"""
regularised_theta_scalar.py — scalar counterpart of SDE.regularised_theta
(sde_routines.py), for the scalar-potential bimodal setup (sde_routines_scalar_new.py).

This is the full-resolution DENSE solve (one (n*r, n*r) linalg.solve, no
block-Thomas / chunked approximation), ported line-for-line from the class
method to scalar (d == 1, unbatched) potentials.

Built directly from the fixed endpoints x0 (noise) and x1 (data), exactly
like the class version — it never touches the walker trajectory xt. x0
should be the SAME noise draw used inside solve_sde for this run (its first
return value), x1 the same data tensor passed into solve_sde.

Assumes the Cos interpolant schedule, matching iteration_step_projection /
solve_sde in sde_routines_scalar_new.py (which hardcode Cos — there's no
interpolant switch to assert against here).

Output units match eta_t2 (the raw corrector returned by solve_sde, BEFORE
dividing by DENOM) — divide by the same DENOM the caller already uses for
the raw MGD trajectory to get a directly comparable regularized theta.
"""

import torch
import numpy as np

from potentials_new import *                      # gradmat — same import as sde_routines_scalar_new.py
from sde_routines_scalar_new import get_potentials


def grad_contract_scalar(S, V, potentials):
    """
    E_n[ phi_j'(S_n) * V_n ]  for each potential j -> shape (r,).

    Scalar (d == 1) special case of SDE._grad_contract: for a scalar signal
    the per-sample gradient field has a single component, so the contraction
    with V collapses to a plain elementwise product and mean.
    """
    return torch.stack([torch.mean(p.grad(S) * V) for p in potentials])


def regularised_theta_scalar(x0, x1, t, potential_names, regularization=0.0, lam=1.0, device='cpu'):
    """
    Scalar port of SDE.regularised_theta — same block-tridiagonal smoothing
    system, solved as one dense (n*r, n*r) linalg.solve (no block-Thomas).

    Parameters
    ----------
    x0 : torch.Tensor, shape (n1,)
        Noise endpoint — the first value returned by solve_sde for this run.
    x1 : torch.Tensor, shape (n1,)
        Data endpoint (same x1 passed to solve_sde).
    t : torch.Tensor or array_like, shape (nt+1,)
        Same time grid passed to solve_sde.
    potential_names : list[str]
        Same potential_names passed to solve_sde.
    regularization : float
        Added to the diagonal of each per-step gradient Gram matrix M_k
        (mirrors the class's `self.regularization`, no diagonal preconditioning
        — matches `regularised_theta`, not `compute_eta`/`compute_theta`).
    lam : float
        Smoothing regularization weight.
    device : str

    Returns
    -------
    Theta : torch.Tensor, shape (n, r)
        Regularized "eta_t2"-equivalent trajectory (pre-DENOM), on t[:-1] if
        t[-1] == 1 (the data endpoint is dropped because cos(pi/2) = 0 there).
    """
    potentials = get_potentials(potential_names, device)
    r = len(potentials)

    t = np.asarray(t.cpu().numpy() if torch.is_tensor(t) else t, dtype=float)
    if np.isclose(t[-1], 1.0):
        t = t[:-1]                                          # drop data endpoint (cos = 0)
    n = len(t)

    x0 = x0.to(device)
    x1 = x1.to(device)

    d = 1                                                    # scalar ambient dimension
    z2 = x0.pow(2)                                           # ||Z||^2 per sample, (n1,)
    zx = x0 * x1                                              # Z . X per sample,  (n1,)
    adot = np.pi / 2.0                                       # d alpha / dt (Cos)
    eye = torch.eye(r, device=device)

    M, Gf, bb, cc = [], [], [], []
    for tk in t:
        ak = np.pi * tk / 2.0
        cos_k, tan_k = np.cos(ak), np.tan(ak)
        I_k = np.cos(ak) * x0 + np.sin(ak) * x1               # interpolant at t_k, (n1,)
        moments = torch.stack([p(I_k) for p in potentials], dim=1)   # (n1, r)
        n1 = I_k.shape[0]

        M.append(gradmat(I_k, potentials) + regularization * eye)        # grad Gram, (r, r)
        Gf.append(moments.T @ moments / n1)                              # moment Gram, (r, r)
        bb.append(grad_contract_scalar(I_k, x0, potentials) / cos_k)     # b_k, (r,)
        tau = -adot * (tan_k * (d - z2) + zx)                            # (n1,)
        cc.append(torch.einsum('sr,s->r', moments, tau) / n1)            # c_k, (r,)

    dt = np.diff(t)
    A = torch.zeros((n, r, n, r), device=device)
    f = torch.zeros((n, r), device=device)
    for k in range(n):
        A[k, :, k, :] += M[k]
        f[k]          += bb[k]
    for k in range(n - 1):
        w = lam / dt[k] ** 2
        A[k,     :, k,     :] += w * Gf[k]
        A[k + 1, :, k + 1, :] += w * Gf[k]
        A[k,     :, k + 1, :] -= w * Gf[k]
        A[k + 1, :, k,     :] -= w * Gf[k]
        f[k]     -= (lam / dt[k]) * cc[k]
        f[k + 1] += (lam / dt[k]) * cc[k]

    Theta = torch.linalg.solve(A.reshape(n * r, n * r), f.reshape(n * r))
    return Theta.reshape(n, r)