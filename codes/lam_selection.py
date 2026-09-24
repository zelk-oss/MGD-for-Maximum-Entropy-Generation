"""Choosing lam for the time-regularised Theta after the MGD run.

- solve_regularised_thomas_batched: SDE._solve_regularised_thomas for many systems
  sharing one time grid (runs of an ensemble) and many lam values at once. Same
  system, same block-Jacobi scaling, same float64 elimination; only the sequential
  sweep over nodes is shared, and all per-node blocks are built vectorised. Meant
  for small r (the scalar experiments); memory is ~3 * R * L * n * r^2 * 8 bytes.
- block_z2 / window_masks / suggest_lam: the lam-selection rule of
  turbulence/lamtune_select.ipynb (see its first cell for the reasoning):
  R = Theta(lam) - Theta(0); while lam only removes noise, block means of R shrink
  like 1/sqrt(w) and E[z^2] ~ 1; once lam flattens real structure E[z^2] >> 1.
  Suggested lam = the largest with E[z^2] <= z2_max in every t-window.
"""

import numpy as np
import torch

WINDOWS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]   # t-windows for all diagnostics
BLOCKS = [10, 50, 200]          # block sizes (nodes) for the residual z^2 test
W_DECIDE = 50                   # block size the suggestion uses
Z2_MAX = 1.0                    # accepted E[z^2]: slow residual no larger than noise


def solve_regularised_thomas_batched(t, M, G, b, c, lams, ridge=0.0):
    """
    t : (n,) strictly increasing, shared by all systems
    M, G : (R, n, r, r);  b, c : (R, n, r)   -- any float dtype/device
    lams : L lam values
    Returns Theta (R, L, n, r) float64 on CPU (no row dropped, as the reference)
    and the relative scaled residual (R, L), the reference's solve check.
    """
    dev, dtype = 'cpu', torch.float64
    t = np.asarray(t, dtype=np.float64)
    dt = np.diff(t)
    if (dt <= 0).any():
        raise ValueError("t must be strictly increasing for the regularised solve")
    lam = torch.as_tensor(np.asarray(lams, dtype=np.float64))
    dt = torch.as_tensor(dt)
    w = (lam[:, None] / dt ** 2)[None, :, :, None, None]         # (1, L, n-1, 1, 1)
    g = (lam[:, None] / dt)[None, :, :, None]                    # (1, L, n-1, 1)

    sym = lambda X: (X + X.transpose(-1, -2)) / 2
    Ms = sym(M.to(dev, dtype))
    if ridge:
        Ms = Ms + ridge * torch.diag_embed(torch.diagonal(Ms, dim1=-2, dim2=-1))
    Gs = sym(G.to(dev, dtype))[:, None, :-1]                     # (R, 1, n-1, r, r)
    b = b.to(dev, dtype)[:, None]                                # (R, 1, n, r)
    cg = g * c.to(dev, dtype)[:, None, :-1]                      # (R, L, n-1, r)

    wG = w * Gs                                                  # (R, L, n-1, r, r)
    D = Ms[:, None].expand(-1, len(lams), -1, -1, -1).clone()    # (R, L, n, r, r)
    D[:, :, :-1] += wG
    D[:, :, 1:] += wG
    f = b.expand(-1, len(lams), -1, -1).clone()                  # (R, L, n, r)
    f[:, :, :-1] -= cg
    f[:, :, 1:] += cg
    del cg

    S = torch.diagonal(D, dim1=-2, dim2=-1).clamp_min(1e-300).sqrt()   # (R, L, n, r)
    D = sym(D / (S[..., :, None] * S[..., None, :]))
    U = -wG / (S[:, :, :-1, :, None] * S[:, :, 1:, None, :])     # scaled A[k, k+1]
    del wG
    f = f / S

    n = D.shape[2]
    c_prime = torch.empty_like(U)
    d_prime = torch.empty_like(f)
    for k in range(n):
        Dk, rhs = D[:, :, k], f[:, :, k]
        if k > 0:
            Lk = U[:, :, k - 1].transpose(-1, -2)
            Dk = Dk - Lk @ c_prime[:, :, k - 1]
            rhs = rhs - (Lk @ d_prime[:, :, k - 1, :, None])[..., 0]
        if k < n - 1:
            sol = torch.linalg.solve(Dk, torch.cat([U[:, :, k], rhs[..., None]], dim=-1))
            c_prime[:, :, k], d_prime[:, :, k] = sol[..., :-1], sol[..., -1]
        else:
            d_prime[:, :, k] = torch.linalg.solve(Dk, rhs)

    Th = torch.empty_like(f)
    Th[:, :, -1] = d_prime[:, :, -1]
    for k in range(n - 2, -1, -1):
        Th[:, :, k] = d_prime[:, :, k] - (c_prime[:, :, k] @ Th[:, :, k + 1, :, None])[..., 0]
    del c_prime, d_prime

    # solve check: residual of the scaled system, as the reference
    Ax = (D @ Th[..., None])[..., 0]
    Ax[:, :, :-1] += (U @ Th[:, :, 1:, :, None])[..., 0]
    Ax[:, :, 1:] += (U.transpose(-1, -2) @ Th[:, :, :-1, :, None])[..., 0]
    residual = (((Ax - f) ** 2).sum((-1, -2)) / ((f ** 2).sum((-1, -2))).clamp_min(1e-300)).sqrt()

    return Th / S, residual


def window_masks(t_reg, windows=WINDOWS):
    return [(lo, hi, (t_reg >= lo) & (t_reg < hi if hi < 1.0 else t_reg <= hi))
            for lo, hi in zip(windows[:-1], windows[1:])]


def block_z2(R, w):
    """E[z^2] of block means of the residual R (n, r), pooled over blocks and coefficients."""
    n = (len(R) // w) * w
    if n < 2 * w:
        return np.nan
    sd = R.std(0, ddof=1)
    ok = sd > 0
    if not ok.any():
        return np.nan
    z = R[:n, ok].reshape(-1, w, ok.sum()).mean(1) / (sd[ok] / np.sqrt(w))
    return float(np.mean(z ** 2))


def suggest_lam(lams, z2, blocks=BLOCKS, w_decide=W_DECIDE, z2_max=Z2_MAX):
    """z2 : (L, n_windows, n_blocks) run-averaged E[z^2]; NaN windows (too short) pass."""
    b = blocks.index(w_decide)
    ok = [lam for i, lam in enumerate(lams)
          if lam > 0 and np.all(np.nan_to_num(z2[i, :, b], nan=0.0) <= z2_max)]
    return max(ok) if ok else None
