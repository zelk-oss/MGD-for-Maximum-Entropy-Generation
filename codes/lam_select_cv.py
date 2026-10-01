"""Choosing lam for the time-regularised Theta on a saved system (2026-09-30).

Used by codes/resolve_theta_reg.py --select. Returns the whole path Theta(lam), not only
the chosen lam. Sign: CODE sign (theta_code = -theta_MGD), as in the saved system.

Held-out moment-matching loss. The nodes are split into even (fit) and odd (held out).
For each lam, Theta is solved on the even nodes, interpolated linearly in t to the odd
nodes, and scored there with the data term
    L(lam) = sum_{k odd} h_k [ 1/2 Th_k^T M_k Th_k - b_k^T Th_k ],
h_k the trapezoid weights of the full grid. Up to a lam-independent constant this is
    1/2 sum_{k odd} h_k || M_k Th_k - b_k ||^2_{M_k^{-1}},
the linearised moment mismatch left by a corrector step with multiplier Th_k:
phibar(y_k + h sigma^2 Th_k . grad phi) - m_{t_k} ≈ h sigma^2 (M_k Th_k - b_k).
With the quadrature weights of the moment mode, lam means the same on the half grid.

Amplitude veto. A_i = max over blocks of |median over the block of Theta(0)_{., i}|
(robust to isolated blow-ups of the per-node solution). lam is vetoed when some
max_k |Theta_{k,i}(lam)| > veto * A_i. The z^2 rule of codes/lam_selection.py cannot
see blow-ups (it normalises by the residual's own spread).

Noise-model scale ([derived, external review], notes/guth_reg_audit_0930, B4):
Cov(theta_hat_k) ≈ 2 M_k^{-1} / (n h_k sigma^2), white in k (n walkers, h_k the SDE
step at node k). So
    chi2(lam) = sum_k (n h_k sigma^2 / 2) (theta_hat_k - Th_k)^T M_k (theta_hat_k - Th_k) / N
(N = number of live entries, theta_hat = Theta(0)) is ~1 when Theta(lam) removes only
noise, larger when it also removes signal (or when the noise model is off).
"""

import contextlib
import io
import types

import numpy as np
import torch

from codes.sde_routines import SDE, trapezoid_node_weights


def _sym(X):
    X = X.to(torch.float64)
    return (X + X.T) / 2


def solve(inp, lam, idx=None):
    """Theta (code sign) for one lam on the nodes idx (all if None). inp: see lam_path."""
    idx = np.arange(len(inp['t'])) if idx is None else np.asarray(idx)
    take = lambda L: [L[i] for i in idx]
    live = inp.get('live')
    s = types.SimpleNamespace(num_potentials=inp['r'])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        Th = SDE._solve_regularised_thomas(
            s, inp['t'][idx], take(inp['M']), take(inp['C']), take(inp['b']), take(inp['v']),
            lam, ridge=inp.get('ridge', 0.0), mode=inp['mode'], dim=inp.get('dim'),
            schedule=inp.get('schedule', 'uniform'),
            live=None if live is None else live[torch.as_tensor(idx)])
    return Th, s.last_reg_residual


def _interp(t_src, Th_src, t_dst):
    """Linear interpolation in t of Th_src (n_src, r) to t_dst (constant outside)."""
    t_src, t_dst = torch.as_tensor(t_src), torch.as_tensor(t_dst)
    j = torch.searchsorted(t_src, t_dst).clamp(1, len(t_src) - 1)
    t0, t1 = t_src[j - 1], t_src[j]
    w = ((t_dst - t0) / (t1 - t0)).clamp(0, 1)[:, None].to(Th_src.dtype)
    return (1 - w) * Th_src[j - 1] + w * Th_src[j]


def _data_term(inp, Th, idx, weights):
    """sum_{k in idx} weights_k [1/2 Th_k^T M_k Th_k - b_k^T Th_k] (M with the ridge)."""
    ridge = inp.get('ridge', 0.0)
    total = 0.0
    for j, k in enumerate(idx):
        Mk = _sym(inp['M'][k])
        if ridge:
            Mk = Mk + ridge * torch.diag(torch.diagonal(Mk))
        th = Th[j]
        total += float(weights[k] * (0.5 * th @ Mk @ th - inp['b'][k].to(torch.float64).reshape(-1) @ th))
    return total


def lam_path(inp, lams, veto=10.0, block_frac=0.01, verbose=True):
    """
    inp: dict with
      t (n,) float64; M, C, b, v: per-node lists as the solver takes them (mode 'legacy':
      C = G, v = c; mode 'moment': C = Sigma_w, v = mdot); r; mode; schedule; dim; ridge;
      live (n, r) bool or None; optional h (n,) SDE step per node, n_walkers, sigma
      (all three needed for chi2).
    Returns a dict: lams, t, Theta (L, n, r) float32 (full-grid solves), cv_loss, veto,
    amp_ratio, chi2 (or None), residual, lam_selected, A (reference amplitudes).
    """
    lams = sorted(set([0.0] + [float(x) for x in lams]))
    t = np.asarray(inp['t'], dtype=np.float64)
    n, r = len(t), inp['r']
    w_nodes = trapezoid_node_weights(t)
    even, odd = np.arange(0, n, 2), np.arange(1, n, 2)
    live = inp.get('live')

    Th0, res0 = solve(inp, 0.0)                                # per-node solves = theta_hat
    nb = max(5, int(block_frac * n))
    blocks = torch.stack([Th0[i:i + nb].median(0).values for i in range(0, n, nb)])
    A = blocks.abs().max(0).values.clamp_min(1e-300)          # (r,)

    chi2_ok = all(inp.get(k_) is not None for k_ in ('h', 'n_walkers', 'sigma'))
    if chi2_ok:
        h = torch.as_tensor(np.asarray(inp['h'], dtype=np.float64))
        scale = inp['n_walkers'] * h * inp['sigma'] ** 2 / 2   # (n,)
        n_live = int(live.sum()) if live is not None else n * r
        ridge = inp.get('ridge', 0.0)

        def Mr(k):                                             # M_k + ridge, float64, built on
            Mk = _sym(inp['M'][k])                             # the fly: a stored float64 copy
            return Mk + ridge * torch.diag(torch.diagonal(Mk)) if ridge else Mk   # = 38 GB at nt 60000

    out = {k_: [] for k_ in ('Theta', 'cv_loss', 'veto', 'amp_ratio', 'chi2', 'residual')}
    for lam in lams:
        Th, res = (Th0, res0) if lam == 0.0 else solve(inp, lam)
        The, _ = solve(inp, lam, even)
        Tho = _interp(t[even], The, t[odd])
        if live is not None:
            Tho = Tho * live[torch.as_tensor(odd)].to(Tho.dtype)
        cv = _data_term(inp, Tho, odd, w_nodes)
        ratio = float((Th.abs().max(0).values / A).max())
        chi2 = None
        if chi2_ok:
            D = Th0 - Th
            q = torch.stack([D[k] @ Mr(k) @ D[k] for k in range(n)])
            chi2 = float((scale * q).sum() / n_live)
        out['Theta'].append(Th.to(torch.float32)); out['cv_loss'].append(cv)
        out['veto'].append(ratio > veto); out['amp_ratio'].append(ratio)
        out['chi2'].append(chi2); out['residual'].append(res)
        if verbose:
            print(f'  lam={lam:9.3g}  held-out loss {cv: .6e}  max|Theta|/A {ratio:9.3g}'
                  f'{"  VETO" if ratio > veto else ""}' + (f'  chi2 {chi2:.3f}' if chi2 is not None else ''))

    cv = np.array(out['cv_loss']); vetoed = np.array(out['veto'])
    ok = np.where(~vetoed)[0]
    lam_sel = lams[ok[np.argmin(cv[ok])]] if len(ok) else None
    if verbose:
        print(f'  selected lam = {lam_sel}  (min held-out loss among {len(ok)}/{len(lams)} not vetoed)')
    return {'lams': lams, 't': torch.as_tensor(t), 'Theta': torch.stack(out['Theta']),
            'cv_loss': cv, 'veto': vetoed, 'amp_ratio': np.array(out['amp_ratio']),
            'chi2': (np.array(out['chi2'], dtype=float) if chi2_ok else None),
            'residual': np.array(out['residual'], dtype=float), 'lam_selected': lam_sel,
            'A': A, 'veto_factor': veto, 'block_nodes': nb}
