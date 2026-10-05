"""Batched moment-mode Thomas solve for SMALL r (bimodal test, r = 4), and the lam rules on top.

solve_batched() computes exactly what SDE._solve_regularised_thomas(mode='moment') computes
(same energy weights, same block-Jacobi scaling, same symmetrisation, same float64 elimination),
for L values of lam and V data masks in one sweep over the nodes. The reference loops over the
nodes in Python once per (lam, mask), ~1.2 ms per node (59 s for one 50000-node solve at r = 4,
measured 2026-10-05); here the per-node work is shared by all (lam, mask) pairs. Checked against
the reference by check_thomas_batched.py (same folder).

Memory ~ 5 * V * L * n * r^2 * 8 bytes: fine for r = 4, NOT for the turbulence systems
(r ~ 258), which keep using the reference solver through codes/lam_select_cv.py.

lam_path_batched() and block_cv_batched() reproduce codes/lam_select_cv.lam_path (even/odd
held-out loss, amplitude veto, chi2) and gauss_d8_lamtest.block_cv (alternate blocks of B nodes
held out with zero data weight) on top of solve_batched. No live mask (all statistics live).
"""
import numpy as np
import torch

from codes.sde_routines import reg_energy_weights, trapezoid_node_weights

DT = torch.float64


def _sym(X):
    return (X + X.transpose(-1, -2)) / 2


def stack_inputs(inp):
    """Per-node lists of build_inputs() -> stacked float64 tensors (n, r, r) / (n, r)."""
    if inp.get('live') is not None and not bool(torch.as_tensor(inp['live']).all()):
        raise ValueError('thomas_batched has no live mask: some statistic is dead at some node')
    st = lambda L: torch.stack([torch.as_tensor(X).to(DT) for X in L])
    return dict(t=np.asarray(inp['t'], dtype=np.float64), M=st(inp['M']), C=st(inp['C']),
                b=st(inp['b']).reshape(len(inp['b']), -1), v=st(inp['v']).reshape(len(inp['v']), -1),
                ridge=float(inp.get('ridge', 0.0)), schedule=inp.get('schedule', 'uniform'), dim=inp.get('dim'),
                h=inp.get('h'), n_walkers=inp.get('n_walkers'), sigma=inp.get('sigma'))


def ridged(M, ridge):
    Ms = _sym(M)
    return Ms + ridge * torch.diag_embed(torch.diagonal(Ms, dim1=-2, dim2=-1)) if ridge else Ms


def solve_batched(S_, lams, data_w=None, idx=None):
    """
    S_: stack_inputs() dict. lams: L values. data_w: (V, n) multipliers of the data weight a_k
    (1 keep, 0 held out); default one mask of ones. idx: optional node subset (the even/odd CV
    solves on the even nodes ALONE, with the weights recomputed on that sub-grid, as
    lam_select_cv.solve(idx=...)).
    Returns Theta (V, L, n, r) float64 (CODE sign) and the relative scaled residual (V, L).
    """
    t, M, C, b, v = S_['t'], S_['M'], S_['C'], S_['b'], S_['v']
    if idx is not None:
        idx_t = torch.as_tensor(np.asarray(idx))
        t, M, C, b, v = t[np.asarray(idx)], M[idx_t], C[idx_t], b[idx_t], v[idx_t]
    n, r = M.shape[0], M.shape[1]
    if (np.diff(t) <= 0).any():
        raise ValueError('t must be strictly increasing')
    a_u, w_u, l_u = reg_energy_weights(t, 1.0, mode='moment', dim=S_['dim'], schedule=S_['schedule'])
    lam = torch.as_tensor(np.asarray(lams, dtype=np.float64))
    L = len(lam)
    dw = torch.ones(1, n, dtype=DT) if data_w is None else torch.as_tensor(np.asarray(data_w), dtype=DT)
    if idx is not None and data_w is not None:
        raise ValueError('data_w and idx together are not needed by any rule')
    V = dw.shape[0]

    Mr = ridged(M, S_['ridge'])                                         # (n, r, r)
    Cs = _sym(C)
    Cc = (Cs[:-1] + Cs[1:]) / 2                                          # (n-1, r, r) cell C
    ct = -(v[:-1] + v[1:]) / 2                                           # (n-1, r) cell ct
    a = dw * torch.as_tensor(a_u)                                        # (V, n)
    W = lam[:, None] * torch.as_tensor(w_u)                              # (L, n-1)
    Lg = lam[:, None] * torch.as_tensor(l_u)                             # (L, n-1)

    D = a[:, None, :, None, None] * Mr                                   # (V, 1, n, r, r)
    D = D.expand(V, L, n, r, r).clone()
    D[:, :, 1:] += (W[:, :, None, None] * Cc)[None]                      # w_{k-1} C_{k-1}
    D[:, :, :-1] += (W[:, :, None, None] * Cc)[None]                     # w_k C_k
    f = (a[:, None, :, None] * b).expand(V, L, n, r).clone()
    f[:, :, :-1] += (Lg[:, :, None] * ct)[None]
    f[:, :, 1:] -= (Lg[:, :, None] * ct)[None]

    S = torch.diagonal(D, dim1=-2, dim2=-1).clamp_min(1e-300).sqrt()    # (V, L, n, r)
    Dsc = _sym(D / (S[..., :, None] * S[..., None, :]))
    del D
    U = -(W[:, :, None, None] * Cc)[None] / (S[:, :, :-1, :, None] * S[:, :, 1:, None, :])   # (V, L, n-1, r, r)
    rhs = f / S

    B = V * L
    Dsc_, U_, rhs_ = Dsc.reshape(B, n, r, r), U.reshape(B, n - 1, r, r), rhs.reshape(B, n, r)
    cp = torch.empty(B, max(n - 1, 0), r, r, dtype=DT)
    dp = torch.empty(B, n, r, dtype=DT)
    for k in range(n):
        Dk, rk = Dsc_[:, k], rhs_[:, k]
        if k > 0:
            Lk = U_[:, k - 1].transpose(-1, -2)
            Dk = Dk - Lk @ cp[:, k - 1]
            rk = rk - (Lk @ dp[:, k - 1, :, None])[..., 0]
        if k < n - 1:
            sol = torch.linalg.solve(Dk, torch.cat([U_[:, k], rk[..., None]], dim=-1))
            cp[:, k], dp[:, k] = sol[..., :-1], sol[..., -1]
        else:
            dp[:, k] = torch.linalg.solve(Dk, rk)
    Th = torch.empty(B, n, r, dtype=DT)
    Th[:, -1] = dp[:, -1]
    for k in range(n - 2, -1, -1):
        Th[:, k] = dp[:, k] - (cp[:, k] @ Th[:, k + 1, :, None])[..., 0]
    del cp, dp

    Ax = (Dsc_ @ Th[..., None])[..., 0]                                 # solve check, as the reference
    Ax[:, :-1] += (U_ @ Th[:, 1:, :, None])[..., 0]
    Ax[:, 1:] += (U_.transpose(-1, -2) @ Th[:, :-1, :, None])[..., 0]
    res = (((Ax - rhs_) ** 2).sum((1, 2)) / ((rhs_ ** 2).sum((1, 2))).clamp_min(1e-300)).sqrt()
    Th = Th.reshape(V, L, n, r) / S
    return Th, res.reshape(V, L)


def data_term(S_, Th, nodes, weights):
    """sum_{k in nodes} weights_k [1/2 Th_k^T Mr_k Th_k - b_k^T Th_k]; Th (..., len(nodes), r)
    -> (...). Same as lam_select_cv._data_term (M with the ridge)."""
    nodes = torch.as_tensor(np.asarray(nodes))
    Mr = ridged(S_['M'][nodes], S_['ridge'])
    w = torch.as_tensor(np.asarray(weights, dtype=np.float64))[nodes]
    q = 0.5 * torch.einsum('...ki,kij,...kj->...k', Th, Mr, Th) - torch.einsum('...ki,ki->...k', Th, S_['b'][nodes])
    return (q * w).sum(-1)


def _interp(t_src, Th_src, t_dst):
    """lam_select_cv._interp along the node axis (-2) of Th_src (..., n_src, r)."""
    t_src, t_dst = torch.as_tensor(t_src), torch.as_tensor(t_dst)
    j = torch.searchsorted(t_src, t_dst).clamp(1, len(t_src) - 1)
    t0, t1 = t_src[j - 1], t_src[j]
    w = ((t_dst - t0) / (t1 - t0)).clamp(0, 1)[:, None]
    return (1 - w) * Th_src[..., j - 1, :] + w * Th_src[..., j, :]


def lam_path_batched(S_, lams, veto=10.0, block_frac=0.01):
    """lam_select_cv.lam_path on solve_batched. Returns the same keys (Theta float64 here)."""
    lams = sorted(set([0.0] + [float(x) for x in lams]))
    t = S_['t']
    n = len(t)
    w_nodes = trapezoid_node_weights(t)
    even, odd = np.arange(0, n, 2), np.arange(1, n, 2)
    Th, res = solve_batched(S_, lams)
    Th, res = Th[0], res[0]                                              # (L, n, r)
    The, _ = solve_batched(S_, lams, idx=even)
    Tho = _interp(t[even], The[0], t[odd])                               # (L, n_odd, r)
    cv = data_term(S_, Tho, odd, w_nodes).numpy()
    Th0 = Th[0]
    nb = max(5, int(block_frac * n))
    A = torch.stack([Th0[i:i + nb].median(0).values for i in range(0, n, nb)]).abs().max(0).values.clamp_min(1e-300)
    ratio = (Th.abs().max(1).values / A).max(1).values.numpy()
    vetoed = ratio > veto
    chi2 = None
    if all(S_.get(k) is not None for k in ('h', 'n_walkers', 'sigma')):
        scale = S_['n_walkers'] * torch.as_tensor(np.asarray(S_['h'], dtype=np.float64)) * S_['sigma'] ** 2 / 2
        Dl = Th0[None] - Th
        q = torch.einsum('lki,kij,lkj->lk', Dl, ridged(S_['M'], S_['ridge']), Dl)
        chi2 = ((scale * q).sum(1) / (n * Th.shape[-1])).numpy()
    ok = np.where(~vetoed)[0]
    sel = lams[ok[np.argmin(cv[ok])]] if len(ok) else None
    return {'lams': lams, 'Theta': Th, 'cv_loss': cv, 'veto': vetoed, 'amp_ratio': ratio, 'chi2': chi2,
            'residual': res.numpy(), 'lam_selected': sel, 'A': A, 'veto_factor': veto, 'block_nodes': nb}


def block_cv_batched(S_, lams, B, vetoed):
    """gauss_d8_lamtest.block_cv on solve_batched: two folds of alternate blocks of B nodes,
    held-out nodes with zero data weight, scored with the data term (trapezoid weights).
    lams as in the path (0 included -> nan); vetoed: bool per lam from the path."""
    t = S_['t']
    n = len(t)
    w = trapezoid_node_weights(t)
    blk = (np.arange(n) // B) % 2
    pos = [i for i, l_ in enumerate(lams) if l_ > 0]
    lp = [lams[i] for i in pos]
    loss = np.full(len(lams), np.nan)
    masks = np.stack([(blk != f).astype(np.float64) for f in (0, 1)])    # data kept outside fold f
    Th, _ = solve_batched(S_, lp, data_w=masks)                         # (2, Lp, n, r)
    tot = np.zeros(len(lp))
    for f in (0, 1):
        held = np.flatnonzero(blk == f)
        tot += data_term(S_, Th[f][:, torch.as_tensor(held)], held, w).numpy()
    loss[pos] = tot
    ok = [i for i in pos if not vetoed[i] and np.isfinite(loss[i])]
    sel = lams[min(ok, key=lambda i: loss[i])] if ok else None
    return loss, sel
