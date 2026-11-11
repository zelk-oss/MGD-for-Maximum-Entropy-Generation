#!/usr/bin/env python3
"""Generalised spectrum of the time-coupling matrix C relative to M, on saved regularised systems (moment mode).

Per selected cell k (between nodes k and k+1) of a system written by SDE.save_regularised_system:
    C = (Sigma_k + Sigma_k+1)/2          (Cov_w(phi), walker covariance, stored float32)
    M = (M_k + M_k+1)/2                  (E[grad phi grad phi^T], raw, no ridge)
on the live statistics (live_k AND live_k+1, diag M > 0). Both are Jacobi-scaled by sqrt(diag M)
(so M has unit diagonal), then the generalised problem  C v = mu (M + ridge I) v  is solved in float64
with scipy.linalg.eigh(C, M + ridge I) (Cholesky of M; M^-1/2 is never formed).

Why mu: for a stochastic increment dx = sqrt(2 h) sigma xi with xi ~ N(0, I), dphi = grad phi dx, so the
covariance of the phi-increments over one step is 2 h sigma^2 M. Directions of C that exist only because the
walkers MOVED between the averaged nodes (not because of the N-1 walker differences at one instant) are
therefore expected at mu ~ 2 h sigma^2 (h = fine step; the node values average n_subsample fine steps, so the
coarse cell width dt_cell = t_k+1 - t_k is reported too), far below the bulk.

Printed per cell:
  r_live, N (n_walkers), N-1
  rank of C (Jacobi-scaled) at the relative thresholds 1e-4 / 1e-6 / 1e-8 of its largest eigenvalue, the
    rank of each node's Sigma at the same thresholds (rank(C) >= max of the two: PSD sum), and the position
    of the largest gap (ratio of consecutive sorted positive eigenvalues) = a threshold-free rank estimate.
    NOTE: Sigma is stored float32 (eps 6e-8), so a null direction carries noise of order 6e-8 * lambda_max,
    and a threshold of 1e-8 counts about half of the null directions (the positive ones) as rank.
  mu: max, median over the bulk (largest N-1 mu), median over mu > 1e-8 mu_max, a log10 histogram (one bin
    per decade), and the counts above / near (within a factor 10) / below 2 h sigma^2 (fine h) and 2 dt_cell sigma^2.

Usage (repo root, CPU):
    python codes/c_spectrum_gen.py <system.pt> [...] [--t_targets 0.05 0.3 0.5 0.7 0.9] [--cells 0 5 ...]
        [--ridge 1e-8] [--sigma S] [--out DIR]
Output: <out>/c_spectrum_gen_<stem>.npz (mu and eigenvalues per cell) and .png (histogram per cell).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import scipy.linalg
import torch

OUT = Path(__file__).resolve().parent / 'results' / 'c_spectrum_gen'
THR = (1e-4, 1e-6, 1e-8)


def rank_at(ev, thr):
    """Number of eigenvalues above thr * largest."""
    return int((ev > thr * ev.max()).sum())


def gap_rank(ev):
    """(rank, ratio): position of the largest ratio between consecutive sorted POSITIVE eigenvalues."""
    p = np.sort(ev[ev > 0])[::-1]
    if len(p) < 2:
        return len(p), float('nan')
    ratio = p[:-1] / p[1:]
    i = int(np.argmax(ratio))
    return i + 1, float(ratio[i])


def node_scaled(S, s):
    return (S / (s[:, None] * s[None, :])).double()


def cell_spectrum(S0, S1, M0, M1, live, ridge):
    Md = (torch.diagonal(M0) + torch.diagonal(M1)).double() / 2
    keep = live & (Md > 0)
    s = Md[keep].sqrt()
    idx = torch.nonzero(keep).squeeze(1)
    sub = lambda A: A.double()[idx][:, idx]
    C0, C1 = node_scaled(sub(S0), s), node_scaled(sub(S1), s)
    C0, C1 = (C0 + C0.T) / 2, (C1 + C1.T) / 2
    C = (C0 + C1) / 2
    Ms = node_scaled(sub((M0 + M1) / 2), s)
    Ms = (Ms + Ms.T) / 2
    out = {'r_live': int(keep.sum())}
    evC = torch.linalg.eigvalsh(C).numpy()
    ev0, ev1 = torch.linalg.eigvalsh(C0).numpy(), torch.linalg.eigvalsh(C1).numpy()
    out['evC'] = evC
    out['lmax'] = float(evC.max())
    for t in THR:
        out[f'rank_C_{t:g}'] = rank_at(evC, t)
        out[f'rank_S0_{t:g}'] = rank_at(ev0, t)
        out[f'rank_S1_{t:g}'] = rank_at(ev1, t)
    out['gap_rank_C'], out['gap_ratio_C'] = gap_rank(evC)
    out['gap_rank_S0'], out['gap_ratio_S0'] = gap_rank(ev0)
    out['gap_rank_S1'], out['gap_ratio_S1'] = gap_rank(ev1)
    Mn, Cn = Ms.numpy(), C.numpy()
    rd = ridge
    for _ in range(8):                               # Cholesky of M + ridge I; raise the ridge if it fails
        try:
            mu = scipy.linalg.eigh(Cn, Mn + rd * np.eye(len(Mn)), eigvals_only=True, check_finite=False)
            break
        except np.linalg.LinAlgError:
            rd *= 10
    else:
        raise RuntimeError('M + ridge is not positive definite even at 1e-1 x the first ridge')
    out['ridge_used'] = rd
    out['mu'] = np.sort(mu)[::-1]
    return out


def describe(o, N, h_fine, dt_cell, sigma):
    mu = o['mu']
    mmax = mu[0]
    nb = min(N - 1, len(mu)) if N else len(mu)
    pos = mu[mu > 1e-8 * mmax]
    lines = [f'    mu: max {mmax:.3e} | bulk (largest {nb}) median {np.median(mu[:nb]):.3e} | '
             f'median of mu > 1e-8 mu_max ({len(pos)} of them) {np.median(pos):.3e} | '
             f'mu <= 1e-8 mu_max or <= 0: {len(mu) - len(pos)} | ridge {o["ridge_used"]:g}']
    hist = []
    for d in range(-14, 3):
        n = int(((mu > 10.0 ** d) & (mu <= 10.0 ** (d + 1))).sum())
        if n:
            hist.append(f'1e{d}:{n}')
    lines.append('    histogram of mu (count with mu in (10^d, 10^(d+1)]): ' + ' '.join(hist)
                 + f' | mu <= 1e-14: {int((mu <= 1e-14).sum())}')
    if sigma:
        for name, hh in (('2 h sigma^2 (fine h)', 2 * h_fine * sigma ** 2),
                         ('2 dt_cell sigma^2', 2 * dt_cell * sigma ** 2)):
            if not hh or not np.isfinite(hh):
                continue
            above, below = int((mu > 10 * hh).sum()), int(((mu > 1e-8 * mmax) & (mu < 0.1 * hh)).sum())
            near = int(((mu >= 0.1 * hh) & (mu <= 10 * hh)).sum())
            lines.append(f'    vs {name} = {hh:.3e}: above x10: {above} | near (x0.1..x10): {near} | '
                         f'below x0.1 but > 1e-8 mu_max: {below}')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('systems', nargs='+', type=Path)
    ap.add_argument('--t_targets', type=float, nargs='+', default=[0.05, 0.3, 0.5, 0.7, 0.9],
                    help='cells whose midpoint is nearest to these t (the last cell is always added)')
    ap.add_argument('--cells', type=int, nargs='+', default=None, help='explicit cell indices (override t_targets)')
    ap.add_argument('--ridge', type=float, default=1e-8, help='ridge on the unit-diagonal M for the Cholesky')
    ap.add_argument('--sigma', type=float, default=None, help='default: meta["sigma"] or parsed from the file name')
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for path in a.systems:
        s = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        meta = s.get('meta', {})
        assert meta.get('reg_mode', 'moment') == 'moment', f'{path.name}: not a moment-mode system'
        t = s['t'].double().numpy()
        n, r = len(t), int(s['num_potentials'])
        N = meta.get('n_walkers')
        sigma = a.sigma or meta.get('sigma')
        if sigma is None:
            m = re.search(r'sigma([0-9.]+?)_nt', path.name)
            sigma = float(m.group(1)) if m else None
        h_all = s['h'].double().numpy() if 'h' in s and s['h'] is not None else None
        print(f'\n=== {path.name[:130]}\n  r = {r}, nodes {n}, N = {N}, n_subsample {meta.get("n_subsample")}, '
              f'sigma {sigma}, meta keys {sorted(meta)[:14]}', flush=True)
        if n < 2:
            print('  fewer than 2 nodes: nothing to measure')
            continue
        mid = (t[:-1] + t[1:]) / 2
        if a.cells is not None:
            cells = a.cells
        else:
            cells = sorted({int(np.argmin(np.abs(mid - x))) for x in a.t_targets} | {n - 2})
        live = s.get('live')
        save, hists = {}, []
        for k in cells:
            lv = torch.ones(r, dtype=torch.bool) if live is None else (live[k].bool() & live[k + 1].bool())
            o = cell_spectrum(s['Sigma'][k], s['Sigma'][k + 1], s['M'][k], s['M'][k + 1], lv, a.ridge)
            dt = float(t[k + 1] - t[k])
            hf = float(h_all[k]) if h_all is not None and np.ndim(h_all) > 0 and k < len(h_all) else float('nan')
            print(f'  cell {k}: t {t[k]:.5f}..{t[k + 1]:.5f} (dt_cell {dt:.3e}, fine h {hf:.3e}) | r_live {o["r_live"]} of {r} | '
                  f'N-1 = {(N - 1) if N else "?"}', flush=True)
            print('    rank C   : ' + ' | '.join(f'{t_:g}: {o[f"rank_C_{t_:g}"]}' for t_ in THR)
                  + f' | largest gap at {o["gap_rank_C"]} (ratio {o["gap_ratio_C"]:.2e})')
            print('    rank S_k : ' + ' | '.join(f'{t_:g}: {o[f"rank_S0_{t_:g}"]}' for t_ in THR)
                  + f' | largest gap at {o["gap_rank_S0"]} (ratio {o["gap_ratio_S0"]:.2e})')
            print('    rank S_k1: ' + ' | '.join(f'{t_:g}: {o[f"rank_S1_{t_:g}"]}' for t_ in THR)
                  + f' | largest gap at {o["gap_rank_S1"]} (ratio {o["gap_ratio_S1"]:.2e})')
            for line in describe(o, N, hf, dt, sigma):
                print(line)
            save[f'mu_cell{k}'] = o['mu']
            save[f'evC_cell{k}'] = o['evC']
            hists.append((k, t[k], t[k + 1], o['mu']))
        f = a.out / f'c_spectrum_gen_{path.stem}.npz'
        np.savez_compressed(f, t=t, cells=np.array(cells), **save)
        print(f'  -> {f}')
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, len(hists), figsize=(3.6 * len(hists), 3.2), squeeze=False, sharey=True)
            for ax, (k, t0, t1, mu) in zip(axs[0], hists):
                p = mu[mu > 1e-14]
                ax.hist(np.log10(p), bins=60, color='tab:blue')
                if sigma and np.isfinite(t1 - t0):
                    ax.axvline(np.log10(2 * (t1 - t0) * sigma ** 2), color='tab:red', ls='--', label='2 dt_cell sigma^2')
                if N:
                    ax.axvline(np.log10(max(mu[min(N - 1, len(mu)) - 1], 1e-14)), color='k', ls=':', label='(N-1)-th mu')
                ax.set_yscale('log')
                ax.set_xlabel('log10 mu (C v = mu M v)')
                ax.set_title(f'cell {k}, t {t0:.3g}..{t1:.3g}', fontsize=9)
            axs[0][0].set_ylabel('number of directions')
            axs[0][0].legend(fontsize=7)
            fig.suptitle(path.name[:90], fontsize=8)
            fig.tight_layout()
            fig.savefig(f.with_suffix('.png'), dpi=110)
            plt.close(fig)
        except Exception as e:                                  # the printed table is the result; the figure is optional
            print('  (figure not written:', e, ')')


if __name__ == '__main__':
    main()
