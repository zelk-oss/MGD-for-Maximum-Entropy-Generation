"""Data vs the chosen model vs the same model before the tail fix, on its own banks.

One figure per bank (Q=1: 7 channels, Q=3: 22 channels), finest channel on top. Two
columns per row:
  left   histogram of the wavelet coefficient z_j, log counts, FULL range (no clipping)
  right  P(|z_j| > u) against u, log-log: the extreme tails, one decade per 10x rarer
Curves: data (x1 of the run), "before" and "after" (default morletQ1Q3_L6_2b ->
mass_L6_2b: same M=128, J=7, Q=1+3, L_6, n1=3000, nt=15000; only the fit of the outer
region differs). The 'final' runs are M=256 with Scat4: a different set of channels, so
they cannot share axes with these; pass --before final --after tailfix_final ... only
for arms of one M.

Usage (repo root on Jean Zay):
    python turbulence/plot_before_after.py
    python turbulence/plot_before_after.py --before final --after mass_noScat4   # M=256
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_scalar_orthog_marginals as C                              # noqa: E402

COLORS = {'data': 'k', 'before': 'C3', 'after': 'C0'}


def coeffs(x, wav, J, Q):
    return C.wavelet_coeffs(x, wav, J, Q=Q).transpose(0, 1).reshape(J * Q, -1).numpy()


def ccdf(a, grid):
    a = np.sort(np.abs(a))
    return 1.0 - np.searchsorted(a, grid, side='right') / a.size


def plot_bank(x1, models, wav, J, Q, out):
    n_ch = J * Q
    zd = coeffs(x1, wav, J, Q)
    zs = {k: coeffs(x, wav, J, Q) for k, x in models.items()}
    fig, axes = plt.subplots(n_ch, 2, figsize=(9, 1.5 * n_ch), squeeze=False)
    for j in range(n_ch):
        allz = np.concatenate([zd[j]] + [z[j] for z in zs.values()])
        lim = np.abs(allz).max()
        bins = np.linspace(-lim, lim, 160)
        ax = axes[j, 0]
        ax.hist(zd[j], bins=bins, log=True, histtype='step', color=COLORS['data'], lw=1.2, label='data')
        for k, z in zs.items():
            ax.hist(z[j], bins=bins, log=True, histtype='step', color=COLORS[k], lw=1.2, label=k)
        ax.set_ylabel(f'ch {j}', fontsize=8)
        ax2 = axes[j, 1]
        pos = np.abs(allz)
        pos = pos[pos > 0]
        grid = np.geomspace(max(np.quantile(pos, 0.01), 1e-12), lim, 200)
        ax2.loglog(grid, ccdf(zd[j], grid), color=COLORS['data'], lw=1.2)
        for k, z in zs.items():
            ax2.loglog(grid, ccdf(z[j], grid), color=COLORS[k], lw=1.2)
        ax2.set_ylim(0.3 / zd[j].size, 1.2)
        for a in (ax, ax2):
            a.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7, frameon=False)
            ax.set_title('histogram of z (log counts)', fontsize=9)
            ax2.set_title('P(|z| > u), log-log', fontsize=9)
    fig.suptitle(f'wavelet coefficients on the Q={Q} bank (ch 0 = finest), M={x1.shape[-1]}',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=140)
    print(f'figure: {out}')


def main(before, after, seed, out_base):
    models = {}
    for key, arm in (('before', before), ('after', after)):
        p = C.find_sample(arm, seed)
        x = torch.load(p, map_location='cpu').float()
        x = x.squeeze(1) if x.dim() == 3 else x
        print(f'{key:6s} {arm:20s} {p.name}  {tuple(x.shape)}')
        models[key] = x
    M = models['before'].shape[-1]
    assert models['after'].shape[-1] == M, 'before/after have different M'
    J = C.ARM_J_FOR_M[M]
    wav = C.WAV_FOR_ARM[after]
    x1 = C.rebuild_data_x1(512, M, models['after'].shape[0])
    for Q in (1, 3):
        plot_bank(x1, models, wav, J, Q, out_base.with_name(f'{out_base.stem}_Q{Q}{out_base.suffix}'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', default='morletQ1Q3_L6_2b', choices=sorted(C.LABEL_GLOB))
    ap.add_argument('--after', default='mass_L6_2b', choices=sorted(C.LABEL_GLOB))
    ap.add_argument('--seed', type=int, default=900)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    main(a.before, a.after, a.seed,
         a.out or HERE / 'figures' / f'before_after_{a.before}_vs_{a.after}.png')
