"""Largest full 2D run that fits in one job: projected wall time of a full SDE run vs number of walkers, per model.

Numbers = steady seconds per SDE step read from the 20-step timing runs of 2026-11-18 (H100, new code: 11-17 port +
11-18 index fix; logs turbulence_2d/saved_results/logs/slurm_t2d_<arm>_<job>_0.log, last tqdm s/it). Projected run
time = (s/step) x nt. Fit: time per step = b * N (least squares through the origin; the fixed part per step is ~0
in every model). The Theta_reg solve (separate job) and the start/end of the run are not included.
Cosmic web costs the same per step (scat4 N 1000: 57.77 vs 57.36 s, job 388761).

Usage: python turbulence_2d/timing/plot_max_walkers.py   -> turbulence_2d/timing/max_walkers_2d.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NT = 3300            # turbulence_2d production steps (cosmic web: 2700 scat4/base, 3000 *_real)
CAP_T3, CAP_DEV = 20.0, 2.0   # hours

# model: (label, r, {N: s/step}, job ids)
DATA = {
    'scat4':      ('Scat3 + Scat4, Re + Im', 4294, {100: 5.83, 400: 22.98, 1000: 57.36, 2000: 114.75},
                   '387976 388323 388240 388239'),
    'scat4_real': ('Scat3 + Scat4, Re only', 2414, {400: 10.47, 1000: 26.09, 2000: 52.12}, '388414 388357 388266'),
    'base':       ('Scat3 Re + Im, no Scat4', 564, {400: 2.11, 1000: 5.21, 2000: 10.42}, '388862 388847 388443'),
    'base_real':  ('Scat3 Re only, no Scat4', 324, {400: 1.17, 1000: 2.91, 2000: 5.83}, '389089 389060 388871'),
}
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']   # dataviz default categorical slots 1-4, fixed order
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def main():
    out = Path(__file__).resolve().parent / 'max_walkers_2d.png'
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    N = np.logspace(np.log10(100), np.log10(20000), 200)
    for k, ((key, (label, r, pts, jobs)), c) in enumerate(zip(DATA.items(), COLORS)):
        n = np.array(list(pts)); s = np.array(list(pts.values()))
        b = (s * n).sum() / (n * n).sum()                     # s per walker per step
        hours = lambda x: b * x * NT / 3600
        nmax = CAP_T3 * 3600 / (b * NT)
        ax.plot(N, hours(N), color=c, lw=2, zorder=3, label=f'{label} (r = {r})')
        ax.plot(n, s * NT / 3600, 'o', ms=8, color=c, mec='white', mew=2, zorder=4)
        ax.plot([nmax, nmax], [0.05, CAP_T3], ls=':', lw=1.2, color=c, zorder=2)
        ax.annotate(f'N ≈ {nmax:,.0f}'.replace(',', ' '), (nmax, 0.06), rotation=90, ha='right', va='bottom',
                    fontsize=9, color=INK, xytext=(-3, 0), textcoords='offset points')
        ax.annotate(f'r = {r}', (N[-1], hours(N[-1])), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=9, color=INK)
    for cap, txt in ((CAP_T3, '20 h: longest job (t3 QoS)'), (CAP_DEV, '2 h: dev QoS')):
        ax.axhline(cap, color=INK2, lw=1, ls='--', zorder=1)
        ax.annotate(txt, (110, cap), xytext=(0, 3), textcoords='offset points', fontsize=9, color=INK2)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(100, 20000); ax.set_ylim(0.05, 2000)
    ax.set_xlabel('number of walkers N', color=INK)
    ax.set_ylabel(f'wall time of one full run, nt = {NT} (hours)', color=INK)
    ax.set_title('2D (128 × 128), H100: full-run time vs walkers — largest N in 20 h at the dotted lines',
                 fontsize=11, color=INK, loc='left')
    ax.grid(True, which='major', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(loc='upper left', fontsize=9, frameon=False, bbox_to_anchor=(0, 0.93))
    fig.text(0.01, 0.01, 'Dots: measured (20-step runs, 2026-11-18, s/step × nt). Lines: fit time per step ∝ N. '
             'SDE only (Θ_reg solve not included).\nCosmic web: same cost per step; its nt is 2700–3000 '
             '→ about 10–20 % more walkers fit in 20 h.', fontsize=8, color=INK2)
    fig.subplots_adjust(right=0.88, bottom=0.15)
    fig.savefig(out)
    print(out)


if __name__ == '__main__':
    main()
