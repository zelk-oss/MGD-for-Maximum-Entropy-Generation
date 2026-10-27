"""Summary figures of the log p per family, from the .npz files of logp_family_contrib.py
(figures/data/logp_C_<label>_lam<lam>[_unc].npz). CPU only, runs in seconds, locally too.

Per model (as notebook cells 28 and 25 of turb_interpretation_rare_typical_final.ipynb):
  1. aggregate log p: histogram of TOTAL log p over all signals (gray), outlines of the rare
     and typical signals on the same bins, log counts.
  2. bar chart per family (TOTAL included): mean and std across signals, typical vs rare,
     then a zoom with its own y-axis per family for the small ones (mean +- standard error).
Plus one figure with the aggregate log p of all the models side by side on a COMMON x-axis.
log p is up to an additive constant that is the same for rare and typical (same theta), so
shapes and the gap between them are exact; the absolute x position is not.

Usage:
    python turbulence/plot_logp_summary.py                       # the two M=128 Scat4 models, lam 5e-4
    python turbulence/plot_logp_summary.py --percentiles 95 99 --suffix _unc
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
if __name__ == '__main__':          # keep the notebook's inline backend when imported
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

HERE = Path(__file__).resolve().parent
DATA = HERE / 'figures' / 'data'
COLORS = {'typical': 'C0', 'rare': 'C3', 'all': 'tab:gray'}
LABELS = {'turbulence_ggd_mass_L6_2b_scat4': 'new: Scat4 + fix',
          'turbulence_final_coarse_noL2_scat4': 'old: Scat4, no fix'}
SMALL = ('Scat4_Real_Q1', 'Scat4_Imag_Q1')          # zoomed with their own y-axis


def short(name):
    return (name.replace('Scattering_Fourth_Order_Mod2_', 'Scat4_').replace('Scalar_', 'S_')
            .replace('_gaussianK', ''))


def load(label, lam, suffix):
    f = DATA / f'logp_C_{label}_lam{lam:g}{suffix}.npz'
    z = np.load(f, allow_pickle=False)
    C = {short(k): z[k] for k in z.files if k not in ('increment', 'theta_desc')}
    return C, z['increment'], str(z['theta_desc'])


def rare_mask(inc, P):
    return inc > np.percentile(inc, P)


def plot_aggregate(ax, C, rare, edges, title):
    tot = C['TOTAL']
    ax.hist(tot, bins=edges, color=COLORS['all'], alpha=0.6, label=f'all (n={len(tot)})')
    for name, m in (('typical', ~rare), ('rare', rare)):
        ax.hist(tot[m], bins=edges, histtype='step', lw=1.3, color=COLORS[name], label=f'{name} (n={int(m.sum())})')
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.5)
    ax.set_xlabel(r'$\log p_\theta(x)$ (up to additive const.)')
    ax.set_ylabel('count')
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)


def plot_bars(C, rare, title, out=None):
    fams = list(C)
    x = np.arange(len(fams))
    w = 0.38
    cats = (('typical', ~rare), ('rare', rare))
    mean = {n: np.array([C[F][m].mean() for F in fams]) for n, m in cats}
    std = {n: np.array([C[F][m].std(ddof=1) for F in fams]) for n, m in cats}
    zoom = [F for F in SMALL if F in C]
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, max(len(zoom), 1), height_ratios=[1.3, 1.0, 1.0, 1.0])
    ax_m = fig.add_subplot(gs[0, :])
    ax_s = fig.add_subplot(gs[1, :], sharex=ax_m)
    for i, (n, m) in enumerate(cats):
        off = (i - 0.5) * w
        ax_m.bar(x + off, mean[n], w, color=COLORS[n], label=f'{n} (n={int(m.sum())})')
        ax_s.bar(x + off, std[n], w, color=to_rgba(COLORS[n], 0.45))
    ax_m.axhline(0, color='k', lw=0.6)
    ax_m.set_ylabel('mean (nats)')
    ax_s.set_ylabel('std across signals (nats)')
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(fams, rotation=30, ha='right')
    ax_m.legend()
    ax_m.set_title(f'{title}: log p contribution per family')
    for j, F in enumerate(zoom):
        for row, (what, fn) in enumerate((('mean +- SE (nats)', 'mean'), ('std (nats)', 'std'))):
            ax = fig.add_subplot(gs[2 + row, j])
            for i, (n, m) in enumerate(cats):
                v = C[F][m]
                val = v.mean() if fn == 'mean' else v.std(ddof=1)
                err = v.std(ddof=1) / np.sqrt(v.size) if fn == 'mean' else None
                ax.bar(i, val, 0.7, yerr=err, capsize=4,
                       color=COLORS[n] if fn == 'mean' else to_rgba(COLORS[n], 0.45))
                ax.annotate(f'{val:.3g}', (i, val), ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
            ax.axhline(0, color='k', lw=0.6)
            ax.margins(y=0.2)
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels([c[0] for c in cats])
            if row == 0:
                ax.set_title(f'zoom: {F} (own y-axis)', fontsize=9)
            if j == 0:
                ax.set_ylabel(what)
    if out is not None:
        fig.savefig(out, dpi=130)
        print(f'figure: {out}')
    return fig


def main(labels, lam, suffix, percentiles):
    data = {lab: load(lab, lam, suffix) for lab in labels}
    for P in percentiles:
        lo = min(C['TOTAL'].min() for C, _, _ in data.values())
        hi = max(C['TOTAL'].max() for C, _, _ in data.values())
        edges = np.linspace(lo, hi, 201)                       # one set of bins for every model
        fig, axes = plt.subplots(1, len(labels), figsize=(6.2 * len(labels), 4.4), sharex=True, sharey=True,
                                 squeeze=False)
        for ax, lab in zip(axes[0], labels):
            C, inc, desc = data[lab]
            plot_aggregate(ax, C, rare_mask(inc, P), edges, f'{LABELS.get(lab, lab)}  [{desc[:40]}]')
        fig.suptitle(f'Overall log p, whole dataset; rare = top {100 - P:g}% of max |lag-1 increment|, '
                     'common x-axis', fontsize=10)
        fig.tight_layout()
        out = HERE / 'figures' / f'logp_aggregate_compare_lam{lam:g}{suffix}_p{P:g}.png'
        fig.savefig(out, dpi=130)
        print(f'figure: {out}')
        for lab in labels:
            C, inc, desc = data[lab]
            rare = rare_mask(inc, P)
            plot_bars(C, rare, f'{LABELS.get(lab, lab)}, rare = top {100 - P:g}%',
                      HERE / 'figures' / f'logp_bars_{lab}_lam{lam:g}{suffix}_p{P:g}.png')
            fig, ax = plt.subplots(figsize=(7, 4.5))
            plot_aggregate(ax, C, rare, edges, f'{LABELS.get(lab, lab)}: overall log p ({desc[:40]})')
            fig.tight_layout()
            out = HERE / 'figures' / f'logp_aggregate_{lab}_lam{lam:g}{suffix}_p{P:g}.png'
            fig.savefig(out, dpi=130)
            print(f'figure: {out}')
            plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', nargs='+', default=list(LABELS))
    ap.add_argument('--lam', type=float, default=5e-4)
    ap.add_argument('--suffix', default='_unc', help="'_unc' = unconstrained Theta_reg files, '' = L_6<0 / default")
    ap.add_argument('--percentiles', type=float, nargs='+', default=[95, 99])
    a = ap.parse_args()
    main(a.labels, a.lam, a.suffix, a.percentiles)
