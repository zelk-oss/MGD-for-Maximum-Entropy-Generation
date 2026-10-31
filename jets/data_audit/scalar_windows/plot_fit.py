"""Figures for the Scalar_morlet_gaussianK fit on jets (Modane): data vs the MaxEnt law of the fitted statistics.

Fits two variants on the same x1 (M=256, J=8, Q=1, n1=5000, spread windows) and draws, per channel,
  fig 1: density of |z|/sigma (log-log): data (train, held-out) and the model of each variant, cuts of variant B
  fig 2: model/data quantile ratio of |z| versus probability, both variants (1 = perfect)
Model = 1-D MaxEnt law whose moments are the potential's own targets (what MGD is asked to reach),
exactly as turbulence/check_fit_offline.py check_potential does. CPU, ~5-8 min in total.

Usage (repo root): python jets/data_audit/scalar_windows/plot_fit.py [--n1 5000]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault('TURBULENCE_1D_DATA_PATH', str(ROOT / 'data/data_files/turbulence_1d_period.pt'))
for p in (ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data',
          ROOT / 'jets', ROOT / 'turbulence'):
    sys.path.insert(0, str(p))
from filters_bank import return_Filters                                   # noqa: E402
from potentials_builder import get_1d_potentials                          # noqa: E402
from jet_data import load_modane_windows                                  # noqa: E402
from check_fit_offline import channel_stats_on_grid                       # noqa: E402
from tail_form_holdout import solve_dual                                  # noqa: E402

OUT = Path(__file__).resolve().parent
VARIANTS = {
    'A default (BIC order)': {},
    'B likelihood + tail q.99 + tail mass': dict(boundary_method='likelihood', tail_quantile=0.99,
                                                 tail_mass=True, min_window_bic=0.0115),
}
PQ = np.array([0.5, 0.9, 0.99, 0.999, 0.9999, 0.99995])


def fit(x1, kw, M, J):
    filters, filters_phi = return_Filters(M, J, 1, device='cpu', include_phi=True)
    pots = get_1d_potentials(['Scalar_morlet_gaussianK'], J, filters, 1, filters_Q=None, filters_Phi=filters_phi,
                             deduplicate_filters=True, dedup_tol=1e-8, ggd_kwargs=kw)
    pot = pots['Scalar_morlet_gaussianK']
    pot.verbose = False
    pot.fit(x1)
    plt.close('all')
    return pot


def model_density(pot, x1, j, a1):
    """(g, p) of the MaxEnt law of channel j (density of |z| on the grid g), None if no active slot."""
    with torch.no_grad():
        target_all = pot._all_slot_forward(x1.double()).mean(0).numpy()
    g = np.concatenate([[0.0], np.geomspace(max(a1[a1 > 0].min(), 1e-14), 20 * a1.max(), 60_000)])
    slots, B = channel_stats_on_grid(pot, j, g)
    if not slots:
        return None
    md = np.array([target_all[k * pot.J + j] for k in slots])
    scale = np.where(np.abs(md) > 0, np.abs(md), 1.0)
    _, p, res, _ = solve_dual(B / scale[:, None], g, md / scale)
    p = np.asarray(p) / np.trapezoid(p, g)
    return g, p, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n1', type=int, default=5000)
    ap.add_argument('--M', type=int, default=256)
    ap.add_argument('--J', type=int, default=8)
    ap.add_argument('--min_window_bic', type=float, default=0.0115)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    VARIANTS['B likelihood + tail q.99 + tail mass']['min_window_bic'] = a.min_window_bic
    X, _ = load_modane_windows(512, a.M, n=a.n1 + 9000, spread=True)
    x1, xh = X[:a.n1], X[a.n1:]
    pots = {}
    for name, kw in VARIANTS.items():
        print('fitting', name, kw, flush=True)
        pots[name] = fit(x1, kw, a.M, a.J)

    pot0 = next(iter(pots.values()))
    z1 = torch.fft.ifft(torch.fft.fft(x1.double()) * pot0.filters.double()).real.numpy()
    zh = torch.fft.ifft(torch.fft.fft(xh.double()) * pot0.filters.double()).real.numpy()
    J = pot0.J
    colors = {'A default (BIC order)': 'tab:orange', 'B likelihood + tail q.99 + tail mass': 'tab:green'}

    fig1, ax1 = plt.subplots(3, 3, figsize=(15, 11))
    fig2, ax2 = plt.subplots(3, 3, figsize=(15, 9), sharey=True)
    for j in range(J):
        a1, ah = np.abs(z1[:, j]).ravel(), np.abs(zh[:, j]).ravel()
        sd = z1[:, j].std()
        r, c = divmod(j, 3)
        bins = np.geomspace(1e-3 * sd, 1.05 * a1.max(), 70)
        for arr, lab, col in ((a1, 'data (train)', 'k'), (ah, 'data (held-out)', 'tab:blue')):
            h, _ = np.histogram(arr, bins=bins)
            dens = h / (len(arr) * np.diff(bins))
            mid = np.sqrt(bins[1:] * bins[:-1])
            ok = h > 0
            if col == 'k':
                dmin, dmax = (dens[ok] * sd).min(), (dens[ok] * sd).max()
            ax1[r, c].plot(mid[ok] / sd, dens[ok] * sd, 'o' if col == 'k' else '.', ms=4 if col == 'k' else 3,
                           color=col, label=lab, alpha=0.8)
        for name, pot in pots.items():
            res = model_density(pot, x1, j, a1)
            if res is None:
                continue
            g, p, resid = res
            ax1[r, c].plot(g[1:] / sd, p[1:] * sd, '-', color=colors[name], lw=1.3, label=name.split(' ')[0])
            qm = np.interp(PQ, np.concatenate([[0], np.cumsum((p[1:] + p[:-1]) / 2 * np.diff(g))]), g)
            ax2[r, c].plot(PQ, qm / np.quantile(a1, PQ), 'o-', color=colors[name], label=name.split(' ')[0] + ' / train')
            ax2[r, c].plot(PQ, qm / np.quantile(ah, PQ), 's--', color=colors[name], alpha=0.5, ms=4,
                           label=name.split(' ')[0] + ' / held-out')
        for cut in pots['B likelihood + tail q.99 + tail mass'].cuts[j].double().numpy():
            ax1[r, c].axvline(cut / sd, color='tab:green', lw=0.6, ls=':')
        ax1[r, c].set_xscale('log'); ax1[r, c].set_yscale('log')
        ax1[r, c].set_xlim(5e-4, 2.5 * a1.max() / sd)                    # data range (model runs to 20 x max)
        ax1[r, c].set_ylim(dmin * 1e-2, 3 * dmax)
        ax1[r, c].set_title(f'ch{j}  std={sd:.2e}' + ('  (low-pass)' if j == J - 1 else ''))
        ax1[r, c].set_xlabel('|z| / sigma'); ax1[r, c].set_ylabel('density of |z|/sigma')
        ax2[r, c].axhline(1, color='k', lw=0.8)
        ax2[r, c].set_xscale('logit'); ax2[r, c].set_title(f'ch{j}'); ax2[r, c].set_xlabel('probability q')
        ax2[r, c].set_ylabel('model / data quantile of |z|')
    ax1[0, 0].legend(fontsize=7)
    ax2[0, 0].legend(fontsize=6)
    fig1.suptitle('Jets (Modane) M=256 Q=1, Scalar_morlet_gaussianK: data vs MaxEnt model (dotted green = cuts of B)')
    fig2.suptitle('Model/data quantile ratio of |z| (1 = perfect)')
    fig1.tight_layout(); fig2.tight_layout()
    fig1.savefig(OUT / f'fit_density{a.tag}.png', dpi=110)
    fig2.savefig(OUT / f'fit_quantile_ratio{a.tag}.png', dpi=110)
    print('saved', OUT / f'fit_density{a.tag}.png', OUT / f'fit_quantile_ratio{a.tag}.png', flush=True)


if __name__ == '__main__':
    main()
