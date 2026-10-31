"""Wavelet-coefficient histograms of the jets (Modane) training data, and the Scalar_*_gaussianK fit on them.

x1 exactly as run_jet_SDE.py builds it (512 -> Db3 -> M=256, n1 windows, --n1_spread default),
bank = return_Filters(M, J, 1, include_phi=True). Read-only on data; CPU, seconds.

Usage (repo root):  python jets/wavelet_histograms.py [--n1 5000] [--J 8] [--no-fit]
Figures -> jets/data_audit/wavelet_hist/
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data', HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from jet_data import load_modane_windows                      # noqa: E402
from filters_bank import return_Filters                       # noqa: E402

QS = (0.99, 0.9999)


def coefficients(x1, filters):
    """(n, nch, M) real part of the wavelet coefficients (what the scalar potentials use)."""
    return torch.fft.ifft(torch.fft.fft(x1) * filters).real.squeeze(1) if x1.shape[1] == 1 and filters.dim() == 3 \
        else torch.fft.ifft(torch.fft.fft(x1) * filters).real


def stats_table(z, labels):
    print(f"{'ch':>3} {'label':>8} {'std':>9} {'flatness':>9} {'q.99/std':>9} {'q.9999/std':>11} {'max/std':>8}")
    for j, lab in enumerate(labels):
        v = z[:, j].reshape(-1).double().numpy()
        s = v.std()
        a = np.abs(v - v.mean()) / s
        print(f'{j:>3} {lab:>8} {s:9.4g} {np.mean(a**4):9.2f} {np.quantile(a, .99):9.2f} '
              f'{np.quantile(a, .9999):11.2f} {a.max():8.1f}')


def plot_grid(z, labels, path, title):
    nch = z.shape[1]
    ncols = 3
    fig, axes = plt.subplots(int(np.ceil(nch / ncols)), ncols, figsize=(5 * ncols, 3.6 * np.ceil(nch / ncols)))
    for j, ax in enumerate(axes.flat):
        if j >= nch:
            ax.axis('off')
            continue
        v = z[:, j].reshape(-1).double().numpy()
        u = (v - v.mean()) / v.std()
        ax.hist(u, bins=120, density=True, log=True, alpha=.7)
        g = np.linspace(u.min(), u.max(), 400)
        ax.plot(g, np.exp(-g**2 / 2) / np.sqrt(2 * np.pi), 'r-', lw=1)
        ax.set_ylim(max(1e-8, 0.3 / len(u)), 1)
        ax.set_title(f'{labels[j]}  flat={np.mean(u**4):.1f}')
        ax.set_xlabel('Re coeff / std')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n1', type=int, default=5000)
    ap.add_argument('--J', type=int, default=8)
    ap.add_argument('--M', type=int, default=256)
    ap.add_argument('--subseries_len', type=int, default=512)
    ap.add_argument('--no-spread', dest='spread', action='store_false')
    ap.add_argument('--no-fit', dest='fit', action='store_false')
    ap.add_argument('--out', default=str(HERE / 'data_audit' / 'wavelet_hist'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    x1, _ = load_modane_windows(a.subseries_len, a.M, n=a.n1, spread=a.spread)
    print('x1', tuple(x1.shape), 'spread', a.spread)
    filters, filters_phi = return_Filters(a.M, a.J, 1, device='cpu', include_phi=True)
    print('filters', tuple(filters.shape), 'filters_phi', tuple(filters_phi.shape))
    nch = filters.shape[-2]
    labels = [f'j{j}' for j in range(nch - 1)] + ['phi(low)']
    z = torch.fft.ifft(torch.fft.fft(x1)[:, :, :] * filters).real        # (n, nch, M)
    print('coefficients', tuple(z.shape))
    print('\nReal part of coefficients, per channel:')
    stats_table(z, labels)
    plot_grid(z, labels, out / 'hist_morlet_Q1.png', f'Modane M={a.M} n1={a.n1} Re coefficients (Q=1)')

    zp = torch.fft.ifft(torch.fft.fft(x1) * filters_phi).real
    print('\nfilters_phi channels:', zp.shape[1])

    if not a.fit:
        return
    from potentials_builder import get_1d_potentials
    pots = get_1d_potentials(['Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK'], a.J, filters, 1,
                             filters_Q=None, filters_Phi=filters_phi,
                             deduplicate_filters=True, dedup_tol=1e-8, ggd_kwargs={})
    for name, pot in pots.items():
        pot.verbose = False
        pot.fit(x1)
        plt.close('all')
        print(f'\n== {name}: channels J={pot.J} K={pot.K} active stats={len(pot.active_flat)} '
              f'num_coefficients={pot.num_coefficients}')
        for j in range(pot.J):
            c = pot.cuts[j].double().numpy()
            al = pot.alpha[j].double().numpy()
            n_act = sum(1 for f in pot.active_flat.tolist() if f % pot.J == j)
            print(f'  ch{j}: cuts(|z|)={np.array2string(c, precision=4)} alpha={np.array2string(al, precision=2)} active={n_act}')


if __name__ == '__main__':
    main()
