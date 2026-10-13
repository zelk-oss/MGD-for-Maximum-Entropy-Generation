#!/usr/bin/env python3
"""Data sanity check for the production turbulence config.

Loads data exactly as the preflight/run_SDE do (target_len, subseries_len,
coarse-graining, normalize), then saves:
  - signal_marginal.png: histogram of x1 pointwise, overlaid with the standard
    normal (post-normalize, so if the normalization is well-calibrated the
    bulk should look Gaussian; tails will be heavier -- intermittency).
  - wavelet_histograms.png: per-scale histogram of |W_j x| in log-y, with
    the Gaussian prediction (filtered white noise) for comparison. Fine scales
    will be heavy-tailed (intermittent), coarse scales closer to Gaussian.

Usage: python turbulence/viz_data.py [--n1 10000 --target_len 256 ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'data'):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from codes.utils import split_periodize_reshape, normalize            # noqa: E402
from codes.filters_bank import return_Filters                          # noqa: E402
from codes.ortho_wavelet.ReadyToUseWavelets import DefineWavelet       # noqa: E402
from data.data_loader import load_turbulence_1d                        # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--n1', type=int, default=10000)
    ap.add_argument('--subseries_len', type=int, default=512)
    ap.add_argument('--target_len', type=int, default=256)
    ap.add_argument('--J', type=int, default=8)
    ap.add_argument('--Q', type=int, default=3)
    ap.add_argument('--out_dir', type=Path, default=Path('/tmp/viz_data_M256'))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data (same as preflight) ----
    W = DefineWavelet('Db', m=3, device='cpu')
    raw = load_turbulence_1d()
    data = split_periodize_reshape(raw, args.subseries_len)
    pre_len = data.shape[-1]
    scales = int(np.log2(pre_len / args.target_len)) if pre_len > args.target_len else 0
    for _ in range(scales):
        data = W.decompose(data)[1]
    x1 = normalize(data[:args.n1])
    print(f'[data] x1 {tuple(x1.shape)}, mean {float(x1.mean()):+.3e}, '
          f'std {float(x1.std()):.3e}')

    # ---- signal marginal vs standard normal ----
    vals = x1.flatten().numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=200, range=(-5, 5), density=True, alpha=0.6, label='x1')
    xx = np.linspace(-5, 5, 400)
    ax.plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), 'r-', label='N(0,1)')
    ax.set_yscale('log')
    ax.set_xlabel('x'); ax.set_ylabel('density'); ax.legend()
    ax.set_title(f'signal marginal (log-y): n1={args.n1}, M={args.target_len}')
    fig.tight_layout()
    fig.savefig(args.out_dir / 'signal_marginal.png', dpi=100)
    plt.close(fig)
    print(f'[saved] {args.out_dir}/signal_marginal.png')

    # ---- wavelet-coefficient histograms per scale ----
    filters, _ = return_Filters(args.target_len, args.J, 1,
                                device='cpu', include_phi=True)
    # filters: (1, J, M) complex; convolve x1 with each filter, per-sample
    X = torch.fft.fft(x1[:, 0, :], dim=-1)             # (B, M) complex
    F = torch.fft.fft(filters[0], dim=-1).conj()       # (J, M) complex, Hermitian conj
    Y = torch.fft.ifft(X[:, None, :] * F[None, :, :], dim=-1)  # (B, J, M)
    coeff_abs = Y.abs().numpy()

    ncols = 3
    nrows = (args.J + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)
    for j in range(args.J):
        ax = axes[j // ncols][j % ncols]
        v = coeff_abs[:, j, :].flatten()
        v = v / (v.std() + 1e-30)
        ax.hist(v, bins=200, range=(0, 8), density=True, alpha=0.6)
        xx = np.linspace(0, 8, 400)
        # half-normal: 2 * phi(x) for x >= 0 (if the real part were N(0,1))
        ax.plot(xx, 2 * np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), 'r-',
                label='half-N(0,1)')
        ax.set_yscale('log')
        ax.set_title(f'|W_j x|/std, j={j}')
        ax.legend(fontsize=8)
    for k in range(args.J, nrows * ncols):
        axes[k // ncols][k % ncols].axis('off')
    fig.suptitle(f'wavelet-coefficient marginals (log-y), n1={args.n1}')
    fig.tight_layout()
    fig.savefig(args.out_dir / 'wavelet_histograms.png', dpi=100)
    plt.close(fig)
    print(f'[saved] {args.out_dir}/wavelet_histograms.png')


if __name__ == '__main__':
    main()
