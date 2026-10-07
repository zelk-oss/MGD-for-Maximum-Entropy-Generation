"""Rare events after coarse-graining 3 times (512-windows -> 64) vs the current data (512 -> 256), same windows.
Pipeline as turbulence/run_SDE.py (split_periodize_reshape, Db3 coarse-graining, normalize). Rare = per-window
max |lag-1 increment| above the 95th percentile (rare/typical notebook definition).
Run from the repo root: TURBULENCE_1D_DATA_PATH=$PWD/data/data_files/turbulence_1d_period.pt python <this>"""
import sys
import numpy as np
import torch
from scipy.stats import spearmanr
sys.path[:0] = ['.', 'codes', 'data']
from data.data_loader import load_turbulence_1d
from codes.utils import split_periodize_reshape
from codes.ortho_wavelet.ReadyToUseWavelets import DefineWavelet

W = DefineWavelet('Db', m=3, device='cpu')
x = split_periodize_reshape(load_turbulence_1d(), 512)
levels = {}
for k in range(4):                                   # k coarse-graining levels: length 512 / 2^k
    levels[k] = x
    x = W.decompose(x)[1]


def maxinc(x):
    a = x[..., 1:] - x[..., :-1]
    return (a.abs().amax(dim=(1, 2)) / a.std()).numpy(), a


def flat(d):
    d = d.flatten().double()
    return float((d ** 4).mean() / (d ** 2).mean() ** 2)


ref, _ = maxinc(levels[1])                           # current data: 1 level, d = 256
rare_ref = ref > np.percentile(ref, 95)
print(f'{"data":28s} {"lag-1 flatness":>14s} {"max|a|/std p50/p99/max":>24s} {"rare kept":>10s} {"Spearman":>9s} {"amp ratio (rare)":>17s}')
for k in (1, 2, 3):
    m, a = maxinc(levels[k])
    rare = m > np.percentile(m, 95)
    kept = (rare & rare_ref).sum() / rare_ref.sum()
    rho = spearmanr(m, ref).correlation
    print(f'{f"coarse x{k}, d = {512 >> k}":28s} {flat(a):14.1f} {np.percentile(m, 50):8.1f}/{np.percentile(m, 99):5.1f}/{m.max():5.1f} '
          f'{kept:10.1%} {rho:9.3f} {np.median(m[rare_ref] / ref[rare_ref]):17.2f}')
    print('      flatness at lags 1/2/4/8/16 (own samples):',
          ' '.join(f'{flat(levels[k][..., L:] - levels[k][..., :-L]):.1f}' for L in (1, 2, 4, 8, 16)))

# how strongly intermittent is the coarse lag-1 increment, in physical terms: lag-1 at x3 spans 8 raw samples
raw = levels[0]
print('\nraw-resolution increment flatness at lag 2 / 4 / 8 (for reference):',
      ' '.join(f'{flat(raw[..., L:] - raw[..., :-L]):.1f}' for L in (2, 4, 8)))
