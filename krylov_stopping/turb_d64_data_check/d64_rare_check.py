"""Does d = 64 (shorter windows, same resolution) keep the rare events of the current d = 256 data?
Pipeline as turbulence/run_SDE.py: load -> split_periodize_reshape(subseries_len) -> Db3 coarse-graining
to target_len -> normalize. Rare = per-window max |lag-1 increment| above the 95th percentile (notebook def.)."""
import sys
import numpy as np
import torch
sys.path[:0] = ['.', 'codes', 'data']
from data.data_loader import load_turbulence_1d
from codes.utils import split_periodize_reshape, normalize
from codes.ortho_wavelet.ReadyToUseWavelets import DefineWavelet

W = DefineWavelet('Db', m=3, device='cpu')
raw = load_turbulence_1d()
print('raw', tuple(raw.shape))


def build(sub, target):
    x = split_periodize_reshape(raw, sub)
    for _ in range(int(np.log2(x.shape[-1] / target))):
        x = W.decompose(x)[1]
    return x


def maxinc(x):                                  # per-window max |lag-1 increment|, in units of the global std
    a = x[..., 1:] - x[..., :-1]
    return (a.abs().amax(dim=(1, 2)) / a.std()).numpy(), a


def flat(a):
    a = a.flatten().double()
    return float((a ** 4).mean() / (a ** 2).mean() ** 2)


cur, new = build(512, 256), build(128, 64)
nps_c, nps_n = raw.shape[-1] // 512, raw.shape[-1] // 128          # windows per series
print(f'current: {tuple(cur.shape)} ({nps_c}/series) | new: {tuple(new.shape)} ({nps_n}/series)')

for name, x in (('current d=256', cur), ('new d=64', new)):
    m, a = maxinc(x)
    print(f'{name:14s} lag-1 increment flatness {flat(a):6.1f} | per-window max|a|/std percentiles '
          f'50 {np.percentile(m, 50):5.1f}  95 {np.percentile(m, 95):5.1f}  99 {np.percentile(m, 99):5.1f}  '
          f'99.9 {np.percentile(m, 99.9):5.1f}  max {m.max():5.1f}')
    for lag in (1, 2, 4, 8, 16, 32):
        if lag < x.shape[-1] // 2:
            d = (x[..., lag:] - x[..., :-lag]).flatten().double()
            print(f'     lag {lag:2d}: flatness {float((d**4).mean() / (d**2).mean()**2):6.2f}')

# event preservation on the same raw stretches: current window (series s, w) = new windows 4w..4w+3
S = raw.shape[0]
mc, _ = maxinc(cur)
mn, _ = maxinc(new)
mc = mc.reshape(S, nps_c)
mn = mn[:S * nps_n].reshape(S, nps_n)[:, :4 * nps_c].reshape(S, nps_c, 4)
cut_c, cut_n = np.percentile(mc, 95), np.percentile(mn, 95)
rare_c = mc > cut_c
hit = (mn > cut_n).any(-1)
ratio = mn.max(-1) / mc
print(f'\nsame raw stretches ({rare_c.sum()} current-rare windows of {rare_c.size}):')
print(f'  current-rare windows with >= 1 rare 64-window inside: {hit[rare_c].mean():.1%}')
print(f'  largest burst amplitude, new/current (rare windows): median {np.median(ratio[rare_c]):.3f}, '
      f'5th pct {np.percentile(ratio[rare_c], 5):.3f}')
print(f'  rare 64-windows overall: {(mn > cut_n).mean():.1%} (5% by construction)')

for n1 in (2000,):
    print(f'\nrun_SDE takes Data[:n1]: n1 = {n1} 64-windows come from {int(np.ceil(n1 / nps_n))} series '
          f'(current n1 = 8500 256-windows: {int(np.ceil(8500 / nps_c))} series)')
