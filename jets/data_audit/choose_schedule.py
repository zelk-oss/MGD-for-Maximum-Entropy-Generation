"""Choose gap_end / tail ratio of the two-phase schedule for jets from the data, the way it was done for turbulence.

Cos interpolant: at t = 1 - gap_end the leftover white noise has amplitude w = sin(pi gap_end / 2)
(7.85e-5 at gap_end = 5e-5). codes/time_schedules.py: for turbulence (M=256) that is ~1% of the data's POWER at
Nyquist. The data are standardised (unit variance), so a white noise of unit variance has a flat spectrum
P_k = E|fft_k|^2 / M = 1 and the comparison is  w / sqrt(P_nyq).

Prints, for Modane (jets) and turbulence on the same footing (512 -> Db3 -> 256 windows, n1 windows):
  sqrt(P) at Nyquist (last bin, mean of the top 4 and top 16 bins), the ratio w/sqrt(P) at gap_end = 5e-5,
  the gap_end that gives a chosen ratio, and the two-phase tail ratio r, h at 1-t = 1e-2 and the
  implied --ggd_bic_min_window bound sqrt(2 h) sigma for nt / n_bulk / t_switch.

Usage (repo root): python jets/data_audit/choose_schedule.py [--n1 10000] [--nt 40000] [--n_bulk 5000]
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('TURBULENCE_1D_DATA_PATH', str(ROOT / 'data/data_files/turbulence_1d_period.pt'))
for p in (ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data',
          ROOT / 'jets', ROOT / 'turbulence'):
    sys.path.insert(0, str(p))
from jet_data import load_modane_windows                      # noqa: E402
from time_schedules import two_phase_schedule                 # noqa: E402


def nyquist_amplitudes(x):
    """sqrt of the mean power per bin (white unit variance = 1) at Nyquist: last bin, top 4, top 16 bins."""
    M = x.shape[-1]
    P = (torch.fft.rfft(x.double(), dim=-1).abs() ** 2 / M).mean((0, 1)).numpy()      # (M/2+1,)
    return math.sqrt(P[-1]), math.sqrt(P[-4:].mean()), math.sqrt(P[-16:].mean()), P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n1', type=int, default=10000)
    ap.add_argument('--M', type=int, default=256)
    ap.add_argument('--nt', type=int, default=40000)
    ap.add_argument('--n_bulk', type=int, default=5000)
    ap.add_argument('--t_switch', type=float, default=0.9)
    ap.add_argument('--sigma', type=float, default=3.5)
    ap.add_argument('--target_ratio', type=float, default=0.1, help='leftover noise / data AMPLITUDE at Nyquist (0.1 = 1% in power, what turbulence has at gap_end 5e-5)')
    a = ap.parse_args()

    xj, _ = load_modane_windows(512, a.M, n=a.n1, spread=True)
    import check_fit_offline as cfo
    xt, _ = cfo.build_data(512, a.M, a.n1, 0)

    w5 = math.sin(math.pi * 5e-5 / 2)
    print(f'leftover noise weight at gap_end=5e-5: {w5:.3e}\n')
    chosen = {}
    for name, x in (('turbulence', xt), ('jets (Modane)', xj)):
        a1, a4, a16, _ = nyquist_amplitudes(x)
        print(f'{name}: x {tuple(x.shape)}  sqrt(P) at Nyquist: last bin {a1:.3e}, top-4 mean {a4:.3e}, top-16 mean {a16:.3e}')
        for lab, amp in (('last bin', a1), ('top 4', a4), ('top 16', a16)):
            need_w = a.target_ratio * amp
            gap = 2 / math.pi * math.asin(min(need_w, 1.0))
            print(f'   {lab:9s}: ratio w/sqrt(P) at gap_end=5e-5 = {w5 / amp:.2e};  gap_end for {a.target_ratio:g} = {gap:.2e}')
            chosen[(name, lab)] = gap
        print()

    print('two-phase schedules (nt=%d, n_bulk=%d, t_switch=%g):' % (a.nt, a.n_bulk, a.t_switch))
    cands = {'5e-5 (turbulence)': 5e-5}
    for lab in ('last bin', 'top 4', 'top 16'):
        cands[f'jets, {lab}'] = chosen[('jets (Modane)', lab)]
    for lab, g in cands.items():
        if not 0 < g < 1 - a.t_switch:
            print(f'   {lab:22s}: gap_end={g:.2e} outside (0, 1-t_switch)')
            continue
        t, info = two_phase_schedule(a.nt, a.n_bulk, a.t_switch, g)
        tt = t.numpy()
        i = int(np.argmin(np.abs((1 - tt) - 1e-2)))
        h_ref = float(tt[i + 1] - tt[i])
        print(f'   {lab:22s}: gap_end={g:.2e}  r={info["tail_ratio"]:.3e}  h(1-t=1e-2)={h_ref:.3e}'
              f'  min_window_bic (C=1) = sqrt(2h) sigma = {math.sqrt(2 * h_ref) * a.sigma:.3e}')


if __name__ == '__main__':
    main()
