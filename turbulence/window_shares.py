"""Where, in |z|, model and data differ: per channel and window of a run's fitted
Scalar_*_gaussianK, the fraction of coefficients that falls in the window and the share of
<z^2> that comes from it, model samples vs data (hard window edges = the fitted cuts).

Settles what the window statistics <w_k |z|^alpha_k> cannot say alone: e.g. "too few
near-zero coefficients" (fraction) vs "energy lost in the tail" (share of <z^2>).

Usage (repo root on Jean Zay):
    python turbulence/window_shares.py --arm mass_L6_2b --channels 0 1 2 3
    python turbulence/window_shares.py --arm mass_L6_2b_drop --channels 0 1 2 3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data', HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from filters_bank import return_Filters                              # noqa: E402
from potentials_builder import get_1d_potentials                     # noqa: E402
from compare_scalar_orthog_marginals import find_sample, LABEL_GLOB, rebuild_data_x1   # noqa: E402
from window_vs_step_noise import newest, SCALAR                      # noqa: E402


def main(arm, seed, channels):
    xs_path = find_sample(arm, seed)
    label = LABEL_GLOB[arm].split('*')[-2].removesuffix('_2')    # e.g. turbulence_ggd_mass_L6_2b
    exp = newest(f'*/*_seed_{seed}_*{label}_2*', HERE / 'experiments')
    a = json.loads((exp / 'config.json').read_text())
    M, J, Q = int(a['M']), int(a['J']), int(a['Q'])
    print(f'samples {xs_path.name}\nfit     {exp.name}')

    xs = torch.load(xs_path, map_location='cpu').double()
    xs = xs if xs.dim() == 3 else xs.unsqueeze(1)
    x1 = rebuild_data_x1(int(a.get('subseries_len', 512)), M, xs.shape[0]).double().unsqueeze(1)

    filters, filters_phi = return_Filters(M, J, 1, include_phi=True, wav_type=a.get('wav_type', 'morlet'))
    filters_Q = return_Filters(M, J, Q, wav_type=a.get('wav_type', 'morlet'))
    pots = get_1d_potentials(a['terms'], J, filters, Q, filters_Q=filters_Q, filters_Phi=filters_phi,
                             deduplicate_filters=bool(a.get('deduplicate_filters', False)),
                             dedup_tol=float(a.get('dedup_tol', 1e-12)))
    for name in SCALAR:
        if name not in pots:
            continue
        pot = type(pots[name]).load_fixed_parameters(exp / 'fitted_potentials' / f'{name}.pt',
                                                     pots[name].filters, map_location='cpu')
        f = pot.filters.to(torch.complex128)      # complex: the _imag terms carry filters * -1j
        with torch.no_grad():
            zd = torch.fft.ifft(f * torch.fft.fft(x1)).real.numpy()
            zs = torch.fft.ifft(f * torch.fft.fft(xs)).real.numpy()
        act = pot.active.numpy()
        print(f'\n--- {name}   (stat = window has an active statistic; "-" = dropped/dead)')
        print(f"{'ch':>3} {'win':>3} {'lo':>9} {'hi':>9} {'stat':>4} | {'frac data':>9} {'model':>7} {'m/d':>6} | "
              f"{'<z2> share data':>15} {'model':>7} | {'m2 m/d':>6}")
        for j in channels:
            if j >= pot.J:
                continue
            ad, as_ = np.abs(zd[:, j]).ravel(), np.abs(zs[:, j]).ravel()
            e2d, e2s = (ad ** 2).sum(), (as_ ** 2).sum()
            edges = np.concatenate([[0.0], pot.cuts[j].double().numpy(), [np.inf]])
            m2 = (as_ ** 2).mean() / (ad ** 2).mean()
            first = True
            for k in range(pot.K):
                lo, hi = edges[k], edges[k + 1]
                if hi <= 1e-6 * max(ad.max(), 1e-300):           # padded slot of a Keff < K channel
                    continue
                md, ms = (ad >= lo) & (ad < hi), (as_ >= lo) & (as_ < hi)
                fd, fs = md.mean(), ms.mean()
                print(f"{j:>3} {k:>3} {lo:9.2e} {hi:9.2e} {'yes' if act[j, k] else '-':>4} | "
                      f"{fd:9.3f} {fs:7.3f} {fs / max(fd, 1e-300):6.2f} | "
                      f"{(ad[md] ** 2).sum() / e2d:15.3f} {(as_[ms] ** 2).sum() / e2s:7.3f} | "
                      + (f'{m2:6.2f}' if first else ''))
                first = False


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='mass_L6_2b', choices=sorted(LABEL_GLOB))
    ap.add_argument('--seed', type=int, default=900)
    ap.add_argument('--channels', type=int, nargs='+', default=[0, 1, 2, 3])
    a = ap.parse_args()
    main(a.arm, a.seed, a.channels)
