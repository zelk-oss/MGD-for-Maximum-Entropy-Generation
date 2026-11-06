#!/usr/bin/env python3
"""How noisy are the raw Lagrange-multiplier trajectories theta_t of the SDE runs, per family?

For a window of --last steps (default 100) ending at the last step, and for comparison one ending at t ~ 0.9 (end of the
uniform bulk of the two_phase grid): per statistic k, the residual of theta_k(step) after a smooth polynomial trend
(--deg, default 3, in the step index; the tail drifts fast as t -> 1 and a trend is not noise) gives
    sigma_k = std of the residual,       rel_k = sigma_k / median |theta_k| over the window.
Per family: median and 90th percentile of rel_k over the statistics. In units of log p:
    noise_F = sqrt( sum_k (sigma_k * s_k)^2 ),   s_k = std over the 100 data maps of phi_k (results/phi cache),
the noise of the family contribution theta_F . phi_F if the errors of the statistics were independent (a rough scale;
correlated errors along ill-conditioned directions can make it larger or smaller), to be compared with the std of -log p
over maps (~2000 nats here). Mean over the seeds.

Usage (repo root, Jean Zay, CPU is enough, seconds):  python cosmic_web/theta_noise_cw.py --arms base_real scat4_real
Output: cosmic_web/results/theta_noise/theta_noise_<arm>.csv (+ printed tables)
"""
import argparse

import numpy as np
import pandas as pd
import torch

import cw_common as cw


def detrended_noise(Y, deg):
    """Y (n, r): std over the window of the residual of a polynomial fit of degree deg in the step index, per column."""
    n = Y.shape[0]
    x = np.linspace(-1, 1, n)
    V = np.vander(x, deg + 1)
    coef, *_ = np.linalg.lstsq(V, Y, rcond=None)
    return (Y - V @ coef).std(0, ddof=deg + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arms', nargs='+', default=['base_real', 'scat4_real'])
    ap.add_argument('--seeds', nargs='+', type=int, default=cw.SEEDS)
    ap.add_argument('--last', type=int, default=100, help='steps in the window')
    ap.add_argument('--deg', type=int, default=3, help='degree of the trend removed')
    a = ap.parse_args()
    out = cw.RESULTS / 'theta_noise'
    out.mkdir(parents=True, exist_ok=True)

    for arm in a.arms:
        runs = cw.find_runs(arm, a.seeds)
        rows = []
        for s in a.seeds:
            exp_dir, cfg = runs[s]
            name = cfg['config_name']
            th = torch.load(cw.CW / 'saved_results' / 'lagrange_multipliers' / f'{name}.pt', map_location='cpu').double().numpy()
            t = torch.load(cw.CW / 'saved_results' / 'sampling_times' / f'{name}.pt', map_location='cpu').double().numpy()
            fam = cw.family_slices(cw.build_potentials(cfg, 'cpu'))
            f_phi = cw.RESULTS / 'phi' / f'{name}.pt'
            sk = torch.load(f_phi, map_location='cpu', weights_only=False)['phi_data'].std(0, ddof=1) if f_phi.exists() else None
            # theta_t is stored one row per step after the first (len(t) - 1 rows) or per step: align the times to the last rows
            tt = t[-len(th):] if len(t) >= len(th) else np.linspace(0, 1, len(th))
            ends = {'last': len(th) - 1, 't~0.9': int(np.argmin(np.abs(tt - 0.9)))}
            if s == a.seeds[0]:
                print(f'{arm}: theta_t {th.shape}, t in [{tt[0]:.4g}, {tt[-1]:.6f}]; window "last": 1-t from {1 - tt[-a.last]:.2e} to {1 - tt[-1]:.2e}')
            for wname, e in ends.items():
                Y = th[max(e - a.last + 1, 0):e + 1]
                sig = detrended_noise(Y, a.deg)
                scale = np.median(np.abs(Y), 0)
                rel = sig / np.maximum(scale, 1e-300)
                for k, (i0, w) in fam.items():
                    r = rel[i0:i0 + w]
                    rows.append({'seed': s, 'window': wname, 'family': k, 'statistics': w,
                                 'median rel noise': np.median(r), 'p90 rel noise': np.percentile(r, 90),
                                 'noise in log p (nats)': np.sqrt(np.sum((sig[i0:i0 + w] * sk[i0:i0 + w]) ** 2)) if sk is not None else np.nan})
        D = pd.DataFrame(rows)
        T = D.groupby(['window', 'family'], sort=False)[['statistics', 'median rel noise', 'p90 rel noise', 'noise in log p (nats)']].mean()
        print(f'\n=== {arm}: theta_t noise (mean over {len(a.seeds)} seeds), polynomial trend of degree {a.deg} removed, {a.last}-step windows')
        print(T.to_string(float_format=lambda v: f'{v:.3g}'))
        T.to_csv(out / f'theta_noise_{arm}.csv')


if __name__ == '__main__':
    main()
