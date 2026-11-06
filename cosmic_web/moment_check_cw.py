#!/usr/bin/env python3
"""Moment check of ALL imposed statistics of the cosmic-web arms (data maps vs generated maps), per family.

Same metric as moment_reproduction/moment_reproduction.py, per statistic k and seed:
    z_k      = (m_k - mhat_k) / sqrt(s2_data_k / n + s2_model_k / n_s)
    e_k / floor_k = z_k^2          (~1 if reproduced, >> 1 if not)
m, s2_data = mean / unbiased variance of phi_k over the n data maps; mhat, s2_model = same over the n_s generated
maps (independent samples). offset_k = |m_k - mhat_k| / sqrt(s2_data_k) is the miss in units of the data spread.
The metric does not change under a rescaling of phi_k, so the potentials' own normalisation does not matter.

Reading: for independent samples z would be ~ N(0, 1) (4.6 % of the statistics beyond 2, 0.27 % beyond 3) and the
offset would be ~ 0.14 data std (the 'sampling noise', n = n_s = 100). MGD pins the empirical mean of the particles to
the data mean (final moment error ~1e-7, see moment_matching.png), so the offsets are expected FAR below that noise:
this check then confirms that every imposed statistic (also the ones the run's own plot hides) was matched; it does
not test the population moments of the model (that would need samples independent of the fit).
Statistics with (almost) no variance over the data maps (structural zeros: the check is meaningless) are left out
and counted: s2_data < --var_floor x the median variance of their family.

Usage (repo root, GPU node on Jean Zay, minutes):  python cosmic_web/moment_check_cw.py --arms base_real scat4_real
Output: cosmic_web/results/moment_check/moment_check_<arm>.{csv,npz}, cosmic_web/figures/moment_check_<arm>.png
"""
import argparse
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import cw_common as cw

warnings.filterwarnings('ignore', message='Mean of empty slice')


def moment_z(phi_d, phi_g, var_floor, fam):
    """z, offset, noise (r,) and the mask of used statistics (r,) for one seed. noise = the offset that independent
    sampling noise would give (std of m - mhat in data std); MGD pins the particles' mean to the data mean, so the
    offsets are expected far BELOW it."""
    n, ns = len(phi_d), len(phi_g)
    m, mh = phi_d.mean(0), phi_g.mean(0)
    s2d, s2m = phi_d.var(0, ddof=1), phi_g.var(0, ddof=1)
    used = np.zeros(phi_d.shape[1], bool)
    for a, w in fam.values():
        v = s2d[a:a + w]
        ref = np.median(v[v > 0]) if (v > 0).any() else 0.0
        used[a:a + w] = (v > var_floor * ref) & (v > 0)
    z = np.full(len(m), np.nan); off = np.full(len(m), np.nan); noise = np.full(len(m), np.nan)
    z[used] = (m - mh)[used] / np.sqrt(s2d[used] / n + s2m[used] / ns)
    off[used] = np.abs(m - mh)[used] / np.sqrt(s2d[used])
    noise[used] = np.sqrt(s2d[used] / n + s2m[used] / ns) / np.sqrt(s2d[used])
    return z, off, noise, used


def family_table(Z, OFF, NOISE, USED, fam):
    """Per family (mean over seeds): the offset in data std (median, max), its ratio to the sampling noise, and the
    fractions of |z| <= 2 / 2-3 / > 3 (the z columns only say 'compatible with noise': MGD does much better)."""
    rows = {}
    for k, (a, w) in fam.items():
        az = np.abs(Z[:, a:a + w])                                   # (seeds, w), NaN = not used
        used = USED[:, a:a + w]
        nu = used.sum(1)
        f2 = (az <= 2).sum(1) / nu * 100
        f3 = ((az > 2) & (az <= 3)).sum(1) / nu * 100
        f4 = (az > 3).sum(1) / nu * 100
        off = np.nanmedian(OFF[:, a:a + w], 1)
        o = OFF[:, a:a + w]
        rows[k] = {'statistics': w, 'used': int(nu.mean()), 'median offset (data std)': off.mean(),
                   'max offset (data std)': np.nanmax(o, 1).mean(), 'median noise (data std)': np.nanmedian(NOISE[:, a:a + w]),
                   'max offset / noise': np.nanmax(o / NOISE[:, a:a + w]),
                   '% |z|<=2': f2.mean(), '% 2<|z|<=3': f3.mean(), '% |z|>3': f4.mean()}
    return pd.DataFrame(rows).T


def plot_arm(arm, OFF, NOISE, fam, T, out):
    """Left: histogram of the offset |m - mhat| / std_data of every statistic (seeds pooled), per family; dashed line =
    the offset independent sampling noise would give. Right: median and max offset per family against that noise."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    noise = float(np.nanmedian(NOISE))
    allv = OFF[np.isfinite(OFF) & (OFF > 0)]
    lo = np.floor(np.log10(allv.min())) - 0.5 if allv.size else -10
    bins = np.logspace(lo, np.ceil(np.log10(max(allv.max(), noise))) + 0.5, 60)
    for k, (a, w) in fam.items():
        v = OFF[:, a:a + w].ravel(); v = v[np.isfinite(v) & (v > 0)]
        axs[0].hist(v, bins=bins, histtype='step', lw=1.6, label=f'{k} ({len(v)})')
    axs[0].axvline(noise, color='k', ls='--', lw=1.2, label=f'sampling noise of independent maps ({noise:.2f})')
    axs[0].set_xscale('log'); axs[0].set_yscale('log')
    axs[0].set_xlabel('offset |data mean - generated mean| / data std, seeds pooled'); axs[0].set_ylabel('statistics')
    axs[0].set_title(f'{arm}: all imposed moments, data vs generated maps', fontsize=10); axs[0].legend(fontsize=7)
    names = list(fam)
    x = np.arange(len(names))
    axs[1].bar(x - 0.2, T['median offset (data std)'].astype(float), 0.4, label='median')
    axs[1].bar(x + 0.2, T['max offset (data std)'].astype(float), 0.4, label='max')
    axs[1].axhline(noise, color='k', ls='--', lw=1.2)
    axs[1].set_xticks(x); axs[1].set_xticklabels([n.replace('Scattering_', '').replace('_Order', '') for n in names], rotation=20, fontsize=8)
    axs[1].set_ylabel('offset (data std)'); axs[1].set_yscale('log')
    axs[1].set_title('per family (dashed: sampling noise)', fontsize=10); axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arms', nargs='+', default=['base_real', 'scat4_real'])
    ap.add_argument('--seeds', nargs='+', type=int, default=cw.SEEDS)
    ap.add_argument('--var_floor', type=float, default=1e-8, help='statistic dropped if s2_data < var_floor x median s2_data of its family')
    ap.add_argument('--worst', type=int, default=20, help='how many worst statistics to list')
    ap.add_argument('--data_path', default=None)
    ap.add_argument('--chunk', type=int, default=10)
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    (cw.RESULTS / 'moment_check').mkdir(parents=True, exist_ok=True); cw.FIGURES.mkdir(parents=True, exist_ok=True)

    for arm in a.arms:
        runs = cw.find_runs(arm, a.seeds)
        Z, OFF, NOISE, USED = [], [], [], []
        for s in a.seeds:
            exp_dir, cfg = runs[s]
            phi_d, phi_g, fam = cw.get_phi(arm, s, exp_dir, cfg, device, a.chunk, a.data_path)
            z, off, noise, used = moment_z(phi_d, phi_g, a.var_floor, fam)
            Z.append(z); OFF.append(off); NOISE.append(noise); USED.append(used)
            print(f'{arm} seed {s}: {used.sum()}/{len(used)} statistics used | offset in data std: median {np.nanmedian(off):.2e}, '
                  f'max {np.nanmax(off):.2e} | sampling noise {np.nanmedian(noise):.2f} | {(np.abs(z[used]) > 2).mean() * 100:.1f} % with |z| > 2')
        Z, OFF, NOISE, USED = np.stack(Z), np.stack(OFF), np.stack(NOISE), np.stack(USED)
        T = family_table(Z, OFF, NOISE, USED, fam)
        print(f'\n{arm}: per family (mean over {len(a.seeds)} seeds). Offset = |data mean - generated mean| / data std; MGD pins '
              f'the particles to the data mean, so it is expected far below the sampling noise of independent maps')
        print(T.to_string(float_format=lambda v: f'{v:.3g}'))
        labels = [f'{k}[{i}]' for k, (_, w) in fam.items() for i in range(w)]
        # largest offset over seeds (a persistent miss shows in every seed); statistics left out in every seed score 0
        score = np.where(np.isnan(OFF).all(0), 0.0, np.nanmax(np.where(np.isnan(OFF), -np.inf, OFF), 0))
        order = np.argsort(-score)[:a.worst]
        print(f'\n{arm}: {a.worst} statistics with the largest offset (max over seeds, data std)')
        for i in order:
            print(f'  {labels[i]:42s} max offset {score[i]:9.2e}   per seed {np.array2string(OFF[:, i], precision=1)}')
        T.to_csv(cw.RESULTS / 'moment_check' / f'moment_check_{arm}.csv')
        np.savez_compressed(cw.RESULTS / 'moment_check' / f'moment_check_{arm}.npz', z=Z, offset=OFF, noise=NOISE, used=USED,
                            labels=np.array(labels), seeds=np.array(a.seeds), families=np.array(list(fam)))
        plot_arm(arm, OFF, NOISE, fam, T, cw.FIGURES / f'moment_check_{arm}.png')
        print(f'{arm}: -> {cw.FIGURES / f"moment_check_{arm}.png"}')


if __name__ == '__main__':
    main()
