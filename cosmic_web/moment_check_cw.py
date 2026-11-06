#!/usr/bin/env python3
"""Moment check of ALL imposed statistics of the cosmic-web arms (data maps vs generated maps), per family.

Same metric as moment_reproduction/moment_reproduction.py, per statistic k and seed:
    z_k      = (m_k - mhat_k) / sqrt(s2_data_k / n + s2_model_k / n_s)
    e_k / floor_k = z_k^2          (~1 if reproduced, >> 1 if not)
m, s2_data = mean / unbiased variance of phi_k over the n data maps; mhat, s2_model = same over the n_s generated
maps (independent samples). offset_k = |m_k - mhat_k| / sqrt(s2_data_k) is the miss in units of the data spread.
The metric does not change under a rescaling of phi_k, so the potentials' own normalisation does not matter.

Reading: if every statistic is reproduced, z is ~ N(0, 1): 4.6 % of the statistics have |z| > 2 and 0.27 % have
|z| > 3 from noise alone. With n = n_s = 100 maps the resolution is about 0.14 data std (noise on the offset), so
this CAN say 'compatible', not 'matched to 1 %'.
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
    """z, offset (r,) and the mask of used statistics (r,) for one seed."""
    n, ns = len(phi_d), len(phi_g)
    m, mh = phi_d.mean(0), phi_g.mean(0)
    s2d, s2m = phi_d.var(0, ddof=1), phi_g.var(0, ddof=1)
    used = np.zeros(phi_d.shape[1], bool)
    for a, w in fam.values():
        v = s2d[a:a + w]
        ref = np.median(v[v > 0]) if (v > 0).any() else 0.0
        used[a:a + w] = (v > var_floor * ref) & (v > 0)
    z = np.full(len(m), np.nan); off = np.full(len(m), np.nan)
    z[used] = (m - mh)[used] / np.sqrt(s2d[used] / n + s2m[used] / ns)
    off[used] = np.abs(m - mh)[used] / np.sqrt(s2d[used])
    return z, off, used


def family_table(Z, OFF, USED, fam):
    """Per family: mean over seeds (and std over seeds) of the fractions and of the mean offset."""
    rows = {}
    for k, (a, w) in fam.items():
        az = np.abs(Z[:, a:a + w])                                   # (seeds, w), NaN = not used
        used = USED[:, a:a + w]
        nu = used.sum(1)
        f2 = (az <= 2).sum(1) / nu * 100
        f3 = ((az > 2) & (az <= 3)).sum(1) / nu * 100
        f4 = (az > 3).sum(1) / nu * 100
        off = np.nanmedian(OFF[:, a:a + w], 1)
        rows[k] = {'statistics': w, 'used': int(nu.mean()), '% |z|<=2': f2.mean(), '± (% |z|<=2)': f2.std(ddof=1) if len(f2) > 1 else np.nan,
                   '% 2<|z|<=3': f3.mean(), '% |z|>3': f4.mean(), 'median offset (data std)': off.mean(),
                   'median z^2': np.nanmedian(az ** 2)}
    return pd.DataFrame(rows).T


def plot_arm(arm, Z, fam, T, out):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    bins = np.linspace(0, 6, 61)
    xs = 0.5 * (bins[1:] + bins[:-1])
    for k, (a, w) in fam.items():
        v = np.abs(Z[:, a:a + w]).ravel(); v = v[np.isfinite(v)]
        axs[0].hist(v, bins=bins, density=True, histtype='step', lw=1.6, label=f'{k} ({len(v)})')
    axs[0].plot(xs, 2 * np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), 'k--', lw=1.2, label='perfect match: half-normal')
    axs[0].set_yscale('log'); axs[0].set_ylim(1e-4, 3)
    axs[0].set_xlabel('|z| of the statistic (miss / noise), seeds pooled'); axs[0].set_ylabel('density')
    axs[0].set_title(f'{arm}: moments, data vs generated'); axs[0].legend(fontsize=7)
    names = list(fam)
    x = np.arange(len(names))
    axs[1].bar(x - 0.2, 100 - T['% |z|<=2'].astype(float), 0.4, label='|z| > 2')
    axs[1].bar(x + 0.2, T['% |z|>3'].astype(float), 0.4, label='|z| > 3')
    axs[1].axhline(4.55, color='C0', ls='--', lw=1); axs[1].axhline(0.27, color='C1', ls='--', lw=1)
    axs[1].set_xticks(x); axs[1].set_xticklabels([n.replace('Scattering_', '').replace('_Order', '') for n in names], rotation=20, fontsize=8)
    axs[1].set_ylabel('% of the statistics'); axs[1].set_yscale('log'); axs[1].set_ylim(0.05, 100)
    axs[1].set_title('outside 2 and 3 sigmas (dashed: noise alone, 4.6 % and 0.27 %)', fontsize=10); axs[1].legend(fontsize=8)
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
        Z, OFF, USED = [], [], []
        for s in a.seeds:
            exp_dir, cfg = runs[s]
            phi_d, phi_g, fam = cw.get_phi(arm, s, exp_dir, cfg, device, a.chunk, a.data_path)
            z, off, used = moment_z(phi_d, phi_g, a.var_floor, fam)
            Z.append(z); OFF.append(off); USED.append(used)
            print(f'{arm} seed {s}: {used.sum()}/{len(used)} statistics used, {(np.abs(z[used]) > 2).mean() * 100:.1f} % with |z| > 2, '
                  f'{(np.abs(z[used]) > 3).mean() * 100:.2f} % with |z| > 3, max |z| {np.nanmax(np.abs(z)):.2f}')
        Z, OFF, USED = np.stack(Z), np.stack(OFF), np.stack(USED)
        T = family_table(Z, OFF, USED, fam)
        print(f'\n{arm}: per family (mean over {len(a.seeds)} seeds); noise alone gives 95.4 / 4.3 / 0.27 %'); print(T.to_string(float_format=lambda v: f'{v:.3g}'))
        labels = [f'{k}[{i}]' for k, (_, w) in fam.items() for i in range(w)]
        # mean |z| over seeds (a persistent miss shows in every seed); statistics left out in every seed score 0
        score = np.where(np.isnan(Z).all(0), 0.0, np.nanmean(np.where(np.isnan(Z), np.nan, np.abs(Z)), 0))
        order = np.argsort(-score)[:a.worst]
        print(f'\n{arm}: {a.worst} statistics with the largest mean |z| over seeds (a persistent miss; noise alone gives ~0.8)')
        for i in order:
            print(f'  {labels[i]:42s} mean |z| {score[i]:6.2f}   per seed {np.round(Z[:, i], 2)}')
        T.to_csv(cw.RESULTS / 'moment_check' / f'moment_check_{arm}.csv')
        np.savez_compressed(cw.RESULTS / 'moment_check' / f'moment_check_{arm}.npz', z=Z, offset=OFF, used=USED,
                            labels=np.array(labels), seeds=np.array(a.seeds), families=np.array(list(fam)))
        plot_arm(arm, Z, fam, T, cw.FIGURES / f'moment_check_{arm}.png')
        print(f'{arm}: -> {cw.FIGURES / f"moment_check_{arm}.png"}')


if __name__ == '__main__':
    main()
