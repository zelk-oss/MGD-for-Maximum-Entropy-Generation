"""Per-family interpretation tables of the pinned-L_6 test (2026-11-17), one set per arm, same code as before:

  1. rare vs typical table per family (turbulence/logp_family_contrib.py): C_F(x) = theta_F . phi_F(x) on the DATA,
     rare = max |lag-1 increment| above the P-th percentile (P = 95, 99), D_F = mean(rare) - mean(typical), its share
     D_F / D_TOTAL and the bootstrap SE; per seed, then mean +- std over the seeds; plus one comparison table (all arms)
  2. histograms of C_F rare vs typical (logp_family_contrib.plot_shapes), seed 900, one figure per arm and P
  3. Var/2d tables (codes/talk_figures.kl_table + plot_kl_table_single, as codes/make_kl_tables.py): per category of
     statistics Var(ln p_k) / (2 d) on the GENERATED signals and on the DATA, mean +- std over the seeds

Theta = Theta_reg at the last node of each arm (free, pinall, pinall10, pinpos, pnewton), read from the
logp_scan.npz files written by pin_l6_test.py (keys 'theta_last|<arm>'). log p = +Theta . phi (code sign).

Usage (repo root on Jean Zay, GPU node):
    python -u theta_sign_constraint/pin_l6_family_tables.py --results theta_sign_constraint/results/pin_l6_test/pinl6_bscat4 \
        --exp_root turbulence/experiments --samples turbulence/saved_results/samples
Output: <results>/family_tables/ (csv, png).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'codes', ROOT / 'turbulence'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from logp_family_contrib import load_potentials, phi_matrix, plot_shapes, short   # noqa: E402  (puts codes/ on sys.path)
from check_potentials import theta_column_map                                     # noqa: E402
from check_fit_offline import build_data                                          # noqa: E402
import talk_figures as tf                                                         # noqa: E402

ARM_ORDER = ['free', 'pinall', 'pinall10', 'pinpos', 'pnewton']


def family_table(C, inc, P, n_boot=500):
    """As logp_family_contrib.main, one seed: rows = families + TOTAL."""
    rare = inc > np.percentile(inc, P)
    rows = {}
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(rare)
    boot_idx = [rng.choice(idx, idx.size) for _ in range(n_boot)]
    for F, c in C.items():
        r, t = c[rare], c[~rare]
        rows[F] = {'typ mean': t.mean(), 'typ std': t.std(ddof=1), 'rare mean': r.mean(), 'rare std': r.std(ddof=1),
                   'D = rare-typ': r.mean() - t.mean(),
                   'SE(D)': np.std([c[b].mean() - t.mean() for b in boot_idx], ddof=1)}
    df = pd.DataFrame(rows).T
    df['D / D_TOTAL'] = df['D = rare-typ'] / df.loc['TOTAL', 'D = rare-typ']
    return df, rare


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, required=True, help='pin_l6_test results folder (seed<N>/ inside)')
    ap.add_argument('--exp_root', type=Path, required=True, help='experiments folder of the runs')
    ap.add_argument('--samples', type=Path, required=True, help='saved_results/samples folder of the runs')
    ap.add_argument('--percentiles', type=float, nargs='+', default=[95, 99])
    ap.add_argument('--hist_seed', type=int, default=900, help='seed whose histograms are drawn')
    a = ap.parse_args()
    out = a.results / 'family_tables'
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed_dirs = sorted(d for d in a.results.glob('seed*') if (d / 'logp_scan.npz').exists())
    assert seed_dirs, f'no seed*/logp_scan.npz in {a.results}'
    C = {}                                     # C[arm][source] = {family: [per-seed arrays]}
    tables = {}                                # tables[(arm, P)] = [per-seed DataFrames]
    widths, d = None, None
    for sd in seed_dirs:
        seed = int(sd.name[4:])
        summ = json.loads((sd / 'summary.json').read_text())
        config = summ['config']
        Z = np.load(sd / 'logp_scan.npz')
        thetas = {k.split('|', 1)[1]: Z[k] for k in Z.files if k.startswith('theta_last|')}
        exp = [p for p in a.exp_root.glob(f'*/{config}') if p.is_dir()]
        assert len(exp) == 1, (config, exp)
        cfg = json.loads((exp[0] / 'config.json').read_text())
        pots = load_potentials(exp[0], cfg, device)
        x1, _ = build_data(int(cfg.get('subseries_len', 512)), int(cfg['M']), int(cfg['n1']), 1)
        xg = torch.load(a.samples / f'{config}.pt', map_location='cpu', weights_only=False).float()
        xg = xg.reshape(xg.shape[0], 1, -1)
        inc = (x1[..., 1:] - x1[..., :-1]).abs().amax(dim=(1, 2)).numpy()          # as logp_family_contrib.py
        phi = {'data': phi_matrix(pots, x1.to(device)), 'generated': phi_matrix(pots, xg.to(device))}
        r = phi['data'].shape[1]
        cmap = theta_column_map(pots, r)
        widths, d = {F: w for F, (_, w) in cmap.items()}, int(cfg['M'])
        print(f'seed {seed}: {config}\n  arms {list(thetas)}; families ' + ', '.join(f'{short(F)}[{s}:{s + w}]' for F, (s, w) in cmap.items()),
              flush=True)
        for arm, th in thetas.items():
            assert th.shape == (r,), (arm, th.shape, r)
            for src, ph in phi.items():
                cf = {F: ph[:, s:s + w] @ th[s:s + w] for F, (s, w) in cmap.items()}
                for F, v in cf.items():
                    C.setdefault(arm, {}).setdefault(src, {}).setdefault(F, []).append(v)
            cd = {F: phi['data'][:, s:s + w] @ th[s:s + w] for F, (s, w) in cmap.items()}
            cd['TOTAL'] = phi['data'] @ th
            for P in a.percentiles:
                df, rare = family_table(cd, inc, P)
                tables.setdefault((arm, P), []).append(df)
                if seed == a.hist_seed:
                    plot_shapes(cd, rare, P, f'{arm} (seed {seed})', out / f'hist_rare_typical_{arm}_seed{seed}_p{P:g}.png')

    arms = [x for x in ARM_ORDER if x in C] + [x for x in C if x not in ARM_ORDER]
    # 1. rare vs typical: per arm mean +- std over seeds; comparison of the shares across arms
    for P in a.percentiles:
        comp = {}
        for arm in arms:
            dfs = tables[(arm, P)]
            mean = sum(dfs) / len(dfs)
            std = (sum((df - mean) ** 2 for df in dfs) / max(len(dfs) - 1, 1)) ** 0.5
            mean.rename(index=short).to_csv(out / f'rare_typical_{arm}_p{P:g}_mean.csv')
            std.rename(index=short).to_csv(out / f'rare_typical_{arm}_p{P:g}_std_over_seeds.csv')
            comp[f'{arm} D'] = mean['D = rare-typ']
            comp[f'{arm} share'] = mean['D / D_TOTAL']
            comp[f'{arm} share sd'] = std['D / D_TOTAL']
        comp = pd.DataFrame(comp).rename(index=short)
        comp.to_csv(out / f'rare_typical_compare_p{P:g}.csv')
        with pd.option_context('display.float_format', '{:,.4g}'.format, 'display.width', 250):
            print(f'\n=== rare (max |lag-1 increment| > {P:g}th pct) vs typical, DATA, mean over {len(seed_dirs)} seed(s): '
                  f'D_F = rare - typical (nats) and share D_F / D_TOTAL (sd over seeds) ===')
            print(comp.to_string())

    # 3. Var/2d tables, generated and data, one per arm
    tf.set_output(out)
    for src in ('generated', 'data'):
        rows = {}
        for arm in arms:
            name = f'kl_table_lagrangian_{arm}_{src}'
            print(f'\n--- Var/2d [{arm}, {src}] ---')
            T = tf.kl_table({F: np.stack(v) for F, v in C[arm][src].items()}, widths, d, name)
            tf.plot_kl_table_single(T, name, f'Lagrangian, {arm}', std=True)
            rows[arm] = T.loc['Var/2'].copy()
            rows[arm][tf.STD_COL] = T.loc['std of total', tf.STD_COL]       # kl_table puts it in its own row
            rows[f'{arm} ±'] = T.loc['± Var/2']
        cmpT = pd.DataFrame(rows).T
        cmpT.to_csv(out / f'kl_table_compare_{src}.csv')
        with pd.option_context('display.float_format', '{:,.4g}'.format, 'display.width', 250):
            print(f'\n=== Var/2d per category, {src} signals (nats per dimension), all arms ===')
            print(cmpT.to_string())
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
