#!/usr/bin/env python3
"""Var/2 tables of -log10 p / d per category of statistics, without running the notebooks: Lagrangian, Eulerian,
S&P 500 and the overall table (the same code as talk_turbulence / talk_jets / talk_snp: talk_figures.kl_table).

TWO versions, never mixed, marked ONLY in the file names (_generated / _data), nothing about it in the figures:
  generated : log p evaluated on the GENERATED signals of the model (5 seeds each)
  data      : log p evaluated on the DATASET signals (the training signals; 5 seeds = 5 different Theta on the same data)
The S&P 500 has no 'data' version: its data is a single series, there is no variance over signals to compute.

Reads
  generated : python turbulence/talk_logp_generated.py --label <label> --seed S --lam 3e-5   (S = 900 ... 904)
              python jets/logp_generated_jets.py ; snp_experiment/figures/logp_snp/logp_<arm>.npz
  data      : turbulence/figures/data/logp_C_<label>[_seed<S>]_lam3e-05.npz (logp_family_contrib.py),
              jets/figures/logp_jets/logp_jets.npz (logp_jets.py)
CPU only, seconds.

Usage (repo root, e.g. on Jean Zay in the pytorch-gpu environment):
    python codes/make_kl_tables.py                                  # generated AND data, all datasets, overall tables
    python codes/make_kl_tables.py --source generated --only snp overall
    python codes/make_kl_tables.py --no_std                          # without the shaded std(-log p) column
Output: <dataset>/figures/talk/kl_table_<dataset>_<source>.{png,csv}; overall: jets/figures/talk/kl_table_<source>.png
(--outdir DIR writes everything to one folder instead).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for p in ('codes', 'codes/filters', 'codes/potentials', 'data', 'turbulence', 'jets', 'snp_experiment'):
    sys.path.insert(0, str(ROOT / p))
import talk_figures as tf                                              # noqa: E402

SOURCE_LABEL = {'generated': 'GENERATED signals', 'data': 'DATASET signals'}
TITLES = {'lagrangian': 'Lagrangian', 'eulerian': 'Eulerian', 'snp': 'S&P 500',
          'cosmic_base': 'Cosmic web', 'cosmic_scat4': 'Cosmic web + order 4'}
FOLDER = {'lagrangian': 'turbulence', 'eulerian': 'jets', 'snp': 'snp_experiment',
          'cosmic_base': 'cosmic_web', 'cosmic_scat4': 'cosmic_web'}
COSMIC_ARM = {'cosmic_base': 'base_real', 'cosmic_scat4': 'scat4_real'}


def lagrangian(a, source):
    """Seeds a.seeds of the run a.label: per-family log p of the generated signals (talk_logp_generated.py) or of the
    dataset (logp_family_contrib.py, lam a.lam); d = M of the run."""
    from logp_family_contrib import HERE, newest, load_potentials
    from check_potentials import theta_column_map
    data = HERE / 'figures' / 'data'
    stag = lambda s: '' if s == 900 else f'_seed{s}'
    if source == 'generated':
        files = [data / f'talk_logp_generated_{a.label}{stag(s)}.npz' for s in a.seeds]
        assert all(f.exists() for f in files), f'run talk_logp_generated.py first, missing: {[f.name for f in files if not f.exists()]}'
    else:
        files = [data / f'logp_C_{a.label}{stag(s)}_lam{a.lam:g}.npz' for s in a.seeds]
        files = [f for f in files if f.exists()]
        assert files, f'no logp_C_{a.label}*_lam{a.lam:g}.npz in {data} (logp_family_contrib.py)'
    GS = [np.load(f) for f in files]
    exp = newest(f'*/*_seed_{a.seeds[0]}_*{a.label}_2*', HERE / 'experiments')
    cfg = json.loads((exp / 'config.json').read_text())
    theta = np.load(data / f'talk_logp_generated_{a.label}.npz')['theta']          # only its length (number of statistics)
    cmap = theta_column_map(load_potentials(exp, cfg, 'cpu'), len(theta))
    widths = {f: w for f, (_, w) in cmap.items()}
    fams = list(cmap)
    assert all(set(fams) <= set(g.files) for g in GS), 'files without the per-family keys'
    print(f'Lagrangian [{source}]: {len(GS)} seed file(s) | d = {int(cfg["M"])}')
    return tf.kl_table({k: np.stack([g[k] for g in GS]) for k in fams}, widths, int(cfg['M']), f'kl_table_lagrangian_{source}')


def eulerian(a, source):
    """Run a.jets_arm, 5 seeds: per-family log p of the generated signals (logp_generated_jets.py) or of the training
    signals (logp_jets.py); d = signal length."""
    import torch
    import logp_jets                                                    # puts the project folders on sys.path
    from logp_family_contrib import load_potentials, short as short_turb
    from check_potentials import theta_column_map
    from logp_seed_check import XSEED, theta_file
    from run_comparison import _find_exp_dir
    xs = torch.load(XSEED / f'xseed_{a.jets_arm}.pt', map_location='cpu', weights_only=False)
    lam = xs['selected']['end0.99']
    config, f_sel = next(iter(xs['theta_files']['end0.99'].items()))
    exp = _find_exp_dir(logp_jets.HERE, config)
    cfg = json.loads((exp / 'config.json').read_text())
    sol = torch.load(theta_file(f_sel, lam, lam), map_location='cpu', weights_only=False)
    cmap = theta_column_map(load_potentials(exp, cfg, 'cpu'), len(sol['Theta_reg'][-1]))
    raw = list(cmap)                                                    # the model's own family names (used for the categories)
    widths = {k: w for k, (_, w) in cmap.items()}
    if source == 'generated':
        f = logp_jets.HERE / 'figures' / 'logp_jets' / f'logp_generated_{a.jets_arm}.npz'
        assert f.exists(), f'run python jets/logp_generated_jets.py first: {f}'
        G = np.load(f)
        assert 'C' in G.files, 'old file without per-family log p: rerun python jets/logp_generated_jets.py'
        C, seeds, saved = G['C'], [int(s) for s in G['seeds']], [str(x) for x in G['families']]
    else:
        L = logp_jets.load(logp_jets.HERE / 'figures' / 'logp_jets' / 'logp_jets.npz')[a.jets_arm]
        C, seeds = L['C']['train'], [int(s) for s in L['seeds']]
        saved = [str(x) for x in L['families'] if str(x).lower() != 'total']
    # the saved files abbreviate the family names in two different ways (logp_jets.short / logp_family_contrib.short):
    # matched by position with the model's families, which must be the same list in the same order
    assert saved in ([logp_jets.short(k) for k in raw], [short_turb(k) for k in raw]), (saved, raw)
    assert C.shape[1] == len(raw) + 1, (C.shape, len(raw))
    assert np.allclose(C[:, :-1].sum(1), C[:, -1], rtol=1e-6, atol=1e-6 * np.abs(C[:, -1]).max()), 'TOTAL != sum of families'
    d = int(cfg.get('M') or cfg['target_len'])                          # length of the signals (256)
    print(f'Eulerian [{source}]: seeds {seeds} | d = {d}')
    return tf.kl_table({k: C[:, i] for i, k in enumerate(raw)}, widths, d, f'kl_table_eulerian_{source}')


def snp(a, source):
    """Per-family log p of the generated series (snp_experiment/logp_snp.py), d = length of the series (6062)."""
    if source == 'data':
        return None                                                     # one series: no variance over signals
    f = Path(a.snp_npz) if a.snp_npz else ROOT / 'snp_experiment' / 'figures' / 'logp_snp' / f'logp_{a.snp_arm}.npz'
    assert f.exists(), f'missing {f} (python snp_experiment/logp_snp.py)'
    D = np.load(f)
    if a.snp_d:
        d = a.snp_d
    else:
        from snp_loader import load_snp_returns
        d = int(load_snp_returns()[0].numel())
    fm = [str(x) for x in D['families']]
    print('S&P 500 [generated]:', f.name, '| seeds', [int(s) for s in D['seeds']], '| d =', d)
    return tf.kl_table({k: D['C_gen'][:, i] for i, k in enumerate(fm)}, {k: int(w) for k, w in zip(fm, D['widths'])},
                       d, 'kl_table_snp_generated')


def cosmic(key, a, source):
    """Cosmic web (Quijote 2D), arm COSMIC_ARM[key]: per-family log p of the generated maps or of the 100 data maps
    (cosmic_web/logp_cw.py, 5 seeds), d = number of pixels (128 x 128). The 'scalar multiscale' column = order 1 + 2."""
    f = Path(os.environ.get('CW_ROOT', ROOT / 'cosmic_web')) / 'results' / 'logp' / f'logp_{COSMIC_ARM[key]}.npz'
    if not f.exists():
        print(f'{TITLES[key]}: MISSING {f} (python cosmic_web/logp_cw.py), skipped')
        return None
    D = np.load(f)
    fm = [str(x) for x in D['families']]
    C = D['C_gen'] if source == 'generated' else D['C_data']             # (seeds, families + total, signals)
    d = int(D['M']) ** 2
    print(f'{TITLES[key]} [{source}]:', f.name, '| seeds', [int(s) for s in D['seeds']], f'| lam {float(D["lam"]):g} | d = {d}')
    return tf.kl_table({k: C[:, i] for i, k in enumerate(fm)}, {k: int(w) for k, w in zip(fm, D['widths'])},
                       d, f'kl_table_{key}_{source}')


BUILD = {'lagrangian': lagrangian, 'eulerian': eulerian, 'snp': snp,
         'cosmic_base': lambda a, source: cosmic('cosmic_base', a, source),
         'cosmic_scat4': lambda a, source: cosmic('cosmic_scat4', a, source)}
ALL_KEYS = ('lagrangian', 'eulerian', 'snp', 'cosmic_base', 'cosmic_scat4')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', nargs='+', choices=['generated', 'data'], default=['generated', 'data'])
    ap.add_argument('--only', nargs='+', choices=list(ALL_KEYS) + ['overall'],
                    default=list(ALL_KEYS) + ['overall'])
    ap.add_argument('--label', default='turbulence_nested_M3_bicw_allmass_scat4')
    ap.add_argument('--seeds', type=int, nargs='+', default=[900, 901, 902, 903, 904])
    ap.add_argument('--lam', type=float, default=3e-5, help='lam of the Lagrangian data files logp_C_*_lam<lam>.npz')
    ap.add_argument('--jets_arm', default='jets_full')
    ap.add_argument('--snp_arm', default='snp_M1')
    ap.add_argument('--snp_npz', default=None, help='explicit logp_snp npz (default snp_experiment/figures/logp_snp/...)')
    ap.add_argument('--snp_d', type=int, default=None, help='length of the S&P series (default: read from the data, 6062)')
    ap.add_argument('--outdir', type=Path, default=None, help='one output folder for everything (default: <dataset>/figures/talk)')
    ap.add_argument('--no_std', action='store_true', help='no shaded std(-log p) column: only the Var/2 of the categories')
    a = ap.parse_args()

    for source in a.source:
        tag = SOURCE_LABEL[source]
        print(f'\n================ {tag} ================')
        tables = {}
        for key in ALL_KEYS:
            folder = a.outdir or ROOT / FOLDER[key] / 'figures' / 'talk'
            if key in a.only:
                tf.set_output(folder)
                T = BUILD[key](a, source)
                if T is None:
                    print(f'{TITLES[key]}: no {source} table (S&P: a single series has no variance over signals; cosmic web: file missing)')
                    continue
                tf.plot_kl_table_single(T, f'kl_table_{key}_{source}', TITLES[key], std=not a.no_std)   # source only in the file name
                tables[TITLES[key]] = T
            elif 'overall' in a.only and (not (key == 'snp' and source == 'data')):    # only the overall: read the saved CSV
                f = folder / f'kl_table_{key}_{source}.csv'
                if f.exists():
                    tables[TITLES[key]] = pd.read_csv(f, index_col=0)
                else:
                    print(f'{TITLES[key]}: MISSING {f} (run with --only {key} --source {source})')
        if 'overall' in a.only:
            if not tables:
                print(f'overall table [{source}]: nothing to stack')
            else:
                tf.set_output(a.outdir or ROOT / 'jets' / 'figures' / 'talk')
                tf.plot_kl_table(tables, f'kl_table_{source}', std=not a.no_std)                      # source only in the file name
                tf.plot_confinement_table(tables, f'kl_table_confinement_{source}')                  # confinement apart
                print('overall table:', list(tables))
        for t, T in tables.items():
            print(f'\n{t}'); print(T.to_string())


if __name__ == '__main__':
    main()
