#!/usr/bin/env python3
"""H_G = D_KL(p || q_G) for all groups of a dataset, from the cached energies of cache_energies.py (CPU, minutes).

Groups (L_6 / confinement never removed):
  category : summed over all families of one kind (scalar, scat1-2, scat3_real/imag, scat4_real/imag)
  family   : one potential family of the model
  single   : one statistic
  pair     : order-4 (j, j') with all l (1D: j < j' < l; 2D: lambda = j*L + angle, ASSUMED j = lambda // L, all angles and
             all second-wavelet (l, angle) summed; j = j' rows split into 'same' / 'cross' angle as well)
Output: kl_groups/results/H_<dataset>.csv (one row per seed x group) and H_vs_N_<dataset>.npz (top groups per seed).
Usage:  python kl_groups/compute_H.py --dataset lagrangian
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from kl_groups import H_estimate, H_vs_N

RES = Path(os.environ.get('KL_RESULTS', Path(__file__).resolve().parent / 'results'))
CONFINEMENT = ('L_6', 'L6', 'L2p')


def category(fam):
    if any(c in fam for c in CONFINEMENT):
        return None
    if 'Scalar' in fam:
        return 'scalar'
    for key, cat in (('Fourth', 'scat4'), ('Third', 'scat3')):
        if key in fam:
            return f'{cat}_{"imag" if "Imag" in fam else "real"}'
    return 'scat1-2'


def pair_groups(fam, start, idx, L):
    g = {}
    for i, (a, b, c) in enumerate(np.asarray(idx).T):
        if L is None:
            keys = [(int(a), int(b), 'all')]
        else:
            j, jp = int(a) // L, int(b) // L
            keys = [(j, jp, 'all')] + ([(j, jp, 'same' if a % L == b % L else 'cross')] if j == jp else [])
        for k in keys:
            g.setdefault(k, []).append(start + i)
    return g


def main(dataset, n_boot, in_dir=None):
    rows, curves = [], {}
    src = Path(in_dir) if in_dir else RES                       # energies E_*.npz read here; CSV / npz written to RES
    for f in sorted(src.glob(f'E_{dataset}_seed*.npz')):
        D = np.load(f)
        seed, E = int(f.stem.split('seed')[1]), D['E']
        L = int(D['L']) if 'L' in D.files else None
        fams = [str(x) for x in D['families']]
        cmap = {k: (int(s), int(w)) for k, s, w in zip(fams, D['starts'], D['widths'])}
        print(f'{f.name}: N = {E.shape[0]}, n = {E.shape[1]}; families: ' +
              ', '.join(f'{k} -> {category(k)}' for k in fams), flush=True)
        groups = {}                                                  # name -> (kind, family, j, j', angle, columns)
        for cat in sorted({category(k) for k in fams} - {None}):
            cols = np.concatenate([np.arange(s, s + w) for k, (s, w) in cmap.items() if category(k) == cat])
            groups[f'cat:{cat}'] = ('category', cat, -1, -1, '', cols)
        for k, (s, w) in cmap.items():
            if category(k) is None:
                continue
            groups[f'fam:{k}'] = ('family', k, -1, -1, '', np.arange(s, s + w))
            idx = D[f'idx_{k}'] if f'idx_{k}' in D.files else None
            for i in range(w):
                lab = '' if idx is None else ','.join(str(int(v)) for v in idx[:, i])
                groups[f'one:{k}[{i}]'] = ('single', k, -1, -1, lab, np.array([s + i]))
            if idx is not None:
                for (j, jp, ang), cols in pair_groups(k, s, idx, L).items():
                    groups[f'pair:{k}({j},{jp},{ang})'] = ('pair', k, j, jp, ang, np.array(cols))
        for name, (kind, fam, j, jp, lab, cols) in groups.items():
            o = H_estimate(E[:, cols].sum(1), n_boot=n_boot if kind != 'single' else n_boot // 2)
            rows.append(dict(dataset=dataset, seed=seed, group=name, kind=kind, family=fam, category=category(fam) or fam,
                             j=j, jp=jp, label=lab, n_terms=len(cols), **o))
        # H vs N for the 5 largest category / family / pair groups of this seed
        sub = sorted([r for r in rows if r['seed'] == seed and r['kind'] != 'single'], key=lambda r: -r['H'])[:5]
        N = E.shape[0]
        Ns = [n for n in (N // 32, N // 16, N // 8, N // 4, N // 2) if n >= 50]
        for r in sub:
            c = E[:, groups[r['group']][5]].sum(1)
            curves[f'{seed}|{r["group"]}'] = np.vstack([H_vs_N(c, Ns), [N, r['H'], r['boot_std']]])
    assert rows, f'no E_{dataset}_seed*.npz in {src}'
    df = pd.DataFrame(rows)
    df.to_csv(RES / f'H_{dataset}.csv', index=False)
    np.savez(RES / f'H_vs_N_{dataset}.npz', **curves)
    show = df[df.kind.isin(['category', 'family'])].groupby('group')[['H', 'boot_std', 'var2', 'ess_frac', 'ln_N_needed', 'flag']].mean()
    print(show.sort_values('H', ascending=False).to_string(float_format=lambda v: f'{v:.4g}'))
    print(f'-> {RES / f"H_{dataset}.csv"} ({len(df)} rows), flagged groups: {int(df.flag.sum())}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--n_boot', type=int, default=200)
    ap.add_argument('--in_dir', default=None, help='folder with the E_<dataset>_seed*.npz (default: KL_RESULTS)')
    a = ap.parse_args()
    main(a.dataset, a.n_boot, a.in_dir)
