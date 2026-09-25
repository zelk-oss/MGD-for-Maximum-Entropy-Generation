"""Which (wavelet channel, magnitude region) is each Scalar_*_gaussianK statistic of a run?

Scalar_GGD_KRegion (codes/potentials/potentials_1d.py) splits |W_j x| of every channel j
into K magnitude regions (region 0 = core, higher = tails; cuts fitted on the data) and
keeps statistic (j, k) only if the region is populated; its local index i maps to
flat = active_flat[i] = k * J + j. This reads the run's fitted state, so it must run where
experiments/<...>/<config>/fitted_potentials/ exists (Jean Zay).

Usage (from turbulence/):
    python decode_scalar_stats.py LABEL [--indices psi:4,5,6 morlet:1,2,13]
    (no --indices: print the whole table for both potentials)
"""
import argparse
import glob
from pathlib import Path

import torch

NAMES = {'psi': 'Scalar_psi_gaussianK', 'morlet': 'Scalar_morlet_gaussianK'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('label')
    ap.add_argument('--seed', type=int, default=900)
    ap.add_argument('--indices', nargs='*', default=[],
                    help='e.g. psi:4,5,6 morlet:1,2,13 (local indices, as in the --top table)')
    args = ap.parse_args()

    dirs = glob.glob(f'experiments/*/*_seed_{args.seed}_*_{args.label}_*/fitted_potentials')
    assert len(dirs) == 1, f'expected one run, found {dirs}'
    print(dirs[0])
    wanted = {NAMES[k]: [int(v) for v in vals.split(',')]
              for k, vals in (s.split(':') for s in args.indices)}

    for name in NAMES.values():
        f = Path(dirs[0]) / f'{name}.pt'
        if not f.exists():
            print(f'  {name}: no fitted state'); continue
        d = torch.load(f, map_location='cpu', weights_only=False)
        J, K, flat = int(d['J']), int(d['K']), d['active_flat'].tolist()
        cuts, pi, alpha = d['cuts'], d['pi'], d.get('alpha')
        print(f'\n{name}: {len(flat)} statistics, {J} channels x {K} regions')
        print(f'  {"local":>5} {"channel j":>9} {"region k":>8} {"|Wx| range":>22} {"data share":>10} {"alpha":>6}')
        for i in (wanted.get(name) or range(len(flat))):
            k, j = divmod(flat[i], J)
            edges = [0.0] + cuts[j].tolist() + [float('inf')]
            a = f'{float(alpha[j, k]):.2f}' if alpha is not None else '-'
            print(f'  {i:5d} {j:9d} {k:8d} {f"[{edges[k]:.3g}, {edges[k + 1]:.3g})":>22} '
                  f'{float(pi[j, k]):10.1%} {a:>6}')


if __name__ == '__main__':
    main()
