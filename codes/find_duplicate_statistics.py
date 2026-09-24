"""Find redundant statistics in a run's regularised system, and name them.

Reads a system saved with --save_reg_system (per-node M_k = E[grad phi grad phi^T]
and G_k = E[phi phi^T]) and, at --nodes evenly spaced nodes:

* groups of statistics whose Jacobi-scaled M_k correlation is |c| > 1 - tol
  (duplicates up to sign/scale), split into PERSISTENT groups (present at every
  node checked -- candidates for "identical by construction", removable without
  losing anything) and TRANSIENT ones (only at some t -- keep them);
* the null space of the scaled M_k (eigenvalues < --null_tol): linear dependencies
  that are not simple pairs, with their largest components.

Indices are mapped to potential names by rebuilding the run's fitted potentials
exactly as SDE.__init__ does (get_1d_potentials + fitted_potentials/ state), from
the run's experiments/<group>/<config>/config.json. For fourth-order scattering
statistics the (j, s1, s2) scales are printed too.

Usage (from the project root):
    python codes/find_duplicate_statistics.py SYSTEM.pt --root turbulence
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
for p in (project_root, project_root / 'codes', project_root / 'data'):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from codes.potentials_builder import get_1d_potentials          # noqa: E402
from codes.filters_bank import return_Filters                   # noqa: E402
from codes.effective_dimension import load_fitted_potentials    # noqa: E402
from codes.check_potentials import _decode_scattering_jsk       # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('system', type=Path, help='file written by --save_reg_system')
    p.add_argument('--root', type=Path, required=True,
                   help='experiment root holding experiments/ (e.g. turbulence)')
    p.add_argument('--nodes', type=int, default=25, help='nodes checked, evenly spaced in t')
    p.add_argument('--tol', type=float, default=1e-9, help='|corr| > 1 - tol counts as duplicate')
    p.add_argument('--null_tol', type=float, default=1e-10, help='null-space eigenvalue cutoff')
    return p.parse_args()


def scaled(X):
    X = X.double(); X = (X + X.T) / 2
    s = torch.diagonal(X).clamp_min(1e-300).sqrt()
    return X / (s[:, None] * s[None, :])


def groups_of(C, tol):
    """Connected components of |C_ij| > 1 - tol (i != j), as sorted tuples."""
    r = C.shape[0]
    adj = (C.abs() > 1 - tol).cpu().numpy()
    np.fill_diagonal(adj, False)
    seen, out = set(), []
    for i in range(r):
        if i in seen or not adj[i].any():
            continue
        stack, comp = [i], set()
        while stack:
            k = stack.pop()
            if k in comp:
                continue
            comp.add(k); stack.extend(np.nonzero(adj[k])[0].tolist())
        seen |= comp
        out.append(tuple(sorted(comp)))
    return out


def statistic_labels(root, config):
    """index -> 'Potential[local]' (+ scales), from the run's fitted potentials."""
    exp_dirs = list((root / 'experiments').glob(f'*/{config}')) + list((root / 'experiments').glob(config))
    if not exp_dirs:
        return None, f'no experiments/*/{config} under {root}'
    exp_dir = exp_dirs[0]
    cfg = json.load(open(exp_dir / 'config.json'))
    M, J, Q = int(cfg['M']), int(cfg['J']), int(cfg['Q'])
    filters, filters_Phi = return_Filters(M, J, 1, device='cpu', include_phi=True)
    filters_Q = return_Filters(M, J, Q, device='cpu')
    pots = get_1d_potentials(cfg['terms'], J, filters, Q, filters_Q=filters_Q,
                             filters_Phi=filters_Phi, scalar_param=None, parallel=False,
                             deduplicate_filters=bool(cfg.get('deduplicate_filters', False)))
    pots = load_fitted_potentials(pots, exp_dir / 'fitted_potentials')
    labels = []
    for name, pot in pots.items():
        n = int(pot.num_coefficients)
        extra = [''] * n
        if 'Scattering_Fourth_Order' in name:
            try:
                j, s1, s2 = _decode_scattering_jsk(pot)
                extra = [f' (j={a}, s1={b}, s2={c})' for a, b, c in zip(j, s1, s2)]
            except Exception:
                pass
        labels += [f'{name}[{i}]{extra[i]}' for i in range(n)]
    return labels, exp_dir


def main():
    args = parse_args()
    # mmap: only the ~--nodes blocks used are read from disk (the file is ~25 GB;
    # loading it whole gets killed on a login node)
    system = torch.load(args.system, map_location='cpu', weights_only=False, mmap=True)
    config = args.system.name[:-len('.pt')]
    r, t = system['num_potentials'], system['t'].double().numpy()
    labels, src = statistic_labels(args.root, config)
    if labels is None:
        print(f'(no names: {src})'); labels = [f'#{i}' for i in range(r)]
    elif len(labels) != r:
        print(f'(rebuilt potentials give r={len(labels)} != system r={r}; names unreliable)')
        labels = [f'#{i}' for i in range(r)]
    else:
        print(f'names from {src}')

    idx = sorted(set(int(i) for i in np.linspace(0, len(t) - 1, args.nodes)))
    seen_at = {}                       # group -> nodes where present
    nulls = []
    for k in idx:
        CM = scaled(system['M'][k])
        for g in groups_of(CM, args.tol):
            seen_at.setdefault(g, []).append(k)
        ev, V = torch.linalg.eigh(CM)
        for m in torch.nonzero(ev < args.null_tol).flatten().tolist():
            v = V[:, m]
            top = torch.argsort(v.abs(), descending=True)[:6]
            nulls.append((k, float(ev[m]), [(int(i), float(v[i])) for i in top if abs(v[i]) > 0.05]))

    print(f'\nr = {r}, {len(idx)} nodes checked (t = {t[idx[0]]:.4f} .. {t[idx[-1]]:.4f})')
    persistent = {g: n for g, n in seen_at.items() if len(n) == len(idx)}
    transient = {g: n for g, n in seen_at.items() if len(n) < len(idx)}
    print(f'\nPERSISTENT duplicate groups (every node) -- removal candidates: {len(persistent)}')
    for g in persistent:
        print('  group:'); [print(f'    {i:4d}  {labels[i]}') for i in g]
    print(f'\nTRANSIENT duplicate groups (some nodes only -- keep): {len(transient)}')
    for g, n in transient.items():
        print(f'  group at {len(n)}/{len(idx)} nodes, t in [{t[min(n)]:.4f}, {t[max(n)]:.4f}]:')
        [print(f'    {i:4d}  {labels[i]}') for i in g]
    print(f'\nnull-space directions of scaled M_k (eig < {args.null_tol:g}), largest components:')
    shown = set()
    for k, e, comps in nulls:
        key = tuple(i for i, _ in comps)
        if key in shown:
            continue
        shown.add(key)
        print(f'  first at node {k} (t={t[k]:.4f}), eig {e:.1e}: '
              + ', '.join(f'{c:+.2f}*[{i}]' for i, c in comps))


if __name__ == '__main__':
    main()
