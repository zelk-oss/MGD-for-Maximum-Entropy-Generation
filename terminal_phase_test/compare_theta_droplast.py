"""Does Theta_reg(1) change when the terminal-phase nodes are left out of the solve? (2026-11-25)

Two solves of codes/resolve_theta_reg.py on the same system (same walkers, lam, ridge): A = all nodes, B = --drop_last N
(the terminal-phase rows removed). Theta(1) = the last row of each file (the Theta used for log p and Var/2d).

Metric (as the diagonal-C validation, 2026-11-24): log p_Theta(x) = Theta . phi(x) - log Z, so on a set of signals with
covariance Sigma of phi, the spread of the log p difference is
    eps = std_x[ dTheta . phi(x) ] = sqrt(dTheta^T Sigma dTheta)   [nats],   dTheta = Theta_A - Theta_B,
and the model width is w = sqrt(Theta_A^T Sigma Theta_A) (spread of log p itself). Sigma = walker covariance at the
system's LAST node (the final samples). eps/w small = same model for local quantities (ranking, rare events, Var/2d).
Per family: the same eps restricted to the family's columns (other columns of dTheta set to 0), plus |dTheta|/|Theta_A|.
Also reported: A at the planned-grid end vs B's last node (same t: what the extra nodes change through the time term).

Usage (repo root, Jean Zay): python terminal_phase_test/compare_theta_droplast.py --system S.pt --all A.pt --drop B.pt
    [--names stats.csv (column 'name', one row per column; the moment_error_variance --dump csv)] --out DIR
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--system', type=Path, default=None)
ap.add_argument('--all', type=Path, default=None, help='Theta_reg file solved on all nodes')
ap.add_argument('--drop', type=Path, default=None, help='Theta_reg file solved with --drop_last N')
ap.add_argument('--names', type=Path, default=None)
ap.add_argument('--out', type=Path, required=True)
# cross-seed mode (2026-11-25): which variant (all nodes / without tail) is more reproducible across seeds?
ap.add_argument('--cross_systems', type=Path, nargs='+', default=None, help='one system per seed')
ap.add_argument('--cross_all', type=Path, nargs='+', default=None, help='all-nodes Theta files, same seed order')
ap.add_argument('--cross_drop', type=Path, nargs='+', default=None, help='without-tail Theta files, same seed order')
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)


def last_sigma(path):
    s = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    S = s['Sigma'][len(s['t']) - 1].double()
    return (S + S.T) / 2


if args.cross_systems is not None:
    # eps_ij = sqrt(dTh^T Sigma_i dTh), dTh = Theta_i(1) - Theta_j(1), Sigma_i = final-walker covariance of seed i;
    # w_i = sqrt(Theta_i^T Sigma_i Theta_i). Both orders (i, j) and (j, i) are printed: Sigma differs by seed.
    n = len(args.cross_systems)
    assert len(args.cross_all) == n and len(args.cross_drop) == n
    Sig = [last_sigma(p) for p in args.cross_systems]
    out = {}
    for variant, files in (('all nodes (A)', args.cross_all), ('without tail (B)', args.cross_drop)):
        Th = [torch.load(f, map_location='cpu', weights_only=False)['Theta_reg'][-1].double() for f in files]
        print(f'\n{variant}:  pair   eps [nats]   w_i [nats]   eps/w_i')
        rows = []
        for i in range(n):
            w = float(torch.sqrt(Th[i] @ Sig[i] @ Th[i]))
            for j in range(n):
                if i == j:
                    continue
                d = Th[i] - Th[j]
                e = float(torch.sqrt(torch.clamp(d @ Sig[i] @ d, min=0)))
                rows.append(dict(i=i, j=j, eps_nats=e, w_i_nats=w, eps_over_w=e / w))
                print(f'  seeds {i}-{j} (Sigma of {i}): {e:11.4g} {w:12.4g} {e / w:9.3g}')
        print(f'  median eps/w: {np.median([q["eps_over_w"] for q in rows]):.3g}')
        out[variant] = rows
    with open(args.out / 'cross_seed_A_vs_B.json', 'w') as fh:
        json.dump(dict(systems=[str(p) for p in args.cross_systems], rows=out), fh, indent=1)
    print(f'\n-> {args.out / "cross_seed_A_vs_B.json"}')
    raise SystemExit(0)
assert args.system and args.all and args.drop, 'pair mode needs --system, --all, --drop'

A = torch.load(args.all, map_location='cpu', weights_only=False)
B = torch.load(args.drop, map_location='cpu', weights_only=False)
assert A['config'] == B['config'] and A['lam'] == B['lam'], 'the two files must come from the same system and lam'
ThA, ThB = A['Theta_reg'].double(), B['Theta_reg'].double()
tA, tB = np.asarray(A['t_reg'], float), np.asarray(B['t_reg'], float)
system = torch.load(args.system, map_location='cpu', weights_only=False, mmap=True)
t_sys = np.asarray(system['t'], float)
r = ThA.shape[1]
S = system['Sigma'][len(t_sys) - 1].double()                    # walker covariance of phi at the last node
S = (S + S.T) / 2
print(f"{A['config']}\n  lam {A['lam']:g}: A {ThA.shape[0]} rows (last 1-t {1 - tA[-1]:.4e}), "
      f"B {ThB.shape[0]} rows (last 1-t {1 - tB[-1]:.4e}); system {len(t_sys)} nodes, r = {r}")

families = {'all': np.arange(r)}
if args.names is not None:
    import csv
    with open(args.names) as fh:
        names = [row['name'] for row in csv.DictReader(line for line in fh if not line.startswith('#'))]
    assert len(names) == r, f'{len(names)} names vs r = {r}'
    fam = np.array([re.sub(r'\[\d+\]$', '', n) for n in names])
    for f in dict.fromkeys(fam):
        families[f] = np.flatnonzero(fam == f)


def compare(a, b, label):
    d = a - b
    w = float(torch.sqrt(a @ S @ a))
    rows = []
    for f, cols in families.items():
        dm = torch.zeros(r, dtype=torch.float64)
        dm[cols] = d[cols]
        am = torch.zeros(r, dtype=torch.float64)
        am[cols] = a[cols]
        eps = float(torch.sqrt(torch.clamp(dm @ S @ dm, min=0)))
        wf = float(torch.sqrt(torch.clamp(am @ S @ am, min=0)))
        rel = float(d[cols].norm() / a[cols].norm().clamp_min(1e-300))
        rows.append(dict(compare=label, family=f, n=len(cols), eps_nats=eps, width_nats=wf,
                         eps_over_total_width=eps / w, theta_rel_change=rel))
    print(f'\n  {label}: model width w = {w:.4g} nats')
    print(f'  {"family":42s} {"n":>4s} {"eps [nats]":>11s} {"eps/w":>9s} {"|dTh|/|Th|":>11s}')
    for q in rows:
        print(f'  {q["family"][:42]:42s} {q["n"]:4d} {q["eps_nats"]:11.4g} {q["eps_over_total_width"]:9.3g} '
              f'{q["theta_rel_change"]:11.3g}')
    return rows


rows = compare(ThA[-1], ThB[-1], 'Theta(1): A last node vs B last node')
k = int(np.argmin(np.abs(tA - tB[-1])))                         # A at the same t as B's last node
assert abs(tA[k] - tB[-1]) < 1e-12, f'no node of A at t = {tB[-1]} (closest {tA[k]})'
rows += compare(ThA[k], ThB[-1], f'same t (planned-grid end, 1-t {1 - tB[-1]:.4e}): A row {k} vs B last')
rows += compare(ThA[-1], ThA[k], 'within A: last node vs planned-grid end (terminal-phase change of Theta)')

stem = args.out / f"droplast_compare_lam{A['lam']:g}"
with open(stem.with_suffix('.json'), 'w') as fh:
    json.dump(dict(config=A['config'], lam=A['lam'], rows=rows), fh, indent=1)
print(f'\n-> {stem}.json')
