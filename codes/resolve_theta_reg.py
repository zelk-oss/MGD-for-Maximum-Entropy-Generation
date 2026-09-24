"""Offline lam sweep for Theta_reg from a saved regularised system.

lam only enters the post-loop solve A Theta = f in SDE.forward_regularised, never
the SDE evolution, so a run launched with --save_reg_system can be re-solved for
any lam here, on CPU, without re-running the SDE.

Each system file (written by SDE._save_reg_system) holds the grid t (float64) and
per-node M, G, b, c (float32). For every lam this runs the same float64
block-Thomas solve as the in-run path (SDE._solve_regularised_thomas) and, like
forward_regularised, drops the first node: Theta_reg = Theta[1:], t_reg = t[1:].

Output, one file per (system, lam):
    <outdir>/<config>/lam<lam>_ridge<ridge>.pt
    = {'Theta_reg', 't_reg', 'lam', 'ridge', 'residual', 'config', 'meta'}

Peak RAM ~ the loaded system (n r^2 * 8 bytes, float32 M and G) plus the float64
elimination factor (n r^2 * 8 bytes): ~47 GB for n = 40000, r = 272.

Usage:
    python codes/resolve_theta_reg.py SYSTEM.pt [SYSTEM2.pt ...] \
        --lams 1e-8 3e-8 1e-7 ... --outdir turbulence/saved_results/theta_reg_lamsweep
"""

import argparse
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
for p in (project_root, project_root / 'codes', project_root / 'data'):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from codes.sde_routines import SDE  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('systems', nargs='+', type=Path,
                   help='System files written by --save_reg_system')
    p.add_argument('--lams', nargs='+', type=float, required=True,
                   help='lam values to solve for')
    p.add_argument('--ridge', type=float, default=0.0,
                   help='M_k += ridge * diag(M_k), as --reg_ridge (0 = off)')
    p.add_argument('--outdir', type=Path, required=True,
                   help='Output root; one subfolder per system/config')
    p.add_argument('--diagnose', type=int, default=5,
                   help='print the spectrum of the Jacobi-scaled M_k and G_k at this many '
                        'evenly spaced nodes before solving (0 = off)')
    p.add_argument('--overwrite', action='store_true',
                   help='Recompute (system, lam) pairs whose output already exists')
    p.add_argument('--threads', type=int,
                   default=int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or None,
                   help='torch CPU threads (default: $SLURM_CPUS_PER_TASK)')
    return p.parse_args()


def out_path(outdir, config, lam, ridge):
    return outdir / config / f'lam{lam:g}_ridge{ridge:g}.pt'


def diagnose(system, n_nodes):
    """Conditioning of the per-node blocks: at lam=0 (no time coupling) the solve is
    M_k theta = b_k with no ridge, so an exactly rank-deficient M_k is singular."""
    t = system['t'].double().numpy()
    idx = sorted(set(int(i) for i in np.linspace(0, len(t) - 1, n_nodes)))
    print(f'  diagnose (Jacobi-scaled blocks, float64), {len(idx)} nodes:')
    print(f'  {"node":>7} {"t":>9} | {"M: zero diag":>12} {"min eig":>9} {"#eig<1e-10":>10} {"#dup pairs":>10}'
          f' | {"G: min eig":>10} {"#eig<1e-10":>10}')
    for k in idx:
        row = []
        for key in ('M', 'G'):
            if key == 'G' and k >= len(system['G']):
                row.append(None); continue
            X = system[key][k].double(); X = (X + X.T) / 2
            d = torch.diagonal(X)
            s = d.clamp_min(1e-300).sqrt()
            Xs = X / (s[:, None] * s[None, :])
            ev = torch.linalg.eigvalsh(Xs)
            off = Xs - torch.diag(torch.diagonal(Xs))
            dup = int(((off.abs() > 1 - 1e-9).sum() // 2).item())
            row.append((int((d == 0).sum()), float(ev.min()), int((ev < 1e-10).sum()), dup))
        m, g = row
        gtxt = f'{g[1]:10.2e} {g[2]:10d}' if g else f'{"-":>10} {"-":>10}'
        print(f'  {k:7d} {t[k]:9.6f} | {m[0]:12d} {m[1]:9.2e} {m[2]:10d} {m[3]:10d} | {gtxt}')


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    for system_path in args.systems:
        config = system_path.name[:-len('.pt')] if system_path.name.endswith('.pt') else system_path.name
        todo = [lam for lam in args.lams
                if args.overwrite or not out_path(args.outdir, config, lam, args.ridge).exists()]
        if not todo:
            print(f'[{config}] all {len(args.lams)} lam values already solved, skipping')
            continue

        t0 = time.time()
        system = torch.load(system_path, map_location='cpu', weights_only=False)
        t = system['t'].double().numpy()
        print(f'[{config}] loaded {len(t)} nodes, r={system["num_potentials"]} '
              f'in {time.time() - t0:.0f} s; solving {len(todo)} lam values')

        if args.diagnose:
            diagnose(system, args.diagnose)

        # the solver only needs num_potentials from `self`
        solver_self = types.SimpleNamespace(num_potentials=system['num_potentials'])

        failed = []
        for lam in todo:
            t1 = time.time()
            try:
                Theta = SDE._solve_regularised_thomas(
                    solver_self, t, system['M'], system['G'], system['b'], system['c'],
                    lam, ridge=args.ridge)
            except torch.linalg.LinAlgError as e:              # log, keep going with the next lam
                print(f'[{config}] lam={lam:g}: FAILED after {time.time() - t1:.0f} s: {e}')
                failed.append(lam)
                continue
            dest = out_path(args.outdir, config, lam, args.ridge)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + '.tmp')
            torch.save({
                'Theta_reg': Theta[1:].clone(),
                't_reg': torch.as_tensor(t[1:]),
                'lam': lam,
                'ridge': args.ridge,
                'residual': solver_self.last_reg_residual,
                'config': config,
                'meta': system.get('meta', {}),
            }, tmp)
            tmp.replace(dest)
            print(f'[{config}] lam={lam:g}: residual {solver_self.last_reg_residual:.2e}, '
                  f'{time.time() - t1:.0f} s -> {dest}')

        if failed:
            print(f'[{config}] {len(failed)} lam value(s) failed: {failed}')
        del system


if __name__ == '__main__':
    main()
