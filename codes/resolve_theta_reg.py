"""Offline lam sweep for Theta_reg from a saved regularised system.

lam only enters the post-loop solve A Theta = f in SDE.forward_regularised, never
the SDE evolution, so a run launched with --save_reg_system can be re-solved for
any lam here, on CPU, without re-running the SDE.

Each system file (written by SDE._save_reg_system) holds the grid t (float64) and
per-node M, G, b, c (float32). For every lam this runs the same float64
block-Thomas solve as the in-run path (SDE._solve_regularised_thomas) and, like
forward_regularised, drops the first node: Theta_reg = Theta[1:], t_reg = t[1:].

Output, one file per (system, lam):
    <outdir>/<config>/lam<lam>_ridge<ridge>[_<mode>].pt   (no suffix for mode legacy)
    = {'Theta_reg', 't_reg', 'lam', 'ridge', 'mode', 'm_source', 'residual', 'config', 'meta'}

--mode picks the energy (see SDE._solve_regularised_thomas): 'legacy' (the old system,
wrong-sign time term), 'fixed' (correct, centred time term, same lam scale), 'guth'
(correct time term with Guth et al.'s weights; lam is a multiplier, 1 = theirs).
'fixed'/'guth' need m = E[phi] per node: taken from the system if it holds one (runs
from 2026-09-29 on), else from the run's aux_moments (barphi_p, the walker moments
after the corrector, a close stand-in for the moments on y_k the system was built on),
found under --results_root/saved_results/{aux_moments,sampling_times}/. 'guth' also
needs the signal dimension: meta['dim'] if saved, else --dim.

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
    p.add_argument('--mode', choices=['legacy', 'fixed', 'guth'], default='legacy',
                   help='energy to solve (default legacy = the old system)')
    p.add_argument('--results_root', type=Path, default=None,
                   help='folder holding saved_results/ of the runs (e.g. turbulence/); '
                        'used for m when the system file has none')
    p.add_argument('--dim', type=int, default=None,
                   help="signal dimension d for --mode guth, if not in the system's meta")
    p.add_argument('--overwrite', action='store_true',
                   help='Recompute (system, lam) pairs whose output already exists')
    p.add_argument('--threads', type=int,
                   default=int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or None,
                   help='torch CPU threads (default: $SLURM_CPUS_PER_TASK)')
    return p.parse_args()


def out_path(outdir, config, lam, ridge, mode='legacy'):
    suffix = '' if mode == 'legacy' else f'_{mode}'
    return outdir / config / f'lam{lam:g}_ridge{ridge:g}{suffix}.pt'


def node_moments(system, config, results_root):
    """m = E[phi] per system node, and its source."""
    if 'm' in system:
        return system['m'].double(), 'system'
    if results_root is None:
        raise ValueError(f'[{config}] system has no m; pass --results_root to read aux_moments')
    base = Path(results_root) / 'saved_results'
    aux = torch.load(base / 'aux_moments' / f'{config}_aux_moments.pt', map_location='cpu')
    t_fine = torch.load(base / 'sampling_times' / f'{config}.pt', map_location='cpu')
    t_fine = np.asarray(t_fine, dtype=np.float64)
    barphi_p = aux['barphi_p'].double()              # one row per step, at t_fine[1:]
    if len(barphi_p) != len(t_fine) - 1:
        raise ValueError(f'[{config}] barphi_p has {len(barphi_p)} rows, t has {len(t_fine)}')
    t_sys = system['t'].double().numpy()
    idx = np.clip(np.searchsorted(t_fine[1:], t_sys), 0, len(t_fine) - 2)
    err = np.abs(t_fine[1:][idx] - t_sys).max()
    if err > 1e-9:
        raise ValueError(f'[{config}] system nodes are not on the saved time grid (max |dt| {err:.1e})')
    return barphi_p[idx], 'aux_moments.barphi_p'


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


def c_vs_m(system, m, tau_mean, n_nodes):
    """How much of c = E[phi tau] is the part m * mean(tau) that centring removes.
    For the exact interpolant E[tau] = 0; |cos(c, m)| ~ 1 with implied tau_mean >> 0
    means c is dominated by that part (the old, uncentred system's time term)."""
    t = system['t'].double().numpy()
    idx = sorted(set(int(i) for i in np.linspace(0, len(t) - 2, n_nodes)))
    print(f'  c vs m at {len(idx)} nodes: {"t":>9} {"cos(c,m)":>9} {"implied tau_mean":>17}'
          f' {"saved tau_mean":>15} {"|c - m tau|/|c|":>16}')
    for k in idx:
        c, mk = system['c'][k].double(), m[k].double()
        cos = float(c @ mk / (c.norm() * mk.norm() + 1e-300))
        implied = float(c @ mk / (mk @ mk + 1e-300))
        tm = float(tau_mean[k]) if tau_mean is not None else float('nan')
        tau_used = tm if tau_mean is not None else 0.0
        rel = float((c - tau_used * mk).norm() / (c.norm() + 1e-300))
        print(f'  {"":>21} {t[k]:9.6f} {cos:9.4f} {implied:17.3e} {tm:15.3e} {rel:16.3e}')


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    for system_path in args.systems:
        config = system_path.name[:-len('.pt')] if system_path.name.endswith('.pt') else system_path.name
        todo = [lam for lam in args.lams
                if args.overwrite or not out_path(args.outdir, config, lam, args.ridge, args.mode).exists()]
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

        m = tau_mean = None
        m_source = None
        dim = system.get('meta', {}).get('dim', args.dim)
        if args.mode != 'legacy':
            m, m_source = node_moments(system, config, args.results_root)
            tau_mean = system.get('tau_mean')                # None -> its expectation, 0
            print(f'[{config}] mode={args.mode}: m from {m_source}, '
                  f'tau_mean {"saved" if tau_mean is not None else "not saved (0 used)"}'
                  + (f', dim={dim}' if args.mode == 'guth' else ''))
            if args.mode == 'guth' and dim is None:
                raise ValueError(f'[{config}] --mode guth needs --dim (not in the system meta)')
            if args.diagnose:
                c_vs_m(system, m, tau_mean, args.diagnose)

        failed = []
        for lam in todo:
            t1 = time.time()
            try:
                Theta = SDE._solve_regularised_thomas(
                    solver_self, t, system['M'], system['G'], system['b'], system['c'],
                    lam, ridge=args.ridge, mode=args.mode, m=m, tau_mean=tau_mean, dim=dim)
            except torch.linalg.LinAlgError as e:              # log, keep going with the next lam
                print(f'[{config}] lam={lam:g}: FAILED after {time.time() - t1:.0f} s: {e}')
                failed.append(lam)
                continue
            dest = out_path(args.outdir, config, lam, args.ridge, args.mode)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + '.tmp')
            torch.save({
                'Theta_reg': Theta[1:].clone(),
                't_reg': torch.as_tensor(t[1:]),
                'lam': lam,
                'ridge': args.ridge,
                'mode': args.mode,
                'm_source': m_source,
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
