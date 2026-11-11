"""Offline re-solve of Theta_reg from a saved regularised system, for any lam and energy.

lam only enters the post-loop solve A Theta = f in SDE.forward_regularised, never
the SDE evolution, so a run launched with --save_reg_system can be re-solved for
any lam here, on CPU, without re-running the SDE. Same float64 block-Thomas solve as
the in-run path (SDE._solve_regularised_thomas); like forward_regularised, the first
node is dropped: Theta_reg = Theta[1:], t_reg = t[1:]. CODE sign throughout
(theta_code = -theta_MGD).

--mode (default: the file's meta['reg_mode'], 'legacy' for files without one):
  legacy -- the pre-2026-09-30 tau-based energy (reproduction only).
  moment -- time term Sigma_w Theta_dot = mdot (see SDE._solve_regularised_thomas).
            On a moment system: Sigma, mdot, live, h from the file.
            On an OLD (legacy) system, rebuilt from the run's aux_moments (needs
            --results_root, param_storage_frequency 1):
              Sigma_k = G_k - m_k m_k^T, m = barphi_p (walkers after the corrector).
                G is float32 and at y_k, so this centring loses precision for statistics
                whose mean is large relative to their spread.
              mdot = 3-point non-uniform finite difference of barphi_e (interpolant moments
                on FIXED pairs, a smooth function of t; error O(h^2)). barphi_e[j] is at
                t_fine[j+1].
--schedule uniform | guth: moment-mode weights (guth needs the dimension: meta or --dim).
--mask auto | none | system | diag | theta: pin dead potentials to 0 per node (auto: the
  saved live mask if the file has one, else none; diag: diag(M_k) > --live_floor; theta:
  the per-step mask of an OLD run rebuilt from its saved theta_t, which the corrector set
  to exactly 0 for every masked potential -- needs --results_root).
--select: codes/lam_select_cv.py -- one file lam_path_<tag>.pt with Theta for every lam,
  the held-out moment-matching loss, the amplitude veto, the noise-model chi2 and the
  selected lam (instead of one file per lam).
--neg_cols START:STOP (2026-10-08, theta_sign_constraint/): force Theta_{k,i} <= -eps_{k,i} for
  the columns START..STOP-1 (e.g. 0:9 = L_6 when it is the first term), at every node with
  t >= --neg_t_min, by the active-set loop of codes/theta_reg_bounds.py (a few Thomas solves).
  eps_{k,i} = --neg_eps_nats / sd of phi_i on the walkers at node k (moment systems only).
  Output name gets _neg<eps>[_from<t_min>][_exact]; the file also holds 'neg' (bounds info).
  --neg_method pnewton (default: projected Newton, feasible and monotone, see
  codes/theta_reg_bounds.py) | exact (plain active set; oscillated on turbulence, job 781019) |
  add_only. Non-default methods add _<method> to the name. A run that does not converge is written
  as <name>_NOTCONVERGED.pt, never under the normal name.

Output, one file per (system, lam):
    <outdir>/<config>/lam<lam>_ridge<ridge>[_moment[_guthsched]][_mask].pt
    = {'Theta_reg', 't_reg', 'lam', 'ridge', 'mode', 'schedule', 'mask', 'inputs',
       'residual', 'config', 'meta'}

Peak RAM ~ the loaded system (2 n r^2 * 4 bytes, float32 blocks) plus the float64
elimination factor (n r^2 * 8 bytes): ~47 GB for n = 40000, r = 272.

Usage:
    python codes/resolve_theta_reg.py SYSTEM.pt [SYSTEM2.pt ...] \
        --lams 1e-8 3e-8 1e-7 ... --outdir turbulence/saved_results/theta_reg_lamsweep
"""

import argparse
import contextlib
import io
import os
import sys
import threading
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
                   help='M_k += ridge * diag(M_k), as --reg_ridge (0 = off; use the run\'s '
                        '--regularization to reproduce theta_t at lam = 0)')
    p.add_argument('--outdir', type=Path, required=True,
                   help='Output root; one subfolder per system/config')
    p.add_argument('--diagnose', type=int, default=5,
                   help='print the spectrum of the Jacobi-scaled M_k and C_k at this many '
                        'evenly spaced nodes before solving (0 = off)')
    p.add_argument('--mode', choices=['legacy', 'moment'], default=None,
                   help="energy to solve (default: the system's meta['reg_mode'], else legacy)")
    p.add_argument('--schedule', choices=['uniform', 'guth'], default='uniform',
                   help='moment-mode weights')
    p.add_argument('--results_root', type=Path, default=None,
                   help='folder holding saved_results/ of the runs (e.g. turbulence/); '
                        'needed for --mode moment on systems saved in legacy mode')
    p.add_argument('--dim', type=int, default=None,
                   help="signal dimension d for --schedule guth, if not in the system's meta")
    p.add_argument('--mask', choices=['auto', 'none', 'system', 'diag', 'theta'], default='auto',
                   help='pin dead potentials: saved live mask / none / diag(M_k) > --live_floor / '
                        'theta_t != 0 of the run (needs --results_root)')
    p.add_argument('--live_floor', type=float, default=0.0,
                   help='with --mask diag: a potential is dead at node k if M_k[i,i] <= floor')
    p.add_argument('--select', action='store_true',
                   help='lam selection (codes/lam_select_cv.py): write one lam_path file')
    p.add_argument('--veto', type=float, default=10.0,
                   help='--select: amplitude veto factor')
    p.add_argument('--n_walkers', type=int, default=None,
                   help='--select chi2: walkers per run, if not in the meta')
    p.add_argument('--sigma', type=float, default=None,
                   help='--select chi2: SDE sigma, if not in the meta')
    p.add_argument('--neg_cols', type=str, default=None, metavar='START:STOP',
                   help='force Theta <= -eps on these columns (e.g. 0:9 = L_6); see the docstring')
    p.add_argument('--neg_eps_nats', type=float, default=1e-2,
                   help='with --neg_cols: eps in nats per standard deviation of phi_i on the walkers')
    p.add_argument('--neg_t_min', type=float, default=0.0,
                   help='with --neg_cols: bound only the nodes with t >= this (0 = every node)')
    p.add_argument('--neg_max_iter', type=int, default=30,
                   help='with --neg_cols: cap on active-set iterations (one Thomas solve each)')
    p.add_argument('--neg_method', choices=['pnewton', 'exact', 'add_only'], default='pnewton',
                   help='with --neg_cols: bound loop (codes/theta_reg_bounds.py)')
    p.add_argument('--c_floor_eps', nargs='+', type=float, default=[0.0],
                   help='method A (moment mode): floor eps*M_cell on C and mdot -> C (C + eps M)^-1 mdot per '
                        'cell (SDE._solve_regularised_thomas). One solve per (lam, eps). eps in the units of the '
                        'generalised eigenvalues mu of C v = mu M v (codes/c_spectrum_gen.py). 0 = unchanged, '
                        'same file name as before; eps > 0 adds _cfl<eps> to the name.')
    p.add_argument('--node_stride', type=int, default=1,
                   help='keep every m-th node (plus the last) before the solve: cells then span m steps, so the '
                        'directions of C filled by one step of SDE noise (mu ~ 2 h sigma^2) move to ~2 m h sigma^2. '
                        'Grid-dependence test (2026-11-11). 1 = all nodes, unchanged; m > 1 adds _stride<m> to the name.')
    p.add_argument('--mmap', action='store_true',
                   help='torch.load(..., mmap=True): the system stays file-backed instead of copied '
                        'into RAM (same numbers)')
    p.add_argument('--mem_every', type=float, default=60.0,
                   help='print process RSS to stderr every this many seconds (0 = off)')
    p.add_argument('--overwrite', action='store_true',
                   help='Recompute (system, lam) pairs whose output already exists')
    p.add_argument('--threads', type=int,
                   default=int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or None,
                   help='torch CPU threads (default: $SLURM_CPUS_PER_TASK)')
    return p.parse_args()


def rss_gb():
    """(current, peak) resident memory of this process in GB (10^9 B), from /proc/self/status."""
    v = {}
    with open('/proc/self/status') as fh:
        for line in fh:
            if line.startswith(('VmRSS:', 'VmHWM:')):
                v[line.split(':')[0]] = int(line.split()[1]) * 1024 / 1e9
    return v.get('VmRSS', float('nan')), v.get('VmHWM', float('nan'))


def log_mem(label):
    """RSS line on stderr (the solver's stdout is captured during the solve)."""
    cur, peak = rss_gb()
    print(f'[mem] {time.strftime("%H:%M:%S")} {label}: RSS {cur:.1f} GB, peak {peak:.1f} GB',
          file=sys.stderr, flush=True)


def start_mem_monitor(every):
    """Background thread: log_mem every `every` seconds until the process exits."""
    def loop():
        while True:
            time.sleep(every)
            log_mem('monitor')
    threading.Thread(target=loop, daemon=True).start()


def tag(mode, schedule, mask):
    s = '' if mode == 'legacy' else ('_moment' if schedule == 'uniform' else f'_moment_{schedule}sched')
    return s + ('' if mask == 'none' else '_mask')


def out_path(outdir, config, lam, ridge, mode='legacy', schedule='uniform', mask='none', extra=''):
    return outdir / config / f'lam{lam:g}_ridge{ridge:g}{tag(mode, schedule, mask)}{extra}.pt'


def neg_tag(args):
    if args.neg_cols is None:
        return ''
    return (f'_neg{args.neg_eps_nats:g}' + (f'_from{args.neg_t_min:g}' if args.neg_t_min > 0 else '')
            + ('' if args.neg_method == 'pnewton' else f'_{args.neg_method}'))


def fd_derivative(t, y):
    """d y / d t on a non-uniform grid: 3-point, second order (one-sided at the ends).
    t (N,) float64 increasing, y (N, r). Returns (N, r)."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(t) < 3:
        raise ValueError('need at least 3 points')
    d = np.empty_like(y)
    h1, h2 = (t[1:-1] - t[:-2])[:, None], (t[2:] - t[1:-1])[:, None]
    d[1:-1] = (-h2 / (h1 * (h1 + h2))) * y[:-2] + ((h2 - h1) / (h1 * h2)) * y[1:-1] \
        + (h1 / (h2 * (h1 + h2))) * y[2:]
    h1, h2 = t[1] - t[0], t[2] - t[1]
    d[0] = (-(2 * h1 + h2) / (h1 * (h1 + h2))) * y[0] + ((h1 + h2) / (h1 * h2)) * y[1] \
        - (h1 / (h2 * (h1 + h2))) * y[2]
    h1, h2 = t[-2] - t[-3], t[-1] - t[-2]
    d[-1] = (h2 / (h1 * (h1 + h2))) * y[-3] - ((h1 + h2) / (h1 * h2)) * y[-2] \
        + ((h1 + 2 * h2) / (h2 * (h1 + h2))) * y[-1]
    return d


def aux_on_nodes(system, config, results_root):
    """For a legacy system: m (barphi_p), mdot (FD of barphi_e) and the SDE step h at the
    system nodes, from the run's aux_moments and sampling_times."""
    if results_root is None:
        raise ValueError(f'[{config}] --mode moment on a legacy system needs --results_root')
    base = Path(results_root) / 'saved_results'
    aux = torch.load(base / 'aux_moments' / f'{config}_aux_moments.pt', map_location='cpu')
    t_fine = np.asarray(torch.load(base / 'sampling_times' / f'{config}.pt', map_location='cpu'),
                        dtype=np.float64)
    bp, be = aux['barphi_p'].double().numpy(), aux['barphi_e'].double().numpy()
    if len(bp) != len(t_fine) - 1 or len(be) != len(t_fine) - 1:
        raise ValueError(f'[{config}] aux rows ({len(bp)}, {len(be)}) != len(t) - 1 = {len(t_fine) - 1}: '
                         f'param_storage_frequency != 1 is not supported')
    t_e = t_fine[1:]                                   # row j of barphi_* is at t_fine[j+1]
    mdot = fd_derivative(t_e, be)
    t_sys = system['t'].double().numpy()
    idx = np.clip(np.searchsorted(t_e, t_sys), 0, len(t_e) - 1)
    err = np.abs(t_e[idx] - t_sys).max()
    if err > 1e-9:
        raise ValueError(f'[{config}] system nodes are not on the saved time grid (max |dt| {err:.1e})')
    h = t_fine[idx + 1] - t_fine[idx]                  # SDE step that ended at the node
    return torch.as_tensor(bp[idx]), torch.as_tensor(mdot[idx]), h


def theta_t_on_nodes(system, config, results_root):
    """The run's per-step theta_t (code sign) at the system nodes (row j at t_fine[j+1])."""
    if results_root is None:
        raise ValueError(f'[{config}] --mask theta needs --results_root')
    base = Path(results_root) / 'saved_results'
    theta_t = torch.load(base / 'lagrange_multipliers' / f'{config}.pt', map_location='cpu').double()
    t_fine = np.asarray(torch.load(base / 'sampling_times' / f'{config}.pt', map_location='cpu'),
                        dtype=np.float64)
    if len(theta_t) != len(t_fine) - 1:
        raise ValueError(f'[{config}] theta_t has {len(theta_t)} rows, t has {len(t_fine)}')
    t_sys = system['t'].double().numpy()
    idx = np.clip(np.searchsorted(t_fine[1:], t_sys), 0, len(t_fine) - 2)
    if np.abs(t_fine[1:][idx] - t_sys).max() > 1e-9:
        raise ValueError(f'[{config}] system nodes are not on the saved time grid')
    return theta_t[idx]


def build_inputs(system, config, args, mode):
    """Per-node lists the solver takes, plus live, h and provenance, for one mode."""
    r = system['num_potentials']
    meta = system.get('meta', {})
    saved = meta.get('reg_mode', 'legacy')
    inp = {'t': system['t'].double().numpy(), 'r': r, 'mode': mode, 'M': system['M'],
           'b': system['b'], 'schedule': args.schedule, 'ridge': args.ridge,
           'dim': meta.get('dim', args.dim), 'n_walkers': meta.get('n_walkers', args.n_walkers),
           'sigma': meta.get('sigma', args.sigma), 'h': None, 'live': None}
    if mode == 'legacy':
        if saved != 'legacy':
            raise ValueError(f'[{config}] --mode legacy needs a legacy system (G, c); this one is {saved}')
        inp.update(C=system['G'], v=system['c'], source='system (legacy G, c)')
    elif saved == 'moment':
        inp.update(C=system['Sigma'], v=list(system['mdot']), h=system['h'].numpy(),
                   source='system (Sigma_w at x_k+1, pathwise mdot)')
    else:
        m, mdot, h = aux_on_nodes(system, config, args.results_root)
        C = [system['G'][k].double() - torch.outer(m[k], m[k]) for k in range(len(m))]
        inp.update(C=C, v=list(mdot), h=h,
                   source='rebuilt: Sigma = G(y_k) - m m^T (m = barphi_p), mdot = FD(barphi_e)')
    if args.schedule == 'guth' and mode == 'moment' and inp['dim'] is None:
        raise ValueError(f'[{config}] --schedule guth needs --dim (not in the system meta)')

    mask = args.mask
    if mask == 'auto':
        mask = 'system' if 'live' in system else 'none'
    if mask == 'system':
        if 'live' not in system:
            raise ValueError(f'[{config}] --mask system: the file has no live mask')
        inp['live'] = system['live'].bool()
    elif mask == 'diag':
        inp['live'] = torch.stack([torch.diagonal(Mk).double() > args.live_floor for Mk in system['M']])
    elif mask == 'theta':
        inp['live'] = theta_t_on_nodes(system, config, args.results_root) != 0
    inp['mask'] = mask
    return inp


def stride_inputs(inp, m):
    """Keep nodes 0, m, 2m, ... and the last one. Per-node blocks stay lazy views (mmap-friendly lists)."""
    n = len(inp['t'])
    idx = list(range(0, n, m))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    out = dict(inp)
    out['t'] = inp['t'][idx]
    for key in ('M', 'C', 'b', 'v'):
        out[key] = [inp[key][k] for k in idx]
    if inp.get('h') is not None:
        out['h'] = np.asarray(inp['h'])[idx]
    if inp.get('live') is not None:
        out['live'] = inp['live'][idx]
    return out


def diagnose(inp, n_nodes):
    """Conditioning of the per-node blocks (Jacobi-scaled, float64): at lam=0 the solve is
    M_k theta = b_k (+ ridge), so an exactly rank-deficient M_k is singular."""
    t = inp['t']
    idx = sorted(set(int(i) for i in np.linspace(0, len(t) - 1, n_nodes)))
    print(f'  diagnose (Jacobi-scaled blocks, float64), {len(idx)} nodes:')
    print(f'  {"node":>7} {"t":>9} | {"M: zero diag":>12} {"min eig":>9} {"#eig<1e-10":>10} {"#dup pairs":>10}'
          f' | {"C: diag<=0":>10} {"min eig":>9} {"#eig<1e-10":>10}')
    for k in idx:
        row = []
        for X in (inp['M'][k], inp['C'][k]):
            X = X.double(); X = (X + X.T) / 2
            d = torch.diagonal(X)
            pos = d > 0                       # scale only entries with a positive diagonal: a
            Xs = X[pos][:, pos]               # C rebuilt as G - m m^T can have d <= 0, and
            s = d[pos].sqrt()                 # 1/sqrt(1e-300) scaling made eigvalsh diverge
            Xs = Xs / (s[:, None] * s[None, :])
            try:
                ev = torch.linalg.eigvalsh(Xs) if Xs.numel() else torch.zeros(1, dtype=X.dtype)
                ev_min, n_small = float(ev.min()), int((ev < 1e-10).sum())
            except torch.linalg.LinAlgError:
                ev_min, n_small = float('nan'), -1
            off = Xs - torch.diag(torch.diagonal(Xs))
            dup = int(((off.abs() > 1 - 1e-9).sum() // 2).item())
            row.append((int((~pos).sum()), ev_min, n_small, dup))
        m, g = row
        print(f'  {k:7d} {t[k]:9.6f} | {m[0]:12d} {m[1]:9.2e} {m[2]:10d} {m[3]:10d}'
              f' | {g[0]:10d} {g[1]:9.2e} {g[2]:10d}')


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    log_mem('start')
    if args.mem_every > 0:
        start_mem_monitor(args.mem_every)

    for system_path in args.systems:
        config = system_path.name[:-len('.pt')] if system_path.name.endswith('.pt') else system_path.name
        t0 = time.time()
        system = torch.load(system_path, map_location='cpu', weights_only=False, mmap=args.mmap)
        log_mem(f'after load ({system_path.stat().st_size / 1e9:.1f} GB file, mmap {args.mmap})')
        mode = args.mode or system.get('meta', {}).get('reg_mode', 'legacy')
        inp = build_inputs(system, config, args, mode)
        if args.node_stride > 1:
            inp = stride_inputs(inp, args.node_stride)
            print(f'[{config}] node_stride {args.node_stride}: {len(inp["t"])} nodes kept')
        log_mem('after build_inputs')
        print(f'[{config}] loaded {len(inp["t"])} nodes, r={inp["r"]} in {time.time() - t0:.0f} s; '
              f'mode {mode}' + (f' ({args.schedule})' if mode == 'moment' else '') +
              f', inputs: {inp["source"]}, mask: {inp["mask"]}')
        if args.diagnose:
            diagnose(inp, args.diagnose)
        meta = system.get('meta', {})
        common = {'ridge': args.ridge, 'mode': mode, 'schedule': args.schedule, 'mask': inp['mask'],
                  'inputs': inp['source'], 'config': config, 'meta': meta}

        neg = None
        if args.neg_cols is not None:
            if args.select or mode != 'moment' or 'Sigma' not in system:
                raise ValueError(f'[{config}] --neg_cols needs a moment-mode system and no --select')
            from codes.theta_reg_bounds import eps_from_sigma
            a_, b_ = (int(x) for x in args.neg_cols.split(':'))
            cols = list(range(a_, b_))
            upper = -eps_from_sigma(system['Sigma'], cols, args.neg_eps_nats)
            upper[torch.as_tensor(inp['t']) < args.neg_t_min] = float('inf')
            neg = dict(cols=cols, eps_nats=args.neg_eps_nats, t_min=args.neg_t_min)
            print(f'[{config}] bounds Theta <= -eps on columns {a_}:{b_} at {int((inp["t"] >= args.neg_t_min).sum())} '
                  f'nodes, eps_nats {args.neg_eps_nats:g} (eps at the last node: '
                  f'{[f"{-float(u):.3g}" for u in upper[-1]]})')
        extra0 = neg_tag(args)
        if any(e for e in args.c_floor_eps) and (neg is not None or args.select or mode != 'moment'):
            raise ValueError(f'[{config}] --c_floor_eps > 0 needs moment mode, no --neg_cols, no --select')
        if args.node_stride > 1 and (neg is not None or args.select):
            raise ValueError(f'[{config}] --node_stride > 1 is not supported with --neg_cols or --select')

        if args.select:
            extra = extra0
            from codes.lam_select_cv import lam_path
            dest = args.outdir / config / f'lam_path_ridge{args.ridge:g}{tag(mode, args.schedule, inp["mask"])}.pt'
            if dest.exists() and not args.overwrite:
                print(f'[{config}] {dest} exists, skipping'); continue
            res = lam_path(inp, args.lams, veto=args.veto)
            res['Theta'] = res['Theta'][:, 1:].clone()          # drop the first node, as below
            res['t'] = res['t'][1:].clone()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + '.tmp')
            torch.save({**res, **common}, tmp)
            tmp.replace(dest)
            print(f'[{config}] lam path ({len(res["lams"])} values) -> {dest}')
            continue

        stride_tag = f'_stride{args.node_stride}' if args.node_stride > 1 else ''
        eps_extra = lambda e: extra0 + (f'_cfl{e:g}' if e else '') + stride_tag
        todo = [(lam, e) for e in args.c_floor_eps for lam in args.lams
                if args.overwrite or not out_path(args.outdir, config, lam, args.ridge, mode,
                                                  args.schedule, inp['mask'], eps_extra(e)).exists()]
        if not todo:
            print(f'[{config}] all {len(args.lams)} x {len(args.c_floor_eps)} (lam, eps) values already solved, skipping')
            continue
        solver_self = types.SimpleNamespace(num_potentials=inp['r'])
        failed = []
        for lam, c_eps in todo:
            extra = eps_extra(c_eps)
            solver_self.last_c_floor_shrink = None
            t1 = time.time()
            try:
                if neg is not None:
                    from codes.theta_reg_bounds import solve_upper_bounded
                    log_buf = io.StringIO()
                    with contextlib.redirect_stdout(log_buf):
                        Theta, info = solve_upper_bounded(
                            solver_self, inp['t'], inp['M'], inp['C'], inp['b'], inp['v'], lam,
                            neg['cols'], upper, ridge=args.ridge, mode=mode, dim=inp['dim'],
                            schedule=args.schedule, live=inp['live'], max_iter=args.neg_max_iter,
                            method=args.neg_method, log=lambda m: print(m, file=sys.__stdout__, flush=True))
                    neg_info = {**neg, **info, 'upper_last': upper[-1].clone()}
                    print(f'[{config}] lam={lam:g}: bounds {"converged" if info["converged"] else "NOT CONVERGED"} '
                          f'after {info["iterations"]} solves, {info["n_pinned"]} entries pinned '
                          f'({info["n_wants_out"]} of them would go below the bound if freed), '
                          f'max Theta on the bounded columns {info["max_theta_bounded"]:.3g}, '
                          f'max excess over the bound {info["max_excess"]:.3g}')
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        Theta = SDE._solve_regularised_thomas(
                            solver_self, inp['t'], inp['M'], inp['C'], inp['b'], inp['v'], lam,
                            ridge=args.ridge, mode=mode, dim=inp['dim'], schedule=args.schedule,
                            live=inp['live'], c_floor_eps=c_eps)
                    neg_info = None
            except torch.linalg.LinAlgError as e:              # log, keep going with the next lam
                print(f'[{config}] lam={lam:g} eps={c_eps:g}: FAILED after {time.time() - t1:.0f} s: {e}')
                failed.append((lam, c_eps))
                continue
            dest = out_path(args.outdir, config, lam, args.ridge, mode, args.schedule, inp['mask'], extra)
            if neg_info is not None and not neg_info['converged']:
                # Timestamp the NOTCONVERGED name so a relaunch never clobbers the previous attempt
                # (lets us keep both for comparison without moving them to a stash dir).
                dest = dest.with_name(dest.stem + '_NOTCONVERGED_' + time.strftime('%Y%m%d_%H%M%S') + '.pt')
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + '.tmp')
            torch.save({'Theta_reg': Theta[1:].clone(), 't_reg': torch.as_tensor(inp['t'][1:]),
                        'lam': lam, 'residual': solver_self.last_reg_residual, 'neg': neg_info,
                        'c_floor_eps': c_eps, 'c_floor_shrink': solver_self.last_c_floor_shrink,
                        **common}, tmp)
            tmp.replace(dest)
            log_mem(f'after lam={lam:g}')
            shr = solver_self.last_c_floor_shrink
            print(f'[{config}] lam={lam:g} eps={c_eps:g}' + (f' (median |mdot_eff|^2/|mdot|^2 {shr:.3g})' if shr is not None else '')
                  + f': residual {solver_self.last_reg_residual:.2e}, '
                  f'finite {bool(torch.isfinite(Theta).all())}, {time.time() - t1:.0f} s -> {dest}')

        if failed:
            print(f'[{config}] {len(failed)} lam value(s) failed: {failed}')
        del system


if __name__ == '__main__':
    main()
