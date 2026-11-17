"""Pinned-L_6 test (2026-11-17): does fixing the L_6 coefficients of Theta_reg at -eps change the interpretation?

Context: the SDE transport is left as it is (L_6 free, moments matched, theta_t -> H_bound unchanged). Only the
offline Theta_reg is changed, so that p_Theta ~ exp(Theta . phi) is normalizable without the slow projected-Newton
bound loop. Checked before (11-17, B+Scat4 and jets_full): no statistic grows as fast as |x|^6 (scalar outer GGD
exponents <= 1.63, Scat4 Mod2 degree 4) and the L_6 bank sees every Fourier mode, so any Theta_L6 <= -eps < 0 at the
last node gives a normalizable density. Open question = what it costs:

  arms (one seed, same lam/ridge/mask as the free fit):
    free        unconstrained Theta_reg (reference; re-solved here so that every energy comes from the same pass)
    pinall      every L_6 entry (all nodes, all filters) fixed at -eps            (one Thomas solve)
    pinall10    the same at -10 eps (does anything depend on the arbitrary eps?)   (one Thomas solve)
    pinpos      only the entries above -eps fixed at -eps, re-solve, repeat until none is above (add_only loop,
                started from the free solve; feasible after every round)
    pnewton     the existing L_6<0 projected-Newton file, if --pnewton is given (read only, not re-solved)

  per arm, at the last node (the Theta used for log p), against free and against pnewton:
    energy E of the Theta_reg problem (E - E_free = cost of the constraint, same pass for all arms)
    -log p ranking on the data and on the generated signals: Spearman, overlap of the 5 % rarest
    per-family change of theta (relative norm) and correlation of each family's log p contribution
    radial scan: Theta . phi(a x) - Theta . phi(x) for a in A_GRID on data signals, generated signals, the 20
      rarest data signals and one wavelet atom per channel of the Q bank (energy matched to the data): a rise above
      the a = 1 value at a > 1 is a far bump of exp(Theta . phi) (the -eps wall only acts at large amplitude)

Theta in the code sign: p ~ exp(+Theta . phi), log p = Theta . phi - log Z (log Z is common to all signals of an arm,
so it does not enter rankings or the radial rise).

Usage (repo root on Jean Zay, CPU node; see launch_pin_l6_test.sh):
    python -u theta_sign_constraint/pin_l6_test.py --system <reg_system .pt> --exp_dir <experiment run dir> \
        --samples <saved_results/samples/<config>.pt> --free_ref <theta_reg_lamsweep file> --pnewton <l6neg file> \
        --lam 3e-5 --ridge 1e-4 --eps 1e-2 --cols 0:9 --theta_out <dir on SCRATCH> --out <small results dir> --mmap
"""
import argparse
import contextlib
import io
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'codes', ROOT / 'turbulence'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from codes.sde_routines import SDE                                       # noqa: E402
from codes.theta_reg_bounds import eps_from_sigma, _bounds              # noqa: E402
from codes.resolve_theta_reg import build_inputs, log_mem               # noqa: E402

A_GRID = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50], dtype=np.float64)
RARE_FRAC = 0.05


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--system', type=Path, required=True, help='moment-mode regularised system (.pt)')
    p.add_argument('--exp_dir', type=Path, required=True, help='experiment run dir (config.json, fitted_potentials/)')
    p.add_argument('--samples', type=Path, required=True, help='generated signals of the run (saved_results/samples)')
    p.add_argument('--free_dir', type=Path, default=None,
                   help='root of the saved unconstrained Theta_reg (<dir>/<config>/lam<lam>_ridge<ridge>_moment_mask.pt), '
                        'compared with the re-solve')
    p.add_argument('--pnewton_dir', type=Path, default=None,
                   help='root of the existing L_6<0 projected-Newton files (newest lam.._neg<eps>_*.pt is used)')
    p.add_argument('--lam', type=float, required=True)
    p.add_argument('--ridge', type=float, required=True)
    p.add_argument('--eps', type=float, default=1e-2, help='eps in nats per sd of phi (as launch_l6neg.sh)')
    p.add_argument('--eps_factor', type=float, default=10.0, help='pinall10 uses eps * eps_factor')
    p.add_argument('--cols', type=str, default='0:9', help='L_6 columns START:STOP')
    p.add_argument('--addonly_max_iter', type=int, default=6, help='max re-solves of the pinpos loop')
    p.add_argument('--theta_out', type=Path, required=True, help='where the full Theta_reg files go (SCRATCH)')
    p.add_argument('--out', type=Path, required=True, help='small results (json, npz)')
    p.add_argument('--n_scan', type=int, default=200, help='data and generated signals in the radial scan (each)')
    p.add_argument('--time_budget_min', type=float, default=95.0,
                   help='skip a remaining solve if it would end after this (analysis still runs)')
    p.add_argument('--mmap', action='store_true')
    p.add_argument('--threads', type=int, default=None)
    return p.parse_args()


# ----------------------------------------------------------------------------------------------- solves
def solve(ctx, pin=None):
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        Th = SDE._solve_regularised_thomas(ctx['self'], ctx['t'], ctx['M'], ctx['C'], ctx['b'], ctx['v'], ctx['lam'],
                                           ridge=ctx['ridge'], mode=ctx['mode'], dim=ctx['dim'],
                                           schedule=ctx['schedule'], live=ctx['live'], pin=pin).cpu()
    dt = time.time() - t0
    ctx['solve_s'].append(dt)
    return Th, float(ctx['self'].last_reg_residual), dt


def pin_all(U, bounded):
    return bounded.clone(), torch.where(bounded, U, torch.zeros_like(U))


def time_left_ok(ctx, args, t_start):
    est = max(ctx['solve_s']) if ctx['solve_s'] else 0.0
    return (time.time() - t_start + est) / 60 < args.time_budget_min


def save_theta(args, config, arm, Th, t, extra):
    d = args.theta_out / config
    d.mkdir(parents=True, exist_ok=True)
    f = d / f'lam{args.lam:g}_ridge{args.ridge:g}_moment_mask_pinl6_{arm}.pt'
    tmp = f.with_name(f.name + '.tmp')
    # same layout as codes/resolve_theta_reg.py (first node dropped), so existing readers work
    torch.save({'Theta_reg': Th[1:].clone(), 't_reg': torch.as_tensor(t[1:]), 'lam': args.lam, 'ridge': args.ridge,
                'config': config, 'arm': arm, **extra}, tmp)
    tmp.replace(f)
    return f


# --------------------------------------------------------------------------------------------- analysis
def ranks_overlap(lp_a, lp_b):
    """Spearman of log p, and the overlap of the RARE_FRAC lowest-log-p signals (the 'rare' set)."""
    rho = float(stats.spearmanr(lp_a, lp_b)[0])
    k = max(1, int(round(RARE_FRAC * len(lp_a))))
    ra, rb = set(np.argsort(lp_a)[:k]), set(np.argsort(lp_b)[:k])
    return rho, len(ra & rb) / k


def atoms(M, J, Q, wav_type, norm):
    """One real wavelet atom per channel of the Q bank, centred, L2 norm = norm. (n_channels, 1, M) float32."""
    from filters_bank import return_Filters
    f = return_Filters(M, J, Q, device='cpu', wav_type=wav_type)[0]          # (channels, M), Fourier
    x = torch.fft.ifft(f.to(torch.complex128), dim=-1).real
    x = torch.roll(x, M // 2, dims=-1)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-300) * norm
    return x[:, None, :].float()


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    t_start = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    log_mem('start')

    # ---- system and solver inputs (as codes/resolve_theta_reg.py, mask auto = saved live mask)
    config = args.system.name[:-3]
    stem = f'lam{args.lam:g}_ridge{args.ridge:g}_moment_mask'               # resolve_theta_reg.py file names
    args.free_ref = None
    if args.free_dir is not None and (args.free_dir / config / f'{stem}.pt').exists():
        args.free_ref = args.free_dir / config / f'{stem}.pt'
    args.pnewton = None
    if args.pnewton_dir is not None:
        c = sorted((args.pnewton_dir / config).glob(f'{stem}_neg{args.eps:g}_*.pt'), key=lambda f: f.stat().st_mtime)
        args.pnewton = c[-1] if c else None
    print(f'free_ref: {args.free_ref}\npnewton: {args.pnewton}', flush=True)
    system = torch.load(args.system, map_location='cpu', weights_only=False, mmap=args.mmap)
    if system.get('meta', {}).get('reg_mode') != 'moment' or 'Sigma' not in system:
        raise ValueError('needs a moment-mode system with Sigma (eps is set per sd of phi)')
    ns = types.SimpleNamespace(schedule='uniform', ridge=args.ridge, dim=None, n_walkers=None, sigma=None,
                               mask='auto', live_floor=0.0, results_root=None)
    inp = build_inputs(system, config, ns, 'moment')
    n, r = len(inp['t']), inp['r']
    ctx = dict(self=types.SimpleNamespace(num_potentials=r), t=inp['t'], M=inp['M'], C=inp['C'], b=inp['b'],
               v=inp['v'], lam=args.lam, ridge=args.ridge, mode='moment', dim=inp['dim'], schedule='uniform',
               live=inp['live'], solve_s=[])
    a_, b_ = (int(x) for x in args.cols.split(':'))
    cols = list(range(a_, b_))
    U1, bounded = _bounds(n, r, cols, -eps_from_sigma(system['Sigma'], cols, args.eps), inp['live'])
    U10, _ = _bounds(n, r, cols, -eps_from_sigma(system['Sigma'], cols, args.eps * args.eps_factor), inp['live'])
    log_mem('after build_inputs')
    u_last = -U1[-1, cols]
    print(f'[{config}]\n  {n} nodes, r = {r}, L_6 columns {a_}:{b_}, {int(bounded.sum())} bounded entries, '
          f'lam {args.lam:g}, ridge {args.ridge:g}, eps {args.eps:g} nats/sd\n'
          f'  pinned value at the last node (-eps/sd per filter): {[f"{-float(u):.3g}" for u in u_last]}', flush=True)
    if not bool((u_last > 0).all()):
        print('  WARNING: eps/sd = 0 at the last node for some filter (sd of phi_6 infinite?): the pin is not a wall there',
              flush=True)

    Theta, info = {}, {}
    # free
    Th, res, dt = solve(ctx)
    Theta['free'] = Th
    info['free'] = dict(residual=res, solves=1, n_pinned=0, solve_s=dt)
    pos = bounded & (Th > U1)
    print(f'  free: {dt:.0f} s, residual {res:.2e}; L_6 entries above -eps: {int(pos.sum())} of {int(bounded.sum())} '
          f'({int(pos[-1].sum())} of {len(cols)} at the last node)', flush=True)
    if not bool(torch.isfinite(Th).all()):
        print(f'  WARNING: free Theta has {int((~torch.isfinite(Th)).sum())} non-finite entries: pinpos cannot find the '
              f'entries above -eps (NaN comparisons are false) and every comparison with free is meaningless', flush=True)
    if args.free_ref is not None:
        ref = torch.load(args.free_ref, map_location='cpu', weights_only=False)['Theta_reg'].double()
        d = float((ref - Th[1:]).abs().max() / Th[1:].abs().max())
        info['free']['rel_diff_vs_saved'] = d
        print(f'  free re-solve vs saved {args.free_ref.name}: max |diff| / max |Theta| = {d:.2e}', flush=True)

    # pin all at -eps, at -10 eps
    for arm, U in (('pinall', U1), ('pinall10', U10)):
        if not time_left_ok(ctx, args, t_start):
            print(f'  {arm}: SKIPPED (time budget)', flush=True); continue
        Th, res, dt = solve(ctx, pin=pin_all(U, bounded))
        Theta[arm] = Th
        info[arm] = dict(residual=res, solves=1, n_pinned=int(bounded.sum()), solve_s=dt)
        print(f'  {arm}: {dt:.0f} s, residual {res:.2e}', flush=True)

    # pin the positive ones only (add_only loop from the free solve)
    pinned = torch.zeros_like(bounded)
    Th = Theta['free']
    hist = []
    for it in range(args.addonly_max_iter):
        viol = bounded & ~pinned & (Th > U1)
        hist.append(int(viol.sum()))
        if not bool(viol.any()):
            break
        if not time_left_ok(ctx, args, t_start):
            print('  pinpos: STOPPED (time budget)', flush=True); break
        pinned |= viol
        Th, res, dt = solve(ctx, pin=(pinned, torch.where(pinned, U1, torch.zeros_like(U1))))
        print(f'  pinpos round {it + 1}: +{hist[-1]} pinned (total {int(pinned.sum())}), {dt:.0f} s, residual {res:.2e}',
              flush=True)
    if int(pinned.sum()):
        left = int((bounded & ~pinned & (Th > U1)).sum())
        Theta['pinpos'] = Th
        info['pinpos'] = dict(residual=res, solves=len([h for h in hist if h]), n_pinned=int(pinned.sum()),
                              added_per_round=hist, feasible=left == 0, still_above=left,
                              n_pinned_last_node=int(pinned[-1].sum()))
        print(f'  pinpos: {"feasible" if left == 0 else f"NOT feasible ({left} entries still above -eps)"}', flush=True)

    # energies, one pass
    arms = list(Theta)
    with contextlib.redirect_stdout(io.StringIO()):
        E, _ = SDE._reg_energy_grad(ctx['self'], ctx['t'], ctx['M'], ctx['C'], ctx['b'], ctx['v'], args.lam,
                                    [Theta[a] for a in arms], ridge=args.ridge, mode='moment', dim=ctx['dim'],
                                    schedule='uniform')
    for a, e in zip(arms, E):
        info[a]['E'] = float(e)
        info[a]['E_minus_free'] = float(e) - float(E[0])
        info[a]['max_L6_last_node'] = float(Theta[a][-1, cols].max())
        info[a]['n_nonfinite'] = int((~torch.isfinite(Theta[a])).sum())
        save = save_theta(args, config, a, Theta[a], ctx['t'], {'info': info[a], 'eps': args.eps})
        print(f'  E[{a}] = {float(e):.6g}  (E - E_free = {float(e) - float(E[0]):+.6g});  max L_6 at the last node '
              f'{info[a]["max_L6_last_node"]:+.3g}  -> {save}', flush=True)
    last = {a: Theta[a][-1].numpy() for a in arms}
    if args.pnewton is not None:
        last['pnewton'] = torch.load(args.pnewton, map_location='cpu', weights_only=False)['Theta_reg'][-1].double().numpy()
        info['pnewton'] = {'file': args.pnewton.name}
    del system, ctx, inp, Theta
    log_mem('after solves')
    print(f'solves done in {(time.time() - t_start) / 60:.1f} min', flush=True)

    # ---- log p on data and generated signals (Theta at the last node)
    from logp_family_contrib import load_potentials, phi_matrix
    from check_potentials import theta_column_map
    from check_fit_offline import build_data
    cfg = json.loads((args.exp_dir / 'config.json').read_text())
    assert cfg['config_name'] == config, (cfg['config_name'], config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pots = load_potentials(args.exp_dir, cfg, device)
    x1, _ = build_data(int(cfg.get('subseries_len', 512)), int(cfg['M']), int(cfg['n1']), 1)
    x1 = x1.to(device)
    xg = torch.load(args.samples, map_location='cpu', weights_only=False).float()
    xg = xg.reshape(xg.shape[0], 1, -1).to(device)
    print(f'potentials on {device}', flush=True)
    phi_d, phi_g = phi_matrix(pots, x1), phi_matrix(pots, xg)
    assert phi_d.shape[1] == r and phi_g.shape[1] == r, (phi_d.shape, phi_g.shape, r)
    fams = {k: (s, s + w) for k, (s, w) in theta_column_map(pots, r).items()}
    print(f'phi on {len(x1)} data and {len(xg)} generated signals; families '
          + ', '.join(f'{k}[{s}:{e}]' for k, (s, e) in fams.items()), flush=True)
    lp = {a: {'data': phi_d @ th, 'gen': phi_g @ th} for a, th in last.items()}

    refs = ['free'] + (['pnewton'] if 'pnewton' in last else [])
    cmp = {}
    for a in last:
        for ref in refs:
            if a == ref:
                continue
            row = {}
            for s in ('data', 'gen'):
                row[f'spearman_{s}'], row[f'rare5_overlap_{s}'] = ranks_overlap(lp[a][s], lp[ref][s])
            for F, (s0, s1) in fams.items():
                d0 = np.linalg.norm(last[ref][s0:s1])
                row[f'dtheta_rel[{F}]'] = float(np.linalg.norm(last[a][s0:s1] - last[ref][s0:s1]) / max(d0, 1e-300))
                ca, cr = phi_d[:, s0:s1] @ last[a][s0:s1], phi_d[:, s0:s1] @ last[ref][s0:s1]
                row[f'corr_contrib_data[{F}]'] = float(np.corrcoef(ca, cr)[0, 1]) if ca.std() > 0 and cr.std() > 0 else float('nan')
            cmp[f'{a}_vs_{ref}'] = row
    print('\n=== against the reference (Theta at the last node) ===')
    for k, row in cmp.items():
        print(f'  {k:22s} Spearman data {row["spearman_data"]:.4f} gen {row["spearman_gen"]:.4f} | 5 % rarest overlap '
              f'data {row["rare5_overlap_data"]:.3f} gen {row["rare5_overlap_gen"]:.3f}')
        print('      ' + ', '.join(f'{F}: dtheta {row[f"dtheta_rel[{F}]"]:.2e} corr {row[f"corr_contrib_data[{F}]"]:+.4f}'
                                   for F in fams))

    # ---- radial scan
    rng = np.random.default_rng(0)
    i_d = rng.choice(len(x1), min(args.n_scan, len(x1)), replace=False)
    i_g = rng.choice(len(xg), min(args.n_scan, len(xg)), replace=False)
    i_r = np.argsort(lp['free']['data'])[:20]
    norm = float(np.median(x1.reshape(len(x1), -1).norm(dim=-1).cpu().numpy()))
    sets = {'data': x1[i_d], 'gen': xg[i_g], 'rare20': x1[i_r],
            'atoms': atoms(int(cfg['M']), int(cfg['J']), int(cfg['Q']), cfg.get('wav_type', 'morlet'), norm).to(device)}
    scan, scan_sum = {}, {}
    for name, xs in sets.items():
        ph = np.stack([phi_matrix(pots, float(a) * xs) for a in A_GRID], axis=1)        # (signals, len(A), r)
        if not np.isfinite(ph).all():
            print(f'  scan {name}: non-finite phi at a = {A_GRID[~np.isfinite(ph).all(axis=(0, 2))]}', flush=True)
        i1 = int(np.flatnonzero(A_GRID == 1.0)[0])
        for a, th in last.items():
            f = ph @ th                                                                     # (signals, len(A))
            rise = f - f[:, i1:i1 + 1]
            scan[f'{name}|{a}'] = rise.astype(np.float32)
            out = rise[:, A_GRID > 1]
            fin = np.where(np.isfinite(out), out, -np.inf)
            if np.isfinite(fin).any():
                j = np.unravel_index(np.argmax(fin), fin.shape)
                mx, at, sig = float(fin[j]), float(A_GRID[A_GRID > 1][j[1]]), int(j[0])
            else:
                mx, at, sig = float('nan'), float('nan'), -1
            scan_sum[f'{name}|{a}'] = dict(max_rise_nats=mx, at_a=at, signal=sig, n_signals=len(xs),
                                           n_rise_gt_1nat=int((fin.max(1) > 1).sum()),
                                           n_nonfinite=int((~np.isfinite(out)).sum()))
    print(f'\n=== radial scan: max over a in {A_GRID[A_GRID > 1].min():g}..{A_GRID.max():g} of '
          f'Theta.phi(a x) - Theta.phi(x) (nats; > 0 = far bump) ===')
    for k, v in scan_sum.items():
        print(f'  {k:22s} max rise {v["max_rise_nats"]:+.4g} at a = {v["at_a"]:g} (signal {v["signal"]}), '
              f'{v["n_rise_gt_1nat"]}/{v["n_signals"]} signals rise > 1 nat')

    summary = dict(config=config, lam=args.lam, ridge=args.ridge, eps=args.eps, eps_factor=args.eps_factor,
                   cols=args.cols, arms=info, compare=cmp, scan=scan_sum, a_grid=A_GRID.tolist(),
                   minutes=(time.time() - t_start) / 60)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    np.savez_compressed(args.out / 'logp_scan.npz',
                        **{f'logp_{s}|{a}': lp[a][s].astype(np.float32) for a in lp for s in ('data', 'gen')},
                        **{f'theta_last|{a}': th for a, th in last.items()},
                        **{f'scan|{k}': v for k, v in scan.items()},
                        a_grid=A_GRID, fam_names=np.array(list(fams)),
                        fam_ranges=np.array(list(fams.values())))
    print(f'\nsummary -> {args.out / "summary.json"}\nnpz -> {args.out / "logp_scan.npz"}\n'
          f'total {(time.time() - t_start) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
