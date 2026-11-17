"""Open-loop synthesis test of the time-regularised Theta on the 1-D bimodal benchmark (2026-11-13).

Question: which Theta(t) works as a model on its own? Replayed on fresh walkers with NO corrector solve
(option B1, codes/sde_routines.py set_prescribed_theta: the corrector move is h sigma^2 grad(Theta(t_{k+1}) . phi)
at the predicted walkers; the predictor eta_k is still solved from G(x_k) and the interpolant's moment velocity,
the moment target b_k never enters), does it reproduce the target moments?

Per beta, dataset k (default 0), SDE seeds s = 0..S-1 (seed k + 1000 s, as bimodal_lamtest.run_replica):
  1. closed-loop MGD run (forward_regularised, moment system KEPT in --system_dir), as bimodal_lamtest; its walkers
     and moments along t are the BASELINE (measure 6).
  2. Theta(lam) for 'uniform' and 'guth' weights on the lam grid (thomas_batched, as bimodal_lamtest).
  3. candidates (one Theta(t) per seed): C0 lam = 0 (uniform); C1 uniform, lam of the cross-seed CV in the M metric
     on t >= 0.99 (= bimodal_lamtest's uniform_xseed_end0.99, recomputed here identically); C2 uniform, lam of the
     NEW cross-seed CV in the Sigma metric on t >= 0.99 (uniform_xseed_sigma_end0.99, below); C3 guth weights, lam = 1;
     C4 uniform_oracle_end (exact KL of p_Theta(1) to theta*; needs the truth). Both CVs are also scored on t >= 0.999.
  4. open-loop syntheses with seed 0's Theta of each candidate (duplicate Theta = same lam and weights: run once),
     on fresh walkers with one fresh seed per dataset, common to all candidates (SYNTH_SEED0 + k).
  5. check (ii): C0 open loop with seed 0's own SDE seed: compare with the baseline (closed loop, same seed).
--replay_check (smoke size): check (i): a closed loop records its theta_k, then the same seed is replayed open loop
with exactly those theta_k; the final walkers must agree to rounding.

New rule C2 (cross-seed CV in the Sigma metric): for every ordered pair (A, B) of seeds,
    L_AB(lam) = sum_{k: t_k >= e} w_k [ 1/2 Th_A,k(lam)^T S_B,k Th_A,k(lam) - Th_A,k(lam)^T S_B,k Th_B,k(0) ],
w_k trapezoid weights in t, S_B,k = seed B's walker covariance Cov_w(phi) at node k (the system's 'Sigma' = C input),
Th_B(0) = seed B's lam = 0 solve with the run's ridge. Up to a lam-independent term, L_AB = 1/2 |Th_A(lam) -
Th_B(0)|^2 in the S_B metric; B's noise is independent of Th_A. Pooled over the pairs, argmin on the lam grid.
(All statistics are live in this benchmark: thomas_batched.stack_inputs refuses a dead one.)

Measures per run (walkers x at the stored times): diverged = non-finite walkers (first t); max|x| along t / max|x1|;
moment error z_i = (phibar_i(walkers) - target_i) / se_i, se_i = std_i(phi(walkers)) / sqrt(n), at t_end against
phibar(x1) and along t (~50 times) against the interpolant's empirical moments m_t; mass of the small well P(x < 0)
vs the truth; exact D_KL(p_theta* || p_Theta(1)) of the candidate's Theta at the last node.

Usage (repo root):
  full (Jean Zay, GPU):  python lam_selection_criterion/bimodal_openloop.py --out DIR --system_dir SDIR --beta 0.9
  smoke:                 python lam_selection_criterion/bimodal_openloop.py --out DIR --system_dir SDIR --beta 0.9 \
                             --nt 50 --n_bulk 10 --n1 200 --lams 0 1e-3 1 --replay_check
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'codes', ROOT / 'data', ROOT / 'lam_selection_criterion'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import bimodal_lamtest as bl                                        # noqa: E402  (Poly, truth, sampler, grid)
from bimodal_lamtest import (Poly, theta_star, sample, kl_exact, log_density, interp_moments,   # noqa: E402
                             two_phase_schedule, build_inputs, tb, edge_flag, log, GRID, DX)
from codes.sde_routines import SDE, trapezoid_node_weights          # noqa: E402

SYNTH_SEED0 = 777_000            # open-loop syntheses: seed SYNTH_SEED0 + k, never an MGD seed (those are k + 1000 s)
ENDS = (0.99, 0.999)
N_TIMES = 50


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--system_dir', type=Path, required=True, help='where the closed-loop systems are written and kept')
    p.add_argument('--beta', type=float, required=True)
    p.add_argument('--run', type=int, default=0, help='dataset k (x1 drawn with seed k, SDE seeds k + 1000 s)')
    p.add_argument('--seeds', type=int, default=3, help='SDE seeds per dataset (>= 2 for the cross-seed rules)')
    p.add_argument('--n1', type=int, default=1_000_000, help='samples of x1 = walkers (as bimodal_lamtest --n)')
    p.add_argument('--sigma', type=float, default=10.0)
    p.add_argument('--nt', type=int, default=50000)
    p.add_argument('--n_bulk', type=int, default=8334)
    p.add_argument('--t_switch', type=float, default=0.9)
    p.add_argument('--gap_end', type=float, default=5e-5)
    p.add_argument('--regularization', type=float, default=1e-4)
    p.add_argument('--lams', type=float, nargs='+', default=[0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])
    p.add_argument('--replay_check', action='store_true', help='check (i) only: exact replay of recorded theta_k')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


# ------------------------------------------------------------------ SDE runs
def make_solver(args, x1, seed, t_grid, sysdir):
    """As bimodal_lamtest.run_replica: the seed is set BEFORE the SDE draws x_0 and the walkers."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    return SDE(x1.clone(), args.n1, args.n1, t_grid.clone(), args.sigma, {'poly': Poly()}, args.n1,
               device=args.device, regularization=args.regularization, interpolant='Cos',
               potentials_save_dir=sysdir / 'unused_potentials_dir', solve_float64=True)


def phi(x):
    z = x.reshape(-1).double()
    return torch.stack([z, z ** 2 / 2, z ** 3 / 3, z ** 4 / 4], 1)


def moment_stats(x):
    """phibar (4,), standard error of the mean (4,), on CPU float64."""
    f = phi(x)
    return f.mean(0).cpu(), (f.std(0) / math.sqrt(f.shape[0])).cpu()


def measure_steps(t_grid):
    """~N_TIMES step indices k (state after step k is at t_grid[k+1]): half uniform in t, half log-uniform in 1 - t."""
    tt = t_grid[1:].double().numpy()
    return set(int(i) for i in bl.plot_nodes(tt, N_TIMES))


def run_loop(solver, t_grid, record_theta=False, label=''):
    """forward()'s loop (same calls, same RNG use) with the measures; stops at the first non-finite walker.
    Returns dict: x (final walkers or None), diverged_t, maxabs (n_steps,), meas {k: (t, phibar, se)}, thetas."""
    steps = measure_steps(t_grid)
    nt = len(t_grid) - 1
    maxabs = np.full(nt, np.nan)
    meas, thetas, diverged_t = {}, [], None
    solver._rhs_dt_cache = None
    t0 = last = time.time()
    for k in range(nt):
        try:
            x, y, I, eta, theta, dH, bk = solver.iteration_step_projection(solver.x_k, k)
        except FloatingPointError as ex:                    # the predictor's solve refuses non-finite walkers
            diverged_t = float(t_grid[k + 1])
            log(f'{label}: step {k} failed on non-finite values (t = {diverged_t:.6g}): {ex}; stopped')
            break
        solver.x_k = x
        if record_theta:
            thetas.append(theta.detach().double().cpu().reshape(-1))
        finite = bool(torch.isfinite(x).all())
        if not finite:
            diverged_t = float(t_grid[k + 1])
            log(f'{label}: NON-FINITE walkers at step {k} (t = {diverged_t:.6g}); stopped')
            break
        maxabs[k] = float(x.abs().max())
        if k in steps or k == nt - 1:
            m, se = moment_stats(x)
            meas[k] = (float(t_grid[k + 1]), m, se)
        if time.time() - last > 60:
            last = time.time()
            log(f'{label}: step {k + 1}/{nt} ({(time.time() - t0) / (k + 1) * 1e3:.1f} ms/step)')
    log(f'{label}: {time.time() - t0:.0f} s')
    return dict(x=None if diverged_t is not None else solver.x_k.detach().clone(), diverged_t=diverged_t,
                maxabs=maxabs, meas=meas, thetas=torch.stack(thetas) if record_theta and thetas else None)


def summarize_run(out, x1, x0, beta, theta_end=None):
    """Measures 1-4 of one run. x0: the run's interpolant noise (for m_t)."""
    th_star = theta_star(beta)
    p_star = log_density(th_star)[0].exp()
    small_true = float((p_star[GRID < 0]).sum() * DX)
    x1max = float(x1.abs().max())
    res = dict(diverged=out['diverged_t'] is not None, diverged_t=out['diverged_t'],
               maxabs_ratio_peak=float(np.nanmax(out['maxabs']) / x1max) if np.isfinite(out['maxabs']).any() else float('nan'),
               t_of_peak=None, small_well_true=small_true)
    if np.isfinite(out['maxabs']).any() and len(out['maxabs']) == len(out['t_grid']) - 1:
        res['t_of_peak'] = float(out['t_grid'][1:][int(np.nanargmax(out['maxabs']))])
    if theta_end is not None:
        res['kl_theta_end'] = float(kl_exact(th_star, theta_end.reshape(1, -1))[0])
    ks = sorted(out['meas'])
    if not ks:                                             # diverged before the first measuring time
        res.update(along=None, along_max_abs_z=float('nan'), along_rms_z=float('nan'), end_z=None,
                   end_max_abs_z=float('nan'), end_rms_z=float('nan'), small_well=float('nan'))
        return res
    ts = np.array([out['meas'][k][0] for k in ks])
    m_t = interp_moments(x0, x1, ts)                                    # (N, 4) empirical interpolant moments
    z_t = torch.stack([(out['meas'][k][1] - m_t[i]) / out['meas'][k][2] for i, k in enumerate(ks)]).numpy()
    res['along'] = dict(t=ts.tolist(), z=z_t.tolist())
    res['along_max_abs_z'] = float(np.abs(z_t).max())
    res['along_rms_z'] = float(np.sqrt((z_t ** 2).mean()))
    if out['x'] is not None:
        m, se = moment_stats(out['x'])
        target = phi(x1).mean(0).cpu()
        z = ((m - target) / se).numpy()
        res.update(end_z=z.tolist(), end_max_abs_z=float(np.abs(z).max()), end_rms_z=float(np.sqrt((z ** 2).mean())),
                   small_well=float((out['x'] < 0).double().mean()))
    else:
        res.update(end_z=None, end_max_abs_z=float('nan'), end_rms_z=float('nan'), small_well=float('nan'))
    return res


# ------------------------------------------------------------------ rules
def xseed(reps, lams, ends, metric):
    """Cross-seed CV pooled over ordered pairs on t >= e. metric 'M': bimodal_lamtest.add_xseed_rules (uniform);
    'Sigma': the new rule C2. Returns {e: dict(loss_minus_min, selected, per_pair)}."""
    t = reps[0]['t']
    w = trapezoid_node_weights(t)
    out = {}
    for e in ends:
        nodes = np.flatnonzero(t >= e)
        nd = torch.as_tensor(nodes)
        wn = torch.as_tensor(w[nodes])
        pooled, pairs = np.zeros(len(lams)), []
        for a, ra in enumerate(reps):
            Th = ra['Theta']['uniform'][:, nd]                              # (L, nodes, 4)
            for b, rb in enumerate(reps):
                if a == b:
                    continue
                if metric == 'M':
                    loss = tb.data_term(rb['S']['uniform'], Th, nodes, w).numpy()
                else:
                    Sg = tb._sym(rb['S']['uniform']['C'][nd])               # (nodes, 4, 4) Cov_w(phi) of seed B
                    T0 = rb['Theta']['uniform'][0][nd]                      # seed B's lam = 0 solve
                    q = (0.5 * torch.einsum('lki,kij,lkj->lk', Th, Sg, Th)
                         - torch.einsum('lki,kij,kj->lk', Th, Sg, T0))
                    loss = (q * wn).sum(1).numpy()
                pooled += loss
                pairs.append(lams[int(np.argmin(loss))])
        i = int(np.argmin(pooled))
        out[e] = dict(selected=lams[i], edge=edge_flag(lams[i], lams), loss_minus_min=(pooled - pooled[i]).tolist(),
                      per_pair={f'{l_:g}': pairs.count(l_) for l_ in sorted(set(pairs))})
    return out


# ------------------------------------------------------------------ main
def replay_check(args, x1, t_grid):
    """Check (i): closed loop records theta_k; same seed replayed open loop with those theta_k."""
    seed = args.run
    sysdir = args.system_dir
    s1 = make_solver(args, x1, seed, t_grid, sysdir)
    a = run_loop(s1, t_grid, record_theta=True, label='replay (i) closed')
    s2 = make_solver(args, x1, seed, t_grid, sysdir)
    s2.set_prescribed_theta(t_grid[1:].double(), a['thetas'])          # theta_k belongs to t_{k+1}
    b = run_loop(s2, t_grid, label='replay (i) open')
    d = (a['x'] - b['x']).abs()
    rel = float(d.max() / a['x'].abs().max())
    log(f'CHECK (i) exact replay: max |x_open - x_closed| = {float(d.max()):.3e}, relative {rel:.3e} '
        f'(walker dtype {a["x"].dtype}, eps {torch.finfo(a["x"].dtype).eps:.1e})')
    return dict(max_abs=float(d.max()), max_rel=rel, dtype=str(a['x'].dtype))


def main():
    args = parse_args()
    out_dir = args.out / f'beta_{args.beta:.2f}'
    out_dir.mkdir(parents=True, exist_ok=True)
    args.system_dir.mkdir(parents=True, exist_ok=True)
    k = args.run
    x1 = sample(args.n1, args.beta, k).to(args.device)
    t_grid, info = two_phase_schedule(args.nt, args.n_bulk, args.t_switch, args.gap_end)
    lams = sorted(set([0.0] + list(args.lams)))
    if 1.0 not in lams:
        sys.exit('--lams must contain 1 (candidate C3: guth weights, lam = 1)')
    th_star = theta_star(args.beta)
    log(f'beta {args.beta:.2f} dataset {k}: nt {args.nt}, n1 {args.n1}, sigma {args.sigma}, seeds {args.seeds}, '
        f'lams {lams}, device {args.device}')

    if args.replay_check:
        r = replay_check(args, x1, t_grid)
        (out_dir / 'replay_check.json').write_text(json.dumps(r, indent=1))
        return

    # 1-2: closed-loop runs, systems, Theta(lam)
    reps = []
    for s in range(args.seeds):
        seed = k + 1000 * s
        sys_path = args.system_dir / f'bimodal_beta{args.beta:.2f}_run{k:03d}_s{s}.pt'
        solver = make_solver(args, x1, seed, t_grid, args.system_dir)
        t0 = time.time()
        res = solver.forward_regularised(lam=0.0, reg_solver='thomas', reg_ridge=args.regularization,
                                         reg_system_path=sys_path, solve_reg=False, reg_mode='moment')
        log(f'seed {s} (SDE seed {seed}): closed loop {time.time() - t0:.0f} s')
        x_closed, barphi_p = res[0].detach().clone(), res[2].double().cpu()
        theta_t = res[4].double().cpu().reshape(len(res[4]), -1)            # row j = step j, at t_grid[j+1]
        system = torch.load(sys_path, map_location='cpu', weights_only=False)
        S_, Th_ = {}, {}
        for sched in ('uniform', 'guth'):
            a_ = SimpleNamespace(schedule=sched, ridge=args.regularization, dim=None, n_walkers=None, sigma=None,
                                 mask='auto', results_root=None, live_floor=0.0)
            S_[sched] = tb.stack_inputs(build_inputs(system, sys_path.stem, a_, 'moment'))
            Th_[sched] = tb.solve_batched(S_[sched], lams)[0][0]              # (L, n, 4), code sign, on system nodes
        rep = dict(seed=s, sde_seed=seed, t=S_['uniform']['t'], S=S_, Theta=Th_, x0=solver.x_0.detach().cpu(),
                   x_closed=x_closed, barphi_p=barphi_p)
        # Theta(lam = 0) from the float32 system vs the theta_k the run used, matched by time (as run_replica)
        tf = t_grid[1:].double().numpy()
        idx = np.clip(np.searchsorted(tf, rep['t']), 0, len(tf) - 1)
        assert np.abs(tf[idx] - rep['t']).max() < 1e-12, 'system nodes are not step times'
        th_run = theta_t[torch.as_tensor(idx)]
        rel = ((Th_['uniform'][0] - th_run).norm(dim=1) / th_run.norm(dim=1).clamp_min(1e-300)).numpy()
        j = int(np.argmax(rel))
        rep['theta0_vs_theta_t'] = dict(max_rel=float(rel.max()), median_rel=float(np.median(rel)), t_of_max=float(rep['t'][j]),
                                        first_nodes_rel=rel[:5].tolist())
        log(f"seed {s}: |Theta(0) - theta_t| / |theta_t|: max {rel.max():.3e} at t = {rep['t'][j]:.6g}, "
            f"median {np.median(rel):.3e}, first nodes {np.array2string(rel[:5], precision=2)}")
        # baseline (measure 6): closed-loop walkers; along t from barphi_p (row j = after step j-1) with se from Sigma
        steps = sorted(measure_steps(t_grid) | {args.nt - 1})
        Sig = {float(tt): torch.diagonal(torch.as_tensor(Sm), dim1=-2, dim2=-1)
               for tt, Sm in zip(system['t'].double().numpy(), system['Sigma'])}
        meas = {}
        for kk in steps:
            tt = float(t_grid[kk + 1])
            if tt in Sig:                                   # barphi_p of forward_regularised: row j = after step j
                meas[kk] = (tt, barphi_p[kk], (Sig[tt].double() / args.n1).sqrt())
        base_out = dict(x=x_closed, diverged_t=None, maxabs=np.array([float(x_closed.abs().max())]), meas=meas,
                        t_grid=t_grid.double().numpy())
        rep['baseline'] = summarize_run(base_out, x1, rep['x0'], args.beta)
        rep['baseline']['note'] = 'closed loop: max|x| at t_end only; se along t from the system Sigma'
        reps.append(rep)
        del system, solver

    # 3: candidates
    t_sys = reps[0]['t']
    for r in reps[1:]:
        assert np.array_equal(r['t'], t_sys), 'seeds on different node grids'
    kl_end = {sched: kl_exact(th_star, reps[0]['Theta'][sched][:, -1]) for sched in ('uniform', 'guth')}
    rules = {'M': xseed(reps, lams, ENDS, 'M'), 'Sigma': xseed(reps, lams, ENDS, 'Sigma')}
    oracle_lam = lams[int(torch.argmin(kl_exact(th_star, reps[0]['Theta']['uniform'][:, -1])))]
    cands = {'C0_lam0': ('uniform', 0.0),
             'C1_uniform_xseed_end0.99': ('uniform', rules['M'][0.99]['selected']),
             'C2_uniform_xseed_sigma_end0.99': ('uniform', rules['Sigma'][0.99]['selected']),
             'C3_guth_lam1': ('guth', 1.0),
             'C4_uniform_oracle_end': ('uniform', oracle_lam)}
    for name, (sched, lam) in cands.items():
        log(f'{name}: {sched} lam {lam:g}, exact KL(p* || p_Theta(1)) seed 0 = {float(kl_end[sched][lams.index(lam)]):.4g}')
    for metric, rr in rules.items():
        for e, d in rr.items():
            log(f'xseed {metric} t >= {e:g}: selected {d["selected"]:g}{" (GRID EDGE)" if d["edge"] else ""}; '
                f'per pair {d["per_pair"]}; pooled loss - min ' +
                ' '.join(f'{l_:g}: {v:.3g}' for l_, v in zip(lams, d['loss_minus_min'])))

    # 4-5: open-loop syntheses with seed 0's Theta
    r0 = reps[0]
    t_nodes = torch.as_tensor(t_sys)
    results, done = {}, {}

    def synth(sched, lam, seed, label):
        Th = r0['Theta'][sched][lams.index(lam)]
        solver = make_solver(args, x1, seed, t_grid, args.system_dir)
        solver.set_prescribed_theta(t_nodes, Th)
        o = run_loop(solver, t_grid, label=label)
        o['t_grid'] = t_grid.double().numpy()
        res = summarize_run(o, x1, solver.x_0.detach().cpu(), args.beta, theta_end=Th[-1])
        res.update(sched=sched, lam=lam, synth_seed=seed)
        return res, o

    # check (ii): C0 with seed 0's own SDE seed vs the closed-loop baseline
    res_ii, o_ii = synth('uniform', 0.0, r0['sde_seed'], 'check (ii) C0 same seed')
    if o_ii['x'] is not None:
        d = (o_ii['x'] - r0['x_closed']).abs()
        res_ii['vs_closed_max_abs'] = float(d.max())
        res_ii['vs_closed_max_rel'] = float(d.max() / r0['x_closed'].abs().max())
        log(f"CHECK (ii) C0 same seed vs closed loop: max |dx| = {res_ii['vs_closed_max_abs']:.3e} "
            f"(relative {res_ii['vs_closed_max_rel']:.3e})")
    results['check_ii_C0_same_seed'] = res_ii

    synth_seed = SYNTH_SEED0 + k
    for name, (sched, lam) in cands.items():
        key = (sched, lam)
        if key in done:
            results[name] = dict(results[done[key]], same_as=done[key])
            log(f'{name}: same Theta as {done[key]}, not rerun')
            continue
        results[name], _ = synth(sched, lam, synth_seed, name)
        done[key] = name

    # report
    base = reps[0]['baseline']
    print(f"\n######## beta {args.beta:.2f} dataset {k}: open-loop synthesis with seed 0's Theta, synthesis seed {synth_seed}")
    print(f"small well P(x<0): truth {base['small_well_true']:.4f}")
    hdr = (f"{'candidate':>34} | {'lam':>7} | {'div':>3} | {'max|x|/max|x1|':>14} | {'end max|z|':>10} {'end rms z':>9} | "
           f"{'along max|z|':>12} {'rms':>8} | {'P(x<0)':>7} | {'KL Theta(1)':>11}")
    print(hdr)
    rows = [('baseline (closed loop, seed 0)', dict(base, lam=float('nan')))] + list(results.items())
    for name, r in rows:
        print(f"{name:>34} | {r.get('lam', float('nan')):7.0e} | {'yes' if r.get('diverged') else 'no':>3} | "
              f"{r.get('maxabs_ratio_peak', float('nan')):14.3g} | {r['end_max_abs_z']:10.3g} {r['end_rms_z']:9.3g} | "
              f"{r['along_max_abs_z']:12.3g} {r['along_rms_z']:8.3g} | {r['small_well']:7.4f} | "
              f"{r.get('kl_theta_end', float('nan')):11.4g}")
    save = dict(args={kk: str(v) if isinstance(v, Path) else v for kk, v in vars(args).items()}, grid=info, lams=lams,
                rules={m: {f'{e:g}': d for e, d in rr.items()} for m, rr in rules.items()},
                kl_end_seed0={s_: v.tolist() for s_, v in kl_end.items()},
                candidates={n: dict(sched=s_, lam=l_) for n, (s_, l_) in cands.items()},
                baseline=[r['baseline'] for r in reps], results=results,
                theta_seed0={f'{s_}|{l_:g}': reps[0]['Theta'][s_][lams.index(l_)].float() for s_, l_ in cands.values()},
                t_sys=t_sys)
    dest = out_dir / f'openloop_run{k:03d}.pt'
    torch.save(save, dest)
    log(f'-> {dest}')


if __name__ == '__main__':
    main()
