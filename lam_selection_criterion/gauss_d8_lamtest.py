"""Known-truth test of the lam choice for the time-regularised Theta (moment mode): Gaussian, d = 8.

What is tested, what is Guth et al.'s and what is ours, and every own decision (D1..D13) with
the argument against it: README.md in this folder.

Runs the real pipeline of the turbulence runs on X ~ N(0, C), C_ij = rho^|i-j|, with the
statistics phi_ij = x_i x_j (i <= j): SDE.forward_regularised (moment mode, system saved to
disk) -> resolve_theta_reg.build_inputs -> SDE._solve_regularised_thomas for every lam and both
weightings ('uniform', 'guth'), lam_select_cv.lam_path for the current even/odd rule, plus a
block cross-validation. The walker law stays exactly Gaussian, so the true theta(t) is known at
every node (D5) and every rule is scored against it (D6, D7).

Usage, from the repo root:
  full (Jean Zay):  python lam_selection_criterion/gauss_d8_lamtest.py --out DIR --seeds 900 901 902 903 904
  combine only:     python lam_selection_criterion/gauss_d8_lamtest.py --out DIR --combine_only
  smoke (seconds):  python lam_selection_criterion/gauss_d8_lamtest.py --out /tmp/x --nt 60 --n_bulk 10 \
                        --n 200 --seeds 900 901 --lams 0 1e-3 1
"""
import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'codes', ROOT / 'data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from codes.sde_routines import SDE, trapezoid_node_weights          # noqa: E402
from codes.time_schedules import two_phase_schedule                 # noqa: E402
from codes.resolve_theta_reg import build_inputs                    # noqa: E402
from codes import lam_select_cv as lsc                              # noqa: E402

WINDOWS = [(0.05, 0.95), (0.95, 0.99), (0.99, 0.999), (0.999, 1.0)]
NOISE_WINDOWS = [(0.1, 0.9), (0.9, 0.99), (0.99, 0.999), (0.999, 1.0)]   # t < 0.1 skipped as on turbulence
SCHEDULES = ('uniform', 'guth')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[900, 901, 902, 903, 904])
    p.add_argument('--d', type=int, default=8)
    p.add_argument('--rho', type=float, default=0.9, help='C_ij = rho^|i-j| (D3)')
    p.add_argument('--data_seed', type=int, default=0, help='x1 is the same for every seed (D4)')
    p.add_argument('--n', type=int, default=8500, help='walkers = interpolant samples (D8)')
    p.add_argument('--sigma', type=float, default=3.5)
    p.add_argument('--nt', type=int, default=60000)
    p.add_argument('--n_bulk', type=int, default=10000)
    p.add_argument('--t_switch', type=float, default=0.9)
    p.add_argument('--gap_end', type=float, default=5e-5)
    p.add_argument('--regularization', type=float, default=1e-4,
                   help='per-step ridge of the run; the offline solves use the same value as RIDGE')
    p.add_argument('--lams', type=float, nargs='+',
                   default=[0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], help='(D12)')
    p.add_argument('--veto', type=float, default=10.0, help='amplitude veto of lam_select_cv')
    p.add_argument('--block', type=int, default=0, help='block length of the block CV; 0 = auto (D10)')
    p.add_argument('--keep_system', action='store_true', help='keep the saved system files')
    p.add_argument('--system_dir', type=Path, default=None,
                   help='where the ~0.6 GB/seed systems are written while a seed runs (default OUT/systems; '
                        'the launcher puts them on $SCRATCH)')
    p.add_argument('--combine_only', action='store_true')
    p.add_argument('--no_combine', action='store_true', help='skip the combined table (array tasks)')
    p.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or None)
    return p.parse_args()


# ------------------------------------------------------------------ statistics and truth
class Quadratic:
    """phi_ij(x) = x_i x_j for i <= j (D2). x: (B, 1, d). Implements what SDE needs:
    forward, grad (stacked or contracted with a coefficient vector v), num_coefficients."""

    def __init__(self, d):
        self.d = d
        self.iu = torch.triu_indices(d, d)                 # (2, r), i <= j, row-major
        self.num_coefficients = self.iu.shape[1]

    def forward(self, x):
        z = x.reshape(x.shape[0], self.d)
        return z[:, self.iu[0]] * z[:, self.iu[1]]

    __call__ = forward

    def grad(self, x, v=None):
        z = x.reshape(x.shape[0], self.d)
        i, j = self.iu[0].to(z.device), self.iu[1].to(z.device)
        if v is None:                                      # (B, r, 1, d): d phi_ij / d z_l
            B, r = z.shape[0], self.num_coefficients
            g = torch.zeros(B, r, self.d, dtype=z.dtype, device=z.device)
            ar = torch.arange(r, device=z.device)
            g[:, ar, i] += z[:, j]
            g[:, ar, j] += z[:, i]                         # diagonal i = j: 2 z_i
            return g.reshape(B, r, 1, self.d)
        W = torch.zeros(self.d, self.d, dtype=z.dtype, device=z.device)
        W[i, j] = v.reshape(-1).to(z.dtype)
        W = W + W.T                                        # sum_ij v_ij grad phi_ij = (W z)
        return (z @ W).reshape(x.shape)


def data_cov(d, rho):
    idx = np.arange(d)
    return rho ** np.abs(idx[:, None] - idx[None, :])     # unit diagonal: tr C / d = 1 (D3)


def make_data(args):
    rng = np.random.default_rng(args.data_seed)
    C = data_cov(args.d, args.rho)
    L = np.linalg.cholesky(C)
    x1 = rng.standard_normal((args.n, args.d)) @ L.T
    return torch.as_tensor(x1, dtype=torch.float32).reshape(args.n, 1, args.d), C


def interp_second_moments(x0, x1, t):
    """Empirical E[I_t I_t^T] of the interpolant samples I_t = cos a x0 + sin a x1 (the
    corrector's target), at the times t: (N, d, d) float64 (D5)."""
    n = x0.shape[0]
    Z, X = x0.reshape(n, -1).double(), x1.reshape(n, -1).double()
    A, B, Cx = Z.T @ Z / n, X.T @ X / n, Z.T @ X / n
    a = torch.as_tensor(np.pi * np.asarray(t, dtype=np.float64) / 2)[:, None, None]
    c, s = torch.cos(a), torch.sin(a)
    return c ** 2 * A + s ** 2 * B + c * s * (Cx + Cx.T)


def theta_of_cov(S, iu):
    """Code-sign theta of N(0, S) for phi_ij = x_i x_j: theta.phi = -1/2 x^T S^-1 x."""
    P = torch.linalg.inv(S)
    th = -P[:, iu[0], iu[1]]
    diag = iu[0] == iu[1]
    th[:, diag] = th[:, diag] / 2
    return th


def local_kl(e, S, iu):
    """1/2 e^T F e at N(0, S), F the Fisher information of phi (D6) = tr((Delta S)^2),
    Delta symmetric with Delta_ii = e_ii, Delta_ij = e_ij / 2. e, S: (N, r), (N, d, d)."""
    out = torch.empty(e.shape[0], dtype=torch.float64)
    for a in range(0, e.shape[0], 8192):
        E, Sa = e[a:a + 8192].double(), S[a:a + 8192]
        D = torch.zeros_like(Sa)
        D[:, iu[0], iu[1]] = E
        D = (D + D.transpose(1, 2)) / 2                   # off-diagonal e_ij / 2, diagonal e_ii
        DS = D @ Sa
        out[a:a + 8192] = (DS * DS.transpose(1, 2)).sum((1, 2))
    return out


# ------------------------------------------------------------------ diagnostics
def acf_tau(e, lmax=50):
    """Median over statistics of acf(1), acf(10) and tau_int = 1 + 2 sum_{L<=lmax} acf(L)
    of the rows of e (time x statistics)."""
    e = e - e.mean(0)
    lmax = max(1, min(lmax, len(e) // 4))
    v = (e ** 2).mean(0)
    v[v == 0] = np.inf
    acf = np.stack([(e[:-L] * e[L:]).mean(0) / v for L in range(1, lmax + 1)])
    tau = 1 + 2 * acf.sum(0)
    a10 = acf[min(9, lmax - 1)]
    return float(np.median(acf[0])), float(np.median(a10)), float(np.median(tau))


def moving_avg(x, w):
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), 0)
    h = w // 2
    lo = np.clip(np.arange(len(x)) - h, 0, len(x))
    hi = np.clip(np.arange(len(x)) + h + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def noise_diagnostics(theta_nodes, theta_star, t, inp, ridge):
    """Exact residual theta_t - theta* (truth known) and the practical moving-average
    estimator (D13). Returns {window: {...}} and the largest practical tau_int."""
    out, tau_est_max = {}, 1.0
    h = np.asarray(inp['h'], dtype=np.float64)
    scale = inp['n_walkers'] * h * inp['sigma'] ** 2 / 2
    for lo, hi in NOISE_WINDOWS:
        s = np.flatnonzero((t >= lo) & (t < hi))
        if len(s) < 20:
            continue
        e = (theta_nodes[s] - theta_star[s]).numpy()
        a1, a10, tau = acf_tau(e)
        w = min(501, max(5, (len(s) // 5) | 1))
        X = theta_nodes[s].numpy()
        r_ = (X - moving_avg(X, w))[w // 2: len(s) - w // 2]
        _, _, tau_est = acf_tau(r_) if len(r_) >= 20 else (np.nan, np.nan, np.nan)
        if np.isfinite(tau_est):
            tau_est_max = max(tau_est_max, tau_est)
        z2 = []                                            # noise model: Cov = 2 M^-1 / (n h sigma^2)
        for k in s[:: max(1, len(s) // 2000)]:
            Mk = lsc._sym(inp['M'][k])
            Mk = Mk + ridge * torch.diag(torch.diagonal(Mk))
            ek = theta_nodes[k] - theta_star[k]
            z2.append(float(scale[k] * ek @ Mk @ ek) / len(ek))
        out[f'[{lo},{hi})'] = dict(steps=int(len(s)), acf1=a1, acf10=a10, tau_exact=tau,
                                   tau_movavg=float(tau_est), movavg_window=w,
                                   z2_median=float(np.median(z2)))
    return out, tau_est_max


# ------------------------------------------------------------------ rules
def block_cv(inp, lams, B, veto_by_lam):
    """D10: two folds of alternate blocks of B nodes; held-out nodes kept with zero data
    weight (M_k = 0, b_k = 0), scored with the data term (trapezoid weights, D11)."""
    n = len(inp['t'])
    w = trapezoid_node_weights(inp['t'])
    blk = (np.arange(n) // B) % 2
    zM, zb = torch.zeros_like(inp['M'][0]), torch.zeros_like(inp['b'][0])
    loss = {}
    for lam in lams:
        if lam == 0:
            loss[lam] = np.nan                             # no prediction at held-out nodes
            continue
        tot = 0.0
        for f in (0, 1):
            held = np.flatnonzero(blk == f)
            hm = blk == f
            inp_f = dict(inp)
            inp_f['M'] = [zM if hm[k] else Mk for k, Mk in enumerate(inp['M'])]
            inp_f['b'] = [zb if hm[k] else bk for k, bk in enumerate(inp['b'])]
            Th, _ = lsc.solve(inp_f, lam)
            tot += lsc._data_term(inp, Th[torch.as_tensor(held)], held, w)
        loss[lam] = tot
    ok = [lam for lam in lams if lam > 0 and not veto_by_lam[lam] and np.isfinite(loss[lam])]
    sel = min(ok, key=lambda l_: loss[l_]) if ok else None
    return loss, sel


def summarize(kl, t, wts):
    d = {'end': float(kl[-1])}
    for lo, hi in WINDOWS:
        s = (t >= lo) & (t < hi) if hi < 1 else (t >= lo)
        d[f'[{lo},{hi})'] = float((kl[s] * wts[s]).sum() / wts[s].sum()) if s.any() else float('nan')
    return d


def edge_flag(lam, lams):
    pos = [l_ for l_ in lams if l_ > 0]
    return lam is not None and (lam == max(lams) or (pos and lam == min(pos)))


# ------------------------------------------------------------------ one seed
def run_seed(args, seed, x1, C):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t_grid, info = two_phase_schedule(args.nt, args.n_bulk, args.t_switch, args.gap_end)
    pot = Quadratic(args.d)
    sysdir = args.system_dir or args.out / 'systems'
    sysdir.mkdir(parents=True, exist_ok=True)
    sys_path = sysdir / f'gauss_d{args.d}_seed{seed}.pt'
    solver = SDE(x1.clone(), args.n, args.n, t_grid.clone(), args.sigma, {'quad': pot}, args.n,
                 device='cpu', regularization=args.regularization, interpolant='Cos',
                 potentials_save_dir=args.out / 'unused_potentials_dir', solve_float64=True)
    t0 = time.time()
    with contextlib.redirect_stderr(io.StringIO()):                 # tqdm bar
        out = solver.forward_regularised(lam=0.0, reg_solver='thomas', reg_ridge=args.regularization,
                                         reg_system_path=sys_path, solve_reg=False, reg_mode='moment')
    t_sde = time.time() - t0
    theta_t = out[4].double()                                        # (nt, r), row j at t_grid[j+1]
    x0 = solver.x_0

    system = torch.load(sys_path, map_location='cpu', weights_only=False)
    t = system['t'].double().numpy()
    iu = pot.iu
    S_hat = interp_second_moments(x0, x1, t)
    theta_star = theta_of_cov(S_hat, iu)
    a_end = np.pi * t[-1] / 2
    S_pop_end = torch.as_tensor(np.cos(a_end) ** 2 * np.eye(args.d) + np.sin(a_end) ** 2 * C)[None]
    theta_pop_end = theta_of_cov(S_pop_end, iu)[0]
    wts = trapezoid_node_weights(t)

    tf = t_grid.double().numpy()
    idx = np.clip(np.searchsorted(tf[1:], t), 0, len(tf) - 2)      # theta_t row on each node
    assert np.abs(tf[1:][idx] - t).max() < 1e-12
    theta_nodes = theta_t[idx]

    res = dict(seed=seed, n_nodes=len(t), r=int(system['num_potentials']), grid=info, t_sde_s=t_sde,
               lams=sorted(set([0.0] + list(args.lams))), rules={}, cv={}, kl_curves={}, theta_end={})
    lams = res['lams']
    kl_by = {}
    for sched in SCHEDULES:
        a_ = SimpleNamespace(schedule=sched, ridge=args.regularization, dim=None, n_walkers=None, sigma=None,
                             mask='auto', results_root=None, live_floor=0.0)
        inp = build_inputs(system, sys_path.stem, a_, 'moment')
        t1 = time.time()
        path = lsc.lam_path(inp, lams, veto=args.veto, verbose=False)
        Theta = path['Theta']                                         # (L, n, r) float32
        veto_by_lam = dict(zip(path['lams'], path['veto']))
        if sched == 'uniform':
            th0 = Theta[0].double()                                   # lam = 0 with the run's ridge
            res['theta0_vs_theta_t_maxrel'] = float(((th0 - theta_nodes).norm(dim=1) /
                                                     theta_nodes.norm(dim=1).clamp_min(1e-300)).max())
            noise, tau_est = noise_diagnostics(theta_nodes, theta_star, t, inp, args.regularization)
            res['noise'] = noise
            B = args.block or max(20, int(math.ceil(10 * tau_est)))
            res['block_B'] = B
        for l_i, lam in enumerate(path['lams']):
            kl = local_kl(Theta[l_i].double() - theta_star, S_hat, iu)
            kl_by[(sched, lam)] = kl
            res['kl_curves'][f'{sched}|{lam:g}'] = kl.float()
            res['theta_end'][f'{sched}|{lam:g}'] = Theta[l_i, -1].double()
        bl_loss, bl_sel = block_cv(inp, path['lams'], res['block_B'], veto_by_lam)
        res['cv'][sched] = dict(eo_loss=path['cv_loss'].tolist(), eo_selected=path['lam_selected'],
                                block_loss=[bl_loss[l_] for l_ in path['lams']], block_selected=bl_sel,
                                veto=[bool(v) for v in path['veto']], amp_ratio=path['amp_ratio'].tolist(),
                                chi2=None if path['chi2'] is None else path['chi2'].tolist(),
                                solve_s=time.time() - t1)
        ends = {lam: float(kl_by[(sched, lam)][-1]) for lam in path['lams']}
        mids = {lam: summarize(kl_by[(sched, lam)].numpy(), t, wts)['[0.05,0.95)'] for lam in path['lams']}
        choices = {f'{sched}_cv_eo': path['lam_selected'], f'{sched}_cv_block': bl_sel,
                   f'{sched}_oracle_end': min(ends, key=ends.get), f'{sched}_oracle_mid': min(mids, key=mids.get)}
        if sched == 'guth':
            choices['guth_lam1'] = 1.0
        for name, lam in choices.items():
            if lam is None or (sched, lam) not in kl_by:
                res['rules'][name] = dict(lam=lam)
                continue
            kl = kl_by[(sched, lam)]
            e_pop = res['theta_end'][f'{sched}|{lam:g}'] - theta_pop_end
            res['rules'][name] = dict(lam=lam, edge=edge_flag(lam, path['lams']), **summarize(kl.numpy(), t, wts),
                                      end_vs_population=float(local_kl(e_pop[None], S_pop_end, iu)[0]))
    kl0 = local_kl(theta_nodes - theta_star, S_hat, iu)
    res['rules']['lam0'] = dict(lam=0.0, edge=False, **summarize(kl0.numpy(), t, wts),
                                end_vs_population=float(local_kl((theta_nodes[-1] - theta_pop_end)[None],
                                                                 S_pop_end, iu)[0]))
    res['kl_curves']['theta_t'] = kl0.float()
    res['t'] = torch.as_tensor(t)
    res['theta_star_end'] = theta_star[-1]
    res['kl_theta_star'] = args.d / 4                                # KL(theta*) for the relative error (D6)
    res['t_total_s'] = time.time() - t0
    dest = args.out / f'seed_{seed}.pt'
    tmp = dest.with_name(dest.name + '.tmp')                         # atomic: array tasks may combine
    torch.save(res, tmp)
    tmp.replace(dest)
    if not args.keep_system:
        sys_path.unlink(missing_ok=True)
    return res


# ------------------------------------------------------------------ report
RULE_ORDER = ['lam0', 'guth_lam1', 'uniform_cv_eo', 'uniform_cv_block', 'guth_cv_eo', 'guth_cv_block',
              'uniform_oracle_end', 'uniform_oracle_mid', 'guth_oracle_end', 'guth_oracle_mid']


def print_seed(res):
    solves = ', '.join(f"{s} solves {res['cv'][s]['solve_s']:.0f} s" for s in SCHEDULES)
    print(f"\n=== seed {res['seed']}: {res['n_nodes']} nodes, r = {res['r']}, total {res['t_total_s']:.0f} s "
          f"(SDE {res['t_sde_s']:.0f} s, {solves}), "
          f"block B = {res['block_B']}, max rel |Theta(0) - theta_t| = {res['theta0_vs_theta_t_maxrel']:.1e}")
    print('noise of theta_t (exact residual vs truth; tau_int = 1 for independent steps; '
          'moving-average estimator alongside):')
    for w, d in res['noise'].items():
        print(f"  t {w:>14}  steps {d['steps']:6d}  acf(1) {d['acf1']: .3f}  acf(10) {d['acf10']: .3f}  "
              f"tau_int exact {d['tau_exact']:6.2f}  moving-avg est {d['tau_movavg']:6.2f}  "
              f"E[z^2] (noise model) {d['z2_median']:.3g}")
    print(f"{'rule':>20} {'lam':>8} | {'local KL (nats): end':>21} {'[0.05,0.95)':>12} {'[0.95,0.99)':>12} "
          f"{'[0.99,0.999)':>12} {'[0.999,1)':>11}")
    for name in RULE_ORDER:
        d = res['rules'].get(name)
        if d is None or 'end' not in d:
            print(f"{name:>20} {'none':>8}")
            continue
        print(f"{name:>20} {d['lam']:8.0e} | {d['end']:21.4g} {d['[0.05,0.95)']:12.4g} {d['[0.95,0.99)']:12.4g} "
              f"{d['[0.99,0.999)']:12.4g} {d['[0.999,1.0)']:11.4g}" + ('  (grid edge)' if d.get('edge') else ''))


def combine(args):
    files = sorted(args.out.glob('seed_*.pt'))
    if not files:
        print('no seed files'); return
    R = [torch.load(f, map_location='cpu', weights_only=False) for f in files]
    seeds = [r['seed'] for r in R]
    print(f"\n######## combined over {len(R)} seeds {seeds}  (local KL in nats; relative = sqrt(KL / {R[0]['kl_theta_star']:g}))")
    print(f"{'rule':>20} | {'lam chosen per seed':>34} | {'KL end: median [min, max]':>30} {'rel':>6} | "
          f"{'KL [0.05,0.95)':>14} {'KL [0.999,1)':>13} | {'KL end vs population':>20}")
    x1, _ = make_data(args)
    phi = Quadratic(args.d).forward(x1).double()
    lp_true = phi @ R[0]['theta_star_end']                       # theta* at t_end, seed 900's Z (D5)
    agree = {}
    for name in RULE_ORDER:
        rows = [r['rules'].get(name, {}) for r in R]
        if not all('end' in d for d in rows):
            print(f"{name:>20} | {'(missing in some seed)':>34}")
            continue
        ends = np.array([d['end'] for d in rows])
        lamtxt = ' '.join(f"{d['lam']:.0e}" for d in rows)
        pop = np.median([d.get('end_vs_population', np.nan) for d in rows])
        print(f"{name:>20} | {lamtxt:>34} | {np.median(ends):10.4g} [{ends.min():8.3g}, {ends.max():8.3g}] "
              f"{math.sqrt(np.median(ends) / R[0]['kl_theta_star']):6.3f} | "
              f"{np.median([d['[0.05,0.95)'] for d in rows]):14.4g} {np.median([d['[0.999,1.0)'] for d in rows]):13.4g} | "
              f"{pop:20.4g}")
        if name == 'lam0':
            continue
        key = lambda r, d: f"{'guth' if name.startswith('guth') else 'uniform'}|{d['lam']:g}"
        lps = np.stack([(phi @ r['theta_end'][key(r, d)]).numpy() for r, d in zip(R, rows)])
        c = np.corrcoef(lps)[np.triu_indices(len(R), 1)] if len(R) > 1 else np.array([np.nan])
        ct = [np.corrcoef(lp, lp_true.numpy())[0, 1] for lp in lps]
        agree[name] = (np.median(c), np.median(ct))
    print('\nlog p on the data x1 at the last node (as the turbulence check): '
          'median corr between seeds | median corr with the true log p')
    for name, (c, ct) in agree.items():
        print(f'  {name:>20}  {c:7.4f} | {ct:7.4f}')
    print('\nnoise of theta_t, median over seeds (tau_int exact | moving-average estimate | E[z^2]):')
    for w in R[0]['noise']:
        print(f"  t {w:>14}  {np.median([r['noise'][w]['tau_exact'] for r in R]):6.2f} | "
              f"{np.median([r['noise'][w]['tau_movavg'] for r in R]):6.2f} | "
              f"{np.median([r['noise'][w]['z2_median'] for r in R]):.3g}")
    plot(R[0], args.out)


def plot(res, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = res['t'].numpy()
    x = np.maximum(1 - t, 1e-7)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    curves = [('theta_t (lam=0)', res['kl_curves']['theta_t'], 'k')]
    for name, col in (('guth_lam1', 'C3'), ('uniform_cv_eo', 'C0'), ('uniform_cv_block', 'C2'),
                      ('uniform_oracle_end', 'C1')):
        d = res['rules'].get(name, {})
        if d.get('lam') is not None:
            sched = 'guth' if name.startswith('guth') else 'uniform'
            curves.append((f"{name} (lam={d['lam']:.0e})", res['kl_curves'][f"{sched}|{d['lam']:g}"], col))
    for lab, kl, col in curves:
        ax[0].plot(x, kl.numpy(), lw=0.7, color=col, label=lab)
    ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].invert_xaxis()
    ax[0].set_xlabel('1 - t'); ax[0].set_ylabel('local KL to the truth (nats)'); ax[0].legend(fontsize=7)
    lams = [l_ for l_ in res['lams'] if l_ > 0]
    for sched, col in (('uniform', 'C0'), ('guth', 'C3')):
        ax[1].plot(lams, [float(res['kl_curves'][f'{sched}|{l_:g}'][-1]) for l_ in lams], 'o-', color=col,
                   label=f'{sched}: KL at the last node')
        for kind, ls in (('eo', '--'), ('block', ':')):
            sel = res['cv'][sched][f'{kind}_selected']
            if sel:
                ax[1].axvline(sel, color=col, ls=ls, lw=1, label=f'{sched} CV {kind} -> {sel:.0e}')
    ax[1].axhline(float(res['kl_curves']['theta_t'][-1]), color='k', lw=0.8, label='lam = 0')
    ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('lam')
    ax[1].legend(fontsize=7)
    fig.suptitle(f"Gaussian d=8, seed {res['seed']}: error of Theta_reg against the exact theta")
    fig.savefig(out / f"gauss_d8_seed{res['seed']}.png", dpi=120)
    print(f"figure -> {out / ('gauss_d8_seed%d.png' % res['seed'])}")


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    # combine() pools every seed_*.pt in OUT, so all seeds there must share the problem settings:
    # adding seeds to an OUT is fine, changing --rho / --nt / ... needs a new OUT.
    run_only = {'seeds', 'out', 'combine_only', 'no_combine', 'threads', 'keep_system', 'system_dir'}
    setting = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in run_only}
    cfg_path = args.out / 'args.json'
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        diff = {k: (old.get(k), v) for k, v in setting.items() if old.get(k) != v}
        if diff and not args.combine_only:
            sys.exit(f'{args.out} holds seeds run with other settings {diff}: use a new --out')
    else:
        cfg_path.write_text(json.dumps(setting, indent=1))
    if not args.combine_only:
        x1, C = make_data(args)
        for seed in args.seeds:
            print_seed(run_seed(args, seed, x1, C))
    if not args.no_combine:
        combine(args)


if __name__ == '__main__':
    main()
