"""Known-truth test of the lam choice for the time-regularised Theta: the paper's 1-D bimodal benchmark.

Target (draft Fig. 1, bimodal_experiment): log p*(x) = -beta x^4 + 5 beta x^2 + beta x / 2 + const,
statistics phi_a = x^a / a (a = 1..4, as bimodal_experiment), so in the CODE sign
(log p = theta . phi + const) theta* = beta (1/2, 10, 0, -4).

Unlike the Gaussian d = 8 test, the walker law is not in the exponential family and the noise of
the per-step theta_t is expected to be correlated in time (1-D validation of 2026-09-30, sigma = 10:
lag-1 0.18-0.99), which is the regime of the turbulence runs. One run = one dataset x1 of n
samples at one beta, as one run of the paper's ensemble.

Pipeline (the turbulence one): SDE.forward_regularised (moment mode, two-phase grid, GPU) ->
saved system -> resolve_theta_reg.build_inputs -> Theta(lam) for both weightings. The solves use
thomas_batched (same solve as SDE._solve_regularised_thomas, all lam at once; checked by
check_thomas_batched.py), which is what makes ~100 solves of a 50000-node system affordable.

Truth, exact in 1-D (no Monte Carlo):
  - theta* (population) and the exact D_KL(p_theta* || p_Theta) at the last node, by quadrature on a
    grid of [-10, 10] (as bimodal_experiment/kl_bimodal.ipynb): the paper's metric.
  - theta_ME(t): the max-ent fit of the interpolant's EMPIRICAL moments m_t (this x1, this run's Z),
    by Newton on the same grid, at every node. At the last node it is the max-likelihood fit of x1:
    the floor no estimator from these n samples can beat. Along the run it is the reference for the
    time windows and for the noise of theta_t. Caveat: MGD's theta_t estimates the Fokker-Planck
    projection, which differs from theta_ME(m_t) at finite sigma (0.13 % at sigma = 10 in the 09-30
    1-D check), so at sigma = 10 that bias is part of the "noise" measured here.
    Where the empirical moments admit no max-ent density on R (theta_4 >= 0; checked 2026-10-05 on a
    beta = 1.9 interpolant: t in [0.20, 0.36], skewed and nearly Gaussian kurtosis), the grid fit only
    exists because the grid is cut at +-10: those nodes are excluded from every theta_ME-based number
    (time windows, mid oracle, noise; the noise uses the longest valid stretch of each window).

Rules (per run, per weighting 'uniform' / 'guth'): even/odd CV (lam_select_cv.lam_path), block CV
for each block length in --blocks, oracle at the end (exact KL to theta*), oracle mid-run (local KL
to theta_ME over [0.05, 0.95)), guth at lam = 1, and lam = 0 (raw theta_t).

Usage, from the repo root:
  one beta, runs 0..9 (Jean Zay, GPU):  python lam_selection_criterion/bimodal_lamtest.py --out DIR --beta 0.9 --runs 0 1 2 ...
  combine only:                         python lam_selection_criterion/bimodal_lamtest.py --out DIR --combine_only
  smoke (seconds, CPU):                 python lam_selection_criterion/bimodal_lamtest.py --out /tmp/x --nt 60 --n_bulk 10 \
                                            --n 200 --beta 0.5 --runs 0 1 --lams 0 1e-3 1 --blocks 5 10
"""
import argparse
import functools
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
for p in (ROOT, ROOT / 'codes', ROOT / 'data', ROOT / 'lam_selection_criterion'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import codes.sde_routines as _sr                                    # noqa: E402
from codes.sde_routines import SDE, trapezoid_node_weights          # noqa: E402
from codes.time_schedules import two_phase_schedule                 # noqa: E402
from codes.resolve_theta_reg import build_inputs                    # noqa: E402
import thomas_batched as tb                                         # noqa: E402

# SDE progress bar: one update a minute, so `tail -f` on the slurm log shows the step rate without
# flooding it (the TQDM_MININTERVAL environment variable has no effect in tqdm 4.65, checked)
_sr.tqdm = functools.partial(_sr.tqdm, mininterval=60, maxinterval=60)

R = 4
DENOM = torch.arange(1, R + 1, dtype=torch.float64)                  # phi_a = x^a / a
WINDOWS = [(0.05, 0.95), (0.95, 0.99), (0.99, 0.999), (0.999, 1.0)]
NOISE_WINDOWS = [(0.1, 0.9), (0.9, 0.99), (0.99, 0.999), (0.999, 1.0)]
ACF_LAGS = (1, 10, 100, 1000)
SCHEDULES = ('uniform', 'guth')
X_RANGE, N_GRID = (-10.0, 10.0), 8001                              # as kl_bimodal.ipynb
GRID = torch.linspace(*X_RANGE, N_GRID, dtype=torch.float64)
PHI = GRID[:, None] ** DENOM / DENOM                                # (G, 4)
DX = float(GRID[1] - GRID[0])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--beta', type=float, default=0.9)
    p.add_argument('--runs', type=int, nargs='+', default=list(range(10)),
                   help='run k: dataset drawn with seed k, SDE with seed k')
    p.add_argument('--n', type=int, default=1_000_000, help='samples of x1 = walkers (paper: n1 = 1e6)')
    p.add_argument('--sigma', type=float, default=10.0, help='paper: 10')
    p.add_argument('--nt', type=int, default=50000, help='paper: 5e4')
    p.add_argument('--n_bulk', type=int, default=8334, help='two-phase bulk steps (turbulence ratio 1:5)')
    p.add_argument('--t_switch', type=float, default=0.9)
    p.add_argument('--gap_end', type=float, default=5e-5)
    p.add_argument('--regularization', type=float, default=1e-4,
                   help='per-step ridge of the run = ridge of the offline solves (turbulence reg1e-4 arm)')
    p.add_argument('--lams', type=float, nargs='+',
                   default=[0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])
    p.add_argument('--blocks', type=int, nargs='+', default=[20, 200, 2000], help='block CV lengths (nodes)')
    p.add_argument('--veto', type=float, default=10.0)
    p.add_argument('--n_plot', type=int, default=3000, help='nodes kept for the Theta(t) curves of the notebook')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--keep_system', action='store_true')
    p.add_argument('--system_dir', type=Path, default=None)
    p.add_argument('--combine_only', action='store_true')
    p.add_argument('--no_combine', action='store_true')
    p.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or None)
    return p.parse_args()


# ------------------------------------------------------------------ model, data, truth
class Poly:
    """phi_a(x) = x^a / a, a = 1..4, for x of shape (B, 1) (scalar signal)."""
    num_coefficients = R

    def forward(self, x):
        z = x.reshape(x.shape[0], 1)
        return torch.cat([z, z ** 2 / 2, z ** 3 / 3, z ** 4 / 4], dim=1)

    __call__ = forward

    def grad(self, x, v=None):
        z = x.reshape(x.shape[0], 1)
        g = torch.cat([torch.ones_like(z), z, z ** 2, z ** 3], dim=1)      # d phi_a / dx = x^(a-1)
        if v is None:
            return g.reshape(x.shape[0], R, 1)
        return (g @ v.reshape(-1).to(g.dtype)).reshape(x.shape)


def theta_star(beta):
    """Code sign: log p* = theta* . phi + const."""
    return torch.tensor([beta / 2, 10 * beta, 0.0, -4 * beta], dtype=torch.float64)


def sample(n, beta, seed):
    """Inverse CDF on a fine grid (the paper's bimodal_utils.bimodal does the same with quad)."""
    x = np.linspace(-6, 6, 200001)
    lp = -beta * x ** 4 + 5 * beta * x ** 2 + beta * x / 2
    p = np.exp(lp - lp.max())
    cdf = np.concatenate([[0], np.cumsum((p[1:] + p[:-1]) / 2)])
    cdf /= cdf[-1]
    u = np.random.default_rng(seed).uniform(size=n)
    return torch.as_tensor(np.interp(u, cdf, x), dtype=torch.float32).reshape(n, 1)


def log_density(theta):
    """Normalised log p_theta on GRID, (N, G); rows with theta_4 >= 0 are +inf-normalisation -> nan."""
    th = torch.as_tensor(theta, dtype=torch.float64).reshape(-1, R)
    lv = th @ PHI.T
    lv = lv - (torch.logsumexp(lv, 1, keepdim=True) + math.log(DX))
    lv[th[:, 3] >= 0] = float('nan')
    return lv


def kl_exact(theta_ref, theta):
    """D_KL(p_ref || p_theta) by quadrature; inf where theta is not normalisable (theta_4 >= 0)."""
    lr = log_density(theta_ref)[0]
    lq = log_density(theta)
    p = lr.exp()
    out = ((p * (lr - lq)).sum(1) * DX)
    out[torch.isnan(out)] = float('inf')
    return out


def interp_moments(x0, x1, t):
    """Empirical E[phi(I_t)], I_t = cos a Z + sin a X, from the mixed sample moments E[Z^j X^l],
    j + l <= 4 (exact for the empirical measure). (N, 4) float64."""
    Z, X = x0.reshape(-1).double().cpu(), x1.reshape(-1).double().cpu()
    mix = {(j, l): float((Z ** j * X ** l).mean()) for j in range(5) for l in range(5 - j)}
    a = np.pi * np.asarray(t, dtype=np.float64) / 2
    c, s = np.cos(a), np.sin(a)
    out = np.zeros((len(t), R))
    for p_ in range(1, R + 1):
        out[:, p_ - 1] = sum(math.comb(p_, j) * c ** j * s ** (p_ - j) * mix[(j, p_ - j)] for j in range(p_ + 1)) / p_
    return torch.as_tensor(out)


def maxent_fit(m, iters=200, chunk=1000, device='cpu'):
    """Max-ent theta (code sign) with E_theta[phi] = m on GRID, for each row of m (N, 4), by damped
    Newton on the convex dual log Z(theta) - theta . m. Returns theta (N, 4), the Fisher matrix
    Cov_theta(phi) (N, 4, 4) and the largest relative moment residual (all on CPU)."""
    N = m.shape[0]
    th_all, F_all, worst = torch.empty(N, R, dtype=torch.float64), torch.empty(N, R, R, dtype=torch.float64), 0.0
    phi = PHI.to(device)
    scale = phi.abs().mean(0)                                           # residual normalisation
    eye = torch.eye(R, dtype=torch.float64, device=device)

    def stats(th):
        lv = th @ phi.T
        lz = torch.logsumexp(lv, 1)
        w = (lv - lz[:, None]).exp()                                    # grid weights, sum 1
        E = w @ phi
        F = (w[:, :, None] * phi).transpose(1, 2) @ phi - E[:, :, None] * E[:, None, :]
        return lz, E, F

    for a in range(0, N, chunk):
        mm = m[a:a + chunk].double().to(device)
        v = (2 * mm[:, 1] - mm[:, 0] ** 2).clamp_min(1e-6)            # Gaussian start
        th = torch.stack([mm[:, 0] / v, -1 / v, torch.zeros_like(v), torch.zeros_like(v)], 1)
        lz, E, F = stats(th)
        obj = lz - (th * mm).sum(1)
        for _ in range(iters):
            g = E - mm
            if float((g.abs() / scale).max()) < 1e-11:
                break
            d = torch.sqrt(torch.diagonal(F, dim1=1, dim2=2)).clamp_min(1e-300)
            step = torch.linalg.solve(F / (d[:, :, None] * d[:, None, :]) + 1e-12 * eye, g / d) / d
            alpha = torch.ones(len(th), 1, dtype=torch.float64, device=device)
            for _ in range(40):                                         # backtracking, per row
                th_new = th - alpha * step
                lz_n, E_n, F_n = stats(th_new)
                obj_n = lz_n - (th_new * mm).sum(1)
                bad = ~(obj_n <= obj + 1e-14 * obj.abs())
                if not bad.any():
                    break
                alpha[bad] /= 2
            th, lz, E, F, obj = th_new, lz_n, E_n, F_n, obj_n
        worst = max(worst, float(((E - mm).abs() / scale).max()))
        th_all[a:a + chunk], F_all[a:a + chunk] = th.cpu(), F.cpu()
    return th_all, F_all, worst


def local_kl(e, F):
    """1/2 e^T F e per node; e (..., n, 4), F (n, 4, 4)."""
    return 0.5 * torch.einsum('...ni,nij,...nj->...n', e, F, e)


# ------------------------------------------------------------------ diagnostics
def acf_stats(e, lmax=1000):
    """Median over the 4 statistics of acf at ACF_LAGS and tau_int = 1 + 2 sum_{L<=lmax} acf(L)
    (lmax capped at len/4) of e (steps, 4) float64 numpy."""
    e = e - e.mean(0)
    v = (e ** 2).mean(0)
    v[v == 0] = np.inf
    lm = max(1, min(lmax, len(e) // 4))
    acf = np.stack([(e[:-L] * e[L:]).mean(0) / v for L in range(1, lm + 1)])
    out = {f'acf{L}': float(np.median(acf[L - 1])) if L <= lm else float('nan') for L in ACF_LAGS}
    out['tau_int_100'] = float(np.median(1 + 2 * acf[:min(100, lm)].sum(0)))
    out['tau_int'] = float(np.median(1 + 2 * acf.sum(0)))
    out['tau_lmax'] = lm
    return out


def longest_run(mask):
    """Indices of the longest stretch of consecutive True in mask."""
    best, cur = (0, 0), None
    for i, m in enumerate(np.append(mask, False)):
        if m and cur is None:
            cur = i
        elif not m and cur is not None:
            if i - cur > best[1] - best[0]:
                best = (cur, i)
            cur = None
    return np.arange(*best)


def noise_diagnostics(theta_nodes, theta_ref, t, S_, sigma, n_walkers, valid):
    """Residual theta_t - theta_ME(t) per window (longest valid stretch): autocorrelation and the
    noise-model z^2 (Cov = 2 M^-1 / (n h sigma^2), white -> E[z^2] = 1)."""
    out = {}
    Mr = tb.ridged(S_['M'], S_['ridge'])
    h = torch.as_tensor(np.asarray(S_['h'], dtype=np.float64))
    for lo, hi in NOISE_WINDOWS:
        s = longest_run((t >= lo) & (t < hi) & valid)
        if len(s) < 20:
            continue
        e = (theta_nodes[s] - theta_ref[s])
        d = acf_stats(e.numpy())
        si = torch.as_tensor(s)
        z2 = n_walkers * h[si] * sigma ** 2 / 2 * torch.einsum('ki,kij,kj->k', e, Mr[si], e) / R
        out[f'[{lo},{hi})'] = dict(steps=int(len(s)), t_from=float(t[s[0]]), t_to=float(t[s[-1]]),
                                   z2_median=float(z2.median()), **d)
    return out


def plot_nodes(t, n_plot):
    """~n_plot node indices: half uniform in t, half log-uniform in 1 - t (the end is where it happens)."""
    n = len(t)
    a = np.searchsorted(t, np.linspace(t[0], t[-1], n_plot // 2))
    b = np.searchsorted(t, 1 - np.logspace(np.log10(1 - t[0]), np.log10(1 - t[-1]), n_plot // 2))
    return np.unique(np.clip(np.concatenate([a, b, [n - 1]]), 0, n - 1))


def summarize(kl, t, wts, valid):
    d = {}
    for lo, hi in WINDOWS:
        s = ((t >= lo) & (t < hi) if hi < 1 else (t >= lo)) & valid
        d[f'[{lo},{hi})'] = float((kl[s] * wts[s]).sum() / wts[s].sum()) if s.any() else float('nan')
    return d


def edge_flag(lam, lams):
    pos = [l_ for l_ in lams if l_ > 0]
    return lam is not None and (lam == max(lams) or (bool(pos) and lam == min(pos)))


# ------------------------------------------------------------------ one run
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_one(args, k):
    torch.manual_seed(k)
    np.random.seed(k)
    th_star = theta_star(args.beta)
    x1 = sample(args.n, args.beta, k).to(args.device)
    t_grid, info = two_phase_schedule(args.nt, args.n_bulk, args.t_switch, args.gap_end)
    sysdir = args.system_dir or args.out / 'systems'
    sysdir.mkdir(parents=True, exist_ok=True)
    sys_path = sysdir / f'bimodal_beta{args.beta:.2f}_run{k:03d}.pt'
    solver = SDE(x1.clone(), args.n, args.n, t_grid.clone(), args.sigma, {'poly': Poly()}, args.n,
                 device=args.device, regularization=args.regularization, interpolant='Cos',
                 potentials_save_dir=sysdir / 'unused_potentials_dir',   # required by SDE, never written
                 solve_float64=True)
    t0 = time.time()
    log(f'beta {args.beta:.2f} run {k}: SDE start ({args.nt} steps, n = {args.n}, device {args.device}); '
        f'progress bar below, one update a minute')
    out = solver.forward_regularised(lam=0.0, reg_solver='thomas', reg_ridge=args.regularization,
                                     reg_system_path=sys_path, solve_reg=False, reg_mode='moment')
    t_sde = time.time() - t0
    log(f'SDE done in {t_sde / 60:.1f} min')
    theta_t = out[4].double().cpu()                                  # (nt, 4), row j at t_grid[j+1]

    system = torch.load(sys_path, map_location='cpu', weights_only=False)
    t = system['t'].double().numpy()
    tf = t_grid.double().numpy()
    idx = np.clip(np.searchsorted(tf[1:], t), 0, len(tf) - 2)
    assert np.abs(tf[1:][idx] - t).max() < 1e-12
    theta_nodes = theta_t[idx]
    wts = trapezoid_node_weights(t)

    t1 = time.time()
    log(f'truth: max-ent fits at {len(t)} nodes')
    m_t = interp_moments(solver.x_0, x1, t)
    theta_me, F_me, me_res = maxent_fit(m_t, device=args.device)
    t_truth = time.time() - t1
    log(f'truth done in {t_truth:.0f} s')
    kl_floor = float(kl_exact(th_star, theta_me[-1:])[0])           # max-likelihood fit of x1 vs theta*
    valid = (theta_me[:, 3] < 0).numpy()                             # theta_ME exists on R

    res = dict(run=k, beta=args.beta, n_nodes=len(t), grid=info, t_sde_s=t_sde, t_truth_s=t_truth,
               maxent_residual=me_res, maxent_theta4_nonneg=int((~valid).sum()),
               maxent_invalid_t=[float(t[~valid].min()), float(t[~valid].max())] if (~valid).any() else None,
               theta_star=th_star, theta_mle=theta_me[-1], kl_floor=kl_floor, cr_bound=R / (2 * args.n),
               rules={}, cv={}, kl_end_true={}, kl_end_sample={}, theta_end={})
    lams = sorted(set([0.0] + list(args.lams)))
    res['lams'] = lams
    pidx = plot_nodes(t, args.n_plot)
    res['plot'] = dict(idx=torch.as_tensor(pidx), t=torch.as_tensor(t[pidx]), theta_t=theta_nodes[pidx].float(),
                       theta_me=theta_me[pidx].float(), valid=torch.as_tensor(valid[pidx]),
                       Theta={}, kl_local={})
    kl_local_by, kl_true_by = {}, {}
    for sched in SCHEDULES:
        a_ = SimpleNamespace(schedule=sched, ridge=args.regularization, dim=None, n_walkers=None, sigma=None,
                             mask='auto', results_root=None, live_floor=0.0)
        inp = build_inputs(system, sys_path.stem, a_, 'moment')
        S_ = tb.stack_inputs(inp)
        t2 = time.time()
        log(f'{sched}: lam path + block CV ({len(lams)} lams, blocks {args.blocks})')
        path = tb.lam_path_batched(S_, lams, veto=args.veto)
        Theta = path['Theta']                                        # (L, n, 4) float64
        if sched == 'uniform':
            res['theta0_vs_theta_t_maxrel'] = float(((Theta[0] - theta_nodes).norm(dim=1) /
                                                     theta_nodes.norm(dim=1).clamp_min(1e-300)).max())
            res['noise'] = noise_diagnostics(theta_nodes, theta_me, t, S_, args.sigma, args.n, valid)
        blocks = {}
        for B in args.blocks:
            loss, sel = tb.block_cv_batched(S_, lams, B, path['veto'])
            blocks[B] = dict(loss=loss.tolist(), selected=sel)
        kl_loc = local_kl(Theta - theta_me[None], F_me)              # (L, n)
        kl_true = kl_exact(th_star, Theta[:, -1])                    # (L,)
        kl_samp = kl_exact(theta_me[-1], Theta[:, -1])
        for i, lam in enumerate(lams):
            key = f'{sched}|{lam:g}'
            kl_local_by[(sched, lam)], kl_true_by[(sched, lam)] = kl_loc[i], float(kl_true[i])
            res['kl_end_true'][key], res['kl_end_sample'][key] = float(kl_true[i]), float(kl_samp[i])
            res['theta_end'][key] = Theta[i, -1]
            res['plot']['Theta'][key] = Theta[i][torch.as_tensor(pidx)].float()
            res['plot']['kl_local'][key] = kl_loc[i][torch.as_tensor(pidx)].float()
        res['cv'][sched] = dict(eo_loss=path['cv_loss'].tolist(), eo_selected=path['lam_selected'],
                                blocks=blocks, veto=[bool(v) for v in path['veto']],
                                amp_ratio=path['amp_ratio'].tolist(),
                                chi2=None if path['chi2'] is None else path['chi2'].tolist(),
                                residual=path['residual'].tolist(), solve_s=time.time() - t2)
        log(f"{sched} done in {res['cv'][sched]['solve_s']:.0f} s")
        mids = {lam: summarize(kl_local_by[(sched, lam)].numpy(), t, wts, valid)['[0.05,0.95)'] for lam in lams}
        ends = {lam: kl_true_by[(sched, lam)] for lam in lams}
        choices = {f'{sched}_cv_eo': path['lam_selected'],
                   **{f'{sched}_cv_block{B}': blocks[B]['selected'] for B in args.blocks},
                   f'{sched}_oracle_end': min(ends, key=ends.get), f'{sched}_oracle_mid': min(mids, key=mids.get)}
        if sched == 'guth':
            choices['guth_lam1'] = 1.0
        for name, lam in choices.items():
            if lam is None or (sched, lam) not in kl_true_by:
                res['rules'][name] = dict(lam=lam)
                continue
            res['rules'][name] = dict(lam=lam, sched=sched, edge=edge_flag(lam, lams),
                                      end_true=kl_true_by[(sched, lam)],
                                      end_sample=res['kl_end_sample'][f'{sched}|{lam:g}'],
                                      **summarize(kl_local_by[(sched, lam)].numpy(), t, wts, valid))
    kl0 = local_kl(theta_nodes - theta_me, F_me)
    res['rules']['lam0'] = dict(lam=0.0, sched=None, edge=False,
                                end_true=float(kl_exact(th_star, theta_nodes[-1:])[0]),
                                end_sample=float(kl_exact(theta_me[-1], theta_nodes[-1:])[0]),
                                **summarize(kl0.numpy(), t, wts, valid))
    res['theta_end']['theta_t'] = theta_nodes[-1]
    res['plot']['kl_local']['theta_t'] = kl0[torch.as_tensor(pidx)].float()
    res['t_total_s'] = time.time() - t0
    dest = args.out / f'beta_{args.beta:.2f}' / f'run_{k:03d}.pt'
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.tmp')
    torch.save(res, tmp)
    tmp.replace(dest)
    if not args.keep_system:
        sys_path.unlink(missing_ok=True)
    return res


# ------------------------------------------------------------------ report
def rule_order(blocks):
    out = ['lam0', 'guth_lam1']
    for s in SCHEDULES:
        out += [f'{s}_cv_eo'] + [f'{s}_cv_block{B}' for B in blocks]
    for s in SCHEDULES:
        out += [f'{s}_oracle_end', f'{s}_oracle_mid']
    return out


def print_run(res, blocks):
    solves = ', '.join(f"{s} solves {res['cv'][s]['solve_s']:.0f} s" for s in SCHEDULES)
    print(f"\n=== beta {res['beta']:.2f} run {res['run']}: {res['n_nodes']} nodes, total {res['t_total_s']:.0f} s "
          f"(SDE {res['t_sde_s']:.0f} s, truth {res['t_truth_s']:.0f} s, {solves}); max rel |Theta(0) - theta_t| "
          f"= {res['theta0_vs_theta_t_maxrel']:.1e}; max-ent residual {res['maxent_residual']:.1e}, "
          f"theta_ME invalid (theta_4 >= 0) at {res['maxent_theta4_nonneg']} nodes, t in {res['maxent_invalid_t']}")
    print(f"floor KL(theta* || max-likelihood fit of x1) = {res['kl_floor']:.3g}   (r/2n = {res['cr_bound']:.3g})")
    print('noise of theta_t (residual vs theta_ME(t); tau_int = 1 for independent steps):')
    for w, d in res['noise'].items():
        print(f"  t {w:>14}  steps {d['steps']:6d} [{d['t_from']:.4f}, {d['t_to']:.6f}]  acf(1) {d['acf1']: .3f}  acf(10) {d['acf10']: .3f}  "
              f"acf(100) {d['acf100']: .3f}  acf(1000) {d['acf1000']: .3f}  tau_int(<=100) {d['tau_int_100']:7.1f}  "
              f"tau_int(<={d['tau_lmax']}) {d['tau_int']:7.1f}  E[z^2] {d['z2_median']:.3g}")
    print(f"{'rule':>22} {'lam':>7} | {'KL end vs theta*':>16} {'vs MLE':>9} | {'local KL [0.05,0.95)':>20} "
          f"{'[0.95,0.99)':>11} {'[0.99,0.999)':>12} {'[0.999,1)':>10}")
    for name in rule_order(blocks):
        d = res['rules'].get(name)
        if d is None or 'end_true' not in d:
            print(f"{name:>22} {'none':>7}")
            continue
        print(f"{name:>22} {d['lam']:7.0e} | {d['end_true']:16.4g} {d['end_sample']:9.3g} | {d['[0.05,0.95)']:20.4g} "
              f"{d['[0.95,0.99)']:11.4g} {d['[0.99,0.999)']:12.4g} {d['[0.999,1.0)']:10.4g}"
              + ('  (grid edge)' if d.get('edge') else ''))


def combine(args):
    for bdir in sorted(args.out.glob('beta_*')):
        files = sorted(bdir.glob('run_*.pt'))
        if not files:
            continue
        Rs = [torch.load(f, map_location='cpu', weights_only=False) for f in files]
        floor = np.array([r['kl_floor'] for r in Rs])
        print(f"\n######## beta {Rs[0]['beta']:.2f}: {len(Rs)} runs; KL to theta* at the last node "
              f"(floor = max-likelihood fit of each x1: mean {floor.mean():.3g}; r/2n = {Rs[0]['cr_bound']:.3g})")
        print(f"{'rule':>22} | {'lam chosen (count)':>34} | {'KL end: mean':>12} {'median':>9} {'[q25, q75]':>21} "
              f"{'/floor':>7} {'inf':>4} | {'local KL [0.05,0.95)':>20}")
        for name in rule_order(args.blocks):
            rows = [r['rules'].get(name, {}) for r in Rs]
            rows = [d for d in rows if 'end_true' in d]
            if not rows:
                print(f"{name:>22} | {'none':>34}")
                continue
            ends = np.array([d['end_true'] for d in rows])
            fin = ends[np.isfinite(ends)]
            lam_counts = {}
            for d in rows:
                lam_counts[d['lam']] = lam_counts.get(d['lam'], 0) + 1
            lamtxt = ' '.join(f'{l_:.0e}x{c}' for l_, c in sorted(lam_counts.items()))
            q = np.percentile(fin, [25, 75]) if len(fin) else [np.nan, np.nan]
            print(f"{name:>22} | {lamtxt:>34} | {fin.mean() if len(fin) else np.nan:12.4g} "
                  f"{np.median(ends):9.4g} [{q[0]:9.3g}, {q[1]:9.3g}] "
                  f"{(fin.mean() / floor.mean()) if len(fin) else np.nan:7.2f} {int((~np.isfinite(ends)).sum()):4d} | "
                  f"{np.median([d['[0.05,0.95)'] for d in rows]):20.4g}")
        print('noise of theta_t, median over runs (tau_int(<=100) | tau_int(<=lmax) | acf(1) | E[z^2]):')
        for w in Rs[0]['noise']:
            g_ = lambda key: np.median([r['noise'][w][key] for r in Rs if w in r['noise']])
            print(f"  t {w:>14}  {g_('tau_int_100'):7.1f} | {g_('tau_int'):7.1f} | {g_('acf1'):.3f} | {g_('z2_median'):.3g}")


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    # combine() pools every run of a beta folder: all runs in OUT must share the settings except beta/runs
    run_only = {'runs', 'beta', 'out', 'combine_only', 'no_combine', 'threads', 'keep_system', 'system_dir',
                'device', 'n_plot'}
    setting = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in run_only}
    cfg_path = args.out / 'args.json'
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        diff = {k: (old.get(k), v) for k, v in setting.items() if old.get(k) != v}
        if diff and not args.combine_only:
            sys.exit(f'{args.out} holds runs made with other settings {diff}: use a new --out')
    elif not args.combine_only:
        cfg_path.write_text(json.dumps(setting, indent=1))
    if args.combine_only and cfg_path.exists():
        args.blocks = json.loads(cfg_path.read_text())['blocks']       # the rules the runs were made with
    if not args.combine_only:
        for k in args.runs:
            done = args.out / f'beta_{args.beta:.2f}' / f'run_{k:03d}.pt'
            if done.exists():                                        # resubmit-safe
                print(f'beta {args.beta:.2f} run {k}: {done} exists, skipped')
                continue
            print_run(run_one(args, k), args.blocks)
    if not args.no_combine:
        combine(args)


if __name__ == '__main__':
    main()
