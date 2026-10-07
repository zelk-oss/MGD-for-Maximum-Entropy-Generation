"""Ridge vs early-stopped Krylov solves of the per-step MGD systems: centred Gaussian, d = 8.

phi_ij = x_i x_j (i <= j), r = 36 (Quadratic of lam_selection_criterion/gauss_d8_lamtest.py, no
linear terms). Target C_1 = Q diag(logspace(-1, 0, d)) Q^T, Q a random rotation (--data_seed).
The walker law stays N(0, C_t) with C_t = cos^2 a I + sin^2 a C_1, so the truth is known at every
step (MGD App. E): code-sign Theta_code = -C_t^-1 (theta_code . phi = -1/2 x^T C_t^-1 x), entropy
change 1/2 log det C_1 - 1/2 log det C_0.

Two truths, each for what it tests (2026-10-06):
  empirical S_t (second moments of the frozen interpolant samples, the corrector's actual target):
      does the stopping rule do what it claims? -- whitened moment residual vs its threshold,
      calibration of the Ito noise model;
  population C_t: estimation quality -- theta error vs C_t^-1, final KL, entropy integral.

Methods (METHODS): ridge lam in {1e-7, 1e-4, 1e-2}; direct (ridge lam = 0); direct theta + tol_minres
eta; ito_cg theta + tol_minres eta (full W); tol_minres theta and eta with W diag / full / shrink.
Sweep: N in {1e3, 1e4} (M = N interpolant samples, M_eff = M), sigma^2 in {1, 10}, nt = 1000 uniform
steps (h sigma^2 in {1e-3, 1e-2}), 5 seeds. sigma is a legitimate lever only here: for Gaussian p_0
and quadratic potentials MGD's solution does not depend on sigma (MGD sec. 3, App. E).

Usage, from the repo root:
  smoke (local, ~1 min):  python krylov_stopping/gauss_d8_krylov.py --out <dir> --smoke
  one array task (JZ):    python krylov_stopping/gauss_d8_krylov.py --out DIR --task $SLURM_ARRAY_TASK_ID --n_tasks 10 --no_combine
  combine (login node):   python krylov_stopping/gauss_d8_krylov.py --out DIR --combine_only
Cells (# %%) run in Jupyter as well: set ARGS by hand in the first cell instead of the CLI.
"""
# %% imports and configuration
import argparse
import contextlib
import io
import itertools
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().parent
for p in (ROOT, ROOT / 'codes', ROOT / 'data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from codes.sde_krylov import SDEKrylov                                          # noqa: E402
from codes import krylov as K                                                  # noqa: E402
from lam_selection_criterion.gauss_d8_lamtest import (Quadratic, interp_second_moments,  # noqa: E402
                                                      theta_of_cov)

class Distorted:
    """Near-copies of Quadratic (--eps > 0): psi_ij = s(x_i) s(x_j), s(u) = u + eps u^3. Their gradients
    are within O(eps) of grad phi_ij, so the Gram matrix of (phi, psi) is ill-conditioned (~1/eps^2), as
    with turbulence's near-duplicate statistics. The walker law stays N(0, C_t): theta_psi = 0 (population)."""

    def __init__(self, d, eps):
        self.d, self.eps = d, eps
        self.iu = torch.triu_indices(d, d)
        self.num_coefficients = self.iu.shape[1]

    def forward(self, x):
        z = x.reshape(x.shape[0], self.d)
        S = z + self.eps * z ** 3
        return S[:, self.iu[0]] * S[:, self.iu[1]]

    def grad(self, x, v=None):
        z = x.reshape(x.shape[0], self.d)
        S, Sp = z + self.eps * z ** 3, 1 + 3 * self.eps * z ** 2
        i, j = self.iu[0].to(z.device), self.iu[1].to(z.device)
        if v is None:                                      # (B, r, 1, d)
            B, r = z.shape[0], self.num_coefficients
            g = torch.zeros(B, r, self.d, dtype=z.dtype, device=z.device)
            ar = torch.arange(r, device=z.device)
            g[:, ar, i] += Sp[:, i] * S[:, j]
            g[:, ar, j] += S[:, i] * Sp[:, j]
            return g.reshape(B, r, 1, self.d)
        W = torch.zeros(self.d, self.d, dtype=z.dtype, device=z.device)
        W[i, j] = v.reshape(-1).to(z.dtype)
        W = W + W.T
        return (Sp * (S @ W)).reshape(x.shape)


METHODS = {
    'ridge1e-7':      dict(regularization=1e-7),
    'ridge1e-4':      dict(regularization=1e-4),
    'ridge1e-2':      dict(regularization=1e-2),
    # dense ridge sweep for the near-duplicate test (ito_cg vs the BEST ridge)
    'ridge1e-3':      dict(regularization=1e-3),
    'ridge3e-3':      dict(regularization=3e-3),
    'ridge3e-2':      dict(regularization=3e-2),
    'ridge1e-1':      dict(regularization=1e-1),
    'ridge3e-1':      dict(regularization=3e-1),
    'ridge1':         dict(regularization=1.0),
    'direct':         dict(regularization=0.0),
    'direct+eta_tol': dict(regularization=0.0, eta_solver='tol_minres', tol_weight='full'),
    'ito_cg':         dict(regularization=0.0, theta_solver='ito_cg', eta_solver='tol_minres', tol_weight='full'),
    'tol_diag':       dict(regularization=0.0, theta_solver='tol_minres', eta_solver='tol_minres', tol_weight='diag'),
    'tol_full':       dict(regularization=0.0, theta_solver='tol_minres', eta_solver='tol_minres', tol_weight='full'),
    'tol_shrink':     dict(regularization=0.0, theta_solver='tol_minres', eta_solver='tol_minres', tol_weight='shrink'),
}
SWEEP = ('ridge1e-3', 'ridge3e-3', 'ridge3e-2', 'ridge1e-1', 'ridge3e-1', 'ridge1')   # opt-in via --methods
T_AVG = 0.05         # time averages over t >= T_AVG: skips the first-step transient (walkers start
                      # from their own N(0, I) draws, not from the target's empirical S_0)
T_CALIB = 0.1         # calibration statistics over t >= T_CALIB


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--d', type=int, default=8)
    p.add_argument('--data_seed', type=int, default=0, help='C_1 (eigenvalues fixed, rotation drawn)')
    p.add_argument('--N', type=int, nargs='+', default=[1000, 10000], help='walkers = interpolant samples')
    p.add_argument('--sigma2', type=float, nargs='+', default=[1.0, 10.0])
    p.add_argument('--nt', type=int, default=1000)
    p.add_argument('--eps', type=float, nargs='+', default=[0.0],
                   help='perturbation strength delta (> 0: add the near-copies Distorted(delta) of the 36 statistics, r = 72); '
                        'named delta in the write-ups to avoid a clash with the stopping threshold eps^2')
    p.add_argument('--seeds', type=int, nargs='+', default=[900, 901, 902, 903, 904])
    p.add_argument('--methods', nargs='+', default=[m for m in METHODS if m not in SWEEP], choices=list(METHODS))
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--kappa', type=float, default=2.0)
    p.add_argument('--var_floor', type=float, default=1e-6)
    p.add_argument('--task', type=int, default=None,
                   help='array task: runs every method for the (N, sigma2, seed) points task, task + n_tasks, ... '
                        'of the grid ordered N, sigma2, seed')
    p.add_argument('--n_tasks', type=int, default=None,
                   help='number of array tasks (default: one task per grid point). qos_gpu_h100-dev allows '
                        'at most 10 submitted jobs per user')
    p.add_argument('--smoke', action='store_true',
                   help='local smoke test: N 200, sigma2 1, nt 100, one seed, every method')
    p.add_argument('--combine_only', action='store_true')
    p.add_argument('--no_combine', action='store_true')
    p.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or None)
    a = p.parse_args(argv)
    if a.smoke:
        a.N, a.sigma2, a.nt, a.seeds = [200], [1.0], 100, a.seeds[:1]
    return a


def cfg_name(N, sigma2, nt, eps=0.0):
    return f'N{N}_s2{sigma2:g}_nt{nt}' + (f'_eps{eps:g}' if eps else '')


# %% truth and helpers
def target_cov(d, data_seed):
    g = np.random.default_rng(data_seed)
    Q, R = np.linalg.qr(g.standard_normal((d, d)))
    Q = Q * np.sign(np.diag(R))                                   # Haar rotation
    return (Q * np.logspace(-1, 0, d)) @ Q.T


def pop_cov(C1, t):
    a = np.pi * np.asarray(t, dtype=np.float64) / 2
    c2, s2 = np.cos(a)[:, None, None] ** 2, np.sin(a)[:, None, None] ** 2
    return torch.as_tensor(c2 * np.eye(len(C1)) + s2 * C1)


def sym_from_phi(v, iu, d):
    """(T, r) moments E[x_i x_j] (i <= j) -> (T, d, d) symmetric matrices."""
    M = torch.zeros(v.shape[0], d, d, dtype=torch.float64)
    M[:, iu[0], iu[1]] = v.double()
    return M + M.transpose(1, 2) - torch.diag_embed(M.diagonal(dim1=1, dim2=2))


def Theta_mgd(theta_code, iu, d):
    """MGD-sign Theta (T, d, d) with theta . grad phi(x) = Theta x: Theta_ii = 2 theta_ii,
    Theta_ij = theta_ij, theta_MGD = -theta_code."""
    M = torch.zeros(theta_code.shape[0], d, d, dtype=torch.float64)
    M[:, iu[0], iu[1]] = theta_code.double()
    return -(M + M.transpose(1, 2))


def rel_fro(A, B):
    return ((A - B).flatten(1).norm(dim=1) / B.flatten(1).norm(dim=1)).numpy()


def kl_gauss(S_hat, S):
    """KL(N(0, S_hat) || N(0, S))."""
    P = torch.linalg.solve(S, S_hat)
    return 0.5 * float(P.trace() - len(S) - torch.logdet(P))


# %% one run
def run_one(args, C1, N, sigma2, seed, method, eps=0.0):
    d = args.d
    g = torch.Generator().manual_seed(seed)
    x1 = (torch.randn(N, d, generator=g, dtype=torch.float64) @ torch.linalg.cholesky(
        torch.as_tensor(C1)).T).float().reshape(N, 1, d)
    t = np.linspace(0.0, 1.0, args.nt + 1)
    h = np.diff(t)
    pot = Quadratic(d)
    iu = pot.iu
    pots = {'quad': pot, 'dup': Distorted(d, eps)} if eps else {'quad': pot}
    torch.manual_seed(seed + 10_000)        # x_0, walkers, noise: same draws for every method
    with contextlib.redirect_stdout(io.StringIO()):
        sde = SDEKrylov(x1, N, N, t, math.sqrt(sigma2), pots, N, device='cpu',
                        interpolant='Cos', potentials_save_dir=args.out / 'unused_potentials_dir',
                        solve_float64=True, alpha=args.alpha, kappa=args.kappa,
                        var_floor=args.var_floor, keep_step_mats=True, **METHODS[method])
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _, phi_I, phi_x, eta, theta, dH = sde.forward()
    wall = time.perf_counter() - t0
    nt = args.nt
    # forward(): row 0 of phi_I / phi_x is the t = 0 seed, the last row of every list repeats the last step
    phi_I, phi_x, theta, dH = phi_I[1:nt + 1], phi_x[1:nt + 1], theta[:nt].double(), dH[:nt].double()
    L = sde.krylov_log()
    tn = t[1:]                                                  # step k targets t_{k+1}
    S_emp = interp_second_moments(sde.x_0, sde.x_1, tn)          # (nt, d, d)
    S_pop = pop_cov(C1, tn)
    r0 = pot.num_coefficients
    C_x = sym_from_phi(phi_x[:, :r0], iu, d)                     # walker second moments after the corrector
    Th = Theta_mgd(theta[:, :r0], iu, d)                         # phi block (= all of theta if eps = 0)
    P_pop = torch.linalg.inv(S_pop)
    if eps:                                                      # copies: theta_psi = 0, sum -> C^-1 as eps -> 0
        Th_psi = Theta_mgd(theta[:, r0:], iu, d)
        extra = dict(psi_norm_pop=(Th_psi.flatten(1).norm(dim=1) / P_pop.flatten(1).norm(dim=1)).numpy(),
                     theta_sum_err_pop=rel_fro(Th + Th_psi, P_pop))
    else:
        extra = {}

    # whitened moment residual against the empirical target, both weights of the tol rule
    res = (phi_x - phi_I).double()
    w_full, w_diag, thr_diag = np.empty(nt), np.empty(nt), np.empty(nt)
    for k in range(nt):
        Sig = L['Sigma_tol'][k]
        Wf, _ = K.tol_weight(Sig, 'full', args.var_floor, args.alpha, args.kappa)
        Wd, thr_diag[k] = K.tol_weight(Sig, 'diag', args.var_floor, args.alpha, args.kappa)
        w_full[k] = float((Wf @ res[k]).pow(2).sum())
        w_diag[k] = float((Wd * res[k]).pow(2).sum())

    # calibration: Ito-whitened theta error against the empirical truth, z ~ chi2_r if the noise
    # model holds (exactly so for the direct solve: theta - theta* = G'^-1 (b - G' theta*))
    # (eps > 0: theta*_psi = 0 is the POPULATION value; the empirical one is unknown, so z is approximate)
    th_star = theta_of_cov(S_emp, iu)
    e = theta - torch.cat([th_star, torch.zeros_like(th_star)], 1) if eps else theta - th_star
    G = L['G_theta']
    z = (N * h * sigma2 / 2) * torch.einsum('kr,krs,ks->k', e, G, e).numpy()
    tr_inv = torch.linalg.inv(S_emp).diagonal(dim1=1, dim2=2).sum(1).numpy()

    col = lambda name, c: L[name][:, L[f'{name}_cols'].index(c)].numpy()
    r = sde.num_potentials
    S0 = interp_second_moments(sde.x_0, sde.x_1, t[:1])[0]
    return dict(
        **extra, eps=eps,
        method=method, N=N, sigma2=sigma2, seed=seed, nt=nt, t=tn, h=h, r=r,
        mom_err_pop=rel_fro(C_x, S_pop), mom_err_emp=rel_fro(C_x, S_emp),
        w_full=w_full, w_diag=w_diag, thr_full=K.chi2_threshold(r, args.alpha), thr_diag=thr_diag,
        theta_err_pop=rel_fro(Th, P_pop), theta_err_emp=rel_fro(Th, torch.linalg.inv(S_emp)),
        j_theta=col('theta', 'j'), j_eta=col('eta', 'j'),
        r_live_theta=col('theta', 'r_live'), r_live_eta=col('eta', 'r_live'),
        stat_theta=col('theta', 'stat'), thr_theta=col('theta', 'thr'),
        stat_eta=col('eta', 'stat'), thr_eta=col('eta', 'thr'),
        ito0=col('theta', 'ito_stat0'), ito0_pred=r + N * h * sigma2 * tr_inv / 2,
        tol0=col('theta', 'tol_stat0'), z_calib=z,
        dH=dH.numpy(), S_int=float((h * dH.numpy()).sum()),
        S_exact_pop=0.5 * float(torch.logdet(torch.as_tensor(C1))),
        S_exact_emp=0.5 * float(torch.logdet(S_emp[-1]) - torch.logdet(S0)),
        KL_pop=kl_gauss(C_x[-1], S_pop[-1]), KL_emp=kl_gauss(C_x[-1], S_emp[-1]),
        wall_per_step=wall / nt, t_theta=col('theta', 't_call'), t_eta=col('eta', 't_call'),
    )


def run_task(args, N, sigma2, eps, seed):
    C1 = target_cov(args.d, args.data_seed)
    out = args.out / 'runs'
    out.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        f = out / f'{cfg_name(N, sigma2, args.nt, eps)}_{method}_seed{seed}.pt'
        if f.exists():
            print(f'skip (exists): {f.name}')
            continue
        t0 = time.time()
        res = run_one(args, C1, N, sigma2, seed, method, eps)
        torch.save(res, f)
        print(f'{f.name}: {time.time() - t0:.1f} s, KL_pop {res["KL_pop"]:.2e}, '
              f'j_theta mean {np.nanmean(np.where(res["j_theta"] >= 0, res["j_theta"], np.nan)):.2f}')


# %% combine: tables
def load_runs(out):
    R = [torch.load(f, weights_only=False) for f in sorted((out / 'runs').glob('*.pt'))]
    by = {}
    for r in R:
        by.setdefault((r['N'], r['sigma2'], r['nt'], r.get('eps', 0.0)), {}).setdefault(r['method'], []).append(r)
    return by


def seed_mean(runs, fn):
    v = np.array([fn(r) for r in runs], dtype=float)
    if np.isnan(v).all():                              # e.g. iteration counts of a ridge row
        return np.nan, np.nan
    return np.nanmean(v), (np.nanstd(v) if len(v) > 1 else np.nan)


def tables(by, methods):
    tavg = lambda key: (lambda r: np.mean(r[key][r['t'] >= T_AVG]))
    jstat = lambda key, f: (lambda r: f(r[key]) if (r[key] >= 0).any() else np.nan)
    rows = [
        ('1. moment error vs population, rel. Frobenius, time avg', tavg('mom_err_pop')),
        ('1. moment error vs population, final', lambda r: r['mom_err_pop'][-1]),
        ('1. moment error vs empirical target, time avg', tavg('mom_err_emp')),
        ('1. whitened residual vs empirical (full W) / chi2 threshold, time avg',
         lambda r: np.mean(r['w_full'][r['t'] >= T_AVG]) / r['thr_full']),
        ('1. fraction of steps with whitened residual (full W) > chi2 threshold',
         lambda r: np.mean(r['w_full'][r['t'] >= T_AVG] > r['thr_full'])),
        ('1. whitened residual vs empirical (diag W) / diag threshold, time avg',
         lambda r: np.mean((r['w_diag'] / r['thr_diag'])[r['t'] >= T_AVG])),
        ('2. theta error vs population C_t^-1, rel. Frobenius, time avg', tavg('theta_err_pop')),
        ('2. theta error vs population, final', lambda r: r['theta_err_pop'][-1]),
        ('2. theta error vs empirical S_t^-1, time avg', tavg('theta_err_emp')),
        ('2b. (eps > 0) copies block ||Theta_psi||_F / ||C_t^-1||_F (truth 0), time avg',
         lambda r: np.mean(r['psi_norm_pop'][r['t'] >= T_AVG]) if 'psi_norm_pop' in r else np.nan),
        ('2b. (eps > 0) copies block, final', lambda r: r['psi_norm_pop'][-1] if 'psi_norm_pop' in r else np.nan),
        ('2b. (eps > 0) sum Theta_phi + Theta_psi vs C_t^-1, rel. Frobenius, time avg',
         lambda r: np.mean(r['theta_sum_err_pop'][r['t'] >= T_AVG]) if 'theta_sum_err_pop' in r else np.nan),
        ('3. j theta, mean (Krylov only)', jstat('j_theta', np.mean)),
        ('3. j theta = 0, fraction of steps', jstat('j_theta', lambda j: np.mean(j == 0))),
        ('3. j eta, mean (Krylov only)', jstat('j_eta', np.mean)),
        ('3. j eta = 0, fraction of steps', jstat('j_eta', lambda j: np.mean(j == 0))),
        ('3. j theta hit max_iter (= r_live) without meeting the rule, fraction',
         lambda r: np.mean((r['j_theta'] == r['r_live_theta']) & (r['stat_theta'] > r['thr_theta']))
         if (r['j_theta'] >= 0).any() else np.nan),
        ('4. entropy integral - exact (population)', lambda r: r['S_int'] - r['S_exact_pop']),
        ('4. entropy integral - exact (empirical)', lambda r: r['S_int'] - r['S_exact_emp']),
        ('5. KL(N(0, C_x(1)) || N(0, C_1)) population', lambda r: r['KL_pop']),
        ('5. KL(N(0, C_x(1)) || N(0, S_1)) empirical', lambda r: r['KL_emp']),
        ('6. wall time per step, ms (whole forward step incl. diagnostics)', lambda r: 1e3 * r['wall_per_step']),
        ('6. compute_theta + compute_eta call, ms (Gram, moments, solve)',
         lambda r: 1e3 * (np.mean(r['t_theta']) + np.mean(r['t_eta']))),
    ]
    cfgs = sorted(by)
    head = '| method | ' + ' | '.join(
        f'N={N:g} s2={s2:g} (Nhs2={N * s2 / nt:g})' + (f' δ={e:g}' if e else '') for N, s2, nt, e in cfgs) + ' |'
    lines = []
    for title, fn in rows:
        lines += [f'\n### {title}\n', head, '|' + '---|' * (len(cfgs) + 1)]
        for m in methods:
            cells = []
            for c in cfgs:
                runs = by[c].get(m, [])
                if not runs:
                    cells.append('-')
                    continue
                mu, sd = seed_mean(runs, fn)
                cells.append(f'{mu:.3g}' + (f' ± {sd:.2g}' if np.isfinite(sd) else ''))
            lines.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    # calibration (direct solve)
    lines += [f'\n### Calibration (direct solve, t >= {T_CALIB}): z = (N h s2 / 2) e^T G\' e, '
              'e = theta - theta*(S_t); chi2_r: mean r, var 2r\n',
              '| config | r | mean z | var z | 2r | mean ito_stat0 (j=0) | predicted r + N h s2 tr(S^-1)/2 |',
              '|---|---|---|---|---|---|---|']
    for c in cfgs:
        runs = by[c].get('direct', [])
        if not runs:
            continue
        zs = np.concatenate([r['z_calib'][r['t'] >= T_CALIB] for r in runs])
        i0 = np.concatenate([r['ito0'][r['t'] >= T_CALIB] for r in runs])
        pr = np.concatenate([r['ito0_pred'][r['t'] >= T_CALIB] for r in runs])
        r_ = runs[0]['r']
        lines.append(f'| N={c[0]:g} s2={c[1]:g}' + (f' eps={c[3]:g} (approx. truth)' if c[3] else '') + f' | {r_} | {zs.mean():.2f} | {zs.var():.1f} | {2 * r_} | '
                     f'{i0.mean():.1f} | {pr.mean():.1f} |')
    return '\n'.join(lines)


# %% combine: plots (color = method family, line style = variant)
STYLE = {
    'ridge1e-7': ('#2a78d6', ':'), 'ridge1e-4': ('#2a78d6', '--'), 'ridge1e-2': ('#2a78d6', '-'),
    'direct': ('#0b0b0b', '-'), 'direct+eta_tol': ('#0b0b0b', '--'),
    'ito_cg': ('#eb6834', '-'),
    'tol_diag': ('#1baf7a', '-'), 'tol_full': ('#1baf7a', '--'), 'tol_shrink': ('#1baf7a', ':'),
    **{m: ('#8fb8ea', ls) for m, ls in zip(SWEEP, (':', '--', '-.', '-', (0, (1, 3)), (0, (5, 2, 1, 2))))},
}


def smooth(x, w):
    if w <= 1 or len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return np.concatenate([np.full(w // 2, np.nan), (c[w:] - c[:-w]) / w, np.full(w - 1 - w // 2, np.nan)])


def plot_over_time(by, methods, key, ylabel, out_png, krylov_only=False, logy=False, w=1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfgs = sorted(by)
    fig, axes = plt.subplots(1, len(cfgs), figsize=(4.2 * len(cfgs), 3.4), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], cfgs):
        for m in methods:
            runs = by[c].get(m, [])
            if not runs or key not in runs[0] or (krylov_only and not (runs[0][key] >= 0).any()):
                continue
            y = np.mean([r[key] for r in runs], axis=0)
            color, ls = STYLE[m]
            ax.plot(runs[0]['t'], smooth(y, w), color=color, ls=ls, lw=1.5, label=m)
        ax.set_title(f'N={c[0]:g}, σ²={c[1]:g}, Nhσ²={c[0] * c[1] / c[2]:g}' + (f', δ={c[3]:g}' if c[3] else ''),
                     fontsize=10)
        ax.set_xlabel('t')
        ax.grid(alpha=0.25, lw=0.5)
        if logy:
            ax.set_yscale('log')
    axes[0, 0].set_ylabel(ylabel)
    h, l = axes[0, 0].get_legend_handles_labels()
    for ax in axes[0][1:]:                                         # union of labels over panels
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.tight_layout()
    fig.legend(h, l, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)


def combine(args):
    by = load_runs(args.out)
    if not by:
        print('no runs found')
        return
    methods = [m for m in METHODS if any(m in v for v in by.values())]
    txt = tables(by, methods)
    print(txt)
    (args.out / 'results.md').write_text(f'# gauss_d8_krylov results\n\nseed mean ± std; time averages over '
                                         f't >= {T_AVG}.\n{txt}\n')
    figs = args.out / 'figures'
    figs.mkdir(exist_ok=True)
    w = max(1, args.nt // 50)
    plot_over_time(by, methods, 'j_theta', f'j_θ (seed mean, moving avg {w} steps)', figs / 'j_theta.png',
                   krylov_only=True, w=w)
    plot_over_time(by, methods, 'j_eta', f'j_η (seed mean, moving avg {w} steps)', figs / 'j_eta.png',
                   krylov_only=True, w=w)
    plot_over_time(by, methods, 'theta_err_pop', '‖Θ̂ − C_t⁻¹‖_F / ‖C_t⁻¹‖_F', figs / 'theta_err_pop.png',
                   logy=True)
    plot_over_time(by, methods, 'theta_err_emp', '‖Θ̂ − S_t⁻¹‖_F / ‖S_t⁻¹‖_F', figs / 'theta_err_emp.png',
                   logy=True)
    if any(c[3] for c in by):
        plot_over_time(by, methods, 'psi_norm_pop', '‖Θ̂_ψ‖_F / ‖C_t⁻¹‖_F (truth 0)', figs / 'psi_norm_pop.png',
                       logy=True)
    print(f'\nwrote {args.out / "results.md"} and {figs}/*.png')


# %% main
def main(argv=None):
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.combine_only:
        grid = list(itertools.product(args.N, args.sigma2, args.eps, args.seeds))
        todo = grid if args.task is None else grid[args.task::args.n_tasks or len(grid)]
        for N, s2, eps, seed in todo:
            print(f'=== N={N} sigma2={s2:g} nt={args.nt} eps={eps:g} seed={seed} (N h sigma2 = {N * s2 / args.nt:g})')
            run_task(args, N, s2, eps, seed)
    if not args.no_combine:
        combine(args)


if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    main()
