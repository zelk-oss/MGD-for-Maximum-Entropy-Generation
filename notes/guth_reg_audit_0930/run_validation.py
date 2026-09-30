"""Part C validation (C1, C2) and the B4 noise-model check, 1-D, CPU (~15 min).

Run from this folder:  python run_validation.py  -> prints the tables, writes results.json

Errors are relative, in the Fisher metric of the reference law (mgd1d.score_err), over the
nodes with 0.05 <= t <= 0.95; all multipliers in the MGD sign. One seed per (sigma, n).
Estimators: per-step theta_hat (= Theta at lam = 0), Theta_legacy, Theta_moment (uniform and
Guth schedules). For each regularised estimator: the error at the best lam of the grid
(oracle) and at the lam chosen by codes/lam_select_cv.lam_path (held-out loss + veto).

C1  bimodal target log p_1 = -b x^4 + 5 b x^2 + b x / 2 (b = 0.5, as bimodal_experiment),
    phi = (x, x^2, x^3, x^4); references: MGD's theta_t (Fokker-Planck, MGD eq. 15) and
    theta_ME(m_t); sigma = 10 (bimodal_experiment) and 2.
C2  Gaussian X ~ N(0, 0.5^2), phi = x^2: theta_t = theta_ME = 1/(2 v_t) exactly.
B4  z_k = sqrt(n h sigma^2 / 2) L_k^T (theta_hat_k - theta_t(t_k)), L_k L_k^T = M_k (+ridge):
    if Cov(theta_hat_k) = 2 M_k^{-1} / (n h sigma^2), white, then E[z^2] = 1 per component and
    the lag-1 autocorrelation of z is 0.
"""
import json
import sys
import time

import numpy as np
import torch

sys.path[:0] = ['../..', '../../codes', '../../data']
from mgd1d import ADOT, Problem, fp_solve, particles, score_err, solve_theta   # noqa: E402
from codes.lam_select_cv import lam_path                                       # noqa: E402

LAMS = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
NT = 2000
NS = [1000, 4000, 16000]
results = {'C1': [], 'C2': [], 'B4': []}


def window(t):
    return (t >= 0.05) & (t <= 0.95)


def evaluate(sysd, refs, G, tag, sigma, n):
    """errors of all estimators vs each reference; refs: name -> (nodes, r) MGD sign."""
    w = window(sysd['t'])
    rows = []
    for mode, sched in [('legacy', 'uniform'), ('moment', 'uniform'), ('moment', 'guth')]:
        C, v = (sysd['G'], sysd['c']) if mode == 'legacy' else (sysd['Sigma'], sysd['mdot'])
        inp = {'t': sysd['t'], 'r': sysd['M'].shape[1], 'mode': mode, 'schedule': sched, 'dim': 1,
               'ridge': sysd['eps'], 'live': None,
               'M': [torch.tensor(a) for a in sysd['M']], 'C': [torch.tensor(a) for a in C],
               'b': [torch.tensor(a) for a in sysd['b']], 'v': [torch.tensor(a) for a in v],
               'h': sysd['h'], 'n_walkers': n, 'sigma': sigma}
        path = lam_path(inp, LAMS, verbose=False)
        Th_all = -path['Theta'].double().numpy()               # MGD sign, (L, nodes, r)
        for ref_name, ref in refs.items():
            errs = [score_err(Th[w], ref[w], G[w]) for Th in Th_all]
            i_or = int(np.argmin(errs))
            sel = path['lam_selected']
            e_sel = errs[path['lams'].index(sel)] if sel is not None else float('nan')
            rows.append(dict(case=tag, sigma=sigma, n=n, estimator=f'{mode}/{sched}', ref=ref_name,
                             err_lam0=errs[0], err_oracle=errs[i_or], lam_oracle=path['lams'][i_or],
                             err_selected=e_sel, lam_selected=sel,
                             err_by_lam=dict(zip(map(str, path['lams']), errs)),
                             chi2_selected=(float(path['chi2'][path['lams'].index(sel)])
                                            if sel is not None and path['chi2'] is not None else None),
                             n_vetoed=int(path['veto'].sum())))
    return rows


def b4(sysd, theta_ref, sigma, n, tag):
    """noise model of the per-step theta_hat (MGD sign) against a reference."""
    w = window(sysd['t'])
    th_hat = -sysd['theta_hat']
    z = []
    for k in np.where(w)[0]:
        Mk = sysd['M'][k]
        Mk = (Mk + Mk.T) / 2
        Mk = Mk + sysd['eps'] * np.diag(np.diag(Mk))
        L = np.linalg.cholesky(Mk)
        z.append(np.sqrt(n * sysd['h'][k] * sigma ** 2 / 2) * L.T @ (th_hat[k] - theta_ref[k]))
    z = np.array(z)
    zc = z - z.mean(0)
    lag1 = (zc[1:] * zc[:-1]).mean(0) / zc.var(0)
    return dict(case=tag, sigma=sigma, n=n, nt=len(sysd['t']) + 1, Ez2=(z ** 2).mean(0).tolist(),
                mean_z=z.mean(0).tolist(), lag1=lag1.tolist())


t0 = time.time()
# ------------------------------------------------------------------ C1
beta = 0.5
pb = Problem(lambda x: -beta * x ** 4 + 5 * beta * x ** 2 + beta * x / 2)
for sigma in (10.0, 2.0):
    fps = {}
    for nt in sorted({NT, NT // 2, NT * 2}):
        t_nodes = np.arange(1, nt) / nt
        fps[nt] = fp_solve(pb, sigma, t_nodes, dt_max=1e-4)
    fp = fps[NT]
    print(f'[C1 sigma={sigma}] FP done ({time.time() - t0:.0f}s); max moment err {fp["merr"].max():.1e}; '
          f'||theta_t - theta_ME||_G / ||theta_ME||_G = '
          f'{score_err(fp["theta_t"][window(fp["t"])], fp["theta_ME"][window(fp["t"])], fp["G"][window(fp["t"])]):.2e}')
    for n in NS:
        sysd = particles(pb, sigma, n, NT, seed=0)
        assert np.allclose(sysd['t'], fp['t'])
        results['C1'] += evaluate(sysd, {'theta_t': fp['theta_t'], 'theta_ME': fp['theta_ME']},
                                  fp['G'], 'C1', sigma, n)
        results['B4'].append(b4(sysd, fp['theta_t'], sigma, n, 'C1'))
        print(f'  n={n} done ({time.time() - t0:.0f}s)')
    for nt in (NT // 2, NT * 2):                                 # B4: h scaling at n = 4000
        sysd = particles(pb, sigma, 4000, nt, seed=1)
        results['B4'].append(b4(sysd, fps[nt]['theta_t'], sigma, 4000, 'C1'))

# ------------------------------------------------------------------ C2
s_data = 0.5
pg = Problem(lambda x: -x ** 2 / (2 * s_data ** 2), powers=(2,))
for sigma in (10.0, 2.0):
    for n in NS:
        sysd = particles(pg, sigma, n, NT, seed=0)
        t = sysd['t']
        v = np.cos(ADOT * t) ** 2 + s_data ** 2 * np.sin(ADOT * t) ** 2
        th = (1 / (2 * v))[:, None]
        G = (4 * v)[:, None, None]
        results['C2'] += evaluate(sysd, {'exact': th}, G, 'C2', sigma, n)
        results['B4'].append(b4(sysd, th, sigma, n, 'C2'))
    print(f'[C2 sigma={sigma}] done ({time.time() - t0:.0f}s)')

json.dump(results, open('results.json', 'w'), indent=1)

# ------------------------------------------------------------------ tables
print('\nerrors (relative, Fisher metric, 0.05 <= t <= 0.95)')
print(f"{'case':4} {'sig':>4} {'n':>6} {'estimator':16} {'ref':9} {'lam=0':>8} {'oracle':>8} {'lam_or':>7} "
      f"{'selected':>9} {'lam_sel':>8} {'chi2':>6}")
for row in results['C1'] + results['C2']:
    print(f"{row['case']:4} {row['sigma']:4.0f} {row['n']:6d} {row['estimator']:16} {row['ref']:9} "
          f"{row['err_lam0']:8.4f} {row['err_oracle']:8.4f} {row['lam_oracle']:7.0e} "
          f"{row['err_selected']:9.4f} {str(row['lam_selected']):>8} "
          f"{row['chi2_selected'] if row['chi2_selected'] is None else round(row['chi2_selected'], 2)!s:>6}")
print('\nB4 noise model: E[z^2] per component (1 if the model holds), lag-1 autocorrelation (0 if white)')
for row in results['B4']:
    print(f"{row['case']:4} sigma={row['sigma']:4.0f} n={row['n']:6d} nt={row['nt']:5d}  "
          f"E[z^2]={np.round(row['Ez2'], 2).tolist()}  lag1={np.round(row['lag1'], 3).tolist()}  "
          f"mean z={np.round(row['mean_z'], 2).tolist()}")
print(f'\ntotal {time.time() - t0:.0f}s')
