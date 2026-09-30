"""What the discrepancy principle would choose as the ridge, step by step, from a run's
conditioning log (--cond_every; codes/sde_routines.py SDE._log_cond).

    python notes/eps_conditioning_0930/discrepancy_from_cond.py <root>/saved_results/aux_moments/<config>_cond.pt
        [--root turbulence] [--tau 1.0] [--norm euclid|chi2] [--plot out.png]

Theta solve (corrector) at a logged step, in Jacobi-scaled coordinates, with
Gs = U diag(mu) U^T (live block, unit diagonal, before the ridge), beta = U^T rhs_s and
nu_i = variance of the rhs noise along u_i (both saved by the log):
    solution   theta_s(lam) = sum_i u_i beta_i / (mu_i + lam)
    residual   r_i(lam)     = -lam beta_i / (mu_i + lam)          (Gs theta_s - rhs_s)
Discrepancy principle (advisor's suggestion, direct-solve version): the smallest
regularisation whose residual is at the noise level,
    euclid:  ||r(lam)||^2           = tau^2 * sum_i nu_i           (noise E||eps||^2)
    chi2:    sum_i r_i(lam)^2 / nu_i = tau^2 * n_live              (noise whitened per
             eigen-direction; exact only if the noise covariance is diagonal in U)
||r(lam)|| increases with lam from 0 to ||beta||, so the root is unique; if even
lam -> inf (theta = 0) stays under the noise level, the rhs is pure noise at that step
and lam_dp = inf. Everything is exact given the log: no approximation beyond the noise
model (independent walker / interpolant sample means).

Per step it reports lam_dp, lam_dp / lam_run (lam_run = the run's --regularization),
the residual of the run's solve relative to the noise level (<< 1 = the run fits
noise; >> 1 = the run over-smooths), and the size of the discrepancy solution relative
to the run's. Summary by time band; per-step arrays saved next to the log.
The eta solve has no noise estimate in the log (only kappa is summarised for it).
"""
import argparse
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('log', type=Path)
ap.add_argument('--root', default=None, help='run output root (for the sampling times); '
                'default: two levels above the log (…/saved_results/aux_moments/)')
ap.add_argument('--tau', type=float, default=1.0, help='safety factor of the discrepancy principle')
ap.add_argument('--norm', choices=['euclid', 'chi2'], default='euclid')
ap.add_argument('--plot', type=Path, default=None)
a = ap.parse_args()

L = torch.load(a.log, map_location='cpu', weights_only=False)
lam_run = float(L['regularization'])
rows, spec = L['theta'].numpy(), L['theta_spectra'].double().numpy()      # (n, 9), (n, 3, r)
labels = L['labels']
config = a.log.name[:-len('_cond.pt')]
root = Path(a.root) if a.root else a.log.parents[2]
t_path = root / 'saved_results' / 'sampling_times' / f'{config}.pt'
t_fine = np.asarray(torch.load(t_path, map_location='cpu'), dtype=np.float64) if t_path.exists() else None
k = rows[:, 0].astype(int)
t = t_fine[np.minimum(k + 1, len(t_fine) - 1)] if t_fine is not None else np.full(len(k), np.nan)   # theta at t[k+1]


def resid2(lam, mu, beta, w):
    """sum_i w_i (lam beta_i / (mu_i + lam))^2; w = 1 (euclid) or 1/nu (chi2)."""
    return np.sum(w * (lam * beta / (mu + lam)) ** 2)


def sol_norm(lam, mu, beta):
    return np.sqrt(np.sum((beta / (mu + lam)) ** 2))


n = len(k)
lam_dp = np.full(n, np.nan); rel_run = np.full(n, np.nan); size_ratio = np.full(n, np.nan)
target = np.full(n, np.nan); kappa = rows[:, 3] / rows[:, 2]
for j in range(n):
    m = int(rows[j, 1])
    mu, beta, nu = spec[j, 0, :m], spec[j, 1, :m], spec[j, 2, :m]
    if not (np.isfinite(mu).all() and np.isfinite(beta).all() and np.isfinite(nu).all()):
        continue
    mu = np.maximum(mu, 1e-300)            # Gs is PSD; rounding can leave mu_min ~ -1e-17
    if a.norm == 'euclid':
        w, tgt = np.ones(m), a.tau ** 2 * nu.sum()
    else:
        w, tgt = 1.0 / np.maximum(nu, 1e-300), a.tau ** 2 * m
    target[j] = tgt
    rel_run[j] = np.sqrt(resid2(lam_run, mu, beta, w) / tgt)
    if np.sum(w * beta ** 2) <= tgt:       # even theta = 0 is within the noise
        lam_dp[j] = np.inf
        size_ratio[j] = 0.0
        continue
    lo, hi = -16.0, 8.0                    # bisection in log10(lam); residual increasing
    if resid2(10 ** lo, mu, beta, w) >= tgt:
        lam_dp[j] = 10 ** lo
    else:
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if resid2(10 ** mid, mu, beta, w) < tgt else (lo, mid)
        lam_dp[j] = 10 ** hi
        assert abs(np.sqrt(resid2(lam_dp[j], mu, beta, w) / tgt) - 1) < 1e-6
    size_ratio[j] = sol_norm(lam_dp[j], mu, beta) / sol_norm(lam_run, mu, beta)

print(f'{config}\n  theta solve: {n} logged steps (every {L["cond_every"]}), run ridge {lam_run:g}, '
      f'norm {a.norm}, tau {a.tau:g}')
print(f'  lam_dp = ridge chosen by the discrepancy principle; res_run = residual of the run\'s solve / '
      f'noise level (<1: fits noise, >1: over-smooths); |theta_dp|/|theta_run| in scaled units')
print(f'\n{"t band":>16} {"steps":>6} {"kappa med":>10} {"lam_dp med":>11} {"lam_dp p10":>11} {"lam_dp p90":>11}'
      f' {"inf":>5} {"res_run med":>12} {"size med":>9}')
bands = [(0, .1), (.1, .5), (.5, .9), (.9, .99), (.99, .999), (.999, 1.0001)] if t_fine is not None else [(-np.inf, np.inf)]
for lo, hi in bands:
    s = (t >= lo) & (t < hi) if t_fine is not None else np.ones(n, bool)
    s &= np.isfinite(rel_run)
    if not s.any():
        continue
    ld = lam_dp[s]; fin = np.isfinite(ld)
    q = (lambda p: np.percentile(ld[fin], p)) if fin.any() else (lambda p: np.nan)
    name = f'[{lo:.3f}, {min(hi, 1):.3f})' if t_fine is not None else 'all'
    print(f'{name:>16} {s.sum():6d} {np.median(kappa[s]):10.3g} {q(50):11.2e} {q(10):11.2e} {q(90):11.2e}'
          f' {int((~fin).sum()):5d} {np.median(rel_run[s]):12.3g} {np.median(size_ratio[s]):9.3g}')

# the statistics behind the worst-conditioned steps
worst = np.argsort(-np.nan_to_num(kappa, nan=-1))[:5]
print('\nworst-conditioned theta steps (kappa, top loadings of the smallest eigen-direction):')
for j in worst:
    tops = [labels[int(i)] for i in rows[j, 4:7] if i >= 0]
    print(f'  k={k[j]:6d} t={t[j]:.6f} kappa={kappa[j]:.3g} n_live={int(rows[j, 1])}  {tops}')

eta = L['eta'].numpy()
if len(eta):
    ke = eta[:, 3] / eta[:, 2]
    print(f'\neta solve: kappa median {np.median(ke):.3g}, p90 {np.percentile(ke, 90):.3g}, max {ke.max():.3g}')

out = a.log.with_name(a.log.name.replace('_cond.pt', f'_discrepancy_{a.norm}_tau{a.tau:g}.pt'))
torch.save({'k': k, 't': t, 'kappa': kappa, 'lam_dp': lam_dp, 'res_run': rel_run,
            'size_ratio': size_ratio, 'noise_target': target, 'lam_run': lam_run,
            'norm': a.norm, 'tau': a.tau}, out)
print(f'\nsaved {out}')

if a.plot:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = 1 - t if t_fine is not None else k
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    ax[0].loglog(x, kappa, '.', ms=2); ax[0].set_ylabel('kappa (theta, before ridge)')
    fin = np.isfinite(lam_dp)
    ax[1].loglog(x[fin], lam_dp[fin], '.', ms=2, label='discrepancy')
    ax[1].axhline(lam_run, color='k', ls='--', label=f'run ({lam_run:g})'); ax[1].legend()
    ax[1].set_ylabel('ridge')
    ax[2].loglog(x, rel_run, '.', ms=2); ax[2].axhline(1, color='k', ls='--')
    ax[2].set_ylabel("run's residual / noise level")
    for axi in ax:
        axi.set_xlabel('1 - t' if t_fine is not None else 'step'); axi.invert_xaxis() if t_fine is not None else None
    fig.savefig(a.plot, dpi=130)
    print(f'saved {a.plot}')
