"""M1 (Scattering_Second_Order_Q1 only) is an exactly Gaussian model: checks that need no sampling.

phi_l(x) = mean_n |x * psi_l(n)|^2 for every channel l of the Q=1 bank (phi included), so by Parseval
    theta . phi(x) = (1/M^2) sum_w g(w) |x^_w|^2,   g(w) = sum_l theta_l |psi^_l(w)|^2,
with g symmetrised, g(w) <- (g(w) + g(-w)) / 2, because the Morlet filters are one-sided and x is real,
i.e. p(x) ~ exp(-1/2 x^T Sigma^-1 x) with a circulant precision of eigenvalues  -2 g(w) / M.
It exists (normalisable) iff g(w) < 0 at every frequency. Then, exactly,
    log Z = 1/2 sum_w log(2 pi / s_w),   s_w = -2 g(w) / M,      H = log Z + M/2   (since E[theta.phi] = -M/2).
and y = 2 (-log p - H) + M follows chi^2_M (mean M, variance 2M) on the model's own samples.
This is the same circulant-Gaussian description as gauss_dim_sweep/analyse.py (g_eff).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / 'codes', HERE.parent / 'codes/filters', HERE.parent / 'codes/potentials'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def spectral_weight(theta, M, J):
    """g(w) (M,) for theta over the J+1 channels of the Q=1 bank, and the filters used."""
    from filters_bank import return_Filters
    filters, _ = return_Filters(M, J, 1, device='cpu', include_phi=True)       # the bank Scattering_Second_Order_Q1 uses
    psi = np.asarray(filters.reshape(-1, M).cpu().numpy())                      # (channels, M)
    assert psi.shape[0] == len(theta), (psi.shape, len(theta))
    g = (np.asarray(theta)[:, None] * np.abs(psi) ** 2).sum(0)
    return 0.5 * (g + np.roll(g[::-1], 1))      # analytic filters are one-sided: x is real, so w and -w are paired


def gaussian_exact(theta, M, J):
    """{'exists': g < 0 at every w, 'g_max', 'log_Z', 'H'} (log_Z and H are nan if it does not exist)."""
    g = spectral_weight(theta, M, J)
    out = {'exists': bool((g < 0).all()), 'g_max': float(g.max())}
    if out['exists']:
        s = -2.0 * g / M
        out['log_Z'] = float(0.5 * np.log(2 * np.pi / s).sum())
        out['H'] = out['log_Z'] + M / 2
    else:
        out['log_Z'] = out['H'] = float('nan')
    return out


def chi2_metrics(nlp_model, d):
    """y = 2 (-log p - mean) + d on the model samples: var(y)/(2d) (target 1) and the KS distance to chi^2_d."""
    from scipy import stats
    y = 2.0 * (np.asarray(nlp_model, float) - np.mean(nlp_model)) + d
    return {'var(y)/2d': float(y.var(ddof=1) / (2 * d)), 'KS to chi2_d': float(stats.kstest(y, 'chi2', args=(d,)).statistic)}


def validation_table(runs):
    """M1 only. One row per seed: exact H and log Z of the fitted Gaussian against the MGD bound H_bound, the lower
    bound log Z (H_bound + theta.m_target) and the sample entropy, plus the existence test and the chi^2 metrics.
    runs: the list of npz dicts of the Gaussian model (logp_entropy.load_all)."""
    import pandas as pd
    rows = []
    for r in runs:
        ex = gaussian_exact(r['theta'], r['d'], len(r['theta']) - 1)
        c2 = chi2_metrics(r['nlp_model'], r['d'])
        rows.append({'seed': r['seed'], 'exists (g<0)': ex['exists'], 'H exact': ex['H'], 'H_bound': r['H_bound'],
                     'H_sample': r['H_sample'], 'H_bound - H exact': r['H_bound'] - ex['H'],
                     'log Z exact': ex['log_Z'], 'log Z lower bound': r['log_Z_lb'], **c2})
    return pd.DataFrame(rows).set_index('seed')
