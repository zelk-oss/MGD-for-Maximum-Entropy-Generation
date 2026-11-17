"""EXACT d16 Gaussian bench (gauss_scat4_test/runs/d16, p* in the Scat2 family with theta_star) for groups of the 5 Scat2 terms.

x ~ N(0, circulant), unnormalised FFT X_w, S_w = E|X_w|^2, y_w = |X_w|^2 / S_w, independent:
  pair modes w = 1..M/2-1 (X_w = conj X_{M-w}):   y ~ Exp(1)       (mean 1, var 1)
  real modes w = 0, M/2:                          y ~ chi^2_1      (mean 1, var 2)
theta.phi(x) = (1/M^2) sum_w c(w) |X_w|^2, c = theta.psi2, so the energy of Scat2 term k is E_k = -sum_w B[k, w] y_w with
  B[k, w] = -(1/M^2) S_w (c_k(w) + c_k(M-w))   (pair modes),     B[k, w] = -(1/M^2) S_w c_k(w)   (real modes)
and A_G = -sum_{k in G} E_k = sum_w b_w y_w, b = sum_{k in G} B[k, :]. log p* = -sum_pair y - 1/2 sum_real y  => sum_k B[k, w] = nu_w
(1 pair, 1/2 real); checked below (check_B).

Exact cumulant generating function (per mode, centred):  pair:  -ln(1 - s) - s  (s = t b < 1);
                                                          real:  -1/2 ln(1 - 2 s) - s  (s < 1/2).
So Lambda_G(t) = D_KL(p || q_{G,t}) is finite iff t z_G < 1,  z_G = max(max_pair b, 2 max_real b).
"""
import itertools
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq, linprog, minimize

ROOT = Path(__file__).resolve().parent.parent
D16 = ROOT / 'gauss_scat4_test/runs/d16/results'


def load_bench():
    b = torch.load(D16 / 'spectrum.pt', map_location='cpu', weights_only=False)
    S, th, psi2, M = b['S'][0].double().numpy(), b['theta_star'].double().numpy(), b['psi2'].double().numpy(), int(b['M'])
    c = th[:, None] * psi2                                                  # (5, M): c_k(w)
    modes = np.arange(M // 2 + 1)                                           # 0..M/2
    B = np.empty((th.size, modes.size))
    for i, w in enumerate(modes):
        if w in (0, M // 2):
            B[:, i] = -S[w] * c[:, w] / M ** 2
        else:
            B[:, i] = -S[w] * (c[:, w] + c[:, M - w]) / M ** 2
    real = np.isin(modes, (0, M // 2))
    return dict(B=B, real=real, M=M, S=S, theta=th, psi2=psi2)


def check_B(bench):
    nu = np.where(bench['real'], 0.5, 1.0)
    return float(np.abs(bench['B'].sum(0) - nu).max())


def all_subsets(n=5):
    return [c for r in range(1, n + 1) for c in itertools.combinations(range(n), r)]


def label(G):
    return '{' + ','.join(str(k) for k in G) + '}'


def coef(bench, G):
    return bench['B'][list(G)].sum(0)


def z_of(bench, G):
    b, real = coef(bench, G), bench['real']
    return float(max(b[~real].max(), 2 * b[real].max(), 0.0))


def lam_exact(bench, G, t):
    """Exact Lambda_G(t) (nats); +inf if t z >= 1. t may be an array."""
    b, real = coef(bench, G), bench['real']
    t = np.atleast_1d(np.asarray(t, dtype=float))
    s = t[:, None] * b[None]
    with np.errstate(invalid='ignore', divide='ignore'):
        pair = -np.log1p(-s) - s
        rl = -0.5 * np.log1p(-2 * s) - s
    val = np.where(real[None], rl, pair)
    val = np.where(np.where(real[None], 2 * s, s) >= 1, np.inf, val)
    return val.sum(1)


def t_exact(bench, G, eps):
    """Exact t_G(eps): Lambda_G(t) = eps; 1 if Lambda_G(1) < eps."""
    z = z_of(bench, G)
    if z < 1.0 and lam_exact(bench, G, 1.0)[0] < eps:
        return 1.0
    hi = 1.0 if z < 1.0 else (1.0 / z) * (1 - 1e-13)                        # Lambda -> +inf at t = 1/z
    f = lambda t: lam_exact(bench, G, t)[0] - eps
    return float(brentq(f, 1e-14, hi, xtol=1e-14, rtol=1e-12))


def draw_y(bench, N, rng):
    real = bench['real']
    n_modes = real.size
    y = rng.standard_exponential((N, n_modes))
    y[:, real] = rng.standard_normal((N, int(real.sum()))) ** 2
    return y


def A_from_y(bench, y, G):
    return y @ coef(bench, G)


# ----------------------------------------------------------------------- method 3 exact references
def P_exact(bench, G):
    """P_G = 1/2 min_beta Var(A_G - beta . A_rest) for the 5-term Scat2 bench, rest = the other Scat2 terms (exact)."""
    real = bench['real']
    var = np.where(real, 2.0, 1.0)
    rest = [k for k in range(bench['B'].shape[0]) if k not in G]
    bG = coef(bench, G)
    if not rest:
        return 0.5 * float(np.sum(var * bG ** 2))
    R = bench['B'][rest]
    W = np.sqrt(var)
    beta, *_ = np.linalg.lstsq((R * W).T, bG * W, rcond=None)
    res = (bG - beta @ R) * W
    return 0.5 * float(res @ res)


def I_exact(bench, G):
    """I_G = S(q_G*) - S(p) = D_KL(p || q_G*) with q_G* the member of the Scat2 family WITHOUT the terms of G, remaining
    coefficients refitted to the exact moments (convex).  Per mode with m_w = sum_{k not in G} gamma_k B[k, w]:
    pair: m - 1 - ln m;  real: 1/2 (r - 1 - ln r), r = m / nu = 2m.  inf if no remaining term (or no admissible m > 0)."""
    real = bench['real']
    nu = np.where(real, 0.5, 1.0)
    n_w = np.where(real, 0.5, 1.0)
    rest = [k for k in range(bench['B'].shape[0]) if k not in G]
    if not rest:
        return np.inf
    R = bench['B'][rest]                                                    # (r, modes)
    # phase 1: feasible gamma with m_w >= s_w > 0 (maximise the smallest m_w / nu_w), bounded box
    r = len(rest)
    c = np.zeros(r + 1); c[-1] = -1.0
    A_ub = np.hstack([-(R / nu).T, np.ones((R.shape[1], 1))])
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(R.shape[1]), bounds=[(-1e3, 1e3)] * r + [(None, 1.0)])
    if not res.success or res.x[-1] <= 1e-9:
        return np.inf
    g0 = res.x[:r]

    def fun(g):
        m = g @ R
        if np.any(m <= 0):
            return 1e30, np.zeros_like(g)
        rr = m / nu
        f = np.sum(n_w * (rr - 1 - np.log(rr)))
        grad = R @ (n_w * (1 - 1 / rr) / nu)
        return f, grad
    out = minimize(fun, g0, jac=True, method='BFGS', options=dict(gtol=1e-12, maxiter=2000))
    out = minimize(fun, out.x, jac=True, method='Nelder-Mead' if out.fun > 1e20 else 'BFGS',
                   options=dict(gtol=1e-13, maxiter=5000)) if out.fun < 1e20 else out
    return float(out.fun)
