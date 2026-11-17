"""Tempered removal (method 1) and dressed response (method 3) of a group G of potentials, from per-sample energies.

Sign convention as in kl_groups.py: p ~ exp(+Theta.phi) in the code, E[:, k] = theta_k phi_k(x_i), A_G = -sum_{k in G} E[:, k].
q_{G,t} = the model with theta_G scaled to (1 - t) theta_G, others fixed, renormalised:  q_{G,t} ~ p exp(t A_G).

Method 1.  Lambda_G(t) = log E_p[exp(t (A_G - E_p A_G))] = D_KL(p || q_{G,t}),   t in [0, 1].
    Lambda is convex, Lambda(0) = 0, Lambda'(0) = 0, so it is increasing on [0, 1]; Lambda(1) = D_G (may be +inf).
    t_G(eps) = the t solving Lambda_G(t) = eps, or 1 if Lambda_G(1) < eps.  Small t_G = the model is stiff w.r.t. G.
    Estimator: Lambda_hat(t) = logsumexp(t (a - mean a)) - ln N  (biased low by Jensen, consistent), a_i = A_G(x_i).
    Its weights e^{t a} have finite variance if t z < 1/2 (z = exponential tail rate of A_G): no need for D_G < inf.

Method 3.  P_G = 1/2 min_beta Var_p(A_G - beta . phi_rest): SECOND-ORDER (quadratic) approximation of the information
    I_G = S(q_G*) - S(p) of the group when the other terms are refitted.  It is an APPROXIMATION (exact only for small A_G).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kl_groups import pareto_khat                                           # noqa: E402

EPS_LIST = (0.05, 0.1, 0.5)
T_MIN = 1e-9
ESS_MIN, KHAT_MAX = 100.0, 0.5


# --------------------------------------------------------------------------------------------- method 1: estimators
def lam_hat(a, t):
    """Lambda_hat(t) along the last axis of a (nats). a: (..., N); t: scalar or broadcastable to a.shape[:-1] + (1,)."""
    a = np.asarray(a, dtype=np.float64)
    return logsumexp(t * (a - a.mean(-1, keepdims=True)), axis=-1) - np.log(a.shape[-1])


def t_hat(a, eps):
    """Root of Lambda_hat(t) = eps on [T_MIN, 1] (brentq); 1.0 if Lambda_hat(1) < eps (censored)."""
    a = np.asarray(a, dtype=np.float64)
    d = a - a.mean()
    lnN = np.log(a.size)
    f = lambda t: logsumexp(t * d) - lnN - eps
    if f(1.0) < 0:
        return 1.0
    if f(T_MIN) > 0:
        return T_MIN
    return float(brentq(f, T_MIN, 1.0, xtol=1e-12, rtol=1e-10))


def t_hat_batch(A, eps_list, n_iter=22, chunk=10):
    """Vectorised version of t_hat for A of shape (B, N): bisection in log t (Lambda_hat is increasing in t), 22 steps
    = relative resolution 5e-6 on [1e-9, 1]. Single-pass log-sum-exp (t > 0: the max of t d is t max d). -> (B, n_eps)."""
    out = np.empty((A.shape[0], len(eps_list)))
    lnN = np.log(A.shape[-1])
    for c0 in range(0, A.shape[0], chunk):
        d = A[c0:c0 + chunk] - A[c0:c0 + chunk].mean(-1, keepdims=True)
        dm = d.max(-1)
        d -= dm[:, None]                                                    # d <= 0
        f = lambda t, eps: t * dm + np.log(np.exp(t[:, None] * d).sum(-1)) - lnN - eps
        n = d.shape[0]
        one = np.ones(n)
        for j, eps in enumerate(eps_list):
            cens = f(one, eps) < 0
            lo, hi = np.full(n, np.log(T_MIN)), np.zeros(n)
            for _ in range(n_iter):
                mid = 0.5 * (lo + hi)
                up = f(np.exp(mid), eps) > 0
                hi = np.where(up, mid, hi)
                lo = np.where(up, lo, mid)
            out[c0:c0 + n, j] = np.where(cens, 1.0, np.exp(0.5 * (lo + hi)))
    return out


def tempered_estimate(a, eps_list=EPS_LIST, n_boot=100, n_half=10, seed=0):
    """a: (N,) samples of A_G = -sum_G E[:, k] on draws of p. Returns {eps: dict} (all Lambda in nats).
    Reliable at eps <=> ESS(e^{t a}) >= 100 AND pareto khat(t a) < 0.5 AND drift <= 2 bootstrap sd, at t = t_hat:
      drift = Lambda_hat(t_hat; N) - mean over n_half random halves of Lambda_hat(t_hat; N/2),
      boot sd = bootstrap sd of Lambda_hat(t_hat; resample).  censored = (Lambda_hat(1) < eps, t_hat set to 1)."""
    a = np.asarray(a, dtype=np.float64)
    N = a.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=(n_boot, N))
    Ab = a[idx]
    tb = t_hat_batch(Ab, eps_list)
    halves = np.stack([rng.permutation(N)[:N // 2] for _ in range(n_half)])
    out = {}
    for j, eps in enumerate(eps_list):
        th = t_hat(a, eps)
        w = np.exp(th * (a - a.max()))
        ess = float(w.sum() ** 2 / np.sum(w ** 2))
        kh = pareto_khat(th * a)
        lam_b = lam_hat(Ab, th)                                                    # Lambda at fixed t_hat, per resample
        lam_sd = float(lam_b.std(ddof=1))
        drift = float(lam_hat(a, th) - lam_hat(a[halves], th).mean())
        ess_ok, kh_ok = ess >= ESS_MIN, bool(kh < KHAT_MAX)
        dr_ok = drift <= 2 * lam_sd
        out[eps] = dict(t_hat=th, t_sd=float(tb[:, j].std(ddof=1)), ess=ess, khat=kh, drift=drift, lam_sd=lam_sd,
                        ess_ok=ess_ok, khat_ok=kh_ok, drift_ok=bool(dr_ok), reliable=bool(ess_ok and kh_ok and dr_ok),
                        censored=bool(th >= 1.0))
    return out


# --------------------------------------------------------------------------------------------- method 3: dressed response
def dressed_response(a, Phi, n_folds=5, ridges=None, seed=0):
    """P_G = 1/2 min_beta Var(a - beta . Phi) with a ridge on beta chosen by n_folds-fold cross-validation.
    a: (N,) = A_G; Phi: (N, n_rest) = energies of the other terms (any sign; columns are standardised, constants dropped).
    Returns dict: bare = 1/2 Var(a); P_cv = 1/2 CV residual mean square at the best ridge (HELD-OUT, the number to use);
    P_in = in-sample 1/2 residual variance at that ridge (optimistic); P_ols_in = in-sample unregularised;
    ridge = chosen ridge (relative to unit-variance columns). The minimum over ridges makes P_cv slightly optimistic too."""
    a = np.asarray(a, dtype=np.float64)
    N = a.size
    sd = Phi.std(0)
    keep = sd > 1e-12 * max(sd.max(), 1e-300)
    X = (Phi[:, keep] - Phi[:, keep].mean(0)) / sd[keep]
    ridges = np.logspace(-8, 3, 23) if ridges is None else ridges
    fold = np.random.default_rng(seed).permutation(N) % n_folds
    mse = np.zeros((len(ridges), n_folds))
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        Xm, am = X[tr].mean(0), a[tr].mean()
        Xt, at = X[tr] - Xm, a[tr] - am
        G, g = Xt.T @ Xt / tr.sum(), Xt.T @ at / tr.sum()
        for i, lam in enumerate(ridges):
            beta = np.linalg.solve(G + lam * np.eye(G.shape[0]), g)
            r = (a[te] - am) - (X[te] - Xm) @ beta
            mse[i, f] = np.mean(r ** 2)
    cv = mse.mean(1)
    ib = int(np.argmin(cv))
    Xc, ac = X - X.mean(0), a - a.mean()
    G, g = Xc.T @ Xc / N, Xc.T @ ac / N
    def in_sample(lam):
        beta = np.linalg.solve(G + lam * np.eye(G.shape[0]), g)
        return 0.5 * float(np.mean((ac - Xc @ beta) ** 2))
    return dict(bare=0.5 * float(a.var()), P_cv=0.5 * float(cv[ib]), P_in=in_sample(ridges[ib]),
                P_ols_in=in_sample(1e-12), ridge=float(ridges[ib]), n_rest=int(keep.sum()))
