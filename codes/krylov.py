"""
Early-stopped Krylov solvers for the per-step MGD solves (2026-10-06, krylov_stopping/).

An alternative to the ridge of SDE.compute_eta / compute_theta: instead of solving
(G + lam diag G) x = b exactly, run CG or MINRES from x = 0 and stop at the first
iterate whose residual is at the noise level of b (a discrepancy-type rule).
Used by codes/sde_krylov.py (SDEKrylov); the ridge path of codes/sde_routines.py is
unchanged.

Solvers
-------
cg(matvec, b, stop_fn, max_iter)      Hestenes-Stiefel CG, A SPD.
minres(matvec, b, stop_fn, max_iter)  Paige-Saunders (1975) MINRES, A symmetric. The
                                      residual norm is monotone non-increasing.
Both start from x = 0 (x0: hook for a warm start, not implemented). matvec is a closure,
so a matrix-free operator v -> (1/N) sum_i grad phi(y_i) [grad phi(y_i)^T v] can replace
the explicit G @ v later. stop_fn(x, residual, j) -> (stop: bool, statistic: float) is
called at every iterate INCLUDING j = 0 (x = 0, residual = b), so it can return x = 0.
Return: (x, j, history), j = number of iterations done, history = statistic at j = 0..j.
(Deviation from the 2026-10-06 spec "stop_fn returns True": it also returns the
statistic, which is how the history gets recorded.)

Stopping rules (factories returning stop_fn)
--------------------------------------------
ito_stop      (N h sigma^2 / 2) r^T A^-1 r <= chi2_{r, 1-alpha}, A^-1 via one Cholesky.
residual_stop ||r||^2 <= eps2 (r already whitened by the caller's change of variables).

Noise weights
-------------
tol_weight    Sigma = sum_parts Cov(X_part) / m_part ('diag' | 'full' | 'shrink'),
              floored, and its inverse square root + threshold eps2.
"""
import math

import torch
from scipy.stats import chi2

_TINY = 1e-300


# ------------------------------------------------------------------ solvers
def cg(matvec, b, stop_fn, max_iter, x0=None):
    """Conjugate gradients on A x = b (A SPD), x_0 = 0. See module docstring."""
    if x0 is not None:
        raise NotImplementedError('warm start: hook only')
    x = torch.zeros_like(b)
    r = b.clone()                                      # r = b - A x
    stop, stat = stop_fn(x, r, 0)
    hist = [stat]
    if stop:
        return x, 0, hist
    p = r.clone()
    rr = float(r @ r)
    j = 0
    for j in range(1, max_iter + 1):
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if pAp <= 0 or rr == 0:                        # A singular along p, or solved
            j -= 1
            break
        alpha = rr / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        stop, stat = stop_fn(x, r, j)
        hist.append(stat)
        if stop:
            break
        rr_new = float(r @ r)
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x, j, hist


def minres(matvec, b, stop_fn, max_iter, x0=None):
    """MINRES on A x = b (A symmetric), x_0 = 0: Paige & Saunders (1975), unpreconditioned,
    following scipy.sparse.linalg.minres. x_j minimises ||b - A x|| over the Krylov space
    K_j(A, b), so ||r_j|| is non-increasing. The residual VECTOR passed to stop_fn is kept
    by the recurrence r_j = s_j^2 r_{j-1} - phibar_j c_j v_{j+1} (no extra matvec; checked
    against b - A x_j in krylov_stopping/tests/test_krylov.py)."""
    if x0 is not None:
        raise NotImplementedError('warm start: hook only')
    x = torch.zeros_like(b)
    r = b.clone()
    stop, stat = stop_fn(x, r, 0)
    hist = [stat]
    beta1 = float(b.norm())
    if stop or beta1 == 0:
        return x, 0, hist
    # Lanczos: r1, r2 = beta_{j-1} v_{j-1}, beta_j v_j
    r1, r2 = b.clone(), b.clone()
    oldb, beta = 0.0, beta1
    # QR of the tridiagonal by Givens rotations (cs, sn)
    dbar, epsln, phibar, cs, sn = 0.0, 0.0, beta1, -1.0, 0.0
    w, w2 = torch.zeros_like(b), torch.zeros_like(b)
    j = 0
    for j in range(1, max_iter + 1):
        v = r2 / beta
        y = matvec(v)
        if j >= 2:
            y = y - (beta / oldb) * r1
        alfa = float(v @ y)
        y = y - (alfa / beta) * r2
        r1, r2 = r2, y
        oldb, beta = beta, float(y.norm())
        oldeps = epsln
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln = sn * beta
        dbar = -cs * beta
        gamma = max(math.hypot(gbar, beta), _TINY)
        cs, sn = gbar / gamma, beta / gamma
        phi, phibar = cs * phibar, sn * phibar
        w1, w2 = w2, w
        w = (v - oldeps * w1 - delta * w2) / gamma
        x = x + phi * w
        r = sn * sn * r
        if beta > 0:
            r = r - (phibar * cs / beta) * r2           # r2 / beta = v_{j+1}
        stop, stat = stop_fn(x, r, j)
        hist.append(stat)
        if stop or beta == 0:                          # beta = 0: invariant subspace, solved
            break
    return x, j, hist


# ------------------------------------------------------------------ stopping rules
def chi2_threshold(r, alpha):
    return float(chi2.ppf(1 - alpha, r))


def ito_stop(A, scale, thr):
    """Rule 'ito_cg': stop when scale * r^T A^-1 r <= thr. With A the Jacobi-scaled Gram
    D^-1/2 G D^-1/2 and r the scaled residual D^-1/2 (b - G theta), r^T A^-1 r = u^T G^-1 u
    for the unscaled residual u: the squared A-norm of the CG error, which CG minimises.
    scale = N / (2 h sigma^2) for a residual in moment units. One Cholesky per call. If A is not
    positive definite (exactly singular, e.g. float32 accumulation), A^-1 is replaced by the
    pseudo-inverse on the range of A (eigenvalues > 1e-12 x the largest): noise along an exactly
    null direction cannot move the moments."""
    L, info = torch.linalg.cholesky_ex(A)
    if int(info) == 0:
        def quad(r):
            z = torch.linalg.solve_triangular(L, r[:, None], upper=False)
            return float((z * z).sum())
    else:
        mu, U = torch.linalg.eigh((A + A.T) / 2)
        keep = mu > 1e-12 * float(mu.abs().max())
        U, mu = U[:, keep], mu[keep]

        def quad(r):
            return float(((U.T @ r) ** 2 / mu).sum())

    def stop_fn(x, r, j):
        stat = scale * quad(r)
        return stat <= thr, stat
    return stop_fn


def residual_stop(eps2):
    """Rule 'tol_minres': stop when ||r||^2 <= eps2 (r whitened by the caller)."""
    def stop_fn(x, r, j):
        stat = float(r @ r)
        return stat <= eps2, stat
    return stop_fn


# ------------------------------------------------------------------ noise weights
def cov(X):
    """Unbiased sample covariance of the rows of X, float64."""
    Xc = X.double() - X.double().mean(0)
    return Xc.T @ Xc / (X.shape[0] - 1)


def shrink_cov(X):
    """Schaefer & Strimmer (2005) shrinkage of the sample covariance toward its diagonal
    (their target D: unequal variances, zero covariances): off-diagonals scaled by
    (1 - lam), lam = sum_{i!=j} Var^(s_ij) / sum_{i!=j} s_ij^2 clipped to [0, 1], with
    Var^(s_ij) = n / (n-1)^3 sum_k (w_kij - wbar_ij)^2, w_kij = xc_ki xc_kj.
    Returns (S_shrunk, lam). O(n r^2), no (n, r, r) buffer."""
    n = X.shape[0]
    Xc = X.double() - X.double().mean(0)
    Wbar = Xc.T @ Xc / n
    S = Wbar * n / (n - 1)
    X2 = Xc * Xc
    var_s = n / (n - 1) ** 3 * (X2.T @ X2 - n * Wbar ** 2)
    off = ~torch.eye(S.shape[0], dtype=torch.bool, device=S.device)
    den = float((S[off] ** 2).sum())
    lam = 1.0 if den == 0 else min(1.0, max(0.0, float(var_s[off].sum()) / den))
    out = S * (1 - lam)
    out.diagonal().copy_(S.diagonal())
    return out, lam


def tol_sigma(parts, mode):
    """Sigma = sum over parts (X, m) of Cov(X) / m: X (n, r) per-sample values whose mean
    enters the right-hand side, m the effective sample size of that mean. mode 'shrink'
    shrinks each covariance separately (shrink_cov) before the sum."""
    Sig = 0
    for X, m in parts:
        S = shrink_cov(X)[0] if mode == 'shrink' else cov(X)
        Sig = Sig + S / m
    return Sig


def tol_weight(Sigma, mode, var_floor=1e-6, alpha=0.05, kappa=2.0):
    """Inverse square root of the noise covariance Sigma (r, r) and the threshold eps2 on
    ||Sigma^-1/2 residual||^2.

    Floor: variances below var_floor * median(variance) are raised to it (added to the
    diagonal). 'full' / 'shrink' also clamp eigenvalues at that floor, so a direction
    with (near-)zero noise, e.g. exactly duplicated statistics, is not whitened by 1/0.
      'diag'            Wih (r,) = diag(Sigma)^-1/2, eps2 = r + kappa sqrt(2) ||R||_F,
                        R the correlation matrix of Sigma (mean + kappa std of the
                        whitened squared norm: E = r, Var = 2 tr R^2).
      'full' | 'shrink' Wih (r, r) = Sigma^-1/2, eps2 = chi2_{r, 1-alpha}.
    Returns (Wih, eps2)."""
    Sigma = Sigma.double()
    r = Sigma.shape[0]
    var = Sigma.diagonal()
    fl = var_floor * float(var.median())
    Sigma = Sigma + torch.diag((fl - var).clamp_min(0))
    var = Sigma.diagonal()
    if mode == 'diag':
        d = var.rsqrt()
        R = Sigma * d[:, None] * d[None, :]
        return d, r + kappa * math.sqrt(2) * float(R.norm())
    if mode not in ('full', 'shrink'):
        raise ValueError(f"weight mode must be 'diag', 'full' or 'shrink', got {mode!r}")
    mu, U = torch.linalg.eigh((Sigma + Sigma.T) / 2)
    mu = mu.clamp_min(max(fl, _TINY))
    return (U * mu.rsqrt()) @ U.T, chi2_threshold(r, alpha)


def whiten(Wih, G, b):
    """G~ = W^-1/2 G W^-1/2, b~ = W^-1/2 b for Wih a vector (diagonal) or a matrix."""
    if Wih.dim() == 1:
        return G * Wih[:, None] * Wih[None, :], b * Wih
    return Wih @ G @ Wih, Wih @ b


def unwhiten(Wih, x_tilde):
    """theta = W^-1/2 theta~."""
    return x_tilde * Wih if Wih.dim() == 1 else Wih @ x_tilde
