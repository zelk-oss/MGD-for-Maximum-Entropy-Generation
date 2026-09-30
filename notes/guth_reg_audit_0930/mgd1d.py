"""1-D ground truth and particle systems for validating Theta_reg (2026-09-30, Part C).

Sign conventions (stated per object):
  * MGD sign: p_theta ∝ exp(-theta . phi). Used for the Fokker-Planck solution, theta_t
    (MGD eq. 15: G_t theta_t = E[Delta phi(X_t)]) and theta_ME(m_t) (moment matching).
  * CODE sign: theta_code = -theta_MGD. Used for the particle systems handed to the real
    solver SDE._solve_regularised_thomas (as codes/sde_routines.py builds them).

Interpolant I_t = cos(a) Z + sin(a) X, a = pi t / 2, Z ~ N(0,1), X ~ p_1 (1-D).
phi = (x^1, ..., x^R) (R = 4 or subsets: `powers`).

fp_solve: MGD eq. 22 on a grid, dp/dt = -d/dx[(eta - sigma^2 theta) . phi' p] + sigma^2 p'',
  with eta, theta recomputed from the current p at every step (G eta = mdot_t exact,
  G theta = E_p[phi'']); Scharfetter-Gummel fluxes, implicit Euler, zero-flux walls.
particles: the code's predictor-corrector (iteration_step_projection) in NumPy float64,
  walkers x_0 ~ N(0,1) drawn INDEPENDENTLY of the interpolant noise Z (as run_experiment),
  collecting per node t_{k+1} both systems of forward_regularised: legacy (M, G, b, c) and
  moment (M, Sigma at x_{k+1}, b, pathwise mdot at t_{k+1}, h).
"""
import contextlib
import io
import types
from math import comb

import numpy as np
import torch
from scipy.linalg import solve_banded

ADOT = np.pi / 2


class Problem:
    """Target density on a grid, statistics phi = x^powers, exact interpolant moments."""

    def __init__(self, logp1, powers=(1, 2, 3, 4), L=7.0, N=1401):
        self.x = np.linspace(-L, L, N)
        self.dx = self.x[1] - self.x[0]
        self.powers = np.array(powers)
        lp = logp1(self.x)
        p1 = np.exp(lp - lp.max())
        self.p1 = p1 / (p1.sum() * self.dx)
        self.muX = np.array([(self.x ** j * self.p1).sum() * self.dx for j in range(max(powers) + 1)])
        self.muZ = np.array([0 if j % 2 else float(np.prod(np.arange(j - 1, 0, -2))) for j in range(max(powers) + 1)])
        self.r = len(powers)

    # statistics (any sign convention: they do not carry one)
    def phi(self, x):
        return np.stack([x ** a for a in self.powers], -1)

    def dphi(self, x):
        return np.stack([a * x ** (a - 1) for a in self.powers], -1)

    def d2phi(self, x):
        return np.stack([a * (a - 1) * x ** max(a - 2, 0) if a >= 2 else 0 * x for a in self.powers], -1)

    def m(self, t):
        """exact E[phi(I_t)] and d/dt of it."""
        c, s = np.cos(ADOT * t), np.sin(ADOT * t)
        m, md = np.zeros(self.r), np.zeros(self.r)
        for i, a in enumerate(self.powers):
            for j in range(a + 1):
                w = comb(a, j) * self.muZ[j] * self.muX[a - j]
                m[i] += w * c ** j * s ** (a - j)
                dj = (-j * c ** (j - 1) * s ** (a - j + 1) if j else 0.0) \
                    + ((a - j) * c ** (j + 1) * s ** (a - j - 1) if a - j else 0.0)
                md[i] += w * ADOT * dj
        return m, md

    def sample_X(self, n, rng):
        cdf = np.cumsum(self.p1) * self.dx
        cdf /= cdf[-1]
        return np.interp(rng.uniform(size=n), cdf, self.x)

    def expect(self, p, f):
        return (f * p[:, None]).sum(0) * self.dx

    def gram(self, p):
        g = self.dphi(self.x)
        return (g * p[:, None]).T @ g * self.dx

    def theta_MGD(self, p):
        """MGD eq. 15 (MGD sign): G theta = E_p[Delta phi]."""
        return np.linalg.solve(self.gram(p), self.expect(p, self.d2phi(self.x)))

    def theta_ME(self, m_target, th0):
        """max-entropy multiplier (MGD sign) with E_theta[phi] = m_target, by Newton."""
        th = th0.copy()
        P = self.phi(self.x)
        for _ in range(200):
            lw = -P @ th; lw -= lw.max(); p = np.exp(lw); p /= p.sum() * self.dx
            m = self.expect(p, P)
            C = (P * p[:, None]).T @ P * self.dx - np.outer(m, m)
            step = np.linalg.solve(C, m - m_target)        # dm/dtheta = -C
            th = th + step
            if np.abs(step).max() < 1e-12 * max(1, np.abs(th).max()):
                break
        return th, C


def _B(z):
    out = np.ones_like(z)
    nz = np.abs(z) > 1e-12
    out[nz] = z[nz] / np.expm1(z[nz])
    return out


def fp_solve(pb, sigma, t_record, dt_max=1e-4, tilt=True):
    """Self-consistent Fokker-Planck solution of MGD eq. 22 from p_0 = N(0,1).
    Returns dict with theta_t (MGD sign), theta_ME (MGD sign), G_t, moment error at t_record
    (t_record increasing, t_record[0] > 0). Implicit-Euler steps of at most dt_max; the
    coefficients eta, theta are taken from the current p, which leaves an O(dt sigma^2 kappa)
    error in the moments (kappa: curvature of -log p). With tilt=True the constraint
    E_p[phi] = m_t, which the exact dynamics conserves, is re-imposed after every step by an
    exponential tilt p <- p exp(-delta . phi) / Z (two Newton steps): a projection that only
    removes that discretisation error."""
    x, dx = pb.x, pb.dx
    p = np.exp(-x ** 2 / 2); p /= p.sum() * dx
    D = sigma ** 2
    xi = (x[:-1] + x[1:]) / 2
    gi = pb.dphi(xi)
    out = {k: [] for k in ('theta_t', 'theta_ME', 'G', 'merr', 't')}
    th_me = np.zeros(pb.r)
    if 2 in pb.powers:
        th_me[list(pb.powers).index(2)] = 0.5                # N(0,1): exp(-x^2/2)
    t = 0.0
    grid = np.concatenate([[0.0], np.asarray(t_record, dtype=float)])
    Px = pb.phi(x)
    for T0, T1 in zip(grid[:-1], grid[1:]):
        n_sub = max(1, int(np.ceil((T1 - T0) / dt_max)))
        dt = (T1 - T0) / n_sub
        for _ in range(n_sub):
            G = pb.gram(p)
            _, md = pb.m(t)
            eta = np.linalg.solve(G, md)
            th = np.linalg.solve(G, pb.expect(p, pb.d2phi(x)))
            v = gi @ (eta - D * th)                           # drift at the interfaces
            P = v * dx / D
            a_, b_ = (D / dx) * _B(-P), (D / dx) * _B(P)      # J = a p_j - b p_{j+1}
            k = dt / dx
            diag = np.ones_like(p); diag[:-1] += k * a_; diag[1:] += k * b_
            ab = np.zeros((3, len(p)))
            ab[0, 1:] = -k * b_; ab[1] = diag; ab[2, :-1] = -k * a_
            p = solve_banded((1, 1), ab, p)
            t += dt
            if tilt:
                m_t, _ = pb.m(t)
                for _ in range(2):
                    m_p = pb.expect(p, Px)
                    C = (Px * p[:, None]).T @ Px * dx - np.outer(m_p, m_p)
                    delta = np.linalg.solve(C, m_p - m_t)
                    lw = -Px @ delta
                    p = p * np.exp(lw - lw.max()); p /= p.sum() * dx
        m_ex, _ = pb.m(T1)
        th_me, _ = pb.theta_ME(m_ex, th_me)
        out['theta_t'].append(pb.theta_MGD(p)); out['theta_ME'].append(th_me.copy())
        out['G'].append(pb.gram(p)); out['t'].append(T1)
        out['merr'].append(np.abs(pb.expect(p, pb.phi(x)) - m_ex) / (np.abs(m_ex) + 1e-12))
    return {k: np.array(v) for k, v in out.items()}


def _ridge_solve(G, rhs, eps):
    """(G + eps diag G)^{-1} rhs, as compute_eta/compute_theta (Jacobi scale + eps I)."""
    d = np.sqrt(np.diag(G))
    Gs = G / np.outer(d, d)
    Gs = (Gs + Gs.T) / 2 + eps * np.eye(len(d))
    return np.linalg.solve(Gs, rhs / d) / d


def particles(pb, sigma, n, nt, seed, eps=1e-6):
    """The code's predictor-corrector, CODE sign. Returns per-node systems + per-step theta."""
    rng = np.random.default_rng(seed)
    Z, X = rng.standard_normal(n), pb.sample_X(n, rng)
    x = rng.standard_normal(n)                                 # walkers: independent of Z
    t = np.linspace(0, 1, nt + 1)
    I = lambda tt: np.cos(ADOT * tt) * Z + np.sin(ADOT * tt) * X
    Idot = lambda tt: ADOT * (-np.sin(ADOT * tt) * Z + np.cos(ADOT * tt) * X)
    mdot_hat = lambda tt: (pb.dphi(I(tt)) * Idot(tt)[:, None]).mean(0)
    gram = lambda y: pb.dphi(y).T @ pb.dphi(y) / n
    keys = ['t', 'M', 'b', 'G', 'c', 'Sigma', 'mdot', 'h', 'theta_hat']
    S = {k: [] for k in keys}
    md = mdot_hat(t[0])
    for k in range(nt):
        h = t[k + 1] - t[k]
        eta = _ridge_solve(gram(x), md, eps)
        y = x + h * pb.dphi(x) @ eta + np.sqrt(2 * h) * sigma * rng.standard_normal(n)
        Py = pb.phi(y)
        braw = pb.phi(I(t[k + 1])).mean(0) - Py.mean(0)
        Mk = gram(y)
        th_raw = _ridge_solve(Mk, braw, eps)
        x = y + pb.dphi(y) @ th_raw
        md = mdot_hat(t[k + 1])
        a = ADOT * t[k + 1]
        if np.sin(a) > 1e-8 and np.cos(a) > 1e-8:
            Xe = (y - np.cos(a) * Z) / np.sin(a)
            tau = -ADOT * (np.tan(a) * (1 - Z ** 2) + Z * Xe)
            P1 = pb.phi(x); P1c = P1 - P1.mean(0)
            S['t'].append(t[k + 1]); S['M'].append(Mk); S['b'].append(braw / (h * sigma ** 2))
            S['G'].append(Py.T @ Py / n); S['c'].append(Py.T @ tau / n)
            S['Sigma'].append(P1c.T @ P1c / n); S['mdot'].append(md.copy()); S['h'].append(h)
            S['theta_hat'].append(th_raw / (h * sigma ** 2))
    out = {k: np.array(v) for k, v in S.items()}
    out.update(n=n, sigma=sigma, eps=eps)
    return out


def solve_theta(sysd, lam, mode, schedule='uniform', dim=1, drop_first=True):
    """Real solver on a particle system. Returns Theta in the MGD sign (= -Theta_code)."""
    T = lambda A: [torch.tensor(a) for a in A]
    C, v = (sysd['G'], sysd['c']) if mode == 'legacy' else (sysd['Sigma'], sysd['mdot'])
    s = types.SimpleNamespace(num_potentials=sysd['M'].shape[1])
    with contextlib.redirect_stdout(io.StringIO()):
        Th = SDE_solver(s, sysd['t'], T(sysd['M']), T(C), T(sysd['b']), T(v), lam,
                        ridge=sysd['eps'], mode=mode, dim=dim, schedule=schedule)
    return -Th.numpy()


def SDE_solver(*a, **k):
    from codes.sde_routines import SDE
    return SDE._solve_regularised_thomas(*a, **k)


def score_err(Th, ref, G):
    """relative error in the Fisher metric of the reference law:
    sqrt(sum_k D_k^T G_k D_k / sum_k ref_k^T G_k ref_k), D = Th - ref (same sign)."""
    D = Th - ref
    num = np.einsum('ki,kij,kj->', D, G, D)
    den = np.einsum('ki,kij,kj->', ref, G, ref)
    return float(np.sqrt(num / den))
