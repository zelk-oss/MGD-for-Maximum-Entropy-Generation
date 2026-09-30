"""C5: interpolant-side time term vs the exponential-family identity (independent of the
agent's interp_cov_bias.py). 1-D, data = 1/2 N(-mu, s^2) + 1/2 N(mu, s^2), phi = (x^2, x^4),
MGD convention p_theta ∝ exp(-theta.phi). theta(t) solves E_theta[phi] = m(t) (quadrature,
Newton). Exact: thetadot = -Cov_theta^{-1} mdot. Interpolant-side TSM: -Cov_I^{-1} mdot."""
import numpy as np

mu, s = 2.0, 0.4
x = np.linspace(-12, 12, 24001); dx = x[1] - x[0]
phi = np.stack([x ** 2, x ** 4])
A = lambda t: np.pi * t / 2

def p_I(t):
    a = A(t); v = (np.sin(a) * s) ** 2 + np.cos(a) ** 2; c = np.sin(a) * mu
    g = lambda m: np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)
    return 0.5 * (g(c) + g(-c))

def moments(p):
    m = phi @ p * dx
    C = (phi * p) @ phi.T * dx - np.outer(m, m)
    return m, C

def theta_of(m_target, th0):
    th = th0.copy()
    for _ in range(200):
        lw = -th @ phi; lw -= lw.max(); p = np.exp(lw); p /= p.sum() * dx
        m, C = moments(p)
        step = np.linalg.solve(C, m - m_target)     # dm/dtheta = -C  ->  Newton
        th = th + step
        if np.abs(step).max() < 1e-13: break
    return th, C

th = np.array([0.5, 0.0])                            # t = 0: N(0, 1)
e = 1e-5
print(f"{'t':>5} {'|err| Cov_theta':>16} {'|err| Cov_I':>12}")
for t in (0.3, 0.5, 0.7, 0.8, 0.9):
    mp, _ = moments(p_I(t + e)); mm, _ = moments(p_I(t - e)); m0, C_I = moments(p_I(t))
    mdot = (mp - mm) / (2 * e)
    th, C_th = theta_of(m0, th)
    thp, _ = theta_of(mp, th); thm, _ = theta_of(mm, th)
    thdot = (thp - thm) / (2 * e)
    r_th = np.linalg.norm(-np.linalg.solve(C_th, mdot) - thdot) / np.linalg.norm(thdot)
    r_I = np.linalg.norm(-np.linalg.solve(C_I, mdot) - thdot) / np.linalg.norm(thdot)
    print(f"{t:5.2f} {r_th:16.2e} {r_I:12.3f}")
