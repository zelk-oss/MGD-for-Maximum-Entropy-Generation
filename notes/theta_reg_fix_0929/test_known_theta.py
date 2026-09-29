# Known-theta test of the three energies. 1-D Cos interpolant, Z ~ N(0,1) at t=0,
# X ~ N(0, s^2) at t=1, so p_t = N(0, v(t)). Features phi = (x^2, x^4); in MGD's
# convention p ∝ exp(+theta.phi) the truth is theta = (-1/(2v), 0).
# Per node: fresh walkers from p_t; M, G, c, m, tau_mean from them exactly as
# forward_regularised builds them (tau with d = 1); b = M theta_true + noise whose
# per-node solution has covariance sig^2 M^-1 (noisy per-step theta, like theta_t).
import os as _os
REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
import sys, types, numpy as np, torch
sys.path[:0] = [REPO,
                REPO + '/codes',
                REPO + '/data']
import io, contextlib
from codes.sde_routines import SDE

s_data, B, n, adot, SIG = 0.2, 20000, 300, np.pi / 2, 0.5
t = np.linspace(0.02, 0.995, n)

def build(seed):
    rng = np.random.default_rng(seed)
    M, G, b, c, m, tm, th = [], [], [], [], [], [], []
    for tk in t:
        a = adot * tk
        Z, X = rng.standard_normal(B), s_data * rng.standard_normal(B)
        x = np.cos(a) * Z + np.sin(a) * X
        v = np.cos(a) ** 2 + s_data ** 2 * np.sin(a) ** 2
        theta = np.array([-1 / (2 * v), 0.0])
        phi = np.stack([x ** 2, x ** 4], 1)
        gphi = np.stack([2 * x, 4 * x ** 3], 1)
        Mk = gphi.T @ gphi / B
        L = np.linalg.cholesky(Mk)
        bk = Mk @ theta + SIG * L @ rng.standard_normal(2)        # theta_k = theta + N(0, SIG^2 M^-1)
        Xeff = (x - np.cos(a) * Z) / np.sin(a)
        tau = -adot * (np.tan(a) * (1 - Z ** 2) + Z * Xeff)
        M.append(torch.tensor(Mk)); G.append(torch.tensor(phi.T @ phi / B))
        b.append(torch.tensor(bk)); c.append(torch.tensor(phi.T @ tau / B))
        m.append(phi.mean(0)); tm.append(tau.mean()); th.append(theta)
    return M, G, b, c, torch.tensor(np.array(m)), np.array(tm), np.array(th)

def solve(sysd, lam, mode, dim=None):
    M, G, b, c, m, tm, _ = sysd
    s = types.SimpleNamespace(num_potentials=2)
    with contextlib.redirect_stdout(io.StringIO()):
        return SDE._solve_regularised_thomas(s, t, M, G, b, c, lam, mode=mode, m=m,
                                             tau_mean=tm, dim=dim).numpy()

systems = [build(sd) for sd in range(4)]
def score(lam, mode, dim=None):
    err_all, err_end = [], []
    for sysd in systems:
        th = sysd[-1]
        E = solve(sysd, lam, mode, dim)[:, 0] - th[:, 0]
        err_all.append(np.sqrt(np.mean((E / th[:, 0]) ** 2)))
        err_end.append(abs(E[-1] / th[-1, 0]))
    return np.mean(err_all), np.mean(err_end)

e0 = score(0.0, 'legacy')
print(f'lam=0 (per-node solve):            rel RMS err {e0[0]:.3f}   final node {e0[1]:.3f}')
for mode in ['legacy', 'fixed']:
    for lam in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        e = score(lam, mode)
        print(f'{mode:6s} lam={lam:7.0e}:                rel RMS err {e[0]:.3f}   final node {e[1]:.3f}')
for mult in [0.1, 1.0, 10.0]:
    e = score(mult, 'guth', dim=1)
    print(f'guth   multiplier={mult:4g} (lam_eff->{mult/np.pi**2:.3f}): rel RMS err {e[0]:.3f}   final node {e[1]:.3f}')
