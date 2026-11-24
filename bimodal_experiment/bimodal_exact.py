"""Exact (quadrature) quantities for the 1-D bimodal test, on top of lam_selection_criterion/bimodal_lamtest.py.

Code convention everywhere: log p_theta = theta . phi_c + const, phi_c = (x, x^2/2, x^3/3, x^4/4),
normalisable iff theta_4 < 0. Moments are returned in the plain monomial basis x^a (as the data moments m).
"""
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'lam_selection_criterion'))
from bimodal_lamtest import DX, GRID, kl_exact, log_density, log_z_quad, theta_star  # noqa: E402,F401

A = torch.arange(1, 5, dtype=torch.float64)
MONO = GRID[:, None] ** A                                   # (G, 4), x^a on the grid

# barrier of the true potential U = beta (x^4 - 5 x^2 - x/2): U' = 0 <=> 4x^3 - 10x - 1/2 = 0 (beta drops out);
# the root between the two wells
X_B = float(min(np.roots([4, 0, -10, -0.5]), key=lambda r: abs(r.real)).real)


def to_user_basis(theta):
    """theta in the convention p ∝ exp(-theta . (x, x^2, x^3, x^4)): theta_user_a = -theta_code_a / a."""
    return -torch.as_tensor(theta, dtype=torch.float64) / A


def moments(theta):
    """E_theta[x^a] (4,) and Cov_theta(x^a) (4, 4), exact on GRID."""
    w = log_density(theta)[0].exp() * DX
    E = w @ MONO
    C = (w[:, None] * MONO).T @ MONO - torch.outer(E, E)
    return E, C


def newton(theta0, m, max_iter=100, tol=1e-10):
    """Max-ent fit E_theta[x^a] = m by damped Newton on the convex dual f(theta) = log Z(theta) - theta . m_c
    (code basis, m_c = m / a), exact on GRID, starting from theta0. Backtracking halves the step until f decreases
    (this also keeps theta_4 < 0, where log Z is finite). Returns the iterates (n_iter + 1, 4)."""
    m_c = torch.as_tensor(m, dtype=torch.float64) / A
    f = lambda t: float(log_z_quad(t)[0] - t @ m_c)
    th, path = torch.as_tensor(theta0, dtype=torch.float64).clone(), []
    path.append(th.clone())
    for _ in range(max_iter):
        E, C = moments(th)
        g = E / A - m_c                                          # gradient of f = E_theta[phi_c] - m_c
        if float((g.abs() / m_c.abs()).max()) < tol:
            break
        d = torch.sqrt(torch.diagonal(C)) / A                    # diagonal scaling: Hessian = Cov(phi_c)
        step = torch.linalg.solve(C / torch.outer(A, A) / torch.outer(d, d), g / d) / d
        f0, eps = f(th), 1.0
        while not f(th - eps * step) < f0 and eps > 1e-10:
            eps /= 2
        th = th - eps * step
        path.append(th.clone())
    return torch.stack(path)


def p_below(theta, xb=X_B):
    """P_theta(x < xb): weight of the left (small) well."""
    w = log_density(theta)[0].exp() * DX
    return float(w[GRID < xb].sum())
