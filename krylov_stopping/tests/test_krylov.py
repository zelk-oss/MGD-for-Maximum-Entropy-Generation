"""Unit tests of codes/krylov.py. Run from the repo root:  python krylov_stopping/tests/test_krylov.py
(plain asserts, no pytest on this machine)."""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import cg as sp_cg, minres as sp_minres

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'codes'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from codes import krylov as K                                       # noqa: E402

torch.set_default_dtype(torch.float64)
NEVER = lambda x, r, j: (False, float(r @ r))                       # run max_iter iterations


def spd(n, cond, seed):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    mu = torch.logspace(0, -np.log10(cond), n)
    return (Q * mu) @ Q.T, torch.randn(n, generator=g)


def scipy_iterates(fn, A, b, n_it):
    xs = []
    fn(A.numpy(), b.numpy(), rtol=0.0, maxiter=n_it, callback=lambda xk: xs.append(np.array(xk)))
    return xs


def collect(solver, A, b, n_it):
    """Iterates x_1..x_n_it of our solver (stop_fn sees every iterate)."""
    xs = []

    def stop_fn(x, r, j):
        if j:
            xs.append(x.clone())
        return False, 0.0
    solver(lambda v: A @ v, b, stop_fn, n_it)
    return xs


def test_vs_scipy():
    """Iterate by iterate, over the first iterations. Later the two implementations drift
    apart (~x8 per iteration at cond 1e2 from j ~ 14 on, ~x100 at cond 1e8 from j ~ 8 on;
    residual norms still equal): finite-precision Lanczos amplifies rounding differences once
    orthogonality is lost, so only early iterates are comparable to rounding; convergence is
    checked in test_vs_direct."""
    for cond, n_it, tol in ((1e2, 12, 1e-12), (1e8, 8, 1e-10)):
        A, b = spd(36, cond, 0)
        for ours, theirs, name in ((K.cg, sp_cg, 'cg'), (K.minres, sp_minres, 'minres')):
            xo = collect(ours, A, b, n_it)
            xs = scipy_iterates(theirs, A, b, n_it)
            m = min(len(xo), len(xs))
            assert m >= n_it - 1, (name, cond, len(xo), len(xs))
            err = max(float(np.linalg.norm(xo[j].numpy() - xs[j]) / np.linalg.norm(xs[j]))
                      for j in range(m))
            assert err < tol, (name, cond, err)
            print(f'  {name:6s} cond {cond:.0e}: {m} iterates match scipy, max rel diff {err:.1e}')


def test_vs_direct():
    A, b = spd(36, 1e3, 1)
    x_star = torch.linalg.solve(A, b)
    for solver in (K.cg, K.minres):
        x, j, _ = solver(lambda v: A @ v, b, NEVER, 200)
        err = float((x - x_star).norm() / x_star.norm())
        assert err < 1e-8, (solver.__name__, err)
        print(f'  {solver.__name__:6s} vs torch.linalg.solve (cond 1e3): rel err {err:.1e} after {j} it')
    # Ill-conditioned: compare the error. The residual is not a fair test here: CG does
    # not minimise it (at cond 1e10 it swings between 1 and 700 over 400 iterations, as
    # scipy's does), and MINRES's attainable residual at cond 1e8 is ~1e-4 (scipy: 1.2e-4).
    A, b = spd(36, 1e8, 2)
    x_star = torch.linalg.solve(A, b)
    for solver in (K.cg, K.minres):
        x, j, _ = solver(lambda v: A @ v, b, NEVER, 2000)
        err = float((x - x_star).norm() / x_star.norm())
        assert err < 1e-8, (solver.__name__, err)
        print(f'  {solver.__name__:6s} vs torch.linalg.solve (cond 1e8): rel err {err:.1e} after {j} it')


def test_minres_residual():
    """||r_j|| non-increasing, and the recurrence residual == b - A x_j. Also on an indefinite
    matrix (MINRES needs only symmetry)."""
    A, b = spd(36, 1e6, 3)
    A_ind = A - 0.3 * torch.eye(36)                                  # indefinite
    for M, name in ((A, 'SPD'), (A_ind, 'indefinite')):
        rec, exp = [], []

        def stop_fn(x, r, j):
            rec.append(float(r.norm()))
            exp.append(float((b - M @ x).norm()))
            return False, rec[-1]
        K.minres(lambda v: M @ v, b, stop_fn, 36)
        rec, exp = np.array(rec), np.array(exp)
        gap = np.abs(rec - exp).max() / exp[0]
        incr = np.diff(exp).max() / exp[0]
        assert gap < 1e-8, (name, gap)
        assert incr < 1e-10, (name, incr)
        print(f'  minres {name}: recurrence vs explicit residual {gap:.1e}, max increase {incr:.1e}')


def test_stop_at_zero():
    A, b = spd(10, 10, 4)
    for solver in (K.cg, K.minres):
        x, j, h = solver(lambda v: A @ v, b, lambda x, r, j: (True, -1.0), 10)
        assert j == 0 and float(x.abs().max()) == 0 and h == [-1.0]
    print('  stop at j = 0 returns x = 0')


def test_ito_stat_is_A_norm_error():
    A, b = spd(36, 1e4, 5)
    x_star = torch.linalg.solve(A, b)
    stop = K.ito_stop(A, scale=3.0, thr=-1.0)                        # never stops
    xs, stats = [], []

    def stop_fn(x, r, j):
        s = stop(x, r, j)
        xs.append(x.clone()); stats.append(s[1])
        return s
    K.cg(lambda v: A @ v, b, stop_fn, 20)
    err = [3.0 * float((x_star - x) @ A @ (x_star - x)) for x in xs]
    rel = max(abs(s - e) / err[0] for s, e in zip(stats, err))
    assert rel < 1e-8, rel
    assert np.all(np.diff(stats) <= 1e-12 * stats[0])                # CG minimises it
    print(f'  ito statistic == scale * ||x* - x_j||_A^2 (rel {rel:.1e}), non-increasing along CG')


def test_shrink():
    g = torch.Generator().manual_seed(6)
    X = torch.randn(40, 6, generator=g) @ torch.randn(6, 6, generator=g)
    S, lam = K.shrink_cov(X)
    S0 = K.cov(X)
    n = X.shape[0]                                                   # naive (n, r, r) formula
    Xc = X - X.mean(0)
    w = Xc[:, :, None] * Xc[:, None, :]
    var_s = n / (n - 1) ** 3 * ((w - w.mean(0)) ** 2).sum(0)
    off = ~torch.eye(6, dtype=torch.bool)
    lam_ref = float(var_s[off].sum() / (S0[off] ** 2).sum())
    assert abs(lam - min(1, max(0, lam_ref))) < 1e-12, (lam, lam_ref)
    assert torch.allclose(S.diagonal(), S0.diagonal())
    assert torch.allclose(S[off], (1 - lam) * S0[off])
    print(f'  Schaefer-Strimmer lam = {lam:.3f} matches the naive formula; diagonal kept')


def test_weights():
    g = torch.Generator().manual_seed(7)
    X = torch.randn(500, 5, generator=g) @ torch.randn(5, 5, generator=g)
    Sig = K.cov(X)
    Wih, eps2 = K.tol_weight(Sig, 'full', alpha=0.05)
    assert torch.allclose(Wih @ Sig @ Wih, torch.eye(5), atol=1e-10)
    assert abs(eps2 - 11.0705) < 1e-3                                 # chi2_{5, 0.95}
    d, eps2d = K.tol_weight(Sig, 'diag', kappa=2.0)
    R = Sig / torch.outer(Sig.diagonal().sqrt(), Sig.diagonal().sqrt())
    assert torch.allclose(d, Sig.diagonal().rsqrt())
    assert abs(eps2d - (5 + 2 * np.sqrt(2) * float(R.norm()))) < 1e-12
    Sig2 = Sig.clone()                                                # floor on a tiny variance
    Sig2[0, :] = Sig2[:, 0] = 0
    Sig2[0, 0] = 1e-20
    d2, _ = K.tol_weight(Sig2, 'diag', var_floor=1e-6)
    assert float(d2[0]) ** -2 >= 1e-6 * float(Sig2.diagonal().median()) * (1 - 1e-12)
    print('  tol_weight: full whitens to I, chi2 / diag thresholds, variance floor')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            print(name)
            fn()
    print('ALL PASSED')
