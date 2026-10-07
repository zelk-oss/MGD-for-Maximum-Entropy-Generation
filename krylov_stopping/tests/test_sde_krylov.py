"""Tests of codes/sde_krylov.py on a tiny Gaussian d = 4 problem (seconds, CPU). From the repo root:
    python krylov_stopping/tests/test_sde_krylov.py
1. ridge path (both solvers 'ridge') bit-identical to SDE: walkers, eta, theta, dH, moments;
2. per-sample mdot averages to the parent's mdot;
3. every Krylov step stops at its threshold or at max_iter; forward_regularised refused."""
import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'codes', ROOT / 'data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from codes.sde_routines import SDE                                  # noqa: E402
from codes.sde_krylov import SDEKrylov                              # noqa: E402
from lam_selection_criterion.gauss_d8_lamtest import Quadratic       # noqa: E402

D, N, NT, SIGMA = 4, 300, 40, 1.0
TMP = Path(tempfile.mkdtemp())


def make(cls, seed=0, **kw):
    torch.manual_seed(seed)
    x1 = torch.randn(N, 1, D) * torch.linspace(0.4, 1.0, D)
    t = np.linspace(0, 1, NT + 1)
    with contextlib.redirect_stdout(io.StringIO()):
        return cls(x1, N, N, t, SIGMA, {'quad': Quadratic(D)}, N, device='cpu',
                   regularization=kw.pop('regularization', 1e-4), interpolant='Cos',
                   potentials_save_dir=TMP, solve_float64=True, **kw)


def run(sde):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return sde.forward()


def test_ridge_bit_identical():
    for lam in (1e-4, 0.0):
        a, b = run(make(SDE, regularization=lam)), run(make(SDEKrylov, regularization=lam))
        names = ['x', 'barphi_e', 'barphi_p', 'eta', 'theta', 'dH']
        for n_, u, v in zip(names, a, b):
            assert torch.equal(u, v), (lam, n_, float((u - v).abs().max()))
        print(f'  regularization {lam:g}: SDEKrylov(ridge) == SDE bit for bit ({", ".join(names)})')


def test_mdot_samples():
    s = make(SDEKrylov, eta_solver='tol_minres')
    for k in (0, 7, NT - 1):
        I_k = s.compute_interpolant(k)
        ref = SDE.compute_rhs_dt_phi_I_t(s, I_k, k)
        ours = s.compute_rhs_dt_phi_I_t(I_k, k)
        assert s._mdot_samples[0] == k and s._mdot_samples[1].shape == (N, s.num_potentials)
        assert torch.allclose(ours, ref, rtol=1e-5, atol=1e-6), float((ours - ref).abs().max())
    print('  per-sample mdot: mean == parent mdot, (M, r) kept')


def test_krylov_stops():
    for th, eta, w in (('ito_cg', 'tol_minres', 'full'), ('tol_minres', 'tol_minres', 'diag'),
                       ('tol_minres', 'ridge', 'shrink')):
        s = make(SDEKrylov, theta_solver=th, eta_solver=eta, tol_weight=w, regularization=0.0)
        out = run(s)
        assert torch.isfinite(out[0]).all()
        L = s.krylov_log()
        for name in ('theta', 'eta'):
            T = L[name]
            c = {n_: i for i, n_ in enumerate(L[f'{name}_cols'])}
            kr = T[:, c['j']] >= 0
            if not kr.any():
                continue
            ok = (T[kr, c['stat']] <= T[kr, c['thr']]) | (T[kr, c['j']] == T[kr, c['r_live']])
            assert bool(ok.all()), (th, name)
            j = T[kr, c['j']]
            print(f'  {th}/{eta}/{w} {name}: {int(kr.sum())} steps, j mean {float(j.mean()):.1f}, '
                  f'j = 0 in {float((j == 0).double().mean()):.0%}, all stops valid')
    # 2026-10-07: Theta_reg is allowed on top of any theta solver (it only reads G(y_k), b_k, moments)
    s = make(SDEKrylov, theta_solver='jac_minres', eta_regularization=1e-7, g64=True, monitor=True,
             regularization=0.0)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        out = s.forward_regularised(lam=1e-3, reg_solver='thomas', reg_mode='moment')
    assert out[6] is not None and torch.isfinite(out[6]).all()
    mon = s.monitor_log()
    assert mon['step'].shape[0] == NT and torch.isfinite(mon['step'][:, 1]).all()
    j = mon['step'][:, 2]
    print(f'  jac_minres + Theta_reg: {NT} steps, j mean {float(j.mean()):.2f}, j=0 {float((j == 0).double().mean()):.0%}, '
          f'R_k median {float(mon["step"][:, 1].median()):.3g}, {mon["eig"].shape[0]} eig rows, '
          f'{mon["mis"].shape[0]} mismatch rows; Theta_reg finite')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            print(name)
            fn()
    print('ALL PASSED')
