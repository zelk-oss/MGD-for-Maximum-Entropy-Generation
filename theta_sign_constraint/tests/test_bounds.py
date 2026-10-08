"""Tests of the pinned/bounded Theta_reg solve (codes/sde_routines.py pin=, codes/theta_reg_bounds.py).

Plain asserts, run from the repo root:  python theta_sign_constraint/tests/test_bounds.py
Small synthetic moment-mode systems (n nodes, r statistics), solved four ways:
  1. pin=None vs the solver as committed in git HEAD: bit-identical.
  2. pinned values vs a dense solve with those entries eliminated.
  3. active-set bounded solve vs scipy L-BFGS-B on the dense energy with box bounds.
  4. KKT of the bounded solution (feasible; multipliers >= 0; complementarity).
"""
import importlib.util
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'codes', ROOT / 'data'):
    sys.path.insert(0, str(p))
from codes.sde_routines import SDE, reg_energy_weights          # noqa: E402
from codes.theta_reg_bounds import solve_upper_bounded            # noqa: E402

torch.set_default_dtype(torch.float64)


def make_system(n=40, r=6, seed=0, dead_frac=0.05, bounded=(0, 1, 2)):
    g = torch.Generator().manual_seed(seed)
    t = np.sort(np.concatenate([[0.0, 1.0], np.random.default_rng(seed).uniform(0.02, 0.98, n - 2)]))
    def spd():
        X = torch.randn(r, 2 * r, generator=g)
        return (X @ X.T / (2 * r) + 0.1 * torch.eye(r)).float()
    M = [spd() for _ in range(n)]
    Sigma = [spd() for _ in range(n)]
    b = [(torch.randn(r, generator=g) + 1.0).float() for _ in range(n)]   # +1: pushes Theta positive
    mdot = [torch.randn(r, generator=g).float() for _ in range(n)]
    live = torch.rand(n, r, generator=g) > dead_frac
    live[:, list(bounded)] = True
    return t, M, Sigma, b, mdot, live


def dense(t, M, C, b, v, lam, ridge, live):
    """A, f of the moment-mode energy, written out from its definition (not from the Thomas code)."""
    n, r = len(t), M[0].shape[0]
    a, w, l = reg_energy_weights(t, lam, mode='moment', schedule='uniform')
    sym = lambda X: (X.double() + X.double().T) / 2
    A = torch.zeros(n * r, n * r); f = torch.zeros(n * r)
    blk = lambda k: slice(k * r, (k + 1) * r)
    for k in range(n):
        Mk = sym(M[k]); Mk = Mk + ridge * torch.diag(torch.diagonal(Mk))
        A[blk(k), blk(k)] += a[k] * Mk
        f[blk(k)] += a[k] * b[k].double()
    for k in range(n - 1):
        Ck = (sym(C[k]) + sym(C[k + 1])) / 2
        ct = -(v[k].double() + v[k + 1].double()) / 2
        A[blk(k), blk(k)] += w[k] * Ck; A[blk(k + 1), blk(k + 1)] += w[k] * Ck
        A[blk(k), blk(k + 1)] -= w[k] * Ck; A[blk(k + 1), blk(k)] -= w[k] * Ck
        f[blk(k)] += l[k] * ct; f[blk(k + 1)] -= l[k] * ct
    return A, f


def dense_fixed(A, f, fixed, val):
    """Minimise 1/2 x A x - f x with x = val on the fixed entries."""
    x = torch.where(fixed, val, torch.zeros_like(val))
    fr = ~fixed
    rhs = f[fr] - A[fr][:, fixed] @ val[fixed]
    x[fr] = torch.linalg.solve(A[fr][:, fr], rhs)
    return x


def head_solver():
    src = subprocess.run(['git', 'show', 'HEAD:codes/sde_routines.py'], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    tmp = Path(tempfile.mkdtemp()) / 'sde_routines_head.py'
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location('sde_routines_head', tmp)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.SDE


def run(fn, *a, **k):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def main():
    lam, ridge = 1e-2, 1e-4
    t, M, S, b, v, live = make_system()
    n, r = len(t), M[0].shape[0]
    me = types.SimpleNamespace(num_potentials=r)

    # 1. pin=None: bit-identical to HEAD (with and without a mask)
    SDE_head = head_solver()
    for lv in (None, live):
        new = run(SDE._solve_regularised_thomas, me, t, M, S, b, v, lam, ridge=ridge, mode='moment', live=lv)
        old = run(SDE_head._solve_regularised_thomas, me, t, M, S, b, v, lam, ridge=ridge, mode='moment', live=lv)
        assert torch.equal(new, old), 'pin=None changed the result'
    print('1. pin=None bit-identical to HEAD: OK')

    A, f = dense(t, M, S, b, v, lam, ridge, live)
    dead = ~live.reshape(-1)

    # 2. pinned nonzero values + dead entries vs dense elimination
    g = torch.Generator().manual_seed(1)
    pm = (torch.rand(n, r, generator=g) < 0.2) & live
    pv = torch.randn(n, r, generator=g)
    Th = run(SDE._solve_regularised_thomas, me, t, M, S, b, v, lam, ridge=ridge, mode='moment', live=live, pin=(pm, pv))
    fixed = dead | pm.reshape(-1)
    ref = dense_fixed(A, f, fixed, torch.where(pm, pv, torch.zeros_like(pv)).reshape(-1))
    err = (Th.reshape(-1) - ref).abs().max() / ref.abs().max()
    assert err < 1e-9, f'pinned solve differs from dense: {err:.2e}'
    gref = (A @ ref - f).reshape(n, r)
    gerr = (me.last_pinned_grad - torch.where(pm, gref, torch.zeros_like(gref))).abs().max() / gref[pm].abs().max()
    assert gerr < 1e-8, f'pinned gradient differs: {gerr:.2e}'
    print(f'2. pinned values vs dense elimination: rel err {err:.1e}, gradient rel err {gerr:.1e}: OK')

    # 3./4. bounded: Theta[:, 0:3] <= -eps vs L-BFGS-B
    cols = [0, 1, 2]
    upper = -0.05 * torch.ones(n, len(cols))
    free = run(SDE._solve_regularised_thomas, me, t, M, S, b, v, lam, ridge=ridge, mode='moment', live=live)
    n_viol0 = int((free[:, cols] > upper).sum())
    assert n_viol0 > 0, 'test system has no violations: make b larger'
    Th, info = run(solve_upper_bounded, me, t, M, S, b, v, lam, cols, upper, ridge=ridge, live=live,
                   log=lambda m: None)
    assert info['converged'], info['history']
    ub = np.full(n * r, np.inf); ub.reshape(n, r)[:, cols] = upper.numpy()
    lb = np.full(n * r, -np.inf)
    lb[dead.numpy()] = 0; ub[dead.numpy()] = 0
    An, fn = A.numpy(), f.numpy()
    res = minimize(lambda x: (0.5 * x @ An @ x - fn @ x, An @ x - fn), np.zeros(n * r), jac=True,
                   method='L-BFGS-B', bounds=list(zip(lb, ub)), options=dict(maxiter=50000, ftol=1e-15, gtol=1e-12))
    xr = torch.as_tensor(res.x)
    e_ours = 0.5 * Th.reshape(-1) @ A @ Th.reshape(-1) - f @ Th.reshape(-1)
    e_ref = 0.5 * xr @ A @ xr - f @ xr
    dx = (Th.reshape(-1) - xr).abs().max() / xr.abs().max()
    assert e_ours <= e_ref + 1e-9 * abs(e_ref), (float(e_ours), float(e_ref))
    assert dx < 1e-4, f'bounded solution differs from L-BFGS-B by {dx:.2e}'
    print(f'3. bounded vs L-BFGS-B: {n_viol0} violations unconstrained, {info["iterations"]} solves, '
          f'{info["n_pinned"]} pinned; energy ours {float(e_ours):.10g} vs ref {float(e_ref):.10g}, '
          f'max rel diff {dx:.1e}: OK')

    gr = (A @ Th.reshape(-1) - f).reshape(n, r)
    tolv = 1e-9
    assert bool((Th[:, cols] <= upper + tolv).all()), 'infeasible'
    act = info['pinned']
    assert bool((gr[:, cols][act] <= 1e-8 * gr.abs().max()).all()), 'a pinned entry has the wrong multiplier sign'
    assert bool((gr[:, cols][~act].abs() <= 1e-8 * gr.abs().max()).all()), 'a free bounded entry is not stationary'
    print('4. KKT (feasible, multipliers >= 0, free entries stationary): OK')

    # 5. bounds that are never active: same as the free solve
    Th2, info2 = run(solve_upper_bounded, me, t, M, S, b, v, lam, cols, torch.full((n, 3), 1e6), ridge=ridge,
                     live=live, log=lambda m: None)
    assert info2['iterations'] == 1 and info2['n_pinned'] == 0 and torch.equal(Th2, free)
    print('5. inactive bounds: one solve, identical to the free solve: OK')
    print('ALL TESTS PASSED')


if __name__ == '__main__':
    main()
