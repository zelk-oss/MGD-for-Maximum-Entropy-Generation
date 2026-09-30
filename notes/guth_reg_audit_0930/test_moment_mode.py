"""Unit tests of the 2026-09-30 solver changes (codes/sde_routines.py, codes/resolve_theta_reg.py).

Run from the repo root:  python notes/guth_reg_audit_0930/test_moment_mode.py
(writes the HEAD version of sde_routines.py to a temporary file to compare against).

T1  legacy mode bit-identical to the solver of commit 4baf4c2 (random SPD systems,
    several lam / ridge).
T2  moment mode, CODE sign (theta_code = -theta_MGD), noise-free Gaussian inputs:
    X_t ~ N(0, v_t), phi = (x^2, x^4). Exact code-sign multipliers theta_c = (-1/(2v), 0),
    M = E[grad phi grad phi^T], b = M theta_c, Sigma = Cov(phi), mdot = d/dt E[phi].
    Sigma theta_c_dot = mdot holds exactly, so Theta must equal theta_c for EVERY lam,
    up to the O(dt^2) midpoint rule; both schedules. The opposite sign of the time term
    (target Sigma theta_dot = -mdot, the MGD-sign identity fed without the flip) must fail.
T3  masking: a potential dead at some nodes is pinned to 0 there; the others are unchanged
    when Sigma and M are diagonal.
T4  fd_derivative exact on quadratics on a non-uniform grid.
T5  Guth schedule: lam_eff = w_T / w_D = sin^2(pi t / 2) / (pi^2 d).
"""
import importlib.util
import io
import contextlib
import os
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path[:0] = [REPO, REPO + '/codes', REPO + '/data']
from codes.sde_routines import SDE, reg_energy_weights          # noqa: E402
from codes.resolve_theta_reg import fd_derivative               # noqa: E402

quiet = lambda: contextlib.redirect_stdout(io.StringIO())
ok_all = True


def report(name, ok, msg):
    global ok_all
    ok_all &= bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}")


# ---------------------------------------------------------------- T1
old_src = subprocess.run(['git', '-C', REPO, 'show', '4baf4c2:codes/sde_routines.py'],
                         capture_output=True, text=True, check=True).stdout
with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
    f.write(old_src)
spec = importlib.util.spec_from_file_location('old_sde_routines', f.name)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
os.unlink(f.name)

rng = np.random.default_rng(0)
worst = 0.0
for n, r in [(7, 3), (40, 5)]:
    t = np.sort(rng.uniform(0.05, 0.95, n))
    rnd = lambda: torch.tensor(rng.standard_normal((r + 3, r)), dtype=torch.float32)
    M = [(lambda A: A.T @ A)(rnd()) for _ in range(n)]
    G = [(lambda A: A.T @ A)(rnd()) for _ in range(n)]
    b = [torch.tensor(rng.standard_normal(r), dtype=torch.float32) for _ in range(n)]
    c = [torch.tensor(rng.standard_normal(r), dtype=torch.float32) for _ in range(n)]
    for lam in [0.0, 1e-4, 1.0]:
        for ridge in [0.0, 1e-3]:
            s1, s2 = types.SimpleNamespace(num_potentials=r), types.SimpleNamespace(num_potentials=r)
            with quiet():
                A = old.SDE._solve_regularised_thomas(s1, t, M, G, b, c, lam, ridge=ridge)
                B = SDE._solve_regularised_thomas(s2, t, M, G, b, c, lam, ridge=ridge)
            worst = max(worst, float((A - B).abs().max()))
report('T1 legacy == 4baf4c2', worst == 0.0, f'max |diff| = {worst:.1e} over 12 cases')

# ---------------------------------------------------------------- T2
s_data, adot = 0.4, np.pi / 2


def gauss_system(t):
    a = adot * t
    v = np.cos(a) ** 2 + s_data ** 2 * np.sin(a) ** 2
    vdot = adot * np.sin(2 * a) * (s_data ** 2 - 1)
    th = np.stack([-1 / (2 * v), 0 * v], 1)                      # code sign
    M, S, b, md = [], [], [], []
    for k in range(len(t)):
        vk = v[k]
        Mk = np.array([[4 * vk, 24 * vk ** 2], [24 * vk ** 2, 240 * vk ** 3]])
        Sk = np.array([[2 * vk ** 2, 12 * vk ** 3], [12 * vk ** 3, 96 * vk ** 4]])
        M.append(torch.tensor(Mk)); S.append(torch.tensor(Sk))
        b.append(torch.tensor(Mk @ th[k])); md.append(torch.tensor([vdot[k], 6 * vk * vdot[k]]))
    return M, S, b, md, th


for schedule in ['uniform', 'guth']:
    errs = []
    for n in [100, 400]:
        t = np.linspace(0.05, 0.95, n)
        M, S, b, md, th = gauss_system(t)
        e_n = []
        for lam in [0.0, 1e-4, 1e-2, 1.0, 1e2, 1e4]:
            s = types.SimpleNamespace(num_potentials=2)
            with quiet():
                Th = SDE._solve_regularised_thomas(s, t, M, S, b, md, lam, mode='moment',
                                                   schedule=schedule, dim=1).numpy()
            e_n.append(np.abs(Th[:, 0] / th[:, 0] - 1).max())
        errs.append(max(e_n))
    ratio = errs[0] / max(errs[1], 1e-300)
    report(f'T2 moment exact, schedule={schedule}', errs[1] < 1e-3,
           f'max rel err over lam in [0, 1e4]: {errs[0]:.2e} (n=100), {errs[1]:.2e} (n=400), '
           f'ratio {ratio:.1f} (O(dt^2) -> ~16)')

t = np.linspace(0.05, 0.95, 200)
M, S, b, md, th = gauss_system(t)
s = types.SimpleNamespace(num_potentials=2)
with quiet():
    Th_wrong = SDE._solve_regularised_thomas(s, t, M, S, b, [-x for x in md], 1e2, mode='moment').numpy()
err_wrong = np.abs(Th_wrong[:, 0] / th[:, 0] - 1).max()
report('T2 sign check (flipped mdot must fail)', err_wrong > 0.1, f'max rel err {err_wrong:.2f}')

# ---------------------------------------------------------------- T3
n, r = 50, 3
t = np.linspace(0.1, 0.9, n)
rng = np.random.default_rng(1)
Md = [torch.diag(torch.tensor(rng.uniform(1, 2, r))) for _ in range(n)]
Sd = [torch.diag(torch.tensor(rng.uniform(1, 2, r))) for _ in range(n)]
bd = [torch.tensor(rng.standard_normal(r)) for _ in range(n)]
vd = [torch.tensor(rng.standard_normal(r)) for _ in range(n)]
live = torch.ones(n, r, dtype=torch.bool)
live[10:20, 1] = False
with quiet():
    A = SDE._solve_regularised_thomas(types.SimpleNamespace(num_potentials=r), t, Md, Sd, bd, vd, 0.3,
                                      mode='moment', live=live)
    B = SDE._solve_regularised_thomas(types.SimpleNamespace(num_potentials=r), t, Md, Sd, bd, vd, 0.3,
                                      mode='moment')
report('T3 mask', bool((A[10:20, 1] == 0).all()) and float((A[:, [0, 2]] - B[:, [0, 2]]).abs().max()) < 1e-12,
       f'pinned entries max |.| {float(A[10:20, 1].abs().max()):.1e}; other potentials unchanged '
       f'(max |diff| {float((A[:, [0, 2]] - B[:, [0, 2]]).abs().max()):.1e}); '
       f'potential 1 away from the mask moved by {float((A[:, 1] - B[:, 1]).abs().max()):.2f}')

# ---------------------------------------------------------------- T4
tt = np.sort(np.random.default_rng(2).uniform(0, 1, 30))
y = np.stack([3 * tt ** 2 - tt + 2, -tt ** 2], 1)
dy = np.stack([6 * tt - 1, -2 * tt], 1)
report('T4 fd_derivative', np.abs(fd_derivative(tt, y) - dy).max() < 1e-8,
       f'max err on quadratics {np.abs(fd_derivative(tt, y) - dy).max():.1e}')

# ---------------------------------------------------------------- T5
t = np.linspace(0.05, 0.95, 50); dim = 256
a, q, l = reg_energy_weights(t, 1.0, mode='moment', dim=dim, schedule='guth')
h = np.diff(t)
# w_D at nodes = a / trapezoid weight; w_T at midpoints = l
from codes.sde_routines import trapezoid_node_weights  # noqa: E402
wD = a / trapezoid_node_weights(t)
wD_mid = np.interp((t[:-1] + t[1:]) / 2, t, wD)
lam_eff = l / (2 * adot * np.cos(adot * (t[:-1] + t[1:]) / 2) / (dim * np.sin(adot * (t[:-1] + t[1:]) / 2)))
target = np.sin(adot * (t[:-1] + t[1:]) / 2) ** 2 / (np.pi ** 2 * dim)
report('T5 guth lam_eff', np.abs(lam_eff / target - 1).max() < 1e-12 and np.allclose(q * h, l),
       f'max rel err vs sin^2/(pi^2 d): {np.abs(lam_eff / target - 1).max():.1e}')

print('\nALL PASS' if ok_all else '\nSOME TESTS FAILED')
