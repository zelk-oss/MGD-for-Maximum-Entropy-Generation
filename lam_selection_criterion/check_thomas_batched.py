"""thomas_batched vs the reference (SDE._solve_regularised_thomas through lam_select_cv, and
gauss_d8_lamtest.block_cv) on synthetic moment-mode systems, r = 4, two-phase grid, both
weightings. CPU, ~1 min. Run from the repo root:  python lam_selection_criterion/check_thomas_batched.py

Tolerances: Theta 1e-5, losses 1e-6, selections identical. Why not round-off: with uniform
weights the time coupling lam/dt reaches ~1e10 at the end of the two-phase grid, and the two
solvers (batched vs per-node LAPACK calls) round differently inside the elimination. Measured
2026-10-05: the difference grows with lam (1e-19, 8e-11, 6e-9, 7e-7 at lam = 0, 1e-3, 1, 100)
while both leave the same relative residual (3.4e-10 vs 3.6e-10 at lam = 100): equally good
solutions of an ill-conditioned system. The guth weights (weaker coupling) agree to ~1e-11.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'codes'), str(ROOT / 'data'), str(ROOT / 'lam_selection_criterion')]

from codes import lam_select_cv as lsc                     # noqa: E402
from codes.time_schedules import two_phase_schedule         # noqa: E402
import thomas_batched as tb                                 # noqa: E402
import gauss_d8_lamtest as g                                # noqa: E402

torch.manual_seed(0)
n_steps, r = 1500, 4
t_grid, _ = two_phase_schedule(n_steps, 250, 0.9, 5e-5)
t = t_grid[2:].numpy()                                       # nodes in (0, 1), as a saved system
n = len(t)
h = np.diff(t_grid.numpy())[1:]
A = torch.randn(n, r, r)
M = A @ A.transpose(1, 2) + 0.1 * torch.eye(r)
Bm = torch.randn(n, r, r)
C = Bm @ Bm.transpose(1, 2) + 0.1 * torch.eye(r)
b = torch.randn(n, r) / torch.as_tensor(h, dtype=torch.float32).sqrt()[:, None]   # noisier at small steps
v = torch.randn(n, r).double().cumsum(0) / 30                                      # smooth-ish mdot
lams = [0.0, 1e-3, 1.0, 100.0]
worst = {'Theta': 0.0, 'loss': 0.0, 'selection': True}


def rel(x, y):
    x, y = torch.as_tensor(np.asarray(x, dtype=np.float64)), torch.as_tensor(np.asarray(y, dtype=np.float64))
    return float((x - y).abs().max() / y.abs().max().clamp_min(1e-300))


for sched in ('uniform', 'guth'):
    inp = dict(t=t, r=r, mode='moment', M=list(M), C=list(C), b=list(b), v=list(v), ridge=1e-4,
               schedule=sched, dim=1, live=None, h=h, n_walkers=1000, sigma=10.0)
    S_ = tb.stack_inputs(inp)
    Th_b, _ = tb.solve_batched(S_, lams)
    e = max(rel(Th_b[0, i], lsc.solve(inp, lam)[0]) for i, lam in enumerate(lams))
    print(f'{sched:8s} full solve, 4 lams          max rel diff {e:.1e}')
    worst['Theta'] = max(worst['Theta'], e)

    even = np.arange(0, n, 2)
    The_b, _ = tb.solve_batched(S_, lams, idx=even)
    e = max(rel(The_b[0, i], lsc.solve(inp, lam, even)[0]) for i, lam in enumerate(lams))
    print(f'{sched:8s} even-node sub-grid solve     max rel diff {e:.1e}')
    worst['Theta'] = max(worst['Theta'], e)

    mask = ((np.arange(n) // 50) % 2 == 1).astype(float)
    Thm_b, _ = tb.solve_batched(S_, lams[1:], data_w=mask[None])
    inp_m = dict(inp)
    inp_m['M'] = [Mk * 0 if mask[k] == 0 else Mk for k, Mk in enumerate(inp['M'])]
    inp_m['b'] = [bk * 0 if mask[k] == 0 else bk for k, bk in enumerate(inp['b'])]
    e = max(rel(Thm_b[0, i], lsc.solve(inp_m, lam)[0]) for i, lam in enumerate(lams[1:]))
    print(f'{sched:8s} masked data (block CV fold)  max rel diff {e:.1e}')
    worst['Theta'] = max(worst['Theta'], e)

    p_ref = lsc.lam_path(inp, lams, verbose=False)
    p_b = tb.lam_path_batched(S_, lams)
    e = max(rel(p_b['cv_loss'], p_ref['cv_loss']), rel(p_b['amp_ratio'], p_ref['amp_ratio']),
            rel(p_b['chi2'], p_ref['chi2']))
    same = (p_b['lam_selected'] == p_ref['lam_selected']) and bool((p_b['veto'] == p_ref['veto']).all())
    print(f'{sched:8s} lam_path: cv/amp/chi2        max rel diff {e:.1e}, selection+veto identical: {same} '
          f'(selected {p_b["lam_selected"]})')
    worst['loss'] = max(worst['loss'], e)
    worst['selection'] &= same

    loss_ref, sel_ref = g.block_cv(inp, p_ref['lams'], 50, dict(zip(p_ref['lams'], p_ref['veto'])))
    loss_b, sel_b = tb.block_cv_batched(S_, p_b['lams'], 50, p_b['veto'])
    lr = np.array([loss_ref[l_] for l_ in p_ref['lams']])
    e = rel(loss_b[1:], lr[1:])
    print(f'{sched:8s} block CV (B = 50)            max rel diff {e:.1e}, same selection: {sel_b == sel_ref} '
          f'(selected {sel_b})')
    worst['loss'] = max(worst['loss'], e)
    worst['selection'] &= sel_b == sel_ref

ok = worst['Theta'] < 1e-5 and worst['loss'] < 1e-6 and worst['selection']
print(f"\nworst Theta {worst['Theta']:.1e} (tol 1e-5), loss {worst['loss']:.1e} (tol 1e-6), "
      f"selections identical {worst['selection']} -> {'PASS' if ok else 'FAIL'}")
