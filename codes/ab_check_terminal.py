"""A/B + mechanics check of the terminal phase / final projection (branch terminal-phase-test, 2026-11-12).

Runs a tiny synthetic problem (scattering potentials, scale-mixture data, M = 64) through
  OLD = codes/sde_routines_before_terminal.py  (frozen copy of sde_routines.py at the tag before-terminal-phase)
  NEW = codes/sde_routines.py                  (with the opt-in terminal phase / final projection)
with the same seeds, in one process, and checks:
  T1  flags OFF: NEW is bit-identical to OLD (samples, moments, eta, theta, dH, Theta_reg, t_reg);
  T2  terminal phase ON but the criterion already met at the end of the planned grid (tol huge): bit-identical to OLD, info says so;
  T3  terminal phase forced to the step cap (tol tiny): the planned part (first n_planned rows) is bit-identical to OLD, the extra rows exist
      and are aligned (t, barphi, eta, theta, dH), h shrinks by the factor, Theta_reg/t_reg unchanged (terminal steps not in the system);
  T4  terminal phase with a reachable tolerance: shrink -> hold -> stop ('hold finished'), hold length exact;
  T5  terminal steps IN the system: runs, the system grid is longer;
  T6  final projection ON: all but the last row bit-identical to OLD, the maximum relative error does not increase.
Usage (a GPU is not needed but is used if present):  python codes/ab_check_terminal.py [NEW_MODULE]   -> prints PASS/FAIL per check, exit code 1 on any FAIL.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE, HERE / 'filters', HERE / 'potentials', ROOT / 'data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import importlib                                                        # noqa: E402
NEW_MODULE = sys.argv[1] if len(sys.argv) > 1 else 'sde_routines'     # e.g. sde_routines_new_candidate (identical copy, tested before the shared file is touched)
new_mod = importlib.import_module(NEW_MODULE)                          # noqa: E402
import sde_routines_before_terminal as old_mod                          # noqa: E402
from filters_bank import return_Filters                                 # noqa: E402
from potentials_builder import get_1d_potentials                        # noqa: E402
from time_schedules import two_phase_schedule                           # noqa: E402

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
M, J, N1, NT, NB, GAP = 64, 4, 300, 80, 30, 2e-3
RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append(bool(ok))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'   ({detail})' if detail else ''), flush=True)


def make_data():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(N1, 1, M, generator=g) * (0.5 + torch.rand(N1, 1, 1, generator=g))
    return ((x - x.mean()) / x.std()).to(dev)


def make_solver(mod, x1, tmpdir, **flags):
    filters, filters_phi = return_Filters(M, J, 1, include_phi=True)
    pots = get_1d_potentials(['L_6', 'Scattering_First_Order', 'Scattering_Second_Order'], J, filters, 1,
                             filters_Q=None, filters_Phi=filters_phi, scalar_param=None, parallel=False)
    t, _ = two_phase_schedule(NT, NB, 0.9, GAP)
    torch.manual_seed(1234); np.random.seed(1234)
    S = mod.SDE(x1, N1, N1, t, 3.5, pots, 300, device=dev, regularization=1e-4, interpolant='Cos',
                potentials_save_dir=tmpdir, solve_float64=True)
    for k, v in flags.items():
        setattr(S, k, v)
    return S


def run(mod, x1, **flags):
    with tempfile.TemporaryDirectory() as tmp:
        S = make_solver(mod, x1, tmp, **flags)
        torch.manual_seed(99)
        out = S.forward_regularised(lam=1e-2, n_subsample=1, reg_solver='thomas', reg_mode='moment', reg_schedule='uniform',
                                    solve_reg=True)
    return S, out


NAMES = ['xt', 'barphi_e', 'barphi_p', 'eta', 'theta', 'dH', 'Theta_reg', 't_reg']


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if torch.is_tensor(a):
        return a.shape == b.shape and torch.equal(a.detach().cpu(), b.detach().cpu())
    return np.array_equal(np.asarray(a), np.asarray(b))


def rows(a, n):
    return a[:n] if a is not None else None


def main():
    x1 = make_data()
    print(f'device {dev}, M {M}, n1 {N1}, nt {NT} (n_bulk {NB}, gap_end {GAP}), 3 potentials')
    S0, old = run(old_mod, x1)
    print('OLD run done; r =', S0.num_potentials)

    print('T1  flags off')
    S1, new = run(new_mod, x1)
    for nm, a, b in zip(NAMES, old, new):
        check(f'T1 {nm} identical', same(a, b))
    check('T1 time grid identical', same(S0.t, S1.t))

    print('T2  terminal phase on, criterion met at the end of the planned grid (tol 1e9)')
    S2, o2 = run(new_mod, x1, terminal_phase=True, terminal_tol=1e9)
    for nm, a, b in zip(NAMES, old, o2):
        check(f'T2 {nm} identical', same(a, b))
    check('T2 info', 'no terminal phase' in S2.terminal_info.get('reason', ''), S2.terminal_info.get('reason', ''))
    z_end = S2.terminal_info['z_end_planned']
    print(f'     max z at the end of the planned grid: {z_end:.3g}')

    print('T3  terminal phase forced to the step cap (tol 1e-12, shrink 0.5, cap 25)')
    S3, o3 = run(new_mod, x1, terminal_phase=True, terminal_tol=1e-12, terminal_shrink=0.5, terminal_hold=3, terminal_max_steps=25)
    n_term = len(S3.terminal_log)
    check('T3 25 terminal steps', n_term == 25, f'{n_term}')
    check('T3 reason', S3.terminal_info.get('reason') == 'terminal_max_steps reached', S3.terminal_info.get('reason', ''))
    check('T3 t length', len(S3.t) == NT + n_term + 1, f'{len(S3.t)} vs {NT + n_term + 1}')
    xt, be, bp, eta, th, dH, Th, tr = o3
    for nm, i in (('barphi_e', 1), ('barphi_p', 2), ('eta', 3), ('theta', 4), ('dH', 5)):
        check(f'T3 {nm} rows = steps', o3[i].shape[0] == NT + n_term, f'{o3[i].shape[0]}')
        check(f'T3 {nm} planned rows identical to OLD', same(rows(o3[i], NT), old[i]))
    check('T3 planned time grid identical', same(S3.t[:NT + 1], S0.t))
    check('T3 Theta_reg / t_reg unchanged (terminal steps not in the system)', same(Th, old[6]) and same(tr, old[7]))
    h_term = np.array([r[1] for r in S3.terminal_log])
    check('T3 h shrinks x0.5 per step', np.allclose(h_term[1:] / h_term[:-1], 0.5, rtol=1e-3), f'ratios {h_term[1:3] / h_term[:2]}')
    check('T3 first terminal h = 0.5 x last planned h', np.isclose(h_term[0], 0.5 * float(S0.t[-1] - S0.t[-2]), rtol=1e-9))
    tt = S3.t.numpy()
    check('T3 t strictly increasing and < 1', np.all(np.diff(tt) > 0) and tt[-1] < 1.0, f'1 - t_end = {1 - tt[-1]:.3e}')
    check('T3 samples differ from OLD (extra steps were taken)', not same(xt, old[0]))

    print('T4  reachable tolerance: shrink -> hold -> stop')
    tol4 = 0.5 * z_end
    S4, o4 = run(new_mod, x1, terminal_phase=True, terminal_tol=tol4, terminal_shrink=0.5, terminal_hold=4, terminal_max_steps=80)
    L4 = np.array(S4.terminal_log)
    check('T4 reason = hold finished', S4.terminal_info.get('reason') == 'hold finished', S4.terminal_info.get('reason', ''))
    n_hold = int(np.sum(np.isclose(L4[:, 1], L4[-1, 1], rtol=1e-3)))
    check('T4 h held for the last >= 4 steps', n_hold >= 4, f'{n_hold} steps at h = {L4[-1, 1]:.3e}')
    check('T4 max z at the end <= tol is what ended the shrink', L4[:, 2].min() <= tol4, f'min z {L4[:, 2].min():.3g} vs tol {tol4:.3g}')
    print('     ', S4.terminal_summary())

    print('T5  terminal steps in the regularised system')
    S5, o5 = run(new_mod, x1, terminal_phase=True, terminal_tol=1e-12, terminal_shrink=0.5, terminal_hold=3,
                 terminal_max_steps=10, terminal_in_system=True)
    check('T5 runs; system grid longer than the planned one', len(o5[7]) > len(old[7]), f'{len(o5[7])} vs {len(old[7])}')

    print('T6  final projection')
    S6, o6 = run(new_mod, x1, final_projection=True, final_projection_tol=1e-12, final_projection_max=3)
    for nm, i in (('barphi_e', 1), ('eta', 3), ('theta', 4), ('dH', 5)):
        check(f'T6 {nm} identical', same(o6[i], old[i]))
    check('T6 barphi_p identical except the last row', same(o6[2][:-1], old[2][:-1]))
    fp = S6.final_projection_log
    check('T6 log present, error not increased', fp is not None and fp[2] <= fp[1] * (1 + 1e-12), str(fp))

    n_ok, n = sum(RESULTS), len(RESULTS)
    print(f'\n{n_ok}/{n} checks passed')
    sys.exit(0 if n_ok == n else 1)


if __name__ == '__main__':
    main()
