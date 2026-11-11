#!/usr/bin/env python3
"""Regression check of method A (c_floor_eps, 2026-11-11): with eps = 0 the Theta_reg re-solve must equal the one
saved before the change.

For one saved system and the directory of its earlier resolve_theta_reg.py outputs, re-solves each lam with
SDE._solve_regularised_thomas(..., c_floor_eps=0.0) using the same inputs as resolve_theta_reg.py (mask, ridge,
schedule) and compares with the saved Theta_reg: prints torch.equal and the max |difference| / max |Theta|.
Also solves once WITHOUT the c_floor_eps keyword in the same process (must be torch.equal to eps = 0).
Bit-identity with a file written on another node / thread count is not guaranteed by BLAS; a difference at the
1e-15 relative level is rounding, anything larger is a regression.

Usage: python codes/check_c_floor_eps0.py <system.pt> <saved theta dir> --lams 1e-4 1e-2 1 --ridge 1e-7
"""
import argparse
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'codes'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from codes.sde_routines import SDE                       # noqa: E402
from codes.resolve_theta_reg import build_inputs, out_path   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('system', type=Path)
    ap.add_argument('saved_dir', type=Path, help='<outdir> of the earlier resolve_theta_reg.py run (contains <config>/)')
    ap.add_argument('--lams', nargs='+', type=float, required=True)
    ap.add_argument('--ridge', type=float, default=1e-7)
    ap.add_argument('--schedule', default='uniform')
    a = ap.parse_args()
    config = a.system.name[:-3]
    system = torch.load(a.system, map_location='cpu', weights_only=False, mmap=True)
    args = types.SimpleNamespace(schedule=a.schedule, ridge=a.ridge, dim=None, n_walkers=None, sigma=None,
                                 mask='auto', live_floor=0.0, results_root=None)
    inp = build_inputs(system, config, args, 'moment')
    me = types.SimpleNamespace(num_potentials=inp['r'])
    call = lambda lam, **kw: SDE._solve_regularised_thomas(me, inp['t'], inp['M'], inp['C'], inp['b'], inp['v'], lam,
                                                           ridge=a.ridge, mode='moment', dim=inp['dim'],
                                                           schedule=a.schedule, live=inp['live'], **kw)
    ok_all = True
    for i, lam in enumerate(a.lams):
        th0 = call(lam, c_floor_eps=0.0)
        if i == 0:
            thk = call(lam)
            same = torch.equal(th0, thk)
            ok_all &= same
            print(f'lam={lam:g}: eps=0 vs no keyword (same process): torch.equal {same}')
        f = out_path(a.saved_dir, config, lam, a.ridge, 'moment', a.schedule, inp['mask'])
        if not f.exists():
            print(f'lam={lam:g}: no saved file {f}'); continue
        sv = torch.load(f, map_location='cpu', weights_only=False)
        old = sv['Theta_reg'].to(th0.dtype)
        new = th0[1:]
        D = (new - old).abs()
        d = float(D.max()) / max(float(old.abs().max()), 1e-300)
        eq = torch.equal(new, old)
        ok_all &= (d < 1e-12)
        kk, ii = divmod(int(D.argmax()), D.shape[1])
        tr = sv['t_reg'].double().numpy()
        print(f'lam={lam:g}: vs saved {f.name}: torch.equal {eq}, max|diff|/max|Theta| {d:.3e} '
              f'| at node {kk} (t={tr[kk]:.6f}) stat {ii}: new {float(new[kk, ii]):.4g} old {float(old[kk, ii]):.4g} '
              f'| saved residual {sv.get("residual")}, now {me.last_reg_residual:.2e} '
              f'| Frobenius rel {float((new - old).norm() / old.norm()):.3e}')
        # same solve with a different thread count: if it also moves by percents, the saved files cannot be
        # reproduced bit-for-bit and the difference is the conditioning of the system, not the code change
        nthr = torch.get_num_threads()
        torch.set_num_threads(1)
        th1 = call(lam, c_floor_eps=0.0)
        torch.set_num_threads(nthr)
        d1 = float((th1 - th0).abs().max()) / max(float(th0.abs().max()), 1e-300)
        print(f'lam={lam:g}: 1 thread vs {nthr} threads (same code, same process): max|diff|/max|Theta| {d1:.3e}, '
              f'Frobenius rel {float((th1 - th0).norm() / th0.norm()):.3e}')
    print('REGRESSION CHECK ' + ('PASSED' if ok_all else 'FAILED (see the thread test: equal sensitivity = conditioning, not the change)'))


if __name__ == '__main__':
    main()
