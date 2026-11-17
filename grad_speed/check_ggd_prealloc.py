"""Check of two exact speed-ups (2026-11-17), on the B+Scat4 model with its real GGD fits (seed 900):

1. Scalar_GGD_KRegion.grad: pull-back (one FFT pair per slot) for the active slots only, instead of all candidate
   slots then index_select. Compared with the old code (verbatim below): grad(x) and the projected grad(x, v).
2. SDE.compute_grad_potentials: one preallocated (B, r, M) output instead of torch.cat of the growing result.
   Compared with the old code (verbatim below) on the full B+Scat4 potential set: the whole (B, r, M) tensor.

Both must be bit-identical (torch.equal). Then wall time old vs new at the production batch (B_TIME walkers).

Usage (repo root, JZ dev GPU job): python -u grad_speed/check_ggd_prealloc.py [--B 64 --B_time 5000]
"""
import argparse
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'codes'))
from filters_bank import return_Filters                                   # noqa: E402
from potentials_builder import get_1d_potentials                         # noqa: E402
from potentials.potentials_1d import Scalar_GGD_KRegion                  # noqa: E402
from codes.sde_routines import SDE                                       # noqa: E402

RUN = ('turbulence/experiments/turbulencesynth_M256_J8_Q3_sigma3.5_nt55000_n1_10000_lam0.0_thomas_nsub1_moment_f64_'
       'dedupfilt1e-08_terms9296e719_turbulence_nested_M3_bicw_allmass_scat4/turbulencesynth_M256_J8_Q3_sigma3.5_'
       'nt55000_n1_10000_lam0.0_thomas_nsub1_moment_f64_dedupfilt1e-08_seed_900_terms9296e719_turbulence_nested_M3_'
       'bicw_allmass_scat4_20261101_1714/fitted_potentials')
TERMS = ['L_6', 'Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK',
         'Scattering_Fourth_Order_Mod2_Real_Q1', 'Scattering_Fourth_Order_Mod2_Imag_Q1']
M, J, Q = 256, 8, 3


# ---------------------------------------------------------------- old code (verbatim, commit b50b9bc)

def old_ggd_grad(self, x, v=None):
    gc = self._all_slot_grad(x).index_select(1, self.active_flat.to(x.device))
    if self.stat_scale is not None:
        gc = gc / self.stat_scale.to(x.device)[None, :, None]
    if v is None:
        return gc
    return (gc * v[None, :, None]).sum(1, keepdim=True)


def old_compute_grad_potentials(self, x):
    grad_potential = torch.tensor([], device=x.device)
    for potential in self.potentials.values():
        grad_potential = torch.cat((grad_potential, potential.grad(x)), dim=1).detach()
    return grad_potential.reshape(x.shape[0], self.num_potentials, x.shape[-1])


# ---------------------------------------------------------------- helpers

def build(device):
    filters = return_Filters(M, J, 1, device=device)
    filters_Q = return_Filters(M, J, Q, device=device)
    pots = get_1d_potentials(TERMS, J, filters, Q, filters_Q=filters_Q,
                             deduplicate_filters=True, dedup_tol=1e-8)
    for name in ('Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK'):
        pots[name] = Scalar_GGD_KRegion.load_fixed_parameters(ROOT / RUN / f'{name}.pt', pots[name].filters,
                                                              map_location=device).to(device)
    return pots


def timed(fn, n_rep):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--B', type=int, default=64)
    p.add_argument('--B_time', type=int, default=5000)
    p.add_argument('--n_rep', type=int, default=5)
    p.add_argument('--M', type=int, default=M)          # passed by the launcher; the saved fits are M 256, J 8
    p.add_argument('--J', type=int, default=J)
    args = p.parse_args()
    if (args.M, args.J) != (M, J):
        raise ValueError(f'the seed-900 fits are for M {M}, J {J}; got M {args.M}, J {args.J}')
    device = 'cuda'
    torch.manual_seed(0)
    print(f'GPU {torch.cuda.get_device_name()}; torch {torch.__version__}')
    pots = build(device)
    r = sum(int(pt.num_coefficients) for pt in pots.values())
    print('potentials:', {k: int(v.num_coefficients) for k, v in pots.items()}, f'r = {r}')
    ns = SimpleNamespace(potentials=pots, num_potentials=r, signal_dim=1)
    x = torch.randn(args.B, 1, M, device=device)
    ok = True

    for name in ('Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK'):
        pot = pots[name]
        n_all = pot.n_slots * pot.J
        v = torch.randn(pot.num_coefficients, device=device)
        same_g = torch.equal(pot.grad(x), old_ggd_grad(pot, x))
        same_v = torch.equal(pot.grad(x, v), old_ggd_grad(pot, x, v))
        ok &= same_g and same_v
        print(f'{"PASS" if same_g and same_v else "FAIL"} {name}: {pot.num_coefficients} active of {n_all} slots | '
              f'grad(x) identical {same_g} | grad(x, v) identical {same_v}')

    g_new = SDE.compute_grad_potentials(ns, x)
    g_old = old_compute_grad_potentials(ns, x)
    same = g_new.shape == g_old.shape and g_new.dtype == g_old.dtype and torch.equal(g_new, g_old)
    ok &= same
    print(f'{"PASS" if same else "FAIL"} compute_grad_potentials (full B+Scat4 set): shape {tuple(g_new.shape)}, '
          f'dtype {g_new.dtype}, identical {same}')
    del g_new, g_old

    xt = torch.randn(args.B_time, 1, M, device=device)
    print(f'\ntiming, float32, B {args.B_time}, median of {args.n_rep}:')
    for name in ('Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK'):
        pot = pots[name]
        t_old = timed(lambda: old_ggd_grad(pot, xt), args.n_rep)
        t_new = timed(lambda: pot.grad(xt), args.n_rep)
        print(f'  {name:32s} grad      old {1e3 * t_old:8.1f} ms | new {1e3 * t_new:8.1f} ms | ratio {t_old / t_new:5.2f}')
    torch.cuda.reset_peak_memory_stats()
    t_old = timed(lambda: old_compute_grad_potentials(ns, xt), args.n_rep)
    mem_old = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    t_new = timed(lambda: SDE.compute_grad_potentials(ns, xt), args.n_rep)
    mem_new = torch.cuda.max_memory_allocated() / 2**30
    print(f'  {"compute_grad_potentials (all)":32s} full Jac. old {1e3 * t_old:8.1f} ms | new {1e3 * t_new:8.1f} ms | '
          f'ratio {t_old / t_new:5.2f} | peak GPU memory old {mem_old:.2f} GiB, new {mem_new:.2f} GiB')
    print('  (new GGD inside both: the compute_grad_potentials line measures the preallocation alone)')
    print('\nALL PASS' if ok else '\nSOME FAILED')


if __name__ == '__main__':
    main()
