"""Cost of adding the Scat4 diagonal (Mod2_Real_Q1 -> Mod2_Real_Q1_diag) to the 1D turbulence (B+Scat4) and jets
(jets_full) models, 2026-11-17.

Times the potential-dependent work of ONE MGD step at production size (N walkers in batches, H100), as in
SDE.iteration_step_projection + forward_regularised:
  - full Jacobian at x_k and y_k, each accumulated into G (bmm), and at I_{k+1} contracted with Idot (moment velocity);
  - projected field sum_k v_k grad phi_k at x_k (drift) and y_k (corrector);
  - moments (forward) at I_{k+1}, y_k, x_{k+1};
  - two float64 r x r solves (Jacobi-scaled G + ridge).
Not included (same for both variants, or not potential-dependent): interpolant, noise, eigh of the conditioning log,
regularised-system bookkeeping. Real GGD fits (turbulence seed 900, jets_full seed 500), random walkers.

Usage (repo root, JZ dev GPU job): python -u grad_speed/bench_diag.py [--B_time 5000 --n_rep 3]
"""
import argparse
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_ggd_prealloc import (ROOT, RUN, M, J, Q, return_Filters, get_1d_potentials,  # noqa: E402
                                Scalar_GGD_KRegion, SDE)

JETS_RUN = ('jets/experiments/jetsynth_modane_M256_J8_Q1_sigma3.5_nt40000_n1_10000_lam0.0_spread_thomas_nsub1_moment_'
            'f64_dedupfilt1e-08_terms939b4857_jets_full/jetsynth_modane_M256_J8_Q1_sigma3.5_nt40000_n1_10000_lam0.0_'
            'spread_thomas_nsub1_moment_f64_dedupfilt1e-08_seed_500_terms939b4857_jets_full_20261028_1405/'
            'fitted_potentials')
SCAT4 = ['Scattering_Fourth_Order_Mod2_Real_Q1', 'Scattering_Fourth_Order_Mod2_Imag_Q1']
TURB = ['L_6', 'Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK']
JETS = ['L_6', 'Scattering_First_Order', 'Scattering_Second_Order', 'Scattering_Third_Order_Real',
        'Scattering_Third_Order_Imag', 'Scalar_morlet_gaussianK']


def build(model, diag, device):
    terms = (TURB if model == 'turbulence' else JETS) + SCAT4
    if diag:
        terms = [t + '_diag' if t == 'Scattering_Fourth_Order_Mod2_Real_Q1' else t for t in terms]
    filters = return_Filters(M, J, 1, device=device)
    if model == 'turbulence':
        pots = get_1d_potentials(terms, J, filters, Q, filters_Q=return_Filters(M, J, Q, device=device),
                                 deduplicate_filters=True, dedup_tol=1e-8)
        fits = {n: ROOT / RUN / f'{n}.pt' for n in ('Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK')}
    else:
        pots = get_1d_potentials(terms, J, filters, 1, filters_Q=filters)
        fits = {'Scalar_morlet_gaussianK': ROOT / JETS_RUN / 'Scalar_morlet_gaussianK.pt'}
    for n, path in fits.items():
        pots[n] = Scalar_GGD_KRegion.load_fixed_parameters(path, pots[n].filters, map_location=device).to(device)
    sizes = [int(p.num_coefficients) for p in pots.values()]
    return SimpleNamespace(potentials=pots, num_potentials=sum(sizes), signal_dim=1,
                           indices_potentials=[0] + torch.tensor(sizes).cumsum(0).tolist())


def one_step(ns, X, batch, ridge=1e-4):
    x, y, I, Idot, x1 = X
    r = ns.num_potentials
    chunks = lambda T: [T[i:i + batch] for i in range(0, T.shape[0], batch)]
    Gs = []
    for W in (x, y):                                                     # G(x_k), G(y_k)
        G = torch.zeros(r, r, device=x.device)
        for b in chunks(W):
            g = SDE.compute_grad_potentials(ns, b)
            G += torch.bmm(g, g.transpose(1, 2)).sum(0)
        Gs.append(G / W.shape[0])
    rhs = torch.zeros(r, 1, device=x.device)                             # moment velocity at I_{k+1}
    for b, bd in zip(chunks(I), chunks(Idot)):
        rhs += torch.matmul(SDE.compute_grad_potentials(ns, b), bd.reshape(b.shape[0], M, 1)).sum(0)
    vec = torch.ones(r, device=x.device) * 1e-3
    for W in (x, y):                                                     # drift, corrector
        for b in chunks(W):
            SDE.compute_grad_potentials(ns, b, vec)
    for W in (I, y, x1):                                                 # moments
        for b in chunks(W):
            torch.cat([p.forward(b) for p in ns.potentials.values()], 1)
    for G in Gs:                                                         # two float64 solves
        G = G.double()
        d = torch.diag(G).clamp_min(1e-30).sqrt()
        Gn = G / (d[:, None] * d[None, :])
        Gn = (Gn + Gn.T) / 2 + ridge * torch.eye(r, dtype=G.dtype, device=G.device)
        torch.linalg.solve(Gn, rhs.double() / d[:, None])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--B_time', type=int, default=5000, help='batch size (production 5000)')
    p.add_argument('--N', type=int, default=10000, help='walkers (production 10000)')
    p.add_argument('--n_rep', type=int, default=3)
    p.add_argument('--nt', type=int, default=60000, help='steps of a run, for the per-run extrapolation')
    p.add_argument('--B', type=int, default=0)                          # accepted from the launcher, unused
    p.add_argument('--M', type=int, default=M)
    p.add_argument('--J', type=int, default=J)
    args = p.parse_args()
    device = 'cuda'
    torch.manual_seed(0)
    print(f'GPU {torch.cuda.get_device_name()}; torch {torch.__version__}; N {args.N}, batch {args.B_time}')
    X = [torch.randn(args.N, 1, M, device=device) for _ in range(5)]
    for model in ('turbulence', 'jets'):
        res = {}
        for diag in (False, True):
            ns = build(model, diag, device)
            fn = lambda: one_step(ns, X, args.B_time)
            fn()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ts = []
            for _ in range(args.n_rep):
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            res[diag] = (ns.num_potentials, statistics.median(ts), torch.cuda.max_memory_allocated() / 2**30)
            del ns
            torch.cuda.empty_cache()
        (r0, t0, m0), (r1, t1, m1) = res[False], res[True]
        print(f'{model:10s} no diagonal: r {r0}, {1e3 * t0:7.1f} ms/step, peak {m0:.1f} GiB | with diagonal: r {r1}, '
              f'{1e3 * t1:7.1f} ms/step, peak {m1:.1f} GiB | +{100 * (t1 / t0 - 1):.1f}% '
              f'= +{(t1 - t0) * args.nt / 60:.0f} min per {args.nt} steps (potential work only)')


if __name__ == '__main__':
    main()
