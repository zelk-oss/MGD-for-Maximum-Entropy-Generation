#!/usr/bin/env python3
"""Exactness check of the whitened potentials (codes/whitening.py) on the B+Scat4 model, 2026-11-13.

With y = C^(-alpha/2) x and phi~ = phi o C^(alpha/2), for every potential and every statistic:
  (1) phi~(y) = phi(x)
  (2) grad_y (v . phi~)(y) = C^(alpha/2) grad_x (v . phi)(x)        (C circulant symmetric, chain rule)
Prints the max relative error per potential (float32 GPU arithmetic: expect ~1e-5).
Negative control: Scat4 with BOTH layers whitened (the wrong implementation) -> (1) must fail by O(1).

Usage (JZ, repo root): python whitening_test/check_whiten_potentials.py --potentials_from <dir> [--alpha 0.75 --n 200]
"""
import argparse, copy, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / 'codes'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from data.data_loader import load_turbulence_1d                                  # noqa: E402
from codes.utils import split_periodize_reshape, normalize                       # noqa: E402
from codes.ortho_wavelet.ReadyToUseWavelets import DefineWavelet                 # noqa: E402
from codes.filters_bank import return_Filters                                    # noqa: E402
from codes.potentials_builder import get_1d_potentials                           # noqa: E402
from codes.gauss_spectrum import average_spectrum, whitening_multiplier, apply_multiplier  # noqa: E402
from codes.whitening import whiten_potentials, x_filter_multiplier, _times        # noqa: E402

TERMS = ['L_6', 'Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK',
         'Scattering_Fourth_Order_Mod2_Real_Q1', 'Scattering_Fourth_Order_Mod2_Imag_Q1']


def build(device, src, M=256, J=8, Q=3):
    filters, filters_Phi = return_Filters(M, J, 1, device=device, include_phi=True)
    filters_Q = return_Filters(M, J, Q, device=device)
    pots = get_1d_potentials(TERMS, J, filters, Q, filters_Q=filters_Q, filters_Phi=filters_Phi,
                             scalar_param=None, parallel=False, deduplicate_filters=True, dedup_tol=1e-8,
                             ggd_kwargs=dict(tail_quantile=[0.99], tail_mass=True, all_masses=True))
    return pots


def load_states(pots, src, device):
    for name, pot in list(pots.items()):
        f = Path(src) / f'{name}.pt'
        if hasattr(pot, 'load_fixed_parameters') and f.exists():
            pots[name] = type(pot).load_fixed_parameters(f, pot.filters, map_location=device)
            pots[name].to(device)
    return pots


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--potentials_from', required=True)
    ap.add_argument('--alpha', type=float, default=0.75)
    ap.add_argument('--n', type=int, default=200)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    data = split_periodize_reshape(load_turbulence_1d(), 512)
    data = DefineWavelet('Db', m=3, device='cpu').decompose(data)[1]
    x = normalize(data[:args.n]).to(dev)                                         # (n, 1, 256), as run_SDE
    S = average_spectrum(x).cpu()
    mult = x_filter_multiplier(S, args.alpha)
    y = apply_multiplier(x, whitening_multiplier(S, args.alpha))
    print(f'alpha {args.alpha}: x var {x.var().item():.4f}, y var {y.var().item():.4f}, '
          f'kappa x {(S.max() / S.min()).item():.2e} -> y {((S / 256 * whitening_multiplier(S, args.alpha) ** 2).max() / (S / 256 * whitening_multiplier(S, args.alpha) ** 2).min()).item():.2e}')

    P = load_states(build(dev, args.potentials_from), args.potentials_from, dev)            # model on x
    Pw = build(dev, args.potentials_from)
    whiten_potentials(Pw, mult)
    Pw = load_states(Pw, args.potentials_from, dev)                                             # model on y
    ok = True
    for name in TERMS:
        f_x, f_y = P[name].forward(x), Pw[name].forward(y)
        v = torch.randn(f_x.shape[1], device=dev, dtype=f_x.dtype)
        g_x = P[name].grad(x, v)                                                 # (n, 1, T)
        g_y = Pw[name].grad(y, v)
        g_x_to_y = apply_multiplier(g_x.real if g_x.is_complex() else g_x, mult.to(dev))   # C^(alpha/2) grad_x
        e1, e2 = rel(f_y, f_x), rel(g_y, g_x_to_y)
        flag = 'OK' if (e1 < 1e-3 and e2 < 1e-3) else 'FAIL'
        ok &= flag == 'OK'
        print(f'{name:40s} stats {f_x.shape[1]:4d} | (1) phi~(y) vs phi(x): {e1:.2e} | '
              f'(2) grad_y vs C^(a/2) grad_x: {e2:.2e}  {flag}')

    # negative control: both Scat4 layers whitened (wrong)
    for name in ('Scattering_Fourth_Order_Mod2_Real_Q1', 'Scattering_Fourth_Order_Mod2_Imag_Q1'):
        bad = copy.deepcopy(P[name])
        bad.filters_Q = _times(bad.filters_Q, mult)
        bad.filters = _times(bad.filters, mult)
        print(f'NEGATIVE CONTROL {name}: both layers whitened -> (1) error {rel(bad.forward(y), P[name].forward(x)):.2e} '
              f'(must be O(1))')
    print('ALL OK' if ok else 'SOME CHECK FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
