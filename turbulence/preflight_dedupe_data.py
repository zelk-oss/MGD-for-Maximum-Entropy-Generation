#!/usr/bin/env python3
"""Offline data-Gram dedupe check (CPU-friendly, no SDE run).

Computes G^data = E_x[<grad phi_i(x), grad phi_j(x)>] on the turbulence data
for a given config, Jacobi-scales it, and reports:
  - NEAR-COPY PAIRS: |corr(i,j)| > 1 - tol
  - NULL-SPACE DIRECTIONS: eigenvalue mu < null_tol, with top contributors

Same tolerance philosophy as prune_collinear (cond_tol = 1e-6 by default) but
cross-family. No new per-step mechanism anywhere -- output is a drop list
compatible with the existing --drop_stats CLI flag and --terms omission.

Usage:
    python turbulence/preflight_dedupe_data.py [--target_len 256 --n1 500 ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'data'):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from codes.utils import split_periodize_reshape, normalize            # noqa: E402
from codes.filters_bank import return_Filters                          # noqa: E402
from codes.potentials_builder import get_1d_potentials                 # noqa: E402
from codes.ortho_wavelet.ReadyToUseWavelets import DefineWavelet       # noqa: E402
from codes.sde_routines import SDE                                     # noqa: E402
from data.data_loader import load_turbulence_1d                        # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--target_len', type=int, default=256,
                    help='M after coarse-graining (default 256 = production)')
    ap.add_argument('--subseries_len', type=int, default=512,
                    help='window length cut from raw data (default 512 = production)')
    ap.add_argument('--J', type=int, default=8,
                    help='wavelet scales (default 8 = production)')
    ap.add_argument('--Q', type=int, default=3,
                    help='wavelets per octave (default 3 = production)')
    ap.add_argument('--n1', type=int, default=10000,
                    help='number of windows, data[:n1] as in run_SDE without --n1_spread '
                         '(default 10000 = production; GGD region cuts depend on it)')
    ap.add_argument('--dedup_tol', type=float, default=1e-8,
                    help='filter dedupe (matches production --dedup_tol)')
    ap.add_argument('--tol', type=float, default=1e-3,
                    help='pair detection: |corr(i,j)| > 1 - tol')
    ap.add_argument('--null_tol', type=float, default=1e-6,
                    help='null-space eigenvalue cutoff '
                         '(matches prune_collinear cond_tol = 1e-6)')
    ap.add_argument('--terms', nargs='+', default=[
        'L_6', 'L_2_lowpass',
        'Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK',
        'Scattering_Fourth_Order_Mod2_Real_Q1',
        'Scattering_Fourth_Order_Mod2_Imag_Q1'])
    ap.add_argument('--potentials_dir', type=Path, default=None,
                    help='where SDE saves/loads fitted potential state '
                         '(default: /tmp/preflight_potentials_M<target_len>_n<n1>)')
    ap.add_argument('--drop_stats', nargs='*', default=[],
                    help='NAME:CHANNEL:REGION stats to drop after fit '
                         '(same syntax as run_SDE.py)')
    ap.add_argument('--batch_size', type=int, default=200)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = 'cpu'

    # The fit cache is reused whenever <name>.pt exists, so the default dir must
    # encode n1 and the drops: otherwise an old n1=500 fit would be loaded silently.
    if args.potentials_dir is None:
        tag = '_'.join(s.replace(':', '-') for s in args.drop_stats)
        args.potentials_dir = Path(f'/tmp/preflight_potentials_M{args.target_len}_n{args.n1}'
                                   + (f'_drop_{tag}' if tag else ''))
    args.potentials_dir.mkdir(parents=True, exist_ok=True)
    print(f'[cache] fitted potentials in {args.potentials_dir}')

    # ---- data ----
    W = DefineWavelet('Db', m=3, device=device)
    raw = load_turbulence_1d().to(device)
    data = split_periodize_reshape(raw, args.subseries_len)
    pre_len = data.shape[-1]
    if pre_len > args.target_len:
        scales = int(np.log2(pre_len / args.target_len))
        for _ in range(scales):
            data = W.decompose(data)[1]
    else:
        scales = 0
    x1 = normalize(data[:args.n1]).to(device)
    M = x1.shape[-1]
    print(f'[data] x1 shape {tuple(x1.shape)}, M={M} after {scales}x coarse-grain')

    # ---- potentials ----
    filters, filters_Phi = return_Filters(M, args.J, 1, device=device, include_phi=True)
    filters_Q = return_Filters(M, args.J, args.Q, device=device)
    potentials = get_1d_potentials(
        args.terms, args.J, filters, args.Q,
        filters_Q=filters_Q, filters_Phi=filters_Phi,
        scalar_param=None, parallel=False,
        deduplicate_filters=True, dedup_tol=args.dedup_tol,
    )
    # Mirror run_experiment's --drop_stats handling (utils_experiment.py:363-368):
    # append (channel, region) tuples to drop_slots on the matching potential,
    # BEFORE the SDE's init-time .fit() is called (so the drops take effect).
    for spec in args.drop_stats:
        name, ch, reg = spec.split(':')
        if not hasattr(potentials.get(name), 'drop_slots'):
            raise ValueError(f'--drop_stats {spec}: {name} is not a region potential')
        potentials[name].drop_slots = tuple(potentials[name].drop_slots) + ((int(ch), int(reg)),)
        print(f'[drop] queued {name} ch={ch} reg={reg}')

    # ---- minimal SDE just for compute_G (and the fitting it drives) ----
    t = torch.linspace(0, 1, 2).to(device)
    sde = SDE(x1, x1.shape[0], x1.shape[0], t, 1.0, potentials, args.batch_size,
              device=device, potentials_save_dir=args.potentials_dir)
    print(f'[Gram] computing on {x1.shape[0]} samples ...')
    G = sde.compute_G(x1).double()
    r = G.shape[0]
    print(f'[Gram] shape {tuple(G.shape)}')

    # ---- labels after fitting (potential sizes may have been pruned) ----
    # For Scalar_GGD_KRegion: self.K = regions, self.J = channels (yes, named
    # this way in the class). active_flat stores region*J_channels + channel.
    def _label(name, p, local_idx):
        if hasattr(p, 'active_flat') and hasattr(p, 'J') and hasattr(p, 'K'):
            flat = int(p.active_flat[local_idx])
            region, channel = flat // p.J, flat % p.J
            return f'{name}[{local_idx}](ch={channel},reg={region})'
        return f'{name}[{local_idx}]'
    labels = [_label(name, p, i) for name, p in sde.potentials.items()
              for i in range(p.num_coefficients)]
    assert len(labels) == r, f'label count {len(labels)} != r {r}'
    print('[layout] ' + ', '.join(f'{n}:{p.num_coefficients}'
                                  for n, p in sde.potentials.items()))

    # ---- Jacobi scale + eigendecompose ----
    G = (G + G.T) / 2
    d = G.diagonal().clamp_min(1e-30).sqrt()
    A = G / (d[:, None] * d[None, :])
    mu, U = torch.linalg.eigh(A)
    print(f'[eigs] min {float(mu[0]):.3e}  max {float(mu[-1]):.3e}  '
          f'cond {float(mu[-1] / mu[0].clamp_min(1e-30)):.3e}')

    # ---- near-copy pairs ----
    print(f'\n==== NEAR-COPY PAIRS  |corr| > {1 - args.tol:.6f}  (tol = {args.tol}) ====')
    C = A - torch.eye(r, dtype=A.dtype)
    pair_count = 0
    for i in range(r):
        for j in range(i + 1, r):
            c = float(C[i, j])
            if abs(c) > 1 - args.tol:
                print(f'  {labels[i]:42s}  <->  {labels[j]:42s}  c = {c:+.6f}')
                pair_count += 1
    print(f'  [{pair_count} pair{"" if pair_count == 1 else "s"}]')

    # ---- null-space directions ----
    print(f'\n==== NULL-SPACE DIRECTIONS  mu < {args.null_tol:g} ====')
    found = 0
    for k in range(r):
        if float(mu[k]) >= args.null_tol:
            break
        u = U[:, k]
        order = sorted(range(r), key=lambda j: -abs(float(u[j])))
        top = [j for j in order[:6] if abs(float(u[j])) > 0.1]
        if len(top) < 2:
            continue
        found += 1
        comps = ', '.join(f'{float(u[j]):+.3f}*{labels[j]}' for j in top)
        print(f'  mu = {float(mu[k]):.3e}  {comps}')
    print(f'  [{found} direction{"" if found == 1 else "s"}]')


if __name__ == '__main__':
    main()
