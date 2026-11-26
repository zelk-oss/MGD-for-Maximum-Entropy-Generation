"""Var/2d of the energy theta.phi of each Scat4 coefficient (j, j', l) of B+Scat4, per pair (j, j') (2026-11-20).

Potentials: Scattering_Fourth_Order_Mod2_Real_Q1 and _Mod2_Imag_Q1. Coefficient (a, b, c) = mean_t Re/Im[(psi_c*|W_a|^2)(psi_c*|W_b|^2)^*],
a < b fine scales (j, j'), c = coarser scale l > b (c = J is the low-pass, Real only). Index 0 = finest scale.
Energy A_i(x) = theta_i phi_i(x), theta = Theta_reg(1) of the chosen lambda; Var over signals x (mean removed), on data and model.

Outputs in --out:
  var2d_scat4_triples.csv  one row per (potential, source, a, b, c): theta, half_var_nats, var2d
  var2d_scat4_pairs.csv    one row per (potential, source, a, b): Var of the SUM over l (covariances included), sum of the single Vars,
                           l with the largest single Var and its share of the sum of single Vars
  var2d_scat4_windows.csv  at fixed j (a): Var of the sum of the pair energies over a window of d: {1,2}, {1,2,3}, {2,3,4}, ...
                           (covariances included) and the sum of the single pair Vars (2026-11-24)
Var/2d = Var / (2 d) [nats per dimension], d = M = signal length (digits until 2026-11-25).
Usage (Jean Zay, repo root): python turbulence/var2d_per_scale/var2d_scat4_triples.py
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in (ROOT / 'turbulence', ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_comparison as rc                                         # noqa: E402
from logp_family_contrib import find_theta, load_potentials         # noqa: E402
from check_potentials import theta_column_map                       # noqa: E402

CHUNK = 500

ap = argparse.ArgumentParser()
ap.add_argument('--label', default='turbulence_nested_M3_bicw_allmass_scat4')
ap.add_argument('--seed', type=int, default=900)
ap.add_argument('--lam', type=float, default=1e-3, help='rule A for B+Scat4')
ap.add_argument('--ridge', type=float, default=1e-4)
ap.add_argument('--sweep_dir', type=Path, default=None)
ap.add_argument('--theta_file', type=Path, default=None,
                help='Theta_reg file to use (e.g. a pinpos or droplast solve); overrides --lam/--ridge/--sweep_dir lookup')
ap.add_argument('--n_max', type=int, default=None, help='use only the first n signals (smoke test)')
ap.add_argument('--out', type=Path, default=HERE / 'results')
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
if args.sweep_dir is None and args.theta_file is None:
    args.sweep_dir = (Path(os.environ['SCRATCH']) / 'MGD-for-Maximum-Entropy-Generation/turbulence'
                      / 'theta_reg_lamsweep_xseed/bscat4')

config = rc._find_config(rc.ROOT, args.label, args.seed)
assert config is not None, f'no samples for {args.label} seed {args.seed}'
exp = rc._find_exp_dir(rc.ROOT, config)
assert exp is not None, f'no experiments folder for {config}'
run = rc.load_config(config, rc.ROOT, label=args.label, extras=False)
cfg = run['cfg']
M = int(cfg['M'])
# octave index in ORIGINAL samples (user decision 2026-11-25): j_orig = j + log2(subseries_len / target_len), so
# M=256 runs (one coarse-graining step) and M=1024 runs compare at the same physical scale
J_SHIFT = int(round(np.log2(int(cfg['subseries_len']) / int(cfg['target_len']))))
if args.theta_file is not None:
    sol = torch.load(args.theta_file, map_location='cpu', weights_only=False)
    assert sol['config'] == config, (sol['config'], config)
    theta, desc = sol['Theta_reg'][-1].double().numpy(), f"{args.theta_file.name} lam={sol['lam']:g}"
else:
    theta, desc = find_theta(config, args.lam, args.ridge, 1e-2, False, args.sweep_dir,
                             Path('/nonexistent'), unconstrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pots = load_potentials(exp, cfg, device)
# B+Scat4 has the off-diagonal Real term, scat34d the one with the diagonal a = b (d = 0)
real = [n for n in ('Scattering_Fourth_Order_Mod2_Real_Q1', 'Scattering_Fourth_Order_Mod2_Real_Q1_diag') if n in pots]
assert len(real) == 1, f'expected one Scat4 Mod2 Real term, found {real}'
NAMES = {'real': real[0], 'imag': 'Scattering_Fourth_Order_Mod2_Imag_Q1'}
cmap = theta_column_map(pots, theta.shape[0])
signals = {'data': rc.reference_data(cfg).double(), 'model': run['xt'].double()}
if args.n_max:
    signals = {k: v[:args.n_max] for k, v in signals.items()}
print(f'run {config}\n  {desc}\n  signals: ' + ', '.join(f'{k} {tuple(v.shape)}' for k, v in signals.items()))

per_dim = 1.0 / (2 * M)                           # [nats per dimension], d = M (user 2026-11-25; was digits)
triple_rows, pair_rows, window_rows = [], [], []
for key, name in NAMES.items():
    pot = pots[name]
    start, width = cmap[name]
    th = np.asarray(theta[start:start + width])
    a, b, c = (np.asarray(i.cpu()).astype(int) for i in pot.indices)
    assert len(a) == width, (name, len(a), width)
    for src, x in signals.items():
        with torch.no_grad():
            phi = torch.cat([pot(x[i:i + CHUNK].reshape(-1, 1, M).to(device)).cpu().double()
                             for i in range(0, len(x), CHUNK)]).numpy()
        A = phi * th[None]                                            # (N, n_stat)
        var = A.var(0, ddof=1)
        for i in range(width):
            triple_rows.append(dict(potential=key, source=src, a=a[i], b=b[i], c=c[i], a_orig=a[i] + J_SHIFT, theta=th[i],
                                    half_var_nats=var[i] / 2, var2d=var[i] * per_dim))
        pair_energy = {}
        for pa, pb in sorted(set(zip(a.tolist(), b.tolist()))):
            cols = np.flatnonzero((a == pa) & (b == pb))
            pair_energy[(pa, pb)] = A[:, cols].sum(1)
            v_sum = pair_energy[(pa, pb)].var(ddof=1)
            k = cols[np.argmax(var[cols])]
            pair_rows.append(dict(potential=key, source=src, a=pa, b=pb, d=pb - pa, a_orig=pa + J_SHIFT, n_l=cols.size,
                                  var2d_sum=v_sum * per_dim, var2d_sum_of_singles=var[cols].sum() * per_dim,
                                  l_max=c[k], share_max=var[k] / var[cols].sum()))
        # Windows over d at fixed j: {d1,d2}, then {d1,d2,d3}, {d2,d3,d4}, ... (the first one has only 2 terms).
        # d = 0 (diagonal a = b, scat34d only) stays out of the windows: it is reported in the pairs file.
        for pa in sorted({p for p, _ in pair_energy}):
            ds = sorted(q - p for p, q in pair_energy if p == pa and q > p)
            if len(ds) < 2:
                continue
            windows = [ds[:2]] + [ds[i:i + 3] for i in range(len(ds) - 2)]
            for w in windows:
                E = np.stack([pair_energy[(pa, pa + dd)] for dd in w], 1)
                window_rows.append(dict(potential=key, source=src, a=pa, a_orig=pa + J_SHIFT, d_first=w[0], d_last=w[-1], n_d=len(w),
                                        var2d_sum=E.sum(1).var(ddof=1) * per_dim,
                                        var2d_sum_of_singles=E.var(0, ddof=1).sum() * per_dim))
    print(f'{key}: {width} coefficients')

for fname, rows in (('var2d_scat4_triples.csv', triple_rows), ('var2d_scat4_pairs.csv', pair_rows),
                    ('var2d_scat4_windows.csv', window_rows)):
    with open(args.out / fname, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
print(f'saved in {args.out}')
