"""Conditioning of the theta solve along the transport, from a SAVED regularised system.

    python notes/eps_conditioning_0930/cond_from_system.py SYSTEM.pt [--root turbulence] [--stride 10]

The system's M_k is G(y_k), the raw Gram the corrector (compute_theta) solves at every
step. This rebuilds that solve's live set from the saved theta_t (the corrector writes
exactly 0 for every masked statistic) and, every --stride nodes, takes the Jacobi-scaled
live block (unit diagonal, BEFORE the ridge) as in compute_theta:
    mu_min, mu_max (kappa = mu_max / mu_min), n_live,
    the 3 statistics with the largest |loading| on the mu_min eigenvector,
    and diag(M_k) of every statistic (raw G_nn(y_k); region statistics are in units of
    their data value, so their live floor is near_empty_tol = 1e-6).
Needs no new run. The eta solve (G(x_k)) is not in the system: the in-run log
(--cond_every) has both.

Output: <root>/saved_results/cond_offline/<config>.pt and a summary by time band.
Peak RAM ~ the loaded system (~38 GB at nt = 60000, r = 281).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]

ap = argparse.ArgumentParser()
ap.add_argument('system', type=Path)
ap.add_argument('--root', default='turbulence')
ap.add_argument('--stride', type=int, default=10)
a = ap.parse_args()

config = a.system.name[:-3] if a.system.name.endswith('.pt') else a.system.name
t0 = time.time()
system = torch.load(a.system, map_location='cpu', weights_only=False)
M, t = system['M'], system['t'].double().numpy()
for key in ('G', 'Sigma', 'c', 'b'):                       # free what is not needed
    system.pop(key, None)
print(f'[{config}] {len(t)} nodes, r={system["num_potentials"]}, loaded in {time.time() - t0:.0f} s')

base = Path(a.root) / 'saved_results'
theta_t = torch.load(base / 'lagrange_multipliers' / f'{config}.pt', map_location='cpu')
t_fine = np.asarray(torch.load(base / 'sampling_times' / f'{config}.pt', map_location='cpu'), dtype=np.float64)
idx_t = np.clip(np.searchsorted(t_fine[1:], t), 0, len(t_fine) - 2)
assert np.abs(t_fine[1:][idx_t] - t).max() < 1e-9, 'system nodes are not on the saved time grid'

nodes = np.arange(0, len(t), a.stride)
rows, diags = [], []
t0 = time.time()
for k in nodes:
    Mk = M[k].double()
    live = (theta_t[idx_t[k]] != 0)
    idx = live.nonzero().flatten()
    X = Mk[idx][:, idx]
    d = torch.diagonal(X).sqrt()
    Xs = X / (d[:, None] * d[None, :]); Xs = (Xs + Xs.T) / 2
    try:
        mu, U = torch.linalg.eigh(Xs)
        top = idx[U[:, 0].abs().argsort(descending=True)[:3]].tolist()
        lmin, lmax = float(mu[0]), float(mu[-1])
    except torch.linalg.LinAlgError:
        lmin = lmax = float('nan'); top = [-1, -1, -1]
    rows.append([k, t[k], int(idx.numel()), lmin, lmax, *top])
    diags.append(torch.diagonal(Mk).float())
rows = np.array(rows)
print(f'{len(nodes)} nodes in {time.time() - t0:.0f} s')

out = base / 'cond_offline'
out.mkdir(parents=True, exist_ok=True)
torch.save({'columns': ['k', 't', 'n_live', 'mu_min', 'mu_max', 'top0', 'top1', 'top2'],
            'rows': torch.as_tensor(rows), 'diag_M': torch.stack(diags), 'stride': a.stride,
            'config': config, 'note': 'theta solve, Jacobi-scaled live G(y_k) before the ridge'},
           out / f'{config}.pt')
print(f'saved {out / (config + ".pt")}')

kappa = rows[:, 4] / rows[:, 3]
print(f'\n{"t band":>16} {"nodes":>6} {"n_live":>11} {"mu_min med":>11} {"kappa med":>10} {"kappa p90":>10} {"kappa max":>10}'
      f'  most frequent top0 (count)')
for lo, hi in [(0, .1), (.1, .5), (.5, .9), (.9, .99), (.99, .999), (.999, 1.0001)]:
    s = (rows[:, 1] >= lo) & (rows[:, 1] < hi)
    if not s.any():
        continue
    vals, cnt = np.unique(rows[s, 5].astype(int), return_counts=True)
    common = ', '.join(f'{v} ({c})' for v, c in sorted(zip(vals, cnt), key=lambda x: -x[1])[:4])
    print(f'  [{lo:.3f}, {min(hi, 1):.3f}) {s.sum():6d} {rows[s, 2].min():5.0f}-{rows[s, 2].max():<5.0f}'
          f' {np.median(rows[s, 3]):11.2e} {np.median(kappa[s]):10.3g} {np.percentile(kappa[s], 90):10.3g}'
          f' {kappa[s].max():10.3g}  {common}')
