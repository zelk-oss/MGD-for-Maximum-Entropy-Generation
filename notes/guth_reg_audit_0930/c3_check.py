"""C3 (run on Jean Zay, after the resolve jobs): offline checks on an existing zfloor system.

    python notes/guth_reg_audit_0930/c3_check.py CONFIG [--root turbulence] [--ridge 1e-4]

Reads (CODE sign throughout):
  <root>/saved_results/lagrange_multipliers/CONFIG.pt       per-step theta_t
  <root>/saved_results/sampling_times/CONFIG.pt             fine grid
  <root>/saved_results/theta_reg_lamsweep/CONFIG/lam0_ridge<r>.pt            legacy, lam 0, no mask
  <root>/saved_results/theta_reg_lamsweep/CONFIG/lam0_ridge<r>_mask.pt       legacy, lam 0, --mask theta
  <root>/saved_results/theta_reg_lamsweep/CONFIG/lam_path_ridge<r>_moment_mask.pt   moment, --select
Prints:
  (a) masked Theta(0) vs theta_t (should agree: same ridge, same mask);
  (b) [hypothesis B5] unmasked Theta(0): entries with |Theta| > 10 A_i (A_i = max over blocks of
      the block median of |theta_t|); how many of them have theta_t == 0 (masked per step);
  (c) moment lam path: selected lam, vetoes, chi2, and the distance of Theta(lam) to the
      block averages of theta_t (200 nodes) for every lam.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('config')
ap.add_argument('--root', default='turbulence')
ap.add_argument('--ridge', type=float, default=1e-4)
ap.add_argument('--block', type=int, default=200)
a = ap.parse_args()
base = Path(a.root) / 'saved_results'
sweep = base / 'theta_reg_lamsweep' / a.config
theta_t = torch.load(base / 'lagrange_multipliers' / f'{a.config}.pt', map_location='cpu').double()
t_fine = np.asarray(torch.load(base / 'sampling_times' / f'{a.config}.pt', map_location='cpu'), dtype=np.float64)


def on(t_nodes):
    idx = np.clip(np.searchsorted(t_fine[1:], t_nodes), 0, len(t_fine) - 2)
    assert np.abs(t_fine[1:][idx] - t_nodes).max() < 1e-9
    return theta_t[idx]


def block_mean(X, b):
    n = (len(X) // b) * b
    return X[:n].reshape(-1, b, X.shape[1]).mean(1)


rg = f'{a.ridge:g}'
f_un, f_m = sweep / f'lam0_ridge{rg}.pt', sweep / f'lam0_ridge{rg}_mask.pt'
if f_m.exists():
    r = torch.load(f_m, weights_only=False)
    th = on(r['t_reg'].numpy())
    live = th != 0
    d = (r['Theta_reg'] - th)[live]
    print(f'(a) masked Theta(0) vs theta_t: rel diff {float(d.norm() / th[live].norm()):.2e}, '
          f'max |diff| {float(d.abs().max()):.2e}, masked entries {int((~live).sum())} / {live.numel()}')
if f_un.exists():
    r = torch.load(f_un, weights_only=False)
    Th = r['Theta_reg'].double()
    th = on(r['t_reg'].numpy())
    nb = max(5, len(th) // 100)
    A = torch.stack([th[i:i + nb].abs().median(0).values for i in range(0, len(th), nb)]).max(0).values.clamp_min(1e-300)
    boom = Th.abs() > 10 * A
    zero = th == 0
    print(f'(b) unmasked Theta(0): {int(boom.sum())} exploded entries (|Theta| > 10 A_i) in '
          f'{int(boom.any(0).sum())} potentials; {int((boom & zero).sum())} of them have theta_t == 0 '
          f'({float((boom & zero).sum()) / max(int(boom.sum()), 1):.1%}); theta_t == 0 entries: {int(zero.sum())}, '
          f'of which exploded {float((boom & zero).sum()) / max(int(zero.sum()), 1):.1%}')
    if boom.any():
        worst = torch.nonzero(boom)[:10].tolist()
        print('    first exploded (node, potential):', worst)
for f in sorted(sweep.glob(f'lam_path_ridge{rg}_moment*.pt')):
    P = torch.load(f, weights_only=False)
    th = on(P['t'].numpy())
    ref = block_mean(th, a.block)
    print(f'(c) {f.name}: selected lam {P["lam_selected"]}, inputs: {P["inputs"]}')
    for i, lam in enumerate(P['lams']):
        Thb = block_mean(P['Theta'][i].double(), a.block)
        dist = float((Thb - ref).norm() / ref.norm())
        chi2 = P['chi2'][i] if P['chi2'] is not None else float('nan')
        print(f'    lam {lam:9.3g}: held-out {P["cv_loss"][i]: .6e}  amp ratio {P["amp_ratio"][i]:9.3g}'
              f'{" VETO" if P["veto"][i] else "     "}  chi2 {chi2:8.3f}  '
              f'|block avg Theta - block avg theta_t| / |.| = {dist:.3e}')
