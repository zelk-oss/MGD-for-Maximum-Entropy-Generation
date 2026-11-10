"""log p of the GENERATED signals of a turbulence run, for talk_turbulence.ipynb (signals ranked from rarest to
least rare). Same Theta_reg(t = 1) as logp_family_contrib.py: the L_6<0 file (SCRATCH, else the fswork backup
theta_reg_backup/theta_reg_l6neg), else the unconstrained sweep file. log p = Theta . phi(x) + const.

Output: turbulence/figures/data/talk_logp_generated_<label>[_seed<N>].npz (logp per generated signal, Theta(t = 1) and
its file name, and the contribution of each family theta_F . phi_F(x) to log p, same keys as logp_C_*.npz + TOTAL).
Usage (repo root, GPU node on Jean Zay):
    python turbulence/talk_logp_generated.py --label turbulence_ggd_mass_final
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from logp_family_contrib import HERE, newest, find_theta, load_potentials, phi_matrix
from check_potentials import theta_column_map                        # on sys.path once logp_family_contrib is imported
from run_comparison import load_config


def main(label, seed, lam, ridge, eps):
    exp = newest(f'*/*_seed_{seed}_*{label}_2*', HERE / 'experiments')
    cfg = json.loads((exp / 'config.json').read_text())
    config = cfg['config_name']
    print(f'run {config}')
    scratch = Path(os.environ.get('SCRATCH', '/nonexistent')) / 'MGD-for-Maximum-Entropy-Generation/turbulence'
    l6neg = scratch / 'theta_reg_l6neg'
    if not (l6neg / config).exists():
        l6neg = HERE / 'theta_reg_backup' / 'theta_reg_l6neg'
    theta, desc = find_theta(config, lam, ridge, eps, False, HERE / 'saved_results' / 'theta_reg_lamsweep', l6neg)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    xt = load_config(config, HERE, label=label, extras=False)['xt']
    xt = xt.reshape(xt.shape[0], 1, -1).to(device)
    print(f'generated signals {tuple(xt.shape)} on {device}')
    pots = load_potentials(exp, cfg, device)
    phi = phi_matrix(pots, xt)
    assert phi.shape[1] == theta.shape[0], (phi.shape, theta.shape)
    logp = phi @ theta
    # contribution of each family to log p on the generated signals, same keys as logp_family_contrib.py (data)
    cmap = theta_column_map(pots, theta.shape[0])
    C = {k: phi[:, a:a + w] @ theta[a:a + w] for k, (a, w) in cmap.items()}
    C['TOTAL'] = logp
    assert np.allclose(sum(C[k] for k in cmap), logp, rtol=1e-8, atol=1e-6 * np.abs(logp).max())

    stag = '' if seed == 900 else f'_seed{seed}'                       # as logp_family_contrib.py
    out = HERE / 'figures' / 'data' / f'talk_logp_generated_{label}{stag}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, logp=logp, theta=theta, theta_desc=desc, config=config, **C)
    print(f'log p: median {np.median(logp):.4g}, min {logp.min():.4g}, max {logp.max():.4g}\n-> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='turbulence_ggd_mass_final')
    ap.add_argument('--seed', type=int, default=900)
    ap.add_argument('--lam', type=float, default=5e-4)
    ap.add_argument('--ridge', type=float, default=1e-4)
    ap.add_argument('--eps', type=float, default=1e-2)
    a = ap.parse_args()
    main(a.label, a.seed, a.lam, a.ridge, a.eps)
