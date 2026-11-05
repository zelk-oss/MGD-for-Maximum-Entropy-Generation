#!/usr/bin/env python3
"""log p of the GENERATED signals of the jets runs, for jets/talk_jets.ipynb (generated signals ranked by -log p, as
turbulence/talk_logp_generated.py does for the Lagrangian case).

Same Theta_reg(t = 1) and same fitted potentials as jets/logp_jets.py (cross-seed lam, window end0.99): for each seed,
log p = Theta . phi(x) + const, evaluated on that seed's own generated signals (run_comparison.load_config(...)['xt'],
same order as the notebook). The constant is unknown, so only differences between signals are meaningful: the notebook
plots -log p relative to the median.

Usage (repo root, GPU node on Jean Zay, ~1 min per seed):  python jets/logp_generated_jets.py
Output: jets/figures/logp_jets/logp_generated_<arm>.npz  (logp (seeds, n_generated), seeds, lam)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from logp_jets import HERE, OUT                                  # also puts the project folders on sys.path


def main(arm, window, out):
    from logp_family_contrib import load_potentials, phi_matrix
    from run_comparison import _find_exp_dir, load_config
    from logp_seed_check import XSEED, theta_file
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    xs = torch.load(XSEED / f'xseed_{arm}.pt', map_location='cpu', weights_only=False)
    lam = xs['selected'][window]
    logps, seeds = [], []
    for config, f_sel in xs['theta_files'][window].items():
        exp = _find_exp_dir(HERE, config)
        assert exp is not None, f'no experiments/ folder for {config}'
        cfg = json.loads((exp / 'config.json').read_text())
        sol = torch.load(theta_file(f_sel, lam, lam), map_location='cpu', weights_only=False)
        assert sol['config'] == config, (sol['config'], config)
        theta = sol['Theta_reg'][-1].double().numpy()
        xt = load_config(config, HERE, label=arm, extras=False)['xt']
        xt = xt.reshape(xt.shape[0], 1, -1).to(device)
        phi = phi_matrix(load_potentials(exp, cfg, device), xt)
        assert phi.shape[1] == theta.shape[0], (phi.shape, theta.shape)
        lp = phi @ theta
        seeds.append(int(cfg['seed']))
        logps.append(lp)
        print(f'seed {seeds[-1]}: generated {tuple(xt.shape)} | log p median {np.median(lp):.4g}, '
              f'min {lp.min():.4g}, max {lp.max():.4g}')
    out.mkdir(parents=True, exist_ok=True)
    f = out / f'logp_generated_{arm}.npz'
    np.savez_compressed(f, logp=np.stack(logps), seeds=np.array(seeds), lam=lam)
    print(f'lam {lam:g} -> {f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='jets_full')
    ap.add_argument('--window', default='end0.99', help='xseed window whose selected lam is used (as logp_jets.py)')
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    main(a.arm, a.window, a.out)
