#!/usr/bin/env python3
"""log p of the S&P data and of the GENERATED samples, per potential family, for snp_experiment/talk_snp.ipynb.

For each model (M0..M3) and seed: Theta_reg(t = 1) at the cross-seed lambda of snp_xseed.sh (window end0.99, files
written by snp_theta_reg.sh), the potentials of the run, and
    C_F(x) = Theta_F . phi_F(x)        (log p = sum_F C_F + const; the notebook plots -log p)
for the data (ONE signal, the whole series) and for the generated samples of that seed (n_particles signals).
The additive constant is unknown, so only differences between signals are meaningful.

Usage (repo root, GPU node on Jean Zay, minutes):  python snp_experiment/logp_snp.py [--models M0 M1 M2 M3]
Output: snp_experiment/figures/logp_snp/logp_snp_<model>.npz
  C_gen (seeds, families + total, n_gen), C_data (seeds, families + total), families, widths, seeds, lam
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import sys
for p in ('codes', 'codes/filters', 'codes/potentials', 'data', 'turbulence', 'jets', 'snp_experiment'):
    sys.path.insert(0, str(ROOT / p))
OUT = HERE / 'figures' / 'logp_snp'
XSEED = HERE / 'results' / 'xseed'
SEEDS = [900, 901, 902, 903, 904]


def main(models, window, out, seeds):
    from logp_family_contrib import load_potentials, phi_matrix
    from run_comparison import _find_exp_dir, load_config
    from check_potentials import theta_column_map
    from snp_loader import load_snp_returns
    import jets_compare as jc
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x1 = load_snp_returns()[0].reshape(1, 1, -1).float().to(device)
    out.mkdir(parents=True, exist_ok=True)
    for model in models:
        arm = f'snp_{model}'
        xs = torch.load(XSEED / f'xseed_{arm}.pt', map_location='cpu', weights_only=False)
        lam = xs['selected'][window]
        configs = jc.find_seed_configs(HERE, arm, seeds)
        Cg, Cd, fams, widths = [], [], None, None
        for s in seeds:
            config = configs[s]
            assert config is not None, f'{arm}: no saved run for seed {s}'
            exp = _find_exp_dir(HERE, config)
            cfg = json.loads((exp / 'config.json').read_text())
            sol = torch.load(xs['theta_files'][window][config], map_location='cpu', weights_only=False)
            assert sol['config'] == config and float(sol['lam']) == float(lam), (sol['config'], sol['lam'], lam)
            theta = sol['Theta_reg'][-1].double().numpy()
            pots = load_potentials(exp, cfg, device)
            cmap = theta_column_map(pots, len(theta))
            xt = load_config(config, HERE, label=arm, extras=False)['xt'].reshape(-1, 1, x1.shape[-1]).float().to(device)
            res = []
            for x in (xt, x1):
                phi = phi_matrix(pots, x)
                assert phi.shape[1] == len(theta), (phi.shape, len(theta))
                C = np.stack([phi[:, a:a + w] @ theta[a:a + w] for a, w in cmap.values()])
                res.append(np.concatenate([C, C.sum(0, keepdims=True)]))               # families + total
            Cg.append(res[0]); Cd.append(res[1][:, 0])
            fams, widths = list(cmap), [w for a, w in cmap.values()]
            print(f'{arm} seed {s}: generated {tuple(xt.shape)} | -log p total: data {-res[1][-1, 0]:.4g}, '
                  f'generated median {-np.median(res[0][-1]):.4g} (min {-res[0][-1].max():.4g}, max {-res[0][-1].min():.4g})')
        f = out / f'logp_snp_{model}.npz'
        np.savez_compressed(f, C_gen=np.stack(Cg), C_data=np.stack(Cd), families=np.array(fams), widths=np.array(widths),
                            seeds=np.array(seeds), lam=lam)
        print(f'{arm}: lam {lam:g} -> {f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['M0', 'M1', 'M2', 'M3'])
    ap.add_argument('--window', default='end0.99')
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    main(a.models, a.window, a.out, a.seeds)
