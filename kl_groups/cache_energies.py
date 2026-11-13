#!/usr/bin/env python3
"""Cache the per-term energies E[i, k] = Theta_k * phi_k(x_i) (CODE sign: log p = +Theta . phi + const) on the GENERATED
signals of a trained model, for kl_groups.py. One file per (dataset, seed): kl_groups/results/E_<dataset>_seed<S>.npz
  E (N, n) float64, families, starts, widths, idx_<family> (3, w) for order-4 families, J, L (2D), theta_desc, config.

ASSUMPTION (flagged): E_p is estimated on the generated signals x_i ~ rho_1 (law of the SDE endpoint), while the energies
use Theta_reg(t = 1). H = D_KL(p || q_G) only if rho_1 = p_{Theta_reg(1)}; both approximate p*, neither exactly.

Loaders reuse the dataset scripts (same Theta and samples as the Var/2 tables):
  lagrangian : B+Scat4 turbulence_nested_M3_bicw_allmass_scat4, sigma 3.5, nt 55000, the 11-01 runs (tag 20261101_*),
               Theta_reg lam 3e-5 ridge 1e-4 (L_6<0 file if present, else unconstrained)     [talk_logp_generated.py]
  jets       : jets_full, xseed end0.99                                                       [jets/logp_generated_jets.py]
  snp        : snp_M1, xseed end0.99                                                          [snp_experiment/logp_snp.py]
  cosmic     : scat4_real, xseed end0.99, phi from the cache results/phi/<config>.pt          [cosmic_web/logp_cw.py]
  fbm        : Gaussian check, fBm H = 0.5 (white), M 32, J 5, Scat2 + Scat4 Mod2 Re/Im, n1 15000, in-run Theta_reg
               (lam 1e-6, September pipeline, no fit_micro)

Usage (repo root, GPU node on Jean Zay):  python kl_groups/cache_energies.py --dataset lagrangian --seeds 900 901
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in ('codes', 'codes/filters', 'codes/potentials', 'data', 'turbulence', 'jets', 'snp_experiment', 'cosmic_web', ''):
    sys.path.insert(0, str(ROOT / p))
OUT = Path(os.environ.get('KL_RESULTS', HERE / 'results'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def scat4_idx(pots):
    """(3, w) index rows of each order-4 family; the 2D classes build them in forward() with indices_fourth_order(J, L)."""
    from utils_potentials import indices_fourth_order
    out = {}
    for k, p in pots.items():
        if 'Fourth_Order' in k:
            idx = p.indices if hasattr(p, 'indices') else indices_fourth_order(p.J, p.L)
            assert idx.shape[1] == p.num_coefficients, (k, idx.shape, p.num_coefficients)
            out[f'idx_{k}'] = np.asarray(idx)
    return out


def lagrangian(seed):
    from logp_family_contrib import find_theta, load_potentials, phi_matrix
    from check_potentials import theta_column_map
    from run_comparison import load_config
    T = ROOT / 'turbulence'
    exps = sorted(T.glob(f'experiments/*sigma3.5_nt55000_*bicw_allmass_scat4/*_seed_{seed}_*_20261101_*'))
    assert len(exps) == 1, f'expected exactly one 11-01 run for seed {seed}: {exps}'
    exp = exps[0]
    cfg = json.loads((exp / 'config.json').read_text())
    config = cfg['config_name']
    l6neg = Path(os.environ.get('SCRATCH', '/nonexistent')) / 'MGD-for-Maximum-Entropy-Generation/turbulence/theta_reg_l6neg'
    if not (l6neg / config).exists():
        l6neg = T / 'theta_reg_backup' / 'theta_reg_l6neg'
    if os.environ.get('LAG_LAM'):        # e.g. the cross-seed lam: UNCONSTRAINED Theta_reg of that sweep (LAG_SWEEP_DIR)
        theta, desc = find_theta(config, float(os.environ['LAG_LAM']), 1e-4, 1e-2, False,
                                 Path(os.environ['LAG_SWEEP_DIR']), l6neg, unconstrained=True)
    else:                                # default (run1): lam 3e-5, L_6<0 file if present
        theta, desc = find_theta(config, 3e-5, 1e-4, 1e-2, False, T / 'saved_results' / 'theta_reg_lamsweep', l6neg)
    xt = load_config(config, T, extras=False)['xt']
    pots = load_potentials(exp, cfg, DEV)
    phi = phi_matrix(pots, xt.reshape(xt.shape[0], 1, -1).to(DEV))
    return phi, theta, theta_column_map(pots, len(theta)), scat4_idx(pots), dict(J=cfg['J'], config=config, theta_desc=desc)


def jets(seed, arm='jets_full', window='end0.99'):
    import logp_jets                                                    # noqa: F401  (sys.path as the jets scripts)
    from logp_family_contrib import load_potentials, phi_matrix
    from run_comparison import _find_exp_dir, load_config
    from logp_seed_check import XSEED, theta_file
    from check_potentials import theta_column_map
    xs = torch.load(XSEED / f'xseed_{arm}.pt', map_location='cpu', weights_only=False)
    lam = xs['selected'][window]
    for config, f_sel in xs['theta_files'][window].items():
        exp = _find_exp_dir(logp_jets.HERE, config)
        cfg = json.loads((exp / 'config.json').read_text())
        if int(cfg['seed']) != seed:
            continue
        f = theta_file(f_sel, lam, lam)
        sol = torch.load(f, map_location='cpu', weights_only=False)
        assert sol['config'] == config
        theta = sol['Theta_reg'][-1].double().numpy()
        xt = load_config(config, logp_jets.HERE, label=arm, extras=False)['xt']
        pots = load_potentials(exp, cfg, DEV)
        phi = phi_matrix(pots, xt.reshape(xt.shape[0], 1, -1).to(DEV))
        return phi, theta, theta_column_map(pots, len(theta)), scat4_idx(pots), \
            dict(J=cfg['J'], config=config, theta_desc=f'{arm} xseed {window} lam {lam:g} {Path(f).name}')
    raise FileNotFoundError(f'{arm}: no xseed run for seed {seed}')


def snp(seed, arm='snp_M1', window='end0.99'):
    from logp_family_contrib import load_potentials, phi_matrix
    from run_comparison import _find_exp_dir, load_config
    from check_potentials import theta_column_map
    import jets_compare as jc
    S = ROOT / 'snp_experiment'
    xs = torch.load(S / 'results' / 'xseed' / f'xseed_{arm}.pt', map_location='cpu', weights_only=False)
    lam = xs['selected'][window]
    config = jc.find_seed_configs(S, arm, [seed])[seed]
    exp = _find_exp_dir(S, config)
    cfg = json.loads((exp / 'config.json').read_text())
    f = xs['theta_files'][window][config]
    sol = torch.load(f, map_location='cpu', weights_only=False)
    assert sol['config'] == config and float(sol['lam']) == float(lam)
    theta = sol['Theta_reg'][-1].double().numpy()
    pots = load_potentials(exp, cfg, DEV)
    xt = load_config(config, S, label=arm, extras=False)['xt']
    phi = phi_matrix(pots, xt.reshape(xt.shape[0], 1, -1).float().to(DEV))
    return phi, theta, theta_column_map(pots, len(theta)), scat4_idx(pots), \
        dict(J=cfg['J'], config=config, theta_desc=f'{arm} xseed {window} lam {lam:g} {Path(f).name}')


def cosmic(seed, arm='scat4_real', window='end0.99'):
    import cw_common as cw
    from logp_cw import theta_file
    xs = torch.load(cw.RESULTS / 'xseed' / f'xseed_{arm}.pt', map_location='cpu', weights_only=False)
    exp_dir, cfg = cw.find_runs(arm, [seed])[seed]
    f, lam = theta_file(arm, window, cfg['config_name'], xs, None, None)
    sol = torch.load(f, map_location='cpu', weights_only=False)
    assert sol['config'] == cfg['config_name']
    theta = sol['Theta_reg'][-1].double().numpy()
    _, phi_g, fam = cw.get_phi(arm, seed, exp_dir, cfg, DEV)              # cached phi of the generated maps
    pots = cw.build_potentials(cfg, 'cpu')
    return phi_g, theta, fam, scat4_idx(pots), \
        dict(J=cfg['J'], L=cfg['L'], config=cfg['config_name'], theta_desc=f'{arm} xseed {window} lam {lam:g} {Path(f).name}')


def fbm(seed):
    from logp_family_contrib import load_potentials, phi_matrix
    from check_potentials import theta_column_map
    from utils import load_results
    G = ROOT / 'gaussian_experiment'
    exps = sorted(G.glob(f'experiments/*M32_J5_Q1_sigma1.0_nt1000_n1_15000_*termsc52d1d13_gaussian_d32/*_seed_{seed}_*'))
    assert len(exps) == 1, exps
    cfg = json.loads((exps[0] / 'config.json').read_text())
    config = exps[0].name
    xt, _, _, _, Th = load_results(G, config)
    assert Th is not None, f'no Theta_reg for {config}'
    Th = Th['Theta_reg'] if isinstance(Th, dict) else Th
    theta = Th[-1].double().cpu().numpy()
    pots = load_potentials(exps[0], cfg, DEV)
    phi = phi_matrix(pots, xt.reshape(xt.shape[0], 1, -1).float().to(DEV))
    return phi, theta, theta_column_map(pots, len(theta)), scat4_idx(pots), \
        dict(J=cfg['J'], config=config, theta_desc=f'in-run Theta_reg lam {cfg["lam"]:g}')


def gauss4(seed, window='end0.99'):
    """In-family Gaussian test (gauss_scat4_test/): run root GAUSS4_ROOT, xseed arm GAUSS4_ARM; truth theta = 0 off Scat2."""
    from logp_family_contrib import load_potentials, phi_matrix
    from check_potentials import theta_column_map
    from logp_gauss_scat2 import load_samples
    R, arm = Path(os.environ['GAUSS4_ROOT']), os.environ['GAUSS4_ARM']
    xs = torch.load(R / 'results' / f'xseed_{arm}.pt', map_location='cpu', weights_only=False)
    lam = xs['selected'][window]
    for config, f in xs['theta_files'][window].items():
        exp = next((R / 'experiments').glob(f'*/{config}'))
        cfg = json.loads((exp / 'config.json').read_text()); cfg = cfg.get('args', cfg)
        if int(cfg['seed']) != seed:
            continue
        theta = torch.load(f, map_location='cpu', weights_only=False)['Theta_reg'][-1].double().numpy()
        pots = load_potentials(exp, cfg, DEV)
        xt = load_samples(R, config)
        phi = phi_matrix(pots, xt.reshape(xt.shape[0], 1, -1).float().to(DEV))
        return phi, theta, theta_column_map(pots, len(theta)), scat4_idx(pots), \
            dict(J=cfg['J'], config=config, theta_desc=f'{arm} xseed {window} lam {lam:g} {Path(f).name}')
    raise FileNotFoundError(f'{arm}: no xseed run for seed {seed}')


LOAD = dict(lagrangian=lagrangian, jets=jets, snp=snp, cosmic=cosmic, fbm=fbm, gauss4=gauss4)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=list(LOAD))
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--n_max', type=int, default=None, help='keep the first n_max samples (smoke tests)')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        try:
            phi, theta, cmap, idx, meta = LOAD[a.dataset](s)
        except Exception as e:                                          # one missing seed does not stop the others
            print(f'{a.dataset} seed {s}: FAILED {type(e).__name__}: {e}', flush=True)
            continue
        phi = phi[:a.n_max] if a.n_max else phi
        assert phi.shape[1] == len(theta) == sum(w for _, w in cmap.values()), (phi.shape, len(theta))
        E = phi.astype(np.float64) * theta[None, :]
        f = OUT / f'E_{a.dataset}_seed{s}.npz'
        np.savez(f, E=E, families=np.array(list(cmap)), starts=np.array([c[0] for c in cmap.values()]),
                 widths=np.array([c[1] for c in cmap.values()]), **idx, **{k: np.array(v) for k, v in meta.items()})
        print(f'{a.dataset} seed {s}: E {E.shape} | {meta["theta_desc"]} | families '
              + ', '.join(f'{k}:{w}' for k, (_, w) in cmap.items()) + f'\n -> {f}', flush=True)
