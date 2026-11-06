#!/usr/bin/env python3
"""log p of the cosmic-web DATA maps and of the GENERATED maps, per potential family (feeds codes/make_kl_tables.py).

For each seed of an arm: Theta_reg(t = 1) at the lam chosen by cosmic_web_xseed.sh (window end0.99, file
results/xseed/xseed_<arm>.pt), the potentials of the run, and
    C_F(x) = Theta_F . phi_F(x)         (log p = sum_F C_F + const; p ~ exp(+Theta.phi), tables use -log p)
for the 100 data maps and the generated maps of that seed. The additive constant is unknown: only differences
between maps are meaningful (that is all Var(C_F) needs).
Families = the potentials of the run: L_6 (confinement), First_Order + Second_Order (the 'scalar multiscale'
category of the tables), Third_Order_Real, Fourth_Order_Mod2_Real.

Usage (repo root, GPU node on Jean Zay, minutes):
    python cosmic_web/logp_cw.py --arms base_real scat4_real
    python cosmic_web/logp_cw.py --arms base_real --lam 1e-3 --theta_dir $SCRATCH/MGD-for-Maximum-Entropy-Generation/cosmic_web/theta_reg/base_real
        (explicit lam, no xseed file: Theta_reg files of resolve_theta_reg.py)
Output: cosmic_web/results/logp/logp_<arm>.npz
  C_gen (seeds, families + total, n_gen), C_data (seeds, families + total, n_data), families, widths, seeds, lam, M
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import cw_common as cw


def theta_file(arm, window, config, xs, lam, theta_dir):
    if xs is not None:
        return Path(xs['theta_files'][window][config]), float(xs['selected'][window])
    hits = sorted((Path(theta_dir) / config).glob(f'lam{lam:g}_ridge*.pt'))
    hits = [h for h in hits if 'neg' not in h.name and 'NOTCONVERGED' not in h.name]
    assert hits, f'no Theta_reg file lam{lam:g}_ridge*.pt in {Path(theta_dir) / config}'
    return hits[0], lam


def family_logp(phi, theta, fam):
    """(families + total, n): C_F = phi[:, F] @ theta[F] for each family F, and their sum."""
    C = np.stack([phi[:, a:a + w] @ theta[a:a + w] for a, w in fam.values()])
    return np.concatenate([C, C.sum(0, keepdims=True)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arms', nargs='+', default=['base_real', 'scat4_real'])
    ap.add_argument('--seeds', nargs='+', type=int, default=cw.SEEDS)
    ap.add_argument('--window', default='end0.99', help='xseed window (what log p at t = 1 uses)')
    ap.add_argument('--lam', type=float, default=None, help='explicit lam instead of the xseed file (needs --theta_dir)')
    ap.add_argument('--theta_dir', default=None, help='folder with <config>/lam<lam>_ridge*.pt (resolve_theta_reg.py output)')
    ap.add_argument('--xseed_dir', type=Path, default=cw.RESULTS / 'xseed')
    ap.add_argument('--data_path', default=None, help='quijote.npy (default: the path in the run config)')
    ap.add_argument('--chunk', type=int, default=10, help='maps per forward pass (order-4 potentials are memory hungry)')
    ap.add_argument('--out', type=Path, default=cw.RESULTS / 'logp')
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a.out.mkdir(parents=True, exist_ok=True)

    for arm in a.arms:
        xs = None
        if a.lam is None:
            xs = torch.load(a.xseed_dir / f'xseed_{arm}.pt', map_location='cpu', weights_only=False)
        else:
            assert a.theta_dir, '--lam needs --theta_dir'
        runs = cw.find_runs(arm, a.seeds)
        Cg, Cd, fam_names, widths, lam = [], [], None, None, None
        for s in a.seeds:
            exp_dir, cfg = runs[s]
            f_theta, lam = theta_file(arm, a.window, cfg['config_name'], xs, a.lam, a.theta_dir)
            sol = torch.load(f_theta, map_location='cpu', weights_only=False)
            assert sol['config'] == cfg['config_name'] and float(sol['lam']) == float(lam), (sol['config'], sol['lam'], lam)
            theta = sol['Theta_reg'][-1].double().numpy()
            phi_d, phi_g, fam = cw.get_phi(arm, s, exp_dir, cfg, device, a.chunk, a.data_path)
            assert phi_d.shape[1] == len(theta), (phi_d.shape, theta.shape)
            Cg.append(family_logp(phi_g, theta, fam)); Cd.append(family_logp(phi_d, theta, fam))
            fam_names, widths = list(fam), [w for _, w in fam.values()]
            print(f'{arm} seed {s}: lam {lam:g} | -log p total (generated): median {-np.median(Cg[-1][-1]):.5g}, '
                  f'std {Cg[-1][-1].std(ddof=1):.4g} | data: median {-np.median(Cd[-1][-1]):.5g}, std {Cd[-1][-1].std(ddof=1):.4g}')
        dest = a.out / f'logp_{arm}.npz'
        np.savez_compressed(dest, C_gen=np.stack(Cg), C_data=np.stack(Cd), families=np.array(fam_names),
                            widths=np.array(widths), seeds=np.array(a.seeds), lam=lam, M=int(cfg['M']))
        print(f'{arm}: -> {dest}')


if __name__ == '__main__':
    main()
