#!/usr/bin/env python3
"""Jets: do the seeds of an arm agree on log p, and what sign do the L_6 coefficients have at t = 1? (2026-10-29)

Per arm, Theta_reg of each seed at the lam chosen by cross-seed CV (lam_selection_criterion/turb_xseed.py,
window end0.99; --lams adds other lam of the same sweep). log p_s(x) = Theta_s(t=1) . phi(x) (+ const: log Z drops
out of a correlation; code sign, p ~ exp(+theta.phi)) on the training data x1, phi from the run's own fitted
potentials (experiments/<config>/fitted_potentials, via turbulence/logp_family_contrib.load_potentials).

Prints per arm and lam: the seed x seed Pearson and Spearman correlation of log p over the signals, and the
L_6 coefficients of Theta(t=1) per filter x seed (turbulence: filters 0-3 negative in 5/5 seeds, filter 4 positive
in 4/5). Saves <out>/logp_seed_check.npz (log p per seed, L_6 table).

Usage (repo root, Jean Zay, a GPU node; the samples / potentials are not pulled locally):
  python jets/logp_seed_check.py
  python jets/logp_seed_check.py --lams 1e-3 1e-2 1e-1        # robustness of the agreement in lam
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

HERE = Path(__file__).resolve().parent                      # jets/
ROOT = HERE.parent
for p in (ROOT / 'turbulence', ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data',
          HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from logp_family_contrib import load_potentials, phi_matrix   # noqa: E402
from check_potentials import theta_column_map                 # noqa: E402
from run_comparison import _find_exp_dir                      # noqa: E402
from jets_compare import reference_data                       # noqa: E402

ARMS = ['jets_full', 'jets_noscalar']
XSEED = ROOT / 'lam_selection_criterion' / 'results' / 'jets_xseed'


def theta_file(selected_file, lam_sel, lam):
    """Same sweep, other lam: resolve_theta_reg.out_path names files lam<lam>_ridge..."""
    f = Path(selected_file)
    assert f.name.startswith(f'lam{lam_sel:g}_'), f.name
    return f.with_name(f'lam{lam:g}_' + f.name[len(f'lam{lam_sel:g}_'):])


def corr_table(name, C, seeds):
    off = C[~np.eye(len(C), dtype=bool)]
    print(f'    {name}: off-diagonal min {off.min():.3f}  median {np.median(off):.3f}')
    print('      seed ' + ' '.join(f'{s:>6d}' for s in seeds))
    for s, row in zip(seeds, C):
        print(f'      {s:4d} ' + ' '.join(f'{v:6.3f}' for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=ARMS)
    ap.add_argument('--window', default='end0.99', help='xseed window whose selected lam is used')
    ap.add_argument('--lams', type=float, nargs='*', default=None, help='default: the selected lam only')
    ap.add_argument('--out', type=Path, default=HERE / 'figures' / 'logp_seed_check')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save = {}
    x1 = None
    for arm in args.arms:
        xs = torch.load(XSEED / f'xseed_{arm}.pt', map_location='cpu', weights_only=False)
        lam_sel, files = xs['selected'][args.window], xs['theta_files'][args.window]
        lams = args.lams or [lam_sel]
        print(f'\n################ {arm}: selected lam ({args.window}) = {lam_sel:g}; evaluated at {lams}')
        seeds, phis, thetas = [], [], {lam: [] for lam in lams}
        for config, f_sel in files.items():
            exp = _find_exp_dir(HERE, config)
            assert exp is not None, f'no experiments/ folder for {config}'
            cfg = json.loads((exp / 'config.json').read_text())
            if x1 is None:
                x1 = reference_data(cfg).reshape(-1, 1, int(cfg['target_len'])).to(device)
            pots = load_potentials(exp, cfg, device)
            phi = phi_matrix(pots, x1)
            seeds.append(int(cfg['seed']))
            phis.append(phi)
            for lam in lams:
                sol = torch.load(theta_file(f_sel, lam_sel, lam), map_location='cpu', weights_only=False)
                assert sol['config'] == config, (sol['config'], config)
                thetas[lam].append(sol['Theta_reg'][-1].double().numpy())
            cmap = theta_column_map(pots, phi.shape[1])
            print(f'  seed {seeds[-1]}: r = {phi.shape[1]}, t_last = {float(sol["t_reg"][-1]):.6f}; '
                  + ', '.join(f'{k}[{a}:{a + d}]' for k, (a, d) in cmap.items()))
        a6, d6 = cmap['L_6']
        for lam in lams:
            LP = np.stack([phi @ th for phi, th in zip(phis, thetas[lam])], axis=1)      # (n1, seeds)
            print(f'\n  lam = {lam:g}: log p over {LP.shape[0]} data signals')
            corr_table('Pearson ', np.corrcoef(LP.T), seeds)
            corr_table('Spearman', stats.spearmanr(LP).statistic, seeds)
            L6 = np.stack([th[a6:a6 + d6] for th in thetas[lam]])                       # (seeds, filters)
            print(f'    L_6 coefficients at t = 1 (rows = filter, columns = seed; code sign):')
            print('      filt ' + ' '.join(f'{s:>10d}' for s in seeds) + '   n_neg')
            for j in range(d6):
                print(f'      {j:4d} ' + ' '.join(f'{v:10.3g}' for v in L6[:, j]) + f'   {int((L6[:, j] < 0).sum())}/{len(seeds)}')
            save[f'{arm}__lam{lam:g}__logp'] = LP
            save[f'{arm}__lam{lam:g}__L6'] = L6
        save[f'{arm}__seeds'] = np.array(seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / 'logp_seed_check.npz', **save)
    print(f'\n-> {args.out / "logp_seed_check.npz"}')


if __name__ == '__main__':
    main()
