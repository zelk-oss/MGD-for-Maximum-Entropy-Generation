#!/usr/bin/env python3
"""Cross-seed coherence of log p for the S&P models, as a function of lam (same test as cosmic_web/seed_coherence_cw.py).

S&P has ONE data series, so a correlation over data signals is impossible, and -log p of that one series cannot be
compared across seeds either (each seed has its own unknown additive constant log Z). Instead every seed's Theta_reg(t=1)
scores the SAME common set of signals: the data series + the generated samples of ALL seeds pooled (1 + 5 x 100).
For each pair of seeds: Pearson correlation of C_F(x) = Theta_F . phi_F(x) over the common set, per family F and total;
the table prints the mean over seed pairs. ~1 = the seeds rank the signals alike, ~0 = log p is noise.
Also printed: std of total -log p over the common set (mean over seeds), and the data series' rank (percentile of
-log p within the common set, mean over seeds; the ranks ARE comparable across seeds).

Theta_reg files: <theta_dir>/<config>/lam<lam>_ridge*.pt (snp_theta_reg.sh -> resolve_theta_reg.py), every lam found.
Usage (repo root, GPU node on Jean Zay, minutes):
    python snp_experiment/seed_coherence_snp.py [--models M0 M1 M2 M3] [--lams 0 1e-5 1e-3]
Output: snp_experiment/results/seed_coherence/coherence_<arm>.npz  (lams, families, coherence (lams, F+1), std, rank)
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in ('codes', 'codes/filters', 'codes/potentials', 'data', 'turbulence', 'jets', 'snp_experiment'):
    sys.path.insert(0, str(ROOT / p))
SEEDS = [900, 901, 902, 903, 904]
THETA_ROOT = Path(os.environ.get('SCRATCH', '/nonexistent')) / 'MGD-for-Maximum-Entropy-Generation' / 'snp_experiment' / 'theta_reg'
OUT = HERE / 'results' / 'seed_coherence'


def coherence(C):
    """(F+1,) mean over seed pairs of the Pearson correlation over signals; C = (seeds, F+1, signals)."""
    S = C.shape[0]
    return np.array([np.mean([np.corrcoef(C[a, i], C[b, i])[0, 1] for a in range(S) for b in range(a + 1, S)])
                     for i in range(C.shape[1])])


def lam_of(f):
    return float(re.match(r'lam([^_]+)_ridge', f.name).group(1))


def main(models, seeds, lams, theta_root, out):
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
        configs = jc.find_seed_configs(HERE, arm, seeds)
        exps, gens = {}, []
        for s in seeds:
            assert configs[s] is not None, f'{arm}: no saved run for seed {s}'
            exps[s] = _find_exp_dir(HERE, configs[s])
            gens.append(load_config(configs[s], HERE, label=arm, extras=False)['xt'].reshape(-1, 1, x1.shape[-1]).float())
        X = torch.cat([x1.cpu()] + gens).to(device)              # common set: data first, then all generated
        tdir = theta_root / arm
        found = sorted({lam_of(f) for f in (tdir / configs[seeds[0]]).glob('lam*_ridge*.pt')})
        use = [l for l in found if lams is None or any(np.isclose(l, q) for q in lams)]
        print(f'\n=== {arm}: common set {X.shape[0]} signals (data + {len(seeds)} seeds generated), lams {use}')
        phis, cmaps = {}, {}
        for s in seeds:                                           # phi once per seed (its own potentials)
            cfg = json.loads((exps[s] / 'config.json').read_text())
            pots = load_potentials(exps[s], cfg, device)
            phis[s] = phi_matrix(pots, X)
        coh, std, rank, fams = [], [], [], None
        for lam in use:
            C = []
            for s in seeds:
                f = sorted((tdir / configs[s]).glob(f'lam{lam:g}_ridge*.pt'))
                assert f, f'no lam{lam:g}_ridge*.pt in {tdir / configs[s]}'
                sol = torch.load(f[0], map_location='cpu', weights_only=False)
                assert sol['config'] == configs[s] and np.isclose(float(sol['lam']), lam), (sol['config'], sol['lam'], lam)
                theta = sol['Theta_reg'][-1].double().numpy()
                if s not in cmaps:
                    cfg = json.loads((exps[s] / 'config.json').read_text())
                    cmaps[s] = theta_column_map(load_potentials(exps[s], cfg, device), len(theta))
                phi = phis[s]
                assert phi.shape[1] == len(theta), (phi.shape, len(theta))
                Cs = np.stack([phi[:, a:a + w] @ theta[a:a + w] for a, w in cmaps[s].values()])
                C.append(np.concatenate([Cs, Cs.sum(0, keepdims=True)]))
                fams = list(cmaps[s]) + ['TOTAL']
            C = np.stack(C)                                       # (seeds, F+1, signals)
            coh.append(coherence(C))
            std.append(C[:, -1].std(1, ddof=1).mean())
            rank.append(np.mean([100 * np.mean(-C[k, -1] < -C[k, -1, 0]) for k in range(len(seeds))]))
            if lam == use[0]:
                print(f'{"lam":>9s} ' + ' '.join(f'{re.sub("Scattering_|_Order", "", f)[:14]:>14s}' for f in fams)
                      + '   std(-log p)  data pct')
            print(f'{lam:9g} ' + ' '.join(f'{v:14.3f}' for v in coh[-1]) + f'   {std[-1]:11.3g}  {rank[-1]:8.1f}')
        np.savez_compressed(out / f'coherence_{arm}.npz', lams=np.array(use), families=np.array(fams),
                            coherence=np.array(coh), std=np.array(std), rank=np.array(rank), seeds=np.array(seeds))
        print(f'-> {out / f"coherence_{arm}.npz"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=['M0', 'M1', 'M2', 'M3'])
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--lams', nargs='+', type=float, default=None, help='subset of the swept lams (default: all found)')
    ap.add_argument('--theta_root', type=Path, default=THETA_ROOT)
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    main(a.models, a.seeds, a.lams, a.theta_root, a.out)
