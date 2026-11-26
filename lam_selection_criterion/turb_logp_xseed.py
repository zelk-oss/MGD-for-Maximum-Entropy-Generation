"""Cross-seed coherence of log p as a function of lam (2026-11-18): the turbulence check of a lam rule.

For each seed s of an arm (same data, potentials and settings; only the SDE seed differs) and each lam,
log p_s(x; lam) = Theta_s(lam)(t_end) . phi_s(x) + const on the SAME data signals x (the training x1, as
turbulence/logp_family_contrib.py, whose loaders this script reuses). Theta = the phase-1 file of the
cross-seed sweep (UNCONSTRAINED Theta_reg, uniform weights, the arm's ridge), so this checks the lam choice
itself, before any L_6 sign constraint. The constant (log Z) differs per seed and is removed by centring
each seed's log p over the signals.

What it measures (per lam), on the centred log p:
  - Pearson and Spearman correlation between seeds (median and worst over the pairs);
  - seed spread: per signal, the std over seeds; reported as its median in nats, overall / typical / rare
    (rare = max |lag-1 increment| above the 95th percentile of x1), and as a ratio to the spread of the
    seed-mean log p over signals (the 10-06 statistic: 'seed spread % of signal spread');
  - the largest per-signal seed spread (nats).
What it does NOT measure: a bias common to all seeds. Coherence rises with lam even when Theta is
over-smoothed (all seeds pulled to the same wrong curve), so it can rule out a lam that is too SMALL, not
one that is too large.

Writes <out>/logp_xseed_<label>.pt (log p per seed and lam, the increment, the table) and prints the table.

Usage (repo root, Jean Zay, GPU node):
  python lam_selection_criterion/turb_logp_xseed.py --label turbulence_nested_M3_bicw_allmass_scat4 \
      --seeds 900 901 902 903 904 --sweep_dir $SCRATCH/.../theta_reg_lamsweep_xseed/bscat4 \
      --lams 0 1e-6 1e-5 1e-4 1e-3 1e-2 --ridge 1e-4 --out lam_selection_criterion/results/turb_logp_xseed
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'turbulence'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import logp_family_contrib as lfc                                   # noqa: E402  (also sets the codes/ paths)
from logp_family_contrib import newest, load_potentials, phi_matrix, build_data   # noqa: E402

TURB = ROOT / 'turbulence'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--label', required=True, help='experiment label of the arm (as in the system file names)')
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--sweep_dir', type=Path, required=True, help='phase-1 Theta files: <sweep_dir>/<config>/lam..pt')
    p.add_argument('--lams', type=float, nargs='+', required=True)
    p.add_argument('--ridge', type=float, required=True)
    p.add_argument('--rare_pct', type=float, default=95.0)
    p.add_argument('--n_signals', type=int, default=None, help='first N signals of x1 only (smoke)')
    p.add_argument('--out', type=Path, required=True)
    return p.parse_args()


def main():
    a = parse_args()
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lams = sorted(set(a.lams))
    logp, configs, x1_ref, phi_ref = {}, {}, None, None
    for s in a.seeds:
        exp = newest(f'*/*_seed_{s}_*{a.label}_2*', TURB / 'experiments')
        cfg = json.loads((exp / 'config.json').read_text())
        config = cfg['config_name']
        configs[s] = config
        x1, _ = build_data(int(cfg.get('subseries_len', 512)), int(cfg['M']), int(cfg['n1']), 1, data_path=cfg.get('data_path'))
        if a.n_signals:
            x1 = x1[:a.n_signals]
        if x1_ref is None:
            x1_ref = x1
        elif not torch.equal(x1, x1_ref):
            sys.exit(f'seed {s}: data x1 differ from the first seed (the coherence needs the same signals)')
        pots = load_potentials(exp, cfg, device)
        phi = phi_matrix(pots, x1.to(device))                           # (N, r) float64 numpy
        if phi_ref is not None and phi.shape == phi_ref.shape:
            d = float(np.abs(phi - phi_ref).max() / max(np.abs(phi_ref).max(), 1e-300))
            print(f'  seed {s}: max |phi_s - phi_first| / max|phi| = {d:.2e} (potentials refitted per run)')
        phi_ref = phi if phi_ref is None else phi_ref
        rows = []
        for lam in lams:
            f = a.sweep_dir / config / f'lam{lam:g}_ridge{a.ridge:g}_moment_mask.pt'
            if not f.exists():
                sys.exit(f'missing Theta file {f}')
            sol = torch.load(f, map_location='cpu', weights_only=False)
            th = sol['Theta_reg'][-1].double().numpy()
            if th.shape[0] != phi.shape[1]:
                sys.exit(f'{f.name}: Theta has {th.shape[0]} entries, phi has {phi.shape[1]}')
            lp = phi @ th
            if not np.isfinite(lp).all():
                sys.exit(f'seed {s} lam {lam:g}: non-finite log p')
            rows.append(lp)
        logp[s] = np.stack(rows)                                         # (L, N)
        print(f'seed {s}: {config[-60:]}  ({time.time() - t0:.0f} s)', flush=True)

    S = np.stack([logp[s] for s in a.seeds])                            # (S, L, N)
    Sc = S - S.mean(-1, keepdims=True)                                  # remove each seed's log Z (and mean)
    inc = (x1_ref[..., 1:] - x1_ref[..., :-1]).abs().amax(dim=(1, 2)).cpu().numpy()
    rare = inc > np.percentile(inc, a.rare_pct)
    pairs = list(itertools.combinations(range(len(a.seeds)), 2))
    table = []
    print(f'\n######## {a.label}: cross-seed coherence of log p (centred), {len(a.seeds)} seeds, {S.shape[-1]} signals '
          f'({int(rare.sum())} rare = top {100 - a.rare_pct:g} % of max |lag-1 increment|)')
    print(f"{'lam':>7} | {'Pearson med':>11} {'worst':>7} | {'Spearman med':>12} {'worst':>7} | "
          f"{'seed spread med (nats): all':>27} {'typical':>8} {'rare':>8} | {'/ signal spread':>15} | {'max spread':>10}")
    for i, lam in enumerate(lams):
        X = Sc[:, i]                                                    # (S, N)
        pr = [np.corrcoef(X[p], X[q])[0, 1] for p, q in pairs]
        sr = [stats.spearmanr(X[p], X[q])[0] for p, q in pairs]
        spread = X.std(0, ddof=1)                                       # per signal, over seeds
        sig = X.mean(0).std(ddof=1)                                     # spread of the seed-mean over signals
        row = dict(lam=lam, pearson_med=float(np.median(pr)), pearson_min=float(np.min(pr)),
                   spearman_med=float(np.median(sr)), spearman_min=float(np.min(sr)),
                   spread_med=float(np.median(spread)), spread_typ=float(np.median(spread[~rare])),
                   spread_rare=float(np.median(spread[rare])), spread_ratio=float(np.median(spread) / sig),
                   spread_max=float(spread.max()), signal_spread=float(sig))
        table.append(row)
        print(f"{lam:7.0e} | {row['pearson_med']:11.4f} {row['pearson_min']:7.4f} | {row['spearman_med']:12.4f} "
              f"{row['spearman_min']:7.4f} | {row['spread_med']:27.4g} {row['spread_typ']:8.4g} {row['spread_rare']:8.4g} | "
              f"{row['spread_ratio']:15.3g} | {row['spread_max']:10.4g}")
    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / f'logp_xseed_{a.label}.pt'
    torch.save(dict(label=a.label, seeds=a.seeds, configs=configs, lams=lams, ridge=a.ridge,
                    sweep_dir=str(a.sweep_dir), logp=torch.as_tensor(S), increment=torch.as_tensor(inc),
                    rare=torch.as_tensor(rare), table=table), dest)
    print(f'\n-> {dest}  (total {time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
