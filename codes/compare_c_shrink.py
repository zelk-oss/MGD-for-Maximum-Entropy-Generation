#!/usr/bin/env python3
"""Diagonal / shrunk C in the Theta_reg time term (2026-11-17): compare re-solves (resolve_theta_reg.py --c_shrink)
of a few-walker run with the full-C solve of a many-walker run of the same seed (the reference).

Metric (nats, no data maps needed): for two energies E_a = Theta_a . phi, E_b = Theta_b . phi at the last node
(t = t_reg[-1]), the spread over samples x of the log-density difference is
    std_x [ log p_a(x) - log p_b(x) ] = std_x [ (Theta_a - Theta_b) . phi(x) ] = sqrt( dTheta^T Sigma dTheta ),
exact for any law of x, with Sigma = Cov(phi) under that law (log Z_a - log Z_b is a constant and drops out).
Sigma = the walker covariance at the last node of the REFERENCE system (many walkers, law ~ data at t ~ 1).
CODE sign throughout; the sign does not matter for this metric.

Rows, per seed and lam:
  arm      : full (C as is), shr1 (pure diagonal), shrG (Gaussian plug-in alpha), at each walker count
  d_ref    : sqrt(dTheta^T Sigma dTheta) against the reference (ref walker count, full C, same lam, same seed)
  rel_ref  : the same divided by sqrt(Theta_ref^T Sigma Theta_ref) (std of the reference energy itself)
  alpha    : median per-cell alpha (shrink arms)
plus, per lam, the cross-seed floor: d between the reference solves of two different seeds (same Sigma of the
first seed), mean over pairs -- how much the reference itself moves from seed to seed.
lam = 0 (no time term, independent of C) is one more row for the 'full' arm.

Files expected: <theta_root>/<config>/lam<lam>_ridge<ridge>_moment_mask[_shr1|_shrG].pt, configs found by
'seed_<s>_' and ending in '_<label>'. Writes a CSV (--csv) and prints a table.
Usage: python codes/compare_c_shrink.py --theta_root DIR --systems_root DIR --labels ho_base_real_np100 ho_base_real_np400
       --ref_label ho_base_real_np400 --seeds 900 901 902 903 904 --lams 0 1e-4 1e-3 1e-2 1e-1 1 --csv out.csv
"""
import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import torch

ARMS = [('full', ''), ('shr1', '_shr1'), ('shrG', '_shrG')]


def find_config(root, seed, label):
    hits = sorted(p for p in root.iterdir() if p.is_dir() and f'seed_{seed}_' in p.name and p.name.endswith('_' + label))
    return hits[0] if len(hits) == 1 else None


def load_theta(cdir, lam, ridge, suffix):
    f = cdir / f'lam{lam:g}_ridge{ridge:g}_moment_mask{suffix}.pt'
    if not f.exists():
        return None
    o = torch.load(f, map_location='cpu', weights_only=False)
    al = o.get('c_shrink_alpha')
    return (float(o['t_reg'][-1]), o['Theta_reg'][-1].double().numpy(),
            float(np.median(al)) if al is not None else float('nan'))


def sigma_last(systems_root, config_name):
    f = systems_root / f'{config_name}.pt'
    s = torch.load(f, map_location='cpu', weights_only=False, mmap=True)
    S = s['Sigma'][-1].double().numpy()
    t_last = float(s['t'][-1])
    del s
    return (S + S.T) / 2, t_last


def dist(d, S):
    return float(np.sqrt(max(d @ S @ d, 0.0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--theta_root', type=Path, required=True)
    ap.add_argument('--systems_root', type=Path, required=True, help='folder holding the reference systems (.pt)')
    ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--ref_label', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, required=True)
    ap.add_argument('--lams', nargs='+', type=float, required=True)
    ap.add_argument('--ridge', type=float, default=1e-7)
    ap.add_argument('--csv', type=Path, required=True)
    a = ap.parse_args()

    rows, refs = [], {}                                    # refs[(seed, lam)] = (Theta_ref, Sigma)
    for seed in a.seeds:
        rc = find_config(a.theta_root, seed, a.ref_label)
        if rc is None:
            print(f'seed {seed}: reference config not found, skipped'); continue
        S, t_sys = sigma_last(a.systems_root, rc.name)
        for lam in a.lams:
            ref = load_theta(rc, lam, a.ridge, '')
            if ref is None:
                print(f'seed {seed} lam {lam:g}: reference solve missing'); continue
            t_ref, Th_ref, _ = ref
            if abs(t_ref - t_sys) > 1e-9:
                raise ValueError(f'seed {seed}: last Theta node t={t_ref} != last system node t={t_sys}')
            refs[(seed, lam)] = (Th_ref, S)
            scale = dist(Th_ref, S)
            for label in a.labels:
                cdir = find_config(a.theta_root, seed, label)
                if cdir is None:
                    continue
                for arm, suf in ARMS:
                    if lam == 0 and arm != 'full':
                        continue
                    got = load_theta(cdir, lam, a.ridge, suf)
                    if got is None:
                        continue
                    t_got, Th, al = got
                    if abs(t_got - t_ref) > 1e-9:
                        raise ValueError(f'{cdir.name}: last node t={t_got} != reference t={t_ref}')
                    d = dist(Th - Th_ref, S)
                    rows.append(dict(seed=seed, lam=lam, label=label, arm=arm, d_ref=d,
                                     rel_ref=d / scale if scale > 0 else float('nan'), alpha=al,
                                     ref_scale=scale, finite=bool(np.isfinite(Th).all())))

    floor = {}
    for lam in a.lams:
        ds = [dist(refs[(s1, lam)][0] - refs[(s2, lam)][0], refs[(s1, lam)][1])
              for s1, s2 in itertools.permutations(a.seeds, 2) if (s1, lam) in refs and (s2, lam) in refs]
        floor[lam] = (float(np.mean(ds)), len(ds)) if ds else (float('nan'), 0)

    a.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['seed', 'lam', 'label', 'arm', 'd_ref', 'rel_ref', 'alpha', 'ref_scale',
                                           'finite', 'cross_seed_floor'])
        w.writeheader()
        for r in rows:
            w.writerow({**r, 'cross_seed_floor': floor[r['lam']][0]})

    print(f'reference: {a.ref_label}, full C, same seed and lam. d_ref = std_x[log p_arm - log p_ref] at t = t_reg[-1] '
          f'(nats); floor = same between reference solves of two seeds (mean over pairs)')
    print(f"{'lam':>7} {'label':>22} {'arm':>5} {'d_ref mean':>10} {'min':>8} {'max':>8} {'rel mean':>9} "
          f"{'alpha':>7} {'#seeds':>6} {'floor':>8}")
    for lam in a.lams:
        for label in a.labels:
            for arm, _ in ARMS:
                rs = [r for r in rows if r['lam'] == lam and r['label'] == label and r['arm'] == arm]
                if not rs:
                    continue
                d = np.array([r['d_ref'] for r in rs]); rl = np.array([r['rel_ref'] for r in rs])
                al = np.nanmedian([r['alpha'] for r in rs]) if arm != 'full' else float('nan')
                nf = '' if all(r['finite'] for r in rs) else ' NONFINITE'
                print(f'{lam:7g} {label[-22:]:>22} {arm:>5} {d.mean():10.3g} {d.min():8.3g} {d.max():8.3g} '
                      f'{rl.mean():9.3g} {al:7.3g} {len(rs):6d} {floor[lam][0]:8.3g}{nf}')
    print(f'\nwrote {a.csv}')


if __name__ == '__main__':
    main()
