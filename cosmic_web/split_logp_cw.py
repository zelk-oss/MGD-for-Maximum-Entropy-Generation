#!/usr/bin/env python3
"""Split log p at t = 1 into the part from range C and the part from ker C, and measure cross-seed coherence of each.

At the last node K, C = (Sigma_K-1 + Sigma_K)/2 (the solver's last-cell walker covariance, from the saved regularised
system). In the metric of the data term (statistics scaled by s = sqrt(diag M_K)), Theta~ = s * Theta is split by the
orthogonal projector P onto range C~ (eigenvalues > --rel_tol x the largest):
    Theta_range = (P Theta~) / s,   Theta_ker = Theta - Theta_range      (dead / zero-M statistics go to the ker part)
so that log p(x) = Theta_range . phi(x) + Theta_ker . phi(x) exactly. For each part and each lam: mean over seed pairs of
the Pearson correlation over the n1 data maps (as seed_coherence_cw.py), and the std over the maps (mean over seeds).
Prediction if ker C is what blocks Theta_reg: range part coherence rises with lam, ker part stays ~0 and grows with lam.

Usage (repo root, Jean Zay, GPU node or CPU if phi is cached in results/phi/):
    python cosmic_web/split_logp_cw.py --arm base_real [--lams 0 1e-4 1e-2] [--seeds 900 901 902 903 904]
Output: results/split_logp/split_logp_<arm>.npz  (lams, pearson (lams, 3), std (lams, 3), dim_range (seeds,)); parts = range, ker, total
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch

import cw_common as cw
from seed_coherence_cw import coherence

SCR = Path(os.environ.get('SCRATCH', '/nonexistent')) / 'MGD-for-Maximum-Entropy-Generation' / 'cosmic_web'


def range_split(system_file, rel_tol):
    """(project, dim_range): project(theta) -> theta_range, from the last cell of a saved system."""
    s = torch.load(system_file, map_location='cpu', weights_only=False, mmap=True)
    C = (s['Sigma'][-2].double() + s['Sigma'][-1].double()) / 2
    C = (C + C.T) / 2
    Md = torch.diagonal(s['M'][-1].double())
    live = s.get('live')
    keep = (Md > 0) if live is None else (live[-1].bool() & live[-2].bool() & (Md > 0))
    sc = Md[keep].sqrt()
    ev, V = torch.linalg.eigh(C[keep][:, keep] / (sc[:, None] * sc[None, :]))
    Vr = V[:, ev > rel_tol * ev[-1]]                       # orthonormal basis of range C~
    keep_np, sc_np, Vr_np = keep.numpy(), sc.numpy(), Vr.numpy()

    def project(theta):
        out = np.zeros_like(theta)
        th = theta[keep_np] * sc_np
        out[keep_np] = (Vr_np @ (Vr_np.T @ th)) / sc_np
        return out
    return project, Vr.shape[1], float(s['t'][-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', default='base_real')
    ap.add_argument('--seeds', nargs='+', type=int, default=cw.SEEDS)
    ap.add_argument('--lams', nargs='+', type=float, default=None, help='default: every lam found for the first seed')
    ap.add_argument('--theta_dir', type=Path, default=None, help='default: $SCRATCH/.../cosmic_web/theta_reg/<arm>')
    ap.add_argument('--reg_dir', type=Path, default=None, help='default: $SCRATCH/.../cosmic_web/reg_system/<arm>')
    ap.add_argument('--rel_tol', type=float, default=1e-8)
    ap.add_argument('--data_path', default=None)
    ap.add_argument('--chunk', type=int, default=10)
    ap.add_argument('--out', type=Path, default=cw.RESULTS / 'split_logp')
    a = ap.parse_args()
    theta_dir = a.theta_dir or SCR / 'theta_reg' / a.arm
    reg_dir = a.reg_dir or SCR / 'reg_system' / a.arm
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    runs = cw.find_runs(a.arm, a.seeds)

    proj, phis, dims = {}, {}, []
    for s in a.seeds:
        exp_dir, cfg = runs[s]
        f_sys = reg_dir / f"{cfg['config_name']}.pt"
        assert f_sys.exists(), f'no saved system {f_sys}'
        proj[s], d, t_last = range_split(f_sys, a.rel_tol)
        dims.append(d)
        phis[s] = cw.get_phi(a.arm, s, exp_dir, cfg, device, a.chunk, a.data_path)[0]     # data maps (n1, r)
        print(f'seed {s}: last node t = {t_last:.6f}, dim range C = {d} of {phis[s].shape[1]}')

    first = runs[a.seeds[0]][1]['config_name']
    found = sorted({float(re.match(r'lam([^_]+)_ridge', f.name).group(1))
                    for f in (theta_dir / first).glob('lam*_ridge*.pt') if 'neg' not in f.name and 'NOTCONVERGED' not in f.name})
    lams = [l for l in found if a.lams is None or any(np.isclose(l, q) for q in a.lams)]

    P, S = [], []
    print(f'\n=== {a.arm}: log p of the {len(a.seeds)}-seed Theta_reg(t=1) on the same data maps, split range C / ker C')
    print(f'{"lam":>9} | {"Pearson range":>13} {"ker":>8} {"total":>8} | {"std range":>10} {"ker":>10} {"total":>10}  (nats)')
    for lam in lams:
        C = []                                            # (seeds, 3, maps)
        for s in a.seeds:
            cfg = runs[s][1]
            hits = [h for h in sorted((theta_dir / cfg['config_name']).glob(f'lam{lam:g}_ridge*.pt'))
                    if 'neg' not in h.name and 'NOTCONVERGED' not in h.name]
            assert hits, f'no lam{lam:g}_ridge*.pt for seed {s} in {theta_dir / cfg["config_name"]}'
            sol = torch.load(hits[0], map_location='cpu', weights_only=False)
            assert sol['config'] == cfg['config_name'] and np.isclose(float(sol['lam']), lam), (sol['config'], sol['lam'], lam)
            th = sol['Theta_reg'][-1].double().numpy()
            th_r = proj[s](th)
            C.append(np.stack([phis[s] @ th_r, phis[s] @ (th - th_r), phis[s] @ th]))
        C = np.stack(C)
        P.append(coherence(C)); S.append(C.std(2, ddof=1).mean(0))
        print(f'{lam:9g} | {P[-1][0]:13.3f} {P[-1][1]:8.3f} {P[-1][2]:8.3f} | {S[-1][0]:10.4g} {S[-1][1]:10.4g} {S[-1][2]:10.4g}')
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f'split_logp_{a.arm}.npz'
    np.savez_compressed(f, lams=np.array(lams), pearson=np.array(P), std=np.array(S), dim_range=np.array(dims),
                        seeds=np.array(a.seeds), parts=np.array(['range', 'ker', 'total']))
    print(f'-> {f}')


if __name__ == '__main__':
    main()
