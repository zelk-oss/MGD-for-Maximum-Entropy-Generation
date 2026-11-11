#!/usr/bin/env python3
"""Grid dependence of Theta_reg (2026-11-11): compare re-solves on every m-th node (resolve_theta_reg.py --node_stride m)
with the full-grid solve, for each (lam, eps of method A).

If the directions of C filled by one step of SDE noise (mu ~ 2 h sigma^2) drive Theta_reg, then C^-1 mdot there scales
like 1/h and Theta_reg changes with the stride at eps = 0; with eps above that tier it should not.

Reads <dir>/<config>/lam<lam>_ridge<ridge>_moment_mask[_cfl<eps>][_stride<m>].pt and prints, per (lam, eps, m > 1),
on the nodes common to both grids (same t values):
  whole  ||Th_m - Th_1|| / ||Th_1||   (Frobenius over all common nodes)
  end    same on common nodes with t >= --t_end
  last   same at the last node (t = t_reg[-1], what log p at t = 1 uses)
Usage: python codes/compare_stride_eps.py <theta dir>/<config> --lams 1e-4 1e-2 --eps 0 5e-4 5e-3 --strides 2 4 8
"""
import argparse
from pathlib import Path

import numpy as np
import torch


def load(d, lam, ridge, eps, m, tag):
    name = f'lam{lam:g}_ridge{ridge:g}{tag}' + (f'_cfl{eps:g}' if eps else '') + (f'_stride{m}' if m > 1 else '') + '.pt'
    f = d / name
    if not f.exists():
        return None
    o = torch.load(f, map_location='cpu', weights_only=False)
    return o['t_reg'].double().numpy(), o['Theta_reg'].double().numpy()


def rel(a, b):
    nb = np.linalg.norm(b)
    return np.linalg.norm(a - b) / nb if nb > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config_dir', type=Path)
    ap.add_argument('--lams', nargs='+', type=float, required=True)
    ap.add_argument('--eps', nargs='+', type=float, default=[0.0])
    ap.add_argument('--strides', nargs='+', type=int, default=[2, 4, 8])
    ap.add_argument('--ridge', type=float, default=1e-7)
    ap.add_argument('--tag', default='_moment_mask', help='mode/schedule/mask part of the file name')
    ap.add_argument('--t_end', type=float, default=0.99)
    a = ap.parse_args()
    print(f'{a.config_dir.name[:110]}\n{"lam":>8} {"eps":>8} {"m":>3} {"nodes":>6} {"whole":>9} {"end":>9} {"last":>9}')
    for lam in a.lams:
        for eps in a.eps:
            ref = load(a.config_dir, lam, a.ridge, eps, 1, a.tag)
            if ref is None:
                print(f'{lam:8g} {eps:8g}   1  missing full-grid file'); continue
            t1, T1 = ref
            for m in a.strides:
                got = load(a.config_dir, lam, a.ridge, eps, m, a.tag)
                if got is None:
                    print(f'{lam:8g} {eps:8g} {m:3d}  missing'); continue
                tm, Tm = got
                common, i1, im = np.intersect1d(t1, tm, return_indices=True)
                e = common >= a.t_end
                print(f'{lam:8g} {eps:8g} {m:3d} {len(common):6d} {rel(Tm[im], T1[i1]):9.3e} '
                      f'{rel(Tm[im][e], T1[i1][e]) if e.any() else float("nan"):9.3e} '
                      f'{rel(Tm[-1], T1[-1]) if np.isclose(tm[-1], t1[-1]) else float("nan"):9.3e}')


if __name__ == '__main__':
    main()
