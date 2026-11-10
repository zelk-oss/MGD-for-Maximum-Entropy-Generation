#!/usr/bin/env python3
"""Sign gate of the L_6 coefficients for one S&P arm (snp_M0_L6 / snp_M1_L6), at the cross-seed lambda.

Reads the xseed result of the arm (snp_xseed.sh), takes for every seed the unconstrained Theta_reg file at the selected
lambda (xs['theta_files'][window]) and counts the L_6 entries with Theta > 0 (code sign: Theta < 0 confines; exact
zeros are dead statistics, not violations), over ALL time nodes (the constrained solve of snp_l6neg.sh bounds every
node, T_MIN = 0, as in the turbulence runs). Writes <out>/gate_<arm>.json:
  {"violated": bool, "n_positive": int, "lam": float, "per_seed": {seed: {...}}}
and prints one line "GATE <arm>: VIOLATED|OK". The whole model is gated at once (never per seed), so all seeds end with
the same kind of Theta (all constrained or all unconstrained). Exit code 0 in both cases.

Usage (repo root):  python snp_experiment/l6_gate.py --arm snp_M0_L6 [--cols 0:7] [--window end0.99]
"""
import argparse
import json
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--cols', default='0:7', help='L_6 columns of Theta (L_6 is first: J + 1 = 7 at J = 6)')
    ap.add_argument('--window', default='end0.99')
    ap.add_argument('--xseed', type=Path, default=HERE / 'results' / 'xseed')
    ap.add_argument('--out', type=Path, default=HERE / 'results' / 'gate')
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.cols.split(':'))
    xs = torch.load(a.xseed / f'xseed_{a.arm}.pt', map_location='cpu', weights_only=False)
    lam = float(xs['selected'][a.window])
    per_seed, total = {}, 0
    for config, f in xs['theta_files'][a.window].items():
        d = torch.load(f, map_location='cpu', weights_only=False)
        assert float(d['lam']) == lam, (f, d['lam'], lam)
        Th, t = d['Theta_reg'].double()[:, lo:hi], d['t_reg'].double()
        pos = Th > 0
        n, n99 = int(pos.sum()), int((pos & (t >= 0.99)[:, None]).sum())
        per_seed[config] = dict(n_positive=n, n_positive_t99=n99, n_live=int((Th != 0).sum()), max=float(Th.max()),
                                first_t=float(t[pos.any(1)].min()) if n else None)
        total += n
        print(f'  {config[-40:]:40s} L_6 cols {lo}:{hi}: {n} positive of {int((Th != 0).sum())} live '
              f'(t >= 0.99: {n99}), max {float(Th.max()):.3e}')
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = a.out / f'.gate_{a.arm}.{os.getpid()}.tmp'                       # atomic: the array tasks all write the same content
    tmp.write_text(json.dumps(dict(violated=total > 0, n_positive=total, lam=lam,
                                                              window=a.window, cols=a.cols, per_seed=per_seed), indent=1))
    os.replace(tmp, a.out / f'gate_{a.arm}.json')
    print(f'GATE {a.arm}: {"VIOLATED" if total else "OK"} ({total} positive L_6 entries, lam {lam:g}, {len(per_seed)} seeds)')


if __name__ == '__main__':
    main()
