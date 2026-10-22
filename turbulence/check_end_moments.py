"""Did MGD reach its own targets at t=1? Transport vs. potential, per arm.

aux_moments stores, along the run, barphi_e (moments of the interpolant, = data moments
at t=1) and barphi_p (moments of the model samples x_k). If the model's statistics match
the data at t=1 but the quantiles of |z| do not, the transport did its job and the gap is
in what the statistics determine (coupling between coefficients). If they do not match,
the transport is the cause.

Usage (repo root on Jean Zay):
    python turbulence/check_end_moments.py
    python turbulence/check_end_moments.py --arms mass_L6_1b morletQ1_L6 --top 15
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
AUX = HERE / 'saved_results' / 'aux_moments'

sys.path.insert(0, str(HERE))
from compare_scalar_orthog_marginals import LABEL_GLOB                  # noqa: E402

# same arm registry as the marginal scripts, aux_moments file instead of the samples
GLOB = {a: g.removesuffix('.pt') + '_aux_moments.pt' for a, g in LABEL_GLOB.items()}


def main(arms, seed, top):
    for arm in arms:
        files = sorted(AUX.glob(GLOB[arm].format(seed=seed)), key=lambda f: f.stat().st_mtime)
        if not files:
            print(f'[{arm}] no aux_moments file'); continue
        d = torch.load(files[-1], map_location='cpu')
        e, p = d['barphi_e'][-1].double(), d['barphi_p'][-1].double()   # t = 1
        rel = (p - e).abs() / e.abs().clamp_min(1e-300)
        idx = torch.argsort(rel, descending=True)[:top]
        print(f'[{arm}]  {files[-1].name}')
        print(f'  {len(e)} statistics at t=1: rel err median {rel.median():.2e}  '
              f'max {rel.max():.2e}  n>1e-2 {int((rel > 1e-2).sum())}  n>1e-1 {int((rel > 1e-1).sum())}')
        print('  worst: ' + '  '.join(f'{int(i)}:{float(rel[i]):.1e}(p/e {float(p[i] / e[i]):+.3f})'
                                      for i in idx))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=['morletQ1_L6', 'mass_L6_1b', 'morletQ1Q3_L6_2b',
                                                   'tailq_L6_2b', 'mass_L6_2b'], choices=list(GLOB))
    ap.add_argument('--seed', type=int, default=900)
    ap.add_argument('--top', type=int, default=10)
    a = ap.parse_args()
    main(a.arms, a.seed, a.top)
