"""Compare moment matching across run variants (e.g. the n1/NT budget tests).

For every run whose label is in --labels, loads the saved aux moments
(saved_results/aux_moments/<config>_aux_moments.pt: barphi_e = interpolant /
target moments, barphi_p = walker moments, one row per SDE step, aligned with
t[1:]) and computes the same relative error as check_moments.plot_moment_matching:

    rel_err = 2 |barphi_e - barphi_p| / (|barphi_e| + |barphi_p|)

on the moments whose final target value barphi_e[-1] exceeds --threshold.

Reports per variant: the final-step error (mean / median / 90th percentile over
moments, averaged over seeds, with the across-seed spread) and the mean error per
t-window; saves a figure with the mean error vs t and the final-error distribution.

Usage (from turbulence/):
    python compare_moment_matching.py --labels lamtune_full_nt33000 lamtune_full_n1_8500 \
        --patterns '*nt40000_n1_5000_lam5e-07_*_20260728_1232'
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAME_RE = re.compile(r'_seed_(\d+)_terms([0-9a-f]{8})_(.+)_(\d{8}_\d{4})$')
WINDOWS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--labels', nargs='*', default=[], help='run labels to compare')
    p.add_argument('--patterns', nargs='*', default=[],
                   help='glob patterns on config names, one variant each (for unlabeled runs, '
                        'e.g. the 2026-07-28 batch: "*nt40000_n1_5000_lam5e-07_*_20260728_1232")')
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent,
                   help='folder holding saved_results/ (default: this script\'s folder)')
    p.add_argument('--threshold', type=float, default=1e-8,
                   help='keep moments with barphi_e[-1] > threshold (as --moment_threshold)')
    p.add_argument('--out', type=Path, default=None,
                   help='figure path (default <root>/figures/moment_matching_<labels>.png)')
    p.add_argument('--top', type=int, default=15,
                   help='list the N moments with the largest final-step error (first seed)')
    return p.parse_args()


SEED_RE = re.compile(r'_seed_(\d+)_')


def find_runs(root, label=None, pattern=None):
    runs = []
    aux_dir = root / 'saved_results' / 'aux_moments'
    for f in sorted(aux_dir.glob(f'{pattern}_aux_moments.pt' if pattern else '*_aux_moments.pt')):
        config = f.name[:-len('_aux_moments.pt')]
        if pattern:
            m = SEED_RE.search(config)
            if m:
                runs.append((int(m[1]), config))
        else:
            m = NAME_RE.search(config)
            if m and m[3] == label:
                runs.append((int(m[1]), config))
    return runs


def load_rel_err(root, config, threshold):
    aux = torch.load(root / 'saved_results' / 'aux_moments' / f'{config}_aux_moments.pt',
                     map_location='cpu', weights_only=False)
    e, p = aux['barphi_e'].double().numpy(), aux['barphi_p'].double().numpy()
    t_path = root / 'saved_results' / 'sampling_times' / f'{config}.pt'
    if not t_path.exists():                               # runs saved before the .pt convention
        t_path = t_path.with_suffix('')
    t = torch.load(t_path, map_location='cpu', weights_only=False).double().numpy()
    t = t[1:len(e) + 1]                                   # one aux row per step, aligned with t[1:]
    keep = e[-1] > threshold
    idx = np.flatnonzero(keep)                            # original moment indices
    e, p = e[:, keep], p[:, keep]
    rel = 2 * np.abs(e - p) / (np.abs(e) + np.abs(p))
    return t, rel, int(keep.sum()), len(keep), e, p, idx


def print_outliers(e, p, rel, idx, top):
    """Largest final-step errors, with the target's size relative to its own path.

    |e(1)| / max_t |e(t)| << 1 means the moment ends near zero, so the relative
    error there is dominated by the denominator, not by a real mismatch.
    abs/scale = |e(1) - p(1)| / max_t |e(t)| is the error on the moment's own scale.
    """
    scale = np.abs(e).max(0)
    order = np.argsort(rel[-1])[::-1][:top]
    print(f'    top {len(order)} final-step errors (first seed):')
    print(f'      {"moment":>6} {"rel":>7} {"e(1)":>11} {"p(1)":>11} '
          f'{"|e(1)|/max|e|":>13} {"abs/scale":>10} {"sign(e) flips":>13}')
    for j in order:
        flips = int((np.diff(np.sign(e[:, j])) != 0).sum())
        print(f'      {idx[j]:>6d} {rel[-1, j]:7.3f} {e[-1, j]:11.3e} {p[-1, j]:11.3e} '
              f'{abs(e[-1, j]) / scale[j]:13.2e} {abs(e[-1, j] - p[-1, j]) / scale[j]:10.2e} '
              f'{flips:13d}')
    s = np.abs(e[-1] - p[-1]) / scale
    print(f'    final abs. error / max_t|e|:  mean {s.mean():.3e}   median {np.median(s):.3e}   '
          f'p90 {np.percentile(s, 90):.3e}')


def main():
    args = parse_args()
    variants = [(l, dict(label=l)) for l in args.labels] + \
               [(p.strip('*_') or p, dict(pattern=p)) for p in args.patterns]
    if not variants:
        raise SystemExit('give --labels and/or --patterns')
    tag = '_vs_'.join(re.sub(r'[^A-Za-z0-9.-]+', '-', v) for v, _ in variants)[:150]
    out = args.out or args.root / 'figures' / f'moment_matching_{tag}.png'

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for label, sel in variants:
        runs = find_runs(args.root, **sel)
        if not runs:
            print(f'\n=== {label}: no runs with aux moments found under {args.root}')
            continue
        finals, curves, t_ref = [], [], None
        print(f'\n=== {label}: {len(runs)} seeds {[s for s, _ in runs]}')
        for seed, config in runs:
            t, rel, n_keep, n_all, e, p, idx = load_rel_err(args.root, config, args.threshold)
            if t_ref is None:
                t_ref = t
                print(f'    {n_keep}/{n_all} moments above threshold, {len(t)} steps')
                if args.top:
                    print_outliers(e, p, rel, idx, args.top)
            finals.append(rel[-1])
            curves.append(rel.mean(1) if len(t) == len(t_ref) else None)
        finals = np.stack(finals)                             # (seeds, moments)
        per_seed = np.stack([finals.mean(1), np.median(finals, 1),
                             np.percentile(finals, 90, axis=1)], 1)
        m, sd = per_seed.mean(0), per_seed.std(0, ddof=1) if len(runs) > 1 else np.zeros(3)
        print(f'    final-step rel. error:  mean {m[0]:.3e} ± {sd[0]:.1e}   '
              f'median {m[1]:.3e} ± {sd[1]:.1e}   p90 {m[2]:.3e} ± {sd[2]:.1e}   (± = across seeds)')
        curves = [c for c in curves if c is not None]
        if curves:
            C = np.stack(curves)
            print('    mean rel. error per t-window (seed-averaged):')
            for lo, hi in zip(WINDOWS[:-1], WINDOWS[1:]):
                w = (t_ref >= lo) & ((t_ref < hi) if hi < 1 else (t_ref <= hi))
                if w.any():
                    print(f'      [{lo:.2f},{hi:.2f})  {C[:, w].mean():.3e}')
            med = np.median(C, 0)
            line, = axes[0].semilogy(t_ref, med, lw=1, label=f'{label} (median of {len(C)} seeds)')
            axes[0].fill_between(t_ref, C.min(0), C.max(0), color=line.get_color(), alpha=0.15)
        # counts, not density: density on log-spaced bins divides by the bin width,
        # which inflates the tiny-error bins by orders of magnitude
        axes[1].hist(finals.ravel(), bins=np.logspace(-8, np.log10(2), 90), histtype='step',
                     lw=1.5, label=label)

    axes[0].set_xlabel('SDE time t'); axes[0].set_ylabel('mean relative error over moments')
    axes[0].set_title('moment matching along the SDE (band = min/max over seeds)')
    axes[0].legend(fontsize=8)
    axes[1].set_xscale('log'); axes[1].set_yscale('log'); axes[1].set_ylabel('moments')
    axes[1].set_xlabel('final-step relative error (all moments, all seeds)')
    axes[1].set_title('final moment mismatch'); axes[1].legend(fontsize=8)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f'\nfigure: {out}')


if __name__ == '__main__':
    main()
