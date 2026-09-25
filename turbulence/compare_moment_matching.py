"""Compare moment matching across run variants (e.g. the n1/NT budget tests).

For every run whose label is in --labels, loads the saved aux moments
(saved_results/aux_moments/<config>_aux_moments.pt: barphi_e = interpolant /
target moments, barphi_p = walker moments, one row per SDE step, aligned with
t[1:]) and computes the same relative error as check_moments.plot_moment_matching:

    rel_err = 2 |barphi_e - barphi_p| / (|barphi_e| + |barphi_p|)

on the moments whose final target value barphi_e[-1] exceeds --threshold, and the
error on each moment's own scale,

    own = |barphi_e - barphi_p| / max_t |barphi_e|

which is the headline measure since 2026-09-25: the relative error blows up whenever a
target moment crosses zero (all 45 bulk spikes of long_full_sched3_reg1e-2 were such
crossings), the own-scale error does not. With ~200 moments the mean is carried by the
worst few, so the summary is median / p90 / max over moments.

Reports per variant: the final-step own-scale error (median / p90 / max, and moments
above --target, averaged over seeds with the across-seed spread), the p90 own-scale
error and the mean relative error per t-window; --top lists the worst moments (named
from the run's fitted potentials when experiments/<...>/fitted_potentials exists);
saves a figure with the p90 own-scale error vs t and the final-error distribution.

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
                   help='list the N moments with the largest final-step own-scale error (first seed)')
    p.add_argument('--target', type=float, default=1e-2,
                   help='own-scale error target: count moments above it at the final step')
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
    own = np.abs(e - p) / np.abs(e).max(0)
    return t, rel, own, int(keep.sum()), len(keep), e, p, idx


def moment_names(root, config, n_expected):
    """index -> potential name, from the run's fitted potentials; None if unavailable.

    The count is checked: without the fitted state (e.g. a local copy whose
    fitted_potentials/ is empty) the rebuild keeps every Scalar_*_gaussianK region and
    gives MORE statistics (301 vs 233 for long_full_*), which would mislabel every index.
    """
    try:
        import sys
        project = Path(__file__).resolve().parent.parent
        for d in (project, project / 'codes', project / 'data'):   # codes/__init__ needs codes/ too
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
        from codes.find_duplicate_statistics import statistic_labels
        labels, _ = statistic_labels(root, config)
    except Exception as exc:                  # no fitted_potentials here (e.g. local copy)
        print(f'    (no moment names: {type(exc).__name__}: {exc})')
        return None
    if labels is None or len(labels) != n_expected:
        print(f'    (no moment names: rebuilt potentials give {None if labels is None else len(labels)} '
              f'statistics, the run has {n_expected} -- fitted_potentials/ missing or incomplete here)')
        return None
    return labels


def print_outliers(e, p, rel, own, idx, top, names=None):
    """The `top` moments with the largest final-step own-scale error.

    rel = the relative error; |e(1)|/max|e| << 1 means the target ends near zero, so its
    relative error says little. sign flips = how often the target changes sign.
    """
    scale = np.abs(e).max(0)
    order = np.argsort(own[-1])[::-1][:top]
    print(f'    top {len(order)} final-step errors on the own scale (first seed):')
    print(f'      {"moment":>6} {"own":>9} {"rel":>7} {"e(1)":>11} {"p(1)":>11} '
          f'{"|e(1)|/max|e|":>13} {"sign flips":>10}  name')
    for j in order:
        flips = int((np.diff(np.sign(e[:, j])) != 0).sum())
        name = names[idx[j]] if names else ''
        print(f'      {idx[j]:>6d} {own[-1, j]:9.2e} {rel[-1, j]:7.3f} {e[-1, j]:11.3e} {p[-1, j]:11.3e} '
              f'{abs(e[-1, j]) / scale[j]:13.2e} {flips:10d}  {name}')


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
        finals, curves, rel_curves, t_ref = [], [], [], None
        print(f'\n=== {label}: {len(runs)} seeds {[s for s, _ in runs]}')
        for seed, config in runs:
            t, rel, own, n_keep, n_all, e, p, idx = load_rel_err(args.root, config, args.threshold)
            if t_ref is None:
                t_ref = t
                print(f'    {n_keep}/{n_all} moments above threshold, {len(t)} steps')
                if args.top:
                    print_outliers(e, p, rel, own, idx, args.top, moment_names(args.root, config, n_all))
            finals.append(own[-1])
            same = len(t) == len(t_ref)
            curves.append(np.percentile(own, 90, axis=1) if same else None)
            rel_curves.append(rel.mean(1) if same else None)
        finals = np.stack(finals)                             # (seeds, moments)
        per_seed = np.stack([np.median(finals, 1), np.percentile(finals, 90, axis=1),
                             finals.max(1), (finals > args.target).sum(1)], 1)
        m, sd = per_seed.mean(0), per_seed.std(0, ddof=1) if len(runs) > 1 else np.zeros(4)
        print(f'    final-step own-scale error:  median {m[0]:.3e} ± {sd[0]:.1e}   '
              f'p90 {m[1]:.3e} ± {sd[1]:.1e}   max {m[2]:.3e} ± {sd[2]:.1e}   '
              f'above {args.target:g}: {m[3]:.1f} ± {sd[3]:.1f} moments   (± = across seeds)')
        curves = [c for c in curves if c is not None]
        rel_curves = [c for c in rel_curves if c is not None]
        if curves:
            C, R = np.stack(curves), np.stack(rel_curves)
            print('    per t-window (seed-averaged):  p90 own-scale error | mean rel. error')
            for lo, hi in zip(WINDOWS[:-1], WINDOWS[1:]):
                w = (t_ref >= lo) & ((t_ref < hi) if hi < 1 else (t_ref <= hi))
                if w.any():
                    print(f'      [{lo:.2f},{hi:.2f})  {C[:, w].mean():.3e} | {R[:, w].mean():.3e}')
            med = np.median(C, 0)
            line, = axes[0].semilogy(t_ref, med, lw=1, label=f'{label} (median of {len(C)} seeds)')
            axes[0].fill_between(t_ref, C.min(0), C.max(0), color=line.get_color(), alpha=0.15)
        # counts, not density: density on log-spaced bins divides by the bin width,
        # which inflates the tiny-error bins by orders of magnitude
        axes[1].hist(finals.ravel(), bins=np.logspace(-9, 1, 90), histtype='step',
                     lw=1.5, label=label)

    for a in axes:
        a.grid(alpha=0.2, which='both')
    axes[0].axhline(args.target, color='0.5', lw=0.8, ls='--')
    axes[1].axvline(args.target, color='0.5', lw=0.8, ls='--')
    axes[0].set_xlabel('SDE time t'); axes[0].set_ylabel('p90 over moments of |e-p| / max_t|e|')
    axes[0].set_title('moment matching along the SDE (band = min/max over seeds; dashed = target)',
                      fontsize=9)
    axes[0].legend(fontsize=8)
    axes[1].set_xscale('log'); axes[1].set_yscale('log'); axes[1].set_ylabel('moments')
    axes[1].set_xlabel('final-step own-scale error (all moments, all seeds)')
    axes[1].set_title('final moment mismatch'); axes[1].legend(fontsize=8)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f'\nfigure: {out}')


if __name__ == '__main__':
    main()
