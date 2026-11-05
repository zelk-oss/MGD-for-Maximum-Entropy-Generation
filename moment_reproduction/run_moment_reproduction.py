#!/usr/bin/env python3
"""Compute moment_reproduction results as a script (SLURM, H100), one model per call; the notebook only loads
and plots them (load_results). Same computation as the notebook's compute cell: load_model (first seed) ->
compute_moment_reproduction(pots, x1, xs, EXTRA, cfg).

Output: moment_reproduction/results/<group slug>/<model slug>.pkl =
    {'group', 'name', 'label', 'config_name', 'MJQ', 'imposed', 'extra', 'df', 'excluded', 'seconds'}

Usage (repo root, Jean Zay):
    python moment_reproduction/run_moment_reproduction.py --list            # numbered task list
    python moment_reproduction/run_moment_reproduction.py --task 3          # one model (SLURM array index)
    python moment_reproduction/run_moment_reproduction.py --task all        # everything, sequentially
    bash moment_reproduction/launch_moment_reproduction.sh                  # SLURM array, H100 dev QoS
Edit GROUPS / EXTRA below (not in the notebook).
"""
import argparse
import pickle
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

# ── MODELS ── group -> {row name: (results root, --label of the run)}; rows of one group share (M, J, Q).
GROUPS = {
    'turbulence (M=256, J=8, Q=3)': {
        'full (Scat4)':      (ROOT / 'turbulence', 'turbulence_ggd_mass_final'),
        'M3':                (ROOT / 'turbulence', 'turbulence_nested_M3'),
        'A (bicw)':          (ROOT / 'turbulence', 'turbulence_nested_M3_bicw'),
        'B (bicw_allmass)':  (ROOT / 'turbulence', 'turbulence_nested_M3_bicw_allmass'),
        'C (bicw_rich)':     (ROOT / 'turbulence', 'turbulence_nested_M3_bicw_rich'),
        'B + Scat4':         (ROOT / 'turbulence', 'turbulence_nested_M3_bicw_allmass_scat4'),
    },
    'jets (M=256, J=8, Q=1)': {
        'jets full':              (ROOT / 'jets', 'jets_full'),
        'no scalar':              (ROOT / 'jets', 'jets_noscalar'),
        'no Scat3/4':             (ROOT / 'jets', 'jets_scat2_scalar'),
        'no Scat3/4, no scalar':  (ROOT / 'jets', 'jets_scat2'),
    },
}

# ── EXTRA (non-imposed) STATISTICS ── evaluated on every model; a family a model imposes stays imposed.
EXTRA = [
    'L_6',                                      # 6th-order marginal moment
    'Scattering_First_Order',                   # E|W x| per channel
    'Scattering_Second_Order',                  # E|W |W x||: scale-to-scale envelope
    'Scattering_Third_Order_Real',              # cross-scale phase (run's Q bank)
    'Scattering_Third_Order_Imag',
    'Scattering_Fourth_Order_Mod2_Real_Q1',     # the Scat4 imposed by mass_final / B + Scat4 / jets full
    'Scattering_Fourth_Order_Mod2_Imag_Q1',
    'Scalar_morlet_gaussianK',                  # marginals, Q=1 bank
    'Scalar_psi_gaussianK',                     # marginals, Q bank of the run (= morlet at Q=1: jets duplicate)
    'Scalar_morlet_gaussianK_imag',             # marginals of the imaginary parts
    'Scalar_psi_gaussianK_imag',
    'Scalar_Q4_gaussianK',                      # marginals, Q=4 / Q=5 banks at the run's J (empty wavelets dropped)
    'Scalar_Q5_gaussianK',
]

OUT = HERE / 'results'


def slug(s):
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')


def tasks():
    return [(g, n, root, lab) for g, models in GROUPS.items() for n, (root, lab) in models.items()]


def run_one(group, name, root, label):
    from moment_reproduction import load_model, compute_moment_reproduction, DEVICE
    t0 = time.time()
    cfg, pots, x1, xs = load_model(label, root)
    extra = [f for f in EXTRA if f not in pots]
    print(f'{group} / {name}: {cfg["config_name"]}  n={len(x1)} n_s={len(xs)}  device {DEVICE}\n'
          f'  imposed: {list(pots)}\n  extra:   {extra}', flush=True)
    df, excl = compute_moment_reproduction(pots, x1, xs, EXTRA, cfg)
    dt = time.time() - t0
    print(f'  excluded (zero variance on data): {list(excl.label)}  ({dt:.0f} s)', flush=True)
    dest = OUT / slug(group) / f'{slug(name)}.pkl'
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.pkl.tmp')
    with open(tmp, 'wb') as fh:
        pickle.dump(dict(group=group, name=name, label=label, config_name=cfg['config_name'],
                         MJQ=(cfg['M'], cfg['J'], cfg['Q']), imposed=list(pots), extra=extra,
                         df=df, excluded=excl, seconds=dt), fh)
    tmp.replace(dest)
    print(f'  -> {dest}', flush=True)


def load_results(groups=None):
    """(results, excluded) as the notebook's compute cell builds them: results[group][name] = df,
    excluded[name] = df, in GROUPS order; models without a results file are skipped (and listed)."""
    results, excluded, missing = {}, {}, []
    for group, models in (groups or GROUPS).items():
        results[group], shapes = {}, set()
        for name in models:
            f = OUT / slug(group) / f'{slug(name)}.pkl'
            if not f.exists():
                missing.append(f'{group} / {name}')
                continue
            with open(f, 'rb') as fh:
                r = pickle.load(fh)
            results[group][name], excluded[name] = r['df'], r['excluded']
            shapes.add(tuple(r['MJQ']))
        assert len(shapes) <= 1, f'{group}: models with different (M, J, Q) {shapes}: split the group'
    if missing:
        print('no results file (not run yet / failed):\n  ' + '\n  '.join(missing))
    return results, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--task', help="index from --list, or 'all'")
    args = ap.parse_args()
    T = tasks()
    if args.list or args.task is None:
        for i, (g, n, _, lab) in enumerate(T):
            print(f'{i:3d}  {g} / {n}  ({lab})')
        return
    for g, n, root, lab in (T if args.task == 'all' else [T[int(args.task)]]):
        run_one(g, n, root, lab)


if __name__ == '__main__':
    main()
