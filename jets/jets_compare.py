#!/usr/bin/env python3
"""Jets (Modane record): are the runs good, and what does the scalar potential change? Several arms x several seeds.

Two questions (2026-10-28), same statistics as the turbulence comparison (turbulence/cross_scale_compare.py, reused
as is: marginal, spectrum, S_p, flatness, zeta4/zeta2, increment and power-increment PDFs, C_pq at tau*=20 and the full
C_pq maps, wavelet bands, burst persistence, extremes per signal, Scat4 statistics):

 (a) run vs data, per arm. Arm curve = average over its seeds of the deviation from the data (log ratio for positive
     statistics, difference otherwise). Noise per point: sigma_i^2 = sigma_ref^2 + s_i^2 / n_i, where sigma_ref = the
     data's own sampling noise (random halves of the data) and s_i = spread over the n_i seeds of arm i (sampling +
     seed noise of the model). "Recovered" = |deviation| <= 2 sigma_i.
 (b) arm vs arm (default: full vs noscalar). Per point, z = (m_a - m_b) / sqrt(s_a^2/n_a + s_b^2/n_b), m = seed-averaged
     deviation from the data (the data cancels in the difference). |z| > 2 = the arms differ by more than their seed
     noise (with 5 seeds per arm this is a ~5-10% false-positive level per point, so look at whole families). Among
     the points that differ, which arm is closer to the data.

The data reference is rebuilt from the runs' config.json (jets/jet_data.load_modane_windows, exactly as
run_jet_SDE.py): all arms must share subseries_len / target_len / n1 / n1_spread. The old 6d216e41 runs (256 -> 256,
first 5000 windows) are a different data set: compare them in a separate call.

Usage (repo root, Jean Zay; samples are not pulled locally):
  python jets/jets_compare.py                                    # jets_full vs jets_noscalar, seeds 500-504
  python jets/jets_compare.py --plot_only                        # re-plot from curves.npz
  python jets/jets_compare.py --labels 'jetsynth_Re_number929_*terms6d216e41_2026072*' --names old --seeds 0 \\
      --out jets/figures/jets_compare_old                       # a glob instead of a label: runs without --label
Outputs (small): jets/figures/jets_compare/{curves.npz, summary.csv/png (a), arm_difference.csv/png (b), figures of
cross_scale_compare.plot}. Cache: <out>/cache/, one file per exact saved run (cross_scale_compare.CACHE_VERSION).
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent                      # jets/
ROOT = HERE.parent
for p in (ROOT / 'turbulence', ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data',
          HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cross_scale_compare as csc                          # noqa: E402
import run_comparison as rc                                 # noqa: E402
import paper_figs_ensemble as pfe                           # noqa: E402
from check_moments import log_spaced_taus                   # noqa: E402
from filters_bank import return_Filters                     # noqa: E402
from potentials_builder import get_1d_potentials            # noqa: E402
from jet_data import load_modane_windows                    # noqa: E402

ARMS = {'full (with Scalar_morlet)': 'jets_full', 'noscalar': 'jets_noscalar'}
SEEDS = [500, 501, 502, 503, 504]
DATA_KEYS = ('subseries_len', 'target_len', 'n1', 'n1_spread')
SEED_RE = re.compile(r'_seed_(\d+)_')
TS_RE = re.compile(r'(\d{8}_\d{4})$')


# ------------------------------------------------------------------- runs / data ----
def find_seed_configs(root, spec, seeds):
    """{seed: config name} of the newest saved run per seed. spec = a --label, or a glob on the samples file names
    (runs saved without --label)."""
    out = {}
    for s in seeds:
        if '*' in spec or '?' in spec:
            hits = [c for c in rc.find_configs(spec if spec.endswith('*') else spec + '*', root=root)
                    if (m := SEED_RE.search(c)) and int(m[1]) == s]
            out[s] = max(hits, key=lambda c: (m := TS_RE.search(c)) and m[1] or '') if hits else None
        else:
            out[s] = rc._find_config(root, spec, s)
            if out[s] is None:                              # run without --timestamp (manual runs): name ends in label
                hits = rc.find_configs(f'*_seed_{s}_terms*_{spec}*', root=root)
                hits = [c for c in hits if re.search(rf'_seed_{s}_terms[0-9a-f]{{8}}_{re.escape(spec)}$', c)]
                out[s] = hits[-1] if hits else None
    return out


def reference_data(cfg):
    """x1 exactly as run_jet_SDE.py builds it, from a run's config.json: (n1, T) float32."""
    x1, _ = load_modane_windows(cfg['subseries_len'], cfg['target_len'], n=cfg['n1'],
                                spread=bool(cfg.get('n1_spread') or False))
    return x1.reshape(x1.shape[0], -1).float()


def _data_params(cfg):
    return tuple(bool(cfg.get(k) or False) if k == 'n1_spread' else cfg.get(k) for k in DATA_KEYS)


# ---------------------------------------------------------------------- compute ----
def compute(args):
    root = Path(args.root)
    t0 = time.time()
    arms = {}                                               # name -> {seed: run dict}
    for name, spec in zip(args.names, args.labels):
        cfgs = find_seed_configs(root, spec, args.seeds)
        found = {s: c for s, c in cfgs.items() if c is not None}
        print(f'{name} ({spec}): seeds {sorted(found)}' + (f', missing {sorted(set(cfgs) - set(found))}'
                                                            if len(found) < len(cfgs) else ''))
        assert found, f'{name}: no saved run for {spec} under {root}'
        arms[name] = {s: rc.load_config(c, root, label=spec, extras=False) for s, c in found.items()}
    runs_all = [r for a in arms.values() for r in a.values()]
    params = {_data_params(r['cfg']) for r in runs_all}
    assert len(params) == 1, f'the runs model different data sets {DATA_KEYS}: {params}; compare them separately'
    cfg0 = runs_all[0]['cfg']
    x1 = reference_data(cfg0)
    for r in runs_all:
        assert r['xt'].shape[-1] == x1.shape[-1], f'{r["config"]}: length {r["xt"].shape[-1]} vs data {x1.shape[-1]}'
    T = x1.shape[-1]
    M, J, Q = int(cfg0.get('M', T)), int(cfg0['J']), int(cfg0['Q'])
    filters, filters_phi = return_Filters(M, J, 1, include_phi=True, wav_type=cfg0.get('wav_type', 'morlet'))
    filters_Q = return_Filters(M, J, Q, wav_type=cfg0.get('wav_type', 'morlet'))
    scat4 = get_1d_potentials(list(csc.SCAT4), J, filters, Q, filters_Q=filters_Q, filters_Phi=filters_phi)
    L = 1.05 * x1.abs().max().item()
    max_tau = T // 2
    ctx = dict(marg_bins=np.linspace(-L, L, pfe.N_BINS), tau_star=args.tau_star,
               taus_f=log_spaced_taus(max_tau, pfe.FLAT_POINTS),
               cpq_taus=np.array(sorted(set(log_spaced_taus(max_tau, args.cpq_points)) |
                                        {t for t in csc.CUT_TAUS if t < max_tau})),
               persist_D=log_spaced_taus(max_tau, 20), wav_J=min(6, J), wav_Q=3,
               x_ref=x1, scat4=scat4, batch=args.batch)
    print(f'data {tuple(x1.shape)} {dict(zip(DATA_KEYS, _data_params(cfg0)))}; C_pq lags {ctx["cpq_taus"].tolist()}')

    cache = args.out / 'cache'
    cache.mkdir(parents=True, exist_ok=True)

    def timed(name, x, key):                                # as cross_scale_compare.compute: cache per exact run
        tag = ''.join(ch if ch.isalnum() else '_' for ch in f'{name}__{key}')
        f = cache / f'v{csc.CACHE_VERSION}_{tag}.npz'
        if f.exists():
            print(f'  {name:36s} cached', flush=True)
            return dict(np.load(f))
        t = time.time()
        s = csc.all_stats(x, ctx)
        np.savez(f, **s)
        print(f'  {name:36s} {tuple(x.shape)}  {time.time() - t:6.1f} s', flush=True)
        return s

    data_key = 'modane_' + '_'.join(str(v) for v in _data_params(cfg0))
    data = timed('data', x1, data_key)
    stats = {a: {s: timed(f'{a} seed {s}', r['xt'].float(), r['config']) for s, r in runs.items()}
             for a, runs in arms.items()}
    rng = np.random.default_rng(0)
    half_dev = {k: [] for k in csc.KINDS}
    for rep in range(args.n_rep):
        perm = rng.permutation(len(x1))
        a = timed(f'data halves {rep + 1}/{args.n_rep} (A)', x1[perm[: len(x1) // 2]], data_key)
        b = timed(f'data halves {rep + 1}/{args.n_rep} (B)', x1[perm[len(x1) // 2:]], data_key)
        for k, (_, kind) in csc.KINDS.items():
            half_dev[k].append(csc.deviation(a[k], b[k], 'ratio' if kind == 'pdf' else kind))

    names = list(arms)
    out = {'T': T, 'tau_star': args.tau_star, 'run_names': np.array(names),
           'n_seeds': np.array([len(arms[a]) for a in names]),
           'marg_bins': ctx['marg_bins'], 'taus_f': ctx['taus_f'], 'cpq_taus': ctx['cpq_taus'],
           'persist_D': ctx['persist_D'], 'skew_taus': log_spaced_taus(max_tau, pfe.SKEW_POINTS)}
    for k, (_, kind) in csc.KINDS.items():
        sig_ref = np.nanstd(np.stack(half_dev[k]), axis=0) / 2.0          # data noise, n_ref = all data
        out[f'data__{k}'], out[f'sigref__{k}'] = data[k], sig_ref
        var_mean = []
        for i, a in enumerate(names):
            dev = np.stack([csc.deviation(stats[a][s][k], data[k], kind) for s in stats[a]])
            n = len(dev)
            with np.errstate(invalid='ignore'):
                m = np.nanmean(dev, 0)
                sd = np.nanstd(dev, 0, ddof=1) if n > 1 else np.zeros_like(m)
            # arm curve back in statistic units: geometric mean over seeds for ratios, mean for differences
            with np.errstate(over='ignore', invalid='ignore'):
                out[f'run{i}__{k}'] = data[k] + m if kind == 'diff' else data[k] * np.exp(m)
            out[f'mdev{i}__{k}'], out[f'sdev{i}__{k}'] = m, sd
            out[f'sigma{i}__{k}'] = np.sqrt(sig_ref ** 2 + np.nan_to_num(sd) ** 2 / n)
            var_mean.append(np.nan_to_num(sd) ** 2 / n)
            for s in stats[a]:
                out[f'arm{i}_seed{s}__{k}'] = stats[a][s][k]
        out[f'sigma__{k}'] = np.sqrt(sig_ref ** 2 + np.mean(var_mean, 0))  # common band for the figures
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / 'curves.npz', **out)
    print(f'-> {args.out / "curves.npz"}  ({time.time() - t0:.0f} s total)')


# --------------------------------------------------------------- arm vs arm (b) ----
def arm_difference(z, out_dir, pair=(0, 1)):
    names = list(z['run_names'])
    ia, ib = pair
    a, b = names[ia], names[ib]
    na, nb = int(z['n_seeds'][ia]), int(z['n_seeds'][ib])
    rows = []
    for fam in dict.fromkeys(f for f, _ in csc.KINDS.values()):
        zs, closer_a, dabs = [], [], []
        for k in [k for k, (f, _) in csc.KINDS.items() if f == fam]:
            ma, mb = z[f'mdev{ia}__{k}'], z[f'mdev{ib}__{k}']
            sd = np.sqrt(z[f'sdev{ia}__{k}'] ** 2 / na + z[f'sdev{ib}__{k}'] ** 2 / nb)
            with np.errstate(divide='ignore', invalid='ignore'):
                zz = (ma - mb) / sd
            ok = np.isfinite(zz)
            zs.append(zz[ok].ravel()); closer_a.append((np.abs(ma) < np.abs(mb))[ok].ravel())
            dabs.append(np.abs(ma - mb)[ok].ravel())
        zs, closer_a, dabs = map(np.concatenate, (zs, closer_a, dabs))
        sig = np.abs(zs) > 2
        rows.append(dict(family=fam, n_points=len(zs), frac_differ=float(sig.mean()) if len(zs) else np.nan,
                         frac_a_closer_among_differ=float(closer_a[sig].mean()) if sig.any() else np.nan,
                         median_abs_z=float(np.median(np.abs(zs))) if len(zs) else np.nan,
                         median_abs_diff=float(np.median(dabs)) if len(zs) else np.nan))
    with open(out_dir / 'arm_difference.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([f'# A = {a} ({na} seeds), B = {b} ({nb} seeds); differ = |m_A - m_B| > 2 seed sigma of the '
                    'difference; A closer = |deviation of A from data| < that of B'])
        w.writerow(list(rows[0]))
        for r in rows:
            w.writerow([r['family'], r['n_points']] + [f'{r[c]:.3f}' for c in list(r)[2:]])
    fams = [r['family'] for r in rows]
    vals = np.array([[r['frac_differ'], r['frac_a_closer_among_differ']] for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(fams) + 1.8))
    im = ax.imshow(vals, cmap='PuOr', vmin=0, vmax=1, aspect='auto')
    for (r, c), v in np.ndenumerate(vals):
        ax.text(c, r, '-' if not np.isfinite(v) else f'{v:.2f}', ha='center', va='center', fontsize=9)
    ax.set_xticks([0, 1], ['fraction of points\nwhere A and B differ', f'among those: fraction\nwhere A is closer'],
                  fontsize=8)
    ax.set_yticks(range(len(fams)), fams, fontsize=9)
    ax.set_title(f'A = {a}  vs  B = {b}\n(differ = > 2 seed sigma; 0.5 in column 2 = no preference)', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / 'arm_difference.png', dpi=130)
    plt.close(fig)
    print(open(out_dir / 'arm_difference.csv').read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', nargs='+', default=list(ARMS.values()), help='--label of each arm, or a glob')
    ap.add_argument('--names', nargs='+', default=list(ARMS))
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--n_rep', type=int, default=6, help='random data halves for sigma_ref')
    ap.add_argument('--cpq_points', type=int, default=14)
    ap.add_argument('--tau_star', type=int, default=20)
    ap.add_argument('--batch', type=int, default=200, help='signals per Scat4 forward batch')
    ap.add_argument('--root', default=str(HERE), help='folder holding saved_results/ and experiments/')
    ap.add_argument('--out', type=Path, default=HERE / 'figures' / 'jets_compare')
    ap.add_argument('--plot_only', action='store_true')
    args = ap.parse_args()
    assert len(args.labels) == len(args.names), '--labels and --names must have the same length'
    args.sigma_note = 'data halves + seed spread of each arm / sqrt(n seeds)'
    if not args.plot_only:
        compute(args)
    csc.plot(args)                                          # all figures + summary.csv/png (a), per-arm sigma
    z = np.load(args.out / 'curves.npz', allow_pickle=True)
    if len(z['run_names']) > 1:
        arm_difference(z, args.out)                         # (b): first arm vs second


if __name__ == '__main__':
    main()
