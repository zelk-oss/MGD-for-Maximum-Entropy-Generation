#!/usr/bin/env python3
"""Do marginal statistics replace Scat4? Single-scale AND cross-scale statistics of several runs vs the data.

Question (2026-10-28): once the imposed moments are matched and enough marginal moments are imposed, does MGD
recover every statistic of the turbulence data without the 4th-order scattering terms (Scat4)? The runs compared by
default (M=256, seed 900): nested M3 (unconstrained scalar windows), A = constrained-BIC windows, B = A + all window
masses, C = B + up to 6 windows, a second tail cut at q.999 and the same marginals of the imaginary parts
(turbulence_nested_M3_bicw_rich, 2026-10-28), and the run WITH Scat4 (turbulence_ggd_mass_final) as the reference.
Cache: turbulence/figures/cross_scale_M3/cache/, one file per exact saved run (see CACHE_VERSION).

Statistics (families), for the data and for each run's final samples:
  - the 9 paper panels (paper_figs_ensemble.curves = codes/paper_figures.make_all_panels as arrays): marginal,
    spectrum, structure functions S_4,6,8, flatness slope, increment PDFs (tau = 1,4,16,64), C4 / C6 / C42 at
    tau* = 20, skewness of power increments vs tau (time asymmetry);
  - scaling-exponent ratio zeta_4/zeta_2 = dlog S4 / dlog S2 (as check_moments.scaling_exponent_ratio, recomputed
    here without its module-level cache);
  - power-increment PDFs (check_moments.pdf_energy_increments, time asymmetry);
  - wavelet bands (run_comparison.compute_stats): energy, flatness, sparsity per band;
  - cross-scale maps C_pq(tau, tau') = <|du_tau|^p |du_tau'|^q> / (S_p(tau) S_q(tau')) on log-spaced lags, for
    (p,q) = (2,2) [C4], (3,3) [C6], (4,2) [C42], (4,4), (3,1) (same definition and common-length truncation as
    check_moments.cross_structure_function / C_pq_structure);
  - burst persistence in time: corr(log|du_tau(t)|, log|du_tau(t+D)|) vs D, tau = 1, 4, 16;
  - extremes per signal: quantiles of the largest |du_tau| of each signal (run_comparison.event_scores, data sigma);
  - Scat4 statistics themselves (Scattering_Fourth_Order_Mod2_Real_Q1 / _Imag_Q1, unfitted, mean over signals).

Noise (A and B have one seed): sigma^2 = sigma_ref^2 + sigma_seed^2 per point, where
  sigma_ref  = spread of (half A - half B) over random halves of the data / 2  (the reference data's own sampling
               noise for n_ref = all data series: Var(A - B) = 4 sigma^2 / n),
  sigma_seed = spread over the seeds of --seed_label (nested M3, 900-904): the run-side noise (sampling + seed),
               assumed to hold for the other runs too.
Deviation = log(run / data) for positive statistics, run - data otherwise; a point is "recovered" if
|deviation| <= 2 sigma. summary.csv: fraction recovered and median |z| per family and run.

Usage (repo root, Jean Zay, CPU; samples are not pulled locally):
  python turbulence/cross_scale_compare.py                       # compute + plot
  python turbulence/cross_scale_compare.py --plot_only           # re-plot from curves.npz
Outputs (small): turbulence/figures/cross_scale_M3/{curves.npz, summary.csv, *.png}.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
if 'ipykernel' not in sys.modules:                          # imported by a notebook: keep its inline backend
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (HERE, ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import paper_figs_ensemble as pfe                                   # noqa: E402
import run_comparison as rc                                         # noqa: E402
from check_moments import log_spaced_taus, logder, pdf_energy_increments   # noqa: E402
from filters_bank import return_Filters                             # noqa: E402
from potentials_builder import get_1d_potentials                    # noqa: E402

RUNS = {'M3 (unconstrained windows)': 'turbulence_nested_M3',
        'A: constrained windows': 'turbulence_nested_M3_bicw',
        'B: constrained + all masses': 'turbulence_nested_M3_bicw_allmass',
        'C: B + 6 windows, q.999, Im parts': 'turbulence_nested_M3_bicw_rich',
        'with Scat4 (mass_final)': 'turbulence_ggd_mass_final'}
CACHE_VERSION = 1      # bump when a statistic in all_stats changes: older cache files are then ignored
CPQ_PAIRS = {'C22': (2, 2), 'C33': (3, 3), 'C42': (4, 2), 'C44': (4, 4), 'C31': (3, 1)}
CUT_TAUS = (1, 4, 20, 64)                                           # tau* of the 1-D cuts of the maps
PERSIST_TAUS = (1, 4, 16)
EXT_TAUS = (1, 4, 16)
EXT_Q = (0.5, 0.9, 0.99, 0.999)
ENERGY_TAUS = (1, 4, 16, 64)
SCAT4 = ('Scattering_Fourth_Order_Mod2_Real_Q1', 'Scattering_Fourth_Order_Mod2_Imag_Q1')
EPS = 1e-12

# key -> (family, kind); kind 'ratio' (positive, compared in log) or 'diff'; 'pdf' = ratio with a floor
KINDS = {
    'marg': ('marginal', 'pdf'), 'psd': ('spectrum', 'ratio'), 'sf': ('structure functions S4,6,8', 'ratio'),
    'flat_slope': ('flatness slope', 'diff'), 'zeta_ratio': ('scaling ratio zeta4/zeta2', 'diff'),
    'inc': ('increment PDFs', 'pdf'), 'skew': ('time asymmetry: skewness', 'diff'),
    'energy_pdf': ('time asymmetry: power-increment PDFs', 'pdf'),
    'C4': ('C_pq at tau*=20 (paper)', 'ratio'), 'C6': ('C_pq at tau*=20 (paper)', 'ratio'),
    'C42': ('C_pq at tau*=20 (paper)', 'ratio'),
    'wav_E': ('wavelet bands', 'ratio'), 'wav_flat': ('wavelet bands', 'ratio'), 'wav_sparsity': ('wavelet bands', 'ratio'),
    'persist': ('burst persistence', 'diff'), 'extremes': ('extremes per signal', 'ratio'),
    'scat4_real': ('Scat4 statistics', 'diff'), 'scat4_imag': ('Scat4 statistics', 'diff'),
    **{f'cpq_{k}': ('C_pq maps (all lag pairs)', 'ratio') for k in CPQ_PAIRS},
}


# ----------------------------------------------------------------- statistics ----
def cpq_maps(x_np, taus):
    """C_pq(tau_i, tau_j) for CPQ_PAIRS on the lags taus: <|du_i|^p |du_j|^q> over the common length
    L = T - max(tau_i, tau_j) (as check_moments.cross_structure_function), divided by S_p(tau_i) S_q(tau_j)
    (full-length moments, as check_moments.C_pq_structure). Returns {name: (n, n)} with rows = tau_i (power p)."""
    T = x_np.shape[-1]
    n = len(taus)

    def powers(t):                                   # |du_t|^p for p = 1..4 by multiplication (much faster than **)
        a1 = np.abs(x_np[..., t:] - x_np[..., :-t])
        a2 = a1 * a1
        return {1: a1, 2: a2, 3: a2 * a1, 4: a2 * a2}

    S = {}
    for t in taus:
        P = powers(t)
        S[t] = {p: P[p].mean() for p in P}
    out = {k: np.zeros((n, n)) for k in CPQ_PAIRS}
    for i, ti in enumerate(taus):
        Pi = powers(ti)
        for j, tj in enumerate(taus):
            Pj = Pi if j == i else powers(tj)
            L = T - max(ti, tj)
            for k, (p, q) in CPQ_PAIRS.items():
                out[k][i, j] = np.mean(Pi[p][..., :L] * Pj[q][..., :L]) / (S[ti][p] * S[tj][q])
    return out


def persistence(x_np, D):
    """corr(log|du_tau(t)|, log|du_tau(t+D)|) over signals and positions, periodic, rows = PERSIST_TAUS."""
    out = np.zeros((len(PERSIST_TAUS), len(D)))
    for r, tau in enumerate(PERSIST_TAUS):
        lg = np.log(np.abs(np.roll(x_np, -tau, axis=-1) - x_np) + EPS)
        lg = lg - lg.mean()
        var = np.mean(lg * lg)
        for c, d in enumerate(D):
            out[r, c] = np.mean(lg * np.roll(lg, -d, axis=-1)) / var
    return out


def scaling_ratio(x_np, taus_f):
    """dlog S4 / dlog S2 on taus_f (check_moments.scaling_exponent_ratio without its cache)."""
    S2, S4 = [], []
    for t in taus_f:
        d2 = (x_np[..., t:] - x_np[..., :-t]) ** 2
        S2.append(d2.mean()); S4.append((d2 * d2).mean())
    return logder(np.array(S4), taus_f) / logder(np.array(S2), taus_f)


def scat4_means(x, pots, batch):
    """Mean over signals of the Scat4 statistics, x (B, 1, T) float32."""
    out = {}
    for name, pot in pots.items():
        acc = None
        with torch.no_grad():
            for b in range(0, x.shape[0], batch):
                s = pot.forward(x[b:b + batch]).double().sum(0)
                acc = s if acc is None else acc + s
        out['scat4_real' if 'Real' in name else 'scat4_imag'] = (acc / x.shape[0]).numpy()
    return out


def all_stats(x, ctx):
    """x: (B, T) float32 tensor -> {key: np.ndarray} for every key of KINDS."""
    x3 = x[:, None, :]
    x_np = x.double().numpy()
    c = pfe.curves(x3, ctx['marg_bins'], ctx['tau_star'])
    s = {k: np.asarray(c[k], float) for k in ('marg', 'psd', 'sf', 'flat_slope', 'inc', 'skew', 'C4', 'C6', 'C42')}
    s['zeta_ratio'] = scaling_ratio(x_np, ctx['taus_f'])
    pdfs = pdf_energy_increments(x3, list(ENERGY_TAUS), mode='power')
    s['energy_pdf'] = np.stack([pdfs[t][1] for t in ENERGY_TAUS])
    w = rc.compute_stats(x, n_wav=10, wav_J=ctx['wav_J'], wav_Q=ctx['wav_Q'])
    s.update(wav_E=w['wav_E'], wav_flat=w['wav_flat'], wav_sparsity=w['wav_sparsity'])
    for k, v in cpq_maps(x_np, ctx['cpq_taus']).items():
        s[f'cpq_{k}'] = v
    s['persist'] = persistence(x_np, ctx['persist_D'])
    s['extremes'] = np.stack([np.quantile(rc.event_scores(x, tau, ref=ctx['x_ref'])[0], EXT_Q) for tau in EXT_TAUS])
    s.update(scat4_means(x3.float(), ctx['scat4'], ctx['batch']))
    return s


def deviation(a, ref, kind):
    """run vs data deviation (log ratio or difference); nan where not defined."""
    a, ref = np.asarray(a, float), np.asarray(ref, float)
    if kind == 'diff':
        return a - ref
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.log(a / ref)
    if kind == 'pdf':
        d[ref < 1e-4 * np.nanmax(ref)] = np.nan                    # floor, as Y_FLOOR in paper_figs_ensemble
    d[~np.isfinite(d)] = np.nan
    return d


# -------------------------------------------------------------------- compute ----
def compute(args):
    root = Path(args.root)
    t0 = time.time()
    runs, missing = rc.load_runs(dict(zip(args.names, args.labels)), root=root, seed=args.seed)
    assert not missing, f'missing runs: {missing}'
    seeds = {}
    for s in args.seeds:
        cfg_name = rc._find_config(root, args.seed_label, s)
        if cfg_name is None:
            print(f'  seed run {args.seed_label} seed {s}: no samples, skipped')
            continue
        seeds[s] = rc.load_config(cfg_name, root, label=args.seed_label, extras=False)['xt']
    cfg0 = next(iter(runs.values()))['cfg']
    x1 = rc.reference_data(cfg0).float()
    for k, r in runs.items():
        assert r['xt'].shape[-1] == x1.shape[-1], f'{k}: length {r["xt"].shape[-1]} vs data {x1.shape[-1]}'
    T = x1.shape[-1]
    M, J = int(cfg0.get('M', T)), int(cfg0['J'])
    filters, filters_phi = return_Filters(M, J, 1, include_phi=True, wav_type=cfg0.get('wav_type', 'morlet'))
    filters_Q = return_Filters(M, J, int(cfg0['Q']), wav_type=cfg0.get('wav_type', 'morlet'))
    scat4 = get_1d_potentials(list(SCAT4), J, filters, int(cfg0['Q']), filters_Q=filters_Q, filters_Phi=filters_phi)
    L = 1.05 * x1.abs().max().item()
    max_tau = T // 2
    ctx = dict(marg_bins=np.linspace(-L, L, pfe.N_BINS), tau_star=args.tau_star,
               taus_f=log_spaced_taus(max_tau, pfe.FLAT_POINTS),
               cpq_taus=np.array(sorted(set(log_spaced_taus(max_tau, args.cpq_points)) | {t for t in CUT_TAUS if t < max_tau})),
               persist_D=log_spaced_taus(max_tau, 20), wav_J=min(6, J), wav_Q=3,
               x_ref=x1, scat4=scat4, batch=args.batch)
    print(f'data {tuple(x1.shape)}; runs {list(runs)}; seed runs {sorted(seeds)}; C_pq lags {ctx["cpq_taus"].tolist()}')

    cache = args.out / 'cache'                       # one npz per data set: a killed run resumes where it stopped
    cache.mkdir(parents=True, exist_ok=True)
    # cache key = the exact saved run (config name with its timestamp) + CACHE_VERSION: a rerun under the same
    # label, or a changed statistic, is recomputed instead of silently reusing old numbers
    runs_cfg = {k: rc._find_config(root, lab, args.seed) for k, lab in zip(args.names, args.labels)}
    seed_cfg = {s: rc._find_config(root, args.seed_label, s) for s in seeds}

    def timed(name, x, key):
        tag = ''.join(ch if ch.isalnum() else '_' for ch in f'{name}__{key}')
        f = cache / f'v{CACHE_VERSION}_{tag}.npz'
        if f.exists():
            print(f'  {name:32s} cached', flush=True)
            return dict(np.load(f))
        t = time.time()
        s = all_stats(x, ctx)
        np.savez(f, **s)
        print(f'  {name:32s} {tuple(x.shape)}  {time.time() - t:6.1f} s', flush=True)
        return s

    data_key = f"{cfg0['subseries_len']}_{cfg0['target_len']}_n{x1.shape[0]}"
    data = timed('data', x1, data_key)
    res = {k: timed(k, r['xt'].float(), runs_cfg[k]) for k, r in runs.items()}
    seed_stats = {s: timed(f'{args.seed_label} seed {s}', x.float(), seed_cfg[s]) for s, x in seeds.items()}
    rng = np.random.default_rng(0)
    half_dev = {k: [] for k in KINDS}
    for rep in range(args.n_rep):
        perm = rng.permutation(len(x1))
        a = timed(f'data halves {rep + 1}/{args.n_rep} (A)', x1[perm[: len(x1) // 2]], data_key)
        b = timed(f'data halves {rep + 1}/{args.n_rep} (B)', x1[perm[len(x1) // 2:]], data_key)
        for k, (_, kind) in KINDS.items():
            half_dev[k].append(deviation(a[k], b[k], 'ratio' if kind == 'pdf' else kind))

    out = {'T': T, 'tau_star': args.tau_star, 'run_names': np.array(list(runs)), 'seed_list': np.array(sorted(seeds)),
           'marg_bins': ctx['marg_bins'], 'taus_f': ctx['taus_f'], 'cpq_taus': ctx['cpq_taus'],
           'persist_D': ctx['persist_D'], 'skew_taus': log_spaced_taus(max_tau, pfe.SKEW_POINTS)}
    for k, (_, kind) in KINDS.items():
        sig_ref = np.nanstd(np.stack(half_dev[k]), axis=0) / 2.0            # sigma for n_ref = all data
        if len(seed_stats) > 1:
            sd = np.stack([deviation(seed_stats[s][k], data[k], kind) for s in seed_stats])
            sig_seed = np.nanstd(sd, axis=0, ddof=1)
        else:
            sig_seed = np.zeros_like(sig_ref)
        out[f'data__{k}'] = data[k]
        out[f'sigma__{k}'] = np.sqrt(sig_ref ** 2 + np.nan_to_num(sig_seed) ** 2)
        out[f'sigref__{k}'], out[f'sigseed__{k}'] = sig_ref, sig_seed
        for i, name in enumerate(runs):
            out[f'run{i}__{k}'] = res[name][k]
        for s in seed_stats:
            out[f'seed{s}__{k}'] = seed_stats[s][k]
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / 'curves.npz', **out)
    print(f'-> {args.out / "curves.npz"}  ({time.time() - t0:.0f} s total)')


# ------------------------------------------------------------------- summarise ----
def summarise(z, out_dir):
    names = list(z['run_names'])
    rows = []
    for fam in dict.fromkeys(f for f, _ in KINDS.values()):
        keys = [k for k, (f, _) in KINDS.items() if f == fam]
        for i, name in enumerate(names):
            zs = []
            for k in keys:
                d = deviation(z[f'run{i}__{k}'], z[f'data__{k}'], KINDS[k][1])
                s = z[f'sigma{i}__{k}'] if f'sigma{i}__{k}' in z else z[f'sigma__{k}']   # per-run sigma (jets)
                with np.errstate(divide='ignore', invalid='ignore'):
                    zz = np.abs(d) / s
                zs.append(zz[np.isfinite(zz)].ravel())
            zs = np.concatenate(zs)
            rows.append((fam, name, len(zs), float(np.mean(zs <= 2)) if len(zs) else np.nan,
                         float(np.median(zs)) if len(zs) else np.nan))
    with open(out_dir / 'summary.csv', 'w', newline='') as f:            # csv quoting: family names contain commas
        w = csv.writer(f)
        w.writerow(['family', 'run', 'n_points', 'frac_within_2sigma', 'median_abs_z'])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f'{r[3]:.3f}', f'{r[4]:.2f}'])
    fams = list(dict.fromkeys(r[0] for r in rows))
    frac = np.array([[next(r[3] for r in rows if r[0] == fa and r[1] == n) for n in names] for fa in fams])
    fig, ax = plt.subplots(figsize=(2.2 + 1.9 * len(names), 0.45 * len(fams) + 1.6))
    im = ax.imshow(frac, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    for (r, c), v in np.ndenumerate(frac):
        ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=9)
    ax.set_xticks(range(len(names)), [n.replace(': ', ':\n').replace(' (', '\n(') for n in names], fontsize=8)
    ax.set_yticks(range(len(fams)), fams, fontsize=9)
    ax.set_title('fraction within 2 sigma of the data\n(1 = recovered everywhere)', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_dir / 'summary.png', dpi=130)
    plt.close(fig)
    print(open(out_dir / 'summary.csv').read())


# ------------------------------------------------------------------------ plot ----
def _band_panel(ax, x, data, sig, runs, kind, cols, logx=False, logy=False):
    if kind == 'diff':
        lo, hi = data - 2 * sig, data + 2 * sig
    else:
        lo, hi = data * np.exp(-2 * sig), data * np.exp(2 * sig)
    ax.fill_between(x, lo, hi, color='0.6', alpha=0.35, lw=0, label='data +- 2 sigma')
    ax.plot(x, data, color='k', lw=2, label='data')
    for name, y in runs.items():
        ax.plot(x, y, color=cols[name], lw=1.3, label=name)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.grid(alpha=0.25, which='both')


def plot(args):
    z = np.load(args.out / 'curves.npz', allow_pickle=True)
    names = list(z['run_names'])
    cols = rc._colors(names)
    T = int(z['T'])
    lags = np.arange(1, T // 2)
    get = lambda k: (z[f'data__{k}'], z[f'sigma__{k}'], {n: z[f'run{i}__{k}'] for i, n in enumerate(names)})

    def fig_of(title, panels, fname, ncol=3):
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False)
        for ax, (sub, fn) in zip(axes.ravel(), panels):
            fn(ax)
            ax.set_title(sub, fontsize=10)
        for ax in axes.ravel()[len(panels):]:
            ax.axis('off')
        axes[0, 0].legend(fontsize=7, frameon=False)
        fig.suptitle(title, y=1.0)
        fig.tight_layout()
        fig.savefig(args.out / fname, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print('  ->', fname)

    # single-scale panels
    mb = z['marg_bins']
    mc = 0.5 * (mb[1:] + mb[:-1])
    f = np.fft.rfftfreq(T)
    dp, sp, rp = get('psd')                          # bound now: later loops reuse the names d, s, r
    p1 = [('marginal of x', lambda ax: _band_panel(ax, mc, *get('marg'), 'pdf', cols, logy=True)),
          ('spectrum', lambda ax, d=dp, s=sp, r=rp: _band_panel(
              ax, f[1:], d[1:], s[1:], {k: v[1:] for k, v in r.items()}, 'ratio', cols, True, True)),
          ('flatness slope dlogF/dlog tau', lambda ax: _band_panel(ax, z['taus_f'], *get('flat_slope'), 'diff', cols, True)),
          ('scaling ratio zeta4/zeta2', lambda ax: _band_panel(ax, z['taus_f'], *get('zeta_ratio'), 'diff', cols, True)),
          ('skewness of power increments', lambda ax: _band_panel(ax, z['skew_taus'], *get('skew'), 'diff', cols, True))]
    for row, p in enumerate((4, 6, 8)):
        d, s, r = get('sf')
        p1.append((f'S_{p}(tau)', lambda ax, row=row, d=d, s=s, r=r: _band_panel(
            ax, lags, d[row], s[row], {k: v[row] for k, v in r.items()}, 'ratio', cols, True, True)))
    fig_of('single-scale statistics (data band = 2 sigma, sigma = '
           f"{getattr(args, 'sigma_note', 'data halves + M3 seed spread')})", p1, 'single_scale.png')

    # PDFs
    pdf_panels = []
    ib = np.linspace(*pfe.INC_XLIM, pfe.N_BINS); ic = 0.5 * (ib[1:] + ib[:-1])
    for row, tau in enumerate(pfe.INC_TAUS):
        d, s, r = get('inc')
        pdf_panels.append((f'increment PDF tau={tau}', lambda ax, row=row, d=d, s=s, r=r: _band_panel(
            ax, ic, d[row], s[row], {k: v[row] for k, v in r.items()}, 'pdf', cols, logy=True)))
    eb = np.linspace(-20, 20, 101); ec = 0.5 * (eb[1:] + eb[:-1])
    for row, tau in enumerate(ENERGY_TAUS):
        d, s, r = get('energy_pdf')
        pdf_panels.append((f'power-increment PDF tau={tau}', lambda ax, row=row, d=d, s=s, r=r: _band_panel(
            ax, ec, d[row], s[row], {k: v[row] for k, v in r.items()}, 'pdf', cols, logy=True)))
    fig_of('PDFs (normalised increments; power increments = time asymmetry)', pdf_panels, 'pdfs.png', ncol=4)

    # cross-scale: paper C_pq, map cuts, persistence, extremes
    cx = [(f'{k} paper (tau*={int(z["tau_star"])})', lambda ax, k=k: _band_panel(ax, lags, *get(k), 'ratio', cols, True))
          for k in ('C4', 'C6', 'C42')]
    ct = z['cpq_taus']
    for k in CPQ_PAIRS:
        for t in CUT_TAUS:
            if t in ct:
                i = int(np.where(ct == t)[0][0])
                d, s, r = get(f'cpq_{k}')
                cx.append((f'{k} cut tau*={t}', lambda ax, i=i, d=d, s=s, r=r: _band_panel(
                    ax, ct, d[i], s[i], {n: v[i] for n, v in r.items()}, 'ratio', cols, True)))
    fig_of('cross-scale C_pq(tau*, tau) = <|du_tau*|^p |du_tau|^q> / (S_p S_q)', cx, 'cross_scale_cuts.png', ncol=4)

    pe = []
    for row, tau in enumerate(PERSIST_TAUS):
        d, s, r = get('persist')
        pe.append((f'burst persistence tau={tau}: corr(log|du(t)|, log|du(t+D)|)', lambda ax, row=row, d=d, s=s, r=r:
                   _band_panel(ax, z['persist_D'], d[row], s[row], {k: v[row] for k, v in r.items()}, 'diff', cols, True)))
    for row, tau in enumerate(EXT_TAUS):
        d, s, r = get('extremes')
        pe.append((f'per-signal max |du_{tau}| / sigma_data: quantiles', lambda ax, row=row, d=d, s=s, r=r: _band_panel(
            ax, np.array(EXT_Q), d[row], s[row], {k: v[row] for k, v in r.items()}, 'ratio', cols)))
    fig_of('bursts in time and extremes per signal', pe, 'persistence_extremes.png')

    # wavelet bands and Scat4 z-scores
    wb = [(k, lambda ax, k=k: _band_panel(ax, np.arange(len(z[f'data__{k}'])), *get(k), 'ratio', cols)) for k in ('wav_E', 'wav_flat', 'wav_sparsity')]
    for k in ('scat4_real', 'scat4_imag'):
        def zplot(ax, k=k):
            for i, n in enumerate(names):
                with np.errstate(divide='ignore', invalid='ignore'):
                    zz = (z[f'run{i}__{k}'] - z[f'data__{k}']) / z[f'sigma__{k}']
                ax.plot(np.arange(len(zz)), zz, 'o', ms=3, color=cols[n], label=n)
            ax.axhline(2, color='k', ls='--', lw=1); ax.axhline(-2, color='k', ls='--', lw=1)
            ax.set_xlabel('statistic index'); ax.set_ylabel('(run - data) / sigma'); ax.grid(alpha=0.25)
        wb.append((f'{k}: z-score per statistic', zplot))
    fig_of('wavelet bands (ratio) and Scat4 statistics (z-scores, +-2 = noise)', wb, 'wavelet_scat4.png')

    # full C_pq maps: log(run/data) per pair and run
    pairs = list(CPQ_PAIRS)
    fig, axes = plt.subplots(len(pairs), len(names), figsize=(3.3 * len(names), 3.0 * len(pairs)), squeeze=False)
    for r_, k in enumerate(pairs):
        d, s, rr = get(f'cpq_{k}')
        for c_, n in enumerate(names):
            lr = np.log(rr[n] / d)
            frac = np.mean(np.abs(lr) <= 2 * s)
            im = axes[r_, c_].imshow(lr, cmap='RdBu_r', vmin=-0.5, vmax=0.5, origin='lower')
            axes[r_, c_].set_title(f'{k}  {n}\nwithin 2 sigma: {frac:.2f}', fontsize=8)
            ticks = range(0, len(ct), max(1, len(ct) // 5))
            axes[r_, c_].set_xticks(ticks, [str(ct[t]) for t in ticks], fontsize=7)
            axes[r_, c_].set_yticks(ticks, [str(ct[t]) for t in ticks], fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.02, label='log(run / data)')
    fig.suptitle('C_pq maps: rows tau* (power p), columns tau (power q)', y=1.0)
    fig.savefig(args.out / 'cross_scale_maps.png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('  -> cross_scale_maps.png')
    summarise(z, args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', nargs='+', default=list(RUNS.values()))
    ap.add_argument('--names', nargs='+', default=list(RUNS))
    ap.add_argument('--seed', type=int, default=900)
    ap.add_argument('--seed_label', default='turbulence_nested_M3')
    ap.add_argument('--seeds', type=int, nargs='+', default=[900, 901, 902, 903, 904])
    ap.add_argument('--n_rep', type=int, default=6, help='random data halves for sigma_ref')
    ap.add_argument('--cpq_points', type=int, default=14, help='log-spaced lags of the C_pq maps (+ the cut lags)')
    ap.add_argument('--tau_star', type=int, default=20)
    ap.add_argument('--batch', type=int, default=200, help='signals per Scat4 forward batch')
    ap.add_argument('--root', default=str(HERE), help='folder holding saved_results/ and experiments/')
    ap.add_argument('--out', type=Path, default=HERE / 'figures' / 'cross_scale_M3')
    ap.add_argument('--plot_only', action='store_true')
    args = ap.parse_args()
    assert len(args.labels) == len(args.names), '--labels and --names must have the same length'
    if not args.plot_only:
        compute(args)
    plot(args)


if __name__ == '__main__':
    main()
