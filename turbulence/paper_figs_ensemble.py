#!/usr/bin/env python3
"""Paper panels (codes/paper_figures.py) as data vs seed-ensemble mean +- std.

Same 9 panels and style as make_all_panels (run_analyse_*.ipynb, last cells), but the
synthetic curve of each panel is computed for every seed separately, on a grid shared
with the data, and drawn as mean (solid) with a +-1 std band over seeds. Data stays dotted.

Two steps, so the plots can be restyled without reloading 30 sample files:
  1. compute: per seed, load saved_results/samples/<config>.pt, compute every panel's curve,
     store all curves in <out_dir>/curves.npz (data curve + (n_seeds, ...) synth curves).
  2. plot: read curves.npz, save panel_*.pdf in <out_dir>.
`--plot_only` skips step 1.

Seeds = every run folder under experiments/<GROUP>/ whose samples file exists. If a seed has
two runs with samples (seed 902: _1542 and _1904), the newest is used and both are printed.

Usage (repo root, on Jean Zay -- samples are not pulled locally):
  python turbulence/paper_figs_ensemble.py                       # all seeds with samples
  python turbulence/paper_figs_ensemble.py --exclude 922
  python turbulence/paper_figs_ensemble.py --plot_only --fmt png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
if 'ipykernel' not in sys.modules:                          # imported by a notebook: keep its inline backend
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'data'):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from codes.utils import split_periodize_reshape, normalize, _load_tensor          # noqa: E402
from codes.ortho_wavelet.ReadyToUseWavelets import DefineWavelet                  # noqa: E402
from data.data_loader import load_turbulence_1d                                   # noqa: E402
from check_moments import logder, skewness_vs_tau, log_spaced_taus             # noqa: E402
from paper_figures import STYLE, COLOR_1, LS_DATA, LS_SYNTH, _apply_style, _finalize, _psd  # noqa: E402

GROUP = ('turbulencesynth_M256_J8_Q3_sigma3.5_nt55000_n1_10000_lam0.0_thomas_nsub1_moment_f64'
         '_dedupfilt1e-08_terms54d17f3b_turbulence_final')

# panel settings, as the defaults of codes/paper_figures.py
SF_ORDERS = (4, 6, 8)
FLAT_POINTS = 30
INC_TAUS = (1, 4, 16, 64)
INC_XLIM = (-6, 6)
N_BINS = 201
SMOOTH_SIGMA = 1.5
CPQ = {'C4': (2, 2), 'C6': (3, 3), 'C42': (4, 2)}
SKEW_POINTS = 45
OMEGA_MAX = 0.4
Y_FLOOR = 1e-4
BAND_ALPHA = 0.3


# ------------------------------------------------------------------ compute ----
def increment_stats(x_np, max_tau, tau_star, eps=1e-8):
    """S_p(tau) for p = 2,4,6,8 and C_pq(tau, tau*) for CPQ, lags 1 .. max_tau-1, in one pass.

    Same quantities as check_moments.second_order_structure_function (|mean du^p|) and
    check_moments.C_pq_structure, but each lag's increments are formed once and every power is
    taken from them; the tau* increments and moments are computed once instead of once per lag.
    """
    T = x_np.shape[-1]
    taus = np.arange(1, max_tau)
    sf = np.zeros((4, len(taus)))                                    # p = 2, 4, 6, 8
    cpq = {name: np.zeros(len(taus)) for name in CPQ}
    a_star = np.abs(x_np[..., tau_star:] - x_np[..., :-tau_star])    # |du_tau*|, full length
    S_star = {p: np.mean(a_star ** p) for p in {p for p, _ in CPQ.values()}}
    for i, t in enumerate(taus):
        du = x_np[..., t:] - x_np[..., :-t]
        d2 = du * du
        d4 = d2 * d2
        sf[:, i] = np.abs([d2.mean(), d4.mean(), (d4 * d2).mean(), (d4 * d4).mean()])
        a = np.abs(du)
        L = min(T - tau_star, T - t)                                 # as C_pq_structure
        a_L, s_L = a[..., :L], a_star[..., :L]
        for name, (p, q) in CPQ.items():
            S_q = sf[0, i] if q == 2 else np.mean(a ** q)
            cpq[name][i] = np.mean(s_L ** p * a_L ** q) / (S_star[p] * S_q + eps)
    return taus, sf, cpq


def curves(x, marg_bins, tau_star):
    """Every panel's curve for one set of signals x, (B, 1, T) float tensor on CPU."""
    x_np = x.numpy()
    T = x.shape[-1]
    max_tau = T // 2
    out = {}

    h, _ = np.histogram(x_np.reshape(-1), bins=marg_bins, density=True)
    out['marg'] = gaussian_filter1d(h, SMOOTH_SIGMA)
    out['marg_out'] = np.mean((x_np < marg_bins[0]) | (x_np > marg_bins[-1]))   # mass outside the bins

    out['psd'] = _psd(x_np)

    taus, sf, cpq = increment_stats(x_np, max_tau, tau_star)
    out['sf'] = sf[[{2: 0, 4: 1, 6: 2, 8: 3}[p] for p in SF_ORDERS]]
    taus_f = log_spaced_taus(max_tau, FLAT_POINTS)                  # subset of taus = 1 .. max_tau-1
    F = sf[1, taus_f - 1] / sf[0, taus_f - 1] ** 2                  # Flat_4 = S_4 / S_2^2, as check_moments.flatness
    out['flat_slope'] = logder(F, taus_f)
    out.update(cpq)

    inc_bins = np.linspace(*INC_XLIM, N_BINS)
    inc = []
    for tau in INC_TAUS:
        d = (x_np[..., tau:] - x_np[..., :-tau]).reshape(-1)
        h, _ = np.histogram(d / (d.std() + 1e-12), bins=inc_bins, density=True)
        h = gaussian_filter1d(h, SMOOTH_SIGMA)
        inc.append(h / (h.max() + 1e-12))       # peak 1; the 10^-k shift is applied at plot time
    out['inc'] = np.stack(inc)

    _, out['skew'] = skewness_vs_tau(x, log_spaced_taus(max_tau, num_points=SKEW_POINTS), mode='power')
    return out


def find_runs(root, group, exclude):
    """seed -> config of the run whose samples exist (newest if several)."""
    samples_dir = root / 'saved_results' / 'samples'
    by_seed = {}
    for d in sorted((root / 'experiments' / group).iterdir()):
        if not d.is_dir() or '_seed_' not in d.name:
            continue
        if not ((samples_dir / f'{d.name}.pt').exists() or (samples_dir / d.name).exists()):
            print(f'  no samples for {d.name}, skipped')
            continue
        seed = int(json.loads((d / 'config.json').read_text())['seed'])
        by_seed.setdefault(seed, []).append(d.name)
    runs = {}
    for seed, configs in sorted(by_seed.items()):
        if seed in exclude:
            continue
        if len(configs) > 1:
            print(f'  seed {seed}: {len(configs)} runs with samples {configs} -> newest')
        runs[seed] = sorted(configs)[-1]        # names end in _<YYYYMMDD_HHMM>
    return runs


def compute(args):
    root = ROOT / 'turbulence'
    runs = find_runs(root, args.group, set(args.exclude))
    assert runs, f'no runs with samples under {root / "experiments" / args.group}'
    print(f'{len(runs)} seeds: {sorted(runs)}')

    cfg = json.loads((root / 'experiments' / args.group / runs[min(runs)] / 'config.json').read_text())
    W = DefineWavelet('Db', m=3, device='cpu')
    Data = split_periodize_reshape(load_turbulence_1d(), cfg['subseries_len'])
    for _ in range(int(np.log2(cfg['subseries_len'] / cfg['target_len']))):   # as run_SDE.py
        Data = W.decompose(Data)[1]
    x1 = normalize(Data[:cfg['n1']]).cpu()
    print('x1:', tuple(x1.shape))

    L = args.marg_range * x1.abs().max().item()
    marg_bins = np.linspace(-L, L, args.marg_bins)
    data = curves(x1, marg_bins, args.tau_star)

    synth = {}
    for i, (seed, config) in enumerate(sorted(runs.items())):
        xt = _load_tensor(root / 'saved_results' / 'samples' / config).detach().float().cpu()
        assert xt.shape[1:] == x1.shape[1:], f'seed {seed}: samples {tuple(xt.shape)} vs data {tuple(x1.shape)}'
        c = curves(xt, marg_bins, args.tau_star)
        for k, v in c.items():
            synth.setdefault(k, []).append(v)
        print(f'  [{i + 1}/{len(runs)}] seed {seed}: {tuple(xt.shape)}, '
              f'mass outside marginal bins {c["marg_out"]:.1e}')
        del xt

    T = x1.shape[-1]
    np.savez(args.out_dir / 'curves.npz',
             seeds=np.array(sorted(runs)), configs=np.array([runs[s] for s in sorted(runs)]),
             tau_star=args.tau_star, T=T, marg_bins=marg_bins,
             **{f'data_{k}': v for k, v in data.items()},
             **{f'synth_{k}': np.stack(v) for k, v in synth.items()})
    print(f'-> {args.out_dir / "curves.npz"}')


# --------------------------------------------------------------------- plot ----
def band(ax, x, Y, color=COLOR_1, lw=None, alpha=1.0, label=None, floor=None):
    """Mean over seeds (axis 0) as a solid line, +-1 std as a band. floor: clip for log axes."""
    m, s = Y.mean(0), Y.std(0, ddof=1)
    lo = m - s if floor is None else np.maximum(m - s, floor)
    ax.fill_between(x, lo, m + s, color=color, alpha=BAND_ALPHA * alpha, lw=0)
    ax.plot(x, m, ls=LS_SYNTH, color=color, lw=lw or STYLE['lw'], alpha=alpha, label=label)


def line(ax, x, y, color=COLOR_1, lw=None, alpha=1.0, label=None):
    ax.plot(x, y, ls=LS_DATA, color=color, lw=lw or STYLE['lw'], alpha=alpha, label=label)


def plot(args):
    z = np.load(args.out_dir / 'curves.npz')
    D = {k[5:]: z[k] for k in z.files if k.startswith('data_')}
    S = {k[6:]: z[k] for k in z.files if k.startswith('synth_')}
    T, tau_star, n_seeds = int(z['T']), int(z['tau_star']), len(z['seeds'])
    max_tau = T // 2
    kw = dict(save_dir=str(args.out_dir), fmt=args.fmt, dpi=args.dpi)
    print(f'{n_seeds} seeds: {z["seeds"].tolist()}')

    # (a) marginal
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    bins = z['marg_bins']
    ctr = 0.5 * (bins[1:] + bins[:-1])
    line(ax, ctr, np.where(D['marg'] > 0, D['marg'], np.nan))
    band(ax, ctr, S['marg'], floor=Y_FLOOR)
    above = ctr[(D['marg'] > Y_FLOOR) | (S['marg'].mean(0) > Y_FLOOR)]
    pad = 0.05 * (above.max() - above.min())
    ax.set_xlim(above.min() - pad, above.max() + pad)
    ax.set_yscale('log')
    ax.set_ylim(Y_FLOOR, ax.get_ylim()[1])
    _apply_style(ax, r'$x$', 'PDF')
    _finalize(fig, name='panel_a_marginals', **kw)

    # (b) spectrum
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    omega = np.fft.rfftfreq(T)
    mask = (omega > 0) & (omega <= OMEGA_MAX)
    line(ax, omega[mask], D['psd'][mask])
    band(ax, omega[mask], S['psd'][:, mask])
    ax.set_xscale('log'); ax.set_yscale('log')
    _apply_style(ax, r'$\omega$', 'PSD')
    _finalize(fig, name='panel_b_spectrum', **kw)

    # (c) structure functions
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    taus = np.arange(1, max_tau)
    for k, a in enumerate(np.linspace(1.0, 0.4, len(SF_ORDERS))):
        line(ax, taus, D['sf'][k], alpha=a)
        band(ax, taus, S['sf'][:, k], alpha=a)
    ax.set_xscale('log'); ax.set_yscale('log')
    _apply_style(ax, r'$\tau$', r'$S_p(\tau)$, $p=4,6,8$')
    _finalize(fig, name='panel_c_structure_functions', **kw)

    # (d) flatness slope
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    taus_f = log_spaced_taus(max_tau, FLAT_POINTS)
    line(ax, taus_f, D['flat_slope'])
    band(ax, taus_f, S['flat_slope'])
    ax.set_xscale('log')
    _apply_style(ax, r'$\tau$', r'$d\log \mathrm{Flat}_4/d\log\tau$')
    _finalize(fig, name='panel_d_flatness_slope', **kw)

    # (e) increment PDFs, shifted by 10^-k
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    inc_bins = np.linspace(*INC_XLIM, N_BINS)
    ctr = 0.5 * (inc_bins[1:] + inc_bins[:-1])
    for k, tau in enumerate(INC_TAUS):
        sh = 10.0 ** (-k)
        line(ax, ctr, np.where(D['inc'][k] > 0, D['inc'][k] * sh, np.nan), lw=STYLE['lw'] * 0.7)
        band(ax, ctr, S['inc'][:, k] * sh, lw=STYLE['lw'] * 0.7, floor=1e-12)
        ax.text(0.0, sh * 1.5, rf'$\tau={tau}$', ha='center', va='bottom', fontsize=STYLE['fontsize'] * 0.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.2))
    ax.set_yscale('log')
    ax.set_xlim(INC_XLIM)
    ax.set_ylim(10.0 ** (-(len(INC_TAUS) + 1)), 5.0)
    _apply_style(ax, r'$\delta u/\sigma_{\delta u}$', 'PDF (shifted)')
    _finalize(fig, name='panel_e_increment_pdfs', **kw)

    # (f, g, h) C_pq
    taus = np.arange(1, max_tau)
    for name, ylabel, panel, inset in [('C4', r'$C_4(\tau,\tau^\star)$', 'panel_f_C4', False),
                                       ('C6', r'$C_6(\tau,\tau^\star)$', 'panel_g_C6', False),
                                       ('C42', r'$C_{4,2}(\tau,\tau^\star)$', 'panel_h_C42', True)]:
        fig, ax = plt.subplots(figsize=STYLE['figsize'])
        line(ax, taus, D[name])
        band(ax, taus, S[name])
        ax.axvline(tau_star, color='grey', ls=':', lw=2)
        if inset:
            axins = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
            line(axins, taus, D[name], lw=STYLE['lw'] * 0.7)
            band(axins, taus, S[name], lw=STYLE['lw'] * 0.7)
            axins.set_xlim(max(1, tau_star // 2), tau_star * 2)
            axins.axvline(tau_star, color='grey', ls=':', lw=1.5)
            axins.grid(True, alpha=STYLE['grid_alpha'])
            axins.tick_params(labelsize=STYLE['fontsize'] * 0.5)
        ax.set_xscale('log')
        _apply_style(ax, r'$\tau$', ylabel)
        _finalize(fig, name=panel, **kw)

    # (i) skewness
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    taus_s = log_spaced_taus(max_tau, num_points=SKEW_POINTS)
    line(ax, taus_s, D['skew'])
    band(ax, taus_s, S['skew'])
    ax.axhline(0, color='black', lw=1)
    ax.set_xscale('log')
    _apply_style(ax, r'$\tau$', 'Skewness')
    _finalize(fig, name='panel_i_skewness', **kw)
    print(f'panels -> {args.out_dir}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--group', default=GROUP, help='experiments/<group>/ holding one folder per seed')
    ap.add_argument('--exclude', type=int, nargs='*', default=[], help='seeds to leave out')
    ap.add_argument('--tau_star', type=int, default=20)
    ap.add_argument('--marg_range', type=float, default=1.5,
                    help='marginal bins span +-marg_range * max|data|; mass outside is printed per seed')
    ap.add_argument('--marg_bins', type=int, default=301)
    ap.add_argument('--out_dir', type=Path, default=ROOT / 'turbulence' / 'paper_figs' / 'turbulence_final_ensemble')
    ap.add_argument('--fmt', default='pdf')
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('--plot_only', action='store_true', help='skip compute, replot from curves.npz')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        compute(args)
    plot(args)


if __name__ == '__main__':
    main()
