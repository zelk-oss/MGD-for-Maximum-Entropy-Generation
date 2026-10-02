"""
Conditioning and moment mismatch along a run, from the (partial) conditioning logs
<config>_cond.pt written every cond_flush_rows theta steps by SDE._flush_cond_log.

Works on runs that are still going: only the cond log and the run's config.json are
needed (the time grid is rebuilt from config.json; sampling_times is only saved at the end).

What is plotted, per logged corrector (theta) step k, in the Jacobi-scaled live system
Gs theta_s = rhs_s,  Gs = U diag(mu) U^T,  beta = U^T rhs_s  (see SDE._log_cond):

  conditioning : kappa = mu_max / mu_min, mu_min, # eigenvalues below the ridge
  moment gap   : |rhs_s| = |beta|, rhs = phi_bar(I_{k+1}) - phi_bar(y_k): how far the
                 predicted walkers are from the target moments BEFORE the correction,
                 each statistic divided by its own gradient size sqrt(G_nn)
  after ridge  : |lam beta / (mu + lam)|, the part of that gap the ridged solve leaves
                 (linear residual of the moment equation, exact for the solve itself)
  noise        : sqrt(sum nu), sampling noise of the gap (walkers + interpolant samples)

These are NOT the own-scale relative errors of compare_moment_matching.py (those need
<config>_aux_moments.pt, written only at the end of the run).

Usage (from turbulence/):
    python ../notes/gram_along_run_1002/partial_cond_plots.py \
        --glob 'saved_results/aux_moments/*noL6psi_noSm30*_cond.pt' \
        --ref  'saved_results/aux_moments/*terms2f1130a8_zfloor_two_phase_reg1e-4_20260930_1852_cond.pt'
"""
import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'codes'))
from time_schedules import build_time_grid                      # noqa: E402

ARM_COLORS = {'1e-4': '#2a78d6', '1e-2': '#eb6834'}             # categorical slots 1, 2
REF_COLOR = '#5f5e5a'


def time_grid(cfg, root):
    """Time grid of a run, rebuilt from experiments/*/<cfg>/config.json."""
    hits = glob.glob(str(root / 'experiments' / '*' / cfg / 'config.json'))
    if not hits:
        raise FileNotFoundError(f'no experiments/*/{cfg}/config.json (needed for the time grid)')
    a = json.load(open(hits[0]))
    t, _ = build_time_grid(SimpleNamespace(**a), logger=SimpleNamespace(info=lambda *_: None))
    return np.asarray(t, dtype=np.float64)


def load_run(path, root):
    cfg = Path(path).name[:-len('_cond.pt')]
    d = torch.load(path, map_location='cpu', weights_only=False)
    cols = list(d['columns'])
    C = d['theta'].numpy()
    S = d['theta_spectra'].double().numpy()                     # (rows, 3, r): mu, beta, nu
    t = time_grid(cfg, root)
    k = C[:, cols.index('k')].astype(int)
    lam = float(d['regularization'])
    mu, beta, nu = S[:, 0], S[:, 1], S[:, 2]
    lmin, lmax = C[:, cols.index('lam_min')], C[:, cols.index('lam_max')]
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = np.where(lmin > 0, lmax / lmin, np.nan)
        res = lam * beta / (mu + lam)
        chi2 = np.nansum(res ** 2 / nu, axis=1) / C[:, cols.index('n_live')]
    seed = re.search(r'seed_(\d+)', cfg)
    arm = re.search(r'reg(1e-\d)', cfg)
    return dict(
        cfg=cfg, seed=int(seed.group(1)) if seed else -1, arm=arm.group(1) if arm else '?',
        lam=lam, labels=list(d['labels']), cols=cols, C=C,
        k=k, t=t[np.minimum(k + 1, len(t) - 1)], nt=len(t) - 1,     # theta at step k targets t[k+1]
        kappa=kappa, mu_min=lmin,
        n_below_1e4=np.sum(mu < 1e-4, axis=1), n_below_1e6=np.sum(mu < 1e-6, axis=1),
        gap=np.sqrt(np.nansum(beta ** 2, axis=1)),
        res=np.sqrt(np.nansum(res ** 2, axis=1)),
        noise=np.sqrt(np.nansum(nu, axis=1)),
        chi2=chi2,
    )


def roll_med(y, w=25):
    """Centred running median over w logged steps (NaN-aware); keeps spikes out of the eye."""
    if len(y) < w:
        return y
    from numpy.lib.stride_tricks import sliding_window_view
    h = w // 2
    yp = np.pad(y.astype(float), (h, h), constant_values=np.nan)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanmedian(sliding_window_view(yp, w), axis=-1)


def setup_x(ax, xaxis):
    if xaxis == 'log1mt':
        ax.set_xscale('log'); ax.invert_xaxis(); ax.set_xlabel('1 - t   (t = 1 on the right)')
        ax.axvline(0.1, color='0.75', lw=0.8, ls=':')               # t_switch of the two-phase grid
    else:
        ax.set_xlabel('t')
    ax.grid(True, which='major', color='0.92', lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def xval(run, xaxis):
    return np.maximum(1 - run['t'], 1e-7) if xaxis == 'log1mt' else run['t']


def plot_series(axs, runs, refs, keys, xaxis, smooth):
    for ax, (key, ylab, ylog) in zip(axs, keys):
        for run in refs:
            y = roll_med(run[key]) if smooth else run[key]
            ax.plot(xval(run, xaxis), y, color=REF_COLOR, lw=1.6, alpha=0.9, zorder=3)
        for run in runs:
            y = roll_med(run[key]) if smooth else run[key]
            ax.plot(xval(run, xaxis), y, color=ARM_COLORS.get(run['arm'], 'k'), lw=0.9, alpha=0.65)
        if ylog:
            ax.set_yscale('log')
        ax.set_ylabel(ylab)
        setup_x(ax, xaxis)


def legend_handles(runs, refs):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=ARM_COLORS[a], lw=2,
                label=f'noL6psi+noSm30, ridge {a} ({sum(r["arm"] == a for r in runs)} seeds)')
         for a in sorted({r['arm'] for r in runs}, reverse=True) if a in ARM_COLORS]
    if refs:
        h.append(Line2D([], [], color=REF_COLOR, lw=2, label='full set, ridge 1e-4 (382722, seed 900)'))
    return h


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--glob', default='saved_results/aux_moments/*noL6psi_noSm30*_cond.pt')
    p.add_argument('--ref', default=None, help='glob of reference cond logs (drawn in grey)')
    p.add_argument('--root', default='.', help='turbulence/ (holds experiments/)')
    p.add_argument('--out', default=str(REPO / 'notes' / 'gram_along_run_1002' / 'figures'))
    p.add_argument('--xaxis', choices=['log1mt', 'linear'], default='log1mt')
    p.add_argument('--raw', action='store_true', help='no running median (default: median over 25 logged steps)')
    a = p.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = [load_run(f, root) for f in sorted(glob.glob(a.glob))]
    refs = [load_run(f, root) for f in sorted(glob.glob(a.ref))] if a.ref else []
    runs = [r for r in runs if r['cfg'] not in {q['cfg'] for q in refs}]
    if not runs:
        sys.exit(f'no cond logs match {a.glob}')
    runs.sort(key=lambda r: (r['arm'], r['seed']))
    smooth = not a.raw
    tag = '' if smooth else '_raw'

    # ---- text summary -------------------------------------------------------------
    bands = [(0, 0.9), (0.9, 0.99), (0.99, 0.999), (0.999, 0.9999), (0.9999, 1.0)]
    print(f'{"run":>22} {"logged to":>16} | ' + ' | '.join(f'[{lo},{hi})'.center(31) for lo, hi in bands))
    print(f'{"":>22} {"":>16} | ' + ' | '.join(f'{"med kappa":>10} {"med gap":>9} {"res/noise":>9}' for _ in bands))
    for run in refs + runs:
        name = ('REF ' if run in refs else '') + f'reg{run["arm"]} s{run["seed"]}'
        last = f'k={run["k"][-1]} t={run["t"][-1]:.6f}'
        row = f'{name:>22} {last:>16} |'
        for lo, hi in bands:
            s = (run['t'] >= lo) & (run['t'] < hi)
            if s.any():
                row += (f' {np.nanmedian(run["kappa"][s]):10.2e} {np.nanmedian(run["gap"][s]):9.2e}'
                        f' {np.nanmedian(run["res"][s] / run["noise"][s]):9.2e} |')
            else:
                row += f' {"-":>10} {"-":>9} {"-":>9} |'
        print(row)
    print('\nkappa undefined (mu_min <= 0, below float32 resolution of G): ' +
          ', '.join(f'reg{r["arm"]} s{r["seed"]}: {np.isnan(r["kappa"]).sum()}/{len(r["kappa"])}'
                    for r in refs + runs))

    # most frequent statistics in the weakest direction: near-copies come back as a pair
    i0 = runs[0]['cols'].index('top0')
    for run in refs + runs:
        trip = [tuple(sorted(int(v) for v in row[i0:i0 + 2] if v >= 0)) for row in run['C']]
        top = Counter(trip).most_common(3)
        lab = run['labels']
        txt = '; '.join(f'{100 * c / len(trip):4.1f}% {"+".join(lab[i] for i in pr)}' for pr, c in top)
        name = ('REF ' if run in refs else '') + f'reg{run["arm"]} s{run["seed"]}'
        print(f'{name:>14}  weakest dir top-2: {txt}')

    # ---- figure 1: conditioning ---------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    plot_series(axs, runs, refs, [
        ('kappa', 'κ of scaled G (before ridge)', True),
        ('mu_min', 'smallest eigenvalue μ_min', True),
        ('n_below_1e4', '# eigenvalues below 1e-4', False)], a.xaxis, smooth)
    for lam, c in [(1e-4, ARM_COLORS['1e-4']), (1e-2, ARM_COLORS['1e-2'])]:
        axs[1].axhline(lam, color=c, ls='--', lw=1)
        axs[1].annotate(f'ridge {lam:g}', (1, lam), xycoords=('axes fraction', 'data'),
                        ha='right', va='bottom', fontsize=8, color='0.3')
    for ax in axs[:-1]:
        ax.set_xlabel('')
    axs[0].legend(handles=legend_handles(runs, refs), loc='upper left', fontsize=8, frameon=False)
    axs[0].set_title('Corrector (θ) Gram along the run' + ('' if smooth else ' (raw)')
                     + ', running median over 25 logged steps' * smooth, fontsize=11, loc='left')
    fig.savefig(out / f'1_conditioning_{a.xaxis}{tag}.png', dpi=130)

    # ---- figure 2: moment gap -----------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    for run in runs + refs:
        run['res_over_noise'] = run['res'] / run['noise']
    plot_series(axs, runs, refs, [
        ('gap', '|target − walker moments| before correction\n(scaled by √G_nn)', True),
        ('res', 'left after the ridged correction\n|λβ/(μ+λ)|', True),
        ('res_over_noise', 'left after correction / sampling noise', True)], a.xaxis, smooth)
    axs[2].axhline(1, color='0.4', ls='--', lw=1)
    for ax in axs[:-1]:
        ax.set_xlabel('')
    axs[0].legend(handles=legend_handles(runs, refs), loc='upper left', fontsize=8, frameon=False)
    axs[0].set_title('Moment mismatch of the corrector solve (from the cond log)', fontsize=11, loc='left')
    fig.savefig(out / f'2_moment_gap_{a.xaxis}{tag}.png', dpi=130)
    print(f'\nfigures in {out}')


if __name__ == '__main__':
    main()
