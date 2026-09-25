"""Side-by-side comparison of finished turbulence runs (used by compare_runs.ipynb).

Loads each run by its --label (samples, sampling times, aux moments, config.json,
run.log), rebuilds the reference data x1 exactly as run_SDE.py does, and draws one
compact figure per statistic with every run overlaid (data always in black):

    time series | marginal PDF | power spectrum | structure functions / flatness /
    skewness | increment PDFs | wavelet-coefficient histograms (+ KS heatmap) |
    moment matching | summary table

Statistics are computed once per array (compute_stats) and reused by every plot.
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent                     # turbulence/
PROJECT = ROOT.parent
for p in (PROJECT, PROJECT / 'codes', PROJECT / 'data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils import normalize, split_periodize_reshape      # noqa: E402
from filters.filters_1d import init_band_pass              # noqa: E402
from ortho_wavelet.ReadyToUseWavelets import DefineWavelet  # noqa: E402

# Categorical slots 1-8 of the dataviz reference palette, in fixed order: a run
# keeps its colour for the whole notebook (colour follows the run, not its rank).
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
           '#4a3aa7', '#e34948']
DATA_COLOR = 'black'
LOCAL_DATA = PROJECT / 'data' / 'data_files' / 'turbulence_1d_period.pt'
RUN_NAME_RE = re.compile(r'_seed_(\d+)_terms([0-9a-f]{8})_(.+)_(\d{8}_\d{4})$')


# ================================================================================
# Loading
# ================================================================================

def _find_config(root, label, seed=None):
    """Config name (no .pt) of the saved run with this --label (latest timestamp)."""
    hits = []
    for f in (root / 'saved_results' / 'samples').glob(f'*_{label}_*'):
        name = f.stem if f.suffix == '.pt' else f.name
        m = RUN_NAME_RE.search(name)
        if m and m[3] == label and (seed is None or int(m[1]) == seed):
            hits.append((m[4], name))
    return max(hits)[1] if hits else None


def _find_exp_dir(root, config):
    hits = list((root / 'experiments').glob(f'*/{config}')) + \
        list((root / 'experiments').glob(config))
    return hits[0] if hits else None


def _load_pt(path):
    if not path.exists():
        path = path.with_suffix('')                          # pre-.pt runs
    return torch.load(path, map_location='cpu', weights_only=False)


def _runtime_h(exp_dir):
    log = exp_dir / 'logs' / 'run.log' if exp_dir else None
    if not log or not log.exists():
        return np.nan
    m = re.findall(r'SDE integration finished in ([\d.]+) s', log.read_text())
    return float(m[-1]) / 3600 if m else np.nan


def load_runs(runs, root=ROOT, seed=None):
    """runs: {display name: --label}. Returns ({name: run dict}, [missing labels])."""
    out, missing = {}, []
    for name, label in runs.items():
        config = _find_config(root, label, seed)
        if config is None:
            missing.append(label)
            continue
        sr = root / 'saved_results'
        xt = _load_pt(sr / 'samples' / f'{config}.pt').float()
        run = dict(label=label, config=config, xt=xt.reshape(xt.shape[0], -1),
                   t=_load_pt(sr / 'sampling_times' / f'{config}.pt').double())
        for key, sub in (('theta', 'lagrange_multipliers'), ('dH', 'entropy_bounds')):
            f = sr / sub / f'{config}.pt'
            if f.exists() or f.with_suffix('').exists():
                v = _load_pt(f).double()
                run[key] = v.reshape(v.shape[0], -1)
        aux = sr / 'aux_moments' / f'{config}_aux_moments.pt'
        if aux.exists():
            a = torch.load(aux, map_location='cpu', weights_only=False)
            run['barphi_e'], run['barphi_p'] = a['barphi_e'].double(), a['barphi_p'].double()
        exp_dir = _find_exp_dir(root, config)
        run['cfg'] = json.loads((exp_dir / 'config.json').read_text()) if exp_dir else {}
        run['runtime_h'] = _runtime_h(exp_dir)
        out[name] = run
    return out, missing


def pull_commands(labels, subdir='turbulence'):
    """Copy-pasteable rsync for the saved_results of the given labels (run it yourself)."""
    jz = f'jz:/lustre/fswork/projects/rech/wbg/ukv59en/MGD-for-Maximum-Entropy-Generation/{subdir}'
    inc = ' '.join(f"--include='*_{l}_*'" for l in labels)
    return (f"cd ~/phd/MGD-for-Maximum-Entropy-Generation\n"
            f"rsync -av --prune-empty-dirs --include='*/' {inc} --exclude='*' \\\n"
            f"  {jz}/saved_results/{{samples,sampling_times,aux_moments,lagrange_multipliers,entropy_bounds}} "
            f"{subdir}/saved_results/")


def reference_data(cfg, device='cpu'):
    """x1 exactly as run_SDE.py builds it, from a run's config.json."""
    if 'TURBULENCE_1D_DATA_PATH' not in os.environ and LOCAL_DATA.exists():
        os.environ['TURBULENCE_1D_DATA_PATH'] = str(LOCAL_DATA)
    from data_loader import load_turbulence_1d
    W = DefineWavelet('Db', m=3, device=device)
    data = split_periodize_reshape(load_turbulence_1d().to(device), cfg['subseries_len'])
    for _ in range(int(np.log2(data.shape[-1] / cfg['target_len']))):
        data = W.decompose(data)[1]
    x1 = normalize(data[:cfg['n1']])
    return x1.reshape(x1.shape[0], -1).cpu()


# ================================================================================
# Statistics (computed once per array)
# ================================================================================

def compute_stats(x, taus=None, pdf_taus=(1, 4, 16, 64), wav_J=6, wav_Q=3,
                  n_wav=2000, seed=0):
    """All statistics used by the plots, for x of shape (B, T), periodic in T."""
    x = x.double().numpy() if torch.is_tensor(x) else np.asarray(x, float)
    B, T = x.shape
    taus = np.unique(np.round(np.logspace(0, np.log10(T // 2), 30)).astype(int)) \
        if taus is None else np.asarray(taus)
    s = dict(T=T, taus=taus, values=x.ravel())

    ps = np.abs(np.fft.rfft(x - x.mean(1, keepdims=True), axis=1)) ** 2
    s['freq'], s['spectrum'] = np.fft.rfftfreq(T)[1:], ps.mean(0)[1:]

    S = {p: [] for p in (2, 3, 4, 6)}
    for tau in taus:
        d = np.roll(x, -tau, axis=1) - x                      # periodic increments
        for p in S:
            S[p].append(np.mean(d ** p))                      # signed for p=3
    S = {p: np.array(v) for p, v in S.items()}
    s.update(S2=S[2], skew=S[3] / S[2] ** 1.5, flat4=S[4] / S[2] ** 2, flat6=S[6] / S[2] ** 3)

    s['pdf_taus'] = pdf_taus
    s['inc'] = {tau: ((np.roll(x, -tau, axis=1) - x) /
                      np.sqrt(np.mean((np.roll(x, -tau, axis=1) - x) ** 2))).ravel()
                for tau in pdf_taus}

    rng = np.random.default_rng(seed)
    sub = x[rng.choice(B, min(n_wav, B), replace=False)]
    psi = init_band_pass('morlet', T, J=wav_J, Q=wav_Q, high_freq=0.49, wav_norm='l1')
    wt = np.fft.ifft(np.fft.fft(sub)[:, None, :] * np.asarray(psi)[None], axis=-1)
    s['wav'] = np.abs(wt).transpose(1, 0, 2).reshape(wt.shape[1], -1)   # (bands, samples)
    s['wav_JQ'] = (wav_J, wav_Q)
    return s


def _colors(names):
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


def _hist_line(ax, v, bins, **kw):
    h, e = np.histogram(v, bins=bins, density=True)
    c = 0.5 * (e[1:] + e[:-1])
    ax.plot(c[h > 0], h[h > 0], **kw)


# ================================================================================
# Plots
# ================================================================================

def plot_time_series(x_ref, runs, n=3, zoom=64, seed=0):
    """One row per source (data first), n random samples, full length + zoom, shared y."""
    rng = np.random.default_rng(seed)
    rows = [('data', x_ref, DATA_COLOR)] + [(k, r['xt'], c) for (k, r), c in
                                            zip(runs.items(), _colors(runs).values())]
    lim = 1.05 * max(float(np.percentile(np.abs(x.numpy()), 99.9)) for _, x, _ in rows)
    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 1.6 * len(rows)), sharey=True,
                             gridspec_kw=dict(width_ratios=[3, 1]))
    for (name, x, c), (a_full, a_zoom) in zip(rows, axes):
        idx = rng.choice(x.shape[0], n, replace=False)
        for k, i in enumerate(idx):
            a_full.plot(x[i].numpy(), color=c, lw=0.8, alpha=1 - 0.25 * k)
            a_zoom.plot(x[i, :zoom].numpy(), color=c, lw=1.2, alpha=1 - 0.25 * k)
        a_full.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=9)
        a_full.set_ylim(-lim, lim)
        for a in (a_full, a_zoom):
            a.grid(alpha=0.2)
    axes[0, 0].set_title(f'{n} random samples (same y-scale everywhere)')
    axes[0, 1].set_title(f'zoom: first {zoom} points')
    fig.tight_layout()
    plt.show()


def plot_marginals(S_ref, S_runs):
    """Small multiples: one panel per run, data in black, log density."""
    cols = _colors(S_runs)
    lo = min(np.percentile(S['values'], 0.001) for S in [S_ref, *S_runs.values()])
    hi = max(np.percentile(S['values'], 99.999) for S in [S_ref, *S_runs.values()])
    bins = np.linspace(1.1 * lo, 1.1 * hi, 161)
    n = len(S_runs)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3), sharey=True, squeeze=False)
    for ax, (k, S) in zip(axes[0], S_runs.items()):
        _hist_line(ax, S_ref['values'], bins, color=DATA_COLOR, lw=2, label='data')
        _hist_line(ax, S['values'], bins, color=cols[k], lw=1.5, label=k)
        ax.set_yscale('log'); ax.set_title(k, fontsize=9); ax.grid(alpha=0.2)
        ax.set_xlabel('x')
    axes[0, 0].set_ylabel('density'); axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle('Marginal PDF of x (log scale shows the tails)')
    fig.tight_layout()
    plt.show()


def plot_spectra(S_ref, S_runs):
    cols = _colors(S_runs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    a1.loglog(S_ref['freq'], S_ref['spectrum'], color=DATA_COLOR, lw=2, label='data')
    for k, S in S_runs.items():
        a1.loglog(S['freq'], S['spectrum'], color=cols[k], lw=1.5, label=k)
        a2.loglog(S['freq'], S['spectrum'] / S_ref['spectrum'], color=cols[k], lw=1.5, label=k)
    a2.axhline(1, color=DATA_COLOR, lw=1, ls='--')
    a1.set_title('Power spectrum'); a2.set_title('Spectrum ratio  gen / data')
    for a in (a1, a2):
        a.set_xlabel('frequency'); a.grid(alpha=0.2, which='both')
    a1.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    plt.show()


def plot_structure(S_ref, S_runs):
    """S2 ratio, flatness S4/S2^2, S6/S2^3 and skewness S3/S2^1.5 vs tau."""
    cols = _colors(S_runs)
    panels = [('S2', r'$S_2(\tau)$ gen / data', True),
              ('flat4', r'flatness $S_4/S_2^2$', False),
              ('flat6', r'$S_6/S_2^3$', False),
              ('skew', r'skewness $S_3/S_2^{3/2}$ (time asymmetry)', False)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    tau = S_ref['taus']
    for ax, (key, title, ratio) in zip(axes, panels):
        if ratio:
            ax.axhline(1, color=DATA_COLOR, lw=1, ls='--', label='data')
        else:
            ax.plot(tau, S_ref[key], 'o-', color=DATA_COLOR, lw=2, ms=3, label='data')
        for k, S in S_runs.items():
            y = S[key] / S_ref[key] if ratio else S[key]
            ax.plot(tau, y, 'o-', color=cols[k], lw=1.5, ms=3, label=k)
        ax.set_xscale('log')
        if key != 'skew':
            ax.set_yscale('log')
        ax.set_title(title); ax.set_xlabel(r'$\tau$'); ax.grid(alpha=0.2, which='both')
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    plt.show()


def plot_increment_pdfs(S_ref, S_runs):
    """PDF of normalised increments dx_tau / std at each tau, runs overlaid."""
    cols = _colors(S_runs)
    taus = S_ref['pdf_taus']
    bins = np.linspace(-15, 15, 241)
    fig, axes = plt.subplots(1, len(taus), figsize=(4.2 * len(taus), 3.6), sharey=True)
    for ax, tau in zip(axes, taus):
        _hist_line(ax, S_ref['inc'][tau], bins, color=DATA_COLOR, lw=2, label='data')
        for k, S in S_runs.items():
            _hist_line(ax, S['inc'][tau], bins, color=cols[k], lw=1.3, label=k)
        ax.set_yscale('log'); ax.set_title(fr'$\tau={tau}$'); ax.grid(alpha=0.2)
        ax.set_xlabel(r'$\delta_\tau x / \sigma_\tau$')
    axes[0].set_ylabel('density'); axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Increment PDFs (small tau = intermittent tails)')
    fig.tight_layout()
    plt.show()


def wavelet_ks(S_ref, S_runs, n=100_000, seed=0):
    """KS distance between data and run |W psi x| per band: DataFrame runs x bands."""
    rng = np.random.default_rng(seed)
    J, Q = S_ref['wav_JQ']
    rows = {}
    for k, S in S_runs.items():
        rows[k] = [stats.ks_2samp(rng.choice(S_ref['wav'][b], n), rng.choice(S['wav'][b], n)).statistic
                   for b in range(J * Q)]
    return pd.DataFrame(rows, index=[f'j{b // Q}q{b % Q}' for b in range(J * Q)]).T


def plot_wavelet_hists(S_ref, S_runs):
    """|W psi x| histograms, one small panel per band (j rows, q columns), plus KS heatmap."""
    cols = _colors(S_runs)
    J, Q = S_ref['wav_JQ']
    fig, axes = plt.subplots(J, Q, figsize=(4 * Q, 2.3 * J), squeeze=False)
    for b in range(J * Q):
        ax = axes[b // Q, b % Q]
        hi = 1.1 * max(np.percentile(S['wav'][b], 99.99) for S in [S_ref, *S_runs.values()])
        bins = np.linspace(0, hi, 101)
        _hist_line(ax, S_ref['wav'][b], bins, color=DATA_COLOR, lw=2, label='data')
        for k, S in S_runs.items():
            _hist_line(ax, S['wav'][b], bins, color=cols[k], lw=1.2, label=k)
        ax.set_yscale('log'); ax.grid(alpha=0.2)
        ax.set_title(f'j={b // Q}, q={b % Q}' + ('  (finest)' if b == 0 else ''), fontsize=9)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(r'Wavelet coefficient magnitudes $|W_\psi x|$ (Morlet, log density)')
    fig.tight_layout()
    plt.show()

    ks = wavelet_ks(S_ref, S_runs)
    fig, ax = plt.subplots(figsize=(1 + 0.55 * ks.shape[1], 0.5 + 0.45 * ks.shape[0]))
    im = ax.imshow(ks.values, cmap='Blues', vmin=0, aspect='auto')     # sequential, one hue
    ax.set_xticks(range(ks.shape[1]), ks.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(ks.shape[0]), ks.index, fontsize=8)
    for i in range(ks.shape[0]):
        for j in range(ks.shape[1]):
            v = ks.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if v > 0.6 * ks.values.max() else 'black')
    fig.colorbar(im, ax=ax, label='KS distance to data')
    ax.set_title('Wavelet-band mismatch (0 = identical distribution)')
    fig.tight_layout()
    plt.show()
    return ks


def _rel_err(run, threshold):
    e, p = run['barphi_e'].numpy(), run['barphi_p'].numpy()
    keep = e[-1] > threshold
    e, p = e[:, keep], p[:, keep]
    t = run['t'].numpy()[1:len(e) + 1]
    return t, 2 * np.abs(e - p) / (np.abs(e) + np.abs(p))


def plot_moment_matching(runs, threshold=1e-8):
    """Mean rel. error vs t, the same vs 1-t (resolves t -> 1), final-step distribution."""
    runs = {k: r for k, r in runs.items() if 'barphi_e' in r}
    if not runs:
        print('no aux moments loaded')
        return
    cols = _colors(runs)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 4))
    bins = np.logspace(-8, np.log10(2), 60)
    for k, r in runs.items():
        t, rel = _rel_err(r, threshold)
        m = rel.mean(1)
        a1.semilogy(t, m, color=cols[k], lw=0.8, label=k)
        a2.loglog(1 - t[t < 1], m[t < 1], color=cols[k], lw=1.2, alpha=0.8, label=k)
        a3.hist(rel[-1], bins=bins, histtype='step', lw=1.5, color=cols[k], label=k)
    a1.set_xlabel('t'); a1.set_title('mean rel. error over moments')
    a2.set_xlabel('1 - t'); a2.invert_xaxis(); a2.set_title('same, zoomed on t -> 1')
    a3.set_xscale('log'); a3.set_yscale('log'); a3.set_xlabel('final-step rel. error')
    a3.set_ylabel('moments'); a3.set_title('final-step distribution (2 = sign flip)')
    for a in (a1, a2, a3):
        a.grid(alpha=0.2, which='both')
    a1.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    plt.show()


def _step_series(run, threshold):
    """Per-step quantities aligned on t[1:]: t, h, mean moment error, theta, dH."""
    t = run['t'].numpy()
    n = min(len(t) - 1, *(len(run[k]) for k in ('theta', 'barphi_e') if k in run))
    out = dict(t=t[1:n + 1], h=np.diff(t)[:n])
    if 'barphi_e' in run:
        out['mm'] = _rel_err(run, threshold)[1][:n].mean(1)
    if 'theta' in run:
        out['theta'] = run['theta'].numpy()[:n]
    if 'dH' in run:
        out['dH'] = run['dH'].numpy().ravel()[:n]
    return out


def plot_theta_check(runs, sigma=3.5, threshold=1e-8, top=8, window=1e-3):
    """Is the late jump in moment error a step-size instability or a blow-up of theta?

    Stacked panels on a shared 1-t axis (log):
      1. mean moment rel. error (where the jump is)
      2. step size h (what the schedule does near t=1)
      3. ||theta_t||, theta as saved (= corrector coefficients / (h sigma^2))
      4. ||theta_t|| * h * sigma^2 = size of the corrector actually applied that step
      5. |dH_t| (entropy increment, = -theta . d/dt phi_bar)
    Step-size instability: panel 4 spikes at the jump while panel 3 is smooth.
    Ill-conditioned solve: panel 3 itself grows / spikes at the jump.
    Also prints, per run, the theta components with the largest |theta| * h * sigma^2
    in the last `window` of 1-t (indices = rows of theta, after deduplication).
    """
    runs = {k: r for k, r in runs.items() if 'theta' in r}
    if not runs:
        print('no lagrange_multipliers loaded')
        return
    cols = _colors(runs)
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    for k, r in runs.items():
        s = _step_series(r, threshold)
        x = 1 - s['t']
        ok = x > 0
        norm = np.linalg.norm(s['theta'], axis=1)
        applied = norm * s['h'] * sigma ** 2
        kw = dict(color=cols[k], lw=1.1, alpha=0.8, label=k)
        if 'mm' in s:
            axes[0].loglog(x[ok], s['mm'][ok], **kw)
        axes[1].loglog(x[ok], s['h'][ok], **kw)
        axes[2].loglog(x[ok], norm[ok], **kw)
        axes[3].loglog(x[ok], applied[ok], **kw)
        if 'dH' in s:
            axes[4].loglog(x[ok], np.abs(s['dH'][ok]) + 1e-300, **kw)

        late = ok & (x < window)
        if late.any():
            a = np.abs(s['theta'][late]) * (s['h'][late] * sigma ** 2)[:, None]
            peak = a.max(0)
            idx = np.argsort(peak)[::-1][:top]
            print(f'{k}: {s["theta"].shape[1]} theta components; largest |theta|*h*sigma^2 '
                  f'for 1-t < {window:g}:')
            print('   ' + '  '.join(f'{i}:{peak[i]:.2e} @1-t={x[late][a[:, i].argmax()]:.1e}'
                                    for i in idx))
    titles = ['mean moment rel. error', 'step size h', r'$\|\theta_t\|$ (saved, / h$\sigma^2$)',
              r'$\|\theta_t\|\,h\,\sigma^2$ (corrector actually applied)', r'$|dH_t|$']
    for a, ti in zip(axes, titles):
        a.set_title(ti, fontsize=10, loc='left'); a.grid(alpha=0.2, which='both')
    axes[-1].set_xlabel('1 - t'); axes[-1].invert_xaxis()
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    plt.show()


def summary_table(S_ref, S_runs, runs, threshold=1e-8, ks=None):
    """One row per run: settings, cost, moment matching, and physics errors."""
    rows = {}
    for k, S in S_runs.items():
        r, c = runs[k], runs[k]['cfg']
        row = {'reg': c.get('regularization'), 'sched': c.get('schedule_exponent'),
               'n_terms': len(c.get('terms', [])), 'nt': c.get('nt'), 'n1': c.get('n1'),
               'runtime_h': round(r['runtime_h'], 1)}
        if 'barphi_e' in r:
            _, rel = _rel_err(r, threshold)
            row.update({'mm_final_mean': rel[-1].mean(), 'mm_final_p90': np.percentile(rel[-1], 90),
                        'mm_n_above_1': int((rel[-1] > 1).sum())})
        row.update({
            'S2_ratio_tau1': S['S2'][0] / S_ref['S2'][0],
            'flat4_ratio_tau1': S['flat4'][0] / S_ref['flat4'][0],
            'flat6_ratio_tau1': S['flat6'][0] / S_ref['flat6'][0],
            'skew_tau1 (data %.3f)' % S_ref['skew'][0]: S['skew'][0],
            'spec_logrms': np.sqrt(np.mean(np.log(S['spectrum'] / S_ref['spectrum']) ** 2)),
            'marg_KS': stats.ks_2samp(S_ref['values'], S['values']).statistic,
        })
        if ks is not None:
            row['wav_KS_max'] = ks.loc[k].max()
        rows[k] = row
    return pd.DataFrame(rows).T
