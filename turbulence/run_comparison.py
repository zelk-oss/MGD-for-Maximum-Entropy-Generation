"""Side-by-side comparison of finished turbulence runs (used by compare_runs.ipynb).

Loads each run by its --label (samples, sampling times, aux moments, config.json,
run.log), rebuilds the reference data x1 exactly as run_SDE.py does, and draws one
compact figure per statistic with every run overlaid (data always in black):

    time series | marginal PDF | power spectrum | structure functions / flatness /
    skewness | increment PDFs | wavelet-coefficient histograms (+ KS heatmap) |
    moment matching | summary table

Statistics are computed once per array (compute_stats) and reused by every plot.

The plot_group_* functions (used by compare_july_vs_sept.ipynb) compare a seed
ensemble with a single run when the two were fitted on DIFFERENT reference data:
everything is shown relative to each group's own data.
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


def find_configs(pattern, root=ROOT):
    """Config names (no .pt) of the saved runs whose samples file matches a glob.

    For runs without a --label (e.g. the 2026-07-28 batch:
    '*_n1_5000_lam5e-07_seed_*_terms2f1130a8_20260728_1232*').
    """
    files = (root / 'saved_results' / 'samples').glob(pattern)
    return sorted({f.stem if f.suffix == '.pt' else f.name for f in files})


def load_config(config, root=ROOT, label=None, extras=True):
    """One saved run by its full config name. extras=False skips theta/dH/aux moments.

    run['t'] is None if the sampling_times file is missing (seen for one July seed).
    """
    sr = root / 'saved_results'
    xt = _load_pt(sr / 'samples' / f'{config}.pt').float()
    tf = sr / 'sampling_times' / f'{config}.pt'
    run = dict(label=label, config=config, xt=xt.reshape(xt.shape[0], -1),
               t=_load_pt(tf).double() if tf.exists() or tf.with_suffix('').exists() else None)
    if extras:
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
    return run


def load_runs(runs, root=ROOT, seed=None):
    """runs: {display name: --label}. Returns ({name: run dict}, [missing labels])."""
    out, missing = {}, []
    for name, label in runs.items():
        config = _find_config(root, label, seed)
        if config is None:
            missing.append(label)
            continue
        out[name] = load_config(config, root, label=label)
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

def log_bands(freq, spectrum, per_octave=4):
    """Spectrum averaged over log-spaced frequency bands (empty bands dropped).

    Every octave then weighs the same; on the raw rfft bins half of the points lie
    in the top octave (f > 0.25), so a mean over bins mostly measures that octave.
    """
    T = int(round(1 / freq[0]))
    edges = np.geomspace(freq[0], 0.5, int(np.log2(T / 2)) * per_octave + 1)
    idx = np.clip(np.searchsorted(edges, freq, side='right') - 1, 0, len(edges) - 2)
    keep = np.bincount(idx, minlength=len(edges) - 1) > 0
    mean = lambda v: np.bincount(idx, weights=v, minlength=len(edges) - 1)[keep] / \
        np.bincount(idx, minlength=len(edges) - 1)[keep]
    return mean(freq), mean(spectrum)


def compute_stats(x, taus=None, pdf_taus=(1, 4, 16, 64), wav_J=6, wav_Q=3,
                  n_wav=2000, seed=0, chunk=1000):
    """All statistics used by the plots, for x of shape (B, T), periodic in T.

    Wavelet coefficients W = x * psi, psi = the first J x Q bands of the runs' psi bank
    (Morlet, Q=3; see BANKS): the per-band moments
    (wav_E = E|W|^2, wav_flat = E|W|^4 / (E|W|^2)^2, wav_sparsity = (E|W|)^2 / E|W|^2)
    use all B series; the samples kept for histograms and KS ('wav' = |W|,
    'wav_re' = Re W, float32) use n_wav random series.
    """
    x = x.double().numpy() if torch.is_tensor(x) else np.asarray(x, float)
    B, T = x.shape
    taus = np.unique(np.round(np.logspace(0, np.log10(T // 2), 30)).astype(int)) \
        if taus is None else np.asarray(taus)
    s = dict(T=T, B=B, taus=taus, values=x.ravel())

    ps = np.abs(np.fft.rfft(x - x.mean(1, keepdims=True), axis=1)) ** 2
    s['freq'], s['spectrum'] = np.fft.rfftfreq(T)[1:], ps.mean(0)[1:]
    s['freq_band'], s['spec_band'] = log_bands(s['freq'], s['spectrum'])

    S = {p: [] for p in (2, 3, 4, 6)}
    for tau in taus:
        d = np.roll(x, -tau, axis=1) - x                      # periodic increments
        d2 = d * d                                            # products, not d ** p: numpy's
        d4 = d2 * d2                                          # float pow is ~12x slower here
        for p, v in ((2, d2), (3, d2 * d), (4, d4), (6, d4 * d2)):
            S[p].append(np.mean(v))                           # signed for p=3
    S = {p: np.array(v) for p, v in S.items()}
    s.update(S2=S[2], skew=S[3] / S[2] ** 1.5, flat4=S[4] / S[2] ** 2, flat6=S[6] / S[2] ** 3)

    s['pdf_taus'] = pdf_taus
    s['inc'] = {tau: ((np.roll(x, -tau, axis=1) - x) /
                      np.sqrt(np.mean((np.roll(x, -tau, axis=1) - x) ** 2))).ravel()
                for tau in pdf_taus}

    psi = np.asarray(init_band_pass('morlet', T, J=wav_J, Q=wav_Q, high_freq=0.49, wav_norm='l1'))
    wavelet = lambda v: np.fft.ifft(np.fft.fft(v)[:, None, :] * psi[None], axis=-1)   # (b, bands, T)
    # the n_wav subsampled series go first, so one pass gives both the moments over all
    # B series and the kept samples (same subsample as rng.choice alone would give)
    sel = np.random.default_rng(seed).choice(B, min(n_wav, B), replace=False)
    order = np.concatenate([sel, np.setdiff1d(np.arange(B), sel)])
    m1, m2, m4 = (np.zeros(len(psi)) for _ in range(3))
    kept = []
    for i in range(0, B, chunk):
        w = wavelet(x[order[i:i + chunk]])
        if i < len(sel):
            kept.append(w[:len(sel) - i])
        a2 = w.real * w.real + w.imag * w.imag
        m1 += np.sqrt(a2).sum((0, 2)); m2 += a2.sum((0, 2)); m4 += (a2 * a2).sum((0, 2))
    m1, m2, m4 = m1 / (B * T), m2 / (B * T), m4 / (B * T)
    s.update(wav_E=m2, wav_flat=m4 / m2 ** 2, wav_sparsity=m1 ** 2 / m2,
             wav_xi=np.abs(np.fft.fftfreq(T)[np.abs(psi).argmax(-1)]))   # band centre frequency

    wt = np.concatenate(kept).transpose(1, 0, 2).reshape(len(psi), -1)     # (bands, samples)
    s['wav'], s['wav_re'] = np.abs(wt).astype(np.float32), wt.real.astype(np.float32)
    s['wav_JQ'], s['n_wav'] = (wav_J, wav_Q), len(sel)
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


def ks_distance(a, b):
    """Two-sample KS statistic (= scipy.stats.ks_2samp(a, b).statistic; ~4x faster at 1e6 points)."""
    a, b = np.sort(np.ravel(a)), np.sort(np.ravel(b))
    v = np.sort(np.concatenate([a, b]))          # sorted queries: searchsorted stays in cache
    return float(np.max(np.abs(np.searchsorted(a, v, 'right') / len(a) -
                               np.searchsorted(b, v, 'right') / len(b))))


def wavelet_ks(S_ref, S_runs, n=100_000, seed=0):
    """KS distance between data and run |W psi x| per band: DataFrame runs x bands."""
    rng = np.random.default_rng(seed)
    J, Q = S_ref['wav_JQ']
    rows = {}
    for k, S in S_runs.items():
        rows[k] = [ks_distance(rng.choice(S_ref['wav'][b], n), rng.choice(S['wav'][b], n))
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


def _own_err(run, threshold):
    """Moment error on each moment's own scale: |e - p| / max_t |e| (whole-run max).

    Unlike the relative error 2|e-p|/(|e|+|p|), it does not blow up when a target
    moment crosses zero: on long_full_sched3_reg1e-2 all 45 bulk spikes of the
    relative error were single moments crossing zero, and vanish on this scale
    (2026-09-25). The whole-run max (not the running max the adaptive time grid
    uses) avoids inflating the first steps, where some targets start near zero.
    Returns t, err (steps x moments) and the kept moment indices.
    """
    e, p = run['barphi_e'].numpy(), run['barphi_p'].numpy()
    keep = e[-1] > threshold                     # same moment set as _rel_err
    e, p = e[:, keep], p[:, keep]
    t = run['t'].numpy()[1:len(e) + 1]
    return t, np.abs(e - p) / np.abs(e).max(0), np.flatnonzero(keep)


def plot_moment_matching(runs, threshold=1e-8, target=1e-2):
    """Own-scale moment error: median / p90 / max over moments vs t and vs 1-t, final step.

    With ~200 moments the mean is carried by the worst few, so the ensemble is shown
    as three curves per run: median (dashed, a typical moment), p90 (solid) and max
    (dotted, the worst moment). The dashed grey line is `target`.
    """
    runs = {k: r for k, r in runs.items() if 'barphi_e' in r}
    if not runs:
        print('no aux moments loaded')
        return
    cols = _colors(runs)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 4.5))
    bins = np.logspace(-9, 1, 60)
    for k, r in runs.items():
        t, err, _ = _own_err(r, threshold)
        stats = [('median', np.median(err, 1), '--'), ('p90', np.percentile(err, 90, 1), '-'),
                 ('max', err.max(1), ':')]
        for name, v, ls in stats:
            kw = dict(color=cols[k], ls=ls, lw=1.0, label=k if name == 'p90' else None)
            a1.semilogy(t, v, **kw)
            a2.loglog(1 - t[t < 1], v[t < 1], **kw)
        a3.hist(err[-1], bins=bins, histtype='step', lw=1.5, color=cols[k], label=k)
    for a in (a1, a2):
        a.axhline(target, color='0.5', lw=0.8, ls='--')
    a3.axvline(target, color='0.5', lw=0.8, ls='--')
    a1.set_xlabel('t'); a1.set_title('own-scale error |e-p| / max_t|e|: p90 (solid), median (--), max (:)',
                                     fontsize=9)
    a2.set_xlabel('1 - t'); a2.invert_xaxis(); a2.set_title('same, zoomed on t -> 1')
    a3.set_xscale('log'); a3.set_yscale('log'); a3.set_xlabel('final-step own-scale error')
    a3.set_ylabel('moments'); a3.set_title(f'final step (dashed: target {target:g})')
    for a in (a1, a2, a3):
        a.grid(alpha=0.2, which='both')
    a1.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    plt.show()


def moments_above(runs, threshold=1e-8, target=1e-2, root=ROOT):
    """Per run: the moments whose final own-scale error exceeds `target`, with names.

    Names come from the run's fitted potentials (codes/find_duplicate_statistics.py,
    statistic_labels): they need experiments/<...>/<config>/fitted_potentials/, i.e.
    run this where the run lives (Jean Zay); otherwise indices are shown as #i.
    """
    from codes.find_duplicate_statistics import statistic_labels
    for k, r in runs.items():
        if 'barphi_e' not in r:
            continue
        t, err, idx = _own_err(r, threshold)
        try:
            labels, _ = statistic_labels(root, r['config'])
        except Exception as exc:                  # missing/partial fitted potentials
            labels = None
            print(f'  ({k}: no names, {type(exc).__name__}: {exc})')
        if labels is not None and len(labels) != len(r['barphi_e'][0]):
            print(f'  ({k}: rebuilt potentials give {len(labels)} statistics, run has '
                  f'{len(r["barphi_e"][0])}; names unreliable, showing indices)')
            labels = None
        bad = np.argsort(err[-1])[::-1]
        bad = bad[err[-1][bad] > target]
        print(f'{k}: {len(bad)}/{len(idx)} moments above {target:g} at the final step')
        for j in bad:
            name = labels[idx[j]] if labels else f'#{idx[j]}'
            e_end = float(r['barphi_e'][-1, idx[j]]); p_end = float(r['barphi_p'][-1, idx[j]])
            print(f'   {err[-1, j]:8.2e}   target {e_end: .3e}  walkers {p_end: .3e}   {name}')


def _step_series(run, threshold):
    """Per-step quantities aligned on t[1:]: t, h, mean moment error, theta, dH."""
    t = run['t'].numpy()
    n = min(len(t) - 1, *(len(run[k]) for k in ('theta', 'barphi_e') if k in run))
    out = dict(t=t[1:n + 1], h=np.diff(t)[:n])
    if 'barphi_e' in run:
        out['mm'] = np.percentile(_own_err(run, threshold)[1][:n], 90, axis=1)
    if 'theta' in run:
        out['theta'] = run['theta'].numpy()[:n]
    if 'dH' in run:
        out['dH'] = run['dH'].numpy().ravel()[:n]
    return out


def plot_theta_check(runs, sigma=3.5, threshold=1e-8, top=8, window=1e-3):
    """Is the late jump in moment error a step-size instability or a blow-up of theta?

    Stacked panels on a shared 1-t axis (log):
      1. moment error, p90 over moments on each moment's own scale (where the jump is)
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
    titles = ['moment error, p90 over moments (own scale)', 'step size h', r'$\|\theta_t\|$ (saved, / h$\sigma^2$)',
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
            _, err, _ = _own_err(r, threshold)
            row.update({'mm_final_median': np.median(err[-1]), 'mm_final_p90': np.percentile(err[-1], 90),
                        'mm_final_max': err[-1].max(), 'mm_n_above_1pct': int((err[-1] > 1e-2).sum())})
        for c, v in physics_metrics(S, S_ref, ks.loc[k] if ks is not None else None).items():
            if c == 'skew_tau1':
                row['skew_tau1 (data %.3f)' % S_ref['skew'][0]] = v
            elif c != 'skew_diff_tau1':
                row[c] = v
        rows[k] = row
    return pd.DataFrame(rows).T


def _logrms(a, b):
    return float(np.sqrt(np.mean(np.log(a / b) ** 2)))


def physics_metrics(S, S_ref, ks_row=None):
    """Sample-quality numbers of one run against ITS OWN reference data (ratios: 1 = perfect).

    spec_logrms is over the raw rfft bins (dominated by the top octave, kept for
    comparison with earlier notes); spec_logrms_oct is over log-spaced bands, every
    octave weighted equally. wav_E_logrms / wav_flat_logrms: same over the Morlet bands.
    """
    out = {
        'S2_ratio_tau1': S['S2'][0] / S_ref['S2'][0],
        'flat4_ratio_tau1': S['flat4'][0] / S_ref['flat4'][0],
        'flat6_ratio_tau1': S['flat6'][0] / S_ref['flat6'][0],
        'skew_tau1': S['skew'][0],
        'skew_diff_tau1': S['skew'][0] - S_ref['skew'][0],
        'spec_logrms': _logrms(S['spectrum'], S_ref['spectrum']),
        'spec_logrms_oct': _logrms(S['spec_band'], S_ref['spec_band']),
        'wav_E_logrms': _logrms(S['wav_E'], S_ref['wav_E']),
        'wav_flat_logrms': _logrms(S['wav_flat'], S_ref['wav_flat']),
        'marg_KS': ks_distance(S_ref['values'], S['values']),
    }
    if ks_row is not None:
        out['wav_KS_max'] = float(np.max(ks_row))
    return out


# ================================================================================
# Groups: a seed ensemble vs a single run, each against its own reference data
# ================================================================================
# A group is dict(ref=<stats of its reference data>, runs=[<stats per seed>],
# ks=[<per-band KS row per seed>], color=..., and optionally noise=data_noise(x_ref)).
# Groups may use DIFFERENT reference data (e.g. the July runs were fitted on
# subseries_len=1024, the September ones on 512), so every quantity is shown as
# run / own data, never run / another group's data. A group with several runs is
# drawn as its median and 5-95% band across seeds; a single run as a line.
#
# Data-noise band: the reference series are split into two random halves A, B many
# times, and every curve / metric is computed for A vs B. Two halves of the same data
# differ only by sampling noise, so a run inside the band is as close to its data as
# the data is to itself. The spread is rescaled to the comparison actually made
# (n_run generated series vs n_ref data series instead of n/2 vs n/2): noise variance
# ~ 1/n_a + 1/n_b, so deviations shrink by c = sqrt((1/n_ref + 1/n_run) / (2 / (n/2))),
# i.e. 1/sqrt(2) when n_run = n_ref (in log for ratios, linearly for differences and
# distances). Statistics computed on the n_wav-series wavelet subsample (the per-band
# KS) use n_wav on both sides of both comparisons, so c = 1 for them. The rescaling
# assumes independent series; MGD samples match the potentials' moments as an
# ensemble, so for statistics close to the potentials the true noise can be smaller.

# kind of each curve / metric: (how it deviates from perfect, which sample size sets its noise)
CURVES = {'spectrum': 'ratio', 'spec_band': 'ratio', 'S2': 'ratio', 'flat4': 'ratio',
          'flat6': 'ratio', 'skew': 'diff', 'wav_E': 'ratio', 'wav_flat': 'ratio',
          'wav_sparsity': 'ratio'}
METRIC_KIND = {'S2_ratio_tau1': ('ratio', 'full'), 'flat4_ratio_tau1': ('ratio', 'full'),
               'flat6_ratio_tau1': ('ratio', 'full'), 'skew_diff_tau1': ('diff', 'full'),
               'spec_logrms': ('dist', 'full'), 'spec_logrms_oct': ('dist', 'full'),
               'wav_E_logrms': ('dist', 'full'), 'wav_flat_logrms': ('dist', 'full'),
               'marg_KS': ('dist', 'full'), 'wav_KS_max': ('dist', 'wav')}


SAMPLE_KEYS = ('values', 'inc', 'wav', 'wav_re')     # the large per-sample arrays of compute_stats


def run_summary(S, R, keep_samples=True):
    """Stats of one run against reference stats R: (S, per-band KS row, physics metrics).
    keep_samples=False drops S's sample arrays once the KS row and metrics are computed
    (~125 MB per 5000-series run), keeping only the curves; then S cannot be used for
    histograms or KS again."""
    ks = wavelet_ks(R, {0: S}).loc[0]
    m = physics_metrics(S, R, ks)
    if not keep_samples:
        S = {k: v for k, v in S.items() if k not in SAMPLE_KEYS}
    return S, ks, m


def compare_curves(S, R):
    """Every curve of run stats S against reference stats R: ratio, or difference for skew."""
    return {k: S[k] - R[k] if kind == 'diff' else S[k] / R[k] for k, kind in CURVES.items()}


def data_halves(x, seed=0):
    """Two disjoint random halves of the reference series (for a data-vs-data noise scale)."""
    idx = np.random.default_rng(seed).permutation(x.shape[0])
    h = x.shape[0] // 2
    return x[idx[:h]], x[idx[h:2 * h]]


def _noise_split(x, r, seed):
    xa, xb = data_halves(x, seed=seed + r)
    Sa = compute_stats(xa, pdf_taus=(), seed=2 * r + 1)
    Sb = compute_stats(xb, pdf_taus=(), seed=2 * r + 2)
    ks = wavelet_ks(Sb, {'a': Sa}, seed=r).loc['a']
    return dict(curves=compare_curves(Sa, Sb), metrics=physics_metrics(Sa, Sb, ks),
                ks=ks, B=Sa['B'], n_wav=Sa['n_wav'])


def data_noise(x, n_rep=50, seed=0, n_jobs=None):
    """Half A vs half B of the reference data x for n_rep random splits: per split the
    curves (compare_curves), physics metrics and per-band KS row (samples dropped).
    Splits run in n_jobs threads (numpy's FFT, sort and array arithmetic release the
    GIL); default min(8, cores available to this process): each split holds ~0.4 GB."""
    from concurrent.futures import ThreadPoolExecutor
    x = x.numpy() if torch.is_tensor(x) else np.asarray(x)
    with ThreadPoolExecutor(n_jobs or min(8, len(os.sched_getaffinity(0)))) as ex:
        return list(ex.map(lambda r: _noise_split(x, r, seed), range(n_rep)))


def _noise_factor(g, size):
    """c = sqrt((1/n_ref + 1/n_run) / (1/m + 1/m)), m = series per half (see above)."""
    rep, R, run = g['noise'][0], g['ref'], g['runs'][0]
    key = 'B' if size == 'full' else 'n_wav'
    return np.sqrt((1 / R[key] + 1 / run[key]) / (2 / rep[key]))


def _shrink(v, kind, c):
    v = np.asarray(v, float)
    return np.exp(c * np.log(v)) if kind == 'ratio' else c * v


def noise_band(g, key, q=(5, 95)):
    """(lo, hi) of the rescaled data-noise spread of a curve (CURVES key), a metric
    (METRIC_KIND key) or 'ks' (per-band KS row), across the random splits."""
    if key in CURVES:
        kind, size, vals = CURVES[key], 'full', [r['curves'][key] for r in g['noise']]
    elif key == 'ks':
        kind, size, vals = 'dist', 'wav', [r['ks'].values for r in g['noise']]
    else:
        (kind, size), vals = METRIC_KIND[key], [r['metrics'][key] for r in g['noise']]
    v = _shrink(np.array(vals), kind, _noise_factor(g, size))
    return np.percentile(v, q[0], axis=0), np.percentile(v, q[1], axis=0)


def group_summary(groups):
    """One row per group: median [p5, p95] across its runs of every physics metric,
    plus one 'data noise' row per group with a noise band: [p5, p95] for ratios and
    differences, p95 for distances (KS, log-rms), after the rescaling above."""
    rows = {}
    for name, g in groups.items():
        m = pd.DataFrame(g['metrics'] if 'metrics' in g else
                         [physics_metrics(S, g['ref'], ks) for S, ks in zip(g['runs'], g['ks'])])
        n = len(m)
        rows[f'{name} (n={n})'] = {c: (f'{m[c].median():.3g}' if n == 1 else
                                       f'{m[c].median():.3g} [{m[c].quantile(.05):.3g}, {m[c].quantile(.95):.3g}]')
                                   for c in m.columns}
        if g.get('noise'):
            row = {}
            for c in m.columns:
                if c not in METRIC_KIND:
                    row[c] = '-'
                    continue
                lo, hi = noise_band(g, c)
                row[c] = f'< {hi:.3g}' if METRIC_KIND[c][0] == 'dist' else f'[{lo:.3g}, {hi:.3g}]'
            rows[f'  data noise ({name})'] = row
    return pd.DataFrame(rows).T


def _band(ax, x, ys, color, label, logy=True):
    ys = np.asarray(ys)
    if len(ys) == 1:
        ax.plot(x, ys[0], color=color, lw=1.8, label=label)
        return
    lo, med, hi = np.percentile(ys, [5, 50, 95], axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.25, lw=0)
    ax.plot(x, med, color=color, lw=1.8, label=f'{label}: median, 5-95% band')


def _noise_lines(ax, x, g, key, name):
    """Edges of the group's data-noise band (5-95%, rescaled) as two thin dotted lines."""
    if not g.get('noise'):
        return
    lo, hi = noise_band(g, key)
    ax.plot(x, lo, color=g['color'], lw=0.9, ls=':', label=f'data noise 5-95% ({name} ref)')
    ax.plot(x, hi, color=g['color'], lw=0.9, ls=':')


def _group_curve_panels(groups, panels, xkey, xlabel, figsize):
    """One panel per (curve key, title): every group's run band / line against its own
    data, plus its data-noise band; x from the group's reference stats[xkey]."""
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, squeeze=False)
    for ax, (key, title) in zip(axes[0], panels):
        ratio = CURVES[key] == 'ratio'
        for name, g in groups.items():
            x = g['ref'][xkey]
            _band(ax, x, [compare_curves(S, g['ref'])[key] for S in g['runs']], g['color'], name)
            _noise_lines(ax, x, g, key, name)
        ax.axhline(1 if ratio else 0, color=DATA_COLOR, lw=1, ls='--')
        ax.set_xscale('log')
        if ratio:
            ax.set_yscale('log')
        ax.set_title(title); ax.set_xlabel(xlabel); ax.grid(alpha=0.2, which='both')
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    plt.show()


def plot_group_spectra(groups):
    """Spectrum / own data spectrum on the raw rfft bins and averaged over log-spaced
    bands (4 per octave), one band or line per group, plus each group's data-noise band."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    for ax, (key, fkey, title) in zip(axes, [('spectrum', 'freq', 'rfft bins'),
                                             ('spec_band', 'freq_band', 'log-spaced bands, 4 per octave')]):
        for name, g in groups.items():
            f = g['ref'][fkey]
            _band(ax, f, [compare_curves(S, g['ref'])[key] for S in g['runs']], g['color'], name)
            _noise_lines(ax, f, g, key, name)
        ax.axhline(1, color=DATA_COLOR, lw=1, ls='--')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('frequency')
        ax.set_title(f'Power spectrum, gen / own data ({title})'); ax.grid(alpha=0.2, which='both')
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    plt.show()


def plot_group_structure(groups):
    """S2, flatness, S6/S2^3 as gen / own data, and skewness gen - own data, vs tau."""
    _group_curve_panels(groups, [('S2', r'$S_2(\tau)$  gen / own data'),
                                 ('flat4', r'flatness $S_4/S_2^2$  gen / own data'),
                                 ('flat6', r'$S_6/S_2^3$  gen / own data'),
                                 ('skew', r'skewness  gen $-$ own data')],
                        'taus', r'$\tau$', (19, 4))


def plot_group_band_moments(groups):
    """Scale-by-scale moments of the psi coefficients W (all series), gen / own data,
    against each band's centre frequency: energy E|W|^2 (a wavelet spectrum), flatness
    E|W|^4 / (E|W|^2)^2 (intermittency at that scale) and sparsity (E|W|)^2 / E|W|^2
    (lower = sparser; the scattering-spectra 'sparsity factor')."""
    _group_curve_panels(groups, [('wav_E', r'energy $E|W|^2$  gen / own data'),
                                 ('wav_flat', r'flatness $E|W|^4/(E|W|^2)^2$  gen / own data'),
                                 ('wav_sparsity', r'sparsity $(E|W|)^2/E|W|^2$  gen / own data')],
                        'wav_xi', 'band centre frequency', (17, 4))


def plot_group_pdfs(groups, x_by_group):
    """Marginal and increment PDFs: one row per group, its data in black, a few of its
    runs pooled (x_by_group: {group: (x_ref, [sample arrays])})."""
    taus = next(iter(groups.values()))['ref']['pdf_taus']
    fig, axes = plt.subplots(len(groups), 1 + len(taus), figsize=(4 * (1 + len(taus)), 3.2 * len(groups)),
                             squeeze=False)
    bins_x, bins_i = np.linspace(-8, 8, 161), np.linspace(-15, 15, 241)
    for row, (name, g) in zip(axes, groups.items()):
        x_ref, xs = x_by_group[name]
        pooled = np.concatenate([np.asarray(x) for x in xs])
        _hist_line(row[0], np.asarray(x_ref).ravel(), bins_x, color=DATA_COLOR, lw=2, label='own data')
        _hist_line(row[0], pooled.ravel(), bins_x, color=g['color'], lw=1.4, label=name)
        row[0].set_title(f'{name}: marginal', fontsize=9)
        for ax, tau in zip(row[1:], taus):
            _hist_line(ax, g['ref']['inc'][tau], bins_i, color=DATA_COLOR, lw=2)
            d = np.roll(pooled, -tau, axis=1) - pooled
            _hist_line(ax, (d / np.sqrt(np.mean(d ** 2))).ravel(), bins_i, color=g['color'], lw=1.4)
            ax.set_title(fr'{name}: increments $\tau={tau}$', fontsize=9)
        for ax in row:
            ax.set_yscale('log'); ax.grid(alpha=0.2)
        row[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    plt.show()


def plot_group_wavelet_ks(groups):
    """Per-band KS to own data: median over each group's runs, plus the p95 of the
    data-noise KS (half A vs half B over the random splits)."""
    rows = {}
    for name, g in groups.items():
        ks = pd.DataFrame(g['ks'])
        rows[name if len(ks) == 1 else f'{name} (median of {len(ks)})'] = ks.median(0)
        if len(ks) > 1:
            rows[f'{name} (p95 of {len(ks)})'] = ks.quantile(0.95)
        if g.get('noise'):
            rows[f'data noise p95 ({name} ref)'] = pd.Series(noise_band(g, 'ks')[1], index=ks.columns)
    ks = pd.DataFrame(rows).T
    fig, ax = plt.subplots(figsize=(1 + 0.55 * ks.shape[1], 0.6 + 0.45 * ks.shape[0]))
    im = ax.imshow(ks.values, cmap='Blues', vmin=0, aspect='auto')
    ax.set_xticks(range(ks.shape[1]), ks.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(ks.shape[0]), ks.index, fontsize=8)
    for i in range(ks.shape[0]):
        for j in range(ks.shape[1]):
            v = ks.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if v > 0.6 * ks.values.max() else 'black')
    fig.colorbar(im, ax=ax, label='KS distance to own data')
    ax.set_title('Wavelet-band mismatch, each group vs its own data')
    fig.tight_layout()
    plt.show()
    return ks


# The runs' two wavelet banks (turbulence/run_SDE.py, codes/filters_bank.return_Filters):
# both Morlet, J = log2(M) octaves, high_freq 0.49, l1 norm.
#   'psi'    = filters_Q, Q = 3 per octave: Scalar_psi_* / L_6_psi. Its first 18 bands are
#              the bank of compute_stats and of hist_plot (codes/check_moments.py).
#   'morlet' = filters,   Q = 1 per octave: Scalar_morlet_* / L_6.
# (The runs also append the low-pass phi; it is not a band and is not shown.)
BANKS = {'psi': 3, 'morlet': 1}


def wavelet_bank(M, bank='psi', J=None):
    """(filters (bands, M) in Fourier space, band labels, centre frequencies) of a bank."""
    Q = BANKS[bank]
    J = int(np.log2(M)) if J is None else J
    psi = np.asarray(init_band_pass('morlet', M, J, Q, high_freq=0.49, wav_norm='l1'))
    labels = [f'j{b // Q}q{b % Q}' for b in range(J * Q)]
    return psi, labels, np.abs(np.fft.fftfreq(M)[np.abs(psi).argmax(-1)])


def _coeffs(x, psi, n_series, rng):
    """Wavelet coefficients (bands, n_series * M) of n_series random series of x."""
    x = np.concatenate([np.asarray(v.numpy() if torch.is_tensor(v) else v, float)
                        for v in (x if isinstance(x, (list, tuple)) else [x])])
    x = x[rng.choice(len(x), min(n_series, len(x)), replace=False)]
    w = np.fft.ifft(np.fft.fft(x)[:, None, :] * psi[None], axis=-1)
    return w.transpose(1, 0, 2).reshape(len(psi), -1)


def plot_band_hists(pairs, bank='psi', kind='real', bands=None, J=None, n_series=2000,
                    bins=100, seed=0, colors=None):
    """hist_plot (codes/check_moments.py) for several (data, generated) pairs side by side:
    one row per wavelet band, one column per pair, data and generated histograms
    overlaid (shared bins), log density, the KS distance in each title.

    pairs: {name: (x_data, x_gen)}, x_gen an array or a list of arrays (pooled, e.g.
    several seeds); n_series random series of each side are used. bank: 'psi' or
    'morlet' (see BANKS). kind: 'real' (Re W, as hist_plot) or 'abs' (|W|, what the
    Scalar_*_gaussianK potentials see). Raw coefficients, not normalised. bands: labels
    like 'j0q1' (default: all bands of the bank). colors: {name: colour of its generated
    histograms} (default: palette order)."""
    M = np.shape(next(iter(pairs.values()))[0])[-1]
    psi, labels, xi = wavelet_bank(M, bank, J)
    idx = list(range(len(psi))) if bands is None else [labels.index(b) for b in bands]
    rng = np.random.default_rng(seed)
    part = np.abs if kind == 'abs' else np.real
    fig, axes = plt.subplots(len(idx), len(pairs), figsize=(6.5 * len(pairs), 2.6 * len(idx)),
                             squeeze=False)
    for col, (name, (xd, xg)) in enumerate(pairs.items()):
        wd = part(_coeffs(xd, psi[idx], n_series, rng))
        wg = part(_coeffs(xg, psi[idx], n_series, rng))
        color = (colors or {}).get(name, PALETTE[col % len(PALETTE)])
        for row, b in enumerate(idx):
            ax = axes[row, col]
            e = np.linspace(min(wd[row].min(), wg[row].min()), max(wd[row].max(), wg[row].max()),
                            bins + 1)
            ax.hist(wd[row], bins=e, density=True, alpha=0.5, color='0.35', label='data')
            ax.hist(wg[row], bins=e, density=True, alpha=0.5, color=color, label='generated')
            ax.set_yscale('log'); ax.grid(alpha=0.2)
            ax.set_title(f'{name}: {bank} {labels[b]}  (f = {xi[b]:.4f})   '
                         f'KS = {ks_distance(wd[row], wg[row]):.3f}', fontsize=9)
            if row == 0:
                ax.legend(fontsize=8, frameon=False)
        axes[-1, col].set_xlabel('Re W' if kind == 'real' else '|W|')
    fig.suptitle(f'{bank} wavelet coefficients ({"Re W" if kind == "real" else "|W|"}), '
                 f'data vs generated, {n_series} series each', y=1.0)
    fig.tight_layout()
    plt.show()


# ================================================================================
# Rare events, series by series
# ================================================================================

def event_scores(x, tau=1, ref=None):
    """Per series: the largest |x(t + tau) - x(t)| (periodic) in units of sigma_tau, and
    where it happens. sigma_tau = rms increment of ref (e.g. the data) if given, else of x,
    so generated and data series can be scored on the same scale."""
    x = x.numpy() if torch.is_tensor(x) else np.asarray(x)
    d = np.abs(np.roll(x, -tau, axis=1) - x)
    r = x if ref is None else (ref.numpy() if torch.is_tensor(ref) else np.asarray(ref))
    sigma = np.sqrt(np.mean((np.roll(r, -tau, axis=1) - r) ** 2))
    return d.max(1) / sigma, d.argmax(1), sigma


def plot_series(x, n=10, idx=None, order='top', tau=1, ref=None, name='', color=PALETTE[0],
                ylim=None, inc_ylim=None, seed=0):
    """One figure per series: the series (top) and its increment
    (x(t + tau) - x(t)) / sigma_tau (bottom), the largest increment marked.

    Which series: idx (explicit indices) or n of them, by order = 'top' (largest events
    first, see event_scores), 'bottom' (quietest first) or 'random'. ref: reference
    series for sigma_tau (pass the data when plotting generated series, so both are on
    the data's scale). ylim / inc_ylim: fixed y-ranges, to compare figures by eye.
    Returns the plotted indices (to plot the same series again, e.g. with another tau)."""
    x = x.numpy() if torch.is_tensor(x) else np.asarray(x)
    score, pos, sigma = event_scores(x, tau, ref)
    if idx is None:
        idx = {'top': np.argsort(-score), 'bottom': np.argsort(score),
               'random': np.random.default_rng(seed).permutation(len(x))}[order][:n]
    t = np.arange(x.shape[1])
    for i in idx:
        d = (np.roll(x[i], -tau) - x[i]) / sigma
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True,
                                     gridspec_kw=dict(height_ratios=[3, 2]))
        a1.plot(t, x[i], color=color, lw=1.2)
        a2.plot(t, d, color=color, lw=1)
        for a in (a1, a2):
            a.axvline(pos[i], color=DATA_COLOR, lw=0.8, ls=':')
            a.grid(alpha=0.2)
        a2.axhline(0, color=DATA_COLOR, lw=0.6)
        a1.set_ylabel('x'); a2.set_ylabel(fr'$\delta_{{{tau}}} x / \sigma_{{{tau}}}$')
        a2.set_xlabel('t')
        if ylim is not None:
            a1.set_ylim(ylim)
        if inc_ylim is not None:
            a2.set_ylim(inc_ylim)
        a1.set_title(f'{name} series {i}:  max |increment| = {score[i]:.1f} sigma at t = {pos[i]}'
                     f'  (rank {int(np.sum(score > score[i])) + 1} of {len(x)})', fontsize=10)
        fig.tight_layout()
        plt.show()
    return np.asarray(idx)


def plot_extremes(x_ref, xs, taus=(1, 4, 16)):
    """How often a series contains an event of a given size: fraction of series whose
    largest |increment| exceeds s sigma_tau (sigma_tau of the DATA for every source), per
    tau. xs: {name: series}. Uses every series, so rare events show up as the tail of
    the curve; compare sources with the same number of series."""
    fig, axes = plt.subplots(1, len(taus), figsize=(4.5 * len(taus), 3.6), squeeze=False)
    cols = _colors(xs)
    for ax, tau in zip(axes[0], taus):
        for name, x, c in [('data', x_ref, DATA_COLOR)] + [(k, v, cols[k]) for k, v in xs.items()]:
            s = np.sort(event_scores(x, tau, ref=x_ref)[0])
            ax.plot(s, 1 - np.arange(len(s)) / len(s), color=c, lw=2 if name == 'data' else 1.4,
                    label=f'{name} ({len(s)} series)')
        ax.set_yscale('log'); ax.grid(alpha=0.2, which='both')
        ax.set_xlabel(fr'largest $|\delta_{{{tau}}} x| / \sigma_{{{tau}}}^{{data}}$ in the series')
        ax.set_title(fr'$\tau = {tau}$')
    axes[0, 0].set_ylabel('fraction of series exceeding')
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    plt.show()
