"""
Sanity audit of the Modane jet record (Vit_Modane95.mat): six figures + one
table of numbers. Scales are in samples (dt not known yet, see jet_data.MODANE_META).

    python audit_modane.py [--window_len 1024 --target_len 256 --n1 5000]

Writes to jets/data_audit/results/: fig1_overview.png ... fig6_model_windows.png,
summary.json.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
from jet_data import load_modane_raw, load_modane_windows, select_windows  # noqa: E402

LAGS = 2 ** np.arange(0, 17)  # increment scales, 1 ... 65536 samples


def zero_increment_runs(v):
    """Lengths and start positions of runs where the signal does not move
    (consecutive equal samples): the signature of a stuck probe."""
    flat = np.diff(v) == 0
    edges = np.diff(np.r_[0, flat.astype(np.int8), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return stops - starts, starts


def block_stats(v, block=100_000):
    nb = v.size // block
    b = v[:nb * block].reshape(nb, block)
    return np.arange(nb) * block + block // 2, b.mean(1), b.std(1)


def autocorr(v, max_lag):
    x = v - v.mean()
    nfft = 1 << int(np.ceil(np.log2(2 * x.size)))
    f = np.fft.rfft(x, nfft)
    c = np.fft.irfft(f * np.conj(f), nfft)[:max_lag + 1]
    return c / c[0]


def integral_scale(rho, floor=0.05):
    """Integral of the autocorrelation up to where it first drops below
    `floor`. The Modane rho never crosses zero: it settles on a ~0.03 plateau
    (slow drift of the mean flow), which would otherwise dominate the integral."""
    below = np.flatnonzero(rho < floor)
    stop = below[0] if below.size else rho.size
    return float(rho[:stop].sum()), int(stop)


def increment_stats(v):
    F, S = [], []
    for l in LAGS:
        d = v[l:] - v[:-l]
        d = d - d.mean()
        m2 = np.mean(d ** 2)
        F.append(np.mean(d ** 4) / m2 ** 2)
        S.append(np.mean(d ** 3) / m2 ** 1.5)
    return np.array(F), np.array(S)


# ── figures ──────────────────────────────────────────────────────────────────
def fig_overview(v, run_len, run_start):
    pos, m, s = block_stats(v)
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    ax[0].fill_between(pos, m - s, m + s, alpha=0.3, label='mean ± std (100k blocks)')
    ax[0].plot(pos, m, lw=1)
    long = run_len >= 3
    ax[0].plot(run_start[long], np.full(long.sum(), v.min()), '|', color='C3', ms=12,
               label=f'signal frozen ≥ 3 samples ({long.sum()} places)')
    ax[0].set(xlabel='sample', ylabel='v', title='Whole record: is it stationary?')
    ax[0].legend(loc='upper right')
    i0 = v.size // 2
    ax[1].plot(np.arange(i0, i0 + 20_000), v[i0:i0 + 20_000], lw=0.5)
    ax[1].set(xlabel='sample', ylabel='v', title='Zoom: 20k samples mid-record')
    return fig


def fig_pdf(v):
    z = (v - v.mean()) / v.std()
    h, e = np.histogram(z, bins=300, density=True)
    c = (e[1:] + e[:-1]) / 2
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.semilogy(c, h, label='data')
    ax.semilogy(c, stats.norm.pdf(c), 'k--', lw=1, label='Gaussian')
    ax.set(xlabel='(v − U) / σ', ylabel='PDF', title='One-point velocity PDF', ylim=(1e-7, 1))
    ax.legend()
    return fig


def fig_spectrum(v):
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    for k, part in enumerate(np.array_split(v, 3)):
        f, P = signal.welch(part - part.mean(), nperseg=2 ** 16)
        ax.loglog(f[1:], P[1:], lw=0.8, label=f'third {k + 1}')
    f_ref = np.array([1e-3, 3e-2])
    ax.loglog(f_ref, 2 * P[np.searchsorted(f, 1e-3)] * (f_ref / 1e-3) ** (-5 / 3), 'k--', label='−5/3')
    ax.set(xlabel='frequency [1/sample]', ylabel='E(f)',
           title='Energy spectrum (three thirds of the record)')
    ax.legend()
    return fig


def fig_autocorr(rho, L, window_len):
    lags = np.arange(rho.size)
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    ax.semilogx(lags[1:], rho[1:])
    ax.axvline(L, color='C1', ls='--', label=f'integral scale L = {L:.0f}')
    ax.axvline(window_len, color='C2', ls=':', label=f'window = {window_len}')
    ax.axhline(0, color='k', lw=0.5)
    ax.set(xlabel='lag [samples]', ylabel='ρ(lag)', title='Velocity autocorrelation')
    ax.legend()
    return fig


def fig_increments(v, F):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for k, l in enumerate([1, 16, 256, 4096]):
        d = v[l:] - v[:-l]
        d = (d - d.mean()) / d.std()
        h, e = np.histogram(d, bins=200, density=True)
        c = (e[1:] + e[:-1]) / 2
        ax[0].semilogy(c, h * 10.0 ** (-2 * k), label=f'ℓ = {l}')
        ax[0].semilogy(c, stats.norm.pdf(c) * 10.0 ** (-2 * k), 'k--', lw=0.6)
    ax[0].set(xlabel='δv / std', ylabel='PDF (shifted)', ylim=(1e-12, 1),
              title='Increment PDFs (dashed: Gaussian)')
    ax[0].legend()
    ax[1].semilogx(LAGS, F, 'o-')
    ax[1].axhline(3, color='k', ls='--', lw=0.8, label='Gaussian (3)')
    ax[1].set(xlabel='ℓ [samples]', ylabel='flatness ⟨δv⁴⟩/⟨δv²⟩²',
              title='Flatness vs scale (intermittency)')
    ax[1].legend()
    return fig


def fig_model_windows(window_len, target_len, n1, spread=True):
    x, _ = load_modane_windows(window_len, target_len, normalized=False)
    x = x.squeeze(1).numpy()
    is_train = np.zeros(len(x), bool)
    is_train[select_windows(len(x), n1, spread)] = True
    x = (x - x[is_train].mean()) / x[is_train].std()  # same scaling as x1
    train, rest = x[is_train], x[~is_train]
    which = f'random {n1}' if spread else f'first {n1}'
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for k in range(4):
        ax[0].plot(train[k * (n1 // 4)] + 4 * k, lw=0.8)
    ax[0].set(xlabel='sample in window', yticks=[],
              title=f'4 training windows ({window_len} → {target_len})')
    bins = np.linspace(-6, 6, 200)
    ax[1].hist(train.ravel(), bins, density=True, histtype='step', label=f'train ({which})')
    ax[1].hist(rest.ravel(), bins, density=True, histtype='step', label=f'rest ({len(rest)})')
    ax[1].set(yscale='log', xlabel='value', title='Pooled values: train vs rest of record')
    ax[1].legend()
    return fig, (len(x), float(rest.mean()), float(rest.std()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--window_len', type=int, default=1024)
    p.add_argument('--target_len', type=int, default=256)
    p.add_argument('--n1', type=int, default=5000)
    args = p.parse_args()
    out = HERE / 'results'
    out.mkdir(exist_ok=True)

    v = load_modane_raw()
    run_len, run_start = zero_increment_runs(v)
    rho = autocorr(v, max_lag=2 ** 17)
    L, first_zero = integral_scale(rho)
    F, S = increment_stats(v)

    def save(fig, name):
        fig.savefig(out / name, dpi=110)
        plt.close(fig)

    save(fig_overview(v, run_len, run_start), 'fig1_overview.png')
    save(fig_pdf(v), 'fig2_pdf.png')
    save(fig_spectrum(v), 'fig3_spectrum.png')
    save(fig_autocorr(rho, L, args.window_len), 'fig4_autocorr.png')
    save(fig_increments(v, F), 'fig5_increments.png')
    fig, (n_win, rest_mean, rest_std) = fig_model_windows(args.window_len, args.target_len, args.n1)
    save(fig, 'fig6_model_windows.png')

    _, bm, bs = block_stats(v)
    summary = {
        'N': int(v.size),
        'nan_or_inf': int((~np.isfinite(v)).sum()),
        'U_mean': float(v.mean()),
        'sigma': float(v.std()),
        'sigma_over_U': float(v.std() / v.mean()),
        'min': float(v.min()), 'max': float(v.max()),
        'n_samples_at_min': int((v == v.min()).sum()), 'n_samples_at_max': int((v == v.max()).sum()),
        'frac_zero_increments': float(np.mean(np.diff(v) == 0)),
        'longest_frozen_run': int(run_len.max() + 1) if run_len.size else 1,
        'block_mean_spread_over_sigma': float(bm.std() / v.std()),
        'block_std_spread_over_sigma': float(bs.std() / v.std()),
        'skewness_v': float(stats.skew(v)),
        'flatness_v': float(stats.kurtosis(v, fisher=False)),
        'integral_scale_L': L,
        'autocorr_below_0.05_at': first_zero,
        'autocorr_plateau': float(rho[-10_000:].mean()),
        'n_jumps_dv1_over_15sd': int(np.sum(np.abs(np.diff(v)) > 15 * np.diff(v).std())),
        'window_len': args.window_len,
        'n_windows': n_win,
        # windows closer than ~2L are not independent
        'n_eff_windows': int(v.size // max(args.window_len, 2 * L)),
        'n1': args.n1,
        'rest_vs_train_mean': rest_mean, 'rest_vs_train_std': rest_std,
        'flatness_vs_lag': dict(zip(map(int, LAGS), map(float, F))),
        'skewness_vs_lag': dict(zip(map(int, LAGS), map(float, S))),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    for k, val in summary.items():
        if not isinstance(val, dict):
            print(f'{k:32s} {val}')


if __name__ == '__main__':
    matplotlib.use('Agg')
    main()
