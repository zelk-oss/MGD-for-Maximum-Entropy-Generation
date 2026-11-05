"""S&P statistics for talk_snp.ipynb: data (ONE series of daily log-returns, d = 6062) against generated series.

Conventions. x = standardized daily log-return X(u). p(u) = cumsum of x = the standardized log-price, so that the
increment over s days is delta_s p(u) = p(u) - p(u - s) = x(u - s + 1) + ... + x(u).
Each statistic is a function  f(series (N, T)) -> (axis (L,), values (N, L))  evaluated per series.
Error bands:
  data   : value on the whole series; band = +- standard error over the BLOCKS (23 blocks of 256 days, i.e. the
           statistic of each block, standard deviation over blocks / sqrt(23)),
  model  : mean over the n_particles samples of each seed, then mean +- std over the seeds.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent
for p in ('codes', 'codes/filters'):
    sys.path.insert(0, str(R / p))
import talk_figures as tf                                       # noqa: E402
from check_moments import signed_leverage_correlation, zumbach_moment   # noqa: E402

BLOCK = 256


def blocks(x, size=BLOCK):
    """(1, T) -> (T // size, size): non-overlapping blocks (the last T % size days are dropped)."""
    x = np.asarray(x).reshape(-1)
    n = len(x) // size
    return x[:n * size].reshape(n, size)


# --------------------------------------------------------------- statistics ----
def stat_cdf(x, n=240, lim=8.0):
    """Both tails of the CDF of X on a common grid: axis = grid of |x| (0..lim), values = [F(-a); 1 - F(a)] stacked
    (length 2n). The notebook splits the two halves."""
    a = np.linspace(0.0, lim, n)
    left = np.stack([(s[:, None] <= -a).mean(0) for s in x])
    right = np.stack([(s[:, None] >= a).mean(0) for s in x])
    return a, np.concatenate([left, right], axis=1)


def stat_increments(x, q=2.0, jmax=8):
    """Ave_u |delta_s p(u)|^q at s = 2^j, j = 0..jmax (s <= T / 4)."""
    p = np.cumsum(x, axis=1)
    js = range(jmax + 1)                                         # common axis; NaN where the series is too short
    val = [np.abs(p[:, 2 ** j:] - p[:, :-2 ** j]) ** q for j in js if 2 ** j <= x.shape[1] // 4]
    val = np.stack([v.mean(1) for v in val], axis=1)
    pad = np.full((x.shape[0], jmax + 1 - val.shape[1]), np.nan)
    return np.array([2 ** j for j in js]), np.concatenate([val, pad], axis=1)


def stat_leverage(x, max_tau=60):
    """Signed leverage L(tau) = <x(t - tau) x(t)^2> / <x^2>^2 (Bouchaud, Matacz, Potters 2001), tau = -(max-1)..(max-1)."""
    out, taus = [], None
    for s in x:
        taus, L, _ = signed_leverage_correlation(np.cumsum(s)[None], scale=1, max_tau=max_tau)
        out.append(L)
    return taus, np.stack(out)


def stat_zumbach(x, scale=5, max_tau=60):
    """Zumbach moment Z(tau) = Ave_t[|delta_s p(t - tau)|^2 Sigma(t)], s = scale days (codes/check_moments.zumbach_moment)."""
    out, taus = [], None
    for s in x:
        taus, Z, _, _, _, _ = zumbach_moment(np.cumsum(s)[None], scale=scale, max_tau=max_tau)
        out.append(Z)
    return taus, np.stack(out)


def stat_rolling_vol(x, window=5, n_bins=60, lo=-3.0, hi=1.5):
    """Histogram of log10 of the rolling volatility sqrt(mean of x^2 over `window` days), density, common bins."""
    edges = np.linspace(lo, hi, n_bins + 1)
    out = []
    for s in x:
        v = np.sqrt(np.convolve(s ** 2, np.ones(window) / window, mode='valid'))
        out.append(np.histogram(np.log10(v), bins=edges, density=True)[0])
    return 0.5 * (edges[1:] + edges[:-1]), np.stack(out)


def stat_spectrum(x, kind='returns', per_octave=2, fmin=0.99 / 6062):
    """Power spectrum averaged in log bins (NaN where a bin holds no frequency, as for short blocks).
    kind = 'returns': |FFT(x)|^2 | 'volatility': |FFT(|x| - mean)|^2 (the long memory of the volatility).
    Frequency in cycles per day."""
    edges = fmin * 2.0 ** (np.arange(0, int(np.ceil(np.log2(0.5 / fmin) * per_octave)) + 1) / per_octave)
    centers = np.sqrt(edges[1:] * edges[:-1])
    out = []
    for s in x:
        v = s if kind == 'returns' else np.abs(s) - np.abs(s).mean()
        v = v - v.mean()
        P = np.abs(np.fft.rfft(v)) ** 2 / len(v)
        f = np.fft.rfftfreq(len(v))[1:]
        P = P[1:]
        out.append([P[(f >= a) & (f < b)].mean() if ((f >= a) & (f < b)).any() else np.nan for a, b in zip(edges[:-1], edges[1:])])
    return centers, np.array(out)


STATS = dict(cdf=stat_cdf, inc1=lambda x: stat_increments(x, 1.0), inc2=lambda x: stat_increments(x, 2.0),
             inc4=lambda x: stat_increments(x, 4.0), leverage=stat_leverage, zumbach=stat_zumbach,
             vol5=stat_rolling_vol, spec_ret=lambda x: stat_spectrum(x, 'returns'),
             spec_vol=lambda x: stat_spectrum(x, 'volatility'))


def compute(x_data, xts, names=None, size=BLOCK):
    """{stat: dict(axis, data (L,), data_err (L,), model (L,), model_err (L,))}.
    x_data: (T,) or (1, T); xts: list over seeds of (n_particles, T)."""
    x_data = np.asarray(x_data, dtype=np.float64).reshape(1, -1)
    out = {}
    for name in names or STATS:
        f = STATS[name]
        ax, v = f(x_data)
        _, vb = f(blocks(x_data, size))
        per_seed = np.stack([np.nanmean(f(np.asarray(s, dtype=np.float64))[1], axis=0) for s in xts])
        with np.errstate(all='ignore'):
            out[name] = dict(axis=ax, data=v[0], data_err=np.nanstd(vb, axis=0, ddof=1) / np.sqrt(np.isfinite(vb).sum(0)),
                             model=np.nanmean(per_seed, axis=0), model_err=np.nanstd(per_seed, axis=0, ddof=1))
        print(f'  {name}: {len(ax)} points, data blocks {len(vb)} x {vb.shape[1]}, model seeds {len(per_seed)}')
    return out


# ------------------------------------------------------------------ plots ----
def _band(ax, x, y, e, color, label, logy=False):
    ax.plot(x, y, color=color, label=label)
    lo, hi = (np.where(y - e > 0, y - e, y / 10), y + e) if logy else (y - e, y + e)
    ax.fill_between(x, lo, hi, color=color, alpha=0.25, lw=0)


def _draw(ax, S, logx=False, logy=False, xlabel='', ylabel='', title=None):
    _band(ax, S['axis'], S['data'], S['data_err'], tf.DATA, 'S&P data', logy)
    _band(ax, S['axis'], S['model'], S['model_err'], tf.MODEL, 'MGD model', logy)
    ax.set_xscale('log' if logx else 'linear')
    ax.set_yscale('log' if logy else 'linear')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def plot_cdf(S, name):
    a, n = S['axis'], len(S['axis'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, sl, ttl in ((axes[0], slice(0, n), 'left tail'), (axes[1], slice(n, 2 * n), 'right tail')):
        T = {k: (v[sl] if np.ndim(v) else v) for k, v in S.items()}
        T['axis'] = a
        _draw(ax, T, logy=True, xlabel=r'$|x|$ (standard deviations)', title=ttl)
        ax.set_ylim(1e-4, 1)
    axes[0].set_ylabel(r'CDF  $P(X \leq -|x|)$' + '\n' + r'(right: $P(X \geq |x|)$)')
    axes[0].legend()
    fig.tight_layout(); tf._save(fig, name)


def plot_increments(Ss, name):
    """Ss = {q: stat}. One panel per order q, log-log in the lag s = 2^j days."""
    fig, axes = plt.subplots(1, len(Ss), figsize=(5.2 * len(Ss), 4.4))
    for ax, (q, S) in zip(np.atleast_1d(axes), Ss.items()):
        _draw(ax, S, logx=True, logy=True, xlabel='lag $s$ (days)', ylabel=rf'$\mathrm{{Ave}}_u\,|\delta_s p(u)|^{{{q:g}}}$',
              title=f'$q = {q:g}$')
    np.atleast_1d(axes)[0].legend()
    fig.tight_layout(); tf._save(fig, name)


def plot_leverage_zumbach(L, Z, name, scale=5):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    _draw(axes[0], L, xlabel=r'$\tau$ (days)', ylabel=r'$L(\tau)$', title='leverage')
    axes[0].axhline(0, color=tf.GREY, lw=0.8); axes[0].axvline(0, color=tf.GREY, lw=0.8)
    _draw(axes[1], Z, xlabel=r'$\tau$ (days)', ylabel=r'$Z(\tau)$', title=f'Zumbach ($s = {scale}$ days)')
    axes[1].axvline(0, color=tf.GREY, lw=0.8)
    axes[0].legend()
    fig.tight_layout(); tf._save(fig, name)


def plot_rolling_vol(S, name, window=5):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    _draw(ax, S, logy=True, xlabel=rf'$\log_{{10}}$ of the {window}-day rolling volatility', ylabel='density')
    ax.set_ylim(1e-3, 3); ax.legend()
    fig.tight_layout(); tf._save(fig, name)


def plot_spectra(Sr, Sv, name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    _draw(axes[0], Sr, logx=True, logy=True, xlabel='frequency (cycles / day)', ylabel='power', title='returns $x$')
    _draw(axes[1], Sv, logx=True, logy=True, xlabel='frequency (cycles / day)', ylabel='power', title=r'volatility $|x|$')
    axes[0].legend()
    fig.tight_layout(); tf._save(fig, name)
