"""Plan items (a) and (c) that the run_comparison functions do not cover. Same layout for every model.

    moment_table        imposed moments at t = 1, per family: relative error |<phi>_samples - m_target| / |m_target|
    fixed_stats_table   out-of-model statistics (Scat2 and L_6 per band), model / data
    modulus_table       marginals of the complex modulus |x * psi| per band (not imposed by any scalar potential)
    k0_summary, plot_k0 the per-sample mean (k = 0 mode)
    plot_l1l2           l1 / l2 of the coefficients per scale, on Re(x * psi) and on |x * psi|
    plot_gallery        model samples at quantiles of -log p, each next to the data signal of closest -log p
Samples are (B, M) arrays or tensors, as run_comparison uses them; the Q=1 / Q=3 banks are its BANKS.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / 'codes', HERE.parent / 'codes/filters', HERE.parent / 'codes/potentials'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from filters.filters_1d import init_band_pass                      # noqa: E402

Q_OF = {'morlet': 1, 'psi': 3}
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']


def _np(x):
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return x.reshape(x.shape[0], -1).astype(float)


def _bank(M, bank, J):
    """(J*Q, M) Fourier filters of the Morlet bank, as run_comparison.wavelet_bank."""
    return np.asarray(init_band_pass('morlet', M, J, Q_OF[bank], high_freq=0.49, wav_norm='l1'))


def _coef(x, psi, n, seed):
    """Complex coefficients (n, bands, M) of n random series of x (B, M)."""
    x = _np(x)
    x = x[np.random.default_rng(seed).choice(len(x), min(n, len(x)), replace=False)]
    return np.fft.ifft(np.fft.fft(x, axis=-1)[:, None, :] * psi[None], axis=-1)


# ---------------------------------------------------------------------------------------------- (a) moments
def moment_table(res, tol=0.1):
    """Per model and family, mean over seeds of: median / max relative error of the moments at the end of the run
    (the walkers' <phi> against the data moments m_target the SDE was driven to), and how many exceed tol."""
    rows = []
    for name, runs in res.items():
        if not runs:
            continue
        for F, (a, b) in zip(runs[0]['families'], runs[0]['fam_ranges']):
            med, mx, bad = [], [], []
            for r in runs:
                t, e = np.abs(r['m_target'][a:b]), np.abs(r['m_part'][a:b] - r['m_target'][a:b])
                rel = e[t > 1e-8] / t[t > 1e-8]
                med.append(np.median(rel)); mx.append(rel.max()); bad.append((rel > tol).sum())
            rows.append({'model': name, 'family': str(F), 'N': int(b - a), 'median': np.mean(med), 'max': np.mean(mx),
                         f'# > {tol:.0%}': f'{np.mean(bad):.1f}'})
    return pd.DataFrame(rows).set_index(['model', 'family'])


def fixed_stats_table(x_data, xs, J=8):
    """Out-of-model check: Scat2 (mean |x * psi|^2) and L_6 (mean |x * psi|^6) per band of the Q=1 bank (phi last),
    generated / data. M1 imposes the first, M2 and M3 the second; the other one is a pure check."""
    from filters_bank import return_Filters
    from potentials_builder import get_1d_potentials
    from logp_entropy import phi_matrix
    M = x_data.shape[-1]
    filters, fphi = return_Filters(M, J, 1, include_phi=True)
    pots = get_1d_potentials(['Scattering_Second_Order_Q1', 'L_6'], J, filters, 1, filters_Q=filters, filters_Phi=fphi,
                             scalar_param=None, parallel=False)
    from codes.gram_preflight import stat_labels            # column order of the potentials dict (L_6 comes first)
    names = [l.replace('Scattering_Second_Order_Q1', 'Scat2') for _, _, l in stat_labels(pots)]
    mean_phi = lambda x: phi_matrix(pots, torch.as_tensor(_np(x), dtype=torch.float32).reshape(-1, 1, M)).mean(0)
    ref = mean_phi(x_data)
    return pd.DataFrame({k: mean_phi(v) / ref for k, v in xs.items()}, index=names)


# ---------------------------------------------------------------------------------------------- marginals
def modulus_table(x_data, xs, bank='morlet', J=8, n_series=2000, seed=0, moments=(2, 4, 6), quantiles=(0.99, 0.999, 0.9999)):
    """Generated / data ratios of the marginal of the complex modulus |x * psi| per band: m_p = E|W|^p and quantiles
    of |W| (1 = perfect). Row j{j}q{q}, j0 = finest. Columns: (metric, model)."""
    M, Q = x_data.shape[-1], Q_OF[bank]
    psi = _bank(M, bank, J)
    flat = lambda x: np.abs(_coef(x, psi, n_series, seed)).transpose(1, 0, 2).reshape(psi.shape[0], -1)
    ref = flat(x_data)
    cols = {}
    for name, x in xs.items():
        g = flat(x)
        for p in moments:
            cols[(f'm{p}', name)] = (g ** p).mean(1) / (ref ** p).mean(1)
        for q in quantiles:
            cols[(f'q{q:g}', name)] = np.quantile(g, q, axis=1) / np.quantile(ref, q, axis=1)
    df = pd.DataFrame(cols, index=[f'j{b // Q}q{b % Q}' for b in range(psi.shape[0])])
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def k0_summary(x_data, xs):
    """Per-sample mean (the k = 0 mode up to sqrt(M)): spread and tails of its marginal, data first."""
    from scipy import stats
    rows = {}
    for name, x in {'data': x_data, **xs}.items():
        m = _np(x).mean(1)
        rows[name] = {'std': m.std(), 'excess kurtosis': stats.kurtosis(m), 'q0.001': np.quantile(m, 1e-3),
                      'q0.999': np.quantile(m, 1 - 1e-3), 'std / data': m.std() / _np(x_data).mean(1).std()}
    return pd.DataFrame(rows).T


def plot_k0(x_data, xs, bins=80):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    m0 = _np(x_data).mean(1)
    edges = np.linspace(*np.percentile(m0, [0.05, 99.95]), bins)
    ax.hist(m0, bins=edges, density=True, histtype='step', color='k', lw=1.6, log=True, label='data')
    for (name, x), c in zip(xs.items(), COLORS):
        ax.hist(np.clip(_np(x).mean(1), edges[0], edges[-1]), bins=edges, density=True, histtype='step', color=c, log=True, label=name)
    ax.set_xlabel('per-sample mean (k = 0 mode / sqrt(M))')
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def plot_l1l2(x_data, xs, bank='morlet', J=8, n_series=2000, seed=0):
    """l1 / l2 per scale, ||c||_1 / (sqrt(M) ||c||_2) over the M positions of one series (Gaussian Re: sqrt(2/pi) = 0.80).
    Mean over series +- std; lower = sparser (more intermittent)."""
    M = x_data.shape[-1]
    psi = _bank(M, bank, J)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for (name, x), c in zip({'data': x_data, **xs}.items(), ['k'] + COLORS):
        w = _coef(x, psi, n_series, seed)
        for ax, a in zip(axes, (w.real, np.abs(w))):
            r = np.abs(a).sum(-1) / (np.sqrt(M) * np.sqrt((a ** 2).sum(-1)))
            ax.errorbar(np.arange(psi.shape[0]), r.mean(0), r.std(0), color=c, marker='o', ms=3, lw=1.2, capsize=2, label=name)
    axes[0].axhline(np.sqrt(2 / np.pi), ls='--', color='gray', lw=1, label='Gaussian')
    for ax, t in zip(axes, ('Re(x * psi)', '|x * psi|')):
        ax.set_title(t, fontsize=9); ax.set_xlabel('band (0 = finest)')
    axes[0].set_ylabel('l1 / l2'); axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- (c) gallery
def plot_gallery(res, xs, x_data, seed=900, quantiles=(0, 1e-3, 1e-2, 0.5, 0.99, 0.999, 1), ylim=None):
    """For each model (rows 1 and 2), its samples at quantiles of -log p (0 = most likely, 1 = rarest) and below each
    the data signal whose -log p is closest. res: logp_entropy.load_all; xs: {model: samples of that seed (B, M)}."""
    names = [n for n in xs if any(r['seed'] == seed for r in res.get(n, []))]
    fig, axes = plt.subplots(2 * len(names), len(quantiles), figsize=(2.3 * len(quantiles), 1.5 * 2 * len(names)),
                             squeeze=False, sharey='row')
    X = _np(x_data)
    for i, n in enumerate(names):
        r = [r for r in res[n] if r['seed'] == seed][0]
        xm, nm, nd = _np(xs[n]), np.asarray(r['nlp_model']), np.asarray(r['nlp_data'])
        order = np.argsort(nm)
        for j, q in enumerate(quantiles):
            k = order[int(round(q * (len(nm) - 1)))]
            kd = int(np.argmin(np.abs(nd - nm[k])))
            axes[2 * i][j].plot(xm[k], lw=.7, color='C0')
            axes[2 * i + 1][j].plot(X[kd], lw=.7, color='k')
            axes[2 * i][j].set_title(f'{n}\nq={q:g}  -log p {nm[k]:.0f}' if j == 0 else f'q={q:g}  {nm[k]:.0f}', fontsize=7)
            axes[2 * i + 1][j].set_title(f'data nearest: {nd[kd]:.0f}', fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.tick_params(labelsize=6)
        if ylim:
            ax.set_ylim(*ylim)
    fig.tight_layout()
    return fig
