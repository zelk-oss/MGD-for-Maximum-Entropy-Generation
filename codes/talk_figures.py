"""Figures for the conference talk (turbulence/talk_turbulence.ipynb, jets/talk_jets.ipynb).

One style for every figure: data in black, model in blue, rare signals in red. Each plot_* function makes one
figure, saves it as PNG (300 dpi) in the folder given to set_output() and shows it in the notebook.
"""
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

DATA, MODEL, RARE, GREY = '#0b0b0b', '#2a78d6', '#e34948', '#a3a29b'
_OUT = None

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 14, 'legend.fontsize': 12,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'lines.linewidth': 2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
    'legend.frameon': False, 'mathtext.fontset': 'dejavusans',
})


def set_output(folder):
    """Folder where every figure is saved as <name>.png."""
    global _OUT
    _OUT = Path(folder)
    _OUT.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    """Save as PNG, then show it in the notebook as an image: works with any matplotlib backend (Agg included, which
    other modules may have selected at import) and never twice (the figure is closed)."""
    if _OUT is not None:
        fig.savefig(_OUT / f'{name}.png')
    try:
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is not None:
            display(fig)
    finally:
        plt.close(fig)


def _np(x):
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return x.reshape(x.shape[0], -1).astype(np.float64)


# ------------------------------------------------------------------ names ----
_LABELS = [(r'^L_6$', 'L⁶ norm'),
           (r'^(Scalar_|S_)psi', 'Marginals (Morlet, Q = {Q})'), (r'^(Scalar_|S_)morlet', 'Marginals (Morlet, Q = 1)'),
           (r'(First_Order|^Scat1$)', 'Scattering 1'), (r'(Second_Order|^Scat2$)', 'Scattering 2'),
           (r'(Third_Order_Real|^Scat3_Real)', 'Scattering 3 (Re)'),
           (r'(Third_Order_Imag|^Scat3_Imag)', 'Scattering 3 (Im)'),
           (r'(Fourth_Order.*Real|^Scat4_Real)', 'Scattering 4 (Re)'),
           (r'(Fourth_Order.*Imag|^Scat4_Imag)', 'Scattering 4 (Im)')]


def family_label(name, Q=None):
    """Readable name of a potential family (full or short name); unknown names are returned unchanged.
    Q = the run's Q (bank of Scalar_psi_*, the filters_Q channels)."""
    for pat, lab in _LABELS:
        if re.search(pat, name):
            return lab.format(Q=Q)
    return name


def count_statistics(pots, x, theta, Q=None):
    """Table per family: number of statistics (width of the potential's output on a few signals x, as in the run,
    after deduplication and pruning) and how many have Theta(t = 1) != 0 (the live mask pins dead statistics to 0).
    The widths must add up to len(theta), in the column order of the run."""
    import pandas as pd
    rows, a = [], 0
    with torch.no_grad():
        for name, p in pots.items():
            f = p(x[:4])
            w = 1 if f.ndim == 1 else int(f.shape[-1])
            rows.append(dict(family=family_label(name, Q), statistics=w, nonzero_theta=int(np.count_nonzero(theta[a:a + w]))))
            a += w
    assert a == len(theta), f'potential widths add up to {a}, Theta has {len(theta)} entries'
    T = pd.DataFrame(rows).set_index('family')
    T.loc['total'] = T.sum()
    return T.rename(columns={'nonzero_theta': 'with Θ ≠ 0 at t = 1'})


# ---------------------------------------------------------------- signals ----
def plot_signals(signals, name, color=DATA, row_labels=None, labels_header=None, title=None,
                 xlabel='time (samples)'):
    """One row per signal, shared axes; row_labels (with a header above them) on the right."""
    s = _np(signals)
    n = len(s)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.5 * n + 0.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lim = 1.05 * np.abs(s).max()
    for i, ax in enumerate(axes):
        ax.plot(s[i], color=color, lw=1.1)
        ax.set_ylim(-lim, lim)
        ax.grid(False)
        ax.tick_params(labelsize=9)
        if i < n - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', length=0)
        if row_labels is not None:
            ax.text(1.01, 0.5, row_labels[i], transform=ax.transAxes, va='center', fontsize=10)
    if labels_header:
        axes[0].text(1.01, 1.25, labels_header, transform=axes[0].transAxes, va='bottom', fontsize=10)
    axes[-1].set_xlabel(xlabel)
    axes[n // 2].set_ylabel('signal / std')
    if title:
        axes[0].set_title(title)
    fig.subplots_adjust(hspace=0.12)
    _save(fig, name)


# ------------------------------------------------------- wavelet marginals ----
def plot_wavelet_histograms(x, J, name, title=None):
    """Histogram of the real part of the Morlet coefficients (Q = 1), one panel per scale j, coefficients divided
    by their std; dashed: standard Gaussian."""
    from filters_bank import return_Filters
    x = torch.as_tensor(_np(x))
    M = x.shape[-1]
    filters = return_Filters(M, J, 1, device='cpu').reshape(-1, M).to(torch.complex128)    # J wavelets + low-pass
    assert filters.shape[0] == J + 1, f'{filters.shape[0]} filters for J = {J}'
    filters = filters[:J]                                                                   # wavelets only
    z = torch.fft.ifft(torch.fft.fft(x)[:, None, :] * filters[None], dim=-1).real.numpy()   # (n, J, M)
    ncol = 4
    nrow = int(np.ceil(J / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), sharey=True)    # own x span per scale
    for j, ax in enumerate(axes.flat):
        if j >= J:
            ax.axis('off')
            continue
        u = z[:, j].ravel()
        u = (u - u.mean()) / u.std()
        lim = 1.05 * np.abs(u).max()
        h, e = np.histogram(u, bins=161, density=True)
        c = 0.5 * (e[1:] + e[:-1])
        ax.plot(c[h > 0], h[h > 0], color=DATA, lw=1.6)
        g = np.linspace(-lim, lim, 400)
        ax.plot(g, stats.norm.pdf(g), color=GREY, ls='--', lw=1.4, label='Gaussian' if j == 0 else None)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(1e-6, 1)
        ax.set_yscale('log')
        ax.set_title(f'j = {j}' + (' (finest)' if j == 0 else ' (coarsest)' if j == J - 1 else ''), fontsize=12)
    axes.flat[0].legend(loc='upper right', fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel('coefficient / std')
    for ax in axes[:, 0]:
        ax.set_ylabel('density')
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    _save(fig, name)


# ---------------------------------------------------- Li et al. statistics ----
def _li(data, model, n_batches, max_lag):
    from paper_figures_li import structure_functions
    taus, S_d, Sb_d = structure_functions(data, n_batches, max_lag)
    _, S_m, Sb_m = structure_functions(model, n_batches, max_lag)
    return taus, (S_d, Sb_d), (S_m, Sb_m)


def _curve(ax, t, y, yb, color, label):
    ax.fill_between(t, yb.min(0), yb.max(0), color=color, alpha=0.12, lw=0)
    ax.plot(t, y, color=color, label=label)


def plot_increment_pdfs(data, model, tau_eta, name, tau_ratios=(1, 2, 5), shift_decades=3, n_bins=161):
    """Standardized PDFs of the increments x(t+tau) - x(t) at tau / tau_eta in tau_ratios, shifted down by
    10^(-shift_decades) per lag for readability."""
    d, m = _np(data), _np(model)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for k, r in enumerate(tau_ratios):
        tau = max(1, int(round(r * tau_eta)))
        if tau > d.shape[-1] - 2:
            print(f'tau/tau_eta = {r} skipped: longer than the signal')
            continue
        incs = [(v[:, tau:] - v[:, :-tau]).ravel() for v in (d, m)]
        incs = [v / v.std() for v in incs]
        lim = max(np.abs(v).max() for v in incs)
        bins = np.linspace(-lim, lim, n_bins)
        c = 0.5 * (bins[1:] + bins[:-1])
        shift = 10.0 ** (-shift_decades * k)
        for v, col, lab in zip(incs, (DATA, MODEL), ('data', 'model')):
            h, _ = np.histogram(v, bins, density=True)
            ax.plot(c[h > 0], h[h > 0] * shift, color=col, lw=1.6, label=lab if k == 0 else None)
        ax.text(0.98, 0.4 * shift, rf'$\tau/\tau_\eta = {r}$', transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=11)
    ax.set_yscale('log')
    ax.set_xlabel(r'$\delta_\tau x\, /\, \sigma(\delta_\tau x)$')
    ax.set_ylabel('PDF (shifted)')
    ax.legend(loc='upper left')
    fig.tight_layout()
    _save(fig, name)


def plot_increment_trajectories(data, model, tau_eta, name, tau_ratios=(20, 5, 1), n_windows=8, seed=None, gap=4):
    """As Li et al. 2024 Fig. 1b-d: the signal V(t) and its increments delta_tau V(t) = V(t+tau) - V(t) at
    tau / tau_eta in tau_ratios, against t / tau_eta, data (left) and model (right). n_windows random signals are
    laid end to end, separated by a break of `gap` samples (no increment spans two signals). Each row is divided by
    the std of the DATA increments at that tau, in both columns. seed=None: new signals at every call."""
    d, m = _np(data), _np(model)
    T = d.shape[-1]
    seed = np.random.SeedSequence().entropy % 10**6 if seed is None else seed
    rng = np.random.default_rng(seed)
    picks = {'data': d[rng.choice(len(d), n_windows, replace=False)],
             'model': m[rng.choice(len(m), n_windows, replace=False)]}
    taus = [max(1, int(round(r * tau_eta))) for r in tau_ratios]
    rows = [('$V$', 0)] + [(rf'$\delta_\tau V$' + '\n' + rf'$\tau/\tau_\eta = {r}$', t) for r, t in zip(tau_ratios, taus)]
    fig, axes = plt.subplots(len(rows), 2, figsize=(16, 1.9 * len(rows) + 0.8), sharex=True, sharey='row',
                             gridspec_kw=dict(hspace=0.12, wspace=0.04))
    for r, (lab, tau) in enumerate(rows):
        sig = d.std() if tau == 0 else (d[:, tau:] - d[:, :-tau]).std()        # data scale, both columns
        for c, (key, col) in enumerate((('data', DATA), ('model', MODEL))):
            ax = axes[r, c]
            for w, x in enumerate(picks[key]):
                y = x if tau == 0 else x[tau:] - x[:-tau]
                t0 = w * (T + gap)
                ax.plot((t0 + np.arange(len(y))) / tau_eta, y / sig, color=col, lw=0.8)
            ax.grid(False)
            ax.tick_params(labelsize=10)
            if r < len(rows) - 1:
                ax.spines['bottom'].set_visible(False)
                ax.tick_params(axis='x', length=0)
        axes[r, 0].set_ylabel(lab, fontsize=12)
    axes[0, 0].set_title('data')
    axes[0, 1].set_title('model')
    for ax in axes[-1]:
        ax.set_xlabel(r'$t / \tau_\eta$')
    print(f'seed {seed}: {n_windows} signals of {T / tau_eta:.0f} tau_eta each, laid end to end')
    _save(fig, name)


def plot_structure_functions(data, model, tau_eta, max_lag, name, orders=(2, 4, 6, 8), n_batches=10):
    """S_p(tau) = <(delta_tau x)^p>; bands = min / max over n_batches disjoint batches of signals."""
    from paper_figures_li import ORDERS
    taus, (S_d, Sb_d), (S_m, Sb_m) = _li(data, model, n_batches, max_lag)
    t = taus / tau_eta
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for k, p in enumerate(orders):
        j = ORDERS.index(p)
        _curve(ax, t, S_d[j], Sb_d[:, j], DATA, 'data' if k == 0 else None)
        _curve(ax, t, S_m[j], Sb_m[:, j], MODEL, 'model' if k == 0 else None)
        ax.text(t[-1] * 1.08, S_d[j][-1], f'p = {p}', va='center', fontsize=11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\tau / \tau_\eta$')
    ax.set_ylabel(r'$S_p(\tau)$')
    ax.legend(loc='upper left')
    fig.tight_layout()
    _save(fig, name)


def plot_flatness(data, model, tau_eta, max_lag, name, orders=(4, 6, 8), n_batches=10):
    """F_p(tau) = S_p / S_2^(p/2); bands as plot_structure_functions."""
    from paper_figures_li import _flat
    taus, (S_d, Sb_d), (S_m, Sb_m) = _li(data, model, n_batches, max_lag)
    t = taus / tau_eta
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for k, p in enumerate(orders):
        _curve(ax, t, _flat(S_d, p), _flat(Sb_d, p), DATA, 'data' if k == 0 else None)
        _curve(ax, t, _flat(S_m, p), _flat(Sb_m, p), MODEL, 'model' if k == 0 else None)
        ax.text(t[-1] * 1.08, _flat(S_d, p)[-1], f'p = {p}', va='center', fontsize=11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\tau / \tau_\eta$')
    ax.set_ylabel(r'$F_p(\tau) = S_p / S_2^{\,p/2}$')
    ax.legend(loc='upper right')
    fig.tight_layout()
    _save(fig, name)


def plot_local_exponents(data, model, tau_eta, max_lag, name, orders=(4, 6, 8), n_batches=10):
    """zeta_p(tau) = dlog S_p / dlog S_2, one panel per order; dotted: p/2 (no intermittency)."""
    from paper_figures_li import _zeta
    taus, (S_d, Sb_d), (S_m, Sb_m) = _li(data, model, n_batches, max_lag)
    t = taus / tau_eta
    fig, axes = plt.subplots(1, len(orders), figsize=(5 * len(orders), 4.5))
    for ax, p in zip(np.atleast_1d(axes), orders):
        for S, Sb, col, lab in ((S_d, Sb_d, DATA, 'data'), (S_m, Sb_m, MODEL, 'model')):
            _curve(ax, t, _zeta(S, taus, p), np.stack([_zeta(s, taus, p) for s in Sb]), col, lab)
        ax.axhline(p / 2, color=GREY, ls=':', lw=1.6)
        ax.set_ylim(p / 4, 1.1 * p / 2)
        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau / \tau_\eta$')
        ax.set_title(rf'$\zeta_{p}(\tau)$')
    np.atleast_1d(axes)[0].legend(loc='lower right')
    fig.tight_layout()
    _save(fig, name)


# -------------------------------------------------------------------- log p ----
def plot_logp_histogram(logp, name, rare=None, gumbel=False, bins=100):
    """Histogram of log p over signals, shifted by its median (log p is known up to a constant); log y.
    rare (bool mask): those signals in red, drawn in front of the light histogram of all signals (both
    normalised by the total number of signals, so the red area = fraction of rare signals). gumbel: fit of a Gumbel distribution for the
    minimum (scipy gumbel_l, left tail) to all signals."""
    v = np.asarray(logp, dtype=np.float64)
    v = v - np.median(v)
    edges = np.linspace(v.min(), v.max(), bins + 1)
    w = np.full(len(v), 1.0 / (len(v) * (edges[1] - edges[0])))                # density of all signals
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(v, edges, weights=w, color=GREY, alpha=0.45, label=f'all signals ({len(v)})')
    if rare is not None:
        rare = np.asarray(rare, bool)
        ax.hist(v[~rare], edges, weights=w[~rare], histtype='step', color=DATA, lw=1.4,
                label=f'other signals ({(~rare).sum()})', zorder=4)
        ax.hist(v[rare], edges, weights=w[rare], color=RARE, label=f'rare signals ({rare.sum()})', zorder=3)
    if gumbel:
        loc, scale = stats.gumbel_l.fit(v)
        g = np.linspace(v.min(), v.max(), 400)
        ax.plot(g, stats.gumbel_l.pdf(g, loc, scale), color=DATA, lw=1.8, ls='--', label='Gumbel fit (minimum)')
    ax.set_yscale('log')
    ax.set_ylim(0.5 * w[0], None)
    ax.set_xlabel(r'$\log p(x)$ $-$ median (nats)')
    ax.set_ylabel('density')
    ax.legend(loc='upper left')
    fig.tight_layout()
    _save(fig, name)


def ranked_indices(logp, n=10):
    """Indices of n signals at evenly spaced quantiles of log p, from the lowest (rarest) to the highest."""
    order = np.argsort(logp)
    return order[np.round(np.linspace(0, len(order) - 1, n)).astype(int)]


def plot_ranked_signals(signals, logp, name, n=10, ncol=5):
    """Left: histogram of log p over all the signals (relative to its median), with an arrow at the log p of each
    of n signals: for n evenly spaced values from the lowest (rarest) to the highest log p, the signal closest to
    it. Right: those signals, framed with the colour of their arrow, log p on top (same y-scale everywhere)."""
    from matplotlib import gridspec
    v = np.asarray(logp, dtype=np.float64)
    v = v - np.median(v)
    idx = np.array([np.argmin(np.abs(v - t)) for t in np.linspace(v.min(), v.max(), n)])
    s = _np(signals)[idx]
    colors = plt.get_cmap('plasma')(np.linspace(0.0, 0.85, n))       # dark (rare) -> light (typical), no pale end
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(17, 2.4 * nrow + 1.2))
    gs = gridspec.GridSpec(nrow, ncol + 2, figure=fig, width_ratios=[2.4, 0.35] + [1] * ncol, wspace=0.08, hspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    ax.hist(v, 80, color=GREY, alpha=0.6)
    ax.set_yscale('log')
    for c, x in zip(colors, v[idx]):
        ax.annotate('', xy=(x, 0), xytext=(x, -0.08), xycoords=ax.get_xaxis_transform(),
                    arrowprops=dict(arrowstyle='->', color=c, lw=2), annotation_clip=False)
    ax.tick_params(axis='x', pad=30)                                   # tick labels below the arrows
    ax.set_xlabel(r'$\log p(x)$ $-$ median (nats)')
    ax.set_ylabel('number of signals')
    lim = 1.05 * np.abs(s).max()
    for k in range(n):
        a = fig.add_subplot(gs[k // ncol, 2 + k % ncol])
        a.plot(s[k], color=colors[k], lw=1.2)
        a.set_ylim(-lim, lim)
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        for sp in a.spines.values():
            sp.set_visible(True)
            sp.set_color(colors[k])
            sp.set_linewidth(3)
        a.set_title(f'{v[idx[k]]:+.0f}', fontsize=11)
    _save(fig, name)


def plot_energy_bars(C, name, title=None, Q=None):
    """C = {family: array (seeds, signals)} of theta_F . phi_F(x), TOTAL last. Three panels: mean over signals
    (level only: a constant added to phi_F changes it without changing p), std across signals, std / std of
    log p. Error bars = std over seeds (none for one seed)."""
    fams = list(C)
    mean = np.array([C[f].mean(1) for f in fams])                                 # (F, seeds)
    std = np.array([C[f].std(1, ddof=1) for f in fams])
    norm = std / std[-1:]
    n_seeds = mean.shape[1]
    x = np.arange(len(fams))
    colors = [MODEL] * (len(fams) - 1) + [DATA]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    panels = ((mean, 'mean (nats)', dict(alpha=0.5)),
              (std, 'std across signals (nats)', {}),
              (norm, 'std / std of log p', dict(hatch='//', edgecolor='white')))
    for ax, (v, lab, style) in zip(axes, panels):
        err = v.std(1, ddof=1) if n_seeds > 1 else None
        ax.bar(x, v.mean(1), 0.6, yerr=err, capsize=3, color=colors, **style)
        ax.axhline(0, color=DATA, lw=0.6)
        ax.set_ylabel(lab)
        ax.grid(axis='x', visible=False)
    axes[2].axhline(1, color=GREY, ls='--', lw=1)
    axes[2].set_xticks(x, [family_label(f, Q) if f != 'TOTAL' else r'total $\log p$' for f in fams],
                       rotation=30, ha='right')
    if title:
        axes[0].set_title(title)
    fig.tight_layout()
    _save(fig, name)
