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
    'font.size': 16, 'axes.titlesize': 17, 'axes.labelsize': 18, 'legend.fontsize': 15,          # talk sizes
    'xtick.labelsize': 15, 'ytick.labelsize': 15, 'lines.linewidth': 2.4,
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
    import io
    if _OUT is not None:
        fig.savefig(_OUT / f'{name}.png')
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
        if get_ipython() is not None:
            # PNG bytes, not display(fig): a Figure is only rendered as an image when the inline backend registered its
            # formatter (%matplotlib inline); under Agg it would print '<Figure size ...>'
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            display(Image(data=buf.getvalue()))
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
            ax.text(1.01, 0.5, row_labels[i], transform=ax.transAxes, va='center', fontsize=14)
    if labels_header:
        axes[0].text(1.01, 1.25, labels_header, transform=axes[0].transAxes, va='bottom', fontsize=14)
    axes[-1].set_xlabel(xlabel)
    axes[n // 2].set_ylabel('signal / std')
    if title:
        axes[0].set_title(title)
    fig.subplots_adjust(hspace=0.12)
    _save(fig, name)


def plot_clean_signals(signals, name, n=3, accel_pcts=None, band=2.5, seed=None, color=DATA, size=(9.0, 2.6),
                       lw=1.4, sharey=True, with_axes=True, tau_eta=None):
    """n separate slide-ready figures, one signal each, nothing else on them: no axes, ticks, labels, tau or markers.
    Saved as <name>_1.png ... <name>_n.png.
    with_axes=True: the same n signals are also saved with axes, as <name>_axes_1.png ...: y = x / std (ticks at
    integer multiples of the std, the amplitude scale), x = time t (index of the sampled signal), or t / tau_eta if tau_eta is given.
    accel_pcts=None: n signals at random. accel_pcts=(20, 70, 97): one signal per entry, drawn at random among those whose
    acceleration (largest one-step increment max_t |x(t+1) - x(t)|) is within +-band percentile points of that percentile
    of `signals`, so the figures go from calm to agitated. sharey: same vertical scale for all n (a calm signal then looks
    calm next to an agitated one); False: each one fills its own panel.
    seed=None: new signals at every call; the chosen indices are printed."""
    s = _np(signals)
    rng = np.random.default_rng(seed)
    if accel_pcts is None:
        idx = rng.choice(len(s), n, replace=False)
    else:
        acc = np.abs(np.diff(s, axis=1)).max(1)
        rank = 100 * (np.argsort(np.argsort(acc)) + 0.5) / len(acc)               # percentile rank of each signal
        idx = []
        for p in accel_pcts:
            pool = np.flatnonzero((np.abs(rank - p) <= band) & ~np.isin(np.arange(len(s)), idx))
            idx.append(int(rng.choice(pool)))
        idx = np.array(idx)
        print('acceleration percentiles:', [f'{rank[i]:.0f}' for i in idx])
    print(f'{name}: signals {idx.tolist()}')
    def _range(a):                                                                  # fitted to the data, 6 % margin
        lo, hi = a.min(), a.max()
        return lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo)
    shared = _range(s[idx])
    for k, i in enumerate(idx):
        fig, ax = plt.subplots(figsize=size)
        ax.plot(s[i], color=color, lw=lw, solid_capstyle='round')
        ax.set_xlim(0, s.shape[1] - 1)
        ax.set_ylim(shared if sharey else _range(s[i]))
        ax.axis('off')
        _save(fig, f'{name}_{k + 1}')
    if not with_axes:
        return
    from matplotlib.ticker import MaxNLocator
    t = np.arange(s.shape[1]) / (tau_eta or 1)
    for k, i in enumerate(idx):
        fig, ax = plt.subplots(figsize=(size[0], size[1] + 0.9))
        ax.plot(t, s[i], color=color, lw=lw, solid_capstyle='round')
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(shared if sharey else _range(s[i]))
        ax.grid(False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.set_ylabel(r'$x/\sigma(x)$')
        ax.set_xlabel(r'$t/\tau_\eta$' if tau_eta else r'time $t$')
        _save(fig, f'{name}_axes_{k + 1}')


# ------------------------------------------------------- wavelet marginals ----
def plot_wavelet_histograms(x, J, name, title=None, Q=1, ncol=4, panel=None):
    """Histogram of the real part of the Morlet coefficients, one panel per wavelet channel, coefficients divided
    by their std; dashed: standard Gaussian.
    Q: wavelets per octave of the bank (Q = 1: one panel per scale j; Q > 1: J*Q channels, k = 0 is the finest).
    ncol: panels per row.  panel: (width, height) in inches of one panel; None = (3.6, 3.0) up to 4 columns,
    (3.0, 2.6) beyond, so a wide grid is not shrunk too much on a slide."""
    panel = panel or ((3.6, 3.0) if ncol <= 4 else (3.0, 2.6))
    from filters_bank import return_Filters
    x = torch.as_tensor(_np(x))
    M = x.shape[-1]
    filters = return_Filters(M, J, Q, device='cpu').reshape(-1, M).to(torch.complex128)    # wavelets + low-pass
    filters = filters[:-1]                                                                  # wavelets only (drop low-pass)
    z = torch.fft.ifft(torch.fft.fft(x)[:, None, :] * filters[None], dim=-1).real.numpy()   # (n, C, M)
    C = filters.shape[0]
    nrow = int(np.ceil(C / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel[0] * ncol, panel[1] * nrow), sharey=True, squeeze=False)    # own x span per scale
    for j, ax in enumerate(axes.flat):
        if j >= C:
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
        lab = f'j = {j}' if Q == 1 else f'j ≈ {j / Q:.1f}'                  # Q > 1: channel k at scale 2^(k/Q)
        ax.set_title(lab + (' (finest)' if j == 0 else ' (coarsest)' if j == C - 1 else ''), fontsize=16)
    axes.flat[0].legend(loc='lower center', fontsize=13)
    for ax in axes[-1]:
        ax.set_xlabel('coefficient / std')
    for ax in axes[:, 0]:
        ax.set_ylabel('density')
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    _save(fig, name)


# ---------------------------------------------------- Li et al. statistics ----
BAND = 'batch'      # default error band of the Li et al. statistics (override per call with band=...):
                    #   'batch': min / max over n_batches disjoint batches of the signals of ONE run (as Li et al.)
                    #   'seeds': mean +- std (ddof = 1) over independent runs; the model is then a LIST of arrays,
                    #            one per seed. The data has one realisation: its band stays min / max over batches.


def _li_curves(data, model, f, n_batches, max_lag, band=None, positive=False):
    """Statistic f(S, taus) -> (K, n_taus) of the moments S (P, n_taus), for the data and the model, with bands.
    Returns taus, label of the data, label of the model, (y, lo, hi) data, (y, lo, hi) model, each (K, n_taus).
    positive: the statistic is > 0 (log axis), so mean - std is clipped at 1 % of the mean."""
    from paper_figures_li import structure_functions
    band = BAND if band is None else band
    assert band in ('batch', 'seeds'), band
    ms = list(model) if isinstance(model, (list, tuple)) else [model]
    if band == 'seeds' and len(ms) < 2:
        print("band='seeds' needs the model as a list of >= 2 arrays: falling back to 'batch'")
        band = 'batch'
    taus, S_d, Sb_d = structure_functions(data, n_batches, max_lag)
    yb = np.stack([f(sb, taus) for sb in Sb_d])
    out_d = (f(S_d, taus), yb.min(0), yb.max(0))
    if band == 'seeds':
        per = np.stack([f(structure_functions(m, n_batches, max_lag)[1], taus) for m in ms])
        y, sd = per.mean(0), per.std(0, ddof=1)
        lo = np.maximum(y - sd, 1e-2 * y) if positive else y - sd
        out_m = (y, lo, y + sd)
        lab_d, lab_m = 'data (batches)', f'model (mean ± std, {len(ms)} seeds)'
    else:
        m = ms[0] if len(ms) == 1 else np.concatenate([_np(v) for v in ms])
        _, S_m, Sb_m = structure_functions(m, n_batches, max_lag)
        yb = np.stack([f(sb, taus) for sb in Sb_m])
        out_m = (f(S_m, taus), yb.min(0), yb.max(0))
        lab_d, lab_m = 'data', 'model'
    return taus, lab_d, lab_m, out_d, out_m


def _top_legend(fig, ax, ncol=2):
    """Legend of ax above the whole figure (never on a curve); call instead of fig.tight_layout()."""
    h, l = ax.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.90 if len(fig.axes) > 1 else 0.88))
    fig.legend(h, l, loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, 1.0))


MK_D, MK_M = 'o', 'x'      # markers on lag curves: one per computed lag (data / model); lines between them = interpolation


def _curve(ax, t, y, lo, hi, color, label, marker=None):
    ax.fill_between(t, lo, hi, color=color, alpha=0.12, lw=0)
    ax.plot(t, y, color=color, label=label, marker=marker, ms=3.5, mew=1.0)


def plot_increment_pdfs(data, model, tau_eta, name, tau_ratios=(1, 2, 5), shift_decades=3, n_bins=161,
                        label_x=0.55, label_gain=3.0, n_batches=10, band=None, min_count=100):
    """Standardized PDFs of the increments x(t+tau) - x(t) at tau / tau_eta in tau_ratios, shifted down by
    10^(-shift_decades) per lag for readability. Each set of signals is divided by the std of its own increments.
    Bands as the other Li et al. statistics (BAND): data = min / max over n_batches disjoint batches of signals;
    model = mean +- std over seeds ('seeds', model = list of arrays) or min / max over batches ('batch').
    On the log axis the lower edge of a band is clipped at 1 % of its line. Bands only where the pooled histogram has
    >= min_count counts (~10 per batch; tails with a few events: line only); lines join the non-empty bins.
    label_x: horizontal position of the tau/tau_eta labels (fraction of the half x-range); label_gain: factor above
    the curve."""
    band = BAND if band is None else band
    ms = [_np(v) for v in (model if isinstance(model, (list, tuple)) else [model])]
    if band == 'seeds' and len(ms) < 2:
        print("band='seeds' needs the model as a list of >= 2 arrays: falling back to 'batch'")
        band = 'batch'
    d = _np(data)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    labels = []

    def incs(v, tau):
        a = v[:, tau:] - v[:, :-tau]
        return a / a.std()                                           # (signals, positions), standardized

    def batch_band(a, bins):
        h = np.histogram(a.ravel(), bins, density=True)[0]
        hb = np.stack([np.histogram(a[i].ravel(), bins, density=True)[0]
                       for i in np.array_split(np.arange(len(a)), n_batches)])
        return h, hb.min(0), hb.max(0)

    for k, r in enumerate(tau_ratios):
        tau = max(1, int(round(r * tau_eta)))
        if tau > d.shape[-1] - 2:
            print(f'tau/tau_eta = {r} skipped: longer than the signal')
            continue
        a_d, a_m = incs(d, tau), [incs(v, tau) for v in ms]
        lim = max(np.abs(a_d).max(), max(np.abs(v).max() for v in a_m))
        bins = np.linspace(-lim, lim, n_bins)
        c = 0.5 * (bins[1:] + bins[:-1])
        shift = 10.0 ** (-shift_decades * k)
        hd = batch_band(a_d, bins)
        if band == 'seeds':
            H = np.stack([np.histogram(v.ravel(), bins, density=True)[0] for v in a_m])
            y, sd = H.mean(0), H.std(0, ddof=1)
            hm = (y, y - sd, y + sd)
        else:
            hm = batch_band(np.concatenate(a_m), bins)
        n_d = np.histogram(a_d.ravel(), bins)[0]
        n_m = np.histogram(np.concatenate([v.ravel() for v in a_m]), bins)[0]
        for (y, lo, hi), n, col, lab in ((hd, n_d, DATA, 'data (batches)' if band == 'seeds' else 'data'),
                                         (hm, n_m, MODEL, f'model (mean ± std, {len(ms)} seeds)' if band == 'seeds' else 'model')):
            ok = n >= min_count
            ax.fill_between(c, np.maximum(lo, 1e-2 * y) * shift, hi * shift, where=ok, color=col, alpha=0.15, lw=0)
            on = y > 0                                                   # line through the non-empty bins (as before)
            ax.plot(c[on], y[on] * shift, color=col, lw=1.6, label=lab if k == 0 else None)
        labels.append((r, shift, lim, c[hd[0] > 0], hd[0][hd[0] > 0]))
    xmax = max(l[2] for l in labels)
    for r, shift, lim, cx, hy in labels:      # label sits just above its own data curve, on the right shoulder
        xl = label_x * xmax
        yl = np.interp(xl, cx, hy) * shift * label_gain if xl < lim else hy[-1] * shift * label_gain
        ax.text(xl, yl, rf'$\tau/\tau_\eta = {r}$', ha='center', va='bottom', fontsize=15)
    ax.set_xlim(-xmax, xmax)
    ax.set_yscale('log')
    ax.set_xlabel(r'$\delta_\tau x\, /\, \sigma(\delta_\tau x)$')
    ax.set_ylabel('PDF (shifted)')
    ax.legend(loc='upper left')
    fig.tight_layout()
    _save(fig, name)


def plot_increment_trajectories(data, model, tau_eta, name, tau_ratios=(20, 5, 1), n_windows=8, seed=None, gap=4):
    """As Li et al. 2024 Fig. 1b-d: the signal x(t) and its increments delta_tau x(t) = x(t+tau) - x(t) at
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
    rows = [('$x$', 0)] + [(rf'$\delta_\tau x$' + '\n' + rf'$\tau/\tau_\eta = {r}$', t) for r, t in zip(tau_ratios, taus)]
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
        axes[r, 0].set_ylabel(lab, fontsize=15)
    axes[0, 0].set_title('data')
    axes[0, 1].set_title('model')
    for ax in axes[-1]:
        ax.set_xlabel(r'$t / \tau_\eta$')
    print(f'seed {seed}: {n_windows} signals of {T / tau_eta:.0f} tau_eta each, laid end to end')
    _save(fig, name)


def plot_rare_events(data, model, tau_eta, name, tau_ratios=(20, 5, 1), window=40, rare_pct=95, seed=None,
                     logp_data=None, logp_model=None, logp_pct=5.0, min_freq=1e-4, fig_width=8, rare_rows=(1,)):
    """Increments at several lags, as Li et al. 2024 Fig. 1b-d (typical trajectories at large, medium and small lag).
    One row per lag tau = ratio * tau_eta; data on the left (black), generated on the right (blue), same y-scale
    everywhere. Rows whose ratio is in rare_rows (default: tau/tau_eta = 1): a RARE signal by acceleration, i.e. its
    largest one-step increment max_t |x(t+1) - x(t)| is above the rare_pct-th percentile of the DATA (same absolute
    threshold for the generated signals), so the intense small-scale event shows. Other rows: a typical signal, at random
    among the non-rare ones. In each panel the largest increment at that row's lag is marked: two red dots, the double
    arrow is the lag, and the increment is written in units of the data std of delta_tau x. Only `window` tau_eta around
    it are shown, not the whole signal.
    logp_data / logp_model (log p of every data / generated signal, same order): select by log p instead of by increment:
    each row shows a different signal among the logp_pct % lowest log p (= highest -log p), after dropping the isolated
    outliers beyond the min_freq cut of plot_ranked_signals (min_freq=None keeps them). The event shown is still the
    largest increment at that row's lag, and -log p - median is written in the panel.
    fig_width: total width of the figure in inches (both columns): smaller = the signal is horizontally more compressed.
    seed=None: new signals at every call; the chosen indices are printed."""
    d, m = _np(data), _np(model)
    T = d.shape[-1]
    rng = np.random.default_rng(seed)
    by_logp = logp_data is not None and logp_model is not None
    if by_logp:
        lps, order = [], []
        for lp in (logp_data, logp_model):
            v = np.asarray(lp, dtype=np.float64)
            v = v - np.median(v)
            ok = np.flatnonzero(v >= (logp_support(v, 80, min_freq)[0] if min_freq else v.min()))
            pool = ok[v[ok] <= np.percentile(v[ok], logp_pct)]
            order.append(rng.permutation(pool)[:len(tau_ratios)])
            lps.append(v)
            print(f'log p selection: {len(pool)} signals in the lowest {logp_pct:g} % (of {len(ok)} kept)')
    W = min(T, int(round(window * tau_eta)))
    acc = [np.abs(np.diff(a_, axis=1)).max(1) for a_ in (d, m)]          # largest one-step increment (acceleration)
    thr_acc = np.percentile(acc[0], rare_pct)                            # threshold from the DATA, used for both
    is_rare = [a_ > thr_acc for a_ in acc]
    panels = []                                                          # [row][col] = dict
    for rr, r in enumerate(tau_ratios):
        tau = max(1, int(round(r * tau_eta)))
        row = []
        incs = [a_[:, tau:] - a_[:, :-tau] for a_ in (d, m)]
        sd = incs[0].std()
        want_rare = r in rare_rows
        for c_, (arr, inc) in enumerate(zip((d, m), incs)):
            cand = np.flatnonzero(is_rare[c_] if want_rare else ~is_rare[c_])
            if by_logp:
                i = int(order[c_][rr % len(order[c_])])
            else:
                i = int(rng.choice(cand)) if len(cand) else int(np.argmax(acc[c_]))
            k = int(np.argmax(np.abs(inc[i])))
            lo = int(np.clip(k + tau // 2 - W // 2, 0, T - W))
            row.append(dict(i=i, k=k - lo, tau=tau, y=arr[i, lo:lo + W], v=inc[i, k] / sd, n=len(cand), lp=lps[c_][i] if by_logp else None))
        panels.append(row)
        kind = 'by log p' if by_logp else ('rare (acceleration)' if want_rare else 'typical')
        print(f'tau/tau_eta = {r}: {kind} | data signal {row[0]["i"]} | generated signal {row[1]["i"]}')
    lim = 1.7 * max(np.abs(p['y']).max() for row in panels for p in row)          # headroom: arrow + label inside
    t = np.arange(W) / tau_eta
    n = len(tau_ratios)
    fig, axes = plt.subplots(n, 2, figsize=(fig_width, 1.9 * n + 0.8), sharex=True, sharey=True, squeeze=False)
    for rr, (r, row) in enumerate(zip(tau_ratios, panels)):
        for c, (col, p) in enumerate(zip((DATA, MODEL), row)):
            ax = axes[rr, c]
            k, tau, y = p['k'], p['tau'], p['y']
            ax.plot(t, y, color=col, lw=1.4)
            k2 = min(k + tau, W - 1)
            ya = 0.62 * lim
            ax.plot([t[k], t[k2]], [y[k], y[k2]], color='k', ls='--', lw=0.8)
            for kk in (k, k2):
                ax.plot([t[kk], t[kk]], [y[kk], ya], color='k', ls=':', lw=0.7)
            ax.plot(t[[k, k2]], y[[k, k2]], 'o', color='red', ms=5, zorder=5)
            short = (t[k2] - t[k]) < 0.06 * t[-1]                               # heads would overlap: plain bar
            ax.annotate('', xy=(t[k2], ya), xytext=(t[k], ya), arrowprops=dict(
                arrowstyle='-' if short else '<->, head_length=0.3, head_width=0.15', color='k', lw=2.0 if short else 1.4,
                shrinkA=0, shrinkB=0))
            mid = 0.5 * (t[k] + t[k2]) / t[-1]                                  # keep the label inside the panel
            ha, xa = ('left', t[k]) if mid < 0.22 else ('right', t[k2]) if mid > 0.78 else ('center', 0.5 * (t[k] + t[k2]))
            xa = min(max(xa, 0.03 * t[-1]), 0.97 * t[-1])                       # never on the axis spines
            ax.text(xa, ya + 0.04 * lim, rf'$\tau/\tau_\eta = {r}$,  $\delta_\tau x = {p["v"]:+.1f}\,\sigma$',
                    ha=ha, va='bottom', fontsize=14)
            if p['lp'] is not None:
                ax.text(0.01, 0.04, rf'$-\log p - \mathrm{{median}} = {-p["lp"]:+.0f}$', transform=ax.transAxes, fontsize=13)
            ax.set_ylim(-lim, lim)
            ax.grid(False)
        axes[rr, 0].set_ylabel(r'$x/\sigma(x)$')
    axes[0, 0].set_title('data', pad=14)
    axes[0, 1].set_title('generated', pad=14)
    for ax in axes[-1]:
        ax.set_xlabel(r'$t / \tau_\eta$')
    fig.tight_layout()
    _save(fig, name)


def plot_structure_functions(data, model, tau_eta, max_lag, name, orders=(2, 4, 6, 8), n_batches=10,
                             shift_decades=1.0, band=None):
    """S_p(tau) = <(delta_tau x)^p>, each order shifted down by 10^(-shift_decades) per order for readability
    (shift_decades = 0: no shift). band: 'batch' | 'seeds' | None (= module BAND), see BAND."""
    from paper_figures_li import ORDERS
    f = lambda S, t: np.stack([S[ORDERS.index(p)] for p in orders])
    taus, lab_d, lab_m, (yd, ld, hd), (ym, lm, hm) = _li_curves(data, model, f, n_batches, max_lag, band, positive=True)
    t = taus / tau_eta
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for k, p in enumerate(orders):
        sh = 10.0 ** (-shift_decades * k)
        _curve(ax, t, yd[k] * sh, ld[k] * sh, hd[k] * sh, DATA, lab_d if k == 0 else None, MK_D)
        _curve(ax, t, ym[k] * sh, lm[k] * sh, hm[k] * sh, MODEL, lab_m if k == 0 else None, MK_M)
        ax.text(t[-1] * 1.08, yd[k][-1] * sh, f'$p = {p}$', va='center', fontsize=15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\tau / \tau_\eta$')
    ax.set_ylabel(r'$S_p(\tau)$' + (' (shifted)' if shift_decades else ''))
    _top_legend(fig, ax)
    _save(fig, name)


def plot_flatness(data, model, tau_eta, max_lag, name, orders=(4, 6, 8), n_batches=10, band=None):
    """F_p(tau) = S_p / S_2^(p/2); bands as plot_structure_functions."""
    from paper_figures_li import _flat
    f = lambda S, t: np.stack([_flat(S, p) for p in orders])
    taus, lab_d, lab_m, (yd, ld, hd), (ym, lm, hm) = _li_curves(data, model, f, n_batches, max_lag, band, positive=True)
    t = taus / tau_eta
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for k, p in enumerate(orders):
        _curve(ax, t, yd[k], ld[k], hd[k], DATA, lab_d if k == 0 else None, MK_D)
        _curve(ax, t, ym[k], lm[k], hm[k], MODEL, lab_m if k == 0 else None, MK_M)
        ax.text(t[-1] * 1.08, yd[k][-1], f'$p = {p}$', va='center', fontsize=15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\tau / \tau_\eta$')
    ax.set_ylabel(r'$F_p(\tau) = S_p / S_2^{\,p/2}$')
    _top_legend(fig, ax)
    _save(fig, name)


def plot_compensated_sf(data, model, tau_eta, max_lag, name, orders=(4, 4.5, 5, 5.5, 6, 6.5), n_batches=10, band=None,
                        ncol=3):
    """S_p^(2/p) / S_2 = F_p^(2/p), S_p = <|delta_tau x|^p>, one panel per order (Valadao's slides, p. 4); dotted: the
    Gaussian value (2^(p/2) Gamma((p+1)/2) / sqrt(pi))^(2/p) (sqrt 3 for p = 4). Bands as plot_structure_functions."""
    from math import gamma, pi, sqrt
    from paper_figures_li import _flat
    f = lambda S, t: np.stack([_flat(S, p) ** (2 / p) for p in orders])
    taus, lab_d, lab_m, (yd, ld, hd), (ym, lm, hm) = _li_curves(data, model, f, n_batches, max_lag, band, positive=True)
    t = taus / tau_eta
    nrow = int(np.ceil(len(orders) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow), squeeze=False)
    for k, (ax, p) in enumerate(zip(axes.flat, orders)):
        _curve(ax, t, yd[k], ld[k], hd[k], DATA, lab_d, MK_D)
        _curve(ax, t, ym[k], lm[k], hm[k], MODEL, lab_m, MK_M)
        ax.axhline((2 ** (p / 2) * gamma((p + 1) / 2) / sqrt(pi)) ** (2 / p), color=GREY, ls=':', lw=1.6)
        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau / \tau_\eta$')
        ax.set_title(rf'$p = {p:g}$')
        ax.set_ylabel(rf'$S_{{{p:g}}}^{{2/{p:g}}} / S_2$')
    for ax in list(axes.flat)[len(orders):]:
        ax.axis('off')
    _top_legend(fig, axes.flat[0])
    _save(fig, name)


def plot_local_exponents(data, model, tau_eta, max_lag, name, orders=(4, 6, 8), n_batches=10, band=None):
    """zeta_p(tau) = dlog S_p / dlog S_2, one panel per order; dotted: p/2 (no intermittency); bands as above
    (with band='seeds', zeta is computed per seed, then averaged)."""
    from paper_figures_li import _zeta
    f = lambda S, taus: np.stack([_zeta(S, taus, p) for p in orders])
    taus, lab_d, lab_m, (yd, ld, hd), (ym, lm, hm) = _li_curves(data, model, f, n_batches, max_lag, band)
    t = taus / tau_eta
    fig, axes = plt.subplots(1, len(orders), figsize=(5 * len(orders), 4.5))
    for k, (ax, p) in enumerate(zip(np.atleast_1d(axes), orders)):
        _curve(ax, t, yd[k], ld[k], hd[k], DATA, lab_d, MK_D)
        _curve(ax, t, ym[k], lm[k], hm[k], MODEL, lab_m, MK_M)
        ax.axhline(p / 2, color=GREY, ls=':', lw=1.6)
        ax.set_ylim(p / 4, 1.1 * p / 2)
        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau / \tau_\eta$')
        ax.set_title(rf'$\zeta_{p}(\tau)$')
    _top_legend(fig, np.atleast_1d(axes)[0])
    _save(fig, name)


# --------------------------------------------------------- seminar extras ----
def plot_signal_pairs(data, gen, i_data, i_gen, name, row_labels=None, tau_eta=None, panel=(5.2, 1.5)):
    """Data signals (left, black) next to generated ones (right, blue), one row per pair, the same y-scale everywhere
    (layout of compare_signals in talk_turbulence.ipynb). row_labels (e.g. the category of the row) on the left.
    tau_eta: time axis in tau_eta instead of samples."""
    d, g = _np(data)[list(i_data)], _np(gen)[list(i_gen)]
    n = len(d)
    lim = 1.05 * max(np.abs(d).max(), np.abs(g).max())
    t = np.arange(d.shape[-1]) / (tau_eta or 1)
    fig, axes = plt.subplots(n, 2, figsize=(2 * panel[0], panel[1] * n + 1.0), sharex=True, sharey=True, squeeze=False)
    for k in range(n):
        axes[k, 0].plot(t, d[k], color=DATA, lw=1.4)
        axes[k, 1].plot(t, g[k], color=MODEL, lw=1.4)
        for ax in axes[k]:
            ax.set_ylim(-lim, lim)
            ax.grid(False)
        if row_labels is not None:
            axes[k, 0].set_ylabel(row_labels[k], rotation=0, ha='right', va='center', labelpad=12)
    axes[0, 0].set_title('data')
    axes[0, 1].set_title('generated')
    for ax in axes[-1]:
        ax.set_xlabel('time (samples)' if tau_eta is None else r'$t/\tau_\eta$')
    fig.subplots_adjust(hspace=0.18, wspace=0.06)
    print(f'{name}: data {list(map(int, i_data))} | generated {list(map(int, i_gen))}')
    _save(fig, name)


WAV_RE, WAV_IM = '#e8871e', '#8a8983'          # real part (orange) and imaginary part (grey) of wavelet coeffs


def plot_wavelet_stack(x, J, name, Q=1, every=1, tau_eta=None, width=14, row_height=None, hspace=0.45):
    """One signal (top) and its Morlet wavelet transform below, one row per wavelet channel from the finest (top) to
    the coarsest: real part (orange) over imaginary part (grey), same two colours at every scale. Each row
    is scaled to its own max |z|, so every scale is visible (amplitudes NOT comparable across rows). Q: wavelets per
    octave (channel k at scale 2^(k/Q)); every: show every k-th channel (Q = 3 has J*Q rows). tau_eta: time axis in
    tau_eta. row_height (inches per wavelet row): None = 0.75 up to 12 rows, 0.55 above; hspace: gap between rows."""
    from filters_bank import return_Filters
    x = _np(x).astype(np.float64).ravel()
    M = len(x)
    filt = return_Filters(M, J, Q, device='cpu').reshape(-1, M).to(torch.complex128)[:-1]      # drop the low-pass
    z = torch.fft.ifft(torch.fft.fft(torch.as_tensor(x, dtype=torch.complex128))[None] * filt, dim=-1).numpy()
    ks = list(range(0, len(z), every))
    many = len(ks) > 12
    row_height = row_height or (0.55 if many else 0.75)
    t = np.arange(M) / (tau_eta or 1)
    hr = [3.0, 0.9] + [1.0] * len(ks)                     # signal, empty gap, one row per scale
    fig, axes = plt.subplots(len(hr), 1, figsize=(width, row_height * sum(hr) + 1.2), sharex=True,
                             gridspec_kw=dict(height_ratios=hr, hspace=hspace))
    axes[0].plot(t, x, color=DATA, lw=1.6)
    axes[0].set_ylabel(r'$x(t)$')
    axes[0].grid(False)
    axes[1].set_visible(False)
    for ax, k in zip(axes[2:], ks):
        sc = np.abs(z[k]).max() or 1.0
        ax.axhline(0, color='#dcdbd5', lw=0.6, zorder=0)
        ax.plot(t, z[k].imag / sc, color=WAV_IM, lw=1.2)
        ax.plot(t, z[k].real / sc, color=WAV_RE, lw=1.4)                 # real part drawn on top
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines['left'].set_visible(False)
        if ax is not axes[-1]:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', length=0)
        lab = rf'$2^{{{k}}}$' if Q == 1 else rf'$2^{{{k}/{Q}}}$'
        ax.text(1.01, 0.5, lab, transform=ax.transAxes, va='center', ha='left', fontsize=13 if many else 15)
    top = axes[2]
    top.text(1.01, 1.0 + hspace + 0.3, 'scale', transform=top.transAxes, va='bottom', ha='left', fontsize=14)
    top.legend(handles=[plt.Line2D([], [], color=WAV_RE, lw=2.4, label='real part'),
                        plt.Line2D([], [], color=WAV_IM, lw=2.4, label='imaginary part')],
               loc='lower left', bbox_to_anchor=(0.0, 1.0 + 0.15), ncol=2, borderaxespad=0, handlelength=1.6)
    axes[-1].set_xlabel('time' if tau_eta is None else r'$t/\tau_\eta$')
    if tau_eta is None:
        axes[-1].set_xlim(0, M - 1)
        axes[-1].set_xticks(list(range(0, M, 64)) + [M - 1])
    _save(fig, name)


def _bank(M, J, Q):
    """The model's Morlet filters in Fourier space (filters_bank.return_Filters), real, shape (channels, M); the
    last channel is the low-pass."""
    from filters_bank import return_Filters
    f = return_Filters(M, J, Q, device='cpu').reshape(-1, M)
    return np.real(f.numpy()).astype(np.float64)


def _pi_label(x, _=None):
    """Tick label of x (radians) as a fraction of pi: 0, π/16, π/8, 3π/16, ..., π."""
    from fractions import Fraction
    fr = Fraction(x / np.pi).limit_denominator(256)
    if fr == 0:
        return '0'
    num = '' if fr.numerator == 1 else str(fr.numerator)
    return f'${num}\\pi$' if fr.denominator == 1 else f'${num}\\pi/{fr.denominator}$'


def _width_bracket(ax, c, half, y, text):
    """Horizontal bracket |<---->| from c - half to c + half at height y, text above it."""
    ax.annotate('', xy=(c - half, y), xytext=(c + half, y),
                arrowprops=dict(arrowstyle='|-|', color=DATA, lw=1.4, shrinkA=0, shrinkB=0, mutation_scale=6))
    ax.text(c, y + 0.05, text, ha='center', va='bottom', fontsize=15)


def plot_wavelet_single(M, J, Q, name, k=None, cut=0.01, pad=1.15, size=(6.2, 4.8), omega_max=None):
    """One wavelet of the model's bank (channel k; default: the one at scale 2^3, i.e. k = 3Q), two figures:
    <name>_time: real part (orange) and imaginary part (grey) against t (samples), divided by max |psi(t)|;
    <name>_fourier: its Fourier transform psi_hat(omega) (real, as defined in the code) against omega (radians /
    sample, ticks in multiples of pi). Each axis is cut where |psi| (|psi_hat|) falls below `cut` of its maximum,
    times `pad`: Q = 1 and Q = 3 get different windows, so each figure carries a width bracket
    (2 x rms width of |psi(t)|^2, resp. |psi_hat(omega)|^2) to compare them. omega_max: Fourier axis from 0 to
    omega_max instead (same window for every Q, so the widths compare by eye too)."""
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    f = _bank(M, J, Q)
    k = 3 * Q if k is None else k
    psi = np.fft.fftshift(np.fft.ifft(f[k]))
    psi = psi / np.abs(psi).max()
    t = np.arange(M) - M // 2
    e = np.abs(psi) ** 2
    sig_t = np.sqrt((t ** 2 * e).sum() / e.sum())
    half = pad * np.abs(t[np.abs(psi) >= cut]).max()
    w = np.abs(t) <= half
    fig, ax = plt.subplots(figsize=size)
    ax.axhline(0, color='#dcdbd5', lw=0.8, zorder=0)
    ax.plot(t[w], psi.imag[w], color=WAV_IM, lw=2.2)
    ax.plot(t[w], psi.real[w], color=WAV_RE, lw=2.4)
    _width_bracket(ax, 0, sig_t, 1.12, f'width ≈ {2 * sig_t:.0f}')
    ax.set_xlim(-half, half)
    ax.set_ylim(-1.1, 1.45)
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\psi(t)$')
    ax.grid(False)
    ax.legend(handles=[plt.Line2D([], [], color=WAV_RE, lw=2.4, label='real part'),
                       plt.Line2D([], [], color=WAV_IM, lw=2.4, label='imaginary part')],
              loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=2, borderaxespad=0, handlelength=1.4)
    fig.tight_layout()
    _save(fig, f'{name}_time')

    om = 2 * np.pi * np.arange(M // 2 + 1) / M                       # 0 ... pi radians / sample
    y = f[k][:M // 2 + 1]
    i0 = int(np.argmax(y))
    p = y ** 2
    sig_w = np.sqrt(((om - om[i0]) ** 2 * p).sum() / p.sum())
    on = om[np.abs(y) >= cut * y.max()]
    hw = pad * max(om[i0] - on.min(), on.max() - om[i0])
    lo, hi = (max(0.0, om[i0] - hw), om[i0] + hw) if omega_max is None else (0.0, omega_max)
    print(f'{name}: |psi_hat| < {cut:g} of max beyond omega = {on.max() / np.pi:.3f} pi')
    fig, ax = plt.subplots(figsize=size)
    ax.fill_between(om, 0, y, color=WAV_RE, alpha=0.18, lw=0)
    ax.plot(om, y, color=WAV_RE, lw=2.4)
    _width_bracket(ax, om[i0], sig_w, 1.12, f'width ≈ {2 * sig_w:.2f}')
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.45)
    ax.set_yticks([0, 0.5, 1])
    step = next(np.pi / d for d in (16, 32, 64, 128) if (hi - lo) / (np.pi / d) >= 3)      # at least 3 ticks
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(FuncFormatter(_pi_label))
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$\hat\psi(\omega)$')
    ax.grid(False)
    fig.tight_layout()
    _save(fig, f'{name}_fourier')
    print(f'{name}: Q = {Q}, channel {k}, centre {_pi_label(om[i0])} rad / sample | width (2 x rms): time {2 * sig_t:.1f} '
          f'samples, frequency {2 * sig_w:.3f} rad / sample | product {4 * sig_t * sig_w:.2f}')


def plot_wavelet_tiling(M, J, Q, name):
    """All the band-pass filters of the model's bank in Fourier space (finest = darkest; low-pass not shown), against
    omega (radians / sample, linear axis from 0 to pi). Exact values on the M-point frequency grid, a dot at each
    grid frequency: the coarse wavelets are resolved by only a few frequencies at M = 256."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    f = _bank(M, J, Q)
    om = 2 * np.pi * np.arange(1, M // 2 + 1) / M
    psi = f[:-1, 1:M // 2 + 1]
    cols = plt.get_cmap('viridis')(np.linspace(0.0, 0.85, len(psi)))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for c, y in zip(cols, psi):
        on = y > 2e-3                                                # skip the flat zero part (no dots there)
        ax.plot(om, y, color=c, lw=1.8)
        ax.plot(om[on], y[on], 'o', color=c, ms=3.2)
    ax.set_xlim(0, np.pi * 1.02)                                     # linear axis: the width of each wavelet is to scale
    ax.xaxis.set_major_locator(FixedLocator([np.pi * k / 8 for k in range(9)]))
    ax.xaxis.set_major_formatter(FuncFormatter(_pi_label))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0, 1.08)
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$\hat\psi_j(\omega)$')
    ax.grid(False)
    ax.set_title(f'Q = {Q}  ({len(psi)} wavelets)', loc='left', fontsize=16, pad=10)
    fig.tight_layout()
    _save(fig, name)


ENERGY_CATEGORIES = (('scalar multiscale', r'(^Scalar_|^S_(psi|morlet)|First_Order|Second_Order|^Scat1|^Scat2)'),
                     ('order 3 Re', r'(Third_Order_Real|^Scat3_Real)'),
                     ('order 3 Im', r'(Third_Order_Imag|^Scat3_Imag)'),
                     ('order 4 Re', r'(Fourth_Order.*Real|^Scat4_Real)'),
                     ('order 4 Im', r'(Fourth_Order.*Imag|^Scat4_Imag)'),
                     ('confinement', r'^L_?6$'))                     # last: rightmost column, can be cropped off
ENERGY_ROWS = ('number of moments', 'mean of -log p (nats)', '± mean', 'std of -log p over signals (nats)', '± std')


def energy_table(C, widths, name=None):
    """Contribution of each category (ENERGY_CATEGORIES) to -log p: scalar multiscale moments (wavelet marginals,
    scattering orders 1-2), order 3 and order 4 across scales (real and imaginary parts separately), confinement (L_6).
    C = {family: (signals,) or (seeds, signals)} contributions theta_F . phi_F(x) to log p (nats; 'TOTAL' ignored);
    the table uses -theta_F . phi_F(x). widths = {family: number of statistics}.
    Per seed: mean and std over signals; the table gives their average over seeds, and '± ...' = std over seeds
    (ddof = 1; NaN with one seed). The mean is a level only (a constant added to phi_F shifts it without changing p;
    convention to be decided). Saved as <name>.csv in the output folder."""
    import pandas as pd
    fams = [f for f in C if f.lower() != 'total']
    cat = {}
    for f in fams:
        hit = [c for c, pat in ENERGY_CATEGORIES if re.search(pat, f)]
        assert len(hit) == 1, f'family {f!r} matches {hit} (expected exactly one category)'
        cat[f] = hit[0]
    miss = [f for f in fams if f not in widths]
    assert not miss, f'no number of statistics for {miss}'
    cols = [c for c, _ in ENERGY_CATEGORIES]
    out = {r: [] for r in ENERGY_ROWS}
    n_seeds = None
    for c in cols:
        fs = [f for f in fams if cat[f] == c]
        if not fs:
            for r in ENERGY_ROWS:
                out[r].append(0 if r.startswith('number') else np.nan)
            continue
        E = -sum(np.atleast_2d(np.asarray(C[f], dtype=np.float64)) for f in fs)         # (seeds, signals), -log p
        n_seeds = E.shape[0]
        m, s = E.mean(1), E.std(1, ddof=1)                                               # per seed
        err = (lambda a: float(a.std(ddof=1))) if n_seeds > 1 else (lambda a: np.nan)
        out['number of moments'].append(int(sum(widths[f] for f in fs)))
        out['mean of -log p (nats)'].append(float(m.mean()))
        out['± mean'].append(err(m))
        out['std of -log p over signals (nats)'].append(float(s.mean()))
        out['± std'].append(err(s))
        print(f'  {c:18s} <- {", ".join(fs)}')
    T = pd.DataFrame(out, index=cols).T
    print(f'energy table over {n_seeds} seed(s); ± = std over seeds')
    if name and _OUT is not None:
        T.to_csv(_OUT / f'{name}.csv')
    return T


def _pm(a, e):
    """'a ± e' with e to 2 significant digits and a rounded to the same decimal place."""
    if not np.isfinite(e) or e == 0:
        return f'{a:.3g}'.replace('-', '−')
    d = max(0, 1 - int(np.floor(np.log10(abs(e)))))
    return f'{a:.{d}f} ± {e:.{d}f}'.replace('-', '−')                 # typographic minus, as in the row labels


_MOMENT_ROWS = (('moments matched (%)', 'moments matched', '{:.0f} %'),             # optional rows of T, above -log p
                ('mean offset (data std)', 'mean offset', '{:.2g}'))
_TAB_ROWS = (('mean of -log p (nats)', '± mean', r'mean of $-\log p$'),
             ('std of -log p over signals (nats)', '± std', r'std of $-\log p$'))   # both over the data signals


def _draw_energy_table(blocks, cols, name, corner=''):
    """Draw a table in booktabs style (horizontal rules only). blocks = [(title or None, T)], T from energy_table;
    per block: optional bold title line, the number of moments (small grey italics: a count, not a result), then
    mean and std of -log p (± std over seeds). corner: text in the top-left cell. Values in nats."""
    lines = []                                                       # (kind, label, cells)
    for title, T in blocks:
        if title:
            lines.append(('title', title, None))
        n = T.loc['number of moments', cols].to_numpy(dtype=float)
        lines.append(('count', 'number of moments', ['—' if a == 0 else f'{int(a)}' for a in n]))
        for row, lab, fmt in _MOMENT_ROWS:                           # moment reproduction (moment_reproduction.py), if given
            if row in T.index:
                v = T.loc[row, cols].to_numpy(dtype=float)
                lines.append(('value', lab, ['—' if k == 0 or not np.isfinite(a) else fmt.format(a) for k, a in zip(n, v)]))
        for row, err, lab in _TAB_ROWS:
            v, e = (T.loc[r, cols].to_numpy(dtype=float) for r in (row, err))
            lines.append(('value', lab, ['—' if k == 0 or not np.isfinite(a) else _pm(a, x) for k, a, x in zip(n, v, e)]))
        lines.append(('gap', None, None))
    lines.pop()
    _render_table(lines, cols, name, corner, 'nats; ± = std over seeds')


def _render_table(lines, cols, name, corner, footer, shade_last=False, cw=2.35):
    """Booktabs-style drawing of lines = [(kind, label, cells)], kind in title / count / value / gap; one column per
    entry of cols; footer: small grey text under the bottom rule, right-aligned; shade_last: grey band behind the last
    column (a quantity of a different kind than the other columns)."""
    LW, CW, RH = 4.3, cw, 0.5                                        # label column, value column, row height (inches)
    H = RH * (1.3 + sum(0.75 if k == 'count' else 0.45 if k == 'gap' else 1.0 for k, _, _ in lines)) + 0.3
    W = LW + CW * len(cols)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    xc = [LW + CW * (i + 0.5) for i in range(len(cols))]
    rule = lambda y, lw: ax.plot([0.1, W - 0.1], [y, y], color=DATA, lw=lw, solid_capstyle='butt')
    y = 0.15
    rule(y, 1.6)
    y += RH * 0.65
    ax.text(0.15, y, corner, ha='left', va='center', fontsize=17, weight='bold')
    for x, c in zip(xc, cols):
        ax.text(x, y, c, ha='center', va='center', fontsize=16, weight='bold')
    y += RH * 0.65
    rule(y, 0.9)
    for kind, lab, cells in lines:
        if kind == 'gap':
            y += RH * 0.2
            ax.plot([0.1, W - 0.1], [y, y], color=GREY, lw=0.8)
            y += RH * 0.25
            continue
        h = RH * (0.75 if kind == 'count' else 1.0)
        yc = y + h / 2
        if kind == 'title':
            ax.text(0.15, yc, lab, ha='left', va='center', fontsize=17, weight='bold')
        elif kind == 'count':
            ax.text(0.35, yc, lab, ha='left', va='center', fontsize=13, style='italic', color='#77766f')
            for x, s in zip(xc, cells):
                ax.text(x, yc, s, ha='center', va='center', fontsize=13, style='italic', color='#77766f')
        else:
            ax.text(0.35, yc, lab, ha='left', va='center', fontsize=16)
            for x, s in zip(xc, cells):
                ax.text(x, yc, s, ha='center', va='center', fontsize=16)
        y += h
    y += 0.1
    rule(y, 1.6)
    if shade_last:
        ax.add_patch(plt.Rectangle((LW + CW * (len(cols) - 1), 0.15), CW, y - 0.15, color=GREY, alpha=0.2, lw=0, zorder=0))
    ax.text(W - 0.1, y + 0.12, footer, ha='right', va='top', fontsize=12, color='#77766f')
    _save(fig, name)


def plot_energy_table(tables, name):
    """tables = {'Lagrangian': energy_table(...), 'Eulerian': ...}: one table for a slide, one block of rows per
    dataset (number of moments; mean and std over signals of -log p, each ± std over seeds), one column per category,
    confinement last; empty category = '—'."""
    cols = list(next(iter(tables.values())).columns)
    _draw_energy_table(list(tables.items()), cols, name)


def plot_energy_table_single(T, name, title):
    """One dataset alone (title in the top-left corner): only the categories the model has (number of moments > 0),
    confinement last."""
    cols = [c for c in T.columns if T.loc['number of moments', c] > 0]
    _draw_energy_table([(None, T)], cols, name, corner=title)


# ---- Var/2 tables: variance of -log10 p / d per category (the width of the histograms, related to the KL divergence) ----
KL_ROWS = ('number of moments', 'Var/2', '± Var/2', 'std of total')
STD_COL = 'std(−log p)'                                   # last column: a different quantity, drawn shaded


def kl_table(C, widths, d, name=None):
    """First-order KL per dimension of each category of ENERGY_CATEGORIES (columns), in log10 units (digits / dimension,
    the units of the u = -log10 p / d axis): Var(ln p_k) / (2 d ln10), row 'Var/2'. Then the column STD_COL: the std of the
    total u = -log10 p / d over the signals (the width of the histogram), a different quantity, in row 'std of total'.
    C = {family: (seeds, signals)} contributions theta_F . phi_F(x) to ln p (nats, as stored; 'TOTAL' ignored);
    d = length of the signals; widths = {family: number of statistics}. Per seed: variance over the signals of the
    category's -log10 p / d (ddof = 1), divided by 2; the table gives the mean over seeds, '± Var/2' = std over seeds
    (NaN with one seed); the std column is the mean over seeds of the per-seed std (ddof = 1).
    Units: (log10 per point)^2 for Var/2, log10 per point for the std. Saved as <name>.csv in the output folder."""
    import pandas as pd
    fams = [f for f in C if f.lower() != 'total']
    cat = {}
    for f in fams:
        hit = [c for c, pat in ENERGY_CATEGORIES if re.search(pat, f)]
        assert len(hit) == 1, f'family {f!r} matches {hit} (expected exactly one category)'
        cat[f] = hit[0]
    miss = [f for f in fams if f not in widths]
    assert not miss, f'no number of statistics for {miss}'
    sc = neglogp_scale(d)
    cols = [c for c, _ in ENERGY_CATEGORIES] + [STD_COL]
    out = {r: [] for r in KL_ROWS}
    n_seeds = None
    Etot = -sum(np.atleast_2d(np.asarray(C[f], dtype=np.float64)) for f in fams) * sc      # (seeds, signals), all families
    vtot = float(Etot.var(1, ddof=1).mean())
    for c in cols:
        fs = fams if c == STD_COL else [f for f in fams if cat[f] == c]
        if not fs:
            out['number of moments'].append(0)
            for r in KL_ROWS[1:]:
                out[r].append(np.nan)
            continue
        E = -sum(np.atleast_2d(np.asarray(C[f], dtype=np.float64)) for f in fs) * sc    # (seeds, signals), -log10 p / d
        n_seeds = E.shape[0]
        # first-order KL per dimension, in log10 units (digits / dimension, the units of the u axis):
        # Var(ln p_k) / (2 d ln10) = (d ln10) Var(u_k) / 2.  A KL changes unit LINEARLY (nats -> digits: / ln10),
        # while a variance would change by ln10^2; and Var(ln p_k) grows like d, so dividing by d (not d^2) is intensive.
        v = d * np.log(10.0) * E.var(1, ddof=1) / 2                                      # per seed
        out['number of moments'].append(int(sum(widths[f] for f in fs)))
        out['Var/2'].append(np.nan if c == STD_COL else float(v.mean()))
        out['± Var/2'].append(float(v.std(ddof=1)) if n_seeds > 1 and c != STD_COL else np.nan)
        out['std of total'].append(float(E.std(1, ddof=1).mean()) if c == STD_COL else np.nan)
        if c != STD_COL:             # diagnostic: share of Var(total) carried by the category, Cov(C, total) / Var(total)
            cov = ((E - E.mean(1, keepdims=True)) * (Etot - Etot.mean(1, keepdims=True))).sum(1) / (E.shape[1] - 1)
            print(f'  {c:18s} Var(C)/Var(total) = {float(E.var(1, ddof=1).mean()) / vtot:8.2f} | '
                  f'Cov(C, total)/Var(total) = {float(cov.mean()) / vtot:6.2f}   (shares sum to 1; > 1 or < 0: families cancel)')
    T = pd.DataFrame(out, index=cols).T
    print(f'Var/2 table over {n_seeds} seed(s), d = {d}; ± = std over seeds')
    if name and _OUT is not None:
        T.to_csv(_OUT / f'{name}.csv')
    return T


def _pm_sci(a, e):
    """'(a ± e) x 10^k', e to 1 significant digit (2 if it starts with 1), a rounded to the same decimal place; each cell
    has its own power of ten. With few seeds the error itself is uncertain (35 % for 5), so more digits would be false. The values of the
    categories span several orders of magnitude, so one common power of ten per table does not read."""
    if not np.isfinite(a) or a == 0:
        return '0'
    k = int(np.floor(np.log10(abs(a))))
    if not np.isfinite(e) or e == 0:
        if round(a / 10 ** k, 2) >= 10:                                    # 9.999 rounds to 10.0: next power of ten
            k += 1
        return rf'${a / 10 ** k:.3g}\times10^{{{k}}}$'
    for _ in range(2):
        am, em = a / 10 ** k, e / 10 ** k
        ke = int(np.floor(np.log10(abs(em))))
        dec = max(0, -ke + (1 if abs(em) / 10 ** ke < 2 else 0))          # error to 1 significant digit (2 if it starts with 1)
        if round(am, dec) < 10:
            break
        k += 1                                                             # 9.999 -> 10.00: write 1.000 x 10^(k+1)
    return rf'$({am:.{dec}f} \pm {em:.{dec}f})\times10^{{{k}}}$'          # mean rounded to the decimal place of the error


def _sci_parts(a, e=np.nan):
    """(text, k) of a cell: text = 'a ± e' (e to 1 significant digit, 2 if it starts with 1, a rounded to the same
    decimal place; just 'a' to 3 digits without e) and the power of ten k of the cell, drawn on a second line."""
    if not np.isfinite(a) or a == 0:
        return '0', None
    k = int(np.floor(np.log10(abs(a))))
    if not np.isfinite(e) or e == 0:
        if round(a / 10 ** k, 2) >= 10:                                    # 9.999 rounds to 10.0: next power of ten
            k += 1
        return f'{a / 10 ** k:.3g}', k
    for _ in range(2):
        am, em = a / 10 ** k, e / 10 ** k
        ke = int(np.floor(np.log10(abs(em))))
        dec = max(0, -ke + (1 if abs(em) / 10 ** ke < 2 else 0))
        if round(am, dec) < 10:
            break
        k += 1
    return f'{am:.{dec}f} ± {em:.{dec}f}', k


def _wrap_header(c):
    """Two-line column header: 'scalar multiscale' -> 'scalar' / 'multiscale', 'order 3 Re' -> 'order 3' / 'Re'."""
    return c if ('(' in c or ' ' not in c) else '\n'.join(c.rsplit(' ', 1))


def _render_kl_table(blocks, cols, name, foot, corner='', shade_last=False, val_label=r'Var$/2d$'):
    """Slide table. blocks = [(title or None, counts (strings), cells (text, k))]; per cell the value in large type with
    its power of ten on a second, smaller line; wrapped headers, little padding; shade_last: grey band behind the last
    column (a different quantity)."""
    LW, CW = 2.0, 3.25                                                   # label column, value column (inches)
    HEAD, TITLE, COUNT, VAL, GAP, FOOT, TOP = 1.0, 0.6, 0.45, 0.8, 0.25, 0.45, 0.12
    H = TOP + HEAD + sum((TITLE if t else 0) + COUNT + VAL for t, _, _ in blocks) + GAP * (len(blocks) - 1) + FOOT
    W = LW + CW * len(cols)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    xc = [LW + CW * (i + 0.5) for i in range(len(cols))]
    rule = lambda y, lw, col=DATA: ax.plot([0.1, W - 0.1], [y, y], color=col, lw=lw, solid_capstyle='butt')
    MUTED = '#77766f'
    y = TOP
    rule(y, 1.8)
    ax.text(0.15, y + HEAD / 2, corner, ha='left', va='center', fontsize=24, weight='bold')
    for x, c in zip(xc, cols):
        ax.text(x, y + HEAD / 2, _wrap_header(c), ha='center', va='center', fontsize=19, weight='bold', linespacing=1.05)
    y += HEAD
    rule(y, 1.0)
    for i, (title, counts, cells) in enumerate(blocks):
        if i:
            y += GAP / 2
            rule(y, 0.8, GREY)
            y += GAP / 2
        if title:
            ax.text(0.15, y + TITLE / 2, title, ha='left', va='center', fontsize=24, weight='bold')
            y += TITLE
        ax.text(0.25, y + COUNT / 2, 'number of moments', ha='left', va='center', fontsize=13, style='italic', color=MUTED)
        for x, s in zip(xc, counts):
            ax.text(x, y + COUNT / 2, s, ha='center', va='center', fontsize=15, style='italic', color=MUTED)
        y += COUNT
        ax.text(0.25, y + VAL / 2, val_label, ha='left', va='center', fontsize=21)
        for x, (txt, k) in zip(xc, cells):                               # value and power of ten on ONE line, same size
            if k is None:
                s = txt
            elif '±' in txt:
                s = rf'$({txt.replace("±", chr(92) + "pm")})\times10^{{{k}}}$'
            else:
                s = rf'${txt}\times10^{{{k}}}$'
            ax.text(x, y + VAL / 2, s, ha='center', va='center', fontsize=20)
        y += VAL
    y += 0.05
    rule(y, 1.8)
    if shade_last:
        ax.add_patch(plt.Rectangle((LW + CW * (len(cols) - 1), TOP), CW, y - TOP, color=GREY, alpha=0.2, lw=0, zorder=0))
    ax.text(W - 0.1, y + 0.1, foot, ha='right', va='top', fontsize=12, color=MUTED)
    _save(fig, name)


def _draw_kl_table(blocks, cols, name, corner='', note=''):
    """blocks = [(title or None, T)], T from kl_table. Per block: number of moments (small grey italics), Var/2 with
    its error (each cell with its own power of ten, on a second line). The last column (STD_COL, shaded) is another
    quantity: the std of the total -log10 p / d, written in the same row."""
    rb = []
    for title, T in blocks:
        n = T.loc['number of moments', cols].to_numpy(dtype=float)
        v, e = (T.loc[r, cols].to_numpy(dtype=float) for r in ('Var/2', '± Var/2'))
        s = T.loc['std of total', cols].to_numpy(dtype=float)
        cells = [_sci_parts(a) if c == STD_COL and np.isfinite(a) else
                 (('—', None) if m == 0 or not np.isfinite(b) else _sci_parts(b, x))
                 for c, m, a, b, x in zip(cols, n, s, v, e)]
        rb.append((title, ['—' if a == 0 else f'{int(a)}' for a in n], cells))
    foot = r'Var$(\ln p_k)/(2d\ln 10)$ in digits per dimension, mean ± std over seeds'
    if STD_COL in cols:
        foot += r'; shaded: std of $-\log_{10}p/d$'
    if note:
        foot = f'{note}  |  {foot}'
    _render_kl_table(rb, cols, name, foot, corner=corner, shade_last=STD_COL in cols)


CONF_COL = 'confinement'


def plot_kl_table(tables, name, std=True, note='', confinement=False):
    """tables = {title: kl_table(...)}: one table, one block of rows per dataset (the overall table).
    std=False: no shaded std(-log p) column, only the Var/2 of the categories. note: text added to the footer.
    confinement=False (default): the confinement column is left out (own table: plot_confinement_table)."""
    cols = [c for c in next(iter(tables.values())).columns
            if any(T.loc['number of moments', c] > 0 for T in tables.values())
            and (std or c != STD_COL) and (confinement or c != CONF_COL)]
    _draw_kl_table(list(tables.items()), cols, name, note=note)


def plot_kl_table_single(T, name, title, std=True, note='', confinement=False):
    """One dataset alone (title in the top-left corner): only the columns with moments; std=False: no std column;
    confinement=False (default): no confinement column."""
    cols = [c for c in T.columns if T.loc['number of moments', c] > 0 and (std or c != STD_COL)
            and (confinement or c != CONF_COL)]
    _draw_kl_table([(None, T)], cols, name, corner=title, note=note)


IM_RE_COLS = (('order 3 Im/Re', 'order 3 Im', 'order 3 Re'), ('order 4 Im/Re', 'order 4 Im', 'order 4 Re'))


def im_re_ratio(C, widths, d):
    """Imaginary / real ratio of the Var/2 of the order-3 and of the order-4 category (ENERGY_CATEGORIES), same
    contributions as kl_table (C = {family: (seeds, signals)} of theta_F . phi_F(x), widths = {family: number of
    statistics}). Per seed: Var over the signals (ddof = 1) of the category's -log10 p / d, Im over Re (the factor
    d ln10 / 2 of Var/2 cancels); the table gives the mean over seeds and the std over seeds (NaN with one seed).
    Returns {column: (mean, std, n_im, n_re)}; None for an order the model has no Re/Im terms of."""
    fams = [f for f in C if f.lower() != 'total']
    cat = {}
    for f in fams:
        hit = [c for c, pat in ENERGY_CATEGORIES if re.search(pat, f)]
        assert len(hit) == 1, f'family {f!r} matches {hit} (expected exactly one category)'
        cat[f] = hit[0]
    out = {}
    for col, c_im, c_re in IM_RE_COLS:
        var, n = {}, {}
        if not all(any(cat[f] == c for f in fams) for c in (c_im, c_re)):
            print(f'  {col}: the model has no {c_im!r} / {c_re!r} terms -> no ratio')
            out[col] = None
            continue
        for c in (c_im, c_re):
            fs = [f for f in fams if cat[f] == c]
            E = -sum(np.atleast_2d(np.asarray(C[f], dtype=np.float64)) for f in fs) * neglogp_scale(d)
            var[c], n[c] = E.var(1, ddof=1), int(sum(widths[f] for f in fs))
        r = var[c_im] / var[c_re]                                           # (seeds,)
        out[col] = (float(r.mean()), float(r.std(ddof=1)) if len(r) > 1 else np.nan, n[c_im], n[c_re])
    return out


def plot_im_re_table(ratios, name):
    """ratios = {dataset title: im_re_ratio(...)}: one block per dataset, columns order 3 Im/Re and order 4 Im/Re;
    per cell the ratio mean ± std over seeds (power of ten on the same line, as the Var/2 tables), above it the number
    of moments (Im / Re). Saved as <name>.png and <name>.csv in the output folder."""
    import pandas as pd
    cols = [c for c, _, _ in IM_RE_COLS]
    blocks, rows = [], {}
    for title, R in ratios.items():
        counts = ['—' if R[c] is None else f'{R[c][2]} / {R[c][3]}' for c in cols]
        cells = [('—', None) if R[c] is None else _sci_parts(R[c][0], R[c][1]) for c in cols]
        blocks.append((title, counts, cells))
        rows[title] = {c: np.nan if R[c] is None else R[c][0] for c in cols}
        rows[title + ' (std over seeds)'] = {c: np.nan if R[c] is None else R[c][1] for c in cols}
    foot = r'Var$(\ln p_k)$ of the imaginary over the real category, per seed, mean ± std over seeds; moments: Im / Re'
    _render_kl_table(blocks, cols, name, foot, val_label='Im / Re')
    if _OUT is not None:
        pd.DataFrame(rows).T.to_csv(_OUT / f'{name}.csv')


def plot_confinement_table(tables, name):
    """Only the confinement term (L^6): one column per dataset that has it (number of moments, Var/2 with its error).
    tables = {title: kl_table(...)}; datasets without a confinement term are left out."""
    have = {t: T for t, T in tables.items() if CONF_COL in T.columns and T.loc['number of moments', CONF_COL] > 0}
    if not have:
        print('confinement table: no dataset has a confinement term')
        return
    counts = [str(int(T.loc['number of moments', CONF_COL])) for T in have.values()]
    cells = [_sci_parts(T.loc['Var/2', CONF_COL], T.loc['± Var/2', CONF_COL]) for T in have.values()]
    _render_kl_table([(None, counts, cells)], list(have), name,
                     r'Var$(\ln p_k)/(2d\ln 10)$ in digits per dimension, confinement term, mean ± std over seeds',
                     corner='confinement')

# -------------------------------------------------------------------- log p ----
def neglogp_scale(d):
    """Factor that turns nats into log10 per point: -log10 p / d = (-ln p) / (d ln 10)."""
    return 1.0 / (d * np.log(10.0))


NLP_LABEL = r'$-\log_{10} p(x)\,/\,d$'


def plot_logp_histogram(logp, name, rare=None, bins=100, rare_label='rare signals', min_freq=None, neg=True, note=None,
                        d=256, center='mean'):
    """Histogram of -log10 p / d over signals (logp = ln p as stored, i.e. Theta . phi; d = length of the signal),
    centred at zero by subtracting its mean (center='median' for the median; log p is known up to a constant),
    normalised to unit area (density); log y. Legend without counts (they do not add up once the tail is cut).
    rare (bool mask): those signals in red, drawn in front of the light histogram of all signals (both
    normalised by the total number of signals, so the red area = fraction of rare signals).
    min_freq: cut the non-significant tail as plot_ranked_signals does (logp_support: walking out from the most
    populated of `bins` bins, stop at the first bin with frequency < min_freq); the signals beyond are left out of
    the plot, the density stays normalised by ALL signals. None = whole range.
    neg=True (default): plot -log p (rare signals on the right); neg=False: +log p.
    note: text written below the plot, under a thin line and a blank band, so it can be cropped off."""
    v = np.asarray(logp, dtype=np.float64)
    v = v - (v.mean() if center == 'mean' else np.median(v))
    if neg:
        v = -v
    v = v * neglogp_scale(d)                                         # nats -> log10 per point
    N = len(v)
    if min_freq:
        lo, hi, *_ = logp_support(v, bins, min_freq)
        keep = (v >= lo) & (v <= hi)
        print(f'log p histogram: tail cut at freq >= {min_freq:g} -> [{lo:+.3f}, {hi:+.3f}] (log10 p / d); '
              f'{(v < lo).sum()} below, {(v > hi).sum()} above of {N} signals left out')
        v = v[keep]
        rare = None if rare is None else np.asarray(rare, bool)[keep]
    edges = np.linspace(v.min(), v.max(), bins + 1)
    w = np.full(len(v), 1.0 / (N * (edges[1] - edges[0])))                     # density, normalised by all N signals
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(v, edges, weights=w, color=GREY, alpha=0.45, label='all signals')
    if rare is not None:
        rare = np.asarray(rare, bool)
        ax.hist(v[~rare], edges, weights=w[~rare], histtype='step', color=DATA, lw=1.4, label='other signals', zorder=4)
        ax.hist(v[rare], edges, weights=w[rare], color=RARE, label=rare_label, zorder=3)
    ax.set_yscale('log')
    ax.set_ylim(0.5 * w[0], None)
    ax.set_xlabel('', labelpad=8)                                    # no axis labels (stated in the caption / by the speaker)
    ax.set_ylabel('', labelpad=8)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=1, borderaxespad=0, handlelength=1.4,
              fontsize=14)                                               # above the axes: never on the bars
    fig.tight_layout()
    if note:
        y = ax.xaxis.label.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y0
        y -= 0.035                                                       # blank band under the x label: crop here
        fig.add_artist(plt.Line2D([0.02, 0.98], [y, y], transform=fig.transFigure, color=GREY, lw=0.8))
        fig.text(0.02, y - 0.025, note, ha='left', va='top', fontsize=13, linespacing=1.5)   # \n in note = line break
    _save(fig, name)


def plot_logp_gumbel(logp, name, d=256, bins=30, min_freq=1e-4, center='mean'):
    """-log10 p / d histogram in the style of the S&P figure (light blue, density, log y, no axis labels) with a
    maximum-likelihood Gumbel fit in red (scipy gumbel_r: long tail on the right, as -log p).
    logp = ln p as stored (negated here), one value per signal, centred as plot_logp_histogram (center='mean'|'median');
    pool several seeds by centring each first and concatenating. The fit uses ALL signals; the histogram and the
    curve are drawn only over the support left after the tail cut (logp_support: frequency per bin >= min_freq);
    density normalised by all signals. Location and scale are printed and written in the legend."""
    v = np.asarray(logp, dtype=np.float64)
    v = -(v - (v.mean() if center == 'mean' else np.median(v)))
    v = v * neglogp_scale(d)                                         # nats -> log10 per point
    N = len(v)
    loc, scale = stats.gumbel_r.fit(v)
    lo, hi, *_ = logp_support(v, bins, min_freq)
    keep = (v >= lo) & (v <= hi)
    print(f'Gumbel fit of -log10 p / d: loc {loc:.4f}, scale {scale:.4f} | tail cut -> [{lo:+.3f}, {hi:+.3f}]: '
          f'{(v < lo).sum()} below, {(v > hi).sum()} above of {N} signals left out')
    v = v[keep]
    edges = np.linspace(v.min(), v.max(), bins + 1)
    w = np.full(len(v), 1.0 / (N * (edges[1] - edges[0])))
    xs = np.linspace(edges[0], edges[-1], 400)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(v, edges, weights=w, color=MODEL, alpha=0.25)
    ax.plot(xs, stats.gumbel_r.pdf(xs, loc, scale), color=RARE, lw=2.4, label=f'Gumbel fit (scale {scale:.3g})')
    ax.set_yscale('log')
    ax.set_ylim(0.5 * w[0], None)
    ax.legend()                                                      # no axis labels
    fig.tight_layout()
    _save(fig, name)
    return loc, scale


def plot_logp_widths(sets, name, bins=40, colors=None):
    """-log10 p / d histograms of several datasets, one separate figure each (saved as <name>_1, <name>_2, ...), all
    with the SAME x and y limits (log y), to see which concentrates more. sets: {label: (nlp, d)}, nlp = -log p in nats of the generated signals,
    (seeds, signals) or (signals,); d = signal length. Convention of plot_logp_histogram: each seed centred on its
    own mean, nats -> log10 per point (neglogp_scale(d)), density (unit area), log y. Every histogram has `bins` bins
    over its own range (a narrow one still gets resolved); the std of each is written in its panel."""
    cols = colors or [MODEL, RARE, DATA, GREY]
    vals = {}
    for lab, (nlp, d) in sets.items():
        a = np.atleast_2d(np.asarray(nlp, dtype=np.float64))
        vals[lab] = ((a - a.mean(1, keepdims=True)) * neglogp_scale(d)).ravel()
    lo = min(v.min() for v in vals.values()); hi = max(v.max() for v in vals.values())
    H = {}
    for lab, v in vals.items():
        edges = np.linspace(v.min(), v.max(), bins + 1)
        H[lab] = (edges, np.histogram(v, edges, density=True)[0])
    ymin = 0.5 * min(h[h > 0].min() for _, h in H.values()); ymax = 2.0 * max(h.max() for _, h in H.values())
    for k, ((lab, v), col) in enumerate(zip(vals.items(), cols)):          # one figure per dataset, same x and y limits
        edges, h = H[lab]
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.hist(v, edges, density=True, color=col, alpha=0.55)
        ax.hist(v, edges, density=True, histtype='step', color=col, lw=1.6)
        ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(ymin, ymax)
        ax.text(0.98, 0.92, f'{lab}\nstd {v.std():.3g}', transform=ax.transAxes, ha='right', va='top', fontsize=15)
        fig.tight_layout()
        _save(fig, f'{name}_{k + 1}')
    stds = {k: float(v.std()) for k, v in vals.items()}
    print('std of -log10 p / d:', {k: round(s, 5) for k, s in stds.items()})
    return stds


def plot_logp_overlay(sets, name, bins=40, colors=None, chi2_dof=None):
    """-log10 p / d histograms of several datasets on ONE axis (concentration of -log p of the generated signals:
    real data model vs an exactly Gaussian data set). sets: {label: (nlp, d)}, nlp = -log p in nats of the generated
    signals, (seeds, signals) or (signals,); d = signal length. Same convention as plot_logp_widths: each seed centred
    on its own mean (log p is known up to a constant), nats -> log10 per point (neglogp_scale(d)), density (unit area),
    log y. All sets share the SAME bin edges (so the heights are comparable); the std of each is in the legend
    (in log10 per point and in nats). No axis labels (stated in the caption / by the speaker)."""
    cols = colors or [MODEL, '#e8a33d', RARE, GREY]
    vals, nats = {}, {}
    for lab, (nlp, d) in sets.items():
        a = np.atleast_2d(np.asarray(nlp, dtype=np.float64))
        centred = a - a.mean(1, keepdims=True)
        vals[lab] = (centred * neglogp_scale(d)).ravel()
        nats[lab] = float(centred.std())
    lo = min(v.min() for v in vals.values()); hi = max(v.max() for v in vals.values())
    edges = np.linspace(lo, hi, bins + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (lab, v), col in zip(vals.items(), cols):
        ax.hist(v, edges, density=True, color=col, alpha=0.30)
        ax.hist(v, edges, density=True, histtype='step', color=col, lw=1.8,
                label=f'{lab}: std {v.std():.3g} ({nats[lab]:.0f} nats)')
    if chi2_dof:
        # exact Gaussian: -ln p = chi2_k / 2 + const (k = d real dimensions), so with the centring (mean 0) and the
        # nats -> log10-per-point factor c, v = (chi2 - k) * c / 2, chi2 = k + 2 v / c, density_v = pdf_chi2 * 2 / c
        k = chi2_dof
        c = neglogp_scale(next(iter(sets.values()))[1])
        xs = np.linspace(lo, hi, 600)
        ax.plot(xs, stats.chi2.pdf(k + 2 * xs / c, k) * 2 / c, color=DATA, ls='--', lw=1.8,
                label=rf'$\chi^2$, {k} dof (exact Gaussian): std {np.sqrt(k / 2) * c:.3g} ({np.sqrt(k / 2):.0f} nats)')
    ax.set_yscale('log')
    H = [np.histogram(v, edges, density=True)[0] for v in vals.values()]      # y range = the histograms, not the
    ax.set_ylim(0.5 * min(h[h > 0].min() for h in H), 2.0 * max(h.max() for h in H))   # far tail of the chi2 curve
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=1, borderaxespad=0, handlelength=1.4, fontsize=12)
    fig.tight_layout()
    _save(fig, name)
    print('std of -log10 p / d (nats in brackets):',
          {k: (round(float(v.std()), 5), round(nats[k], 1)) for k, v in vals.items()})
    return {k: float(v.std()) for k, v in vals.items()}


def plot_logp_gaussian(npz, name, seed_index=0, bins=80, n_scatter=3000):
    """-log p for the Gaussian with the data spectrum (gauss_spectrum/, all_seeds.npz of logp_gauss_scat2.py; the npz
    stores +log p, negated here). Left: on the Gaussian training signals, the exact (black) and the MGD (blue), each relative to the
    mean over its own law's samples (exact: training signals; MGD: generated signals), with N(0, M/2) (dashed), the
    exact law of the exact log p (chi-squared with M dof, ~Gaussian at M = 256). Right: MGD vs exact on the same
    training signals, one point per signal; on the dashed line MGD gives a signal exactly its true log p."""
    z = np.load(npz)
    M, s = int(z['M']), int(z['seeds'][seed_index])
    ex, mg = -z['exact_data'][seed_index], -z['logp_data'][seed_index]          # -log p (nats)
    sc = neglogp_scale(M)                                                       # nats -> log10 per point, d = M
    ex, mg = ex * sc, mg * sc
    sd = np.sqrt(M / 2) * sc
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    edges = np.linspace(*np.quantile(np.r_[ex, mg], [0.0005, 0.9995]), bins + 1)
    a.hist(ex, edges, density=True, histtype='step', color=DATA, lw=2, label=f'exact (std {ex.std():.4f})')
    a.hist(mg, edges, density=True, histtype='step', color=MODEL, lw=2, label=f'MGD (std {mg.std():.4f})')
    g = np.linspace(edges[0], edges[-1], 400)
    a.plot(g, stats.norm.pdf(g, 0, sd), color=GREY, ls='--', lw=1.5, label=f'N(0, M/2), std {sd:.4f}')
    a.set_title(f'Gaussian training signals ({len(ex)}), seed {s}')
    a.legend(loc='upper left')
    idx = np.random.default_rng(0).choice(len(ex), min(n_scatter, len(ex)), replace=False)
    b.scatter(ex[idx], mg[idx], s=4, alpha=0.35, color=MODEL)
    lim = [min(ex.min(), mg.min()), max(ex.max(), mg.max())]
    b.plot(lim, lim, color=DATA, ls='--', lw=1.2, label='MGD = exact')
    b.set_title(f'same signals: Pearson {stats.pearsonr(ex, mg)[0]:.3f}')
    b.legend(loc='upper left')
    fig.tight_layout()
    _save(fig, name)
    stds = z['logp_data'].std(1)
    print(f'MGD std on training signals over seeds {list(z["seeds"])}: {np.round(stds, 2)} | '
          f'exact {z["exact_data"].std(1).round(2)} | on generated: {z["logp_model"].std(1).round(2)} (theory {np.sqrt(M / 2):.2f}), nats')


def ranked_indices(logp, n=10):
    """Indices of n signals at evenly spaced quantiles of log p, from the lowest (rarest) to the highest."""
    order = np.argsort(logp)
    return order[np.round(np.linspace(0, len(order) - 1, n)).astype(int)]


def logp_support(v, bins=80, min_freq=1e-3):
    """Support of log p used by plot_ranked_signals. v: log p minus its median. Walking out from the most populated
    bin, stop at the first bin whose frequency (fraction of all signals) is below min_freq, on each side. Returns
    lo, hi (smallest / largest v inside), a, b (first / last kept bin), edges, freq. min_freq falsy: full range."""
    v = np.asarray(v, dtype=np.float64)
    counts, edges = np.histogram(v, bins)
    freq = counts / v.size
    lo, hi, a, b = v.min(), v.max(), 0, len(freq) - 1
    if min_freq:
        i0 = int(np.argmax(freq))
        ok = freq >= min_freq
        a = i0
        while a > 0 and ok[a - 1]:
            a -= 1
        b = i0
        while b < len(freq) - 1 and ok[b + 1]:
            b += 1
        inside = (v >= edges[a]) & (v <= edges[b + 1])
        lo, hi = v[inside].min(), v[inside].max()
    return lo, hi, a, b, edges, freq


def plot_ranked_signals(signals, logp, name, n=10, ncol=5, bins=80, min_freq=1e-3, truncate=True, center='mean',
                        random=False, seed=None):
    """Left: histogram of -log10 p / d over all the signals (centred at zero by its mean, or median with
    center='median'; logp = ln p as stored, negated here; d = length of the signals), normalised to unit area
    (density; min_freq still acts on the fraction of signals per bin), with an arrow at the -log p of each of n signals: for n evenly
    spaced values across the support, the signal closest to it. Right: those signals, framed with the colour of their
    arrow, -log p on top (same y-scale everywhere), from the most typical (left of the histogram, light) to the rarest
    (right, dark).
    The support is capped so isolated outliers are not picked: walking out from the most populated bin, it stops
    at the first bin whose frequency is below min_freq (on each side). Signals beyond the cap are still in the
    histogram, shaded. min_freq=None (or 0) uses the full range, as before. Note the frequency is per bin, so the
    cap depends on `bins`.
    truncate=True (with min_freq): the histogram is also cut at the cap: only the bins inside it are drawn (no shaded
    outer part, no threshold line); truncate=False keeps the whole histogram with the outside shaded.
    random=True: instead of the single closest signal, each of the n slots takes a random signal among those within
    half a slot spacing of its target (inside the support), so the arrows stay evenly spread but the signals change
    at every call (seed=None: new draw each time; an int fixes it). The chosen indices are printed."""
    from matplotlib import gridspec
    v = np.asarray(logp, dtype=np.float64)
    v = -(v - (v.mean() if center == 'mean' else np.median(v)))      # -log p, centred at zero (nats)
    lo, hi, a, b, edges, freq = logp_support(v, bins, min_freq)      # support cut on the fraction per bin
    if min_freq:
        print(f'ranked signals: support capped at freq >= {min_freq:g} -> [{lo:+.1f}, {hi:+.1f}] nats; '
              f'{(v < lo).sum()} below, {(v > hi).sum()} above of {v.size} signals left out')
    sc = neglogp_scale(_np(signals).shape[-1])                       # nats -> log10 per point, d = signal length
    v, lo, hi, edges = v * sc, lo * sc, hi * sc, edges * sc
    binw = edges[1] - edges[0]
    dens = freq / binw                                               # unit area
    targets = np.linspace(lo, hi, n)
    if random:
        rng = np.random.default_rng(seed)
        half = 0.5 * (hi - lo) / max(n - 1, 1)                           # half a slot spacing
        idx = []
        for t in targets:
            cand = np.flatnonzero((np.abs(v - t) <= half) & (v >= lo) & (v <= hi))
            idx.append(int(rng.choice(cand)) if len(cand) else int(np.argmin(np.abs(v - t))))
        idx = np.array(idx)
        print('ranked signals: random choice, indices', idx.tolist())
    else:
        idx = np.array([np.argmin(np.abs(v - t)) for t in targets])
    s = _np(signals)[idx]
    colors = plt.get_cmap('plasma')(np.linspace(0.85, 0.0, n))       # light (typical, left) -> dark (rare, right)
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(17, 2.4 * nrow + 1.2))
    gs = gridspec.GridSpec(nrow, ncol + 2, figure=fig, width_ratios=[2.4, 0.35] + [1] * ncol, wspace=0.08, hspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_yscale('log')
    if min_freq and truncate:
        ax.stairs(dens[a:b + 1], edges[a:b + 2], fill=True, color=GREY, alpha=0.6)
        ax.set_xlim(edges[a], edges[b + 1])
        ax.set_ylim(bottom=min_freq / binw / 2)
    else:
        ax.stairs(dens, edges, fill=True, color=GREY, alpha=0.6)
    if min_freq and not truncate:
        ax.axhline(min_freq / binw, color=GREY, ls='--', lw=1)
        for x0, x1 in ((edges[0], lo), (hi, edges[-1])):
            if x1 > x0:
                ax.axvspan(x0, x1, color=GREY, alpha=0.12, lw=0)
    for c, x in zip(colors, v[idx]):
        ax.annotate('', xy=(x, 0), xytext=(x, -0.08), xycoords=ax.get_xaxis_transform(),
                    arrowprops=dict(arrowstyle='->', color=c, lw=2), annotation_clip=False)
    ax.tick_params(axis='x', pad=30)                                   # tick labels below the arrows
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
        a.set_title(f'{v[idx[k]]:+.3f}', fontsize=16)
    _save(fig, name)


def plot_energy_bars(C, name, title=None, Q=None):
    """C = {family: array (seeds, signals)} of theta_F . phi_F(x) (contribution to log p, as stored), TOTAL last;
    shown as contributions to -log p (negated). Three panels: mean over signals (level only: a constant added to
    phi_F changes it without changing p), std across signals, std / std of -log p. Error bars = std over seeds
    (none for one seed)."""
    fams = list(C)
    mean = np.array([-C[f].mean(1) for f in fams])                                # (F, seeds), -log p
    std = np.array([C[f].std(1, ddof=1) for f in fams])
    norm = std / std[-1:]
    n_seeds = mean.shape[1]
    x = np.arange(len(fams))
    colors = [MODEL] * (len(fams) - 1) + [DATA]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, gridspec_kw=dict(hspace=0.3))
    panels = ((mean, 'mean (nats)', dict(alpha=0.5)),
              (std, 'std (nats)', {}),
              (norm, 'std / std of\n' + r'total $-\log p$', dict(hatch='//', edgecolor='white')))
    for ax, (v, lab, style) in zip(axes, panels):
        err = v.std(1, ddof=1) if n_seeds > 1 else None
        ax.bar(x, v.mean(1), 0.6, yerr=err, capsize=3, color=colors, **style)
        ax.axhline(0, color=DATA, lw=0.6)
        ax.set_ylabel(lab)
        ax.grid(axis='x', visible=False)
    axes[2].axhline(1, color=GREY, ls='--', lw=1)
    axes[2].set_xticks(x, [family_label(f, Q) if f != 'TOTAL' else r'total $-\log p$' for f in fams],
                       rotation=30, ha='right')
    if title:
        axes[0].set_title(title)
    fig.align_ylabels(axes)
    fig.tight_layout()
    _save(fig, name)


# ------------------------------------------------- draft Fig. 6 statistics ----
# Formulas: turbulence/paper_figs_ensemble.curves (= codes/paper_figures.make_all_panels, the draft's Fig. 6),
# computed once by fig6_stats, drawn in the talk style by plot_fig6.
FIG6_PANELS = {
    'marg': 'marginal PDF', 'psd': 'power spectrum', 'sf': 'structure functions', 'flat_slope': 'flatness slope',
    'inc': 'increment PDFs', 'C4': 'C_4', 'C6': 'C_6', 'C42': 'C_{4,2}', 'skew': 'skewness'}
FIG6_GROUPS = (('marg', 'psd', 'sf'), ('flat_slope', 'inc', 'skew'), ('C4', 'C6', 'C42'))   # 3 figures of 3 panels


def _pfe():
    import sys
    from pathlib import Path
    turb = Path(__file__).resolve().parent.parent / 'turbulence'
    if str(turb) not in sys.path:
        sys.path.insert(0, str(turb))
    import paper_figs_ensemble
    return paper_figs_ensemble


def fig6_stats(data, model, tau_star=20, band=None, n_batches=10, marg_range=1.5, marg_bins=301, skew_mode='power'):
    """Every curve of the draft's Fig. 6 for the data and the model, with bands; pass the result to plot_fig6.
    data: (n, T). model: (n, T) or a list of (n, T) arrays, one per seed. band (None = module BAND):
      'batch': line = all signals, band = min / max over n_batches disjoint batches (data and model);
      'seeds': model line = mean over seeds, band = +- std (ddof = 1); data band stays min / max over batches.
    Bins and lags as paper_figs_ensemble (marginal bins +- marg_range * max|data|). skew_mode 'power' (default, as the
    draft) = skewness of (x(t+tau)-x(t)) x(t); 'exact' = skewness of the energy increment x(t+tau)^2/2 - x(t)^2/2."""
    pfe = _pfe()
    band = BAND if band is None else band
    assert band in ('batch', 'seeds'), band
    as_t = lambda a: torch.as_tensor(_np(a), dtype=torch.float32).reshape(len(_np(a)), 1, -1)
    d = _np(data)
    ms = [_np(m) for m in (model if isinstance(model, (list, tuple)) else [model])]
    if band == 'seeds' and len(ms) < 2:
        print("band='seeds' needs the model as a list of >= 2 arrays: falling back to 'batch'")
        band = 'batch'
    L = marg_range * np.abs(d).max()
    bins = np.linspace(-L, L, marg_bins)
    cur = lambda a: pfe.curves(as_t(a), bins, tau_star, skew_mode=skew_mode)
    keys = list(FIG6_PANELS)

    def batched(a):
        full = cur(a)
        per = [cur(a[idx]) for idx in np.array_split(np.arange(len(a)), n_batches)]
        return {k: (full[k], np.min([p[k] for p in per], 0), np.max([p[k] for p in per], 0)) for k in keys}, full['marg_out']

    out_d, out_d_mass = batched(d)
    if band == 'seeds':
        per = [cur(m) for m in ms]
        out_m = {}
        for k in keys:
            Y = np.stack([p[k] for p in per])
            y, sd = Y.mean(0), Y.std(0, ddof=1)
            out_m[k] = (y, y - sd, y + sd)
        mass = max(p['marg_out'] for p in per)
        lab_d, lab_m = 'data (batches)', f'model (mean ± std, {len(ms)} seeds)'
    else:
        out_m, mass = batched(ms[0] if len(ms) == 1 else np.concatenate(ms))
        lab_d, lab_m = 'data', 'model'
    print(f'fig6_stats: band {band}, skewness {skew_mode}, tau* = {tau_star}; mass outside the marginal bins: '
          f'data {out_d_mass:.1e}, model {mass:.1e}')
    return dict(data=out_d, model=out_m, lab_d=lab_d, lab_m=lab_m, T=d.shape[-1], tau_star=tau_star, bins=bins,
                band=band, skew_mode=skew_mode)


def _fig6_panel(ax, st, key, tau_eta=None, shift_decades=1.0, legend=False):
    pfe = _pfe()
    D, Mo = st['data'][key], st['model'][key]
    T, ts = st['T'], st['tau_star']
    max_tau = T // 2
    u = 1.0 if tau_eta is None else tau_eta                     # lags shown in samples, or in tau_eta
    tl = r'$\tau$' if tau_eta is None else r'$\tau/\tau_\eta$'
    pos = lambda c: tuple(np.where(v > 0, v, np.nan) for v in c)
    clip = lambda c, f: (c[0], np.maximum(c[1], f * c[0]), c[2])            # band floor on log axes

    def pair(x, dd, mm, points=True):                           # points: markers at the computed lags
        _curve(ax, x, *dd, DATA, st['lab_d'], MK_D if points else None)
        _curve(ax, x, *mm, MODEL, st['lab_m'], MK_M if points else None)

    if key == 'marg':
        b = st['bins']
        c = 0.5 * (b[1:] + b[:-1])
        pair(c, pos(D), pos(clip(Mo, 1e-2)), points=False)
        keep = c[(D[0] > pfe.Y_FLOOR) | (Mo[0] > pfe.Y_FLOOR)]
        pad = 0.05 * (keep.max() - keep.min())
        ax.set_xlim(keep.min() - pad, keep.max() + pad)
        ax.set_yscale('log')
        ax.set_ylim(pfe.Y_FLOOR, None)
        ax.set_xlabel(r'$x/\sigma(x)$')
        ax.set_ylabel('PDF')
    elif key == 'psd':
        om = np.fft.rfftfreq(T)
        k = (om > 0) & (om <= pfe.OMEGA_MAX)
        pair(om[k], tuple(v[k] for v in D), tuple(v[k] for v in clip(Mo, 1e-2)), points=False)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$\omega$ (cycles / sample)')
        ax.set_ylabel('power spectrum')
    elif key == 'sf':
        t = np.arange(1, max_tau) / u
        for i, p in enumerate(pfe.SF_ORDERS):
            sh = 10.0 ** (-shift_decades * i)
            _curve(ax, t, *(v[i] * sh for v in D), DATA, st['lab_d'] if i == 0 else None, MK_D)
            _curve(ax, t, *(v[i] * sh for v in clip(Mo, 1e-2)), MODEL, st['lab_m'] if i == 0 else None, MK_M)
            ax.text(t[0] * 0.85, D[0][i][0] * sh, f'$p = {p}$', ha='right', va='center', fontsize=15)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(t[0] * 0.25, None)                                   # room for the p labels on the left
        ax.set_xlabel(tl)
        ax.set_ylabel(r'$S_p(\tau)$' + (' (shifted)' if shift_decades else ''))
    elif key == 'flat_slope':
        t = pfe.log_spaced_taus(max_tau, pfe.FLAT_POINTS) / u
        pair(t, D, Mo)
        ax.axhline(0, color=GREY, ls=':', lw=1.2)
        ax.set_xscale('log')
        ax.set_xlabel(tl)
        ax.set_ylabel(r'$d\log F_4 / d\log\tau$')
    elif key == 'inc':
        b = np.linspace(*pfe.INC_XLIM, pfe.N_BINS)
        c = 0.5 * (b[1:] + b[:-1])
        for i, tau in enumerate(pfe.INC_TAUS):
            sh = 10.0 ** (-i)
            _curve(ax, c, *pos(tuple(v[i] * sh for v in D)), DATA, st['lab_d'] if i == 0 else None)
            _curve(ax, c, *pos(tuple(v[i] * sh for v in clip(Mo, 1e-2))), MODEL, st['lab_m'] if i == 0 else None)
            lab = rf'$\tau = {tau}$' if tau_eta is None else rf'$\tau/\tau_\eta = {tau / u:.2g}$'
            ax.text(0, sh * 1.45, lab, ha='center', va='bottom', fontsize=13,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.85))
        ax.set_yscale('log')
        ax.set_xlim(pfe.INC_XLIM)
        ax.set_ylim(10.0 ** (-(len(pfe.INC_TAUS) + 1)), 12.0)
        ax.set_xlabel(r'$\delta_\tau x\, /\, \sigma(\delta_\tau x)$')
        ax.set_ylabel('PDF (peak 1, shifted)')
    elif key in ('C4', 'C6', 'C42'):
        t = np.arange(1, max_tau) / u
        pair(t, D, Mo)
        ax.axvline(ts / u, color=GREY, ls=':', lw=1.4)
        ax.axhline(1, color=GREY, ls='-', lw=0.6)
        ax.set_xscale('log')
        ax.set_xlabel(tl)
        sym = {'C4': 'C_4', 'C6': 'C_6', 'C42': 'C_{4,2}'}[key]          # draft names: C_4 = C_{2,2}, C_6 = C_{3,3}
        ax.set_ylabel(rf'${sym}(\tau, \tau^\star)$')
        ax.set_title(rf'$\tau^\star = {ts}$' if tau_eta is None else rf'$\tau^\star/\tau_\eta = {ts / u:.2g}$')
    elif key == 'skew':
        t = pfe.log_spaced_taus(max_tau, num_points=pfe.SKEW_POINTS) / u
        pair(t, D, Mo)
        ax.axhline(0, color=GREY, ls=':', lw=1.2)
        ax.set_xscale('log')
        ax.set_xlabel(tl)
        ax.set_ylabel(r'skewness of $\delta_\tau (x^2/2)$' if st['skew_mode'] == 'exact' else r'skewness of $x\,\delta_\tau x$')
    else:
        raise KeyError(key)
    if legend:
        ax.legend(loc='best')


def plot_fig6(st, name, groups=FIG6_GROUPS, tau_eta=None, shift_decades=1.0, panel=(5.6, 4.6)):
    """Draft Fig. 6 in the talk style: one figure per group of panels (default 3 figures of 3, as the draft's
    'split into 3' note), saved as <name>_1, <name>_2, ...; a single key ('sf') or a tuple of keys draws one figure.
    st = fig6_stats(...). tau_eta: lags in tau_eta (Lagrangian) instead of samples. Keys: see FIG6_PANELS."""
    if isinstance(groups, str):
        groups = ((groups,),)
    elif groups and isinstance(groups[0], str):
        groups = (tuple(groups),)
    for g, keys in enumerate(groups, 1):
        fig, axes = plt.subplots(1, len(keys), figsize=(panel[0] * len(keys), panel[1]), squeeze=False)
        for i, (ax, k) in enumerate(zip(axes[0], keys)):
            _fig6_panel(ax, st, k, tau_eta=tau_eta, shift_decades=shift_decades)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.legend(h, l, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))         # above the panels, never on a curve
        _save(fig, f'{name}_{g}' if len(groups) > 1 else name)
