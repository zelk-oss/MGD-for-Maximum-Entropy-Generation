"""
Figures reproducing the single-component statistics of Li et al. 2024 (Nat. Mach. Intell. 6,
393-403, "Synthetic Lagrangian turbulence by generative diffusion models"), PDF in turbulence/.

Each panel_li_* function makes ONE standalone matplotlib figure comparing data (dotted) and MGD
synthesis (solid), same style as codes/paper_figures.py:
    fig1a  standardized PDFs of delta_tau V at tau/tau_eta = 1, 2, 5, 100   (no error bars, as paper)
    fig2   standardized PDF of the acceleration a = delta_1 V               (no error bars, as paper)
    fig2i  acceleration correlation C_tau = <a(t+tau) a(t)>                 (no error bars, as paper)
    fig4a  structure functions S_p(tau) = <(delta_tau V)^p>, p = 2, 4, 6 (paper) + 8
    fig4b  generalized flatness F_p(tau) = S_p / S_2^(p/2), p = 4, 6, 8
    fig5   local scaling exponent zeta(p, tau) = dlog S_p / dlog S_2, p = 4 (6, 8 also made)
    table2 E = <V^2>, A = <a^2>
The 3-component statistics (mixed flatness F^(4,ij), Sigma_V, Sigma_A) are not reproduced.

Error bars as in the paper: the curve is the statistic on the whole set, the band spans the
min/max of the same statistic over n_batches disjoint batches of trajectories (rows of x),
for data and synthesis alike. Fig 4: 10 batches. Fig 5: 30 batches for the data (DNS) and
10 for the synthesis (DM-1c), as in its caption.

Units: one raw sample = dt_s ~ 0.1 tau_eta (Li et al. Table 1: dt_s = 2.25e-3, tau_eta = 2.1e-2),
so TAU_ETA = 9.33 raw samples. A run whose x1 was coarse-grained by run_SDE (subseries_len ->
target_len) has one sample = step dt_s, step = subseries_len / target_len: the panels take
tau_eta = TAU_ETA / step (make_li_panels does it from `step`). a is the one-step increment of x
(a = delta_{dt_s} V for step 1, the increment of the coarse-grained signal otherwise).
Lags run over 1 .. min(max_lag, T-2), max_lag = MAX_LAG / step = the paper's zeta grid tau <= 1024 dt_s.

Input: data, synth of shape (B, 1, T), torch or numpy, e.g. x1 and the MGD samples of a run.
x1 is the run_SDE windows (periodized by split_periodize_reshape), so large-tau values are
not the raw-trajectory ones of the paper (shift a*tau per window); data vs synth is like for like.

If save_dir is None -> plt.show(). If save_dir is given -> each panel is saved as
save_dir/<name>.<fmt> and the figure is closed (no display).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from check_moments import logder
from paper_figures import STYLE, COLOR_1, LS_DATA, LS_SYNTH, _to_np, _apply_style, _finalize

TAU_ETA = 2.1e-2 / 2.25e-3     # Kolmogorov time in samples (Li et al. Table 1), ~ 9.33
MAX_LAG = 1024                 # paper's zeta grid: tau = 1 .. 1024
ORDERS = (2, 4, 6, 8)          # even orders only (odd S_p ~ 0 by symmetry, as in the paper)
BAND_ALPHA = 0.3


# ------------------------------------------------------------ moments cache ----
class PerSampleMomentCache:
    """Per-trajectory increment moments m[b, p, tau] = mean_t (delta_tau x_b)^p, for ORDERS.

    Batch statistics are means of m over a subset of rows (every row has the same number of
    pairs at a given lag), so the whole set and every batching come from one pass over the lags.
    Keyed by id(data) like check_moments.sf_cache; the data object is kept in the cache so its
    id cannot be reused by another tensor.
    """
    def __init__(self):
        self.cache = {}

    def get_or_compute(self, data, max_lag=MAX_LAG):
        key = (id(data), max_lag)
        if key not in self.cache:
            x = torch.as_tensor(_to_np(data) if not torch.is_tensor(data) else data).to(torch.float64)
            x = x.reshape(x.shape[0], -1, x.shape[-1])
            # stop at T-2: on periodized windows x(T-1) = x(0), so every lag T-1 increment is 0
            taus = np.arange(1, min(max_lag, x.shape[-1] - 2) + 1)
            print(f"Computing per-sample increment moments for {len(taus)} lags...")
            m = torch.zeros(x.shape[0], len(ORDERS), len(taus), dtype=torch.float64, device=x.device)
            for i, tau in enumerate(taus):
                d2 = (x[..., tau:] - x[..., :-tau]) ** 2
                for j, p in enumerate(ORDERS):
                    m[:, j, i] = (d2 ** (p // 2)).mean(dim=(1, 2))
            self.cache[key] = (data, taus, m.cpu().numpy())
        return self.cache[key][1], self.cache[key][2]


moment_cache = PerSampleMomentCache()


def _batches(m, n_batches):
    """Row means of m over n_batches disjoint contiguous batches: (n_batches, P, n_taus)."""
    return np.stack([m[idx].mean(axis=0) for idx in np.array_split(np.arange(m.shape[0]), n_batches)])


def structure_functions(data, n_batches=10, max_lag=MAX_LAG):
    """S_p(tau) for ORDERS on the whole set, and per batch.

    Returns taus, S (P, n_taus), S_b (n_batches, P, n_taus).
    """
    taus, m = moment_cache.get_or_compute(data, max_lag)
    return taus, m.mean(axis=0), _batches(m, n_batches)


def _flat(S, p):
    k = ORDERS.index(p)
    return S[..., k, :] / S[..., 0, :] ** (p / 2)


def _zeta(S, taus, p):
    """zeta(p, tau) = [dlog S_p/dlog tau] / [dlog S_2/dlog tau], second-order differences on
    the unit-spaced tau grid (check_moments.logder), as in the paper."""
    k = ORDERS.index(p)
    return logder(S[k], taus) / logder(S[0], taus)


def _band(ax, x, lo, hi, color):
    ax.fill_between(x, lo, hi, color=color, alpha=BAND_ALPHA, lw=0)


# ---------------------------------------------------- (fig 1a) increment PDFs ----
def _std_pdf(z, bins, smooth, smooth_sigma):
    h, _ = np.histogram(z / (z.std() + 1e-12), bins=bins, density=True)
    if smooth:
        h = gaussian_filter1d(h, smooth_sigma)
    return np.where(h > 0, h, np.nan)


def panel_li_fig1a_increment_pdfs(data, synth, label="Lagrangian",
                                  tau_ratios=(1, 2, 5, 100), tau_eta=TAU_ETA, shift_decades=3,
                                  n_bins=201, smooth=True, smooth_sigma=1.5,
                                  save_dir=None, fmt='pdf', dpi=200):
    """Standardized PDFs of delta_tau V at tau/tau_eta in tau_ratios, shifted down by
    10^(-shift_decades) per curve (paper: 'vertically shifted for the sake of presentation').
    Ratios whose lag does not fit in the signal (tau > T-2) are skipped, with a message."""
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    d_np, s_np = _to_np(data), _to_np(synth)
    T = min(d_np.shape[-1], s_np.shape[-1])
    reachable = [r for r in tau_ratios if int(round(r * tau_eta)) <= T - 2]
    if len(reachable) < len(tau_ratios):
        print(f"fig1a: tau/tau_eta {sorted(set(tau_ratios) - set(reachable))} skipped "
              f"(largest lag T-2 = {T - 2} samples = {(T - 2) / tau_eta:.1f} tau_eta)")
    for k, r in enumerate(reachable):
        tau = max(1, int(round(r * tau_eta)))
        d_inc = (d_np[..., tau:] - d_np[..., :-tau]).reshape(-1)
        s_inc = (s_np[..., tau:] - s_np[..., :-tau]).reshape(-1)
        z_max = max(np.abs(d_inc).max() / d_inc.std(), np.abs(s_inc).max() / s_inc.std())
        bins = np.linspace(-z_max, z_max, n_bins)
        ctr = 0.5 * (bins[1:] + bins[:-1])
        shift = 10.0 ** (-shift_decades * k)
        ax.plot(ctr, _std_pdf(d_inc, bins, smooth, smooth_sigma) * shift, ls=LS_DATA, color=COLOR_1,
                lw=STYLE['lw'] * 0.7, label=f'{label} data' if k == 0 else None)
        ax.plot(ctr, _std_pdf(s_inc, bins, smooth, smooth_sigma) * shift, ls=LS_SYNTH, color=COLOR_1,
                lw=STYLE['lw'] * 0.7, label=f'{label} synth' if k == 0 else None)
        ax.text(0.0, 0.5 * shift, rf'$\tau/\tau_\eta={r}$',
                ha='center', va='bottom', fontsize=STYLE['fontsize'] * 0.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='black', linewidth=1.2))
    ax.set_yscale('log')
    _apply_style(ax, r'$\delta_\tau V/\sigma(\delta_\tau V)$', r'PDF$(\delta_\tau V)$ (shifted)')
    _finalize(fig, save_dir, 'li_fig1a_increment_pdfs', fmt, dpi)


# ------------------------------------------------- (fig 2) acceleration PDF ----
def panel_li_fig2_acceleration_pdf(data, synth, label="Lagrangian", n_bins=201,
                                   smooth=True, smooth_sigma=1.5,
                                   save_dir=None, fmt='pdf', dpi=200):
    """Standardized PDF of a = delta_1 V (dt_s = 1): PDF(a) sigma(a) vs a/sigma(a)."""
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    d_np, s_np = _to_np(data), _to_np(synth)
    a_d = (d_np[..., 1:] - d_np[..., :-1]).reshape(-1)
    a_s = (s_np[..., 1:] - s_np[..., :-1]).reshape(-1)
    z_max = max(np.abs(a_d).max() / a_d.std(), np.abs(a_s).max() / a_s.std())
    bins = np.linspace(-z_max, z_max, n_bins)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    ax.plot(ctr, _std_pdf(a_d, bins, smooth, smooth_sigma), ls=LS_DATA, color=COLOR_1,
            lw=STYLE['lw'], label=f'{label} data')
    ax.plot(ctr, _std_pdf(a_s, bins, smooth, smooth_sigma), ls=LS_SYNTH, color=COLOR_1,
            lw=STYLE['lw'], label=f'{label} synth')
    ax.set_yscale('log')
    _apply_style(ax, r'$a/\sigma(a)$', r'PDF$(a)\,\sigma(a)$')
    _finalize(fig, save_dir, 'li_fig2_acceleration_pdf', fmt, dpi)


# ------------------------------------- (fig 2 inset) acceleration correlation ----
def acceleration_correlation(data, max_lag=MAX_LAG):
    """C_tau = <a(t+tau) a(t)>, a = delta_1 x, tau = 0 .. max_lag (computed directly, not
    from S_2). Returns taus, C."""
    x = torch.as_tensor(_to_np(data) if not torch.is_tensor(data) else data).to(torch.float64)
    a = x[..., 1:] - x[..., :-1]
    taus = np.arange(0, min(max_lag, a.shape[-1] - 1) + 1)
    C = np.array([(a[..., tau:] * a[..., :a.shape[-1] - tau]).mean().item() for tau in taus])
    return taus, C


def panel_li_fig2_inset_acceleration_correlation(data, synth, label="Lagrangian", max_lag=MAX_LAG,
                                                 tau_eta=TAU_ETA, save_dir=None, fmt='pdf', dpi=200):
    """C_tau vs tau/tau_eta, lin-log (tau = 0 is not shown on the log axis)."""
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    taus, C_d = acceleration_correlation(data, max_lag)
    _, C_s = acceleration_correlation(synth, max_lag)
    ax.plot(taus[1:] / tau_eta, C_d[1:], ls=LS_DATA, color=COLOR_1, lw=STYLE['lw'], label=f'{label} data')
    ax.plot(taus[1:] / tau_eta, C_s[1:], ls=LS_SYNTH, color=COLOR_1, lw=STYLE['lw'], label=f'{label} synth')
    ax.axhline(0, color='black', lw=1)
    ax.set_xscale('log')
    _apply_style(ax, r'$\tau/\tau_\eta$', r'$C_\tau$')
    _finalize(fig, save_dir, 'li_fig2_inset_acceleration_correlation', fmt, dpi)


# --------------------------------------------- (fig 4a) structure functions ----
def panel_li_fig4a_structure_functions(data, synth, label="Lagrangian", orders=(2, 4, 6, 8),
                                       n_batches=10, max_lag=MAX_LAG, tau_eta=TAU_ETA,
                                       save_dir=None, fmt='pdf', dpi=200):
    # orders distinguished by alpha only (1.0 .. 0.4), as panel_c_structure_functions
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    taus, S_d, Sb_d = structure_functions(data, n_batches, max_lag)
    _, S_s, Sb_s = structure_functions(synth, n_batches, max_lag)
    t = taus / tau_eta
    for k, (p, alpha) in enumerate(zip(orders, np.linspace(1.0, 0.4, len(orders)))):
        j = ORDERS.index(p)
        _band(ax, t, Sb_d[:, j].min(0), Sb_d[:, j].max(0), COLOR_1)
        _band(ax, t, Sb_s[:, j].min(0), Sb_s[:, j].max(0), COLOR_1)
        ax.plot(t, S_d[j], ls=LS_DATA, color=COLOR_1, lw=STYLE['lw'], alpha=alpha,
                label=f'{label} data' if k == 0 else None)
        ax.plot(t, S_s[j], ls=LS_SYNTH, color=COLOR_1, lw=STYLE['lw'], alpha=alpha,
                label=f'{label} synth' if k == 0 else None)
    ax.set_xscale('log')
    ax.set_yscale('log')
    _apply_style(ax, r'$\tau/\tau_\eta$',
                 rf'$S^{{(p)}}_\tau$, $p={",".join(map(str, orders))}$')
    _finalize(fig, save_dir, 'li_fig4a_structure_functions', fmt, dpi)


# ---------------------------------------------- (fig 4b) generalized flatness ----
def panel_li_fig4b_flatness(data, synth, label="Lagrangian", orders=(4, 6, 8),
                            n_batches=10, max_lag=MAX_LAG, tau_eta=TAU_ETA,
                            save_dir=None, fmt='pdf', dpi=200):
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    taus, S_d, Sb_d = structure_functions(data, n_batches, max_lag)
    _, S_s, Sb_s = structure_functions(synth, n_batches, max_lag)
    t = taus / tau_eta
    for k, (p, alpha) in enumerate(zip(orders, np.linspace(1.0, 0.4, len(orders)))):
        Fb_d, Fb_s = _flat(Sb_d, p), _flat(Sb_s, p)
        _band(ax, t, Fb_d.min(0), Fb_d.max(0), COLOR_1)
        _band(ax, t, Fb_s.min(0), Fb_s.max(0), COLOR_1)
        ax.plot(t, _flat(S_d, p), ls=LS_DATA, color=COLOR_1, lw=STYLE['lw'], alpha=alpha,
                label=f'{label} data' if k == 0 else None)
        ax.plot(t, _flat(S_s, p), ls=LS_SYNTH, color=COLOR_1, lw=STYLE['lw'], alpha=alpha,
                label=f'{label} synth' if k == 0 else None)
    ax.set_xscale('log')
    ax.set_yscale('log')
    _apply_style(ax, r'$\tau/\tau_\eta$',
                 rf'$F^{{(p)}}_\tau$, $p={",".join(map(str, orders))}$')
    _finalize(fig, save_dir, 'li_fig4b_flatness', fmt, dpi)


# ------------------------------------------- (fig 5) local scaling exponent ----
def panel_li_fig5_local_exponent(data, synth, label="Lagrangian", p=4,
                                 n_batches_data=30, n_batches_synth=10, max_lag=MAX_LAG, ylim=None,
                                 tau_eta=TAU_ETA, save_dir=None, fmt='pdf', dpi=200):
    """zeta(p, tau), lin-log, dotted line at the non-intermittent value p/2 (S_p ~ S_2^(p/2)).

    ylim defaults to (p/4, 1.1 p/2), i.e. the paper's (1.0, 2.2) at p = 4: zeta diverges where
    dlog S_2/dlog tau = 0, which on periodized windows happens near tau = T/2 (S_2 peaks there).
    """
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    taus, S_d, Sb_d = structure_functions(data, n_batches_data, max_lag)
    _, S_s, Sb_s = structure_functions(synth, n_batches_synth, max_lag)
    t = taus / tau_eta
    zb_d = np.stack([_zeta(S, taus, p) for S in Sb_d])
    zb_s = np.stack([_zeta(S, taus, p) for S in Sb_s])
    _band(ax, t, zb_d.min(0), zb_d.max(0), COLOR_1)
    _band(ax, t, zb_s.min(0), zb_s.max(0), COLOR_1)
    ax.plot(t, _zeta(S_d, taus, p), ls=LS_DATA, color=COLOR_1, lw=STYLE['lw'], label=f'{label} data')
    ax.plot(t, _zeta(S_s, taus, p), ls=LS_SYNTH, color=COLOR_1, lw=STYLE['lw'], label=f'{label} synth')
    ax.axhline(p / 2, color='grey', ls=':', lw=2)
    ax.set_ylim(ylim if ylim is not None else (p / 4, 1.1 * p / 2))
    ax.set_xscale('log')
    _apply_style(ax, r'$\tau/\tau_\eta$', rf'$\zeta({p},\tau)$')
    _finalize(fig, save_dir, f'li_fig5_local_exponent_p{p}', fmt, dpi)


# ------------------------------------------------------------------ table 2 ----
def li_table2(data, synth):
    """E = <V^2>, A = <a^2> (a = delta_1 V) for data and synth; printed and returned.
    Paper (DNS, raw trajectories, physical units): E = 3.0, A = 1.7e-3."""
    out = {}
    for name, x in (('data', data), ('synth', synth)):
        x_np = _to_np(x).astype(np.float64)
        out[name] = dict(E=float(np.mean(x_np ** 2)), A=float(np.mean(np.diff(x_np, axis=-1) ** 2)))
    print(f"{'':6s} {'E':>10s} {'A':>10s}")
    for name, v in out.items():
        print(f"{name:6s} {v['E']:10.4g} {v['A']:10.4g}")
    return out


# --------------------------------------------------------------- driver ----
def make_li_panels(data, synth, label="Lagrangian", step=1,
                   save_dir=None, fmt='png', dpi=200):
    """All Li et al. 2024 single-component panels + Table 2 for one (data, synth) pair.
    step = raw samples (dt_s) per sample of x, i.e. subseries_len / target_len of the run."""
    kwargs = dict(label=label, save_dir=save_dir, fmt=fmt, dpi=dpi)
    tau_eta, max_lag = TAU_ETA / step, MAX_LAG // step
    print(f"step {step}: tau_eta = {tau_eta:.2f} samples, max_lag = {max_lag}")
    panel_li_fig1a_increment_pdfs(data, synth, tau_eta=tau_eta, **kwargs)
    panel_li_fig2_acceleration_pdf(data, synth, **kwargs)
    panel_li_fig2_inset_acceleration_correlation(data, synth, max_lag=max_lag, tau_eta=tau_eta, **kwargs)
    panel_li_fig4a_structure_functions(data, synth, max_lag=max_lag, tau_eta=tau_eta, **kwargs)
    panel_li_fig4b_flatness(data, synth, max_lag=max_lag, tau_eta=tau_eta, **kwargs)
    for p in (4, 6, 8):
        panel_li_fig5_local_exponent(data, synth, p=p, max_lag=max_lag, tau_eta=tau_eta, **kwargs)
    return li_table2(data, synth)


# Usage (codes/ on sys.path, as for paper_figures):
# from paper_figures_li import make_li_panels
# make_li_panels(x1, synth, save_dir=None)                                # display
# make_li_panels(x1, synth, save_dir='figs/li2024', fmt='pdf')            # save
