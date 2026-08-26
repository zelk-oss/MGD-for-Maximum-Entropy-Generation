"""
Per-panel figure generator for the turbulence diagnostics paper figure.

Each panel_* function makes ONE standalone matplotlib figure comparing one
OR two (data, synth) pairs -- generically "Type 1" (blue) and "Type 2"
(orange), with training data as dotted lines and MGD synthesis as solid
lines -- so the same code works whether you're showing a single data type
(e.g. just Lagrangian turbulence, or just jets) or overlaying two. Pass only
data1/synth1 for a single-type figure; add data2/synth2 (and label1/label2)
to overlay a second type on the same panels. Call make_all_panels() once per
group; run it again for a different dataset by calling it again with a
different save_dir.

Depends on (assumed already imported / defined in the notebook):
    second_order_structure_function, flatness, logder, C_pq_structure,
    leverage_correlation

If save_dir is None -> plt.show(). If save_dir is given -> each panel is
saved as save_dir/panel_x.<fmt> and the figure is closed (no display).
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter1d

from check_moments import C_ORIG, C_SYNTH, C_pq_structure, second_order_structure_function, \
    flatness, logder, skewness_vs_tau, log_spaced_taus

# ---------------------------------------------------------------- style ----
STYLE = dict(fontsize=24, lw=4.0, grid_alpha=0.3, figsize=(7, 6))
COLOR_1 = 'tab:blue'
COLOR_2 = 'tab:orange'
LS_DATA = ':'          # training data
LS_SYNTH = '-'         # MGD synthesis


def _to_np(x):
    return x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x)


def _build_groups(data1, synth1, data2, synth2, label1, label2):
    """(data, synth, color, label) tuples for whichever of the 1 or 2 data
    types were actually passed in -- data2/synth2 are optional, so a
    single-type figure just omits the second entry entirely."""
    groups = [(data1, synth1, COLOR_1, label1)]
    if data2 is not None:
        groups.append((data2, synth2, COLOR_2, label2))
    return groups


def _apply_style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=STYLE['fontsize'])
    ax.set_ylabel(ylabel, fontsize=STYLE['fontsize'])
    ax.tick_params(labelsize=STYLE['fontsize'] * 0.8)
    ax.grid(True, alpha=STYLE['grid_alpha'])


def _finalize(fig, save_dir, name, fmt='pdf', dpi=200):
    fig.tight_layout()
    if save_dir is None:
        plt.show()
    else:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'{name}.{fmt}'), dpi=dpi, bbox_inches='tight')
        plt.close(fig)


def _legend(ax, ncol=1):
    ax.legend(fontsize=STYLE['fontsize'] * 0.6, frameon=False, ncol=ncol)


# ------------------------------------------------------------- (a) PDFs ----
def panel_a_marginals(data1, synth1, data2=None, synth2=None,
                       label1="Type 1", label2="Type 2", n_bins=201,
                       smooth=True, smooth_sigma=1.5, y_floor=1e-4,
                       save_dir=None, fmt='pdf', dpi=200):
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    x_lo, x_hi = np.inf, -np.inf
    for data, synth, color, label in _build_groups(data1, synth1, data2, synth2, label1, label2):
        d, s = _to_np(data).reshape(-1), _to_np(synth).reshape(-1)
        lo, hi = min(d.min(), s.min()), max(d.max(), s.max())
        bins = np.linspace(lo, hi, n_bins)
        ctr = 0.5 * (bins[1:] + bins[:-1])
        pd_, _ = np.histogram(d, bins=bins, density=True)
        ps_, _ = np.histogram(s, bins=bins, density=True)
        if smooth:
            pd_ = gaussian_filter1d(pd_, smooth_sigma)
            ps_ = gaussian_filter1d(ps_, smooth_sigma)

        # Track the x-range over which either curve is still above the
        # y-axis floor (y_floor), so the horizontal extent adapts to the
        # vertical cutoff instead of just spanning the full (often much
        # wider) raw data range.
        above = ctr[(pd_ > y_floor) | (ps_ > y_floor)]
        if above.size:
            x_lo = min(x_lo, above.min())
            x_hi = max(x_hi, above.max())

        pd_ = np.where(pd_ > 0, pd_, np.nan)
        ps_ = np.where(ps_ > 0, ps_, np.nan)
        ax.plot(ctr, pd_, ls=LS_DATA, color=color, lw=STYLE['lw'], label=f'{label} data')
        ax.plot(ctr, ps_, ls=LS_SYNTH, color=color, lw=STYLE['lw'], label=f'{label} synth')
    ax.set_yscale('log')
    _, hi_y = ax.get_ylim()
    ax.set_ylim(y_floor, hi_y)
    if np.isfinite(x_lo) and np.isfinite(x_hi):
        pad = 0.05 * (x_hi - x_lo)
        ax.set_xlim(x_lo - pad, x_hi + pad)
    #_legend(ax)
    _apply_style(ax, r'$x$', 'PDF')
    _finalize(fig, save_dir, 'panel_a_marginals', fmt, dpi)


# --------------------------------------------------------- (b) spectrum ----
def _psd(x):
    x2 = _to_np(x).reshape(-1, _to_np(x).shape[-1])
    f = np.fft.rfft(x2, axis=-1)
    return np.mean(np.abs(f) ** 2, axis=0)


def panel_b_spectrum(data1, synth1, data2=None, synth2=None,
                      label1="Type 1", label2="Type 2",
                      omega_max=0.4, save_dir=None, fmt='pdf', dpi=200):
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    T = data1.shape[-1]
    omega = np.fft.rfftfreq(T)
    # Cut the spectrum off at omega_max (drop the omega=0 bin too, since it's
    # plotted on a log-x axis). Slicing before plotting -- rather than just
    # calling set_xlim -- means the y-axis autoscale also reflects only the
    # visible frequency range.
    mask = (omega > 0) & (omega <= omega_max)
    omega_plot = omega[mask]
    for data, synth, color, label in _build_groups(data1, synth1, data2, synth2, label1, label2):
        P_o, P_s = _psd(data), _psd(synth)
        ax.plot(omega_plot, P_o[mask], ls=LS_DATA, color=color, lw=STYLE['lw'], label=f'{label} data')
        ax.plot(omega_plot, P_s[mask], ls=LS_SYNTH, color=color, lw=STYLE['lw'], label=f'{label} synth')
    ax.set_xscale('log')
    ax.set_yscale('log')
    #_legend(ax)
    _apply_style(ax, r'$\omega$', 'PSD')
    _finalize(fig, save_dir, 'panel_b_spectrum', fmt, dpi)


# ------------------------------------------------- (c) structure functions
def panel_c_structure_functions(data1, synth1, data2=None, synth2=None,
                                 label1="Type 1", label2="Type 2",
                                 orders=(4, 6, 8), save_dir=None, fmt='pdf', dpi=200):
    # NOTE: only the first order is labeled per group to keep the legend
    # readable; the 3 orders are distinguished by alpha only (1.0/0.7/0.4).
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    max_tau = data1.shape[-1] // 2
    taus = np.arange(1, max_tau)
    alphas = np.linspace(1.0, 0.4, len(orders))
    for data, synth, color, label in _build_groups(data1, synth1, data2, synth2, label1, label2):
        S_o = second_order_structure_function(data, p=np.array(orders), max_tau=max_tau)
        S_s = second_order_structure_function(synth, p=np.array(orders), max_tau=max_tau)
        for k, p in enumerate(orders):
            ax.plot(taus, S_o[k], ls=LS_DATA, color=color, lw=STYLE['lw'], alpha=alphas[k],
                    label=f'{label} data' if k == 0 else None)
            ax.plot(taus, S_s[k], ls=LS_SYNTH, color=color, lw=STYLE['lw'], alpha=alphas[k],
                    label=f'{label} synth' if k == 0 else None)
    ax.set_xscale('log')
    ax.set_yscale('log')
    #_legend(ax)
    _apply_style(ax, r'$\tau$', r'$S_p(\tau)$, $p=4,6,8$')
    _finalize(fig, save_dir, 'panel_c_structure_functions', fmt, dpi)


# --------------------------------------------------- (d) flatness slope ----
def panel_d_flatness_slope(data1, synth1, data2=None, synth2=None,
                            label1="Type 1", label2="Type 2",
                            num_points=30, save_dir=None, fmt='pdf', dpi=200):
    # ASSUMPTION: "local slope of the flatness" = d(log Flat_4)/d(log tau).
    # Swap in plain flatness(...) below if you meant Flat_4(tau) itself.
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    for data, synth, color, label in _build_groups(data1, synth1, data2, synth2, label1, label2):
        taus_o, F_o = flatness(data, num_points=num_points)
        taus_s, F_s = flatness(synth, num_points=num_points)
        slope_o = logder(F_o, taus_o)
        slope_s = logder(F_s, taus_s)
        ax.plot(taus_o, slope_o, ls=LS_DATA, color=color, lw=STYLE['lw'], label=f'{label} data')
        ax.plot(taus_s, slope_s, ls=LS_SYNTH, color=color, lw=STYLE['lw'], label=f'{label} synth')
    ax.set_xscale('log')
    #_legend(ax)
    _apply_style(ax, r'$\tau$', r'$d\log \mathrm{Flat}_4/d\log\tau$')
    _finalize(fig, save_dir, 'panel_d_flatness_slope', fmt, dpi)


# --------------------------------------------------- (e) increment PDFs ----
def panel_e_increment_pdfs(data1, synth1, data2=None, synth2=None,
                            label1="Type 1", label2="Type 2",
                            taus=(1, 4, 16, 64),
                            n_bins=201, xlim=(-6, 6), smooth=True, smooth_sigma=1.5,
                            save_dir=None, fmt='pdf', dpi=200):
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    taus = list(taus)
    groups = _build_groups(data1, synth1, data2, synth2, label1, label2)
    bins = np.linspace(xlim[0], xlim[1], n_bins)
    ctr = 0.5 * (bins[1:] + bins[:-1])

    # loop over tau on the outside so one label per tau can be placed after
    # both groups have been drawn for that scale
    for k, tau in enumerate(taus):
        for data, synth, color, label in groups:
            d_np, s_np = _to_np(data), _to_np(synth)
            d_inc = (d_np[..., tau:] - d_np[..., :-tau]).reshape(-1)
            s_inc = (s_np[..., tau:] - s_np[..., :-tau]).reshape(-1)
            d_inc = d_inc / (d_inc.std() + 1e-12)
            s_inc = s_inc / (s_inc.std() + 1e-12)
            pd_, _ = np.histogram(d_inc, bins=bins, density=True)
            ps_, _ = np.histogram(s_inc, bins=bins, density=True)
            if smooth:
                pd_ = gaussian_filter1d(pd_, smooth_sigma)
                ps_ = gaussian_filter1d(ps_, smooth_sigma)
            pd_ = pd_ / (pd_.max() + 1e-12) * 10.0 ** (-k)
            ps_ = ps_ / (ps_.max() + 1e-12) * 10.0 ** (-k)
            pd_ = np.where(pd_ > 0, pd_, np.nan)
            ps_ = np.where(ps_ > 0, ps_, np.nan)
            ax.plot(ctr, pd_, ls=LS_DATA, color=color, lw=STYLE['lw'] * 0.7,
                    label=f'{label} data' if k == 0 else None)
            ax.plot(ctr, ps_, ls=LS_SYNTH, color=color, lw=STYLE['lw'] * 0.7,
                    label=f'{label} synth' if k == 0 else None)

        # label placed at the (normalized) peak height for this scale
        peak_y = 10.0 ** (-k)
        ax.text(0.0, peak_y * 1.5, rf'$\tau={tau}$',
                ha='center', va='bottom', fontsize=STYLE['fontsize'] * 0.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='black', linewidth=1.2))

    ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(10.0 ** (-(len(taus) + 1)), 5.0)
    #_legend(ax, ncol=2)
    _apply_style(ax, r'$\delta u/\sigma_{\delta u}$', 'PDF (shifted)')
    _finalize(fig, save_dir, 'panel_e_increment_pdfs', fmt, dpi)


# ------------------------------------------------------ (f,g,h) C_{p,q} ----
def _panel_cpq(data1, synth1, data2, synth2, label1, label2, p, q, tau_star, ylabel, name,
               inset=False, save_dir=None, fmt='pdf', dpi=200):
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    max_tau = data1.shape[-1] // 2
    axins = ax.inset_axes([0.55, 0.55, 0.4, 0.4]) if inset else None
    for data, synth, color, label in _build_groups(data1, synth1, data2, synth2, label1, label2):
        d_np, s_np = _to_np(data), _to_np(synth)
        taus, r_o = C_pq_structure(d_np, p, q, tau_star, max_tau=max_tau)
        _, r_s = C_pq_structure(s_np, p, q, tau_star, max_tau=max_tau)
        ax.plot(taus, r_o, ls=LS_DATA, color=color, lw=STYLE['lw'], label=f'{label} data')
        ax.plot(taus, r_s, ls=LS_SYNTH, color=color, lw=STYLE['lw'], label=f'{label} synth')
        if inset:
            axins.plot(taus, r_o, ls=LS_DATA, color=color, lw=STYLE['lw'] * 0.7)
            axins.plot(taus, r_s, ls=LS_SYNTH, color=color, lw=STYLE['lw'] * 0.7)
    ax.axvline(tau_star, color='grey', ls=':', lw=2)
    if inset:
        lo, hi = max(1, tau_star // 2), tau_star * 2  # ASSUMPTION: zoom window
        axins.set_xlim(lo, hi)
        axins.axvline(tau_star, color='grey', ls=':', lw=1.5)
        axins.grid(True, alpha=STYLE['grid_alpha'])
        axins.tick_params(labelsize=STYLE['fontsize'] * 0.5)
    ax.set_xscale('log')
    #_legend(ax)
    _apply_style(ax, r'$\tau$', ylabel)
    _finalize(fig, save_dir, name, fmt, dpi)


def panel_f_C4(data1, synth1, data2=None, synth2=None,
               label1="Type 1", label2="Type 2",
               tau_star=20, save_dir=None, fmt='pdf', dpi=200):
    _panel_cpq(data1, synth1, data2, synth2, label1, label2, 2, 2, tau_star,
               r'$C_4(\tau,\tau^\star)$', 'panel_f_C4', inset=False,
               save_dir=save_dir, fmt=fmt, dpi=dpi)


def panel_g_C6(data1, synth1, data2=None, synth2=None,
               label1="Type 1", label2="Type 2",
               tau_star=20, save_dir=None, fmt='pdf', dpi=200):
    _panel_cpq(data1, synth1, data2, synth2, label1, label2, 3, 3, tau_star,
               r'$C_6(\tau,\tau^\star)$', 'panel_g_C6', inset=False,
               save_dir=save_dir, fmt=fmt, dpi=dpi)


def panel_h_C42(data1, synth1, data2=None, synth2=None,
                 label1="Type 1", label2="Type 2",
                 tau_star=20, save_dir=None, fmt='pdf', dpi=200):
    _panel_cpq(data1, synth1, data2, synth2, label1, label2, 4, 2, tau_star,
               r'$C_{4,2}(\tau,\tau^\star)$', 'panel_h_C42', inset=True,
               save_dir=save_dir, fmt=fmt, dpi=dpi)


# --------------------------------------------------------- (i) skewness ---
def panel_i_skewness(data1, synth1, data2=None, synth2=None,
                      label1="Type 1", label2="Type 2",
                      num_points=45, mode='power',
                      save_dir=None, fmt='pdf', dpi=200):
    # taus log-spaced over [1, M/2), M = data1.shape[-1] (i.e. stop at tau=M/2)
    fig, ax = plt.subplots(figsize=STYLE['figsize'])
    max_tau = data1.shape[-1] // 2
    taus = log_spaced_taus(max_tau, num_points=num_points)
    for data, synth, color, label in _build_groups(data1, synth1, data2, synth2, label1, label2):
        _, s_o = skewness_vs_tau(data, taus, mode=mode)
        _, s_s = skewness_vs_tau(synth, taus, mode=mode)
        ax.plot(taus, s_o, ls=LS_DATA, color=color, lw=STYLE['lw'], label=f'{label} data')
        ax.plot(taus, s_s, ls=LS_SYNTH, color=color, lw=STYLE['lw'], label=f'{label} synth')
    ax.axhline(0, color='black', lw=1)
    ax.set_xscale('log')
    #_legend(ax)
    _apply_style(ax, r'$\tau$', 'Skewness')
    _finalize(fig, save_dir, 'panel_i_skewness', fmt, dpi)


# --------------------------------------------------------------- driver ----
def make_all_panels(data1, synth1, data2=None, synth2=None,
                     label1="Type 1", label2="Type 2",
                     tau_star=20, save_dir=None, fmt='png', dpi=200):
    """Run all 9 panels for one data type, or two overlaid together.
    Pass only data1/synth1 for a single-type figure (e.g. just Lagrangian
    turbulence, or just jets); add data2/synth2 (and label1/label2) to
    overlay a second type on the same panels. Call again with a different
    save_dir for another dataset/group."""
    kwargs = dict(label1=label1, label2=label2, save_dir=save_dir, fmt=fmt, dpi=dpi)
    panel_a_marginals(data1, synth1, data2, synth2, **kwargs)
    panel_b_spectrum(data1, synth1, data2, synth2, **kwargs)
    panel_c_structure_functions(data1, synth1, data2, synth2, **kwargs)
    panel_d_flatness_slope(data1, synth1, data2, synth2, **kwargs)
    panel_e_increment_pdfs(data1, synth1, data2, synth2, **kwargs)
    panel_f_C4(data1, synth1, data2, synth2, tau_star=tau_star, **kwargs)
    panel_g_C6(data1, synth1, data2, synth2, tau_star=tau_star, **kwargs)
    panel_h_C42(data1, synth1, data2, synth2, tau_star=tau_star, **kwargs)
    panel_i_skewness(data1, synth1, data2, synth2, **kwargs)


# Usage:
# from paper_figures import make_all_panels
# make_all_panels(data1, synth1, tau_star=20, save_dir=None)                                   # single type, display
# make_all_panels(data1, synth1, label1="Lagrangian",
#                 tau_star=20, save_dir='figs/lagrangian')                                      # single type, save
# make_all_panels(data1, synth1, data2, synth2, label1="Jets", label2="Lagrangian",
#                 tau_star=20, save_dir='figs/combined')                                        # two types overlaid


def _shade(color, frac):
    """Lighten `color` slightly toward white for small frac; never dark.
    frac in [0,1]: 0 -> a little lighter, 1 -> base color."""
    r, g, b = to_rgb(color)
    t = 0.30 * (1.0 - frac)          # gentle: max 0.30 toward white
    return (r + (1 - r) * t,
            g + (1 - g) * t,
            b + (1 - b) * t)


def visual_comparison(Data, synth,
                      pdf_taus=(1, 2, 4, 8, 16, 32),
                      sf_orders=(4, 6, 8),
                      c4_fixed_tau=10,
                      n_bins=201, smooth=True, smooth_sigma=1.5,
                      pdf_xlim=(-8, 8), a_ylim=None, epsilon=1e-8):
    """
    Statistical comparison: original (blue) vs synthesis (red).
    Layout: 2x2 block on the left (a, b, d, e), tall panel (c) on the right.
    """
    data_np  = Data.cpu().numpy()
    synth_np = synth.cpu().numpy()
    max_tau  = Data.shape[-1] // 2

    plt.rcParams.update({
        'font.size': 20,
        'font.weight': 'normal',
        'axes.labelweight': 'normal',
        'axes.titleweight': 'normal',
        'axes.linewidth': 1.4,
    })
    LW    = 5.0
    LW_A  = 4.0
    LABEL_FS = 22

    taus_arr = np.arange(1, max_tau)

    def abs_increments_pow_mean(x, tau, p):
        d = np.abs(x[..., tau:] - x[..., :-tau])
        return np.mean(np.power(d.reshape(-1), p))

    fig = plt.figure(figsize=(20, 12))
    gs  = fig.add_gridspec(2, 3)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axC = fig.add_subplot(gs[:, 2])

    # ----- (a) marginal distribution -----
    s_orig  = data_np.reshape(-1)
    s_synth = synth_np.reshape(-1)
    lo = min(s_orig.min(), s_synth.min())
    hi = max(s_orig.max(), s_synth.max())
    bins = np.linspace(lo, hi, n_bins)
    ctr  = 0.5 * (bins[1:] + bins[:-1])
    p_o, _ = np.histogram(s_orig,  bins=bins, density=True)
    p_s, _ = np.histogram(s_synth, bins=bins, density=True)
    if smooth:
        p_o = gaussian_filter1d(p_o, smooth_sigma)
        p_s = gaussian_filter1d(p_s, smooth_sigma)
    p_o = np.where(p_o > 0, p_o, np.nan)
    p_s = np.where(p_s > 0, p_s, np.nan)
    axA.plot(ctr, p_o, color=C_ORIG,  lw=LW_A, label='Original')
    axA.plot(ctr, p_s, color=C_SYNTH, lw=LW_A, label='Synthesis')
    axA.set_yscale('log')
    axA.set_ylabel('PDF')
    if a_ylim is not None:
        axA.set_ylim(a_ylim)
    axA.legend(frameon=False, fontsize=16)
    axA.grid(True, ls=':', alpha=0.5)

    # ----- (b) power spectrum -----
    def psd(x):
        x2 = x.reshape(-1, x.shape[-1])
        f  = np.fft.rfft(x2, axis=-1)
        return np.mean(np.abs(f) ** 2, axis=0)
    P_o = psd(data_np)
    P_s = psd(synth_np)
    omega = np.fft.rfftfreq(data_np.shape[-1])
    axB.plot(omega[1:], P_o[1:], color=C_ORIG,  lw=LW)
    axB.plot(omega[1:], P_s[1:], color=C_SYNTH, lw=LW)
    axB.set_xscale('log')
    axB.set_yscale('log')
    #axB.set_xlabel(r'$\omega$')
    axB.set_ylabel('PSD')
    axB.grid(True, which='both', ls=':', alpha=0.5)

    # ----- (d) structure functions SF_k -----
    sf_colors = {4: 'tab:red', 6: 'tab:green', 8: 'tab:blue'}
    for p in sf_orders:
        SF_o = np.array([abs_increments_pow_mean(data_np,  t, p) for t in taus_arr])
        SF_s = np.array([abs_increments_pow_mean(synth_np, t, p) for t in taus_arr])
        col = sf_colors.get(p, 'k')
        axD.plot(taus_arr, SF_o + epsilon, '--', color=col, lw=LW)
        axD.plot(taus_arr, SF_s + epsilon, '-',  color=col, lw=LW,
                 label=rf'$\mathrm{{SF}}_{{{p}}}$')
    axD.set_xscale('log')
    axD.set_yscale('log')
    axD.set_xlabel(r'$\tau$')
    axD.set_ylabel(r'$\mathrm{SF}_k(\tau)$')
    axD.legend(frameon=False, fontsize=16)
    axD.grid(True, which='both', ls=':', alpha=0.5)

    # ----- (e) C_4 coefficient -----
    if not (1 <= c4_fixed_tau < max_tau):
        raise ValueError(f"c4_fixed_tau={c4_fixed_tau} must be in [1, {max_tau-1}]")

    _, r_o = C_pq_structure(data_np,  2, 2, c4_fixed_tau, max_tau=max_tau, epsilon=epsilon)
    _, r_s = C_pq_structure(synth_np, 2, 2, c4_fixed_tau, max_tau=max_tau, epsilon=epsilon)
    axE.plot(taus_arr, r_o, 'o-', color=C_ORIG,  lw=LW, ms=5)
    axE.plot(taus_arr, r_s, 's-', color=C_SYNTH, lw=LW, ms=5)
    axE.set_xscale('log')
    axE.set_xlabel(r'$\tau$')
    axE.set_ylabel(r'$C_4(\tau,\tau^\star)$')
    axE.grid(True, ls=':', alpha=0.5)

    # ----- (c) increment PDFs (waterfall), gentle gradient -----
    taus = list(pdf_taus)
    n = len(taus)
    for k, tau in enumerate(taus):
        frac = k / max(n - 1, 1)
        col_o = _shade(C_ORIG,  frac)
        col_s = _shade(C_SYNTH, frac)

        d_o = (data_np[...,  tau:] - data_np[...,  :-tau]).reshape(-1)
        d_s = (synth_np[..., tau:] - synth_np[..., :-tau]).reshape(-1)
        d_o = d_o / (d_o.std() + 1e-12)
        d_s = d_s / (d_s.std() + 1e-12)
        b   = np.linspace(pdf_xlim[0], pdf_xlim[1], n_bins)
        c   = 0.5 * (b[1:] + b[:-1])
        h_o, _ = np.histogram(d_o, bins=b, density=True)
        h_s, _ = np.histogram(d_s, bins=b, density=True)
        if smooth:
            h_o = gaussian_filter1d(h_o, smooth_sigma)
            h_s = gaussian_filter1d(h_s, smooth_sigma)
        h_o = h_o / (h_o.max() + 1e-12) * 10.0 ** (-k)
        h_s = h_s / (h_s.max() + 1e-12) * 10.0 ** (-k)
        peak = np.nanmax(h_o)
        h_o = np.where(h_o > 0, h_o, np.nan)
        h_s = np.where(h_s > 0, h_s, np.nan)
        axC.plot(c, h_o, color=col_o, lw=LW)
        axC.plot(c, h_s, color=col_s, lw=LW)
        axC.text(0.0, peak * 1.9, rf'$\tau={tau}$',
                 color='black', fontsize=16,
                 ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='white', edgecolor='black', linewidth=1.5))
    axC.set_yscale('log')
    axC.set_xlim(pdf_xlim)
    axC.set_ylim(10.0 ** (-(n + 1)), 8.0)
    axC.set_ylabel('PDF (shifted)')
    axC.grid(True, ls=':', alpha=0.5)

    # ----- finalize layout FIRST, then place panel letters -----
    plt.tight_layout()
    fig.canvas.draw()

    def letter(ax, text, dy):
        pos = ax.get_position()
        x_fig = 0.5 * (pos.x0 + pos.x1)
        fig.text(x_fig, pos.y0 - dy, text, ha='center', va='top',
                 fontsize=LABEL_FS)

    for ax, txt, dy in [(axA, '(a)', 0.03), (axB, '(b)', 0.03),
                        (axD, '(d)', 0.075), (axE, '(e)', 0.075),
                        (axC, '(c)', 0.075)]:
        letter(ax, txt, dy)

    #plt.savefig('visual_comparison.pdf', bbox_inches='tight')

    plt.show()


def plot_kregion_fit_paper(x, model, channel, label="", n_grid=800,
                            figsize=(9, 8), fontsize=15):
    """
    Single-channel, publication-ready figure for Scalar_GGD_KRegion.

    Two stacked panels sharing the x-axis (coefficient value):
      - top:    the smooth partition-of-unity windows w_k(z) that blend
                the K regions (bulk -> mid -> ... -> tail)
      - bottom: empirical histogram of the wavelet coefficients (log
                density) with the fitted truncated-GGD density overlaid
                per region, colored to match the window it corresponds to

    Does not refit -- uses model.alpha/scale/cuts/sw/pi/active as already
    fitted by model.fit_reference(...).
    """
    model._check_fitted()
    j = channel

    filters = model.filters.to(x.device)
    wt = torch.fft.ifft(torch.fft.fft(x) * filters).real

    A = model.alpha.cpu().numpy();  S = model.scale.cpu().numpy()
    C = model.cuts.cpu().numpy();   SW = model.sw.cpu().numpy()
    P = model.pi.cpu().numpy();     ACT = model.active.cpu().numpy()
    Keff = model.Keff.cpu().numpy()
    K = A.shape[1]

    h = wt[:, j, :].detach().cpu().flatten().numpy()
    h = h[np.isfinite(h)]
    ah = np.abs(h)
    xmax = float(ah.max()) * 1.02
    edges = [0.0] + list(C[j]) + [xmax]

    region_colors = plt.cm.plasma(np.linspace(0.12, 0.85, K))
    region_names = (["bulk"] + [f"mid_{{{k}}}" for k in range(1, K - 1)]
                     + (["tail"] if K > 1 else []))

    fig, (ax_w, ax_d) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw=dict(height_ratios=[1, 2.4], hspace=0.06))

    # ------------------------------------------------------------------
    # top panel: windows w_k(z)
    # ------------------------------------------------------------------
    zg = np.linspace(-xmax, xmax, 1000)
    az = np.abs(zg)
    if K == 1:
        ws = [np.ones_like(az)]
    else:
        g = [1.0 / (1.0 + np.exp(-(C[j, m] - az) / SW[j, m])) for m in range(K - 1)]
        ws = [g[0]]
        for m in range(1, K - 1):
            ws.append(g[m] - g[m - 1])
        ws.append(1.0 - g[K - 2])

    for k in range(K):
        name = region_names[k] if k < len(region_names) else f"r_{{{k}}}"
        ax_w.plot(zg, ws[k], lw=2.6, color=region_colors[k],
                  label=rf"$w_{{{name}}}$")

    for c_m, s_m in zip(C[j], SW[j]):
        for sign in (-1, 1):
            ax_w.axvline(sign * c_m, color="k", ls=":", lw=1.0, alpha=0.6)
            ax_d.axvline(sign * c_m, color="k", ls=":", lw=1.0, alpha=0.5)
            ax_w.axvspan(sign * c_m - 2 * s_m, sign * c_m + 2 * s_m,
                         color="gray", alpha=0.12)
            ax_d.axvspan(sign * c_m - 2 * s_m, sign * c_m + 2 * s_m,
                         color="gray", alpha=0.12)

    ax_w.set_ylim(-0.05, 1.08)
    ax_w.set_ylabel("window\nweight", fontsize=fontsize)
    ax_w.legend(fontsize=fontsize - 3, ncol=K, loc="upper center",
                bbox_to_anchor=(0.5, 1.35), frameon=False)
    ax_w.tick_params(labelsize=fontsize - 2)

    # ------------------------------------------------------------------
    # bottom panel: data histogram + fitted densities
    # ------------------------------------------------------------------
    ax_d.hist(h, bins=120, density=True, log=True, alpha=0.35,
              color="0.55", label="data", zorder=1)

    for k in range(K):
        if P[j, k] < 1e-4:
            continue
        lo, hi = edges[k], edges[k + 1]
        hi_fit = np.inf if k == K - 1 else hi
        xp = np.linspace(max(lo, 1e-6), hi, n_grid)
        lp = model._logpdf_trunc(xp, A[j, k], S[j, k], lo, hi_fit, P[j, k])
        xx = np.concatenate([-xp[::-1], xp])
        yy = np.exp(np.concatenate([lp[::-1], lp]))
        name = region_names[k] if k < len(region_names) else f"r_{{{k}}}"
        star = r"$^{*}$" if ACT[j, k] else ""
        ax_d.plot(xx, yy, lw=3, color=region_colors[k], zorder=3,
                  label=rf"${name}${star}: $\alpha$={A[j,k]:.2f}, $\pi$={P[j,k]:.1%}")

    ax_d.set_xlabel("coefficient value $x$", fontsize=fontsize)
    ax_d.set_ylabel("log density", fontsize=fontsize)
    ax_d.legend(fontsize=fontsize - 3, loc="upper right", framealpha=0.9)
    ax_d.tick_params(labelsize=fontsize - 2)

    fig.suptitle(f"{label}  —  channel {j}  ($K_{{eff}}$={int(Keff[j])})",
                 fontsize=fontsize + 2, y=1.0)
    y_thresh = 1e-4
    ax_d.set_ylim(bottom=y_thresh)
    plt.tight_layout()
    plt.savefig("fitting_Q3_marginal.png")

    plt.show()
    return fig