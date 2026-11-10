"""2D figures and tables of the cosmic-web talk notebook (cosmic_web/talk_cosmic.ipynb).

What is generic is taken from codes/talk_figures.py (tf: colours, _save, -log p histogram, energy bars, statistic count,
Var/2 table); this module adds what is specific to maps:
  load_run          data maps, generated maps per seed, contributions C_F = Theta_F . phi_F to ln p (nats) per family
  plot_maps         data and generated maps on one colour scale
  plot_pdf_spectrum pixel PDF and isotropic power spectrum P(|k|)
  plot_wavelet_hist histograms of Re W_j (Morlet coefficients, orientations pooled) per scale j
  plot_peaks        number of local maxima per map vs height
  plot_minkowski    the three Minkowski functionals vs threshold (pixel estimators)
  moment_check      the imposed-moment check of moment_check_cw.py (in-sample: confirms the SDE only)
  entropy_table     MGD entropy bound vs Gaussian with the data covariance, per pixel
  dk_table          exact D_k = log E_p exp(-(C_k - E C_k)) per category, next to its first-order term Var(C_k)/2
Bands (talk convention): data = min-max over batches of maps; model = mean +- std over seeds. The two are not
like-for-like (batch spread vs seed spread); the caption should say so. Maps are standardised (zero mean, unit std of
the data), so heights and thresholds are in data std.
"""
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import maximum_filter
from scipy.special import logsumexp

import cw_common as cw
import talk_figures as tf
from logp_cw import family_logp, theta_file

SCR = Path(os.environ.get('SCRATCH', '/nonexistent')) / 'MGD-for-Maximum-Entropy-Generation' / 'cosmic_web'


# ------------------------------------------------------------------ loading ----
def load_run(arm, seeds, lam, theta_dir=None, data_path=None, device='cpu', chunk=10):
    """Everything the notebook needs for one arm. Theta = Theta_reg(t = 1) at lam (files <theta_dir>/<config>/lam<lam>_ridge*.pt,
    default $SCRATCH/.../cosmic_web/theta_reg/<arm>). C_data / C_gen = {family: (seeds, maps)} + 'TOTAL' (last), ln p up to
    a constant per seed (nats). The data maps are the same for every seed; each seed scores them with its own Theta."""
    theta_dir = Path(theta_dir) if theta_dir else SCR / 'theta_reg' / arm
    runs = cw.find_runs(arm, seeds)
    cfg0 = runs[seeds[0]][1]
    out = dict(arm=arm, seeds=list(seeds), lam=lam, cfg=cfg0, runs=runs, x_data=cw.load_x1(cfg0, data_path).float(),
               x_gen={}, theta={})
    Cd, Cg = [], []
    for s in seeds:
        exp_dir, cfg = runs[s]
        out['x_gen'][s] = cw.load_samples(cfg)
        f, _ = theta_file(arm, None, cfg['config_name'], None, lam, theta_dir)
        sol = torch.load(f, map_location='cpu', weights_only=False)
        assert sol['config'] == cfg['config_name'] and np.isclose(float(sol['lam']), lam), (sol['config'], sol['lam'], lam)
        th = sol['Theta_reg'][-1].double().numpy()
        out['theta'][s] = th
        phi_d, phi_g, fam = cw.get_phi(arm, s, exp_dir, cfg, device, chunk, data_path)
        Cd.append(family_logp(phi_d, th, fam)); Cg.append(family_logp(phi_g, th, fam))
    names = list(fam) + ['TOTAL']
    Cd, Cg = np.stack(Cd), np.stack(Cg)                                  # (seeds, F+1, maps)
    out['C_data'] = {k: Cd[:, i] for i, k in enumerate(names)}
    out['C_gen'] = {k: Cg[:, i] for i, k in enumerate(names)}
    out['widths'] = {k: w for k, (_, w) in fam.items()}
    out['fam'] = fam
    out['M'] = int(cfg0['M'])
    out['d'] = out['M'] ** 2
    print(f'{arm}: {len(out["x_data"])} data maps, generated {[len(v) for v in out["x_gen"].values()]} per seed, '
          f'lam {lam:g}, r = {sum(out["widths"].values())}, d = {out["d"]}')
    return out


# ------------------------------------------------------------------ helpers ----
def _np(x):
    return x[:, 0].numpy() if torch.is_tensor(x) else np.asarray(x)


def _batches(n, n_batches):
    return np.array_split(np.arange(n), n_batches)


def _bands(stat, x_data, x_gen, n_batches=5):
    """stat(maps (n, M, M)) -> curve (K,). Returns the data curve on all maps, data min/max over batches, model mean/std
    over seeds."""
    xd = _np(x_data)
    d_all = stat(xd)
    d_b = np.stack([stat(xd[b]) for b in _batches(len(xd), n_batches)])
    g = np.stack([stat(_np(x)) for x in x_gen.values()])
    return d_all, d_b.min(0), d_b.max(0), g.mean(0), g.std(0, ddof=1) if len(g) > 1 else np.zeros_like(g[0])


def _draw(ax, xv, bands, logy=True, label_data='data', label_model='MGD'):
    d, dlo, dhi, gm, gs = bands
    ax.fill_between(xv, dlo, dhi, color=tf.GREY, alpha=0.35, lw=0)
    ax.plot(xv, d, color=tf.DATA, lw=1.6, label=label_data)
    ax.fill_between(xv, gm - gs, gm + gs, color=tf.MODEL, alpha=0.25, lw=0)
    ax.plot(xv, gm, color=tf.MODEL, lw=1.4, ls='--', label=label_model)
    if logy:
        ax.set_yscale('log')


# ------------------------------------------------------------------- maps ------
def plot_maps(R, name='maps', n=4, seed=None):
    """Row 1: n data maps; row 2: n generated maps of one seed (default the first); same colour scale."""
    seed = seed or R['seeds'][0]
    xd, xg = _np(R['x_data'])[:n], _np(R['x_gen'][seed])[:n]
    vmin, vmax = np.percentile(np.concatenate([xd.ravel(), xg.ravel()]), [0.5, 99.5])
    fig, axs = plt.subplots(2, n, figsize=(2.6 * n, 5.4))
    for i in range(n):
        for r, x in enumerate((xd, xg)):
            axs[r, i].imshow(x[i], vmin=vmin, vmax=vmax, cmap='inferno'); axs[r, i].set_xticks([]); axs[r, i].set_yticks([])
    axs[0, 0].set_ylabel('data'); axs[1, 0].set_ylabel(f'MGD (seed {seed})')
    fig.tight_layout()
    tf._save(fig, name)


# ------------------------------------------------- PDF and power spectrum -------
def radial_power_spectrum(x):
    """(K,) mean |FFT|^2 / (M N) per integer |k| shell, |k| = 1 .. M/2 (as run_SDE.radial_power_spectrum)."""
    x = torch.as_tensor(x, dtype=torch.float64)
    M, N = x.shape[-2:]
    P = (torch.fft.fft2(x).abs() ** 2).mean(0) / (M * N)
    k = torch.sqrt(torch.fft.fftfreq(M, 1.0 / M)[:, None] ** 2 + torch.fft.fftfreq(N, 1.0 / N)[None, :] ** 2).round().long()
    return np.array([P[k == r].mean().item() for r in range(1, min(M, N) // 2 + 1)])


def plot_pdf_spectrum(R, name='pdf_spectrum', bins=80, n_batches=5):
    xd = _np(R['x_data'])
    edges = np.linspace(np.percentile(xd, 0.01) - 0.5, np.percentile(xd, 99.99) + 0.5, bins + 1)
    ctr = (edges[1:] + edges[:-1]) / 2
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    _draw(axs[0], ctr, _bands(lambda x: np.histogram(x, edges, density=True)[0], R['x_data'], R['x_gen'], n_batches))
    axs[0].set_xlabel('pixel value (data std)'); axs[0].set_ylabel('PDF'); axs[0].legend(frameon=False)
    K = np.arange(1, R['M'] // 2 + 1)
    _draw(axs[1], K, _bands(radial_power_spectrum, R['x_data'], R['x_gen'], n_batches))
    axs[1].set_xscale('log'); axs[1].set_xlabel('|k| (cycles per map)'); axs[1].set_ylabel('P(|k|)')
    axs[1].xaxis.set_minor_formatter(plt.NullFormatter())
    fig.tight_layout()
    tf._save(fig, name)


# ------------------------------------------------------ wavelet histograms -------
def wavelet_coeffs(x, filters, J, L):
    """Re W_j(x) for every scale j, orientations pooled: list of J arrays (n * L * M * M,). filters = return_Filters
    output (1, L*J + 1, M, M) in Fourier space, band-pass index = j * L + l (checked by its centre frequency in the nb)."""
    X = torch.fft.fft2(torch.as_tensor(x, dtype=torch.float32))[:, None]
    W = torch.fft.ifft2(X * filters[:, :L * J]).real                       # (n, L*J, M, M)
    return [W[:, j * L:(j + 1) * L].reshape(-1).numpy() for j in range(J)]


def filter_centres(filters, J, L):
    """|k| of the maximum of each band-pass filter (cycles per map): should decrease with j (check of the index order)."""
    M = filters.shape[-1]
    k = torch.sqrt(torch.fft.fftfreq(M, 1.0 / M)[:, None] ** 2 + torch.fft.fftfreq(M, 1.0 / M)[None, :] ** 2)
    c = [float(k.flatten()[filters[0, i].abs().flatten().argmax()]) for i in range(L * J)]
    return np.array(c).reshape(J, L)


def plot_wavelet_hist(R, name='wavelet_hist', bins=60, n_batches=5, lim=8.0):
    """One panel per scale j: histogram of Re W_j / std_data(Re W_j), orientations pooled; log y."""
    from codes.filters_bank import return_Filters
    cfg = R['cfg']
    J, L = int(cfg['J']), int(cfg['L'])
    filt = return_Filters(R['M'], J, cfg['Q'], L=L)
    print('centre |k| of the band-pass filters (rows j, columns orientation):\n', filter_centres(filt, J, L).round(1))
    sd = [w.std() for w in wavelet_coeffs(_np(R['x_data']), filt, J, L)]
    edges = np.linspace(-lim, lim, bins + 1); ctr = (edges[1:] + edges[:-1]) / 2
    fig, axs = plt.subplots(1, J, figsize=(3.2 * J, 3.2), sharey=True)
    for j in range(J):
        stat = (lambda x, j=j: np.histogram(wavelet_coeffs(x, filt, J, L)[j] / sd[j], edges, density=True)[0])
        _draw(axs[j], ctr, _bands(stat, R['x_data'], R['x_gen'], n_batches))
        axs[j].set_title(f'j = {j}', fontsize=10); axs[j].set_xlabel(r'Re $W_j$ / data std')
    axs[0].set_ylabel('PDF'); axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    tf._save(fig, name)


# ----------------------------------------------------------------- peaks ------
def peak_heights(x):
    """Heights of the local maxima (3x3 neighbourhood, periodic) of every map, pooled."""
    m = maximum_filter(x, size=(1, 3, 3), mode='wrap')
    return x[(x == m)]


def plot_peaks(R, name='peaks', bins=40, n_batches=5):
    """Number of local maxima per map and per unit height, vs height (data std)."""
    xd = _np(R['x_data'])
    h = peak_heights(xd)
    edges = np.linspace(h.min() - 0.5, h.max() + 0.5, bins + 1); ctr = (edges[1:] + edges[:-1]) / 2
    stat = lambda x: np.histogram(peak_heights(x), edges)[0] / len(x) / np.diff(edges)
    b = _bands(stat, R['x_data'], R['x_gen'], n_batches)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    with np.errstate(divide='ignore'):
        _draw(ax, ctr, b)
    ax.set_xlabel('peak height (data std)'); ax.set_ylabel('peaks per map per unit height'); ax.legend(frameon=False)
    fig.tight_layout()
    tf._save(fig, name)


# ------------------------------------------------------------- Minkowski ------
def minkowski(x, nus):
    """(3, len(nus)) Minkowski functionals of the excursion sets {x > nu}, per pixel, mean over maps (periodic maps).
    V0 = area fraction; V1 = number of pixel edges between the set and its complement, per pixel (pixel estimator of the
    boundary length; it overestimates an isotropic length by 4/pi, the same for data and model); V2 = Euler
    characteristic per pixel, chi = vertices - edges + faces of the union of closed pixels (8-connected set)."""
    out = np.zeros((3, len(nus)))
    n, M, N = x.shape
    for i, nu in enumerate(nus):
        a = x > nu
        ah, av = np.roll(a, 1, axis=1), np.roll(a, 1, axis=2)
        out[0, i] = a.mean()
        out[1, i] = ((a != ah).sum() + (a != av).sum()) / (n * M * N)
        F = a.sum()
        E = (a | ah).sum() + (a | av).sum()
        V = (a | ah | av | np.roll(ah, 1, axis=2)).sum()
        out[2, i] = (V - E + F) / (n * M * N)
    return out


def plot_minkowski(R, name='minkowski', nus=None, n_batches=5):
    nus = np.linspace(-3, 4, 57) if nus is None else nus
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    labels = ('$V_0$: area fraction', '$V_1$: boundary edges per pixel', '$V_2$: Euler characteristic per pixel')
    for k in range(3):
        _draw(axs[k], nus, _bands(lambda x, k=k: minkowski(x, nus)[k], R['x_data'], R['x_gen'], n_batches), logy=False)
        axs[k].set_xlabel(r'threshold $\nu$ (data std)'); axs[k].set_title(labels[k], fontsize=10)
    axs[2].axhline(0, color=tf.GREY, lw=0.8)
    axs[0].legend(frameon=False)
    fig.tight_layout()
    tf._save(fig, name)


# ------------------------------------------------- -log p vs a Gaussian -------
def plot_logp_vs_chi2(R, name='neglogp_vs_gaussian', bins=60, source='gen'):
    """Histogram of u = -log10 p / d of the MGD maps (each seed centred on its own median, seeds pooled) against the
    exact distribution of the same quantity for a Gaussian field's own samples: -ln p = chi2_d / 2 + const for ANY
    covariance (x^T Sigma^-1 x ~ chi2_d), so no spectrum is needed. Centred on the median of chi2_d / 2. Legend: std in
    nats (Gaussian: sqrt(d/2); MGD: mean +- std over seeds)."""
    from scipy.stats import chi2
    d = R['d']
    sc = tf.neglogp_scale(d)                                           # nats -> log10 per pixel
    C = R['C_gen' if source == 'gen' else 'C_data']['TOTAL']           # ln p, (seeds, maps)
    u = np.concatenate([-(c - np.median(c)) for c in C]) * sc
    sd = np.array([c.std(ddof=1) for c in C])
    y = np.linspace(chi2.ppf(1e-6, d), chi2.ppf(1 - 1e-6, d), 2000) / 2      # -ln p - const of the Gaussian
    ug = (y - chi2.median(d) / 2) * sc
    dens = 2 * chi2.pdf(2 * y, d) / sc                                 # density in u
    lim = max(np.abs(u).max(), np.abs(ug[dens > dens.max() * 1e-4]).max())
    edges = np.linspace(-lim, lim, bins + 1)
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.hist(u, edges, density=True, color=tf.MODEL, alpha=0.45,
            label=f'MGD, cosmic web (std {sd.mean():.1f} ± {sd.std(ddof=1) if len(sd) > 1 else 0:.1f} nats)')
    ax.plot(ug, dens, color=tf.DATA, lw=1.8, label=rf'Gaussian field, $\chi^2_d/2$ (std {np.sqrt(d / 2):.1f} nats)')
    ax.set_yscale('log'); ax.set_ylim(bottom=max(dens.max(), 1) * 1e-4)
    ax.set_xlabel(r'$-\log_{10} p\,/\,d$ (median subtracted)'); ax.set_ylabel('density')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    tf._save(fig, name)
    return sd, np.sqrt(d / 2)


# --------------------------------------------------------- moment check -------
def moment_check(R, var_floor=1e-8, device='cpu', chunk=10):
    """Imposed-moment check (moment_check_cw.py): per family, offset |data mean - generated mean| / data std against the
    offset of independent sampling. In-sample: MGD pins the walkers' mean to the data mean, so this confirms the SDE."""
    import moment_check_cw as mc
    Z, OFF, NOISE, USED = [], [], [], []
    for s in R['seeds']:
        exp_dir, cfg = R['runs'][s]
        phi_d, phi_g, fam = cw.get_phi(R['arm'], s, exp_dir, cfg, device, chunk)
        z, off, noise, used = mc.moment_z(phi_d, phi_g, var_floor, fam)
        Z.append(z); OFF.append(off); NOISE.append(noise); USED.append(used)
    Z, OFF, NOISE, USED = map(np.stack, (Z, OFF, NOISE, USED))
    T = mc.family_table(Z, OFF, NOISE, USED, fam)
    out = (tf._OUT or Path('.')) / 'moment_check.png'
    mc.plot_arm(R['arm'], OFF, NOISE, fam, T, out)
    try:
        from IPython.display import Image, display
        display(Image(str(out)))
    except ImportError:
        pass
    return T


# ---------------------------------------------------------------- entropy -----
def entropy_table(R):
    """Final MGD entropy bound (a lower bound on the entropy of the model) and entropy of the Gaussian with the data
    covariance, from each run's summary.json; the gap per pixel (nats) = (H_Gauss - H_bound) / d."""
    rows = {}
    for s in R['seeds']:
        js = json.loads((R['runs'][s][0] / 'summary.json').read_text())
        rows[s] = {'H bound (nats)': js['H_bound_final'], 'H Gaussian (nats)': js['H_gaussian_final'],
                   'gap per pixel (nats)': (js['H_gaussian_final'] - js['H_bound_final']) / R['d']}
    T = pd.DataFrame(rows).T
    T.loc['mean'] = T.mean(); T.loc['std over seeds'] = T.iloc[:-1].std(ddof=1)
    return T


# ---------------------------------------------------------- Var/2 and D_k -----
def category_contributions(C):
    """{category: (seeds, maps)} summed over the families of each tf.ENERGY_CATEGORIES category; also the widths."""
    out = {}
    for f, v in C.items():
        if f == 'TOTAL':
            continue
        hit = [c for c, pat in tf.ENERGY_CATEGORIES if re.search(pat, f)]
        assert len(hit) == 1, f'family {f}: categories {hit}'
        out[hit[0]] = out.get(hit[0], 0) + v
    return out


def dk_table(R, source='gen', n_boot=200, seed=0):
    """Exact D_k = D_KL(p || q_k), q_k = p with the category's term removed and renormalised. With ln p = sum_k C_k + const
    (our sign, C_k = Theta_k . phi_k), q_k proportional to p exp(-C_k), hence
        D_k = E_p[C_k] + ln E_p exp(-C_k) = ln E_p exp(-(C_k - E_p C_k)),
    estimated with the empirical mean over the maps (source 'gen': generated maps = samples of p; 'data': the data maps).
    First-order term: Var_p(C_k)/2 (cumulant series 1/2 k2 - 1/6 k3 + 1/24 k4 ...). Per seed; table = mean +- std over
    seeds. 'boot rel. err' = bootstrap std of D_k over the maps / D_k (mean over seeds); 'ESS' = effective number of maps
    in the exp-average, (sum w)^2 / sum w^2 with w = exp(-(C_k - mean)) (min over seeds): small ESS = D_k set by few maps.
    'per pixel (log10)' = D_k / (d ln 10), the units of the Var/2 table."""
    rng = np.random.default_rng(seed)
    C = category_contributions(R['C_gen' if source == 'gen' else 'C_data'])
    widths = {}
    for f, w in R['widths'].items():
        c = [c for c, pat in tf.ENERGY_CATEGORIES if re.search(pat, f)][0]
        widths[c] = widths.get(c, 0) + w
    rows = {}
    for c, v in C.items():
        D, V, B, E = [], [], [], []
        for x in v:                                                     # one seed: (maps,)
            u = -(x - x.mean())
            n = len(u)
            D.append(logsumexp(u) - np.log(n))
            V.append(x.var(ddof=1) / 2)
            idx = rng.integers(0, n, (n_boot, n))
            xb = x[idx]
            B.append(np.std(logsumexp(-(xb - xb.mean(1, keepdims=True)), axis=1) - np.log(n), ddof=1))
            w = np.exp(u - u.max())
            E.append(w.sum() ** 2 / (w ** 2).sum())
        D, V = np.array(D), np.array(V)
        rows[c] = {'number of moments': widths[c], 'D_k (nats)': D.mean(), '± D_k': D.std(ddof=1) if len(D) > 1 else np.nan,
                   'Var/2 (nats)': V.mean(), '± Var/2': V.std(ddof=1) if len(V) > 1 else np.nan,
                   'D_k / (Var/2)': (D / V).mean(), 'D_k per pixel (log10)': D.mean() / (R['d'] * np.log(10)),
                   'boot rel. err': float(np.mean(np.array(B) / D)), 'ESS (maps)': float(np.min(E))}
    order = [c for c, _ in tf.ENERGY_CATEGORIES if c in rows]
    return pd.DataFrame(rows)[order]
