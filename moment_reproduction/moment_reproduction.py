"""Moment-reproduction diagnostic for MGD runs.

Per scalar statistic k, one model, one seed:
    e_k     = (m_k - mhat_k)^2 / s2_data_k
    floor_k = (s2_data_k / n + s2_model_k / n_s) / s2_data_k
reported as e_k / floor_k (~1 reproduced, >> 1 not). m, s2_data: mean / unbiased variance of phi_k on the
training data x1 (n windows); mhat, s2_model: same on the synthetic samples (n_s). Samples treated as independent.
(Same quantity as z_n^2 in turbulence/moment_error_variance.py; that script covers imposed statistics only.)

Imposed = every potential the run fitted (terms in config.json, fitted states from fitted_potentials/).
Extra (non-imposed) families: any get_1d_potentials term, or 'Scalar_Q<q>_gaussianK' (GGD on a Q=q bank).
The metric is invariant under a constant rescaling of phi_k, so each potential's own normalisation is irrelevant.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data',
          ROOT / 'turbulence', ROOT / 'jets', ROOT / 'nested_models'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from logp_entropy import rebuild, phi_matrix, find_run, RUN_NAME_RE      # noqa: E402  (restores the mpl backend)
from check_fit_offline import build_data                                 # noqa: E402
from codes.gram_preflight import stat_labels                             # noqa: E402
from filters_bank import return_Filters, deduplicate_filters             # noqa: E402
from potentials_builder import get_1d_potentials                         # noqa: E402
from potentials.potentials_1d import Scalar_GGD_KRegion                  # noqa: E402

# GGD settings of the extra (non-imposed) scalar families: the final model's fix
# (outer cut at q.99 + outer-window mass, mass_L6_2b, 2026-10-22). auto_prune off: pruning only keeps the
# SDE's Gram invertible (a near-copy column is still a statistic to check) and is a per-window Python loop
# over all n windows -- the slow part of the fit at n=10000.
EXTRA_GGD = dict(tail_quantile=[0.99], tail_mass=True, auto_prune=False, verbose=False)
# Extra families keep every channel, also duplicate ones the run's deduplicate_filters drops (e.g. Q=3 bank
# 21-23 at M=256, J=8): dedup only keeps the SDE's Gram invertible, a copy is still a statistic to check
# (2026-11-03). Imposed families are untouched: they are the run's own potentials. Zero (empty) wavelets of
# Scalar_Q<q> banks are still dropped (_bank_drop_empty): they have no statistic.
EXTRA_DEDUP = False
# GGD settings of the Scalar_Q<q> extras only (Q4/Q5; decided 2026-11-03): windows only, no outer-window mass;
# outer cut at q.99; every window >= EXTRA_Q_STEP_NOISE x the noise of one SDE step at 1 - t =
# ggd_noise_ref_gap (A/bicw rule, utils_experiment.py), computed from each run's own time grid. The other
# extra scalar families keep EXTRA_GGD (as already run).
EXTRA_GGD_Q = dict(tail_quantile=[0.99], tail_mass=False, auto_prune=False, verbose=False)
EXTRA_Q_STEP_NOISE = 1.0
# Display rules: a statistic is "within noise" if its offset <= NOISE_K x noise (2 = ~95% band); the
# percentage-of-mean error is shown only where |mean| >= REL_MEAN_MIN x std_data (then |rel| <= offset / 0.5).
NOISE_K = 2.0
# within noise splits in two: "matched" if the resolution NOISE_K x noise < RESOLVED_MAX (the measurement is
# sharp enough to call it a match), else "unresolved" (too few samples to judge that statistic: heavy tails).
RESOLVED_MAX = 0.05
REL_MEAN_MIN = 0.5
OFFSET_RANGE = (1e-2, 1.0)        # fixed colour scale of the heatmap (offset in data std)
# GPU when there is one (potentials, data, samples, extra fits); results come back as CPU float64.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SCALAR_Q_RE = re.compile(r'^Scalar_Q(\d+)_gaussianK$')


# ------------------------------------------------------------------------------------------- loading ----
def _need(cfg, key):
    if cfg.get(key) is None:
        raise KeyError(f'config.json has no {key!r}: cannot determine it, not guessing')
    return cfg[key]


def training_data(cfg):
    """x1 (n1, 1, M) float32 as the run built it."""
    n1, L, T = int(_need(cfg, 'n1')), int(_need(cfg, 'subseries_len')), int(_need(cfg, 'target_len'))
    if cfg.get('dataset') == 'modane':                                   # jets/run_jet_SDE.py
        from jet_data import load_modane_windows
        x1, _ = load_modane_windows(L, T, n=n1, spread=bool(cfg.get('n1_spread')))
        return x1.float()
    if not str(_need(cfg, 'config_name')).startswith('turb'):
        raise ValueError(f"unknown dataset for {cfg['config_name']}")
    if not cfg.get('n1_spread'):
        return build_data(L, T, n1, 0)[0]
    # --n1_spread: same selection as turbulence/run_SDE.py:461-472
    from utils import normalize, split_periodize_reshape
    from ortho_wavelet.ReadyToUseWavelets import DefineWavelet
    from data.data_loader import load_turbulence_1d
    raw = load_turbulence_1d().cpu()
    D = split_periodize_reshape(raw, L)
    W = DefineWavelet('Db', m=3, device='cpu')
    for _ in range(int(np.log2(L / T)) if L > T else 0):
        D = W.decompose(D)[1]
    per_series = D.shape[0] // raw.shape[0]
    g = np.random.default_rng(0)
    series = g.choice(raw.shape[0], n1, replace=False)
    return normalize(D[torch.as_tensor(series * per_series + g.integers(0, per_series, n1))])


def first_seed(label, root):
    """Smallest seed with saved samples for this label."""
    seeds = sorted({int(m[1]) for f in (Path(root) / 'saved_results' / 'samples').glob(f'*{label}_2*.pt')
                    if (m := RUN_NAME_RE.search(f.stem)) and m[3] == label})
    if not seeds:
        raise FileNotFoundError(f'no samples for label {label!r} under {root}')
    return seeds[0]


def load_model(label, root, seed=None, device=DEVICE):
    """(cfg, fitted potentials, x1, synthetic samples) of one run, on `device`; seed None = first seed.
    The run's potentials are loaded with their saved fit (fitted_potentials/), never refitted."""
    root = Path(root)
    seed = first_seed(label, root) if seed is None else seed
    exp_dir, config = find_run(label, seed, root=root)
    cfg = json.loads((exp_dir / 'config.json').read_text())
    cfg.setdefault('config_name', config)
    x1 = training_data(cfg).to(device)
    pots = rebuild(cfg, exp_dir, device)
    for p in pots.values():               # as SDE.__init__: everything not loaded from a state is fit on x1
        if not hasattr(p, 'is_fitted'):
            p.fit(x1)
    xs = torch.load(root / 'saved_results' / 'samples' / f'{config}.pt', map_location='cpu')
    xs = xs[:, None] if xs.dim() == 2 else xs
    return cfg, pots, x1, xs.float().to(device)


def build_extra(names, cfg, x1):
    """Non-imposed potentials, configured from the run's config (M, J, Q, wav_type), fit on x1."""
    M, J, Q = int(_need(cfg, 'M')), int(_need(cfg, 'J')), int(_need(cfg, 'Q'))
    wav = cfg.get('wav_type') or 'morlet'                 # return_Filters' own default (older configs lack it)
    dev = x1.device
    filters, fphi = return_Filters(M, J, 1, device=dev, include_phi=True, wav_type=wav)
    filters_Q = return_Filters(M, J, Q, device=dev, wav_type=wav)
    dedup = EXTRA_DEDUP and bool(cfg.get('deduplicate_filters'))
    tol = float(cfg.get('dedup_tol') or 1e-12)
    out = {}
    for name in names:
        if (m := SCALAR_Q_RE.match(name)):
            q = int(m[1])
            f = _bank_drop_empty(M, J, q, dev, wav)
            settings = dict(EXTRA_GGD_Q, min_window_bic=_step_noise_width(cfg))
            pot = Scalar_GGD_KRegion(deduplicate_filters(f, tol)[0] if dedup else f, **settings)
            pot.to(dev)
            out[name] = _fit_or_load_ggd(name, pot, x1, settings, key_extra=f'J{J}_dropempty')
            continue
        else:
            built = get_1d_potentials([name], J, filters, Q, filters_Q=filters_Q, filters_Phi=fphi,
                                      deduplicate_filters=dedup, dedup_tol=tol, ggd_kwargs=EXTRA_GGD)
            if name not in built:
                raise ValueError(f'{name!r} is not a get_1d_potentials term nor Scalar_Q<q>_gaussianK')
            pot = built[name]
        pot.to(dev)
        if isinstance(pot, Scalar_GGD_KRegion):
            out[name] = _fit_or_load_ggd(name, pot, x1, EXTRA_GGD)
        else:
            pot.fit(x1)                                   # scattering / L_p: no-op
            out[name] = pot
    return out


def _bank_drop_empty(M, J, q, dev, wav):
    """(1, C, M) bank as return_Filters(M, J, q) (band-passes then low-pass), without the wavelets that are empty
    on the M-point grid. Morlet at M=256, J=8 (checked 2026-11-03): Q=4 last wavelet (xi*M = 0.58, sigma*M = 0.06)
    and Q=5 last two (xi*M = 0.65, 0.56) sit below the first Fourier bin and are narrower than one bin, so
    filters_1d's l1 normalisation raises "Zero division"; they carry nothing, so they are dropped (the run's J kept).
    Q<=3 banks have no such wavelet: identical to return_Filters."""
    try:
        return return_Filters(M, J, q, device=dev, wav_type=wav)
    except ValueError:
        if wav != 'morlet':
            raise
    from filters.filters_1d import init_wavelet_param, morlet_1d, init_low_pass
    from filters_bank import _as_filter
    psi, dropped = [], []
    for i, (xi, s) in enumerate(zip(*init_wavelet_param('morlet', J, q, 0.49))):   # 0.49 = return_Filters' value
        try:
            psi.append(morlet_1d(M, xi, s, 'l1'))
        except ValueError:
            dropped.append(i)
    phi = init_low_pass('morlet', M, J, q, 0.49)
    print(f'[extra] Q={q} bank at J={J} (M={M}): dropped empty wavelets {dropped} of {len(psi) + len(dropped)}')
    return torch.cat([_as_filter(np.stack(psi), dev)[None], _as_filter(phi, dev)[None, None]], dim=1)


def _step_noise_width(cfg):
    """min_window_bic (per unit-noise channel) of the A/bicw rule, from the run's own grid: EXTRA_Q_STEP_NOISE x
    sqrt(2 h_ref) x sigma, h_ref = the step at 1 - t = ggd_noise_ref_gap (default 1e-2). Same formula as
    utils_experiment.py (--ggd_bic_min_window); fixed grids only."""
    from types import SimpleNamespace
    from codes.time_schedules import build_time_grid
    if cfg.get('schedule') == 'adaptive':
        raise ValueError('step-noise window width needs a fixed time grid (schedule is adaptive)')
    t, _ = build_time_grid(SimpleNamespace(**cfg))
    tt = np.asarray(t, dtype=np.float64)
    gap_ref = float(cfg.get('ggd_noise_ref_gap') or 1e-2)
    i_ref = int(np.argmin(np.abs((1.0 - tt[1:]) - gap_ref)))
    h_ref = float(tt[i_ref + 1] - tt[i_ref])
    w = EXTRA_Q_STEP_NOISE * (2.0 * h_ref) ** 0.5 * float(_need(cfg, 'sigma'))
    print(f'[extra] Scalar_Q windows >= {EXTRA_Q_STEP_NOISE:g} x step noise at 1-t = {1.0 - tt[i_ref + 1]:.3g} '
          f'(h = {h_ref:.3e}) -> {w:.3e} per unit-noise channel')
    return w


def _fit_or_load_ggd(name, pot, x1, settings, key_extra=''):
    """GGD extra family: the window fit is deterministic in (x1, settings) and ~98% of its time is the
    per-channel BIC window search (scipy, CPU; profiled 2026-11-01), so it is done once and cached in
    moment_reproduction/cache/, keyed by the exact bytes of x1 + filter bank + family + GGD settings (the bank so
    that a deduplicated and a full bank never share a cache file). Same windows as a refit."""
    import hashlib
    h = hashlib.sha1(x1.detach().cpu().numpy().tobytes())
    h.update(pot.filters.detach().cpu().to(torch.complex128).numpy().tobytes())
    h.update(f'{name}|{sorted(settings.items())}'.encode())
    if key_extra:                                         # e.g. the J actually used for a Scalar_Q<q> bank
        h.update(f'|{key_extra}'.encode())
    path = HERE / 'cache' / f'{name}_{h.hexdigest()[:16]}.pt'
    if path.exists():
        print(f'[extra] {name}: windows loaded from {path.name}')
        return Scalar_GGD_KRegion.load_fixed_parameters(path, pot.filters, map_location=x1.device).to(x1.device)
    pot.plot_fit = lambda *a, **k: None                   # fit_reference ends with a per-channel plot: not wanted
    pot.fit(x1)
    path.parent.mkdir(parents=True, exist_ok=True)
    # atomic: parallel SLURM tasks (run_moment_reproduction.py) may fit the same windows at the same time
    import os
    tmp = path.with_name(f'{path.stem}.tmp{os.getpid()}.pt')
    pot.save_fixed_parameters(tmp)
    os.replace(tmp, path)
    print(f'[extra] {name}: windows fitted and saved to {path.name}')
    return pot


# ------------------------------------------------------------------------------------------- compute ----
def compute_moment_reproduction(model, data, synthetic, extra_potentials=None, dataset_config=None,
                                rel_tol=1e-12):
    """model: {name: fitted potential} (imposed). data, synthetic: (n, 1, M) tensors (synthetic = one seed).
    extra_potentials: list of family names (non-imposed), configured from dataset_config (the run's config.json).
    Returns (DataFrame one row per component, DataFrame of excluded components: s2_data <= rel_tol x family median).
    """
    pots = dict(model)
    imposed = set(model)
    if extra_potentials:
        if dataset_config is None:
            raise ValueError('extra_potentials need dataset_config (the run config.json) for M, J, Q')
        new = [n for n in extra_potentials if n not in imposed]          # an imposed family stays imposed
        pots.update(build_extra(new, dataset_config, data))

    phi_d = phi_matrix(pots, data.double())                              # (n, r) float64
    phi_s = phi_matrix(pots, synthetic.double())
    n, n_s = phi_d.shape[0], phi_s.shape[0]
    lab = stat_labels(pots)
    df = pd.DataFrame({'family': [f for f, _, _ in lab], 'label': [l for _, _, l in lab]})
    df['imposed'] = df['family'].isin(imposed)
    m, s2d = phi_d.mean(0), phi_d.var(0, ddof=1)
    mh, s2m = phi_s.mean(0), phi_s.var(0, ddof=1)
    df['s2_data'] = s2d

    fam_med = df.groupby('family')['s2_data'].transform('median').to_numpy()
    dead = s2d <= rel_tol * fam_med
    with np.errstate(divide='ignore', invalid='ignore'):
        df['e'] = (m - mh) ** 2 / s2d
        df['floor'] = (s2d / n + s2m / n_s) / s2d
        df['ratio'] = df['e'] / df['floor']
        # what the plots show (2026-11-01):
        #   offset = sqrt(e) = |m - mhat| / std_data   (main colour: size of the miss, in data std)
        #   noise  = sqrt(floor) = std of (m - mhat) from finite n, n_s alone, same units
        #   within_noise = offset <= NOISE_K x noise    (indistinguishable from a perfect match)
        #   rel_mean = (mhat - m) / |m|, only where |m| >= REL_MEAN_MIN x std_data (else NaN: mean ~ 0,
        #   a percentage of it would blow up)
        df['offset'] = np.sqrt(df['e'])
        df['noise'] = np.sqrt(df['floor'])
        df['within_noise'] = df['offset'] <= NOISE_K * df['noise']
        df['resolved'] = NOISE_K * df['noise'] < RESOLVED_MAX
        # verdict: 'matched' (within noise, sharp), 'unresolved' (within noise, noise too large), 'miss'
        df['verdict'] = np.where(~df['within_noise'], 'miss', np.where(df['resolved'], 'matched', 'unresolved'))
        df['rel_mean'] = np.where(np.abs(m) >= REL_MEAN_MIN * np.sqrt(s2d), (mh - m) / np.abs(m), np.nan)
    return df[~dead].reset_index(drop=True), df[dead].reset_index(drop=True)


# ------------------------------------------------------------------------------------------- plots ----
# miss-size classes of the summary plot (offset in data std), with their colours
MISS_BINS = [(0.0, 0.05, 'tiny miss < 0.05', '#fde8a6'), (0.05, 0.2, 'small 0.05–0.2', '#f6a35c'),
             (0.2, 0.5, 'clear 0.2–0.5', '#d6404e'), (0.5, np.inf, 'big > 0.5', '#4a0c4f')]


_SCAT_ORDER = {'First': 1, 'Second': 2, 'Third': 3, 'Fourth': 4}
_FAMILY_GROUPS = ['moments', 'scalar', 'scattering', 'other']


def family_key(fam):
    """Sort key of a family name: L_p moments, then Scalar_* marginals (bank morlet, psi, Q4, Q5, ...; real
    before imag), then Scattering by order (Real before Imag), then anything else; ties by name."""
    imag = fam.endswith('_imag') or '_Imag' in fam
    if (m := re.fullmatch(r'L_(\d+)', fam)):
        return (0, int(m[1]), 0, fam)
    if fam.startswith('Scalar_'):
        bank = fam.split('_')[1]
        rank = {'morlet': 0, 'psi': 1}.get(bank, int(bank[1:]) + 1 if re.fullmatch(r'Q\d+', bank) else 99)
        return (1, rank, imag, fam)
    if fam.startswith('Scattering_'):
        return (2, _SCAT_ORDER.get(fam.split('_')[1], 9), imag, fam)
    return (3, 0, imag, fam)


def ordered_families(fams):
    return sorted(dict.fromkeys(fams), key=family_key)


def plot_summary(results, title=None):
    """One cell per (model, family): a horizontal bar split by the fraction of that family's statistics that is
    matched (grey), can't tell (hatched), or a miss of each size class (MISS_BINS). Thick black frame = family
    imposed by that model; number = statistics in the cell. results: {model name: DataFrame}.
    One block per category of families (_FAMILY_GROUPS: moments, scalar, scattering, other), stacked vertically, each
    with every model as a row; same cell size in all blocks, so only the categories present are drawn."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    fams = ordered_families(f for df in results.values() for f in df.family)
    models = list(results)
    blocks = {}
    for f in fams:                                           # sorted: each category is contiguous
        g = _FAMILY_GROUPS[family_key(f)[0]]
        blocks.setdefault('moments & other' if g in ('moments', 'other') else g, []).append(f)
    blocks = {g: blocks[g] for g in ('scalar', 'scattering', 'moments & other') if g in blocks}   # small block last
    width = max(len(v) for v in blocks.values())
    row_h = 0.62
    fig, axes = plt.subplots(len(blocks), 1, squeeze=False, figsize=(1.25 * width + 2.5, (row_h * len(models) + 1.5) * len(blocks) + 1.0),
                             gridspec_kw=dict(height_ratios=[len(models) + 2.4] * len(blocks)))
    for ax, (group, bfams) in zip(axes[:, 0], blocks.items()):
        for r, (name, df) in enumerate(results.items()):
            y = len(models) - 1 - r
            for c, fam in enumerate(bfams):
                d = df[df.family == fam]
                if d.empty:
                    ax.text(c + 0.5, y + 0.5, '—', ha='center', va='center', color='0.6')
                    continue
                parts = [((d.verdict == 'matched').mean(), dict(facecolor='0.6')),
                         ((d.verdict == 'unresolved').mean(), dict(facecolor='0.93', edgecolor='0.55', hatch='////'))]
                miss = d[d.verdict == 'miss'].offset
                parts += [(((miss >= lo) & (miss < hi)).sum() / len(d), dict(facecolor=col))
                          for lo, hi, _, col in MISS_BINS]
                x0 = c + 0.05
                for frac, style in parts:
                    if frac > 0:
                        ax.add_patch(Rectangle((x0, y + 0.2), 0.9 * frac, 0.5, lw=0, **style))
                        x0 += 0.9 * frac
                if d.imposed.any():
                    ax.add_patch(Rectangle((c + 0.05, y + 0.2), 0.9, 0.5, fill=False, lw=2.2, edgecolor='k'))
                ax.text(c + 0.5, y + 0.78, f'{len(d)}', ha='center', va='bottom', fontsize=7, color='0.3')
        ax.set_xlim(0, width); ax.set_ylim(0, len(models))   # same cell size in every block, left-aligned
        ax.set_xticks(np.arange(len(bfams)) + 0.5)
        ax.set_xticklabels([f.replace('Scattering_', 'Scat_').replace('_gaussianK', '') for f in bfams],
                           rotation=35, ha='right', fontsize=8)
        ax.set_yticks(np.arange(len(models)) + 0.45)
        ax.set_yticklabels(models[::-1])
        ax.set_title(group, loc='left', fontsize=11, fontweight='bold', color='0.25', pad=14)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    handles = [Patch(facecolor='0.6', label='matched'),
               Patch(facecolor='0.93', edgecolor='0.55', hatch='////', label="can't tell"),
               *[Patch(facecolor=col, label=lab) for _, _, lab, col in MISS_BINS],
               Patch(facecolor='white', edgecolor='k', lw=2.2, label='family imposed')]
    H = fig.get_figheight()                                  # header placed in inches from the top edge
    if title:
        fig.text(0.5, 1 - 0.12 / H, title, ha='center', va='top', fontsize=12)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1 - (0.45 if title else 0.1) / H), ncol=len(handles),
               fontsize=8, frameon=False,
               title="fraction of the family's statistics   (miss size = offset in data std; number = statistics in the family)",
               title_fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 1 - (1.25 if title else 0.9) / H))
    return fig


def plot_strip(df, families=None):
    """One panel per family: offset (log axis) vs component; filled circle = imposed, cross = not imposed;
    grey band = NOISE_K x noise of each component (below it: indistinguishable from a perfect match);
    dashed line = RESOLVED_MAX (a band reaching above it means: can't tell)."""
    import matplotlib.pyplot as plt
    families = families or ordered_families(df.family)
    fig, axes = plt.subplots(len(families), 1, figsize=(12, 2.6 * len(families)), squeeze=False)
    for ax, fam in zip(axes[:, 0], families):
        d = df[df.family == fam]
        y_min = OFFSET_RANGE[0] / 10                    # below this: always deep inside the noise, nothing to read
        x, y = np.arange(len(d)), np.clip(d['offset'].to_numpy(), y_min, None)   # lower points pinned to the edge
        imp = d['imposed'].to_numpy()
        # band from the very bottom: everything below NOISE_K x noise is indistinguishable from a perfect match
        ax.fill_between(x, 1e-300, NOISE_K * d['noise'].to_numpy(), step='mid', color='0.85', lw=0,
                        label=f'within noise (< {NOISE_K:g} × noise)')
        ax.plot(x[imp], y[imp], 'o', ms=4, color='C0', label='imposed')
        ax.plot(x[~imp], y[~imp], 'x', ms=5, color='C3', label='not imposed')
        ax.axhline(RESOLVED_MAX, color='0.4', lw=0.8, ls='--',
                   label=f'{RESOLVED_MAX:g}: sharpness limit (band above it = can\'t tell)')
        ax.set_yscale('log')
        ax.set_ylim(y_min * 0.8, max(2.0, np.nanmax(y) * 1.5))
        ax.set_title(fam, fontsize=9)
        ax.set_ylabel(r'offset / std$_\mathrm{data}$')
        if len(d) <= 60:
            ax.set_xticks(x)
            ax.set_xticklabels([l.split('[', 1)[-1].rstrip(']') for l in d.label], rotation=90, fontsize=6)
        ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    return fig


# ------------------------------------------------------- moment reproduction vs -log p contributions ----
# Same shortening as jets/logp_jets.py (SHORT): the jets log p file names families S_psi, Scat4_Real_Q1, ...; the turbulence
# one uses the full potential names. Both sides are shortened the same way before they are matched.
_SHORT = {'Scattering_Fourth_Order_Mod2_': 'Scat4_', 'Scattering_Third_Order_': 'Scat3_',
          'Scattering_Second_Order': 'Scat2', 'Scattering_First_Order': 'Scat1', 'Scalar_': 'S_', '_gaussianK': ''}
_NOT_FAMILY = ('increment', 'theta_desc', 'TOTAL')


def _canon(name):
    for a, b in _SHORT.items():
        name = name.replace(a, b)
    return name


def load_logp_contributions(label, root, seed=None):
    """Contribution of each potential family to -log p of the data signals, for the run (label, root) at `seed`
    (None = first seed, as load_model). Returns ({family: array (signals,)}, source) or (None, reason).
    -log p(x) = sum_F C_F(x) + const with C_F = -Theta_F . phi_F(x); 'TOTAL' = the sum.
      turbulence: turbulence/figures/data/logp_C_<label>[_seed<s>]_lam<lam>[_unc].npz (logp_family_contrib.py); the
                  newest file with the L_6<0 Theta (no '_unc') is used, else the newest unconstrained one.
      jets:       jets/figures/logp_jets/logp_jets.npz (logp_jets.py), training signals, arm = label."""
    root = Path(root)
    seed = first_seed(label, root) if seed is None else seed
    if root.name == 'jets':
        f = root / 'figures' / 'logp_jets' / 'logp_jets.npz'
        if not f.exists():
            return None, f'{f} not found'
        z = np.load(f, allow_pickle=False)
        if label not in list(z['arms']):
            return None, f'arm {label!r} not in {f.name} (arms {list(z["arms"])})'
        seeds = [int(s) for s in z[f'seeds__{label}']]
        if seed not in seeds:
            return None, f'seed {seed} not in {f.name} for {label} (seeds {seeds})'
        fams = list(z[f'families__{label}'])
        C = -z[f'C__{label}__{seed}__train'].astype(np.float64)           # (families, signals)
        out = dict(zip(fams, C))
        out['TOTAL'] = C.sum(0)
        return out, f'{f.name} [{label}, seed {seed}, lam {float(z[f"lam__{label}"]):g}]'
    stag = '' if seed == 900 else f'_seed{seed}'                          # as logp_family_contrib.py
    found = sorted((root / 'figures' / 'data').glob(f'logp_C_{label}{stag}_lam*.npz'), key=lambda p: p.stat().st_mtime)
    cons = [p for p in found if not p.stem.endswith('_unc')]
    if not found:
        return None, f'no {root.name}/figures/data/logp_C_{label}{stag}_lam*.npz (run logp_family_contrib.py)'
    f = (cons or found)[-1]
    z = np.load(f, allow_pickle=True)
    out = {k: -z[k].astype(np.float64) for k in z.files if k not in ('increment', 'theta_desc')}
    return out, f'{f.name}' + ('' if cons else '  (UNCONSTRAINED: no L_6<0 file)') + (f'  [{len(found)} files match]' if len(found) > 1 else '')


def comparison_table(res, logp):
    """Moment reproduction next to the -log p contribution, per model and family (the families of plot_summary,
    same blocks). res = {model: df of compute_moment_reproduction}; logp = {model: {family: (signals,)} or None}.
    Moment reproduction: number of statistics, % matched / can't tell / miss, mean offset (data std). -log p: mean and
    std over the data signals of the family's contribution C_F (nats; the mean is a level only, std^2 = variance).
    Families the model does not impose have no Theta, hence no contribution: blank. Row 'total' of a model: moment
    columns over all its imposed statistics, -log p columns of the total."""
    rows = []
    for model, df in res.items():
        lp = {_canon(k): v for k, v in (logp.get(model) or {}).items()}
        used = set()
        for fam in ordered_families(df.family):
            d = df[df.family == fam]
            cat = _FAMILY_GROUPS[family_key(fam)[0]]
            cat = 'moments & other' if cat in ('moments', 'other') else cat
            v = lp.get(_canon(fam))
            used.add(_canon(fam))
            rows.append((model, cat, fam, d, v))
        extra = [k for k in lp if k not in used and k != 'TOTAL']
        if extra:
            print(f'{model}: log p families with no moment-reproduction family (not in the table): {extra}')
        rows.append((model, 'total', 'total (imposed)', df[df.imposed], lp.get('TOTAL')))
    cols = pd.MultiIndex.from_tuples([('moment reproduction', c) for c in ('n', 'matched %', "can't tell %", 'miss %', 'mean offset')]
                                     + [('−log p', 'mean'), ('−log p', 'std')] + [('', 'imposed')])
    order = {'scalar': 0, 'scattering': 1, 'moments & other': 2, 'total': 3}
    data, idx = [], []
    for model, cat, fam, d, v in rows:
        n = len(d)
        pct = (lambda s: 100 * (d.verdict == s).mean()) if n else (lambda s: np.nan)
        data.append([n, pct('matched'), pct('unresolved'), pct('miss'), d.offset.mean() if n else np.nan,
                     float(np.mean(v)) if v is not None else np.nan, float(np.std(v, ddof=1)) if v is not None else np.nan,
                     '✓' if n and d.imposed.any() else ''])
        idx.append((model, cat, fam.replace('Scattering_', 'Scat_').replace('_gaussianK', '')))
    T = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(idx, names=['model', 'block', 'family']), columns=cols)
    mo = {m: i for i, m in enumerate(dict.fromkeys(m for m, *_ in idx))}
    key = [(mo[m], order[b], i) for i, (m, b, _) in enumerate(idx)]            # model, block, original (family) order
    return T.iloc[[i for *_, i in sorted(key)]]


# ----------------------------------------- the talk's energy table, with moment reproduction added (per category) ----
# The talk's categories (codes/talk_figures.ENERGY_CATEGORIES), with the marginals of the imaginary parts split off the
# scalar column. Only the families the model IMPOSES count (they are the ones with a Theta, hence a -log p contribution).
TABLE_CATEGORIES = (('scalar multiscale', r'^(Scalar_(?!.*imag)|Scattering_(First|Second)_Order)'),
                    ('scalar Im', r'^Scalar_.*imag'),
                    ('order 3 Re', r'Third_Order_Real'), ('order 3 Im', r'Third_Order_Imag'),
                    ('order 4 Re', r'Fourth_Order.*Real'), ('order 4 Im', r'Fourth_Order.*Imag'),
                    ('confinement', r'^L_?6$'))


def category_table(df, lp):
    """One column per TABLE_CATEGORIES entry, in the format of talk_figures.energy_table (+ two moment-reproduction rows).
    df = compute_moment_reproduction result of one model, lp = {family: -log p contribution per signal} from
    load_logp_contributions (or None). Per category, over the model's imposed families: number of statistics, % matched,
    mean offset (data std); -log p = sum of the families' contributions, then mean and std over the data signals
    (the std is of the sum, not a sum of stds). '± ...' rows are NaN (one seed)."""
    imp = df[df.imposed]
    lpc = {_canon(k): v for k, v in (lp or {}).items()}
    cols = [c for c, _ in TABLE_CATEGORIES]
    out = {r: [] for r in ('number of moments', 'moments matched (%)', 'mean offset (data std)', 'mean of -log p (nats)', '± mean',
                           'std of -log p over signals (nats)', '± std')}
    for cat, pat in TABLE_CATEGORIES:
        fams = [f for f in dict.fromkeys(imp.family) if re.search(pat, f)]
        d = imp[imp.family.isin(fams)]
        out['number of moments'].append(len(d))
        out['moments matched (%)'].append(100 * (d.verdict == 'matched').mean() if len(d) else np.nan)
        out['mean offset (data std)'].append(d.offset.mean() if len(d) else np.nan)
        have = [_canon(f) in lpc for f in fams]
        if fams and not all(have):
            print(f'  {cat}: no log p contribution for {[f for f, h in zip(fams, have) if not h]} -> -log p left blank')
        if fams and all(have):
            E = sum(np.asarray(lpc[_canon(f)], dtype=np.float64) for f in fams)
            out['mean of -log p (nats)'].append(float(E.mean()))
            out['std of -log p over signals (nats)'].append(float(E.std(ddof=1)))
        else:
            out['mean of -log p (nats)'].append(np.nan)
            out['std of -log p over signals (nats)'].append(np.nan)
        out['± mean'].append(np.nan)
        out['± std'].append(np.nan)
    return pd.DataFrame(out, index=cols).T


def plot_moment_vs_logp(res, logp, name):
    """The talk's energy table, one block per model of `res` ({model: df}); `logp` = {model: {family: contribution} or
    None}. Rows: number of moments, % matched, mean offset, mean and std of -log p, per category (columns)."""
    from codes import talk_figures as tf                      # sets the talk plot style (rcParams) on first import
    tf.plot_energy_table({m: category_table(df, logp.get(m)) for m, df in res.items()}, name)
