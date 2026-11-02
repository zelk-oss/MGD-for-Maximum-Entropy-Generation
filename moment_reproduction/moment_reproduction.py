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
    dedup = bool(cfg.get('deduplicate_filters'))
    tol = float(cfg.get('dedup_tol') or 1e-12)
    out = {}
    for name in names:
        if (m := SCALAR_Q_RE.match(name)):
            # A Q>=4 Morlet bank cannot be built at the run's J when its coarsest wavelets are too narrow in
            # frequency for M (M=256: Q=4/5 fail at J=8 with filters_1d's "Zero division" ValueError, work at
            # J=7; checked 2026-11-02): use the largest J that works and say so.
            q, Jq = int(m[1]), J
            while True:
                try:
                    f = return_Filters(M, Jq, q, device=dev, wav_type=wav)
                    break
                except ValueError:
                    if Jq <= 1:
                        raise
                    Jq -= 1
            if Jq != J:
                print(f'[extra] {name}: Q={q} bank not buildable at J={J} (M={M}); using J={Jq}, {f.shape[1]} channels')
            pot = Scalar_GGD_KRegion(deduplicate_filters(f, tol)[0] if dedup else f, **EXTRA_GGD)
            pot.to(dev)
            out[name] = _fit_or_load_ggd(name, pot, x1, key_extra=f'J{Jq}')
            continue
        else:
            built = get_1d_potentials([name], J, filters, Q, filters_Q=filters_Q, filters_Phi=fphi,
                                      deduplicate_filters=dedup, dedup_tol=tol, ggd_kwargs=EXTRA_GGD)
            if name not in built:
                raise ValueError(f'{name!r} is not a get_1d_potentials term nor Scalar_Q<q>_gaussianK')
            pot = built[name]
        pot.to(dev)
        if isinstance(pot, Scalar_GGD_KRegion):
            out[name] = _fit_or_load_ggd(name, pot, x1)
        else:
            pot.fit(x1)                                   # scattering / L_p: no-op
            out[name] = pot
    return out


def _fit_or_load_ggd(name, pot, x1, key_extra=''):
    """GGD extra family: the window fit is deterministic in (x1, settings) and ~98% of its time is the
    per-channel BIC window search (scipy, CPU; profiled 2026-11-01), so it is done once and cached in
    moment_reproduction/cache/, keyed by the exact bytes of x1 + family + EXTRA_GGD. Same windows as a refit."""
    import hashlib
    h = hashlib.sha1(x1.detach().cpu().numpy().tobytes())
    h.update(f'{name}|{sorted(EXTRA_GGD.items())}'.encode())
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


def plot_summary(results, title=None):
    """One cell per (model, family): a horizontal bar split by the fraction of that family's statistics that is
    matched (grey), can't tell (hatched), or a miss of each size class (MISS_BINS). Thick black frame = family
    imposed by that model; number = statistics in the cell. results: {model name: DataFrame}."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    fams = list(dict.fromkeys(f for df in results.values() for f in df.family))
    models = list(results)
    fig, ax = plt.subplots(figsize=(1.25 * len(fams) + 2.5, 0.75 * len(models) + 2.2))
    for r, (name, df) in enumerate(results.items()):
        y = len(models) - 1 - r
        for c, fam in enumerate(fams):
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
    ax.set_xlim(0, len(fams)); ax.set_ylim(0, len(models))
    ax.set_xticks(np.arange(len(fams)) + 0.5)
    ax.set_xticklabels([f.replace('Scattering_', 'Scat_').replace('_gaussianK', '') for f in fams],
                       rotation=35, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(models)) + 0.45)
    ax.set_yticklabels(models[::-1])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [Patch(facecolor='0.6', label='matched'),
               Patch(facecolor='0.93', edgecolor='0.55', hatch='////', label="can't tell"),
               *[Patch(facecolor=col, label=lab) for _, _, lab, col in MISS_BINS],
               Patch(facecolor='white', edgecolor='k', lw=2.2, label='family imposed')]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=len(handles), fontsize=7,
              frameon=False, title='fraction of the family\'s statistics   (miss size = offset in data std; '
                                   'number = statistics in the family)', title_fontsize=7)
    if title:
        ax.set_title(title, pad=40)
    fig.tight_layout()
    return fig


def plot_strip(df, families=None):
    """One panel per family: offset (log axis) vs component; filled circle = imposed, cross = not imposed;
    grey band = NOISE_K x noise of each component (below it: indistinguishable from a perfect match);
    dashed line = RESOLVED_MAX (a band reaching above it means: can't tell)."""
    import matplotlib.pyplot as plt
    families = families or list(dict.fromkeys(df.family))
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
