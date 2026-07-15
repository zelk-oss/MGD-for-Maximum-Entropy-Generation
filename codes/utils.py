"""
Moment-Guided Diffusion (MGD) - utilities.
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Dict, Any, Tuple
import sys 

_UTILS_DIR = Path(__file__).resolve().parent
_UTILS_PROJECT_ROOT = _UTILS_DIR.parent
 
def _extend_syspath(root: Path, project_root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    codes_path = project_root / 'codes'
    if codes_path.is_dir() and str(codes_path) not in sys.path:
        sys.path.insert(0, str(codes_path))
    data_path = project_root / 'data'
    if data_path.is_dir() and str(data_path) not in sys.path:
        sys.path.insert(0, str(data_path))

_extend_syspath(_UTILS_DIR, _UTILS_PROJECT_ROOT)


from codes.check_moments import *     # noqa: E402
from codes.ortho_wavelet.ReadyToUseWavelets import *

import math
import pandas as pd


class TensorDataset(torch.utils.data.Dataset):
    """ We have to create our own class because PyTorch's TensorDataset returns lists... """

    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item]

def add_noise(x, t):
    """Apply Ornstein-Uhlenbeck noising at time ``t``.
 
    Returns ``x_t = e^{-t} x + sqrt(1 - e^{-2t}) z`` with ``z`` standard Gaussian.
 
    Parameters
    ----------
    x : torch.Tensor
        Clean input.
    t : float
        Noise time.
 
    Returns
    -------
    x_t : torch.Tensor
        Noised input.
    z : torch.Tensor
        The Gaussian noise used.
    """

    e_minus_t = np.exp(-t)
    std = np.sqrt(1 - e_minus_t ** 2)

    #torch.manual_seed(13)
    z = torch.randn_like(x)
    x_t = e_minus_t * x + std * z

    return x_t, z


def symmetrize_functional(x):
    """
    Functional version (no class needed).
    
    Args:
        x: Tensor of shape (B, C, M, N)
    Returns:
        Symmetrized tensor of shape (B, C, M*2, N*2)
    """

    # Create the 4 quadrants
    top = torch.cat([x, torch.flip(x, dims=[3])], dim=3)
    bottom = torch.cat([torch.flip(x, dims=[2]), torch.flip(x, dims=[2, 3])], dim=3)
    return torch.cat([top, bottom], dim=2)

def save_results(
    xt,
    theta_t,
    dH_t_bound,
    t,
    root,
    config,
):
    base = root / 'saved_results'

    torch.save(xt.cpu(), base / 'samples' / config)
    torch.save(theta_t.cpu(), base / 'lagrange_multipliers' / config)
    torch.save(dH_t_bound.cpu(), base / 'entropy_bounds' / config)
    torch.save(t.cpu(), base / 'sampling_times' / config)

def save_results_theta_reg(
    xt,
    theta_t,
    dH_t_bound,
    t,
    root,
    config,
    Theta_reg=None,  # Set to None by default
):
    base = root / 'saved_results'

    # 1. Save the core variables first to secure them on disk
    torch.save(xt.cpu(), base / 'samples' / config)
    torch.save(theta_t.cpu(), base / 'lagrange_multipliers' / config)
    torch.save(dH_t_bound.cpu(), base / 'entropy_bounds' / config)
    torch.save(t.cpu(), base / 'sampling_times' / config)

    # 2. Safely process Theta_reg only if it was passed in
    if Theta_reg is not None:
        try:
            torch.save(
                Theta_reg.cpu(),
                base / 'lagrange_multipliers_regularised' / config
            )
        except RuntimeError as e:
            # Catch the OOM (or any other PyTorch error) so the run doesn't crash entirely
            print(f"Warning: Failed to save Theta_reg. Error: {e}")

def load_results(root: Path, exact_config: str) -> Tuple[Any, Any, Any, Any, Any]:
    base = root / 'saved_results'

    x_t = torch.load(base / 'samples' / exact_config)
    theta_t = torch.load(base / 'lagrange_multipliers' / exact_config)
    dH_t_bound = torch.load(base / 'entropy_bounds' / exact_config)
    t = torch.load(base / 'sampling_times' / exact_config)

    path_theta_reg = base / 'lagrange_multipliers_regularised' / exact_config

    if path_theta_reg.exists():
        Theta_reg = torch.load(path_theta_reg)
    else: 
        Theta_reg = None # Always return 5 items to prevent unpacking crashes
        
    return (x_t, theta_t, dH_t_bound, t, Theta_reg)

def normalize(Data):
    """Standardize ``Data`` to zero mean and unit std, cast to float32.
 
    Parameters
    ----------
    Data : torch.Tensor
        Input tensor.
 
    Returns
    -------
    torch.Tensor
        Normalized float32 tensor.
    """

    Data = (Data-Data.mean())/Data.std()
    Data = Data.to(torch.float32)
    return Data


def split_periodize_reshape(Data, n1):
    """
    Splits Data into non-overlapping subseries of length n1 along the last axis,
    discards the end if not divisible, periodizes each subseries, and reshapes
    the output to [batch * num_subseries, channels, n1].
    """
    num_subseries = Data.size(-1) // n1
    Data_sub = Data[:, :, :num_subseries * n1]
    Data_sub = Data_sub.unfold(-1, n1, n1)

    first = Data_sub[..., 0:1]
    last = Data_sub[..., -1:]
    a = (last - first) / (n1 - 1)
    b = first
    x = torch.arange(n1, device=Data.device, dtype=Data.dtype)
    linear = a * x + b 
    Data_periodized = Data_sub - linear 
    Data_periodized = Data_periodized + first 

    batch, channels, _, _ = Data_periodized.shape
    return Data_periodized.reshape(batch * num_subseries, channels, n1)



"""
theta_analysis.py

Analysis utilities for Scattering_Fourth_Order_{Real,Imag,Mod2_Real,Mod2_Imag}
theta coefficients produced by run_experiment().

Harmonized with the current notebook structure:
    result = run_experiment(args, M, config, x1, filters, t, logger, root, device=device)
    potentials = get_1d_potentials(terms, J, filters, Q, scalar_param=None, parallel=False)
There is no `experiments`/`results` dict in the current notebook -- use `result`
and `potentials` directly.

Usage:
    from theta_analysis import build_theta_dataframe, plot_xi_decay, ...

    df, extras = build_theta_dataframe(
        result, potentials, target_key="Scattering_Fourth_Order_Mod2_Real_Q1",
        x1=x1, filters=filters, burnin=5000, norm_order=4,
    )
    plot_xi_decay(df)
    plot_theta_vs_b(df)
    plot_theta_vs_a_fixed_b(df, value_col="theta_raw_mean")
    plot_index_structure(df, J=J)
    plot_theta_heatmap(df, value_col="theta_raw_mean")
    plot_theta_grouped_by_j(df)

norm_order: use 2 for Real/Imag potentials (order-2 micro-norm, theta_physical
= theta_fitted * sqrt(E1*E2)); use 4 for Mod2_Real/Mod2_Imag potentials
(theta_physical = theta_fitted * E1*E2).
"""

# ---------------------------------------------------------------------
# 1. Column mapping: potentials dict -> theta_t column slices
# ---------------------------------------------------------------------
def _potential_width(p, key, known_dims=None):
    """(dim, source) for potential p. Never silently assumes dim=1."""
    known_dims = known_dims or {}
    for attr in ('n_theta', 'num_theta', 'theta_dim'):
        if hasattr(p, attr):
            val = getattr(p, attr)
            val = val() if callable(val) else val
            if torch.is_tensor(val):
                val = val.numel()
            return int(val), f"'{attr}'"
    if hasattr(p, 'num_coefficients'):
        return int(p.num_coefficients), "'num_coefficients'"
    if hasattr(p, 'indices'):
        idx = p.indices
        width = idx.shape[-1] if torch.is_tensor(idx) else np.asarray(idx).shape[-1]
        return int(width), "'.indices.shape[-1]'"
    for attr in ('num_regions', 'n_regions'):
        if hasattr(p, attr):
            n_reg = int(getattr(p, attr))
            n_par = int(getattr(p, 'n_params_per_region', 1))
            return n_reg * n_par, f"'{attr}' x n_params_per_region"
    if key in known_dims:
        return int(known_dims[key]), "known_dims override (UNVERIFIED by introspection)"
    raise RuntimeError(f"Cannot determine theta width for '{key}' ({type(p).__name__}).")


def theta_column_map(potentials, theta_t_np, known_dims=None, verbose=True):
    """
    dict: key -> (start, dim) column slice into theta_t, for every potential
    in `potentials` (dict, insertion-order must equal theta_t column order,
    which is how get_1d_potentials + run_experiment build it).
    """
    keys = list(potentials.keys())
    dims, sources = [], []
    for k in keys:
        d, src = _potential_width(potentials[k], k, known_dims)
        dims.append(d)
        sources.append(src)
    offsets = np.cumsum([0] + dims)
    n_theta = theta_t_np.shape[1]
    if sum(dims) != n_theta:
        lines = "\n".join(f"  {k:35s} {d:5d}  {s}" for k, d, s in zip(keys, dims, sources))
        raise AssertionError(f"sum(dims)={sum(dims)} != theta_t.shape[1]={n_theta}\n{lines}")
    if verbose:
        for k, d, o, s in zip(keys, dims, offsets[:-1], sources):
            print(f"{k:35s} {d:5d}  [{o:4d}:{o+d:<4d}]  {s}")
    return {k: (int(o), int(d)) for k, o, d in zip(keys, offsets[:-1], dims)}


# ---------------------------------------------------------------------
# 2. Load thetas for one target potential
# ---------------------------------------------------------------------
def load_theta_for_potential(result, potentials, target_key, known_dims=None, verbose=True):
    """
    Slice theta_t (from `result['theta_t']`) down to the columns of `target_key`.
    Returns (theta_raw: np.ndarray (T, dim), potential, start, dim).
    Handles theta_t on any device / dtype.
    """
    theta_t = result['theta_t']
    theta_t_np = theta_t.detach().cpu().numpy() if torch.is_tensor(theta_t) else np.asarray(theta_t)

    col_map = theta_column_map(potentials, theta_t_np, known_dims, verbose)
    if target_key not in col_map:
        raise KeyError(f"'{target_key}' not in potentials: {list(potentials.keys())}")
    start, dim = col_map[target_key]

    potential = potentials[target_key]
    if hasattr(potential, 'num_coefficients') and hasattr(potential, 'indices'):
        assert dim == potential.num_coefficients == potential.indices.shape[-1], (
            f"dim mismatch: col_map={dim}, num_coefficients={potential.num_coefficients}, "
            f"indices={potential.indices.shape[-1]}"
        )
    theta_raw = theta_t_np[:, start:start + dim]
    if verbose:
        print(f"-> {target_key}: cols [{start}:{start + dim}] out of {theta_t_np.shape[1]}")
    return theta_raw, potential, start, dim


# ---------------------------------------------------------------------
# 3. Decode (j, a, b) indices for a 4th-order potential
# ---------------------------------------------------------------------
def decode_jab(potential, j_row=None, verbose=True):
    """
    potential.indices is (3, n_coeff): one row is the shared first-layer
    scale j, the other two are absolute second-layer scales s1,s2. Detect
    which row is j via the combinatorial fingerprint for offset=1, lite=True,
    no-diag: #pairs at fixed j == C(J-j, 2). Pass j_row explicitly to skip
    detection / override a bad guess.
    Returns j, a, b, s1, s2 (np.ndarray, a=min(s)-j < b=max(s)-j), j_row.
    """
    idx = potential.indices.long().detach().cpu()
    J = potential.J

    if j_row is None:
        expected = {j: math.comb(J - j, 2) for j in range(J)}
        expected = {j: c for j, c in expected.items() if c > 0}
        row_vc = []
        for r in range(3):
            vals = idx[r].numpy()
            uniq, counts = np.unique(vals, return_counts=True)
            row_vc.append(dict(zip(uniq.tolist(), counts.tolist())))
        j_row = next((r for r in range(3) if row_vc[r] == expected), None)
        if j_row is None:
            def mismatch(vc):
                ks = set(vc) | set(expected)
                return sum(abs(vc.get(k, 0) - expected.get(k, 0)) for k in ks)
            scores = [mismatch(vc) for vc in row_vc]
            j_row = int(np.argmin(scores))
            if verbose:
                print(f"WARNING: no exact fingerprint match, best guess j_row={j_row} "
                      f"(mismatch scores {scores}); pass j_row= explicitly if this is wrong.")
        elif verbose:
            print(f"-> detected j_row={j_row} from combinatorial fingerprint.")

    other = [r for r in range(3) if r != j_row]
    j = idx[j_row].numpy()
    s1_raw, s2_raw = idx[other[0]].numpy(), idx[other[1]].numpy()
    lo, hi = np.minimum(s1_raw, s2_raw), np.maximum(s1_raw, s2_raw)
    a, b = lo - j, hi - j

    if verbose and not ((a >= 1).all() and (b > a).all()):
        print(f"WARNING: offset/lite assumptions violated: {(a < 1).sum()} entries a<1, "
              f"{(b <= a).sum()} entries b<=a. j_row={j_row} may be wrong -- check manually.")
    return j, a, b, lo, hi, j_row


# ---------------------------------------------------------------------
# 4. Cross-scale normalization theta_fitted -> theta_physical
# ---------------------------------------------------------------------
def compute_norm_jab(x1, filters, s1_arr, s2_arr, order=2):
    """
    norm_jab[i] = (E[|Wx_s1|^2] * E[|Wx_s2|^2]) ** (order/4), Wx_s = single-
    layer wavelet transform of x1 at absolute scale s, using the same J+1-
    channel `filters` bank the potential's second layer uses.
    order=2 -> sqrt(E1*E2)  (Real/Imag potentials, micro-norm on |x|)
    order=4 -> E1*E2        (Mod2 potentials, micro-norm on |x|^2)
    Runs on x1's device; filters moved to match, no_grad, no side effects.
    """
    filters = filters.to(x1.device)
    with torch.no_grad():
        Wx_full = torch.fft.ifft(filters * torch.fft.fft(x1))                    # (B, J+1, T)
        E_scale = (Wx_full.abs() ** 2).mean(dim=(0, 2)).detach().cpu().numpy()   # (J+1,)
    s1_arr, s2_arr = np.asarray(s1_arr), np.asarray(s2_arr)
    return (E_scale[s1_arr] * E_scale[s2_arr]) ** (order / 4)


# ---------------------------------------------------------------------
# 5. Build one tidy DataFrame: index structure + cleaned/normalized theta
# ---------------------------------------------------------------------
def build_theta_dataframe(result, potentials, target_key, x1, filters,
                           burnin=5000, norm_order=2, j_row=None,
                           known_dims=None, verbose=True):
    """
    Single entry point. Loads theta columns for `target_key`, decodes
    (j,a,b), drops burn-in + non-finite rows, de-normalizes, and returns:
      df      : one row per (j,a,b) coefficient, columns
                j, a, b, s1, s2, norm_jab, theta_raw_mean, theta_mean,
                theta_std, cv, reliable
      extras  : dict with theta_scat (per-kept-timestep, physical units),
                keep (bool mask into original T), potential, start, dim, j_row
    """
    theta_raw, potential, start, dim = load_theta_for_potential(
        result, potentials, target_key, known_dims, verbose)
    j_arr, a_arr, b_arr, s1_arr, s2_arr, j_row = decode_jab(potential, j_row, verbose)

    finite_mask = np.isfinite(theta_raw)
    bad_rows = np.where((~finite_mask).any(axis=1))[0]
    keep = np.ones(theta_raw.shape[0], dtype=bool)
    keep[:burnin] = False
    keep[bad_rows] = False
    if verbose:
        print(f"dropping {(~keep).sum()} rows ({burnin} burn-in + "
              f"{int(np.sum(bad_rows >= burnin))} non-finite past burn-in), "
              f"{keep.sum()} rows kept")
    if keep.sum() == 0:
        raise RuntimeError("No rows survive burn-in + finite filtering -- check burnin/theta_t.")

    norm_jab = compute_norm_jab(x1, filters, s1_arr, s2_arr, order=norm_order)

    theta_raw_kept = theta_raw[keep]
    theta_raw_mean = theta_raw_kept.mean(axis=0)
    theta_scat = theta_raw_kept * norm_jab[None, :]      # physical units, per-timestep
    theta_mean = theta_scat.mean(axis=0)
    theta_std = theta_scat.std(axis=0)
    cv = theta_std / (np.abs(theta_mean) + 1e-12)

    df = pd.DataFrame({
        'j': j_arr, 'a': a_arr, 'b': b_arr, 's1': s1_arr, 's2': s2_arr,
        'norm_jab': norm_jab,
        'theta_raw_mean': theta_raw_mean,
        'theta_mean': theta_mean,
        'theta_std': theta_std,
        'cv': cv,
        'reliable': cv < 0.2,
    })
    extras = dict(theta_scat=theta_scat, keep=keep, potential=potential,
                  start=start, dim=dim, j_row=j_row)
    return df, extras


# ---------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------
def plot_xi_decay(df, value_col='theta_mean'):
    """Fit theta_{j,a,b} ~ exp(-xi_{j,a}*b) per (j,a); plot |xi| vs j."""
    xi = {}
    for (j, a), g in df.groupby(['j', 'a']):
        g = g.sort_values('b')
        vals = g[value_col].abs().values
        bs = g['b'].values
        mask = (vals > 0) & np.isfinite(vals)
        if mask.sum() < 2:
            continue
        slope, _ = np.polyfit(bs[mask], np.log(vals[mask]), 1)
        xi[(j, a)] = -slope

    plt.figure(figsize=(6, 4))
    for a in sorted({a for (_, a) in xi}):
        js = sorted(j for (j, aa) in xi if aa == a)
        xis = [xi[(j, a)] for j in js]
        kw = dict(marker='o', label=f'a={a}')
        if a == 1:
            kw.update(linewidth=2.5, markersize=7, zorder=10)
        plt.plot(js, np.abs(xis), **kw)
    plt.xlabel('scale j'); plt.ylabel(r'$|\xi_{j,a}|$')
    plt.title(r'Decay rate of $\theta_{j,a,b}$ in $b$, vs $j$')
    plt.legend(fontsize=8); plt.tight_layout(); plt.show()
    return xi


def plot_theta_vs_b(df, value_col='theta_mean', signed=False):
    """One subplot per j: theta_{j,a,b} vs b, one line per a."""
    js_unique = sorted(df['j'].unique())
    fig, axes = plt.subplots(1, len(js_unique), figsize=(4 * len(js_unique), 3.5),
                              sharey=True, squeeze=False)
    for ax, j in zip(axes[0], js_unique):
        sub = df[df['j'] == j]
        for a, g in sub.groupby('a'):
            g = g.sort_values('b')
            if signed:
                ax.plot(g['b'], g[value_col].values, 'o-', label=f'a={a}')
            else:
                ax.semilogy(g['b'], g[value_col].abs().values, 'o-', label=f'a={a}')
        if signed:
            ax.axhline(0, color='k', ls='--', lw=0.8)
        ax.set_title(f'j={j}'); ax.set_xlabel('b')
    ylab = r'$\theta_{j,a,b}$' if signed else r'$|\theta_{j,a,b}|$'
    axes[0, 0].set_ylabel(ylab)
    axes[0, 0].legend(fontsize=7)
    plt.tight_layout(); plt.show()


def plot_theta_vs_a_fixed_b(df, value_col='theta_raw_mean'):
    """For each b: theta vs a (scale separation), one line per j."""
    for b, sub in df.groupby('b'):
        plt.figure(figsize=(6, 4))
        for j, g in sub.groupby('j'):
            g = g.sort_values('a')
            plt.plot(g['a'], g[value_col].abs(), 'o-', label=f'j={j}')
        plt.axhline(0, color='k', ls='--', lw=0.8)
        plt.yscale('log')
        plt.xlabel('a (scale separation)'); plt.ylabel(rf'$|{value_col}|$')
        plt.title(f'Coupling vs separation a, fixed b={b}')
        plt.legend(fontsize=7); plt.tight_layout(); plt.show()


def plot_index_structure(df, J):
    """Diagnostic: which (a,b) pairs exist per j (pure index structure)."""
    js_unique = sorted(df['j'].unique())
    fig, axes = plt.subplots(1, len(js_unique), figsize=(2.6 * len(js_unique), 3),
                              sharex=True, sharey=True, squeeze=False)
    for ax, j in zip(axes[0], js_unique):
        sub = df[df['j'] == j]
        ax.scatter(sub['a'], sub['b'], s=30)
        ax.plot([0, J], [0, J], 'k--', lw=0.6)
        ax.set_title(f'j={j}  (n={len(sub)})'); ax.set_xlabel('a')
    axes[0, 0].set_ylabel('b')
    plt.suptitle('Which (a,b) pairs exist per j  (s1=j+a, s2=j+b, both <= J)')
    plt.tight_layout(); plt.show()


def plot_theta_heatmap(df, value_col='theta_raw_mean'):
    """Per-a heatmap of |theta| over (j,b)."""
    max_j = int(df['j'].max()) + 1
    max_b = int(df['b'].max()) + 1
    unique_a = sorted(df['a'].unique())
    fig, axes = plt.subplots(1, len(unique_a), figsize=(4 * len(unique_a), 3.5), squeeze=False)
    for idx, a in enumerate(unique_a):
        grid = np.full((max_j, max_b), np.nan)
        sub = df[df['a'] == a]
        for _, row in sub.iterrows():
            j, b, v = int(row['j']), int(row['b']), abs(row[value_col])
            if np.isfinite(v):
                grid[j, b] = v
        ax = axes[0, idx]
        if not (np.isfinite(grid).any() and np.nanmax(grid) > 0):
            ax.set_title(f'Offset a = {a} (no valid data)')
            continue
        im = ax.imshow(grid, origin='lower', cmap='viridis',
                        norm=LogNorm(vmin=np.nanmin(grid[grid > 0]), vmax=np.nanmax(grid)),
                        aspect='auto')
        ax.set_title(f'Offset a = {a}'); ax.set_xlabel('Relative Scale b')
        if idx == 0:
            ax.set_ylabel('First-layer Scale j')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(rf'Coupling Magnitude Topology $|{value_col}|$', y=1.02)
    plt.tight_layout(); plt.show()


def plot_theta_grouped_by_j(df, value_col='theta_mean'):
    """Mean_b |theta| vs j, one line per a, + overall mean across a."""
    df = df.copy()
    df['abs_theta'] = df[value_col].abs()
    mean_over_b = df.groupby(['j', 'a'])['abs_theta'].mean().unstack('a')
    plt.figure(figsize=(6, 4))
    for a in mean_over_b.columns:
        plt.semilogy(mean_over_b.index, mean_over_b[a], 'o-', label=f'a={a}')
    plt.semilogy(mean_over_b.index, mean_over_b.mean(axis=1), 'ko-', lw=2.5, label='overall mean')
    plt.xlabel('scale j'); plt.ylabel(r'mean$_b\,|\theta|$')
    plt.legend(fontsize=7); plt.tight_layout(); plt.show()
