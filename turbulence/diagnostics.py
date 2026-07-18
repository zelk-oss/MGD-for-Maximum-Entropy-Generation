import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats, ndimage
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import to_rgb
from pathlib import Path
import sys
import copy

root = Path().resolve()

sys.path.insert(0, str(root / '../codes'))
from potentials_builder import *
from filters_bank import * 
from utils import *
from utils_experiment import * 
from utils_entropy import *
from check_moments import *
from potentials import *
from filters import *
from mala import *
from ortho_wavelet import *
from utils_entropy import *

sys.path.insert(0, str(root / '../data'))
from data_loader import *
# ================================================================================
# I/O helpers
# ================================================================================


def load_all_experiments(
    base_args: Any,
    results: Dict[str, Dict],
    M: int,
    root: Path,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    """
    Dynamically resolve and load every experiment defined in ``results``.
    """
    results = {}
    
    for key, overrides in results.items():
        print(f"Loading {key} ...")
        
        # 1. Copy base args and apply the specific overrides for this run
        current_args = copy.copy(base_args)
        for k, v in overrides.items():
            setattr(current_args, k, v)
            
        # 2. Build the config prefix (exclude timestamp so we can search the directory)
        current_args.timestamp = "" 
        config_prefix = build_config_name(current_args, M=M, include_timestamp=False)
        
        # 3. Resolve the exact folder/file name (handles the timestamp matching)
        resolved_config = resolve_config_for_loading(root, config_prefix)
        
        if not resolved_config:
            print(f"  -> ⚠️ Warning: No matching runs found for prefix:\n     {config_prefix}\n")
            continue
            
        # 4. Load the data using your existing SDE utility
        loaded_data = try_load_experiment(root, resolved_config, device)
        
        if loaded_data and loaded_data.get('loaded', False):
            # Save the resolved config string into the dictionary for plotting labels
            loaded_data['config'] = resolved_config 
            
            # Alias Theta_reg to theta_reg just in case your old plotting code expects lowercase
            if 'Theta_reg' in loaded_data:
                loaded_data['theta_reg'] = loaded_data['Theta_reg']
            
            results[key] = loaded_data
            print(f"  -> Successfully loaded: {resolved_config}\n")
        else:
            print(f"  -> ❌ Failed to read data files for {resolved_config}\n")
            
    print("All experiments processed.")
    return results


# ================================================================================
# Internal helpers
# ================================================================================

def _to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _average_power_spectrum(x: np.ndarray) -> np.ndarray:
    """Power spectrum averaged over batch and channels, shape (freq,)."""
    fft = np.fft.rfft(x, axis=-1)
    return np.abs(fft) ** 2


def _structure_functions(x: np.ndarray, qs=(1, 2, 3, 4), lags=None):
    T = x.shape[-1]
    if lags is None:
        max_lag = max(2, T // 4)
        lags = np.unique(np.logspace(0, np.log10(max_lag), 20).astype(int))
    sf = np.zeros((len(qs), len(lags)))
    for j, lag in enumerate(lags):
        inc = np.abs(x[..., lag:] - x[..., :-lag])
        for i, q in enumerate(qs):
            sf[i, j] = np.mean(inc ** q)
    return np.asarray(qs), np.asarray(lags), sf


def _cross_structure_function(data: np.ndarray, pq=((2, 2),), max_tau: int = 10):
    """
    Cross structure function S_{p,q}(tau1, tau2) = <|du1|^p |du2|^q>.
    Returns array of shape (len(pq), max_tau-1, max_tau-1).
    """
    taus = np.arange(1, max_tau)
    out = np.zeros((len(pq), len(taus), len(taus)))
    for i, ti in enumerate(taus):
        for j, tj in enumerate(taus):
            di = data[..., ti:] - data[..., :-ti]
            dj = data[..., tj:] - data[..., :-tj]
            L = min(di.shape[-1], dj.shape[-1])
            for k, (p, q) in enumerate(pq):
                out[k, i, j] = (np.abs(di[..., :L]) ** p
                                 * np.abs(dj[..., :L]) ** q).mean()
    return out


def _relative_mse(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.mean((a - b) ** 2) / (np.mean(a ** 2) + eps))


def _shade(color, frac: float):
    """Lighten color gently toward white; frac=1 → base color."""
    r, g, b = to_rgb(color)
    t = 0.30 * (1.0 - frac)
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


# ================================================================================
# Scalar metrics
# ================================================================================

def compute_metrics(x_ref: torch.Tensor, x_gen: torch.Tensor) -> Dict[str, float]:
    """Return a dict of scalar comparison metrics."""
    ref = _to_numpy(x_ref)
    gen = _to_numpy(x_gen)
    ref_flat = ref.reshape(ref.shape[1], -1)
    gen_flat = gen.reshape(gen.shape[1], -1)

    mean_err = (np.linalg.norm(gen_flat.mean(1) - ref_flat.mean(1))
                / (np.linalg.norm(ref_flat.mean(1)) + 1e-12))
    std_err  = (np.linalg.norm(gen_flat.std(1) - ref_flat.std(1))
                / (np.linalg.norm(ref_flat.std(1)) + 1e-12))

    ps_ref = _average_power_spectrum(ref).mean(axis=(0, 1))
    ps_gen = _average_power_spectrum(gen).mean(axis=(0, 1))
    spectrum_err = _relative_mse(ps_ref, ps_gen)

    wass = [stats.wasserstein_distance(ref_flat[c], gen_flat[c])
            for c in range(ref_flat.shape[0])]

    qs, lags, sf_ref = _structure_functions(ref)
    _, _, sf_gen     = _structure_functions(gen, qs=qs, lags=lags)
    sf_err = _relative_mse(sf_ref, sf_gen)

    return {
        "mean_rel_error":    float(mean_err),
        "std_rel_error":     float(std_err),
        "spectrum_rel_mse":  float(spectrum_err),
        "wasserstein_mean":  float(np.mean(wass)),
        "structure_rel_mse": float(sf_err),
    }


def metrics_table(
    x_ref: torch.Tensor,
    results: Dict[str, Dict],
) -> pd.DataFrame:
    """Return a DataFrame of scalar metrics, one row per experiment."""
    rows = {key: compute_metrics(x_ref, results[key]["xt"])
            for key, res in results.items()}
    return pd.DataFrame(rows).T


# ================================================================================
# 1. Marginal histogram overlay
# ================================================================================

def plot_histogram_overlay(
    x_ref: torch.Tensor,
    results: Dict[str, Dict],
    bins: int = 100,
    alpha: float = 0.35,
) -> None:
    """Overlay marginal value histograms for all experiments."""
    def _flat(x):
        return _to_numpy(x).reshape(-1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(_flat(x_ref), bins=bins, density=True,
            histtype="step", linewidth=2.5, label="Data", color="black")
    for key, res in results.items():
        ax.hist(_flat(results[key]["xt"]), bins=bins, density=True,
                alpha=alpha, label=key)
    ax.set_yscale("log")
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.set_title("Marginal histogram")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()





def plot_entropy_curves(
    x_ref: torch.Tensor,
    results: Dict[str, Dict],
    interpolant: Any,
    nt: Optional[int] = None,
) -> None:
    """Plot entropy bounds and Gaussian entropy estimates for all experiments."""

    def _entropy_curves(x_ref, dH_t_bound, t_used, nt_val):
        dH_t_bound = dH_t_bound.detach().cpu()
        t_used = t_used.detach().cpu()

        if nt_val is None:
            nt_val = t_used.shape[0]

        d = x_ref.shape[-2] * x_ref.shape[-1]
        H_p_0 = (np.log(2 * np.pi) + 1) * d / 2
        H_t_bound_full = dH_t_bound.cumsum(0) / nt_val + H_p_0

        # Align lengths: dH_t_bound may be logged on a coarser/shorter grid than t.
        min_len = min(t_used.shape[0], H_t_bound_full.shape[0])
        t_aligned = t_used[:min_len]
        H_t_bound = H_t_bound_full[:min_len]

        H_t_gaussian = compute_gaussian_entropy(
            x_ref.detach().cpu(), interpolant, t_aligned
        )
        return t_aligned, H_t_bound, H_t_gaussian

    fig, ax = plt.subplots(figsize=(7, 5))

    last_gaussian = None
    last_t = None

    for key, res in results.items():
        t_used = results[key]['t']
        t_aligned, H_t_bound, H_t_gaussian = _entropy_curves(
            x_ref, results[key]['dH_t_bound'], t_used, nt
        )

        last_gaussian = H_t_gaussian
        last_t = t_aligned

        print(f"{key}:")
        val_bound = H_t_bound[-1].item() if torch.is_tensor(H_t_bound[-1]) else H_t_bound[-1]
        val_gauss = H_t_gaussian[-1].item() if torch.is_tensor(H_t_gaussian[-1]) else H_t_gaussian[-1]
        print(f"  Entropy bound: {val_bound:.4f}")
        print(f"  Gaussian estimation: {val_gauss:.4f}")

        ax.plot(t_aligned, H_t_bound, label=f"Bound: {key}")

    if last_gaussian is not None and last_t is not None:
        y_gauss = last_gaussian.detach().cpu() if torch.is_tensor(last_gaussian) else last_gaussian
        ax.plot(last_t, y_gauss, '--', color="black", label='Gaussian entropy estimate')

    ax.set_xlabel('t')
    ax.set_ylabel('entropy')
    ax.set_title('Entropy Curves')
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()



def plot_entropy_bound_evolution_panels(
    x_ref: torch.Tensor,
    results: Dict[str, Dict],
    interpolant: Any,
    nt: Optional[int] = None,
) -> None:
    """
    Per-experiment two-panel entropy diagnostic:
    left = raw dH_t_bound increment, right = cumulative bound vs Gaussian entropy.
    """
    d = x_ref.shape[-2] * x_ref.shape[-1]
    H_p_0 = (np.log(2 * np.pi) + 1) * d / 2

    for key, res in results.items():
        dH_t_bound = results[key]['dH_t_bound'].detach().cpu()
        t_used     = results[key]['t'].detach().cpu()
        nt_val     = nt if nt is not None else t_used.shape[0]

        H_t_bound_full = dH_t_bound.cumsum(0) / nt_val + H_p_0
        min_len = min(t_used.shape[0], H_t_bound_full.shape[0])
        t_aligned  = t_used[:min_len]
        H_t_bound  = H_t_bound_full[:min_len]
        dH_aligned = dH_t_bound[:min_len]

        H_t_gaussian = compute_gaussian_entropy(
            x_ref.detach().cpu(), interpolant, t_aligned
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(t_aligned[1:-1], dH_aligned[1:-1], marker='.')
        ax1.set_xlabel('t')
        ax1.set_title(r'$\theta_t^T \frac{d}{dt} m_t$')

        ax2.plot(t_aligned[1:-1], H_t_bound[1:-1],
                 label=r'$H(p_0) + \int \theta_t^T \frac{d}{dt} m_t\,dt$')
        if H_t_gaussian is not None:
            y_gauss = H_t_gaussian.detach().cpu() if torch.is_tensor(H_t_gaussian) else H_t_gaussian
            ax2.plot(t_aligned[1:-1], y_gauss[1:-1], label=r'$H(p_t^{\text{gaussian}})$')
        ax2.set_xlabel('t')
        ax2.legend(loc='best')

        fig.suptitle(key)
        plt.tight_layout()
        plt.show()


# ================================================================================
# 2. Per-band wavelet coefficient histograms  (from hist_plot)
# ================================================================================

def plot_wavelet_histograms_overlay(
    x_ref: torch.Tensor,
    results: Dict[str, Dict],
    psi: Optional[torch.Tensor] = None,
    max_j: Optional[int] = None,
    Q: int = 3,
) -> None:
    """
    Per-scale, per-Q wavelet coefficient magnitude histograms, one figure per
    (j, q) pair overlaying data and all models.

    Parameters
    ----------
    psi:
        Precomputed Morlet filter bank (Fourier). Built automatically if None.
    max_j:
        Number of scales. Defaults to log2(M) - 2.
    Q:
        Voices per octave used when building psi automatically.
    """
    M = x_ref.shape[-1]
    if psi is None:
        J = (int(np.log2(M)) - 2) if max_j is None else max_j
        psi = torch.tensor(
            init_band_pass('morlet', M, J=J, Q=Q, high_freq=0.49, wav_norm='l1'),
            dtype=x_ref.dtype,
        )
    if max_j is None:
        max_j = int(np.log2(M)) - 2

    def _wt(x: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft(torch.fft.fft(x.cpu()) * psi)

    wt_ref = _wt(x_ref)
    wt_gen = {key: _wt(results[key]["xt"]) for key in results}

    for j in range(max_j):
        for q in range(Q):
            idx = j * Q + q
            vals_ref = wt_ref[:, idx].flatten().detach().abs().numpy()

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(vals_ref, bins=100, density=True, histtype="step",
                    linewidth=2, label="Data", color="black")
            for key, res in results.items():
                vals = wt_gen[key][:, idx].flatten().abs().numpy()
                ax.hist(vals, bins=100, density=True, alpha=0.35,
                        label=key)
            ax.set_yscale("log")
            ax.set_title(f"Wavelet coefficients  j={j}, q={q}")
            ax.set_xlabel("|Wψ x|")
            ax.set_ylabel("density")
            ax.legend(frameon=False, fontsize=9)
            plt.tight_layout()
            plt.show()


# ================================================================================
# 3. Power spectrum overlay  (from spec_plot)
# ================================================================================

def plot_spectrum_overlay(
    x_ref: torch.Tensor,
    results: Dict[str, Dict],
) -> None:
    """Overlay power spectra (averaged over batch and channels) for all experiments."""
    ps_ref = _average_power_spectrum(_to_numpy(x_ref)).mean(axis=(0, 1))
    k = np.arange(len(ps_ref))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k[1:], ps_ref[1:], label="Data", linewidth=3, color="black")
    for key, res in results.items():
        ps = _average_power_spectrum(_to_numpy(results[key]["xt"])).mean(axis=(0, 1))
        ax.loglog(k[1:], ps[1:], label=key)
    ax.set_xlabel("wavenumber")
    ax.set_ylabel("PSD")
    ax.set_title("Power spectrum")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


# ================================================================================
# 4. Structure functions overlay  (from structure_plot)
# ================================================================================

def plot_structure_overlay(
    x_ref: torch.Tensor,
    results: Dict[str, Dict],
    qs: tuple = (2, 4, 6, 8),
    show_self_similarity: bool = True,
) -> None:
    """
    One subplot per order q. Data always in black (dashed), each experiment
    in a distinct colour (solid). Two figures: raw S_q(τ) and, if
    show_self_similarity, the ratios S_q / S_2^{q/2}.
    """
    ref_np = _to_numpy(x_ref)
    qs_arr, lags, sf_ref = _structure_functions(ref_np, qs=qs)

    gen_sfs = {}
    for key in results:
        _, _, sf = _structure_functions(_to_numpy(results[key]["xt"]),
                                        qs=qs_arr, lags=lags)
        gen_sfs[key] = sf

    # One colour per experiment, consistent across all subplots
    exp_keys   = list(results.keys())
    exp_colors = plt.get_cmap("tab10")(np.linspace(0, 0.7, len(exp_keys)))
    color_of   = dict(zip(exp_keys, exp_colors))

    nq   = len(qs_arr)
    ncols = min(nq, 4)
    nrows = int(np.ceil(nq / ncols))

    # --- Raw structure functions ---
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    fig.suptitle("Structure functions  S_q(τ)", fontsize=14)
    for i, q in enumerate(qs_arr):
        ax = axes[i // ncols][i % ncols]
        ax.loglog(lags, sf_ref[i], "k--", linewidth=2.5, label="Data")
        for key in exp_keys:
            ax.loglog(lags, gen_sfs[key][i], "-",
                      color=color_of[key], linewidth=2,
                      label=key)
        ax.set_title(f"q = {q}")
        ax.set_xlabel("lag τ")
        ax.set_ylabel(f"S_{q}(τ)")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, which="both", ls=":", alpha=0.4)
    # hide unused axes
    for i in range(nq, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)
    plt.tight_layout()
    plt.show()

    # --- Self-similarity ratios S_q / S_2^{q/2} ---
    if show_self_similarity and 2 in list(qs_arr):
        i2   = list(qs_arr).index(2)
        qs_r = [(i, q) for i, q in enumerate(qs_arr) if q != 2]
        nqr  = len(qs_r)
        ncols_r = min(nqr, 4)
        nrows_r = int(np.ceil(nqr / ncols_r))

        fig, axes = plt.subplots(nrows_r, ncols_r,
                                 figsize=(5 * ncols_r, 4 * nrows_r),
                                 squeeze=False)
        fig.suptitle("Self-similarity ratios  S_q / S_2^{q/2}", fontsize=14)
        for idx, (i, q) in enumerate(qs_r):
            ax = axes[idx // ncols_r][idx % ncols_r]
            ratio_ref = sf_ref[i] / (sf_ref[i2] ** (q / 2) + 1e-30)
            ax.loglog(lags, ratio_ref, "k--", linewidth=2.5, label="Data")
            for key in exp_keys:
                ratio = gen_sfs[key][i] / (gen_sfs[key][i2] ** (q / 2) + 1e-30)
                ax.loglog(lags, ratio, "-",
                          color=color_of[key], linewidth=2,
                          label=key)
            ax.set_title(f"q = {q}")
            ax.set_xlabel("lag τ")
            ax.set_ylabel(f"S_{q} / S_2^{{{q}/2}}")
            ax.legend(frameon=False, fontsize=9)
            ax.grid(True, which="both", ls=":", alpha=0.4)
        for idx in range(nqr, nrows_r * ncols_r):
            axes[idx // ncols_r][idx % ncols_r].set_visible(False)
        plt.tight_layout()
        plt.show()


# ================================================================================
# 5. Cross structure functions  (from cross_plot)
# ================================================================================

def plot_cross_structure_overlay(
    x_ref: torch.Tensor,
    results: Dict[str, Dict],
    pq: list = ((2, 1), (2, 2), (3, 1), (3, 3)),
    max_tau: Optional[int] = None,
    epsilon: float = 1e-8,
) -> None:
    """
    For each (p, q) pair show three images side by side: data log S_{p,q},
    model log S_{p,q}, and relative error — one row of figures per experiment.

    Parameters
    ----------
    pq:
        List of (p, q) exponent pairs.
    max_tau:
        Upper lag limit. Defaults to T // 2.
    """
    ref_np = _to_numpy(x_ref)
    if max_tau is None:
        max_tau = x_ref.shape[-1] // 2

    s_ref = _cross_structure_function(ref_np, pq=pq, max_tau=max_tau)
    log_ref = np.log(s_ref + epsilon)

    for key, res in results.items():
        gen_np = _to_numpy(results[key]["xt"])
        s_gen  = _cross_structure_function(gen_np, pq=pq, max_tau=max_tau)
        log_gen = np.log(s_gen + epsilon)
        error   = np.abs(s_ref - s_gen) / (s_ref + epsilon)

        for k, (p, q) in enumerate(pq):
            vmin = min(log_ref[k].min(), log_gen[k].min())
            vmax = max(log_ref[k].max(), log_gen[k].max())

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"{key}  —  (p, q) = ({p}, {q})")

            for ax, img, title in zip(
                axes,
                [log_ref[k], log_gen[k], error[k]],
                ["log S (data)", f"log S ({key})", "relative error"],
            ):
                im = ax.imshow(img,
                               vmin=(vmin if title != "relative error" else 0),
                               vmax=(vmax if title != "relative error" else 1),
                               cmap=("viridis" if title != "relative error" else "Greys"),
                               origin="lower")
                ax.set_title(title)
                ax.set_xlabel("τ₂")
                ax.set_ylabel("τ₁")
                plt.colorbar(im, ax=ax, shrink=0.8)

            plt.tight_layout()
            plt.show()


# ================================================================================
# 6. Increment PDFs waterfall  (from increment_pdf_plot)
# ================================================================================

def plot_increment_pdf_overlay(
    x_ref: torch.Tensor,
    results: Dict[str, Dict],
    taus: tuple = (1, 2, 4, 8, 16, 32),
    n_bins: int = 201,
    xlim: Tuple[float, float] = (-8, 8),
    smooth_sigma: float = 1.5,
) -> None:
    """
    Decade-shifted increment PDF waterfall. Data in black, each model in a
    distinct colour with dashed lines. All scales on one figure.
    """
    ref_np = _to_numpy(x_ref)
    taus   = list(taus)
    n      = len(taus)
    cmap   = plt.get_cmap("tab10")
    b      = np.linspace(xlim[0], xlim[1], n_bins)
    c      = 0.5 * (b[1:] + b[:-1])

    def _pdf(x_np, tau, shift):
        d = (x_np[..., tau:] - x_np[..., :-tau]).reshape(-1)
        d = d / (d.std() + 1e-12)
        h, _ = np.histogram(d, bins=b, density=True)
        h = gaussian_filter1d(h, smooth_sigma)
        h = h / (h.max() + 1e-12) * shift
        return np.where(h > 0, h, np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))

    for k, tau in enumerate(taus):
        shift = 10.0 ** (-k)
        ax.plot(c, _pdf(ref_np, tau, shift), color="black", lw=2.5,
                label="Data" if k == 0 else None)
        for j, (key, exp) in enumerate(results.items()):
            gen_np = _to_numpy(results[key]["xt"])
            ax.plot(c, _pdf(gen_np, tau, shift), color=cmap(j), lw=1.8,
                    linestyle="--", label=key if k == 0 else None)
        ax.text(0.0, shift * 1.8, f"τ={tau}", fontsize=11,
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor="gray", lw=1))

    ax.set_yscale("log")
    ax.set_xlim(xlim)
    ax.set_ylim(10.0 ** (-(n + 1)), 5.0)
    ax.set_xlabel("δu / σ")
    ax.set_ylabel("PDF (decade-shifted)")
    ax.set_title("Increment PDFs across scales")
    ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.show()


# ================================================================================
# 7. Moment matching overlay
# ================================================================================

def plot_moment_matching_overlay(
    results: Dict[str, Dict],
    threshold: float = 1e-8,
) -> None:
    """
    Overlay relative moment-matching error across experiments.
    Left panel: mean relative error vs t. Right panel: final-step error distribution.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for key, res in results.items():
        label = key
        try:
            barphi_e = results[key]["barphi_e"].cpu()
            barphi_p = results[key]["barphi_p"].cpu()
            t        = results[key]["t"].cpu()

            keep_mask = barphi_e[-1] > threshold

            if not keep_mask.any():
                print(f"  Warning: {label} — no moments exceeded threshold "
                      f"{threshold}; showing final-error histogram only.")
                error_last = (
                    2 * (barphi_e - barphi_p).abs()
                    / (barphi_e.abs() + barphi_p.abs())
                )[-1]
                ax2.hist(error_last.numpy(), bins=100, alpha=0.5, label=label)
                continue

            e = barphi_e[:, keep_mask]
            p = barphi_p[:, keep_mask]
            rel_error = 2 * (e - p).abs() / (e.abs() + p.abs())

            min_len    = min(t.shape[0], rel_error.shape[0])
            t_sliced   = t[:min_len][2:-1]
            error_mean = rel_error.mean(dim=1)[:min_len][2:-1]
            error_last = rel_error[-1]

            ax1.plot(t_sliced, error_mean, marker='.', label=label)
            ax2.hist(error_last.numpy(), bins=100, alpha=0.5, label=label)

        except Exception as exc:
            print(f"  Skipping {label}: {exc}")

    ax1.set_xlabel("t")
    ax1.set_ylabel("Mean relative error")
    ax1.set_yscale("log")
    ax1.set_title("Time evolution")
    ax1.legend(frameon=False, fontsize=9)

    ax2.set_xlabel("Relative error")
    ax2.set_ylabel("Count")
    ax2.set_yscale("log")
    ax2.set_title("Final distribution")
    ax2.legend(frameon=False, fontsize=9)

    fig.suptitle("Moment matching")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_scaling_ratio_overlay(
    Data,
    results: Dict[str, Dict],
    synth_key: str = "x",
    num_points: int = 30,
):
    """
    Overlay local scaling exponent ratio
        d(log S4) / d(log S2)
    for multiple experiments.

    Parameters
    ----------
    Data : torch.Tensor or np.ndarray
        Reference dataset.
    results : dict
        Dictionary of experiment outputs.
    synth_key : str
        Key in each experiment containing the synthesized data.
    num_points : int
        Number of logarithmically spaced tau values.
    """

    # Reference curve
    taus, ratio_data = scaling_exponent_ratio(
        Data,
        num_points=num_points,
    )
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        taus,
        ratio_data,
        "k-o",
        lw=2,
        ms=4,
        label="Data",
    )
    for key, res in results.items():
        try:
            synth = res[synth_key]

            # move to cpu if needed
            if hasattr(synth, "cpu"):
                synth = synth.cpu()

            _, ratio = scaling_exponent_ratio(
                synth,
                taus=taus,
            )

            ax.plot(
                taus,
                ratio,
                marker=".",
                label=key,
            )
        except Exception as exc:
            print(f"Skipping {key}: {exc}")
    ax.axhline(
        2.0,
        color="grey",
        ls="--",
        lw=1,
        label=r"$\zeta_4/\zeta_2 = 2$",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$d\log S_4 / d\log S_2$")
    ax.set_title("Scaling exponent ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()


# ================================================================================
# 8. Visual comparison panel  (from visual_comparison)
# ================================================================================

def plot_visual_comparison(
    x_ref: torch.Tensor,
    results: Dict[str, Dict],
    pdf_taus: tuple = (1, 2, 4, 8, 16, 32),
    sf_orders: tuple = (4, 6, 8),
    c4_fixed_tau: int = 10,
    n_bins: int = 201,
    smooth_sigma: float = 1.5,
    pdf_xlim: Tuple[float, float] = (-8, 8),
) -> None:
    """
    Full 5-panel visual comparison figure (one per experiment): marginal PDF,
    power spectrum, structure functions, C4 coefficient, increment PDF waterfall.
    Mirrors the visual_comparison() function from check_moments.py.
    """
    C_ORIG  = "tab:blue"
    C_SYNTH = "tab:red"
    LW      = 3.0
    ref_np  = _to_numpy(x_ref)
    max_tau = x_ref.shape[-1] // 2
    taus_arr = np.arange(1, max_tau)

    def abs_inc_mean(x, tau, p):
        d = np.abs(x[..., tau:] - x[..., :-tau])
        return np.mean(np.power(d.reshape(-1), p))

    for key, res in results.items():
        gen_np = _to_numpy(results[key]["xt"])

        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(key, fontsize=18, fontweight="bold")
        gs  = fig.add_gridspec(2, 3)
        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[0, 1])
        axD = fig.add_subplot(gs[1, 0])
        axE = fig.add_subplot(gs[1, 1])
        axC = fig.add_subplot(gs[:, 2])

        # (a) marginal PDF
        b   = np.linspace(ref_np.min(), ref_np.max(), n_bins)
        ctr = 0.5 * (b[1:] + b[:-1])
        for data, color, label in [(ref_np, C_ORIG, "Data"), (gen_np, C_SYNTH, key)]:
            h, _ = np.histogram(data.reshape(-1), bins=b, density=True)
            h = gaussian_filter1d(h, smooth_sigma)
            axA.plot(ctr, np.where(h > 0, h, np.nan), color=color, lw=LW, label=label)
        axA.set_yscale("log")
        axA.set_ylabel("PDF")
        axA.legend(frameon=False, fontsize=12)
        axA.grid(True, ls=":", alpha=0.5)

        # (b) power spectrum
        def psd(x):
            return np.mean(np.abs(np.fft.rfft(x.reshape(-1, x.shape[-1]), axis=-1)) ** 2, axis=0)
        omega = np.fft.rfftfreq(ref_np.shape[-1])
        axB.loglog(omega[1:], psd(ref_np)[1:], color=C_ORIG,  lw=LW)
        axB.loglog(omega[1:], psd(gen_np)[1:], color=C_SYNTH, lw=LW)
        axB.set_ylabel("PSD")
        axB.grid(True, which="both", ls=":", alpha=0.5)

        # (d) structure functions
        sf_colors = {4: "tab:red", 6: "tab:green", 8: "tab:blue"}
        for p in sf_orders:
            col = sf_colors.get(p, "k")
            SF_o = np.array([abs_inc_mean(ref_np, t, p) for t in taus_arr])
            SF_s = np.array([abs_inc_mean(gen_np, t, p) for t in taus_arr])
            axD.loglog(taus_arr, SF_o + 1e-8, "--", color=col, lw=LW)
            axD.loglog(taus_arr, SF_s + 1e-8, "-",  color=col, lw=LW,
                       label=rf"SF$_{{{p}}}$")
        axD.set_xlabel("τ")
        axD.set_ylabel("SF_k(τ)")
        axD.legend(frameon=False, fontsize=10)
        axD.grid(True, which="both", ls=":", alpha=0.5)

        # ----- (e) C_4 coefficient -----
        if not (1 <= c4_fixed_tau < max_tau):
            raise ValueError(f"c4_fixed_tau={c4_fixed_tau} must be in [1, {max_tau-1}]")
        epsilon=1e-8
        _, r_o = C_pq_structure(ref_np,  2, 2, c4_fixed_tau, max_tau=max_tau, epsilon=epsilon)
        _, r_s = C_pq_structure(gen_np, 2, 2, c4_fixed_tau, max_tau=max_tau, epsilon=epsilon)
        axE.plot(taus_arr, r_o, 'o-', color=C_ORIG,  lw=LW, ms=5)
        axE.plot(taus_arr, r_s, 's-', color=C_SYNTH, lw=LW, ms=5)
        axE.set_xscale('log')
        axE.set_xlabel(r'$\tau$')
        axE.set_ylabel(r'$C_4(\tau,\tau^\star)$')
        axE.grid(True, ls=':', alpha=0.5)

        # (c) increment PDFs waterfall
        taus_list = list(pdf_taus)
        bpdf = np.linspace(pdf_xlim[0], pdf_xlim[1], n_bins)
        cpdf = 0.5 * (bpdf[1:] + bpdf[:-1])
        for k, tau in enumerate(taus_list):
            shift = 10.0 ** (-k)
            for data, color in [(ref_np, C_ORIG), (gen_np, C_SYNTH)]:
                d = (data[..., tau:] - data[..., :-tau]).reshape(-1)
                d = d / (d.std() + 1e-12)
                h, _ = np.histogram(d, bins=bpdf, density=True)
                h = gaussian_filter1d(h, smooth_sigma)
                h = h / (h.max() + 1e-12) * shift
                axC.plot(cpdf, np.where(h > 0, h, np.nan), color=color, lw=LW)
            axC.text(0.0, shift * 1.8, f"τ={tau}", fontsize=10,
                     ha="center", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.2",
                               facecolor="white", edgecolor="gray", lw=1))
        axC.set_yscale("log")
        axC.set_xlim(pdf_xlim)
        axC.set_ylim(10.0 ** -(len(taus_list) + 1), 5.0)
        axC.set_ylabel("PDF (shifted)")
        axC.grid(True, ls=":", alpha=0.5)

        plt.tight_layout()
        plt.savefig("visual_comparison.png")
        plt.show()


# ================================================================================
# Master runner
# ================================================================================
def run_diagnostics(
    x_ref: torch.Tensor,
    results: Dict[str, Dict],
    threshold: float = 1e-8,
    interpolant: str = 'Cos',
    nt: int = 1000,
    Q: int = 1,
    structure_qs: tuple = (2, 4, 6, 8),
    pdf_taus: tuple = (1, 2, 4, 8, 16, 32),
    cross_pq: list = ((2, 1), (2, 2), (3, 1), (3, 3)),
    show_wavelet_histograms: bool = False,
    complex_wavelet_histograms: bool = False,
    show_cross_plots: bool = False,
    show_visual_comparison: bool = True,
    show_metrics_table: bool = True,
    show_entropy_detail: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Full diagnostic suite. `Q` sets voices/octave for wavelet histograms
    (1 or 3, matches whatever the model was trained/evaluated with).
    """
    sep = "=" * 72

    print(sep); print("Marginal histograms"); print(sep)
    plot_histogram_overlay(x_ref, results)

    if show_wavelet_histograms:
        print(sep); print(f"Per-band wavelet histograms (Q={Q})"); print(sep)
        if complex_wavelet_histograms:
            plot_wavelet_real_imag_histograms_overlay(
                x_ref, results, Q=Q
            )
        else:
            plot_wavelet_histograms_overlay(
                x_ref, results, Q=Q
            )

    print(sep); print("Moment matching"); print(sep)
    plot_moment_matching_overlay(results, threshold=threshold)

    print(sep); print("Power spectra"); print(sep)
    plot_spectrum_overlay(x_ref, results)

    print(sep); print("Structure functions"); print(sep)
    plot_structure_overlay(x_ref, results, qs=structure_qs)

    print(sep); print("Increment PDFs"); print(sep)
    plot_increment_pdf_overlay(x_ref, results, taus=pdf_taus)

    if show_cross_plots: 
        print(sep); print("Cross structure functions"); print(sep)
        plot_cross_structure_overlay(x_ref, results, pq=cross_pq)

    print(sep); print("Entropy bound"); print(sep)
    plot_entropy_curves(x_ref, results, interpolant, nt)

    if show_entropy_detail:
        print(sep); print("Entropy bound evolution (per experiment)"); print(sep)
        plot_entropy_bound_evolution_panels(x_ref, results, interpolant, nt)
    

    if show_visual_comparison:
        print(sep); print("Visual comparison panels"); print(sep)
        plot_visual_comparison(x_ref, results,
                               pdf_taus=pdf_taus, sf_orders=(4, 6, 8))
        

    if show_metrics_table:
        print(sep); print("Scalar metrics"); print(sep)
        df = metrics_table(x_ref, results)
        print(df.to_string())
        return df
    

    return None



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

def _wavelet_coeffs(x: torch.Tensor, psi: torch.Tensor):
    """Return complex wavelet coefficients (B, JQ, T)."""
    x_fft = torch.fft.fft(x.cpu())
    return torch.fft.ifft(x_fft * psi)

def plot_wavelet_real_imag_histograms_overlay(
    x_ref: torch.Tensor,
    results: Dict[str, Dict],
    psi: Optional[torch.Tensor] = None,
    max_j: Optional[int] = None,
    Q: int = 3,
    bins: int = 100,
) -> None:

    M = x_ref.shape[-1]

    if psi is None:
        J = (int(np.log2(M)) - 2) if max_j is None else max_j
        psi = torch.tensor(
            init_band_pass(
                'morlet', M, J=J, Q=Q,
                high_freq=0.49, wav_norm='l1'
            ),
            dtype=x_ref.dtype,
        )

    if max_j is None:
        max_j = int(np.log2(M)) - 2

    wt_ref = _wavelet_coeffs(x_ref, psi)
    wt_gen = {
        key: _wavelet_coeffs(results[key]["xt"], psi)
        for key in results
    }

    for j in range(max_j):
        for q in range(Q):
            idx = j * Q + q

            ref = wt_ref[:, idx].detach().cpu().numpy()
            ref_r = ref.real.flatten()
            ref_i = ref.imag.flatten()

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # ---------------- REAL PART ----------------
            axes[0].hist(
                ref_r,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                color="black",
                label="Data",
            )

            for key, res in results.items():
                vals = wt_gen[key][:, idx].cpu().numpy().real.flatten()
                axes[0].hist(vals, bins=bins, density=True, alpha=0.5,
                             label=key)

            axes[0].set_title(f"Real part — j={j}, q={q}")
            axes[0].set_xlabel("Re(Wψ x)")
            axes[0].set_ylabel("density")
            axes[0].legend(frameon=False)
            axes[0].set_yscale("log")

            # ---------------- IMAG PART ----------------
            axes[1].hist(
                ref_i,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                color="black",
                label="Data",
            )

            for key, res in results.items():
                vals = wt_gen[key][:, idx].cpu().numpy().imag.flatten()
                axes[1].hist(vals, bins=bins, density=True, alpha=0.5,
                             label=key)

            axes[1].set_title(f"Imag part — j={j}, q={q}")
            axes[1].set_xlabel("Im(Wψ x)")
            axes[1].set_ylabel("density")
            axes[1].legend(frameon=False)
            axes[1].set_yscale("log")

            plt.tight_layout()
            plt.show()