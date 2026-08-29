#!/usr/bin/env python
"""
run_SDE.py — CLI launcher for MGD/SDE synthesis experiments on fractional
Brownian motion (fBm) / multifractal random walk (MRW) increments.

Structurally a copy of turbulence/run_SDE.py (same CLI conventions, same
config.json/logs/figures output layout, same wavelet-scattering + SDE
machinery) with the data source swapped: instead of loading a real 1D
turbulence recording, this script *generates* synthetic data via
`load_synthetic_data('mrw', ...)` (data/synthetic_data_generator.py,
backed by data/standard_models/, ported from
https://github.com/RudyMorel/scattering_spectra).

Why increments, not the raw process: fractional Brownian motion itself is
not stationary (its marginal variance grows with time), so the SDE/
wavelet-scattering machinery -- built around stationary statistics -- is
run on its increments (fractional Gaussian noise) instead, which *are*
stationary. Setting --intermittency 0 reduces the MRW model to a pure fBm
(hence the increments are pure fGn); intermittency > 0 adds Bacry-Delour-
Muzy multifractal (lognormal-cascade) modulation on top.

NOTE on imports: the sys.path manipulation below is intentional and must
stay exactly as written — the project's 'codes' and 'data' packages rely on
these paths being inserted, in this order, before anything under them is
imported.

NOTE on output layout: every run gets its own self-contained folder,
  <outdir>/experiments/<config>/
      config.json
      logs/run.log
      figures/*.png
  See main() below.
"""
import argparse
import math
import sys
import time as timer
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


# ── sys.path setup ──────────────────────────────────────────────────────────
# Script lives in .../MGD.../gaussian_experiment/; 'codes' and 'data' live in
# the parent folder .../MGD.../, not alongside this script.
root = Path(__file__).resolve().parent
project_root = root.parent

def _extend_syspath(root: Path, project_root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Add the 'codes' directory itself to sys.path
    # This is the "magic" that lets 'from potentials...' work
    codes_path = project_root / 'codes'
    if codes_path.is_dir() and str(codes_path) not in sys.path:
        sys.path.insert(0, str(codes_path))

    # 'data' itself must be on sys.path (not just importable as a package)
    # so that data/synthetic_data_generator.py's own bare `from
    # standard_models import *` resolves regardless of how this script's
    # imports below reach it.
    data_path = project_root / 'data'
    if data_path.is_dir() and str(data_path) not in sys.path:
        sys.path.insert(0, str(data_path))

_extend_syspath(root, project_root)

from codes.sde_routines import *              # noqa: E402
from codes.utils import *                     # noqa: E402
from codes.utils_experiment import *          # noqa: E402
from codes.check_moments import *             # noqa: E402
from codes.ortho_wavelet.ReadyToUseWavelets import *  # noqa: E402
from data.synthetic_data_generator import *   # noqa: E402
from synthetic_data_generator import *        # noqa: E402

root = Path.cwd()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── argument parsing ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='MGD/SDE synthesis experiment launcher on fBm/MRW increments'
    )
    p.add_argument('--timestamp', type=str, default=None, help='Timestamp for the run')

    # Data
    p.add_argument('--n1', type=int, default=3000,
                    help='Dataset size: number of i.i.d. increment paths to generate')
    p.add_argument('--M', type=int, default=128,
                    help='Signal length: number of increments per path')
    p.add_argument('--hurst', type=float, default=0.5,
                    help='Hurst exponent of the underlying fBm (0.5 = standard '
                         'Brownian motion)')
    p.add_argument('--intermittency', type=float, default=0.0,
                    help='MRW intermittency parameter (Bacry-Delour-Muzy lam). '
                         '0.0 reduces the MRW to a pure fBm, so its increments '
                         'are plain fractional Gaussian noise; > 0 adds '
                         'multifractal (fat-tailed) modulation on top. Distinct '
                         'from --lam below, which is the SDE regularization.')

    # Wavelet scattering
    p.add_argument('--J', type=int, default=7, help='Number of wavelet scales')
    p.add_argument('--Q', type=int, default=3, help='Wavelets per octave')
    p.add_argument('--terms', nargs='+', type=str,
                    default=[
                        'L_2_lowpass',
                        'Scattering_Fourth_Order_Mod2_Real_Q1',
                        'Scalar_psi_GGG',
                        'Scalar_morlet_GGG',
                    ],
                    help='Potential terms making up the maxent target')

    # SDE
    p.add_argument('--nt', type=int, default=8000,
                    help='Number of SDE integration steps')
    p.add_argument('--sigma', type=float, default=0.3, help='Diffusion coefficient sigma')
    p.add_argument('--schedule_exponent', type=int, default=2,
                    help='Exponent in t = 1-(1-linspace)^exponent')
    p.add_argument('--interpolant', type=str, default='Cos',
                    help='Interpolant type: Cos | Linear | VarPreserv | Sqrt')
    p.add_argument('--regularization', type=float, default=1e-1,
                    help='Tikhonov regularization for the Gram matrix')
    p.add_argument('--lam', type=float, default=5e-6,
                    help='lambda passed to Solver.forward_regularised (SDE '
                         'regularization strength -- not the MRW intermittency)')
    p.add_argument('--n_subsample', type=int, default=100,
                    help='n_subsample passed to Solver.forward_regularised')

    # Batch
    p.add_argument('--batch_size', type=int, default=None,
                    help='Batch size for potential evaluations (default: B)')

    # Diagnostics
    p.add_argument('--n_traj_groups', type=int, default=5,
                    help='Number of groups of 5 trajectories to plot for the '
                         'true-vs-synthesized comparison')
    p.add_argument('--n_tau', type=int, default=30,
                    help='Number of lags used in the scaling-exponent-ratio diagnostic')
    p.add_argument('--moment_threshold', type=float, default=1e-8,
                    help='Threshold used in plot_moment_matching')
    p.add_argument('--n_acf_lags', type=int, default=50,
                    help='Number of lags shown in the fBm/MRW increment '
                         'autocorrelation diagnostic')

    # Experiment / bookkeeping
    p.add_argument('--outdir', type=str, default=None,
                    help='Base directory. Defaults to this script\'s own '
                         'directory. Each run gets its own subfolder at '
                         '<outdir>/experiments/<config>/ containing config.json, '
                         'logs/ and figures/.')
    p.add_argument('--label', type=str, default=None,
                    help='Optional extra label appended to the config name')
    p.add_argument('--force_rerun', action='store_true',
                    help='Ignore any existing saved results and rerun from scratch')
    p.add_argument('--no_save_aux_moments', action='store_true',
                    help='Disable saving of barphi_e / barphi_p aux moments')
    p.add_argument('--seed', type=int, default=0, help='Random seed')

    return p.parse_args()


def plot_wavelet_coeff_histograms(x1, J, Q, config, figdir, logger, save=None):
    B, channels, M = x1.shape
    filters = return_Filters(M, J, Q, device=device)
    wt = torch.fft.ifft(torch.fft.fft(x1) * filters).real  # (B, J, T)
    n_wavelets = filters.shape[1]

    ncols = 3
    nrows = math.ceil(n_wavelets / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i in range(n_wavelets):
        vals = wt[:, i, :].detach().cpu().flatten().numpy()
        axes[i].hist(vals, bins=50, density=True, log=True)
        axes[i].set_title(f"ch={i}")
        axes[i].set_xlabel("Coefficient value")
        axes[i].set_ylabel("Density")

    for j in range(n_wavelets, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Wavelet coefficient histograms (Q={Q})", fontsize=25)

    if save is not None:
        fig.suptitle(save["title"], fontsize=25)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(save["filename"], dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        fig.tight_layout()
        plt.show()


# ── fBm/MRW increment sanity-check figure ───────────────────────────────────
def plot_increment_diagnostics(x1, args, fig_dir, config, n_show=5):
    """ Sanity-check figure for the generated fBm/MRW increments, saved
    *before* the (expensive) SDE run so a bad data-generation config is
    caught early: a few example traces, a marginal histogram (Gaussian for
    intermittency=0, fat-tailed for intermittency>0), and the empirical
    autocorrelation against the theoretical fractional-Gaussian-noise ACF
    (exact only for intermittency=0 -- shown as a reference curve
    regardless, since MRW's ACF coincides with fGn's at lag>=1 by
    construction of the model).
    """
    x = x1[:, 0, :].detach().cpu().numpy()  # (B, M)
    B, M = x.shape
    max_lag = min(args.n_acf_lags, M - 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. example traces
    for i in range(min(n_show, B)):
        axes[0].plot(x[i], lw=1, alpha=0.8)
    axes[0].set_title('Example increment traces')
    axes[0].set_xlabel('index')
    axes[0].set_ylabel('increment')

    # 2. marginal histogram
    axes[1].hist(x.ravel(), bins=100, density=True, log=True)
    axes[1].set_title('Increment marginal (log density)')
    axes[1].set_xlabel('increment value')
    axes[1].set_ylabel('density')

    # 3. empirical vs theoretical autocorrelation
    x_centered = x - x.mean()
    var = x_centered.var()
    acf_emp = np.array([
        1.0 if lag == 0 else
        np.mean(x_centered[:, :M - lag] * x_centered[:, lag:]) / var
        for lag in range(max_lag + 1)
    ])
    k = np.arange(0, max_lag + 1)
    acf_theory = 0.5 * (
        np.abs(k + 1) ** (2 * args.hurst)
        - 2 * np.abs(k) ** (2 * args.hurst)
        + np.abs(k - 1) ** (2 * args.hurst)
    )
    acf_theory[0] = 1.0

    axes[2].plot(k, acf_emp, 'o-', ms=4, label='empirical')
    axes[2].plot(k, acf_theory, 'k--', label=f'theoretical fGn (H={args.hurst})')
    axes[2].axhline(0, color='grey', lw=0.5)
    axes[2].set_xlabel('lag')
    axes[2].set_ylabel('autocorrelation')
    axes[2].legend()
    axes[2].set_title('Increment autocorrelation')

    fig.suptitle(
        f'{config}\nH={args.hurst}, intermittency={args.intermittency}',
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / 'increment_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── diagnostics ───────────────────────────────────────────────────────────────
def save_diagnostics(x1, res, t, args, config, fig_dir, logger):
    plot_wavelet_coeff_histograms(
        x1, args.J, 1, config, fig_dir, logger,
        save={"filename": fig_dir / "wavelet_histo_Q1.png", "title": config}
    )

    plot_wavelet_coeff_histograms(
        x1, args.J, 3, config, fig_dir, logger,
        save={"filename": fig_dir / "wavelet_histo_Q3.png", "title": config}
    )

    threshold = args.moment_threshold

    # 1. moment matching
    if res.get('barphi_e') is not None and res.get('barphi_p') is not None:
        plot_moment_matching(res['barphi_e'], res['barphi_p'], res['t'], threshold, save={"filename": fig_dir / "moment_matching.png", "title": config})
    else:
        logger.warning('No aux moments available (loaded run without saved aux file?); '
                        'skipping moment-matching plot.')

    # 2. trajectories: true vs synthesized
    n_groups = max(1, min(args.n_traj_groups, x1.shape[0] // 5))
    for i in range(n_groups):
        Compare_time_series_row(x1[i * 5:i * 5 + 5], res['xt'][i * 5:i * 5 + 5], 5, save={"filename": fig_dir / "compare_time_series.png", "title": config})

    # 3. wavelet histogram collection
    hist_plot(x1, res['xt'], save={"filename": fig_dir / "histograms.png", "title": config})

    # 4. power spectrum
    spec_plot(x1, res['xt'], save={"filename": fig_dir / "spectrum.png", "title": config})

    # 5. structure functions
    structure_plot(x1, res['xt'], save={"filename": fig_dir / "structure_functions.png", "title": config})

    # 6. scaling exponent ratio (zeta4 / zeta2)
    taus, ratio_data, ratio_synth = scaling_exponent_ratio_compare(x1, res['xt'], args.n_tau)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(taus[:-1], ratio_data[:-1], 'ko-', ms=4, label='Data')
    ax.plot(taus[:-1], ratio_synth[:-1], 'ro-', ms=4, label='Synth')
    ax.axhline(2.0, color='grey', ls='--', lw=1, label=r'$\zeta_4/\zeta_2=2$ (dimensional)')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$\zeta_4^L/\zeta_2^L = d\log S_4 / d\log S_2$')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(config)
    fig.savefig(fig_dir / 'scaling_exponent_ratio.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info('Saved diagnostic figures to %s', fig_dir)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # `root` = this script's own directory (gaussian_experiment/), set at the
    # top of the file.
    outdir = Path(args.outdir) if args.outdir else root
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    # T = M + 1 time samples of the process -> M increments after .diff().
    # intermittency=0 reduces MRW to a pure fBm (see module docstring).
    raw = load_synthetic_data(
        'mrw', args.n1, args.M + 1,
        lam=args.intermittency, H=args.hurst,
    ).to(device)
    increments = raw.diff(dim=-1)  # (n1, 1, M)

    x1 = normalize(increments).to(device)
    B, channels, M = x1.shape

    config, exp_dir, fig_dir, potentials_dir, logger, loaded = resolve_or_setup_experiment_output(
        outdir, args, M, device, coarse_grained=False,
        extra_metadata={'M': M, 'B': B, 'channels': channels},
        include_potentials_dir=True,
    )
    logger.info('Generated raw process shape: %s', raw.shape)
    logger.info('x1 (increments, normalized) final shape: %s', tuple(x1.shape))
    logger.info('Hurst = %s, intermittency = %s', args.hurst, args.intermittency)

    plot_increment_diagnostics(x1, args, fig_dir, config)
    logger.info('Saved increment sanity-check figure to %s', fig_dir / 'increment_diagnostics.png')

    filters, filters_Phi = return_Filters(M, args.J, 1, device=device, include_phi=True)
    filters_Q = return_Filters(M, args.J, args.Q, device=device)  # unused downstream — see note below

    t = 1 - (1 - torch.linspace(0, 1, args.nt + 1)) ** args.schedule_exponent

    t_rounded = torch.round(t, decimals=4)
    t_final = int((t_rounded == 1.0).nonzero(as_tuple=True)[0][0])
    logger.info(f"t_final = {t_final}/{len(t)} (last t = {t[t_final-1].item():.6f}, "
        f"dropping {len(t) - t_final} redundant trailing points at 1.0000)")

    if loaded is not None:
        result = loaded
    else:
        result = run_experiment(args, M, config, x1, filters,
                                t[:t_final], logger, outdir, device,
                                filters_Q=filters_Q, filters_Phi=filters_Phi,
                                potentials_save_dir=potentials_dir,
                                )

    save_diagnostics(x1, result, t, args, config, fig_dir, logger)
    logger.info('Done.')


if __name__ == '__main__':
    main()
