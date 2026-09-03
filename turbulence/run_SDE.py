#!/usr/bin/env python
"""
run_SDE.py — CLI launcher for MGD/SDE turbulence synthesis experiments.

Replaces the ad-hoc notebook workflow. This script:
  * is fully driven from the command line (see launch_SDE.sh for a SLURM
    array wrapper),
  * writes every user-chosen parameter (plus a few derived ones, like the
    final signal length M) to a JSON config file next to the results,
  * saves all diagnostic figures to disk instead of plt.show()-ing them,
  * can be re-loaded later for interactive analysis via reload_experiment.py.

NOTE on imports: the sys.path manipulation below is intentional and must
stay exactly as written — the project's 'codes' and 'data' packages rely on
these paths being inserted, in this order, before anything under them is
imported.

NOTE on output layout: every run now gets its own self-contained folder,
  <outdir>/experiments/<config>/
      config.json
      logs/run.log
      figures/*.png
  instead of dumping everything with config-name-prefixed filenames into a
  shared saved_results/ directory. See main() below.
"""
import argparse
import contextlib
import hashlib
import json
import logging
import sys
import time as timer
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats, ndimage


# ── sys.path setup ──────────────────────────────────────────────────────────
# Script lives in .../MGD.../turbulence/; 'codes' and 'data' live in
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
 
    data_path = project_root / 'data'
    if data_path.is_dir() and str(data_path) not in sys.path:
        sys.path.insert(0, str(data_path))
 
_extend_syspath(root, project_root)
 
from codes.sde_routines import *      # noqa: E402
from codes.utils import *             # noqa: E402
from codes.utils_experiment import * 
from codes.check_moments import *     # noqa: E402
from codes.ortho_wavelet.ReadyToUseWavelets import *
from data.data_loader import *        # noqa: E402
from data_loader import *             # noqa: E402

root = Path.cwd() 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── argument parsing ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='MGD/SDE turbulence synthesis experiment launcher'
    )
    p.add_argument('--timestamp', type=str, default=None, help='Timestamp for the run')

    # Data
    p.add_argument('--n1', type=int, default=3000,
                    help='Dataset size: number of samples kept from the '
                         'preprocessed data, i.e. Data[:n1] (replaces the '
                         'hardcoded 3000 slice)')

    # Coarse-graining
    p.add_argument('--subseries_len', type=int, default=1024, help='Base length after reshaping')
    p.add_argument('--target_len', type=int, default=128, help='Target signal length after coarse-graining')

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
                    help='lambda passed to Solver.forward_regularised')
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

    # Experiment / bookkeeping
    p.add_argument('--outdir', type=str, default=None,
                    help='Base directory. Defaults to this script\'s own '
                         'directory (matching the notebook workflow, which '
                         'always uses its own cwd as outdir with no '
                         'saved_results/ nesting — the two must agree, or '
                         'runs launched each way never see each other as '
                         'duplicates). Each run gets its own subfolder at '
                         '<outdir>/experiments/<config>/ containing config.json, '
                         'logs/ and figures/.')
    p.add_argument('--label', type=str, default=None,
                    help='Optional extra label appended to the config name')
    p.add_argument('--force_rerun', action='store_true',
                    help='Ignore any existing saved results and rerun from scratch')
    p.add_argument('--time_limit_min', type=float, default=None,
                    help='SLURM wall-clock budget in minutes. If set, the SDE loop '
                         'aborts (raises) once the projected total runtime, '
                         'extrapolated from the average iteration time after a '
                         '30-iteration warm-up, exceeds 90%% of this budget. '
                         'Disabled (no check) when omitted.')
    p.add_argument('--no_save_aux_moments', action='store_true',
                    help='Disable saving of barphi_e / barphi_p aux moments')
    p.add_argument('--seed', type=int, default=0, help='Random seed')

    return p.parse_args()


def plot_wavelet_coeff_histograms(x1, J, Q, config, figdir, logger, save=None): 
    B, channels, M = x1.shape
    filters = return_Filters(M, J, Q, device=device)
    wt = torch.fft.ifft(torch.fft.fft(x1) * filters).real  # (B, J, T)
    n_wavelets = filters.shape[1]
    
    # ------------------------------------------------------------------
    # 1. Overview grid
    # ------------------------------------------------------------------
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
    
    # Matching the exact saving/showing logic of hist_plot
    if save is not None:
        fig.suptitle(save["title"], fontsize=25)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(save["filename"], dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        fig.tight_layout()
        plt.show()

# ── diagnostics ───────────────────────────────────────────────────────────────
def save_diagnostics(x1, res, t, args, config, fig_dir, logger):
    # Pass config, fig_dir, and logger explicitly into the functions
    # and give each Q its own dedicated output filename.
    
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
        plot_moment_matching(res['barphi_e'], res['barphi_p'], res['t'], threshold, save={"filename": fig_dir / "moment_matching.png","title": config})
    else:
        logger.warning('No aux moments available (loaded run without saved aux file?); '
                        'skipping moment-matching plot.')

    # 2. trajectories: true vs synthesized
    n_groups = max(1, min(args.n_traj_groups, x1.shape[0] // 5))
    for i in range(n_groups):
        Compare_time_series_row(x1[i * 5:i * 5 + 5], res['xt'][i * 5:i * 5 + 5], 5, save={"filename": fig_dir / "compare_time_series.png","title": config})

    # 3. wavelet histogram collection
    # This is the section that was coming out "superimposed on one canvas".
    # We force every plt.hist() call inside hist_plot to open its own new
    # figure (see isolate_pyplot_calls docstring for the caveat on when this
    # can't help), then save every figure that resulted, individually.
    hist_plot(x1, res['xt'], save={"filename": fig_dir / "histograms.png","title": config})

    # 4. power spectrum
    spec_plot(x1, res['xt'], save={"filename": fig_dir / "spectrum.png","title": config})

    # 5. structure functions
    structure_plot(x1, res['xt'], save={"filename": fig_dir / "structure_functions.png","title": config})

    # 6. scaling exponent ratio (zeta4 / zeta2)
    # NOTE: the original notebook compared against an undefined `xt_cg` here;
    # fixed to compare against the actual generated trajectories `res['xt']`.
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

    # 7. entropy bound evolution
    #H_t_bound, H_t_gaussian = entropy_curves(x1, res['dH_t_bound'], res['t'], args.interpolant)
    #logger.info('Final entropy bound: %s',
    #            H_t_bound[-1].item() if torch.is_tensor(H_t_bound[-1]) else H_t_bound[-1])
    #logger.info('Final Gaussian entropy estimate: %s',
    #            H_t_gaussian[-1].item() if torch.is_tensor(H_t_gaussian[-1]) else H_t_gaussian[-1])

    #fig = plt.figure(figsize=(7, 5))
    #plt.plot(res['t'].detach().cpu(), H_t_bound, label='bound')
    #gauss_cpu = H_t_gaussian.detach().cpu() if torch.is_tensor(H_t_gaussian) else H_t_gaussian
    #plt.plot(t.detach().cpu(), gauss_cpu, '--', label='Gaussian entropy estimate')
    #plt.xlabel('t')
    #plt.ylabel('entropy')
    #plt.legend()
    #plt.title(config)
    #plt.tight_layout()
    #fig.savefig(fig_dir / f'{config}_entropy_bound.png', dpi=150, bbox_inches='tight')
    #plt.close(fig)

    #plot_entropy_bound_evolution(res['dH_t_bound'], H_t_bound, H_t_gaussian, res['t'])
    #plt.suptitle(config)
    #plt.savefig(fig_dir / f'{config}_entropy_bound_detail.png', dpi=150, bbox_inches='tight')
    #plt.close('all')

    logger.info('Saved diagnostic figures to %s', fig_dir)


def entropy_curves(x_ref, dH_t_bound, t_used, interpolant):
    d = x_ref.shape[-2] * x_ref.shape[-1]
    H_p_0 = (np.log(2 * np.pi) + 1) * d / 2
    H_t_bound = dH_t_bound.cumsum(0).detach().cpu() / (t_used.shape) + H_p_0
    H_t_gaussian = compute_gaussian_entropy(x_ref, interpolant, t_used)
    return H_t_bound, H_t_gaussian


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # `root` = this script's own directory (turbulence/), set at the top of
    # the file. Default outdir to root itself (no saved_results/ nesting) —
    # must match the notebook workflow's outdir exactly, or the two can
    # never see each other's runs as duplicates via resolve_config_for_loading.
    outdir = Path(args.outdir) if args.outdir else root
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    W = DefineWavelet('Db', m=3, device=device)
    raw_data = load_turbulence_1d().to(device)
    data_tensor = split_periodize_reshape(raw_data, args.subseries_len)
    B, C, pre_len = data_tensor.shape

    coarse_grained = False
    scales_needed = 0
    if pre_len > args.target_len:
        scales_needed = int(np.log2(pre_len / args.target_len))
        for _ in range(scales_needed):
            data_tensor = W.decompose(data_tensor)[1]
        coarse_grained = scales_needed > 0
    else:
        print(f'-- Current length {pre_len} <= target_len {args.target_len}; skipping coarse graining.')

    x1 = normalize(data_tensor[:args.n1]).to(device)
    B, channels, M = x1.shape

    config, exp_dir, fig_dir, potentials_dir, logger, loaded = resolve_or_setup_experiment_output(
        outdir, args, M, device, coarse_grained=coarse_grained,
        extra_metadata={
            'M': M, 'B': B, 'channels': channels,
            'coarse_grained': coarse_grained, 'pre_coarse_grain_length': pre_len,
        },
        include_potentials_dir=True,
    )
    logger.info('Data original shape: %s', raw_data.shape)
    logger.info('Calculated scales applied: %d', scales_needed)
    logger.info('x1 final shape: %s (coarse_grained=%s)', tuple(x1.shape), coarse_grained)

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