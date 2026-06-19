"""
variance_experiment.py — Variance comparison: MGD raw theta_t vs regularized Theta_reg
                  (Florentin's regularised_theta post-processing) over K
                  independent trials, on synthetic Brownian-motion (BM) data.

This supersedes the old run_variance_experiment.py, updated to the current
codebase API as used in the production notebook (forward_regularised /
potentials_builder / filters_bank), instead of the old Solver() + smooth_theta
/ smooth_theta_renormalized API.

Usage (direct):
    python variance_BM.py [options]

Usage (via SLURM launcher):
    bash launch_variance_BM.sh

What it does:
    For each of K independent trials:
      1. Re-initialises the SDE with fresh Gaussian workers (x_k ~ N(0,I));
         data x1 and potentials are shared across trials.
      2. Runs Solver.forward_regularised(lam, n_subsample) → returns both the
         raw MGD theta_t path and the regularized Theta_reg path.
    After all K trials:
      - Saves per-trial arrays and cross-trial variance/mean to a .npz file.
      - Saves diagnostic figures (variance over time, per-component variance
        at final time, trajectories with ±1 std shading) for both estimators.

NOTE — two assumptions you should double-check against your current code:

  (1) Theta_reg is returned on its own (subsampled) time grid whose length is
      not given explicitly by forward_regularised, exactly as in the
      notebook (theta_t and Theta_reg have different lengths, plotted on a
      normalized [0,1] x-axis there). We do the same here: the regularized
      estimator's time axis is `linspace(0, 1, T_reg)`. If Theta_reg's actual
      subsampled times are available some other way (e.g. an attribute on
      Solver), swap that in for a faithful t-axis.

Output (in --outdir):
    {config}__variance_results.npz containing:
        t_mgd, t_reg     : time axes for the two estimators
        theta_mgd        : (K, T_mgd, r)  raw MGD theta_t per trial
        theta_reg        : (K, T_reg, r)  regularized Theta_reg per trial
        var_mgd, var_reg : (T, r) variance across K trials
        mean_mgd, mean_reg: (T, r) mean across K trials
"""

import argparse
import logging
import sys
import time as timer
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── sys.path setup ──────────────────────────────────────────────────────────
root = Path(__file__).resolve().parent


def _extend_syspath(root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Add the 'codes' directory itself to sys.path so flat imports
    # (e.g. `from sde_routines import *`) work the same way they do in the
    # notebook, which inserts '../codes' directly.
    codes_path = root / 'codes'
    if codes_path.is_dir() and str(codes_path) not in sys.path:
        sys.path.insert(0, str(codes_path))

    if (root / 'data').is_dir():
        sys.path.insert(0, str(root / 'data'))


_extend_syspath(root)

# ── current API (matches the production notebook) ────────────────────────
from codes.sde_routines import *          # noqa: E402  -> SDE, .forward_regularised
from codes.potentials_builder import *    # noqa: E402  -> get_1d_potentials
from codes.filters_bank import *          # noqa: E402  -> return_Filters
from codes.utils import *                 # noqa: E402  -> normalize, etc.

from codes.ortho_wavelet.ReadyToUseWavelets import * 


from data.data_loader import *         # noqa: E402  -> generate_BM (see NOTE 1 above)



# ── argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='Variance experiment: MGD raw theta vs regularized Theta_reg (BM data)'
    )

    # Data
    p.add_argument('--n1', type=int, default=100_000,
                    help='Number of BM sample paths (dataset size)')
    p.add_argument('--scales', type=int, default=3,
                    help='coarse graning scales)')

    # Scattering
    p.add_argument('--J', type=int, default=7,
                    help='Number of wavelet scales')
    p.add_argument('--Q', type=int, default=1,
                    help='Wavelets per octave (for the Q-resolution filter bank)')
    p.add_argument(
        '--terms',
        nargs='+',
        type=str,
        default=['Scattering_Second_Order'],
        help='List of scattering terms to include',
    )

    # SDE
    p.add_argument('--nt', type=int, default=5000,
                    help='Number of SDE integration steps')
    p.add_argument('--sigma', type=float, default=5.0,
                    help='Diffusion coefficient sigma')
    p.add_argument('--schedule_exponent', type=int, default=1,
                    help='Exponent for time schedule: t = 1-(1-linspace)^exp')
    p.add_argument('--interpolant', type=str, default='Cos',
                    help='Interpolant type: Cos | Linear | VarPreserv | Sqrt')
    p.add_argument('--regularization', type=float, default=1e-10,
                    help='Tikhonov regularization for the Gram matrix (SDE constructor)')

    # Florentin's regularized-theta post-processing
    p.add_argument('--lam', type=float, default=2e-5,
                    help='Regularization weight lambda for forward_regularised')
    p.add_argument('--n_subsample', type=int, default=100,
                    help='Number of subsampled time points for the regularized estimator')

    # Batch
    p.add_argument('--batch_size', type=int, default=None,
                    help='Batch size for potential evaluations (default: n1)')

    # Experiment
    p.add_argument('--K', type=int, default=50,
                    help='Number of independent trials')
    p.add_argument('--seed', type=int, default=0,
                    help='Base random seed (trial k uses seed + k)')

    # Output
    p.add_argument('--outdir', type=str, default='saved_results',
                    help='Directory for outputs')

    return p.parse_args()


# ── logging ───────────────────────────────────────────────────────────────────
def setup_logging(outdir: Path, config: str) -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / f'{config}.log'

    logger = logging.getLogger('variance_bm')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s  %(levelname)-8s  %(message)s',
                             datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ── data ──────────────────────────────────────────────────────────────────────
def make_data(args, device, logger):
    """
    Generate synthetic Brownian-motion sample paths.
    See NOTE (1) at the top of this file if `generate_BM` does not match
    your actual data_generator API.
    """
    # Orthogonal wavelet
    W = DefineWavelet('Db', m=args.scales, device=device)

    Data = load_turbulence_1d()
    n1 = 1024
    Data = split_periodize_reshape(Data, n1)

    # Same preprocessing as in the original notebook.
    for j in range(3):
        Data = W.decompose(Data)[1]

    x1 = normalize(Data[:args.n1]).to(device) # choose here if you want less data 
    print('Data shape after preprocessing:', Data.shape)
    print('x1 shape:', x1.shape)
    M = x1.shape[-1]

    logger.info(f'  x1.shape = {x1.shape}')
    logger.info(f'  M = {M}')
    return x1, M


# ── potentials ───────────────────────────────────────────────────────────────
def make_potentials(M, args, device, logger):
    logger.info(f'  terms   : {args.terms}')

    # Q=1 filters + phi (low-pass), and a separate Q-per-octave filter bank,
    # exactly as built in the production notebook.
    filters, filters_Phi = return_Filters(M, args.J, 1, device=device, include_phi=True)
    filters_Q = return_Filters(M, args.J, args.Q, device=device)

    potentials = get_1d_potentials(
        args.terms,
        args.J,
        filters,
        args.Q,
        filters_Q,
        filters_Phi,
        scalar_param=None,
        parallel=False,
    )
    return potentials


# ── time axis inference ─────────────────────────────────────────────────────
def infer_time_axis(T: int, t_sched: torch.Tensor) -> np.ndarray:
    """Map an estimator's output length T onto a time axis, using t_sched
    when the lengths line up exactly and falling back to a normalized
    [0, 1] axis otherwise (mirrors the notebook's normalized-progress plot)."""
    t_sched_np = t_sched.cpu().numpy()
    if T == t_sched_np.shape[0]:
        return t_sched_np
    if T == t_sched_np.shape[0] - 1:
        return t_sched_np[:-1]
    return np.linspace(0, 1, T)


# ── single trial ─────────────────────────────────────────────────────────────
def run_one_trial(
    trial_idx: int,
    x1: torch.Tensor,
    t_sched: torch.Tensor,
    potentials: dict,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> dict:
    """
    Run one full MGD forward pass (with regularized post-processing) with
    freshly sampled workers. Data x1 and potentials are shared across trials.
    """
    torch.manual_seed(args.seed + trial_idx)
    np.random.seed(args.seed + trial_idx)

    batch_size = args.batch_size if args.batch_size is not None else x1.shape[0]
    nb_workers = x1.shape[0]
    nb_interpolants = x1.shape[0]

    Solver = SDE(
        x1,
        nb_workers,
        nb_interpolants,
        t_sched,
        args.sigma,
        potentials,
        batch_size,
        device=device,
        regularization=args.regularization,
        interpolant=args.interpolant,
    )

    t0 = timer.time()
    xt, barphi_e, barphi_p, eta_t, theta_t, dH_t_bound, Theta_reg = Solver.forward_regularised(
        lam=args.lam, n_subsample=args.n_subsample
    )
    logger.info(
        f'  [trial {trial_idx:3d}] forward_regularised done in {timer.time()-t0:.1f}s  '
        f'| theta_t.shape={tuple(theta_t.shape)}  Theta_reg.shape={tuple(Theta_reg.shape)}'
    )

    theta_mgd = theta_t.detach().cpu().numpy() if torch.is_tensor(theta_t) else np.asarray(theta_t)
    theta_reg = Theta_reg.detach().cpu().numpy() if torch.is_tensor(Theta_reg) else np.asarray(Theta_reg)

    return {
        'theta_mgd': theta_mgd,   # (T_mgd, r)
        'theta_reg': theta_reg,   # (T_reg, r)
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    term_tag = "-".join(t.replace("Scattering_", "").replace("_Order", "") for t in args.terms)

    config = (
        f'variance_bm'
        f'_scales{args.scales}_J{args.J}_Q{args.Q}'
        f'_sigma{args.sigma}'
        f'_n1_{args.n1}'
        f'_nt{args.nt}'
        f'_K{args.K}'
        f'_lam{args.lam}_nsub{args.n_subsample}'
        f'_terms{term_tag}'
    )

    outdir = Path(args.outdir)
    logger = setup_logging(outdir, config)

    logger.info('=' * 70)
    logger.info('Variance experiment: MGD raw theta vs regularized Theta_reg')
    logger.info(f'  config  : {config}')
    logger.info(f'  device  : {device}')
    logger.info(f'  K trials: {args.K}')
    logger.info(f'  lam     : {args.lam}')
    logger.info(f'  n_subsample: {args.n_subsample}')
    logger.info('=' * 70)

    # ── data (shared across all trials) ──────────────────────────────────────
    logger.info('Generating BM data …')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x1, M = make_data(args, device, logger)
    n1_actual, channels, M_actual = x1.shape

    logger.info("M and M actual", M, M_actual)

    # ── potentials (shared across all trials) ─────────────────────────────────
    logger.info('Building potentials …')
    potentials = make_potentials(M_actual, args, device, logger)

    # ── time schedule (shared) ────────────────────────────────────────────────
    t_sched = 1 - (1 - torch.linspace(0, 1, args.nt + 1)) ** args.schedule_exponent

    # ── K trials ─────────────────────────────────────────────────────────────
    all_mgd = []   # will become (K, T_mgd, r)
    all_reg = []   # will become (K, T_reg, r)

    for k in range(args.K):
        logger.info(f'--- Trial {k+1}/{args.K} ---')
        result = run_one_trial(k, x1, t_sched, potentials, args, device, logger)
        all_mgd.append(result['theta_mgd'])
        all_reg.append(result['theta_reg'])

    # ── stack → (K, T, r) ────────────────────────────────────────────────────
    all_mgd = np.stack(all_mgd, axis=0) # (K, T, r) 
    all_reg = np.stack(all_reg, axis=0)

    t_mgd_np = infer_time_axis(all_mgd.shape[1], t_sched)
    t_reg_np = infer_time_axis(all_reg.shape[1], t_sched)

    # ── compute variance and mean across trials (axis=0) ─────────────────────
    var_mgd = np.var(all_mgd, axis=0, ddof=1)   # (T_mgd, r)
    var_reg = np.var(all_reg, axis=0, ddof=1)   # (T_reg, r)

    mean_mgd = np.mean(all_mgd, axis=0) # (T, r) 
    mean_reg = np.mean(all_reg, axis=0)

    # ── print final statistics ────────────────────────────────────────────────
    logger.info('=' * 70)
    logger.info('Final statistics at last time step of each estimator')
    logger.info('=' * 70)

    def log_final_stats(name, mean, var):
        mean_T = mean[-1]
        var_T = var[-1]
        std_T = np.sqrt(var_T)
        logger.info(f'{name}:')
        logger.info(f'  mean theta(T): {np.array2string(mean_T, precision=4)}')
        logger.info(f'  std  theta(T): {np.array2string(std_T,  precision=4)}')
        logger.info(f'  var  theta(T): {np.array2string(var_T,  precision=4)}')
        logger.info('')

    log_final_stats('MGD (raw)', mean_mgd, var_mgd)
    log_final_stats('Regularized (Theta_reg)', mean_reg, var_reg)

    # ── save ─────────────────────────────────────────────────────────────────
    npz_path = outdir / f'{config}__variance_results.npz'
    np.savez(
        npz_path,
        # metadata
        K=np.array(args.K),
        sigma=np.array(args.sigma),
        lam=np.array(args.lam),
        n_subsample=np.array(args.n_subsample),
        # time axes
        t_mgd=t_mgd_np,
        t_reg=t_reg_np,
        # per-trial trajectories
        theta_mgd=all_mgd,   # (K, T_mgd, r)
        theta_reg=all_reg,   # (K, T_reg, r)
        # statistics
        var_mgd=var_mgd,
        var_reg=var_reg,
        mean_mgd=mean_mgd,
        mean_reg=mean_reg,
    )
    logger.info(f'Results saved -> {npz_path}')

    # ── diagnostic figure: total variance over time + per-component bars ─────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogy(t_mgd_np, var_mgd.sum(axis=1), label='MGD (raw)', color='steelblue')
    ax.semilogy(t_reg_np, var_reg.sum(axis=1), label='Regularized', color='tomato')
    ax.set_xlabel('t (normalized)')
    ax.set_ylabel('total variance  sum_r Var[theta_r]')
    ax.set_title(f'Variance over time  (K={args.K})')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    r = var_mgd.shape[1]
    x_idx = np.arange(r)
    width = 0.35
    ax.bar(x_idx - width / 2, var_mgd[-1], width=width, label='MGD (raw)', color='steelblue', alpha=0.8)
    ax.bar(x_idx + width / 2, var_reg[-1], width=width, label='Regularized', color='tomato', alpha=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('theta component index')
    ax.set_ylabel('Var[theta_hat] at final t')
    ax.set_title('Per-component variance at final t')
    ax.legend()
    ax.grid(True, axis='y', which='both', alpha=0.3)

    fig.tight_layout()
    fig_path = outdir / f'{config}__variance_summary.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Diagnostic figure -> {fig_path}')

    # ── trajectory plots with variance shading ───────────────────────────────
    r = mean_mgd.shape[1]
    fig, axes = plt.subplots(r, 1, figsize=(10, 3 * r), sharex=True)
    if r == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t_mgd_np, mean_mgd[:, i], color='steelblue', label='MGD (raw)')
        ax.fill_between(
            t_mgd_np,
            mean_mgd[:, i] - np.sqrt(var_mgd[:, i]),
            mean_mgd[:, i] + np.sqrt(var_mgd[:, i]),
            color='steelblue', alpha=0.2,
        )

        ax.plot(t_reg_np, mean_reg[:, i], color='tomato', label='Regularized')
        ax.fill_between(
            t_reg_np,
            mean_reg[:, i] - np.sqrt(var_reg[:, i]),
            mean_reg[:, i] + np.sqrt(var_reg[:, i]),
            color='tomato', alpha=0.2,
        )

        ax.set_ylabel(f'theta[{i}]')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('t (normalized)')
    axes[0].set_title('Theta trajectories with +/-1 std shading')
    axes[0].legend()

    fig.tight_layout()
    traj_fig_path = outdir / f'{config}__theta_trajectories.png'
    fig.savefig(traj_fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Trajectory figure -> {traj_fig_path}')
    logger.info('Done.')


if __name__ == '__main__':
    main()