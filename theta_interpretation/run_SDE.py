#!/usr/bin/env python
"""
run_SDE.py — CLI launcher for the "theta_interpretation" experiment: fit
MGD/SDE on scalar 1D Gaussian data with a correctly-specified potential set
(e.g. just 'x2') vs. a deliberately misspecified one (e.g. 'x2' + 'x4'), and
inspect what theta does in each case.

Data layout: x1 ~ N(0, data_sigma^2), n1 i.i.d. scalar draws, shape (n1, 1).
This is genuinely (B, C) -- NOT (B, C, 1) -- because codes/sde_routines.py's
SDE class dispatches on len(x_1.shape): 2 -> scalar (no wavelet machinery),
3 -> 1D field of length T (wavelet scattering over the last axis). A single
real number per sample has no signal-length axis to convolve over, so this
experiment uses the scalar branch and codes/potentials_builder.py's
get_scalar_potentials() (pointwise Monomial(degree) potentials: 'x1'..'x9'
-> x**degree), not the wavelet-scattering get_1d_potentials() that
run_experiment() in codes/utils_experiment.py hardcodes.

NOTE on why this file doesn't just call the shared run_experiment()/
build_config_name()/resolve_or_setup_experiment_output() from
codes/utils_experiment.py: those hardcode get_1d_potentials() (wavelet-only
-- would silently build an empty potentials dict for terms like 'x2') and a
naming scheme built around a signal length M and jets/turbulence/gaussian-fBm
prefixes, none of which apply to a scalar target. Local equivalents are
defined below instead of editing that shared file, so jets/turbulence/
gaussian_experiment are untouched.

NOTE on imports: the sys.path manipulation below is intentional and must
stay exactly as written, matching the other run_SDE.py entry points -- the
project's 'codes' package relies on these paths being inserted, in this
order, before anything under it is imported.

NOTE on output layout: every run gets its own self-contained folder,
  <outdir>/experiments/<group_name>/<config>/
      config.json
      logs/run.log
      figures/*.png
      fitted_potentials/   (created for structural parity; Monomial has no
                             .fit()/.is_fitted, so nothing actually lands
                             here -- see SDE.__init__'s hasattr(pot, "fit")
                             gate in codes/sde_routines.py)
  See main() below.
"""
import argparse
import hashlib
import json
import sys
import time as timer
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


# ── sys.path setup ──────────────────────────────────────────────────────────
root = Path(__file__).resolve().parent
project_root = root.parent

def _extend_syspath(root: Path, project_root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    codes_path = project_root / 'codes'
    if codes_path.is_dir() and str(codes_path) not in sys.path:
        sys.path.insert(0, str(codes_path))

_extend_syspath(root, project_root)

from codes.sde_routines import *      # noqa: E402  (also pulls in get_scalar_potentials
                                       # transitively, via potentials_builder.py)
from codes.utils import *             # noqa: E402  (save_results_theta_reg, normalize, ...)
from codes.check_moments import *     # noqa: E402  (plot_moment_matching)
from codes.utils_experiment import (  # noqa: E402  -- generic helpers only; naming/run
    setup_logging,                    # helpers are defined locally below (see module
    resolve_config_for_loading,       # docstring) instead of importing the shared
    try_load_experiment,              # build_config_name/run_experiment.
)

root = Path.cwd()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── argument parsing ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='MGD/SDE theta-interpretation experiment on scalar 1D Gaussian data: '
                    'study fitted theta under a correctly-specified potential set '
                    '(e.g. x^2) vs. a misspecified one (e.g. x^2 + x^4).'
    )
    p.add_argument('--timestamp', type=str, default=None, help='Timestamp for the run')

    # Data
    p.add_argument('--n1', type=int, default=5000,
                    help='Number of i.i.d. scalar Gaussian samples')
    p.add_argument('--data_sigma', type=float, default=1.0,
                    help='Standard deviation of the true target N(0, data_sigma^2). '
                         'Distinct from --sigma below, which is the SDE diffusion coefficient.')

    # Potentials
    p.add_argument('--terms', nargs='+', type=str, default=['x2'],
                    help="Scalar potential terms, from get_scalar_potentials's registry "
                         "(codes/potentials_builder.py): 'x1'..'x9' (Monomial(i) = x**i), "
                         "'x_abs' (|x|), 'bimodal'. The correctly-specified model for a "
                         "Gaussian is ['x2']; adding e.g. 'x4' on top is the deliberately "
                         "misspecified case this experiment is meant to probe.")

    # SDE (same roles/defaults as gaussian_experiment/run_SDE.py)
    p.add_argument('--nt', type=int, default=2000,
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
                         'regularization strength)')
    p.add_argument('--n_subsample', type=int, default=100,
                    help='n_subsample passed to Solver.forward_regularised')

    # Batch
    p.add_argument('--batch_size', type=int, default=None,
                    help='Batch size for potential evaluations (default: n1)')

    # Diagnostics
    p.add_argument('--n_bins', type=int, default=100,
                    help='Histogram bins for the marginal-density diagnostic')
    p.add_argument('--moment_threshold', type=float, default=1e-8,
                    help='Threshold used in plot_moment_matching')

    # Experiment / bookkeeping
    p.add_argument('--outdir', type=str, default=None,
                    help='Base directory. Defaults to this script\'s own '
                         'directory. Each run gets its own subfolder at '
                         '<outdir>/experiments/<group_name>/<config>/ containing '
                         'config.json, logs/ and figures/.')
    p.add_argument('--label', type=str, default=None,
                    help='Optional extra label appended to the config name')
    p.add_argument('--force_rerun', action='store_true',
                    help='Ignore any existing saved results and rerun from scratch')
    p.add_argument('--no_save_aux_moments', action='store_true',
                    help='Disable saving of barphi_e / barphi_p aux moments')
    p.add_argument('--seed', type=int, default=0, help='Random seed')

    return p.parse_args()


# ── local config naming (deliberately not codes/utils_experiment.py's
#    build_config_name/build_group_name -- see module docstring) ────────────
def _config_name_parts(args, include_seed=True):
    terms_hash = hashlib.md5('|'.join(sorted(args.terms)).encode()).hexdigest()[:8]
    parts = [
        'thetainterp',
        f'sigmadata{args.data_sigma}',
        f'sigma{args.sigma}',
        f'nt{args.nt}',
        f'n1_{args.n1}',
        f'lam{args.lam}',
    ]
    if include_seed:
        parts.append(f'seed_{args.seed}')
    parts.append(f'terms{terms_hash}')
    return parts


def build_config_name(args, include_timestamp=True):
    parts = _config_name_parts(args, include_seed=True)
    if args.label:
        parts.append(args.label)
    if include_timestamp and args.timestamp:
        parts.append(args.timestamp)
    return '_'.join(parts)


def build_group_name(args):
    parts = _config_name_parts(args, include_seed=False)
    if args.label:
        parts.append(args.label)
    return '_'.join(parts)


# ── local experiment output setup (mirrors codes/utils_experiment.py's
#    setup_experiment_output/resolve_or_setup_experiment_output, but keyed
#    off the local naming above and without an M/signal-length field) ───────
def setup_experiment_output(outdir, config, args, extra_metadata=None,
                             include_potentials_dir=False):
    group_name = build_group_name(args)
    exp_dir = outdir / 'experiments' / group_name / config
    fig_dir = exp_dir / 'figures'
    log_dir = exp_dir / 'logs'
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    potentials_dir = None
    if include_potentials_dir:
        potentials_dir = exp_dir / 'fitted_potentials'
        potentials_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir, config)
    logger.info('Config: %s', config)
    logger.info('Experiment folder: %s', exp_dir)
    logger.info('Arguments: %s', vars(args))

    config_dict = dict(vars(args))
    config_dict['config_name'] = config
    if extra_metadata:
        config_dict.update(extra_metadata)
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

    return exp_dir, fig_dir, potentials_dir, logger


def resolve_or_setup_experiment_output(outdir, args, device, extra_metadata=None,
                                        include_potentials_dir=False):
    group_name = build_group_name(args)
    config_prefix = build_config_name(args, include_timestamp=False)
    resolved = None if args.force_rerun else resolve_config_for_loading(outdir, config_prefix)

    if resolved is not None:
        loaded = try_load_experiment(outdir, resolved, device)
        if loaded is not None:
            exp_dir = outdir / 'experiments' / group_name / resolved
            fig_dir = exp_dir / 'figures'
            potentials_dir = exp_dir / 'fitted_potentials' if include_potentials_dir else None
            exp_dir.mkdir(parents=True, exist_ok=True)
            fig_dir.mkdir(parents=True, exist_ok=True)
            if potentials_dir is not None:
                potentials_dir.mkdir(parents=True, exist_ok=True)
            logger = setup_logging(exp_dir / 'logs', resolved)
            logger.info('Reusing existing experiment: %s (no new folder created)', resolved)
            return resolved, exp_dir, fig_dir, potentials_dir, logger, loaded

    config = build_config_name(args, include_timestamp=True)
    exp_dir, fig_dir, potentials_dir, logger = setup_experiment_output(
        outdir, config, args, extra_metadata=extra_metadata,
        include_potentials_dir=include_potentials_dir,
    )
    return config, exp_dir, fig_dir, potentials_dir, logger, None


# ── data ──────────────────────────────────────────────────────────────────────
def generate_gaussian_data(args, device):
    """x1 ~ N(0, data_sigma^2), shape (n1, 1) -- the (B, C) 'scalar' layout
    SDE.__init__ dispatches on (see module docstring)."""
    return args.data_sigma * torch.randn(args.n1, 1, device=device)


# ── run / reuse (local counterpart of codes/utils_experiment.py's
#    run_experiment(), dispatching to get_scalar_potentials instead of
#    get_1d_potentials) ───────────────────────────────────────────────────────
def run_experiment(args, config, x1, t, logger, outdir, device, potentials_save_dir=None):
    if not args.force_rerun:
        config_prefix = build_config_name(args, include_timestamp=False)
        resolved = resolve_config_for_loading(outdir, config_prefix)
        if resolved is not None:
            loaded = try_load_experiment(outdir, resolved, device)
            if loaded is not None:
                logger.info('Loaded existing results for %s', resolved)
                return loaded

    logger.info('Running experiment: %s', config)
    logger.info('terms: %s', args.terms)

    potentials = get_scalar_potentials(args.terms)
    if not potentials:
        raise ValueError(
            f"get_scalar_potentials(args.terms) built an empty potentials dict for "
            f"terms={args.terms}. Expected values from its registry: 'x1'..'x9', "
            f"'x_abs', 'bimodal' (see codes/potentials_builder.py)."
        )

    batch_size = args.batch_size or x1.shape[0]
    nb_workers = x1.shape[0]
    nb_interpolants = x1.shape[0]

    t0 = timer.time()
    Solver = SDE(
        x1, nb_workers, nb_interpolants, t, args.sigma, potentials, batch_size,
        device=device, regularization=args.regularization, interpolant=args.interpolant,
        potentials_save_dir=potentials_save_dir,
    )
    xt, barphi_e, barphi_p, eta_t, theta_t, dH_t_bound, Theta_reg = Solver.forward_regularised(
        lam=args.lam, n_subsample=args.n_subsample,
    )
    logger.info('SDE integration finished in %.1f s', timer.time() - t0)

    save_results_theta_reg(xt, theta_t, dH_t_bound, t, outdir, config, Theta_reg=Theta_reg)

    if not args.no_save_aux_moments:
        torch.save(
            {'barphi_e': barphi_e.detach().cpu(), 'barphi_p': barphi_p.detach().cpu()},
            outdir / 'saved_results' / 'aux_moments' / f'{config}_aux_moments.pt',
        )

    return {
        'xt': xt, 'theta_t': theta_t, 'Theta_reg': Theta_reg, 't': t,
        'dH_t_bound': dH_t_bound, 'barphi_e': barphi_e, 'barphi_p': barphi_p,
        'loaded': False,
    }


# ── diagnostics ───────────────────────────────────────────────────────────────
def plot_marginal_histogram(x1, xt, data_sigma, fig_dir, config, n_bins=100):
    """Data vs. synthesized marginal, against the analytic N(0, data_sigma^2)
    density -- the ground truth this whole experiment is trying to recover."""
    x1_np = x1.detach().cpu().numpy().ravel()
    xt_np = xt.detach().cpu().numpy().ravel()

    x_min = min(x1_np.min(), xt_np.min())
    x_max = max(x1_np.max(), xt_np.max())
    x_grid = np.linspace(x_min, x_max, 400)
    pdf_true = np.exp(-x_grid**2 / (2 * data_sigma**2)) / np.sqrt(2 * np.pi * data_sigma**2)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(x1_np, bins=n_bins, density=True, alpha=0.5, label='data ($x_1$)')
    ax.hist(xt_np, bins=n_bins, density=True, alpha=0.5, label='synthesized ($x_t$)')
    ax.plot(x_grid, pdf_true, 'k--', lw=1.5, label=fr'target $\mathcal{{N}}(0, {data_sigma}^2)$')
    ax.set_xlabel('x')
    ax.set_ylabel('density')
    ax.legend()
    ax.set_title(config)
    fig.tight_layout()
    fig.savefig(fig_dir / 'marginal_histogram.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_theta_trajectory(theta_t, term_names, fig_dir, config):
    theta_t = theta_t.detach().cpu()
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(theta_t.shape[-1]):
        label = term_names[i] if i < len(term_names) else f'coef_{i}'
        ax.plot(theta_t[:, i], label=label)
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_xlabel('SDE step (stored)')
    ax.set_ylabel(r'$\theta_t$')
    ax.legend()
    ax.set_title(config)
    fig.tight_layout()
    fig.savefig(fig_dir / 'theta_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_diagnostics(x1, result, args, config, fig_dir, logger):
    term_names = list(get_scalar_potentials(args.terms).keys())

    plot_marginal_histogram(x1, result['xt'], args.data_sigma, fig_dir, config,
                             n_bins=args.n_bins)
    plot_theta_trajectory(result['theta_t'], term_names, fig_dir, config)

    if result.get('barphi_e') is not None and result.get('barphi_p') is not None:
        plot_moment_matching(
            result['barphi_e'], result['barphi_p'], result['t'], args.moment_threshold,
            save={"filename": fig_dir / "moment_matching.png", "title": config},
        )
    else:
        logger.warning('No aux moments available (loaded run without saved aux file?); '
                        'skipping moment-matching plot.')

    theta_final = result['theta_t'][-1].detach().cpu()
    logger.info('Final theta (order matches %s): %s', term_names, theta_final.tolist())
    if 'x2' in term_names:
        i2 = term_names.index('x2')
        # Reference value only, NOT independently re-derived against this
        # codebase's current SDE formulation -- see the notebook for the
        # derivation and cross-check it empirically against this run.
        target_theta_x2 = -1.0 / (2 * args.data_sigma**2)
        logger.info(
            'Reference (unverified against this run): for target N(0, data_sigma^2) '
            'under p(x) ~ exp(sum_i theta_i * phi_i(x)), theta on x^2 is expected near '
            '%.6f. Fitted theta_x2[-1] = %.6f',
            target_theta_x2, theta_final[i2].item(),
        )

    logger.info('Saved diagnostic figures to %s', fig_dir)


# ── notebook-friendly entry point ────────────────────────────────────────────
def make_args(terms, **overrides):
    """Same field set as parse_args() above, as a plain namespace -- lets a
    notebook call run_and_diagnose()/run_experiment() without going through
    argparse/sys.argv. `outdir` defaults to this script's own directory
    (`root`, set at module import time); pass outdir=... to redirect it (e.g.
    for a scratch/smoke-test run)."""
    defaults = dict(
        timestamp=None,
        n1=5000, data_sigma=1.0, terms=terms,
        nt=2000, sigma=0.3, schedule_exponent=2, interpolant='Cos',
        regularization=1e-1, lam=5e-6, n_subsample=100, batch_size=None,
        n_bins=100, moment_threshold=1e-8,
        outdir=str(root), label=None, force_rerun=False,
        no_save_aux_moments=False, seed=0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def run_and_diagnose(args, outdir=None):
    """Mirrors main() below: generate data, resolve/run the experiment, save
    diagnostics, return everything needed for inspection. `outdir` defaults
    to `Path(args.outdir)`."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(outdir) if outdir is not None else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # main() creates these once before its first run; this bypasses main()
    # so it has to do the same here.
    for sub in ('samples', 'lagrange_multipliers', 'lagrange_multipliers_regularised',
                'entropy_bounds', 'sampling_times', 'aux_moments'):
        (outdir / 'saved_results' / sub).mkdir(parents=True, exist_ok=True)

    x1 = generate_gaussian_data(args, device)

    config, exp_dir, fig_dir, potentials_dir, logger, loaded = resolve_or_setup_experiment_output(
        outdir, args, device,
        extra_metadata={'B': x1.shape[0], 'channels': x1.shape[1]},
        include_potentials_dir=True,
    )

    t = 1 - (1 - torch.linspace(0, 1, args.nt + 1)) ** args.schedule_exponent

    if loaded is not None:
        result = loaded
    else:
        result = run_experiment(args, config, x1, t, logger, outdir, device,
                                 potentials_save_dir=potentials_dir)

    save_diagnostics(x1, result, args, config, fig_dir, logger)

    return dict(args=args, x1=x1, config=config, fig_dir=fig_dir, result=result)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir) if args.outdir else root
    outdir.mkdir(parents=True, exist_ok=True)
    for sub in ('samples', 'lagrange_multipliers', 'lagrange_multipliers_regularised',
                'entropy_bounds', 'sampling_times', 'aux_moments'):
        (outdir / 'saved_results' / sub).mkdir(parents=True, exist_ok=True)

    x1 = generate_gaussian_data(args, device)
    B, channels = x1.shape

    config, exp_dir, fig_dir, potentials_dir, logger, loaded = resolve_or_setup_experiment_output(
        outdir, args, device,
        extra_metadata={'B': B, 'channels': channels},
        include_potentials_dir=True,
    )
    logger.info('x1 (scalar Gaussian data) shape: %s', tuple(x1.shape))
    logger.info('data_sigma = %s', args.data_sigma)

    t = 1 - (1 - torch.linspace(0, 1, args.nt + 1)) ** args.schedule_exponent
    t_rounded = torch.round(t, decimals=4)
    t_final = int((t_rounded == 1.0).nonzero(as_tuple=True)[0][0])
    logger.info(f"t_final = {t_final}/{len(t)} (last t = {t[t_final-1].item():.6f}, "
        f"dropping {len(t) - t_final} redundant trailing points at 1.0000)")

    if loaded is not None:
        result = loaded
    else:
        result = run_experiment(args, config, x1, t[:t_final], logger, outdir, device,
                                 potentials_save_dir=potentials_dir)

    save_diagnostics(x1, result, args, config, fig_dir, logger)
    logger.info('Done.')


if __name__ == '__main__':
    main()
