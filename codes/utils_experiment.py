"""
Moment-Guided Diffusion (MGD) - expeiriment saving and loading utilities.
"""

from pathlib import Path
import numpy as np
import torch
from typing import Dict, Any, Tuple
import sys
import logging
import json
import time as timer
from pathlib import Path

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


from codes.sde_routines import *      # noqa: E402
from codes.utils import *             # noqa: E402
from codes.check_moments import *     # noqa: E402
from codes.ortho_wavelet.ReadyToUseWavelets import *



#################################################################################
#### designing and running experiment. works the same for lagrangian turbulence and jets 
import hashlib
import contextlib

# ── logging ───────────────────────────────────────────────────────────────────
def setup_logging(log_dir: Path, config: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'run.log'

    logger = logging.getLogger(config)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s  %(levelname)-8s  %(message)s',
                             datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


@contextlib.contextmanager
def logged_run(logger):
    """Wrap the body of main() in `with logged_run(logger):` so run.log
    always ends with an explicit success/failure line and total duration,
    instead of just stopping mid-stream on an uncaught exception (e.g. the
    SDE time-budget guard in sde_routines.py) with the actual error visible
    only in the SLURM stderr file, separate from run.log.
    """
    t0 = timer.time()
    logger.info('Run started')
    try:
        yield
    except Exception:
        logger.exception('Run FAILED after %.1f s', timer.time() - t0)
        raise
    else:
        logger.info('Run succeeded in %.1f s', timer.time() - t0)


# ── experiment output setup ─────────────────────────────────────────────────
def setup_experiment_output(outdir: Path, config: str, args, M, extra_metadata: dict = None,
                             include_potentials_dir: bool = False):
    """Create a run's output directory tree, start its logger, and write
    config.json — the provenance record every entry point (CLI scripts,
    notebooks) should produce identically, so a run's parameters stay
    recoverable from disk however it was launched.

    `args` must support vars(args) (argparse.Namespace / SimpleNamespace).
    `extra_metadata` is merged into config.json on top of vars(args) — use
    it for derived values (M, coarse_grained, etc.) that aren't CLI args.
    `include_potentials_dir` creates exp_dir/fitted_potentials/ (turbulence
    needs this for Solver's potentials_save_dir; jets doesn't use it).

    exp_dir nests under experiments/<group_name>/<config>/, where
    group_name (build_group_name) is the same fields as config minus
    seed_<N> and the timestamp -- so a seed sweep launched with otherwise
    identical args lands together under one parent folder instead of each
    seed dumping a folder directly under experiments/.

    Returns (exp_dir, fig_dir, potentials_dir, logger). potentials_dir is
    None when include_potentials_dir is False.
    """
    group_name = build_group_name(args, M)
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


def _config_name_parts(args, M, include_seed=True):
    """Shared field-list builder behind build_config_name/build_group_name --
    keeps the two in sync (a field added to one can't silently drift out of
    sync with the other, the way the earlier terms-hash mismatch happened
    between a run's actual --terms and a notebook's copy of it)."""
    terms_hash = hashlib.md5('|'.join(sorted(args.terms)).encode()).hexdigest()[:8]

    re_number = getattr(args, 'Re_number', None)
    hurst = getattr(args, 'hurst', None)
    if re_number is not None:
        prefix = 'jetsynth'
    elif hurst is not None:
        prefix = 'gaussiansynth'
    else:
        prefix = 'turbulencesynth'

    parts = [prefix]
    if re_number is not None:
        parts.append(f'Re_number{re_number}')
    if hurst is not None:
        parts.append(f'H{hurst}_intermittency{getattr(args, "intermittency", 0.0)}')
    parts += [
        f'M{M}',
        f'J{args.J}',
        f'Q{args.Q}',
        f'sigma{args.sigma}',
        f'nt{args.nt}',
        f'n1_{args.n1}',
        f'lam{args.lam}',
    ]
    if include_seed:
        parts.append(f'seed_{args.seed}')
    parts.append(f'terms{terms_hash}')
    return parts


def build_config_name(args, M, coarse_grained=False, include_timestamp=True):
    parts = _config_name_parts(args, M, include_seed=True)
    if args.label:
        parts.append(args.label)
    if include_timestamp and args.timestamp:
        parts.append(args.timestamp)
    return '_'.join(parts)


def build_group_name(args, M, coarse_grained=False):
    """Same fields as build_config_name but WITHOUT seed_<N> or the
    timestamp -- the shared identity of a seed sweep (everything that stays
    fixed while --seed varies). Used to nest
    experiments/<group_name>/<config_name>/ so a sweep's many per-seed
    folders land under one shared parent instead of directly under
    experiments/."""
    parts = _config_name_parts(args, M, include_seed=False)
    if args.label:
        parts.append(args.label)
    return '_'.join(parts)


def resolve_config_for_loading(outdir, config_prefix):
    """Find saved configs matching config_prefix (timestamp-agnostic).
    Returns the exact config string to load, or None if nothing matches."""
    samples_dir = Path(outdir) / 'saved_results' / 'samples'
    if not samples_dir.exists():
        return None

    # Results are now saved as '<config>.pt'; strip that suffix so the
    # returned string is the plain config identifier either way — the same
    # thing callers (aux_moments lookup, load_results, logging) expect,
    # whether the match is a current .pt file or a legacy extensionless one.
    matches = sorted(
        p.stem if p.suffix == '.pt' else p.name
        for p in samples_dir.glob(f'{config_prefix}*')
    )
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    print(f"Found {len(matches)} saved runs matching '{config_prefix}':")
    for i, m in enumerate(matches):
        print(f"  [{i}] {m}")
    choice = input(f"Which one do you want to load? (0-{len(matches)-1}): ")
    return matches[int(choice)]


# ── experiment run / reload ──────────────────────────────────────────────────
def try_load_experiment(outdir, config, device):
    """Try loading saved SDE outputs and auxiliary moment-matching data."""
    try:
        # Pass the global outdir so it finds the old shared folders
        loaded = load_results(outdir, config)
        xt, theta_t, dH_t_bound, t_loaded, Theta_reg = loaded

        out = {
            'xt': xt.to(device) if torch.is_tensor(xt) else xt,
            'theta_t': theta_t,
            'dH_t_bound': dH_t_bound,
            't': t_loaded.to(device) if torch.is_tensor(t_loaded) else t_loaded,
            'Theta_reg': Theta_reg,
            'loaded': True,
        }
        aux_path = outdir / 'saved_results' / 'aux_moments' / f'{config}_aux_moments.pt'
        if aux_path.exists():
            aux = torch.load(aux_path, map_location=device)
            out.update(aux)
        else:
            out['barphi_e'] = None
            out['barphi_p'] = None
        return out
    except Exception as e:
        print(f'Could not load {config}: {e}')
        return None


def resolve_or_setup_experiment_output(outdir, args, M, device, coarse_grained=False,
                                        extra_metadata=None, include_potentials_dir=False):
    """Check whether a matching experiment already exists before creating a
    new output folder — the counterpart to calling setup_experiment_output()
    unconditionally, which creates a fresh experiments/<config>/ folder
    (with a brand-new timestamp) on every call, even when the exact same
    parameters were already run and would just get loaded from disk a
    moment later inside run_experiment()'s own duplicate check.

    Returns (config, exp_dir, fig_dir, potentials_dir, logger, loaded).
    `loaded` is the previously-saved result dict if a match was found (no
    new folder was created — call run_experiment() only if you want to
    force a re-run instead). `loaded` is None for a genuinely new run (a
    fresh folder was created via setup_experiment_output; proceed to call
    run_experiment() with the returned config/exp_dir/etc.).

    exp_dir nests under experiments/<group_name>/<config>/ in both branches
    (reused and freshly-created) — see build_group_name/setup_experiment_output.
    """
    group_name = build_group_name(args, M, coarse_grained)
    config_prefix = build_config_name(args, M, coarse_grained, include_timestamp=False)
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

    config = build_config_name(args, M, coarse_grained, include_timestamp=True)
    exp_dir, fig_dir, potentials_dir, logger = setup_experiment_output(
        outdir, config, args, M, extra_metadata=extra_metadata,
        include_potentials_dir=include_potentials_dir,
    )
    return config, exp_dir, fig_dir, potentials_dir, logger, None


def run_experiment(args, M, config, x1, filters, t, logger, outdir, device,
                   filters_Q=None, filters_Phi=None, normalize_potentials=False,
                   potentials_save_dir=None,
                   ):
    if not args.force_rerun:
        config_prefix = build_config_name(args, M=M, include_timestamp=False)  # see note below
        resolved = resolve_config_for_loading(outdir, config_prefix)
        if resolved is not None:
            loaded = try_load_experiment(outdir, resolved, device)
            if loaded is not None:
                logger.info('Loaded existing results for %s', resolved)
                return loaded

    logger.info('Running experiment: %s', config)
    logger.info('terms: %s', args.terms)

    potentials = get_1d_potentials(
        args.terms, args.J, filters, args.Q, filters_Q=filters_Q, filters_Phi=filters_Phi,
        scalar_param=None, parallel=False,
    )

    if normalize_potentials: 
        for pot in potentials.values():          # <-- add this
            if hasattr(pot, 'fit_micro'):        # <-- add this
                pot.fit_micro(x1)                # <-- add this
                print(f"Successfully normalized potential {pot} with attribute fit_micro")


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
        time_limit_min=getattr(args, 'time_limit_min', None),
    )
    logger.info('SDE integration finished in %.1f s', timer.time() - t0)

    # Pass the global outdir as 'root' so it targets the original global directories
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
