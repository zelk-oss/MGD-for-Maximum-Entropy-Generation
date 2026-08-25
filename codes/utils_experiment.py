"""
Moment-Guided Diffusion (MGD) - expeiriment saving and loading utilities.
"""

from pathlib import Path
import numpy as np
import torch
from typing import Dict, Any, Tuple
import sys 
import logging
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


def build_config_name(args, M, coarse_grained=False, include_timestamp=True):
    terms_hash = hashlib.md5('|'.join(sorted(args.terms)).encode()).hexdigest()[:8]

    re_number = getattr(args, 'Re_number', None)
    if re_number is not None:
        prefix = 'jetsynth'
    else:
        prefix = 'turbulencesynth'

    parts = [prefix]
    if re_number is not None: 
        parts.append(f'Re_number{re_number}')
    parts += [
        f'M{M}',
        f'J{args.J}',
        f'Q{args.Q}',
        f'sigma{args.sigma}',
        f'nt{args.nt}',
        f'n1_{args.n1}',
        f'lam{args.lam}',
        f'seed_{args.seed}',
        f'terms{terms_hash}',
    ]
    if args.label:
        parts.append(args.label)
    if include_timestamp and args.timestamp:
        parts.append(args.timestamp)
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


def run_experiment(args, M, config, x1, filters, t, logger, outdir, device, 
                   filters_Q=None, filters_Phi=None, 
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
