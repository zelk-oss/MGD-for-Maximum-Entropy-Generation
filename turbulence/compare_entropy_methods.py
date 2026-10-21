"""Compare two ways of computing the entropy of a regularised-MGD model, run on
the 30-seed ``turbulence_final`` runs (terms 54d17f3b, M=256, J=8, Q=3,
nt=55000, n1=10000).

Convention: this codebase's exponential family is  p_theta(x) = Z^{-1} exp(+theta^T phi(x))
(see codes/utils_entropy.py's "Sign convention audit").

Method 1 -- MGD tool (lower bound, Prop. 4.3 / Eq. 30)

    H_bound = H(p_0) + integral_0^1 dH_t_bound dt         # dH_t = -theta_t^T dm_t/dt

implemented in codes/utils_entropy.py::entropy_bound(). H(p_0) = standard
Gaussian entropy in d = M dimensions, since x_0 ~ N(0, I_d).

Method 2 -- sample-based  -log p over the generated samples

    -log p(x) = log Z - theta^T phi(x)
    H_sample  = -mean_i log p(x_i)
              = log Z - theta^T m_particles                # m_particles = <phi(x)>_samples

    We don't know log Z; the pipeline only gives log_Z_bound (also a lower
    bound), so this script substitutes log_Z_bound for log Z and reports
    H_sample under that approximation.

Identity ( codes/utils_entropy.py::log_Z_bound ):

    log_Z_bound = H_bound + theta^T m_target                # m_target = barphi_e at t~1

    => H_sample = H_bound + theta^T (m_target - m_particles)

    i.e. the two methods agree *exactly* iff the generated particles'
    empirical moments match the solver's constraint targets. The gap is
    literally the moment-matching residual, projected onto theta_1.

theta at t=1 is Theta_reg[-1] (the regularised fit at the last coarse node).

Theta_reg source
----------------
turbulence_final was launched with --skip_reg_solve (see launch/turbulence_final.sh);
Theta_reg is computed offline by codes/resolve_theta_reg.py and written to

    saved_results/theta_reg_lamsweep/<config>/lam<LAM>_ridge<RIDGE>_moment_mask.pt

so lagrange_multipliers_regularised/ is empty for these runs. The notebooks use
LAM=0.0005 and RIDGE=0.0001; this script matches that by default.

Isolation + resilience
----------------------
Each seed is processed in a FRESH CHILD PROCESS (subprocess.run, spawn-equivalent),
so Python's allocator and PyTorch's caches cannot accumulate state across seeds
-- the OOM-on-seed-7 pattern of a single-process loop is impossible by
construction. The parent streams the child's stdout to its own terminal so you
see progress live, and the child writes its one-line result dict to a temp
JSON path that the parent reads once the child exits.

A `--save_csv` is written row-by-row (appended after each seed finishes), so a
partial run isn't lost if the whole script is killed.

Usage (from the repo root, on Jean Zay):

    # Full 30-seed sweep, GPU recommended (CPU is ~10x slower for scattering).
    python turbulence/compare_entropy_methods.py --device cuda

    # CPU with a 5-seed smoke test + 1000 subsampled particles (fast).
    python turbulence/compare_entropy_methods.py --device cpu \
        --max_seeds 5 --max_particles 1000

    # Different Theta_reg (lam, ridge, suffix) picks.
    python turbulence/compare_entropy_methods.py --lam 0.0005 --ridge 0.0001 --reg_suffix _moment_mask

Outputs:
    * per-seed line:  H_bound, H_sample, gap, moment-discrepancy
    * aggregate:      mean / std across seeds, for each column
    * optional CSV:   --save_csv path/to/out.csv  (append-safe)
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent          # turbulence/
ROOT = HERE.parent                               # repo root
for p in (ROOT, ROOT / 'codes', ROOT / 'data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils import load_results, load_t_reg                                      # noqa: E402
from utils_entropy import (                                                     # noqa: E402
    _phi, entropy_bound, standard_gaussian_entropy,
)
from sde_routines import *                                                      # noqa: E402,F401,F403 (brings get_1d_potentials)
from filters_bank import return_Filters                                         # noqa: E402


DEFAULT_GROUP = (
    'turbulencesynth_M256_J8_Q3_sigma3.5_nt55000_n1_10000_lam0.0_thomas_nsub1'
    '_moment_f64_dedupfilt1e-08_terms54d17f3b_turbulence_final'
)


def rebuild_potentials(cfg, device):
    """Instantiate the potentials dict for one run, then replace each entry
    with its fitted-parameter counterpart loaded from the exp_dir's
    fitted_potentials/ folder. Mirrors run_analyse / logp_decompose."""
    J, Q, M = cfg['J'], cfg['Q'], cfg['M']
    filters, filters_Phi = return_Filters(M, J, 1, device=device, include_phi=True)
    filters_Q = return_Filters(M, J, Q, device=device)
    pots = get_1d_potentials(                                                   # noqa: F405
        cfg['terms'], J, filters, Q, scalar_param=None, parallel=False,
        filters_Q=filters_Q, filters_Phi=filters_Phi,
        deduplicate_filters=bool(cfg.get('deduplicate_filters') or False),
        dedup_tol=cfg.get('dedup_tol') or 1e-12,
    )
    return pots


def load_fitted_potentials(pots, fitted_dir, device):
    for name, p in list(pots.items()):
        if hasattr(p, 'is_fitted') and (fitted_dir / f'{name}.pt').exists():
            pots[name] = type(p).load_fixed_parameters(
                fitted_dir / f'{name}.pt', p.filters, map_location=device,
            )
    return pots


def phi_mean_over_samples(xt, pots, chunk, device):
    """<phi(x)>_samples over the generated particle batch, chunked to keep
    peak memory bounded. Returns a 1D tensor of length r."""
    n = xt.shape[0]
    acc = None
    count = 0
    with torch.no_grad():
        for i in range(0, n, chunk):
            x = xt[i:i + chunk].to(device)
            phi = _phi(x, pots)                                                  # (b, r)
            phi = phi.detach().cpu().double()
            if acc is None:
                acc = phi.sum(0)
            else:
                acc = acc + phi.sum(0)
            count += phi.shape[0]
    return acc / count


def load_theta_reg_from_lamsweep(outdir, config_name, lam, ridge, suffix):
    """Load the offline-resolved Theta_reg + t_reg for this config.

    Expected path:
        saved_results/theta_reg_lamsweep/<config>/lam<LAM>_ridge<RIDGE><suffix>.pt
    which codes/resolve_theta_reg.py writes as a dict with
    ('Theta_reg', 't_reg', 'lam', 'ridge', ...).

    LAM / RIDGE are matched by string equality against resolve_theta_reg.py's
    format: f'lam{lam}_ridge{ridge}{suffix}.pt'. Python's repr of 0.0005 is
    '0.0005', of 1e-4 is '0.0001' -- both match what the launcher produces.
    """
    sweep_dir = outdir / 'saved_results' / 'theta_reg_lamsweep' / config_name
    if not sweep_dir.is_dir():
        return None, None, f'no lamsweep dir ({sweep_dir.name})'
    fname = f'lam{lam}_ridge{ridge}{suffix}.pt'
    path = sweep_dir / fname
    if not path.exists():
        # Fallback: glob tolerantly for small formatting drift (e.g. 5e-04 vs 0.0005).
        candidates = sorted(sweep_dir.glob(f'lam*{suffix}.pt'))
        return None, None, (
            f'missing {fname}; have {[p.name for p in candidates[:3]]}'
            + (f'... (+{len(candidates)-3} more)' if len(candidates) > 3 else '')
        )
    blob = torch.load(path, map_location='cpu', weights_only=False)
    return blob['Theta_reg'], blob.get('t_reg'), None


def one_run(exp_dir, outdir, device, chunk, lam, ridge, reg_suffix, max_particles=None):
    """Return a row (dict) for one seed, or None if the run can't be loaded.

    exp_dir       : experiments/<group>/<config>/
    outdir        : the turbulence/ root (where saved_results/ lives)
    max_particles : if given, subsample xt to its first N walkers before
                    computing <phi(x)>_samples. 1000 is plenty for a (r~272)-dim
                    moment vector; the across-seed std of the gap dominates
                    this residual anyway.
    """
    cfg_path = exp_dir / 'config.json'
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    cfg = cfg.get('args', cfg)
    config_name = exp_dir.name

    # ---- load xt / theta_t / dH_t_bound / t -----------------------------------
    try:
        xt, theta_t, dH_t_bound, t, Theta_reg_in_run = load_results(outdir, config_name)
    except FileNotFoundError as e:
        return {'seed': cfg.get('seed'), 'status': f'missing-pt: {e}'}

    # ---- Theta_reg: in-run first, else from offline lamsweep ------------------
    if Theta_reg_in_run is not None:
        Theta_reg = Theta_reg_in_run
        t_reg = load_t_reg(outdir, config_name)
        reg_src = 'in-run'
    else:
        Theta_reg, t_reg, err = load_theta_reg_from_lamsweep(
            outdir, config_name, lam, ridge, reg_suffix,
        )
        if Theta_reg is None:
            return {'seed': cfg.get('seed'), 'status': err}
        reg_src = f'lamsweep lam={lam} ridge={ridge}{reg_suffix}'

    # ---- barphi_e (constraint moments at t~1), optional -----------------------
    aux_path = outdir / 'saved_results' / 'aux_moments' / f'{config_name}_aux_moments.pt'
    barphi_e = None
    if aux_path.exists():
        barphi_e = torch.load(aux_path, map_location='cpu').get('barphi_e')

    # ---- rebuild the fitted potentials ----------------------------------------
    pots = rebuild_potentials(cfg, device)
    pots = load_fitted_potentials(pots, exp_dir / 'fitted_potentials', device)

    # ---- Method 1: MGD tool (entropy lower bound) -----------------------------
    # Call entropy_bound() directly rather than log_Z_bound(), because log_Z_bound()
    # internally does a FULL-BATCH _phi(xt) (no chunking) which OOM-kills on
    # n1=10000, M=256, r~270 turbulence runs. We rebuild its outputs manually
    # from m_target (= barphi_e[-1], small) and m_particles (chunked below).
    results = {config_name: {
        'xt': xt, 'theta_t': theta_t, 'dH_t_bound': dH_t_bound, 't': t,
        'Theta_reg': Theta_reg, 't_reg': t_reg, 'barphi_e': barphi_e,
    }}
    eb = entropy_bound(results, config_name, pots, check_p0=False)
    H_bound = eb['H_bound']                                                     # Prop. 4.3 integral + H(p_0)

    theta_1 = Theta_reg[-1].detach().cpu().double()                             # (r,)

    # m_target: last fine-grid barphi_e, matching log_Z_bound's default indexing
    if barphi_e is not None:
        m_target = barphi_e[-1].detach().cpu().double()
        theta_dot_mT = float(theta_1 @ m_target)
        log_Z_lb = H_bound + theta_dot_mT
        m1_source = 'target'
    else:
        m_target = None
        theta_dot_mT = float('nan')
        log_Z_lb = float('nan')
        m1_source = 'no barphi_e'

    # ---- Method 2: sample-based entropy  (-log p averaged over samples) -------
    xt_eval = xt if (max_particles is None or xt.shape[0] <= max_particles) else xt[:max_particles]
    m_particles = phi_mean_over_samples(xt_eval, pots, chunk, device)           # chunked over n_eval
    n_eval = int(xt_eval.shape[0])
    theta_dot_mP = float(theta_1 @ m_particles)

    if m_target is not None:
        H_sample = log_Z_lb - theta_dot_mP                                       # = H_bound + theta^T(m_target - m_particles)
        denom = float(m_target.norm())
        m1_disc = float((m_target - m_particles).norm() / denom) if denom > 0 else float('nan')
    else:
        # Without m_target we have no log_Z_bound; fall back to using m_particles
        # for both sides so the identity collapses to H_sample == H_bound.
        log_Z_lb = H_bound + theta_dot_mP
        H_sample = H_bound
        m1_disc = float('nan')

    d = int(np.prod(xt.shape[1:]))
    gap = H_bound - H_sample                                                    # == theta^T (m_particles - m_target)

    return {
        'seed': cfg.get('seed'),
        'd': d,
        'n_eval': n_eval,
        'H_p0': standard_gaussian_entropy(d),
        'H_bound (Method 1)': H_bound,
        'H_sample (Method 2)': H_sample,
        'gap (M1 - M2)': gap,
        'log_Z_bound': log_Z_lb,
        'theta.m_target': theta_dot_mT if m1_source == 'target' else float('nan'),
        'theta.m_particles': theta_dot_mP,
        'm1_source': m1_source,
        'm1_discrepancy': m1_disc,
        'reg_src': reg_src,
    }


def _run_worker(args):
    """Worker mode: process one exp_dir, write JSON result to args.worker_json, exit."""
    device = torch.device(args.device)
    outdir = Path(args.outdir)
    row = one_run(
        Path(args.worker_exp), outdir, device,
        chunk=args.chunk, lam=args.lam, ridge=args.ridge, reg_suffix=args.reg_suffix,
        max_particles=args.max_particles,
    )
    if row is None:
        row = {'seed': Path(args.worker_exp).name, 'status': 'no config.json'}
    # torch tensors / numpy scalars -> plain python for JSON
    serial = {}
    for k, v in row.items():
        if isinstance(v, torch.Tensor):
            serial[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            serial[k] = v.item()
        else:
            serial[k] = v
    Path(args.worker_json).write_text(json.dumps(serial))


def _spawn_child_for_seed(exp_dir, out_json, args, log_prefix):
    """Run this same script in --worker mode for one seed. Returns the parsed
    row dict (always a dict, with a 'status' key if the child failed).
    stdout/stderr are inherited so the user sees per-seed progress live."""
    cmd = [
        sys.executable, '-u', str(Path(__file__).resolve()),
        '--worker_exp', str(exp_dir),
        '--worker_json', str(out_json),
        '--outdir', str(args.outdir),
        '--device', args.device,
        '--chunk', str(args.chunk),
        '--lam', args.lam,
        '--ridge', args.ridge,
        '--reg_suffix', args.reg_suffix,
    ]
    if args.max_particles is not None:
        cmd += ['--max_particles', str(args.max_particles)]

    print(f'{log_prefix} spawning: {exp_dir.name}')
    try:
        proc = subprocess.run(cmd, check=False)
    except Exception as e:
        return {'seed': exp_dir.name, 'status': f'spawn-failed: {e!r}'}

    if not out_json.exists():
        return {'seed': exp_dir.name,
                 'status': f'no result file (child rc={proc.returncode})'}
    try:
        return json.loads(out_json.read_text())
    except Exception as e:
        return {'seed': exp_dir.name, 'status': f'bad result json: {e!r}'}


def _write_csv(path, rows):
    """Rewrite the CSV with the current rows list. We overwrite rather than
    appending so column headers stay consistent across rows that happen to
    carry different keys (status-only rows vs full-result rows)."""
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--group', default=DEFAULT_GROUP,
                    help='experiments/<group>/ to iterate over (default: turbulence_final 30-seed).')
    ap.add_argument('--outdir', default=str(HERE),
                    help='Project root containing saved_results/ (default: turbulence/).')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--chunk', type=int, default=250, help='Batch size for phi evaluation on samples.')
    ap.add_argument('--max_particles', type=int, default=None,
                    help='Subsample xt to its first N walkers before computing <phi(x)>_samples '
                          '(default: use all). 1000 is more than enough for H_sample precision.')
    ap.add_argument('--max_seeds', type=int, default=None,
                    help='Process only the first N seed subdirs (for quick smoke runs).')
    ap.add_argument('--lam', default='0.0005', help='LAM for the offline Theta_reg file (default: 0.0005).')
    ap.add_argument('--ridge', default='0.0001', help='RIDGE for the offline Theta_reg file (default: 0.0001).')
    ap.add_argument('--reg_suffix', default='_moment_mask',
                    help='Suffix after lam/ridge in resolve_theta_reg filenames (default: _moment_mask).')
    ap.add_argument('--save_csv', type=Path, default=None,
                    help='Optional path to write per-seed CSV (appended row-by-row so partial runs survive crashes).')
    # Worker-mode flags, only used by the spawned children -- not for user invocation.
    ap.add_argument('--worker_exp', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--worker_json', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    # ---------- worker-mode branch: process one seed, write JSON, exit --------
    if args.worker_exp is not None:
        _run_worker(args)
        return

    # ---------- orchestrator-mode branch --------------------------------------
    outdir = Path(args.outdir)
    group_dir = outdir / 'experiments' / args.group
    if not group_dir.exists():
        sys.exit(f'[compare_entropy_methods] group not found: {group_dir}')

    seed_dirs = sorted([p for p in group_dir.iterdir() if p.is_dir()])
    if args.max_seeds is not None:
        seed_dirs = seed_dirs[:args.max_seeds]

    print(f'[compare_entropy_methods] group {args.group}')
    print(f'  seed subdirs to process: {len(seed_dirs)}'
          + (f' (max_seeds={args.max_seeds})' if args.max_seeds else ''))
    print(f'  device: {args.device}'
          + (f' | max_particles: {args.max_particles}' if args.max_particles else ''))
    print(f'  Theta_reg source: in-run file if present, else '
          f'theta_reg_lamsweep/<config>/lam{args.lam}_ridge{args.ridge}{args.reg_suffix}.pt')
    print(f'  isolation: one subprocess per seed (fresh Python / PyTorch state each time)')

    if args.save_csv is not None:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with tempfile.TemporaryDirectory(prefix='entropy_cmp_') as tmp:
        tmp = Path(tmp)
        for d_i, exp_dir in enumerate(seed_dirs, 1):
            out_json = tmp / f'row_{d_i:03d}.json'
            prefix = f'[{d_i:2d}/{len(seed_dirs)}]'
            row = _spawn_child_for_seed(exp_dir, out_json, args, prefix)
            rows.append(row)
            if args.save_csv is not None:
                _write_csv(args.save_csv, rows)                         # rewrite after each seed

    df = pd.DataFrame(rows)
    print('\n==== per-seed results ' + '=' * 60)
    with pd.option_context('display.max_columns', None,
                            'display.width', 200,
                            'display.float_format', '{:.6g}'.format):
        print(df.to_string(index=False))

    numeric_cols = [c for c in ['H_bound (Method 1)', 'H_sample (Method 2)',
                                 'gap (M1 - M2)', 'log_Z_bound',
                                 'theta.m_target', 'theta.m_particles',
                                 'm1_discrepancy'] if c in df.columns]
    if numeric_cols:
        print('\n==== aggregate across seeds ' + '=' * 56)
        agg = df[numeric_cols].apply(pd.to_numeric, errors='coerce').agg(['mean', 'std', 'min', 'max']).T
        with pd.option_context('display.float_format', '{:.6g}'.format):
            print(agg.to_string())

    print(
        '\nIdentity being tested:'
        '\n    H_sample = log_Z_bound - theta^T m_particles'
        '\n             = H_bound + theta^T (m_target - m_particles)'
        '\n  so (M1 - M2) = theta^T (m_particles - m_target).'
        '\n  Perfect moment match -> M1 = M2.  Imperfect match -> the gap *is* the'
        '\n  constraint-residual energy, signed by theta_1.'
    )
    if args.save_csv is not None:
        print(f'\nPer-seed CSV written incrementally -> {args.save_csv}')


if __name__ == '__main__':
    main()
