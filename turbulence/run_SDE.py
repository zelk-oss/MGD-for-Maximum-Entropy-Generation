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
  shared saved_results/ directory. See build_config_name / main() below.

NOTE on figure saving (see save_all_open_figures / isolate_pyplot_calls):
  the previous version did
      hist_plot(x1, res['xt'])
      plt.suptitle(...)
      plt.savefig(single_file)
      plt.close('all')
  plt.suptitle/plt.savefig only ever touch the *current* figure
  (plt.gcf()). If hist_plot() opens a new figure per channel/scale inside
  its own loop, only the last one was ever saved — the rest were silently
  dropped by plt.close('all'). If instead it never opens a new figure at
  all and just calls plt.hist() repeatedly, everything piles onto a single
  axes ("superimposed on one canvas"). Both failure modes are fixed below:
  see the docstring on isolate_pyplot_calls() for the caveat on when the
  monkeypatch trick can and can't help.
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
# Script lives in .../conditional_mgd/turbulence/; 'codes' and 'data' live in
# the parent folder .../conditional_mgd/, not alongside this script.
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
from codes.check_moments import *     # noqa: E402
from codes.ortho_wavelet.ReadyToUseWavelets import *
from data.data_loader import *        # noqa: E402
from data_loader import *             # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── argument parsing ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='MGD/SDE turbulence synthesis experiment launcher'
    )

    # Data
    p.add_argument('--n1', type=int, default=3000,
                    help='Dataset size: number of samples kept from the '
                         'preprocessed data, i.e. Data[:n1] (replaces the '
                         'hardcoded 3000 slice)')
    p.add_argument('--subseries_len', type=int, default=1024,
                    help='Subseries length passed to split_periodize_reshape '
                         '(previously hardcoded to 1024)')

    # Coarse-graining
    p.add_argument('--coarse_grain', action='store_true',
                    help='If set, and the signal length after split/reshape '
                         'is > 256, coarse-grain the raw signal via `scales` '
                         'steps of wavelet decomposition before building x1. '
                         'Because x1 is what the SDE trains and generates '
                         'from, xt automatically comes out at the same, '
                         'coarser resolution -- no separate post-hoc '
                         'coarse-graining of xt is needed.')
    p.add_argument('--scales', type=int, default=2,
                    help='Number of wavelet decomposition steps '
                         '(Db, m=3, detail/high-pass branch) removed from '
                         'the raw signal. Replaces the old hardcoded '
                         '`for j in range(2): Data = W.decompose(Data)[1]`.')

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
    p.add_argument('--outdir', type=str, default='saved_results',
                    help='Base directory. Each run gets its own subfolder at '
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


def build_config_name(args, M, coarse_grained):
    terms_hash = hashlib.md5('|'.join(sorted(args.terms)).encode()).hexdigest()[:8]
    parts = [
        'turbsynth',
        f'M{M}',
        f'J{args.J}',
        f'Q{args.Q}',
        f'sigma{args.sigma}',
        f'nt{args.nt}',
        f'n1_{args.n1}',
        f'lam{args.lam}',
        f'terms{terms_hash}',
    ]
    if coarse_grained:
        parts.append(f'cg{args.scales}')
    if args.label:
        parts.append(args.label)
    return '_'.join(parts)


# ── figure-saving helpers ─────────────────────────────────────────────────────
@contextlib.contextmanager
def isolate_pyplot_calls(*fn_names):
    """
    Temporarily patch selected matplotlib.pyplot plotting primitives
    (e.g. 'hist') so that every call opens its own fresh figure first.
    Restores the originals on exit.

    Why this can work at all without touching hist_plot's source: Python
    modules are singletons, so if codes/utils.py did
    `import matplotlib.pyplot as plt` and calls `plt.hist(...)`, that `plt`
    is the SAME module object we hold here. Patching the attribute on the
    module affects every caller that looks it up afterwards, including
    inside hist_plot.

    CAVEAT (please read before assuming this "fixes" everything): this only
    helps if the function actually calls the *module-level* `plt.hist` /
    `plt.plot` for each trace. If it instead does something like
    `fig, axes = plt.subplots(n, 1)` once and then calls `axes[i].hist(...)`
    on its own Axes objects, this patch is a complete no-op — plt.hist is
    never invoked, so there's nothing to intercept, and you'd need to either
    edit hist_plot directly or share its source with me so I can patch it
    precisely instead of guessing.
    """
    originals = {name: getattr(plt, name) for name in fn_names}

    def make_wrapper(name, orig):
        def wrapper(*args, **kwargs):
            plt.figure()
            return orig(*args, **kwargs)
        return wrapper

    for name in fn_names:
        setattr(plt, name, make_wrapper(name, originals[name]))
    try:
        yield
    finally:
        for name, orig in originals.items():
            setattr(plt, name, orig)


def save_all_open_figures(fig_dir, tag, logger, split_panels=True):
    """
    Save every currently-open matplotlib figure to its own file under
    fig_dir, named '<tag>_fig<N>.png'. If split_panels is True and a figure
    has more than one Axes (e.g. a subplot grid), also save each individual
    Axes cropped to its own bounding box as '<tag>_fig<N>_panel<M>.png', so a
    crowded multi-panel figure is still available as separate, readable
    per-panel images.

    This does NOT and cannot separate traces that share a single Axes (real
    overplotting within one panel) — that requires either the monkeypatch in
    isolate_pyplot_calls (if the culprit is a bare plt.<fn> call) or an edit
    to the plotting function itself.

    Does not close figures — call plt.close('all') after, once you're done
    saving from this batch.
    """
    saved = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        fname = fig_dir / f'{tag}_fig{num}.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        saved.append(fname)

        if split_panels and len(fig.axes) > 1:
            for j, ax in enumerate(fig.axes):
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                sub_fname = fig_dir / f'{tag}_fig{num}_panel{j}.png'
                fig.savefig(sub_fname, dpi=150, bbox_inches=extent.expanded(1.2, 1.3))
                saved.append(sub_fname)

    logger.info('Saved %d figure/panel file(s) for %s', len(saved), tag)
    return saved


# ── experiment run / reload ──────────────────────────────────────────────────
def try_load_experiment(outdir, config):
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
        aux_path = outdir / f'{config}_aux_moments.pt'
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


def run_experiment(args, config, x1, filters, filters_Phi, filters_Q, t, logger, outdir):
    if not args.force_rerun:
        loaded = try_load_experiment(outdir, config)
        if loaded is not None:
            logger.info('Loaded existing results for %s', config)
            return loaded

    logger.info('Running experiment: %s', config)
    logger.info('terms: %s', args.terms)

    potentials = get_1d_potentials(
        args.terms, args.J, filters, args.Q, filters_Q, filters_Phi,
        scalar_param=None, parallel=False,
    )

    batch_size = args.batch_size or x1.shape[0]
    nb_workers = x1.shape[0]
    nb_interpolants = x1.shape[0]

    t0 = timer.time()
    Solver = SDE(
        x1, nb_workers, nb_interpolants, t, args.sigma, potentials, batch_size,
        device=device, regularization=args.regularization, interpolant=args.interpolant,
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
            outdir / f'{config}_aux_moments.pt',
        )

    return {
        'xt': xt, 'theta_t': theta_t, 'Theta_reg': Theta_reg, 't': t,
        'dH_t_bound': dH_t_bound, 'barphi_e': barphi_e, 'barphi_p': barphi_p,
        'loaded': False,
    }


# ── diagnostics ───────────────────────────────────────────────────────────────
def save_diagnostics(x1, res, t, args, config, fig_dir, logger):
    threshold = args.moment_threshold

    # 1. moment matching
    if res.get('barphi_e') is not None and res.get('barphi_p') is not None:
        plot_moment_matching(res['barphi_e'], res['barphi_p'], res['t'], threshold, save=True)
    else:
        logger.warning('No aux moments available (loaded run without saved aux file?); '
                        'skipping moment-matching plot.')

    # 2. trajectories: true vs synthesized
    n_groups = max(1, min(args.n_traj_groups, x1.shape[0] // 5))
    for i in range(n_groups):
        Compare_time_series_row(x1[i * 5:i * 5 + 5], res['xt'][i * 5:i * 5 + 5], 5, save=True)

    # 3. wavelet histogram collection
    # This is the section that was coming out "superimposed on one canvas".
    # We force every plt.hist() call inside hist_plot to open its own new
    # figure (see isolate_pyplot_calls docstring for the caveat on when this
    # can't help), then save every figure that resulted, individually.
    hist_plot(x1, res['xt'], save=True)

    # 4. power spectrum
    spec_plot(x1, res['xt'], save=True)

    # 5. structure functions
    structure_plot(x1, res['xt'], save=True)

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

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    W = DefineWavelet('Db', m=3, device=device)
    Data = load_turbulence_1d()

    Data = split_periodize_reshape(Data, args.subseries_len)
    B, C, L = Data.shape
    Data = Data.view(B * 2, C, L // 2)

    pre_len = Data.shape[-1]
    coarse_grained = False
    if args.coarse_grain:
        if pre_len > 256:
            for _ in range(args.scales):
                Data = W.decompose(Data)[1]
            coarse_grained = True
        else:
            print(f'--coarse_grain set but signal length {pre_len} <= 256; skipping.')

    x1 = normalize(Data[:args.n1]).to(device)
    B, channels, M = x1.shape

    config = build_config_name(args, M, coarse_grained)

    # ---- per-experiment folder: outdir/experiments/<config>/ ----
    exp_dir = outdir / 'experiments' / config
    fig_dir = exp_dir / 'figures'
    log_dir = exp_dir / 'logs'
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir, config)
    logger.info('Config: %s', config)
    logger.info('Experiment folder: %s', exp_dir)
    logger.info('Arguments: %s', vars(args))
    logger.info('Data shape after split/reshape (pre coarse-grain): (%d, %d, %d)', B, C, pre_len)
    logger.info('x1 final shape: %s (coarse_grained=%s)', tuple(x1.shape), coarse_grained)

    # Dump the full config (every user-chosen parameter + a few derived ones)
    # so the run can be reloaded later without guessing what was used.
    config_dict = dict(vars(args))
    config_dict.update({
        'config_name': config,
        'M': M,
        'B': B,
        'channels': channels,
        'coarse_grained': coarse_grained,
        'pre_coarse_grain_length': pre_len,
    })
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

    filters, filters_Phi = return_Filters(M, args.J, 1, device=device, include_phi=True)
    filters_Q = return_Filters(M, args.J, args.Q, device=device)

    # save dataset wavelet histograms 
    wt = torch.fft.ifft(torch.fft.fft(x1) * filters).real  # (B, J, T)
    n_wavelets = filters.shape[1]
    # ------------------------------------------------------------------
    # 1. Overview grid - Q=1 
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
    plt.suptitle(f"Wavelet coefficient histograms", fontsize=25)
    plt.tight_layout()
    plt.savefig("wavelet_histo_Q=1.png")

    wtQ3 = torch.fft.ifft(torch.fft.fft(x1) * filters_Q).real  # (B, J, T)
    n_waveletsQ3 = filters_Q.shape[1]
    # ------------------------------------------------------------------
    # 1. Overview grid - Q=3 
    # ------------------------------------------------------------------
    ncols = 5
    nrows = math.ceil(n_waveletsQ3 / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i in range(n_waveletsQ3):
        vals = wtQ3[:, i, :].detach().cpu().flatten().numpy()
        axes[i].hist(vals, bins=50, density=True, log=True)
        axes[i].set_title(f"ch={i}")
        axes[i].set_xlabel("Coefficient value")
        axes[i].set_ylabel("Density")
    for j in range(n_waveletsQ3, len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"Wavelet coefficient histograms", fontsize=25)
    plt.tight_layout()
    plt.savefig("wavelet_histo_Q=3.png")

    t = 1 - (1 - torch.linspace(0, 1, args.nt + 1)) ** args.schedule_exponent

    result = run_experiment(args, config, x1, filters, filters_Phi, filters_Q, t, logger, outdir)

    save_diagnostics(x1, result, t, args, config, fig_dir, logger)

    logger.info('Done.')


if __name__ == '__main__':
    main()