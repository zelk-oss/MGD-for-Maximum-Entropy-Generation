#!/usr/bin/env python
"""
run_SDE.py -- CLI launcher for the 2D cosmic-web (Quijote) MGD experiment.

Science and settings are those of notebooks/mgd_sampling/mgd_2d_example.ipynb, unchanged:

    data      Quijote density maps, log(x - min + 0.1), db4 low-pass down-sampled by --fact (2: 256 -> 128),
              first --n1 maps, standardised (codes.utils.normalize)
    model     2D Morlet bank J=5, Q=1, L=4; terms Scattering_{Second,First,Third_Order_Real}
              (arm "base", 303 potentials) and the same + Scattering_Fourth_Order_Real (arm "scat4")
    SDE       Cos interpolant, sigma = 1, uniform grid of nt steps, no regularisation (0,0,0),
              (scalar ridge --regularization 0.0 in this SDE version), plain predictor-corrector
              Solver() (not forward_regularised), batch_size 20,
              one walker and one interpolant per data map

The seed ONLY changes the initial white noise (x_0 of the interpolants and the starting walkers x_k): the
data are always the same first --n1 maps, exactly as in the other experiments (jets, turbulence, S&P).

Output layout (as in the other experiments):
    <outdir>/saved_results/{samples,lagrange_multipliers,entropy_bounds,sampling_times,aux_moments}/<config>.pt
    <outdir>/experiments/<group>/<config>/{config.json, logs/run.log, figures/*.png, summary.json}
A finished run (samples file present) is reloaded, not recomputed, unless --force_rerun.

Local use is for SMOKE TESTS ONLY (--smoke_data builds a smoothed Gaussian field, no Quijote file needed;
tiny n1 / nt). Real runs go through launch/*.sh on Jean Zay.
"""
import argparse
import hashlib
import json
import sys
import time as timer
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── sys.path setup (same mechanism as the other run_SDE.py) ───────────────────
root = Path(__file__).resolve().parent
project_root = root.parent

def _extend_syspath(root: Path, project_root: Path):
    for p in (root, project_root, project_root / 'codes', project_root / 'data'):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

_extend_syspath(root, project_root)

from codes.sde_routines import *              # noqa: E402
from codes.utils import *                     # noqa: E402
from codes.utils_experiment import setup_logging, logged_run   # noqa: E402
from codes.check_moments import *             # noqa: E402
from codes.utils_entropy import compute_gaussian_entropy        # noqa: E402
from codes.filters_bank import return_Filters  # noqa: E402
from codes.potentials_builder import get_2d_potentials          # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# terms of the two arms (the notebook's list, and the same with order 4)
ARM_TERMS = {
    'base':  ['Scattering_Second_Order', 'Scattering_First_Order', 'Scattering_Third_Order_Real'],
    'scat4': ['Scattering_Second_Order', 'Scattering_First_Order', 'Scattering_Third_Order_Real',
              'Scattering_Fourth_Order_Real'],
}


# ── data ──────────────────────────────────────────────────────────────────────
def load_quijote_2d(path, fact=2, n_max=1000):
    """data/data_loader.load_quijote with an explicit file path (same numerics): shift to min 0,
    log(x + 0.1), then keep the db4 'periodization' approximation at level log2(fact)."""
    import pywt
    if fact < 1 or (fact & (fact - 1)) != 0:
        raise ValueError(f'--fact must be a power of two, got {fact}')
    data = np.load(path)[:n_max]
    data = data - data.min()
    data = np.log(data + 1e-1)
    if fact == 1:
        return torch.from_numpy(data)[:, None]
    n, h, w = data.shape
    out = torch.zeros((n, 1, h // fact, w // fact))
    level = int(np.log2(fact))
    for i in range(n):
        out[i] = torch.from_numpy(pywt.wavedec2(data[i], 'db4', mode='periodization', level=level)[0])
    return out


def smoke_data(n, size, seed=0):
    """SMOKE TEST ONLY: smoothed Gaussian noise maps (n, 1, size, size). Not Quijote."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    x = np.stack([gaussian_filter(rng.standard_normal((size, size)), 2.0, mode='wrap') for _ in range(n)])
    return torch.from_numpy(x).float()[:, None]


# ── names ─────────────────────────────────────────────────────────────────────
def build_names(args, M):
    terms_hash = hashlib.md5('|'.join(sorted(args.terms)).encode()).hexdigest()[:8]
    head = (f'cosmicweb_quijote_M{M}_J{args.J}_Q{args.Q}_L{args.L}_sigma{args.sigma:g}'
            f'_nt{args.nt}_n1_{args.n1}')
    tail = f'_terms{terms_hash}' + (f'_{args.label}' if args.label else '')
    return f'{head}{tail}', f'{head}_seed_{args.seed}{tail}'      # group (all seeds), config


# ── args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='2D cosmic-web (Quijote) MGD experiment')
    p.add_argument('--seed', type=int, default=900, help='seed of the initial white noise only')
    p.add_argument('--arm', choices=list(ARM_TERMS), default='base',
                   help='base = notebook terms; scat4 = + Scattering_Fourth_Order_Real')
    p.add_argument('--terms', nargs='+', default=None, help='override the terms of --arm')
    p.add_argument('--label', type=str, default=None, help='suffix of the config name (default: the arm)')
    # data
    p.add_argument('--data_path', type=str, default=None, help='Quijote_Fidu_15000_256.npy')
    p.add_argument('--fact', type=int, default=2, help='down-sampling factor (2: 256 -> 128)')
    p.add_argument('--n1', type=int, default=100, help='number of data maps (the first n1)')
    p.add_argument('--smoke_data', action='store_true', help='SMOKE TEST: smoothed Gaussian maps instead of Quijote')
    p.add_argument('--smoke_size', type=int, default=32)
    # model
    p.add_argument('--J', type=int, default=5)
    p.add_argument('--Q', type=int, default=1)
    p.add_argument('--L', type=int, default=4)
    # SDE
    p.add_argument('--interpolant', type=str, default='Cos')
    p.add_argument('--nt', type=int, default=1000)
    p.add_argument('--sigma', type=float, default=1.0)
    p.add_argument('--batch_size', type=int, default=20)
    p.add_argument('--regularization', type=float, default=0.0,
                   help='ridge added to the per-step Gram. 0.0 = the notebook\'s regularization=(0,0,0) '
                        '(the older SDE took a tuple; this one a scalar)')
    p.add_argument('--time_limit_min', type=float, default=None,
                   help='abort cleanly if the SDE loop is projected past this (set by the launcher)')
    # io
    p.add_argument('--outdir', type=str, default=None, help='default: this folder')
    p.add_argument('--force_rerun', action='store_true')
    p.add_argument('--moment_threshold', type=float, default=1e-6)
    p.add_argument('--timestamp', type=str, default=None, help='accepted for launcher symmetry; unused')
    args = p.parse_args()
    if args.terms is None:
        args.terms = list(ARM_TERMS[args.arm])
    if args.label is None:
        args.label = args.arm
    if not args.smoke_data and not args.data_path:
        p.error('--data_path is required (or --smoke_data for a smoke test)')
    return args


# ── diagnostics ───────────────────────────────────────────────────────────────
def radial_power_spectrum(x):
    """Isotropic power spectrum of maps (B,1,M,M): mean |FFT|^2 per integer |k| shell."""
    B, _, M, N = x.shape
    P = (torch.fft.fft2(x[:, 0].double()).abs() ** 2).mean(0) / (M * N)
    ky = torch.fft.fftfreq(M, 1.0 / M).double()
    kx = torch.fft.fftfreq(N, 1.0 / N).double()
    k = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2).round().long()
    kmax = min(M, N) // 2
    out = torch.zeros(kmax)
    for r in range(1, kmax + 1):
        out[r - 1] = P[k == r].mean()
    return torch.arange(1, kmax + 1), out


def save_diagnostics(x1, res, args, config, fig_dir, logger):
    x1c, xtc = x1.cpu(), res['xt'].cpu()
    vmin, vmax = np.percentile(x1c.numpy(), [0.5, 99.5])

    # 1. data vs samples (same colour scale)
    n = 4
    fig, axs = plt.subplots(2, n, figsize=(3.2 * n, 6.6))
    for i in range(n):
        axs[0, i].imshow(x1c[i, 0], vmin=vmin, vmax=vmax); axs[0, i].axis('off')
        axs[1, i].imshow(xtc[i, 0], vmin=vmin, vmax=vmax); axs[1, i].axis('off')
    axs[0, 0].set_title('data', loc='left'); axs[1, 0].set_title('MGD samples', loc='left')
    fig.suptitle(config, fontsize=8); fig.tight_layout()
    fig.savefig(fig_dir / 'samples_vs_data.png', dpi=130); plt.close(fig)

    # 2. pixel PDF and isotropic power spectrum, data vs samples
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(min(x1c.min(), xtc.min()).item(), max(x1c.max(), xtc.max()).item(), 120)
    axs[0].hist(x1c.flatten().numpy(), bins=bins, density=True, histtype='step', color='k', label='data')
    axs[0].hist(xtc.flatten().numpy(), bins=bins, density=True, histtype='step', color='r', label='MGD samples')
    axs[0].set_yscale('log'); axs[0].set_xlabel('pixel value (standardised)'); axs[0].set_ylabel('PDF')
    axs[0].legend(); axs[0].set_title('pixel distribution')
    k, Pd = radial_power_spectrum(x1c); _, Ps = radial_power_spectrum(xtc)
    axs[1].loglog(k, Pd, 'k-', label='data'); axs[1].loglog(k, Ps, 'r--', label='MGD samples')
    axs[1].set_xlabel('|k| (pixels$^{-1}$ x M)'); axs[1].set_ylabel('P(k)'); axs[1].legend()
    axs[1].set_title('isotropic power spectrum')
    fig.tight_layout(); fig.savefig(fig_dir / 'pdf_and_spectrum.png', dpi=130); plt.close(fig)

    # 3. moments along the dynamics
    try:
        # plot_moment_matching aligns row k with t[k+1]. SDE.forward() returns nt+2 rows: the initial moments,
        # nt steps, and the last step stored a second time. Keep the nt step rows (plot only, files untouched).
        ne = len(res['t'])
        plot_moment_matching(res['barphi_e'][1:ne], res['barphi_p'][1:ne], res['t'], args.moment_threshold,
                             save={'filename': fig_dir / 'moment_matching.png', 'title': config})
    except Exception as e:
        logger.warning('moment-matching plot failed (%s: %s)', type(e).__name__, e)

    # 4. entropy bound vs equivalent Gaussian (formula of the notebook)
    d = x1.shape[-2] * x1.shape[-1]
    nt = len(res['t']) - 1
    H_p_0 = (np.log(2 * np.pi) + 1) * d / 2
    H_bound = res['dH_t_bound'].cumsum(0).cpu() / nt + H_p_0
    H_gauss = compute_gaussian_entropy(x1, args.interpolant, res['t'])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(H_bound.numpy(), 'r', label='MGD entropy bound (lower bound)')
    ax.plot(np.asarray(H_gauss.cpu()), 'k--', label='Gaussian with data covariance')
    ax.set_xlabel('step'); ax.set_ylabel('entropy'); ax.legend(); ax.set_title(config, fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / 'entropy_bound.png', dpi=130); plt.close(fig)
    summary = {'H_bound_final': float(H_bound[-1]), 'H_gaussian_final': float(H_gauss[-1]),
               'n_potentials': int(res['barphi_e'].shape[1]), 'nt': nt}
    with open(fig_dir.parent / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info('Entropy bound %.2f vs Gaussian %.2f (final)', summary['H_bound_final'], summary['H_gaussian_final'])
    logger.info('Saved diagnostic figures to %s', fig_dir)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    outdir = Path(args.outdir) if args.outdir else root
    outdir.mkdir(parents=True, exist_ok=True)

    # data: identical for every seed; the seed is applied only after loading
    if args.smoke_data:
        Data = smoke_data(args.n1, args.smoke_size)
    else:
        Data = load_quijote_2d(args.data_path, fact=args.fact)
    if Data.shape[0] < args.n1:
        raise ValueError(f'only {Data.shape[0]} maps available, --n1 = {args.n1}')
    x1 = normalize(Data[:args.n1]).to(device)
    n1, channels, M, _ = x1.shape

    group, config = build_names(args, M)
    exp_dir = outdir / 'experiments' / group / config
    fig_dir = exp_dir / 'figures'
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(exp_dir / 'logs', config)
    cfg = dict(vars(args)); cfg.update(config_name=config, M=M, channels=channels, device=str(device))
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2, default=str)
    logger.info('Config: %s', config)
    logger.info('Arguments: %s', vars(args))

    base = outdir / 'saved_results'
    paths = {k: base / k / f'{config}.pt' for k in
             ('samples', 'lagrange_multipliers', 'entropy_bounds', 'sampling_times')}
    aux_path = base / 'aux_moments' / f'{config}_aux_moments.pt'

    with logged_run(logger):
        logger.info('x1: %s (n1 maps of %dx%d, standardised)', tuple(x1.shape), M, M)
        t = torch.linspace(0, 1, args.nt + 1)

        if all(p.exists() for p in paths.values()) and aux_path.exists() and not args.force_rerun:
            logger.info('Loaded existing results for %s (use --force_rerun to recompute)', config)
            res = {'xt': torch.load(paths['samples']), 'dH_t_bound': torch.load(paths['entropy_bounds']),
                   't': torch.load(paths['sampling_times'])}
            res.update({k: v for k, v in torch.load(aux_path).items()})
        else:
            torch.manual_seed(args.seed)      # ONLY the initial white noise depends on the seed
            np.random.seed(args.seed)
            filters = return_Filters(M, args.J, args.Q, L=args.L, device=device)
            potentials = get_2d_potentials(args.terms, args.J, args.L, filters, parallel=False)
            logger.info('terms: %s', args.terms)

            Solver = SDE(x1, n1, n1, t, args.sigma, potentials, args.batch_size, device=device,
                         regularization=args.regularization, interpolant=args.interpolant, x_0=None, x_k=None,
                         potentials_save_dir=exp_dir / 'fitted_potentials')
            logger.info('The model has %d potentials', Solver.num_potentials)
            t0 = timer.time()
            xt, barphi_e, barphi_p, eta_t, theta_t, dH_t_bound = Solver(time_limit_min=args.time_limit_min)
            logger.info('SDE integration finished in %.1f s', timer.time() - t0)

            make_result_dirs(base)
            (base / 'aux_moments').mkdir(parents=True, exist_ok=True)
            save_results(xt, theta_t, dH_t_bound, t, outdir, config)
            torch.save({'barphi_e': barphi_e.detach().cpu(), 'barphi_p': barphi_p.detach().cpu()}, aux_path)
            res = {'xt': xt.detach(), 'dH_t_bound': dH_t_bound, 't': t,
                   'barphi_e': barphi_e.detach().cpu(), 'barphi_p': barphi_p.detach().cpu()}

        # figures after the results are saved: a failing plot must not fail the run
        try:
            save_diagnostics(x1, res, args, config, fig_dir, logger)
        except Exception as e:
            logger.warning('Diagnostic figures failed (%s: %s); results are saved', type(e).__name__, e)


if __name__ == '__main__':
    main()
