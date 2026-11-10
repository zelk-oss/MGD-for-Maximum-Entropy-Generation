#!/usr/bin/env python
"""run_SDE.py -- MGD/SDE on the daily S&P 500 log-returns: the whole series is ONE sample (d = 6062).

Thin entry point: the CLI (parse_args), the SDE/potential machinery and the diagnostics are those of
turbulence/run_SDE.py, loaded by path. Only the data block differs (snp_loader.load_snp_returns), plus
--n_particles (number of walkers/interpolants, read by codes/utils_experiment.run_experiment; the turbulence
parser does not know it, so it is taken out of argv here).

Data: x1 = standardized daily log-returns, shape (1, 1, d): no windows, no per-window anything; moments are
averages over u of the full series. Use --n1 1 (the launcher does); --subseries_len / --target_len are ignored
(M = d = 6062); they only have to be given because the shared launcher passes them.

Not supported (as in turbulence, but not wired here): --n1_spread, --gauss_spectrum, --potentials_from.
Config names keep the turbulence prefix ("turbulencesynth_..._np<N>_..."); runs are told apart by --label.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parent
project_root = root.parent
for p in (root, project_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# turbulence/run_SDE.py by path (its module-level code only sets up paths/imports; main() is not run)
_spec = importlib.util.spec_from_file_location('turb_run_SDE', project_root / 'turbulence' / 'run_SDE.py')
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

from snp_loader import load_snp_returns  # noqa: E402
from codes.gauss_spectrum import replace_with_gaussian  # noqa: E402

device = base.device


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--n_particles', type=int, default=None,
                     help='walkers = interpolants (MGD paper: 100). Default: as many as data samples (here 1).')
    pre.add_argument('--welch_nperseg', type=int, default=512,
                     help='with --gauss_spectrum: Welch segment length for the S&P spectrum (Hann, 50%% overlap). '
                          '0 = raw periodogram.')
    known, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = base.parse_args()
    args.n_particles = known.n_particles
    args.welch_nperseg = known.welch_nperseg
    return args


def save_diagnostics_snp(x1, res, t, args, config, fig_dir, logger):
    """The 'did the run finish fine' figures only (no turbulence-specific ones: structure functions, zeta4/zeta2,
    several-trajectory comparison). Each in its own try: one failing plot does not stop the others."""
    import matplotlib.pyplot as plt

    def attempt(name, fn):
        try:
            fn()
            logger.info('figure ok: %s', name)
        except Exception as e:  # noqa: BLE001
            logger.warning('figure %s skipped: %r', name, e)
        plt.close('all')

    attempt('wavelet_histo_Q1', lambda: base.plot_wavelet_coeff_histograms(
        x1, args.J, 1, config, fig_dir, logger,
        save={'filename': fig_dir / 'wavelet_histo_Q1.png', 'title': config}))
    if res.get('barphi_e') is not None and res.get('barphi_p') is not None:
        attempt('moment_matching', lambda: base.plot_moment_matching(
            res['barphi_e'], res['barphi_p'], res['t'], args.moment_threshold,
            save={'filename': fig_dir / 'moment_matching.png', 'title': config}))
    else:
        logger.warning('No aux moments available: skipping the moment-matching plot')
    attempt('histograms', lambda: base.hist_plot(
        x1, res['xt'], save={'filename': fig_dir / 'histograms.png', 'title': config}))
    attempt('spectrum', lambda: base.spec_plot(
        x1, res['xt'], save={'filename': fig_dir / 'spectrum.png', 'title': config}))

    def series():
        xs = res['xt'].detach().cpu()
        fig, ax = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
        ax[0].plot(x1[0, 0].detach().cpu().numpy()[:1500], lw=0.6); ax[0].set_ylabel('S&P (data)')
        for i in range(3):
            ax[i + 1].plot(xs[i, 0].numpy()[:1500], lw=0.6, color='C1'); ax[i + 1].set_ylabel(f'sample {i}')
        ax[-1].set_xlabel('u (trading days, first 1500)')
        fig.suptitle(config, fontsize=7)
        fig.savefig(fig_dir / 'series_data_vs_samples.png', dpi=130, bbox_inches='tight')
    attempt('series_data_vs_samples', series)


def main():
    args = parse_args()
    for flag in ('n1_spread', 'potentials_from'):
        if getattr(args, flag, None):
            raise NotImplementedError(f'--{flag} is not wired in snp_experiment/run_SDE.py')
    if args.n1 != 1:
        raise ValueError('S&P: the whole series is one sample, use --n1 1')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir) if args.outdir else root
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- data: one standardized series ----
    x1, raw, dates = load_snp_returns()
    x1 = x1.to(device)
    gauss_S = None
    if args.gauss_spectrum:     # x1 = ONE stationary Gaussian series with the Welch spectrum of the S&P (data seed 0)
        x1, gauss_S = replace_with_gaussian(x1, seed=0, nperseg=args.welch_nperseg or None)
    B, channels, M = x1.shape

    config, exp_dir, fig_dir, potentials_dir, logger, loaded = base.resolve_or_setup_experiment_output(
        outdir, args, M, device, coarse_grained=False,
        extra_metadata={
            'M': M, 'B': B, 'channels': channels, 'n_particles': args.n_particles,
            'snp_dates': [str(dates[0].date()), str(dates[-1].date())],
            'snp_raw_return_mean': float(raw.mean()), 'snp_raw_return_std': float(raw.std()),
        },
        include_potentials_dir=True,
    )
    logger.info('S&P daily log-returns: d=%d, %s -> %s; x1 %s, std %.4f; particles %s',
                M, dates[0].date(), dates[-1].date(), tuple(x1.shape), x1.std().item(), args.n_particles)
    with open(exp_dir / 'snp_dates.json', 'w') as f:
        json.dump([str(d.date()) for d in dates], f)
    if gauss_S is not None:
        torch.save({'S': gauss_S, 'data_seed': 0, 'welch_nperseg': args.welch_nperseg}, exp_dir / 'gauss_spectrum.pt')
        logger.info('gauss_spectrum: x1 replaced by a Gaussian with the S&P Welch spectrum (nperseg %s, var %.4f)',
                    args.welch_nperseg, x1.var().item())

    with base.logged_run(logger):
        filters, filters_Phi = base.return_Filters(M, args.J, 1, device=device, include_phi=True,
                                                   wav_type=args.wav_type)
        filters_Q = base.return_Filters(M, args.J, args.Q, device=device, wav_type=args.wav_type)

        t, adaptive = base.build_time_grid(args, logger)

        if loaded is not None:
            result = loaded
        else:
            dwt = base.PeriodicDWT(args.dwt_order, args.J, M, device=device) if args.dwt_order else None
            result = base.run_experiment(args, M, config, x1, filters,
                                         t, logger, outdir, device, dwt=dwt,
                                         filters_Q=filters_Q, filters_Phi=filters_Phi,
                                         potentials_save_dir=potentials_dir,
                                         adaptive=adaptive)

        # diagnostics only after the results are saved
        save_diagnostics_snp(x1, result, t, args, config, fig_dir, logger)


if __name__ == '__main__':
    main()
