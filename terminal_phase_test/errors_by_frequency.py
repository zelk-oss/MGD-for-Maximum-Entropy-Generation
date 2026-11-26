"""Moment-matching errors at the end of the planned grid, per statistic, with the period of its wavelet channel and
its collinearity with the other statistics (2026-11-25, explore_errors_by_frequency_m1024.ipynb, steps 1, 2, 5).

Inputs per run (label, seed), all written by terminal_phase_test/launch_terminal_errors_1118.sh into --errors_dir:
  <config>_before_after.csv  (terminal_phase_errors.py)  target / generated before the tail (row k0) and after it
  <config>.csv               (moment_error_variance.py --dump)  sd of each statistic on the data and on the final samples
Outputs (small) in --out:
  stats_<config>.csv  one row per statistic: family, period_finest [original samples], rel / z before and after the
                      tail, rel_se_target (sampling error of the target / |target|), one_minus_R2
  gram_summary.csv    one row per run: r, condition number and small eigenvalues of the correlation matrix

Definitions:
  rel = (generated - target) / |target|;  z = (generated - target) / SE, SE^2 = sd_model^2 / n_model + sd_data^2 / n_data
      (sd from the final samples, used for both columns: the walkers at the end of the planned grid were not saved);
  period of a coefficient = 1 / centre frequency of its filter (argmax of |filter^|), x subseries_len / target_len;
      Scat3 / Scat4 / Scalar: the FINEST filter the coefficient uses (user choice A); low-pass = inf;
  Gram = covariance of the statistics over the walkers at the end of the planned grid (system Sigma at that node),
      Jacobi-scaled to a correlation matrix C; one_minus_R2_i = 1 / (C^-1)_ii (fraction of statistic i NOT explained
      linearly by the others; ~0 = nearly redundant), C^-1 from the eigendecomposition with eigenvalues floored at
      1e-12 x the largest.

Usage (repo root, Jean Zay): python terminal_phase_test/errors_by_frequency.py --labels L1 L2 --seeds 900 901 \
    --errors_dir terminal_phase_test/errors_m1024 --out terminal_phase_test/errors_by_frequency_m1024
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / 'turbulence', ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_comparison as rc                                         # noqa: E402
from logp_family_contrib import load_potentials                     # noqa: E402
from filters_bank import return_Filters                             # noqa: E402

TURB = ROOT / 'turbulence'
RES = TURB / 'saved_results'

ap = argparse.ArgumentParser()
ap.add_argument('--labels', nargs='+', required=True)
ap.add_argument('--seeds', type=int, nargs='+', default=[900, 901, 902])
ap.add_argument('--errors_dir', type=Path, required=True)
ap.add_argument('--reg_root', type=Path,
                default=Path(os.environ.get('SCRATCH', '/nonexistent')) / 'MGD-for-Maximum-Entropy-Generation/turbulence/reg_system')
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)


def periods_of_rows(filters, M, factor):
    f = torch.as_tensor(filters).reshape(-1, M)
    xi = np.abs(np.fft.fftfreq(M))[f.abs().argmax(-1).cpu().numpy()]
    return np.where(xi == 0, np.inf, 1.0 / np.maximum(xi, 1e-12)) * factor


def stat_periods(pots, cfg):
    """{name[i]: period of the finest filter of coefficient i}, in original samples."""
    M, J = int(cfg['M']), int(cfg['J'])
    factor = int(cfg['subseries_len']) / int(cfg['target_len'])
    f1, phi = return_Filters(M, J, 1, device='cpu', include_phi=True, wav_type=cfg.get('wav_type', 'morlet'))
    bank1 = periods_of_rows(torch.cat([torch.as_tensor(f1).reshape(-1, M), torch.as_tensor(phi).reshape(-1, M)]), M, factor)
    out = {}
    for name, pot in pots.items():
        if name.startswith('Scalar_'):
            per = periods_of_rows(pot.filters, M, factor)
            Jc = int(pot.J)
            chans = [divmod(int(fl), Jc)[1] for fl in pot.active_flat.cpu().tolist()]
            vals = [per[c] for c in chans]
        elif name == 'L_6':
            vals = list(periods_of_rows(pot.filters, M, factor))
        elif name.startswith('Scattering_Third_Order'):
            per = periods_of_rows(pot.filters, M, factor)
            idx = pot._indices().cpu().numpy()
            vals = [min(per[a], per[b]) for a, b in zip(idx[0], idx[1])]
        elif name.startswith('Scattering_Fourth_Order'):
            idx = pot.indices.cpu().numpy().astype(int)
            vals = [min(bank1[a], bank1[b]) for a, b in zip(idx[0], idx[1])]
        else:
            raise SystemExit(f'no period rule for {name}')
        out.update({f'{name}[{i}]': v for i, v in enumerate(vals)})
    return out


gram_rows = []
for label in args.labels:
    for seed in args.seeds:
        config = rc._find_config(TURB, label, seed)
        assert config is not None, f'no run {label} seed {seed}'
        exp = rc._find_exp_dir(TURB, config)
        cfg = json.loads((exp / 'config.json').read_text())
        pots = load_potentials(exp, cfg, 'cpu')
        per = stat_periods(pots, cfg)

        ba = pd.read_csv(args.errors_dir / f'{config}_before_after.csv', comment='#')
        dm = pd.read_csv(args.errors_dir / f'{config}.csv', comment='#')
        assert list(ba['name']) == list(dm['name']), f'{config}: statistic order differs between the two csv'
        assert set(ba['name']) == set(per), f'{config}: {len(per)} periods vs {len(ba)} statistics'
        se = np.sqrt(dm['sd_model'] ** 2 / dm['n_model'] + dm['sd_data'] ** 2 / dm['n_data'])
        S = pd.DataFrame({'name': ba['name'], 'family': [n.rsplit('[', 1)[0] for n in ba['name']],
                          'period_finest': [per[n] for n in ba['name']],
                          'rel_before': ba['rel_before'],
                          'z_before': (ba['generated_before'] - ba['target_before']) / se,
                          'rel_after': ba['rel_after'],
                          'z_after': (ba['generated_after'] - ba['target_after']) / se,
                          'rel_se_target': dm['sd_data'] / np.sqrt(dm['n_data']) / ba['target_after'].abs().clip(lower=1e-300)})

        # Gram at the end of the planned grid
        t_run = np.asarray(torch.load(RES / 'sampling_times' / f'{config}.pt', map_location='cpu', weights_only=False), float).ravel()
        term = torch.load(RES / 'aux_moments' / f'{config}_terminal.pt', map_location='cpu', weights_only=False)
        t_end = t_run[int(term['terminal_start'])]
        system = torch.load(args.reg_root / label / f'{config}.pt', map_location='cpu', weights_only=False, mmap=True)
        t_sys = np.asarray(system['t'], float)
        k = int(np.argmin(np.abs(t_sys - t_end)))
        assert abs(t_sys[k] - t_end) < 1e-12, f'{config}: no system node at the planned end {t_end}'
        Sig = system['Sigma'][k].double()
        Sig = (Sig + Sig.T) / 2
        d = torch.sqrt(torch.clamp(torch.diagonal(Sig), min=1e-300))
        C = Sig / (d[:, None] * d[None, :])
        ev, U = torch.linalg.eigh(C)
        evf = torch.clamp(ev, min=1e-12 * float(ev.max()))
        diag_inv = ((U ** 2) / evf[None, :]).sum(1)
        S['one_minus_R2'] = (1.0 / diag_inv).numpy()
        S.to_csv(args.out / f'stats_{config}.csv', index=False)
        evn = ev.numpy()
        gram_rows.append(dict(label=label, seed=seed, config=config, r=len(evn), t_planned_end=t_end,
                              cond=float(evn.max() / max(evn.min(), 1e-300)) if evn.min() > 0 else float('inf'),
                              eig_min=float(evn.min()), n_eig_below_1e_10=int((evn < 1e-10).sum()),
                              n_eig_below_1e_6=int((evn < 1e-6).sum()),
                              n_stats_rel_before_gt_1pct=int((S['rel_before'].abs() > 1e-2).sum())))
        print(f'{label} seed {seed}: r {len(evn)}, eig min {evn.min():.2e}, '
              f'{(S["rel_before"].abs() > 1e-2).sum()} stats > 1 % before the tail', flush=True)

pd.DataFrame(gram_rows).to_csv(args.out / 'gram_summary.csv', index=False)
print(f'-> {args.out}')
