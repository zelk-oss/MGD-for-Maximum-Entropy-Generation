"""Compare the two coarse-grained diagnostic runs (2026-10-19), one with L_2_lowpass
and one without, on identical setup otherwise:

    turbulence_final_coarse_full   = L_6, L_2_lowpass, Scalar_psi, Scalar_morlet
    turbulence_final_coarse_noL2   = L_6,              Scalar_psi, Scalar_morlet

Each arm has one seed (900) at target_len = 128, n1 = 3000, nt = 15000.

What this script answers
------------------------
1. Does removing L_2_lowpass break the physics (bimodal marginal, broken spectrum,
   flatness)?  --> look at the per-run block "run vs data"
2. Are the two runs sample-equivalent on everything?  --> look at the direct
   "full vs noL2" block and the plots

What it does
------------
- Reloads the reference data x1 EXACTLY the way run_SDE.py did (2 DWT passes with
  Db3, 512 -> 128).
- Computes compute_stats(x) on data, full, noL2 (one pass each, cached in memory).
- Prints:
    * physics_metrics of each run vs its own data (ratios, 1 = perfect)
    * direct full-vs-noL2 ratios on the same quantities
    * the ten largest absolute log-ratio per-band spectral / wavelet differences
- Saves three figures to turbulence/figures/compare_coarse_L2/:
    marginal.png       marginal PDFs on log-y
    spectrum.png       spectrum + ratio to data
    wavelet_bands.png  per-band wavelet energy and flatness ratios

Usage (from the repo root on Jean Zay):
    python turbulence/compare_coarse_L2.py                      # 2026-10-19 arms (no Scat4)
    python turbulence/compare_coarse_L2.py --with-scat4         # the companion arms with Scat4
    python turbulence/compare_coarse_L2.py --seed 900 --with-scat4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent         # turbulence/
ROOT = HERE.parent                              # repo root
for p in (ROOT, ROOT / 'codes', ROOT / 'data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils import normalize, split_periodize_reshape                  # noqa: E402
from ortho_wavelet.ReadyToUseWavelets import DefineWavelet            # noqa: E402
from data.data_loader import load_turbulence_1d                      # noqa: E402

from run_comparison import (                                          # noqa: E402
    load_config, compute_stats, physics_metrics, ks_distance, _logrms,
)


# ----- 1. locate the two configs by label --------------------------------------
def find_one(label, seed):
    """Pick the one saved samples file for a (label, seed), newest if several."""
    pat = f'*seed_{seed}*{label}*.pt'
    files = sorted((HERE / 'saved_results' / 'samples').glob(pat),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f'no saved_results/samples/{pat}')
    if len(files) > 1:
        print(f'[{label}] several matches, picking newest: {files[-1].name}')
    return files[-1].stem


def rebuild_data_x1(cfg, device='cpu'):
    """x1 exactly the way run_SDE.py built it (subseries + K DWT passes with Db3)."""
    W = DefineWavelet('Db', m=3, device=device)
    Data = load_turbulence_1d().to(device)
    Data = split_periodize_reshape(Data, cfg['subseries_len'])
    for _ in range(int(np.log2(cfg['subseries_len'] / cfg['target_len']))):
        Data = W.decompose(Data)[1]
    x1 = normalize(Data[:cfg['n1']])
    return x1.squeeze(1)                                   # (n1, T)


def main(seed, with_scat4):
    tag = '_scat4' if with_scat4 else ''
    cfg_full = find_one(f'turbulence_final_coarse_full{tag}', seed)
    cfg_noL2 = find_one(f'turbulence_final_coarse_noL2{tag}', seed)
    print(f'full : {cfg_full}')
    print(f'noL2 : {cfg_noL2}')

    full = load_config(cfg_full, label='full')
    noL2 = load_config(cfg_noL2, label='noL2')

    # one shared reference data, built from either config (they must agree)
    cfg = full['cfg']
    assert cfg['target_len'] == noL2['cfg']['target_len'] == 128
    assert cfg['n1'] == noL2['cfg']['n1']
    data_x1 = rebuild_data_x1(cfg)
    print(f'reference data: x1 shape {tuple(data_x1.shape)}')

    print('computing statistics ...')
    S_data = compute_stats(data_x1)
    S_full = compute_stats(full['xt'])
    S_noL2 = compute_stats(noL2['xt'])

    # ----- 2. physics metrics of each run against its own data ----------------
    print('\n=== each run vs data (ratios: 1 = perfect) ===')
    rows = []
    for name, S in (('full', S_full), ('noL2', S_noL2)):
        rows.append({'run': name, **physics_metrics(S, S_data)})
    df_vs_data = pd.DataFrame(rows).set_index('run')
    with pd.option_context('display.float_format', '{:.4g}'.format):
        print(df_vs_data.T)

    # ----- 3. direct full vs noL2 --------------------------------------------
    print('\n=== direct full vs noL2 (should be ~0 if L_2_lowpass is redundant) ===')
    direct = {
        'S2_tau1_ratio_full/noL2': S_full['S2'][0] / S_noL2['S2'][0],
        'flat4_tau1_ratio_full/noL2': S_full['flat4'][0] / S_noL2['flat4'][0],
        'flat6_tau1_ratio_full/noL2': S_full['flat6'][0] / S_noL2['flat6'][0],
        'skew_tau1_diff_full-noL2': S_full['skew'][0] - S_noL2['skew'][0],
        'spec_logrms_full_vs_noL2': _logrms(S_full['spectrum'], S_noL2['spectrum']),
        'spec_logrms_oct_full_vs_noL2': _logrms(S_full['spec_band'], S_noL2['spec_band']),
        'wav_E_logrms_full_vs_noL2': _logrms(S_full['wav_E'], S_noL2['wav_E']),
        'wav_flat_logrms_full_vs_noL2': _logrms(S_full['wav_flat'], S_noL2['wav_flat']),
        'marg_KS_full_vs_noL2': ks_distance(S_full['values'], S_noL2['values']),
    }
    for k, v in direct.items():
        print(f'  {k:40s} {v:+.4g}')

    # ----- 4. biggest log-ratio differences in low-frequency / coarse bands ---
    print('\n=== spectrum, octave-binned, full / noL2 (ordered by |log-ratio|) ===')
    r = np.log(S_full['spec_band'] / S_noL2['spec_band'])
    order = np.argsort(-np.abs(r))
    for i in order[:min(len(order), 10)]:
        f = S_full['freq_band'][i]
        print(f'  band f~{f:.4g}  log(full/noL2) = {r[i]:+.3f}')

    # ----- 5. figures ---------------------------------------------------------
    out = HERE / 'figures' / f'compare_coarse_L2{tag}'
    out.mkdir(parents=True, exist_ok=True)

    # Marginal PDFs on log-y: bimodality is here
    lo = min(np.percentile(S['values'], 0.01) for S in (S_data, S_full, S_noL2))
    hi = max(np.percentile(S['values'], 99.99) for S in (S_data, S_full, S_noL2))
    bins = np.linspace(lo, hi, 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    for S, name, c in ((S_data, 'data', 'k'), (S_full, 'full', 'tab:blue'),
                       (S_noL2, 'noL2', 'tab:orange')):
        h, _ = np.histogram(S['values'], bins=bins, density=True)
        xc = 0.5 * (bins[1:] + bins[:-1])
        ax.semilogy(xc, h, label=name, color=c)
    ax.set_title('Marginal PDF (coarse L_2_lowpass test)'); ax.set_xlabel('x'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'marginal.png', dpi=150); plt.close(fig)

    # Spectrum + ratio to data
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.loglog(S_data['freq'], S_data['spectrum'], 'k', lw=2, label='data')
    a1.loglog(S_full['freq'], S_full['spectrum'], 'tab:blue', label='full')
    a1.loglog(S_noL2['freq'], S_noL2['spectrum'], 'tab:orange', label='noL2')
    a1.set_xlabel('f'); a1.set_ylabel('S(f)'); a1.legend(); a1.set_title('spectrum')
    a2.semilogx(S_data['freq'], S_full['spectrum'] / S_data['spectrum'], 'tab:blue', label='full / data')
    a2.semilogx(S_data['freq'], S_noL2['spectrum'] / S_data['spectrum'], 'tab:orange', label='noL2 / data')
    a2.axhline(1, color='k', lw=0.5); a2.set_xlabel('f'); a2.set_title('ratio to data'); a2.legend()
    fig.tight_layout(); fig.savefig(out / 'spectrum.png', dpi=150); plt.close(fig)

    # Per-band wavelet moments
    bands_i = np.arange(len(S_data['wav_E']))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for S, name, c in ((S_data, 'data', 'k'), (S_full, 'full', 'tab:blue'),
                       (S_noL2, 'noL2', 'tab:orange')):
        a1.plot(bands_i, S['wav_E'] / S_data['wav_E'], label=name, color=c)
        a2.plot(bands_i, S['wav_flat'] / S_data['wav_flat'], label=name, color=c)
    for ax, title in ((a1, 'wavelet energy / data'), (a2, 'wavelet flatness / data')):
        ax.axhline(1, color='k', lw=0.5); ax.set_xlabel('band index (coarse->fine)')
        ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'wavelet_bands.png', dpi=150); plt.close(fig)

    (out / 'metrics_vs_data.csv').write_text(df_vs_data.to_csv())
    (out / 'direct_full_vs_noL2.json').write_text(
        json.dumps({k: float(v) for k, v in direct.items()}, indent=2))
    print(f'\nfigures and tables in: {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=900)
    ap.add_argument('--with-scat4', action='store_true',
                    help='look for the companion runs with Scattering_Fourth_Order_* appended '
                    '(label suffix _scat4).')
    a = ap.parse_args()
    main(a.seed, a.with_scat4)
