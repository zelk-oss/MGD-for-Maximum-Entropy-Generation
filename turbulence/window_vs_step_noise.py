"""Are the statistics MGD misses at t=1 exactly the windows narrower than one SDE step?

For every active statistic of Scalar_psi_gaussianK / Scalar_morlet_gaussianK of a run:
(channel, window), window edges in |z|, data fraction inside, and the ratio
    width / (noise one SDE step adds to z_j)
at 1-t = 1e-2, 1e-3 and at the last step (step noise in x is sqrt(2h)*sigma per component;
in z_j it is that times the std of the filter applied to unit white noise, measured
directly). Next to it: the relative error of the statistic at t=1 (aux_moments).
If the misses are the windows with the smallest ratio and the matched ones all sit above
some value, that value is the minimum width to impose in the fit.

Moment order in aux_moments = potentials order in codes/potentials_builder.py:
L_6, Scalar_psi_gaussianK, Scalar_morlet_gaussianK; inside a potential, active_flat
(flat = window * n_channels + channel).

Usage (repo root on Jean Zay):
    python turbulence/window_vs_step_noise.py --label turbulence_ggd_mass_L6_2b
    python turbulence/window_vs_step_noise.py --label turbulence_scalar_orthog_morletQ1Q3_L6_2b
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data', HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from filters_bank import return_Filters                              # noqa: E402
from potentials_builder import get_1d_potentials                     # noqa: E402
from check_fit_offline import build_data                             # noqa: E402

SCALAR = ('Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK')        # builder order
MISS = 1e-2


def newest(pattern, base):
    files = sorted(base.glob(pattern), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f'nothing matches {base}/{pattern}')
    return files[-1]


def main(label, seed, show):
    exp = newest(f'*/*_seed_{seed}_*{label}_2*', HERE / 'experiments')
    a = json.loads((exp / 'config.json').read_text())
    cfg = a.get('config_name', exp.name)
    M, J, Q = int(a.get('M', a.get('target_len', 128))), int(a['J']), int(a['Q'])
    sigma, terms = float(a['sigma']), list(a['terms'])
    print(f'run {cfg}\n  M={M} J={J} Q={Q} sigma={sigma} dedup={a.get("deduplicate_filters")} '
          f'tol={a.get("dedup_tol")} terms={terms}')

    aux = torch.load(HERE / 'saved_results' / 'aux_moments' / f'{cfg}_aux_moments.pt', map_location='cpu')
    e, p = aux['barphi_e'][-1].double(), aux['barphi_p'][-1].double()
    rel = ((p - e).abs() / e.abs().clamp_min(1e-300)).numpy()
    ratio_pe = (p / e).numpy()

    t = torch.load(newest(f'{cfg}*', HERE / 'saved_results' / 'sampling_times'), map_location='cpu')
    t = np.asarray(t, dtype=float).ravel()
    h = np.diff(t)
    gap = 1.0 - t[1:]
    h_at = {f'1-t={g:g}': float(h[np.argmin(np.abs(gap - g))]) for g in (1e-2, 1e-3)}
    h_at['end'] = float(h[-1])
    print('  step h: ' + '  '.join(f'{k} {v:.2e}' for k, v in h_at.items()))

    filters, filters_phi = return_Filters(M, J, 1, include_phi=True, wav_type=a.get('wav_type', 'morlet'))
    filters_Q = return_Filters(M, J, Q, wav_type=a.get('wav_type', 'morlet'))
    ggd = {k: v for k, v in dict(tail_quantile=a.get('ggd_tail_quantile'),
                                 tail_alpha_bounds=a.get('ggd_tail_alpha_bounds'),
                                 tail_mass=a.get('ggd_tail_mass', False)).items() if v}
    pots = get_1d_potentials(terms, J, filters, Q, filters_Q=filters_Q, filters_Phi=filters_phi,
                             deduplicate_filters=bool(a.get('deduplicate_filters', False)),
                             dedup_tol=float(a.get('dedup_tol', 1e-12)), ggd_kwargs=ggd)

    x1, _ = build_data(int(a.get('subseries_len', 512)), M, int(a['n1']), 1)
    white = torch.randn(4000, 1, M, dtype=torch.float64)

    loaded = {}
    for name in SCALAR:
        if name not in pots:
            continue
        pot = type(pots[name]).load_fixed_parameters(exp / 'fitted_potentials' / f'{name}.pt',
                                                     pots[name].filters, map_location='cpu')
        loaded[name] = pot
    n_scalar = sum(pt.num_coefficients for pt in loaded.values())
    offset = len(e) - n_scalar                     # L_6 (and any other term) comes first
    print(f'  {len(e)} statistics: {offset} before the Scalar blocks, '
          + ', '.join(f'{n} {pt.num_coefficients}' for n, pt in loaded.items()))

    rows = []
    for name, pot in loaded.items():
        f = pot.filters.double()
        with torch.no_grad():
            zd = torch.fft.ifft(f * torch.fft.fft(x1.double())).real.numpy()
            zw = torch.fft.ifft(f * torch.fft.fft(white)).real.numpy()
        nch, K = pot.J, pot.K
        unit = zw.std(axis=(0, 2))                 # z_j std for unit white noise in x
        for i, fl in enumerate(pot.active_flat.tolist()):
            j, k = fl % nch, fl // nch
            mass = k == K                          # tail_mass slot: same window as the outer one
            kk = K - 1 if mass else k
            c = pot.cuts[j].double().numpy()
            edges = np.concatenate([[0.0], c, [np.inf]])
            lo, hi = edges[kk], edges[kk + 1]
            ad = np.abs(zd[:, j]).ravel()
            frac = float(((ad >= lo) & (ad < hi)).mean())
            width = (hi if np.isfinite(hi) else np.quantile(ad, 0.9999)) - lo
            noise = {kk_: np.sqrt(2 * hv) * sigma * unit[j] for kk_, hv in h_at.items()}
            idx = offset + sum(pt.num_coefficients for pt in list(loaded.values())[:list(loaded).index(name)]) + i
            rows.append(dict(idx=idx, pot='psi' if 'psi' in name else 'morlet', ch=j,
                             win=('mass' if mass else str(k)), lo=lo, hi=hi, frac=frac,
                             med=float(np.median(ad)),
                             r={kk_: width / v for kk_, v in noise.items()},
                             rel=float(rel[idx]), pe=float(ratio_pe[idx])))

    rows.sort(key=lambda r: r['r']['end'])
    keys = list(h_at)
    print(f"\n{'idx':>4} {'pot':>6} {'ch':>3} {'win':>4} {'lo':>9} {'hi':>9} {'frac':>6} "
          + ' '.join(f'{"w/noise " + k:>16}' for k in keys) + f" {'rel err':>8} {'p/e':>6}")
    for r in rows[:show]:
        flag = '  <-- MISS' if r['rel'] > MISS else ''
        print(f"{r['idx']:>4} {r['pot']:>6} {r['ch']:>3} {r['win']:>4} {r['lo']:9.2e} {r['hi']:9.2e} "
              f"{r['frac']:6.3f} " + ' '.join(f"{r['r'][k]:16.2f}" for k in keys)
              + f" {r['rel']:8.1e} {r['pe']:6.3f}{flag}")

    miss = [r for r in rows if r['rel'] > MISS]
    ok = [r for r in rows if r['rel'] <= MISS]
    if miss and ok:
        print(f"\n{len(miss)} misses (rel > {MISS:g}) of {len(rows)} Scalar statistics.")
        for k in keys:
            mx = max(r['r'][k] for r in miss)
            n_ok_below = sum(r['r'][k] <= mx for r in ok)
            print(f"  {k:>9}: misses have width/noise <= {mx:.2f}; matched statistics at or below "
                  f"that value: {n_ok_below} of {len(ok)}")
        print('  misses not on the narrowest windows (rank by width/noise at end): '
              + ', '.join(f"{r['idx']}(rank {rows.index(r)})" for r in miss if rows.index(r) >= len(miss)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='turbulence_ggd_mass_L6_2b')
    ap.add_argument('--seed', type=int, default=900)
    ap.add_argument('--show', type=int, default=40, help='rows printed, narrowest windows first')
    a = ap.parse_args()
    main(a.label, a.seed, a.show)
