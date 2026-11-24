"""Fast vs original potentials over a whole MGD run (2026-11-23): does the fast code accumulate error?

Two finished runs with the same settings and seed, one with turbulence/run_SDE.py, one with grad_speed/run_SDE_fast.py.
Criterion (agreed 11-23), judged statistically, not walker by walker (rounding differences reach ~1-2 SE of the moments
within ~100 steps for any code change, float64 included):
  1. moments at t = 1: z_n = (p_n - e_n) / SE_n per run (p = walker mean, e = target, SE_n = std over walkers of
     phi_n(x) / sqrt(n1), from the run's own final samples); same distribution of |z| for fast and original.
     Plus d_n = |p_fast - p_orig| / SE_n along the run (SE at t = 1 for every row) and the relative error per row.
  2. theta_t: ||theta_fast - theta_orig|| per step vs the sampling noise of theta at a few t. Noise = split-half
     difference ||theta(A) - theta(B)|| / sqrt(2) on n1/2 + n1/2 interpolant samples: two independent full-n1
     estimates differ by 1/sqrt(2) of two independent half-n1 estimates.
Only loads saved results (aux_moments, lagrange_multipliers, samples, fitted_potentials); the potentials are rebuilt
from the original run's fitted states, which must equal the fast run's (checked, printed).

Usage (repo root on JZ, GPU job; the aux_moments files are ~76 MB each at M = 128, too big to pull):
  python -u grad_speed/compare_fast_orig_run.py --M 128 --J 7 --subseries_len 256 --target_len 128 \
      --orig speedcheck_M128_nt27500 --fast pf_fast_M128 --out grad_speed/results_fast_vs_orig
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'codes', ROOT / 'grad_speed'):
    sys.path.insert(0, str(p))
from bench_signal_length import load_x1, build_sde, set_time, state_at       # noqa: E402

RES = ROOT / 'turbulence' / 'saved_results'
EXP = ROOT / 'turbulence' / 'experiments'


def find_cfg(label, M, seed):
    c = sorted(RES.glob(f'samples/turbulencesynth_M{M}_*_seed_{seed}_*_{label}_n15000_2*.pt'))
    if len(c) != 1:
        raise FileNotFoundError(f'{label}: {len(c)} sample files at M={M} seed={seed}: {[f.name for f in c]}')
    return c[0].stem


def fitted_dir(cfg):
    d = list(EXP.glob(f'*/{cfg}/fitted_potentials'))
    if len(d) != 1:
        raise FileNotFoundError(f'fitted_potentials of {cfg}: {d}')
    return d[0]


def same_fit(da, db):
    out = {}
    for f in sorted(da.glob('*.pt')):
        a, b = torch.load(f, map_location='cpu'), torch.load(db / f.name, map_location='cpu')
        same = a.keys() == b.keys() and all(
            (torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k]) for k in a)
        out[f.name] = bool(same)
    return out


def q(v):
    v = np.asarray(v, dtype=float)
    return {'median': float(np.median(v)), 'p90': float(np.quantile(v, 0.9)), 'max': float(v.max()),
            'n>2': int((v > 2).sum()), 'n>3': int((v > 3).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, required=True)
    ap.add_argument('--J', type=int, required=True)
    ap.add_argument('--Q', type=int, default=3)
    ap.add_argument('--subseries_len', type=int, required=True)
    ap.add_argument('--target_len', type=int, required=True)
    ap.add_argument('--n1', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=900)
    ap.add_argument('--orig', required=True, help='label of the original-code run')
    ap.add_argument('--fast', required=True, help='label of the fast-code run')
    ap.add_argument('--t_noise', type=float, nargs='+', default=[0.5, 0.9, 0.99, 0.999])
    ap.add_argument('--out', default=str(ROOT / 'grad_speed' / 'results_fast_vs_orig'))
    a = ap.parse_args()

    cfg = {k: find_cfg(getattr(a, k), a.M, a.seed) for k in ('orig', 'fast')}
    fd = {k: fitted_dir(c) for k, c in cfg.items()}
    res = {'args': vars(a), 'configs': cfg, 'same_fit': same_fit(fd['orig'], fd['fast'])}
    print('configs', json.dumps(cfg, indent=1), '\nsame fitted states:', res['same_fit'], flush=True)

    aux = {k: torch.load(RES / 'aux_moments' / f'{c}_aux_moments.pt', map_location='cpu') for k, c in cfg.items()}
    th = {k: torch.load(RES / 'lagrange_multipliers' / f'{c}.pt', map_location='cpu').double() for k, c in cfg.items()}
    xs = {k: torch.load(RES / 'samples' / f'{c}.pt', map_location='cuda') for k, c in cfg.items()}
    t = torch.load(RES / 'sampling_times' / f'{cfg["orig"]}.pt', map_location='cpu').double()

    x1 = load_x1(a.subseries_len, a.target_len, a.n1, 'cuda')
    S = build_sde(x1, a.M, a.J, a.Q, fd['orig'], x1.shape[0], 'cuda')

    # 1. moments
    with torch.no_grad():
        se = {k: (S.compute_moments(x).std(0).double() / np.sqrt(x.shape[0])).cpu() for k, x in xs.items()}
    P = {k: d['barphi_p'].double() for k, d in aux.items()}
    E = {k: d['barphi_e'].double() for k, d in aux.items()}
    mom = {}
    for k in cfg:
        z = ((P[k][-1] - E[k][-1]) / se[k].clamp_min(1e-30)).abs()
        rel = ((P[k] - E[k]).abs() / E[k].abs().clamp_min(1e-300))
        mom[k] = {'abs_z_final': q(z), 'rel_err_final': q(rel[-1]),
                  'rel_err_max_per_row_every_1000': [float(v) for v in rel.max(1).values[::1000]]}
    d = (P['fast'] - P['orig']).abs() / se['orig'].clamp_min(1e-30)
    mom['fast_minus_orig_in_SE'] = {'final': q(d[-1]), 'max_per_row_every_1000': [float(v) for v in d.max(1).values[::1000]],
                                    'max_all_rows': float(d.max())}
    res['moments'] = mom
    for k in cfg:
        print(f'[moments] {k}: |z| at t=1 {mom[k]["abs_z_final"]}, rel {mom[k]["rel_err_final"]}', flush=True)
    print(f'[moments] |p_fast - p_orig| / SE: final {mom["fast_minus_orig_in_SE"]["final"]}, '
          f'max over run {mom["fast_minus_orig_in_SE"]["max_all_rows"]:.2f}', flush=True)

    # 2. theta vs its sampling noise
    tt = t[-th['orig'].shape[0]:]
    dth = (th['fast'] - th['orig']).norm(dim=1)
    gen = torch.Generator(device='cuda').manual_seed(a.seed)
    n = x1.shape[0]
    noise = {}
    for tc in a.t_noise:
        set_time(S, tc)
        y = state_at(S, tc, gen)
        with torch.no_grad():
            S._rhs_dt_cache = (0, S.compute_rhs_dt_phi_I_t(S.compute_interpolant(0), 0))
            thA = S.compute_theta(y[:n // 2], 0)[0].double().cpu()
            S._rhs_dt_cache = (0, S.compute_rhs_dt_phi_I_t(S.compute_interpolant(0), 0))
            thB = S.compute_theta(y[n // 2:], 0)[0].double().cpu()
        j = int((tt - tc).abs().argmin())
        nz = float((thA - thB).norm() / np.sqrt(2))
        noise[str(tc)] = {'row': j, 't_row': float(tt[j]), 'noise_norm': nz, 'diff_norm': float(dth[j]),
                          'diff_over_noise': float(dth[j]) / nz,
                          'theta_orig_norm': float(th['orig'][j].norm())}
        print(f'[theta] t={tc}: ||fast-orig|| / noise = {noise[str(tc)]["diff_over_noise"]:.3f}', flush=True)
    res['theta'] = {'vs_noise': noise,
                    'rel_diff_every_1000': [float(v) for v in
                                            (dth / th['orig'].norm(dim=1).clamp_min(1e-300))[::1000]],
                    'dead_set_differs_total': int(((th['fast'] == 0) != (th['orig'] == 0)).sum())}

    Path(a.out).mkdir(parents=True, exist_ok=True)
    f = Path(a.out) / f'fast_vs_orig_M{a.M}.json'
    with open(f, 'w') as fh:
        json.dump(res, fh, indent=1)
    print('written', f, flush=True)


if __name__ == '__main__':
    main()
