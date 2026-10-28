"""Moment misfit of MGD runs at t=1: sampling noise or bias, and does the Ito term explain it?

Per statistic n (aux_moments order: L_6, then the other terms in builder order):
  d_n     = (p_n - e_n) / |e_n|, rel_n = |d_n|; e = barphi_e (target), p = barphi_p (walkers), t=1
  z_n     = |p_n - e_n| / SE_n, SE_n^2 = Var phi_n(walkers)/n1 + Var phi_n(data)/n1.
            ~|N(0,1)| (median 0.67) if the gap were the noise of two independent samples.
            Walkers are NOT independent of the data (the corrector matches empirical moments),
            so z >> 1 only excludes "too few samples in the window", not other biases.
  along the run (aux has one row per step): peak of rel_n(t) and its 1-t, onset (first 1-t with
            rel > 1e-2); sign persistence = fraction of the steps with 1-t < 1e-2 where
            sign d_n(t) = sign d_n(1). Noise around 0 -> ~50%; a lag -> ~100%.
  Ito term the predictor leaves to the corrector (Scalar_*_gaussianK statistics only):
            I_n(h) = h sigma^2 <Laplacian_x phi_n> / |e_n|,
            Laplacian_x phi_{j,k} = ||g_j||^2 mean_s f_k''(z_j(s)), g_j = Re ifft(psi_hat_j),
            on the t=1 samples (not on the walkers at time t), h of 1-t = 1e-2, 1e-3, last step.
            The mechanism predicts rel_n growing with |I_n| and sign(d_n) = sign(I_n).
With several seeds: per statistic mean and std of d_n across seeds; |mean| / (std/sqrt(S)) >> 1
-> systematic, ~1 -> seed noise.

Needs, per run: saved_results/{aux_moments,samples,sampling_times} and
experiments/<group>/<config>/{config.json,fitted_potentials/}. Data: TURBULENCE_1D_DATA_PATH.

Usage (repo root on Jean Zay; CPU is enough, a few minutes per n1=10000 run):
    python turbulence/moment_error_variance.py --label turbulence_ggd_mass_L6_2b --seeds 900
    python turbulence/moment_error_variance.py --label turbulence_nested_M2 --seeds 900 901 902 903 904
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data', HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from filters_bank import return_Filters                              # noqa: E402
from potentials_builder import get_1d_potentials                     # noqa: E402
from check_fit_offline import build_data                             # noqa: E402
from window_vs_step_noise import newest                              # noqa: E402

MISS = 1e-2
SCALAR = ('Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK',
          'Scalar_psi_gaussianK_imag', 'Scalar_morlet_gaussianK_imag')
H_KEYS = ('1e-2', '1e-3', 'end')


def scalar_laplacian(pt, x, batch=250):
    """(N, n_active) Laplacian in x of each active statistic of a Scalar_GGD_KRegion
    (same scaling as forward, i.e. divided by stat_scale). Checked against a central
    difference of pt.grad: max rel err 1e-4 at eps 1e-7 (2026-10-23)."""
    assert pt.dwt is None, 'DWT potentials not supported'
    f = pt.filters.to(torch.complex128)           # complex: the _imag terms carry filters * -1j
    g2 = torch.fft.ifft(f).real.pow(2).sum(-1).reshape(-1)              # ||g_j||^2, (J,)
    c, s, alpha, K = pt.cuts.double(), pt.sw.double(), pt.alpha.double(), pt.K
    eps = pt.eps_ch.double()[None, :, None] if pt.eps_ch is not None else pt.eps_abs

    def slot_values(z, k):                       # same formulas as _all_slot_forward, before the mean
        az = torch.sqrt(z ** 2 + eps)
        ws = pt._windows_from_sigmoids(torch.sigmoid(-(az.unsqueeze(-1) - c[None, :, None, :])
                                                     / s[None, :, None, :]))
        m, is_mass = pt.slot_window(k)
        return ws[m] if is_mass else ws[m] * az ** alpha[:, m][None, :, None]

    out = []
    for b in range(0, x.shape[0], batch):
        z0 = torch.fft.ifft(f * torch.fft.fft(x[b:b + batch].double())).real.detach()
        slots = []
        for k in range(pt.n_slots):
            z = z0.clone().requires_grad_(True)
            d1, = torch.autograd.grad(slot_values(z, k).sum(), z, create_graph=True)
            d2, = torch.autograd.grad(d1.sum(), z)                         # f'' elementwise (separable)
            slots.append(d2.mean(-1) * g2[None, :])                       # (B, J)
        lap = torch.cat(slots, dim=1).index_select(1, pt.active_flat.cpu())
        out.append(lap / pt.stat_scale.double()[None, :])
    return torch.cat(out).numpy()


def analyse(label, seed, root, with_ito=True):
    """All per-statistic quantities of one run (dict of arrays + metadata)."""
    res = root / 'saved_results'
    # newest run folder of this label/seed that has results (pre-flight-only runs share the label)
    exps = sorted((root / 'experiments').glob(f'*/*_seed_{seed}_*{label}_2*'), key=lambda f: f.stat().st_mtime)
    exps = [x for x in exps if (res / 'aux_moments' / f'{x.name}_aux_moments.pt').exists()]
    if not exps:
        raise FileNotFoundError(f'no run of {label} seed {seed} with aux_moments under {root}')
    exp = exps[-1]
    a = json.loads((exp / 'config.json').read_text())
    cfg = a.get('config_name', exp.name)
    M, J, Q, sigma = int(a.get('M', a.get('target_len', 128))), int(a['J']), int(a['Q']), float(a['sigma'])
    n1, terms = int(a['n1']), list(a['terms'])
    print(f'\n=== seed {seed}: {cfg}\n  n1={n1} M={M} J={J} Q={Q} sigma={sigma} terms={terms}')
    if a.get('n1_spread'):
        print('  WARNING: run used --n1_spread; data x1 rebuilt as the first n1 series (SE of the data side approximate)')

    aux = torch.load(res / 'aux_moments' / f'{cfg}_aux_moments.pt', map_location='cpu')
    BE, BP = aux['barphi_e'].double().numpy(), aux['barphi_p'].double().numpy()
    e, p = BE[-1], BP[-1]
    xs = torch.load(newest(f'{cfg}*', res / 'samples'), map_location='cpu').double()
    if xs.dim() == 2:
        xs = xs[:, None]
    t = np.asarray(torch.load(newest(f'{cfg}*', res / 'sampling_times'), map_location='cpu'), float).ravel()
    # aux rows: the x_0 row + one per step (+ a repeated final row) -> row j at t[j]
    if len(BE) in (len(t), len(t) + 1):
        t_rows = t[:len(BE)] if len(BE) == len(t) else np.append(t, t[-1])
    elif len(BE) == len(t) - 1:
        t_rows = t[1:]
    else:
        raise SystemExit(f'{len(BE)} aux rows vs {len(t)} times: param_storage_frequency != 1?')
    h, gap = np.diff(t), 1.0 - t[1:]
    h_at = dict(zip(H_KEYS, [float(h[np.argmin(np.abs(gap - g))]) for g in (1e-2, 1e-3)] + [float(h[-1])]))

    filters, filters_phi = return_Filters(M, J, 1, include_phi=True, wav_type=a.get('wav_type', 'morlet'))
    filters_Q = return_Filters(M, J, Q, wav_type=a.get('wav_type', 'morlet'))
    ggd = {k: v for k, v in dict(tail_quantile=a.get('ggd_tail_quantile'),
                                 tail_alpha_bounds=a.get('ggd_tail_alpha_bounds'),
                                 tail_mass=a.get('ggd_tail_mass', False),
                                 all_masses=a.get('ggd_all_masses', False)).items() if v}
    pots = get_1d_potentials(terms, J, filters, Q, filters_Q=filters_Q, filters_Phi=filters_phi,
                             deduplicate_filters=bool(a.get('deduplicate_filters', False)),
                             dedup_tol=float(a.get('dedup_tol', 1e-12)), ggd_kwargs=ggd)
    for name in list(pots):                      # the run's fitted states (incl. drops), no refit
        state = exp / 'fitted_potentials' / f'{name}.pt'
        if hasattr(pots[name], 'load_fixed_parameters') and state.exists():
            pots[name] = type(pots[name]).load_fixed_parameters(state, pots[name].filters, map_location='cpu')

    x1, _ = build_data(int(a.get('subseries_len', 512)), M, n1, 1)
    x1 = x1.double()

    def per_sample(x):                           # (N, r), same concatenation as SDE.compute_moments
        return torch.cat([torch.cat([pt.forward(x[b:b + 500]).double() for pt in pots.values()], 1)
                          for b in range(0, x.shape[0], 500)]).numpy()

    phi_s, phi_d = per_sample(xs), per_sample(x1)
    if phi_s.shape[1] != len(e):
        raise SystemExit(f'{phi_s.shape[1]} rebuilt statistics vs {len(e)} in aux_moments: terms/order mismatch')
    chk = np.abs(phi_s.mean(0) - p) / np.abs(p).clip(1e-300)
    print(f'  check: rebuilt phi on saved samples vs aux barphi_p: median rel {np.median(chk):.1e}, '
          f'max {chk.max():.1e}' + ('   <-- MISMATCH, numbers below unreliable' if chk.max() > 1e-3 else ''))

    sizes = [pt.forward(x1[:1]).shape[1] for pt in pots.values()]
    names = [f'{nm}[{i}]' for nm, k in zip(pots, sizes) for i in range(k)]
    start = dict(zip(pots, np.cumsum([0] + sizes[:-1])))

    d = (p - e) / np.abs(e).clip(1e-300)
    se = np.sqrt(phi_s.var(0, ddof=1) / phi_s.shape[0] + phi_d.var(0, ddof=1) / phi_d.shape[0])
    d_t = (BP - BE) / np.abs(BE).clip(1e-300)                            # (rows, r)
    tail = (1.0 - t_rows) < 1e-2
    tail[0] = False
    persist = (np.sign(d_t[tail]) == np.sign(d)[None, :]).mean(0)

    ito = np.full((len(e), len(H_KEYS)), np.nan)
    for name in SCALAR if with_ito else ():
        if name in pots:
            lap = scalar_laplacian(pots[name], xs).mean(0)
            sl = slice(start[name], start[name] + len(lap))
            ito[sl] = np.outer(lap / np.abs(e[sl]).clip(1e-300), [hv * sigma ** 2 for hv in h_at.values()])
    return dict(seed=seed, names=names, e=e, p=p, d=d, rel=np.abs(d), z=np.abs(p - e) / se.clip(1e-300),
                rel_t=np.abs(d_t), g_rows=1.0 - t_rows, persist=persist, ito=ito, h_at=h_at)


def report(r):
    rel, z, miss, rel_t, g = r['rel'], r['z'], r['rel'] > MISS, r['rel_t'], r['g_rows']

    def summ(mask, tag):
        if not mask.any():
            return
        x = rel[mask]
        print(f'  {tag:<9} n={mask.sum():4d}  rel mean {x.mean():.2e} median {np.median(x):.2e} '
              f'p90 {np.quantile(x, .9):.2e} max {x.max():.2e} | z median {np.median(z[mask]):.2g} '
              f'| sign persistence (1-t<1e-2) median {np.median(r["persist"][mask]):.0%}')

    print(f'1. {len(rel)} statistics at t=1 (miss: rel > {MISS:g}); z median 0.67 under independent-sample noise')
    summ(np.ones_like(miss), 'all')
    summ(~miss, 'matched')
    summ(miss, 'missed')

    has = ~np.isnan(r['ito'][:, 0])
    if has.sum() > 2:
        print(f'2. Ito term, {has.sum()} Scalar statistics (Spearman of |I_n| with rel_n across statistics):')
        for i, (k, hv) in enumerate(r['h_at'].items()):
            I = r['ito'][has, i]
            rho, pv = spearmanr(np.abs(I), rel[has])
            m = miss[has]
            print(f'   1-t={k:>5} h={hv:.2e}: Spearman {rho:+.2f} (p {pv:.0e}); |I| median missed '
                  f'{np.median(np.abs(I[m])) if m.any() else np.nan:.2e} vs matched {np.median(np.abs(I[~m])):.2e}')
        mi = np.nonzero(has & miss)[0]
        if len(mi):
            agree = np.mean(np.sign(r['d'][mi]) == np.sign(r['ito'][mi, -1]))
            print(f'   sign(p - e) == sign(I) for {agree:.0%} of the {len(mi)} missed Scalar statistics')

    if miss.any():
        print('3. missed statistics (peak/onset along the run; I = Ito term / |e| at 1-t = 1e-2, 1e-3, end)')
        print(f"{'stat':<30}{'rel':>9}{'p/e':>7}{'z':>7}{'persist':>8}{'peak':>9}{'@1-t':>9}{'onset 1-t':>10}"
              f"{'I(1e-2)':>10}{'I(1e-3)':>10}{'I(end)':>10}")
        for i in np.argsort(-rel):
            if not miss[i]:
                break
            jp = int(rel_t[1:, i].argmax()) + 1
            above = np.nonzero(rel_t[1:, i] > MISS)[0]
            onset = f'{g[above[0] + 1]:.1e}' if len(above) else '-'
            Is = ''.join(f'{v:10.2e}' if np.isfinite(v) else f"{'-':>10}" for v in r['ito'][i])
            print(f"{r['names'][i]:<30}{rel[i]:9.2e}{r['p'][i] / r['e'][i]:7.3f}{z[i]:7.0f}{r['persist'][i]:8.0%}"
                  f"{rel_t[jp, i]:9.2e}{g[jp]:9.1e}{onset:>10}{Is}")


def across_seeds(runs):
    names = runs[0]['names']
    if any(r['names'] != names for r in runs[1:]):
        print('\nseeds have different statistics (fits differ): no cross-seed table')
        return
    D = np.stack([r['d'] for r in runs])                                # (S, r)
    S = len(runs)
    mean, std = D.mean(0), D.std(0, ddof=1)
    tr = np.divide(np.abs(mean), std / np.sqrt(S), out=np.full_like(mean, np.inf), where=std > 0)
    miss = np.abs(mean) > MISS
    print(f'\n4. across {S} seeds: d_n = (p - e)/|e|; |mean|/(std/sqrt(S)) >> 1 -> systematic')
    for mask, tag in ((~miss, 'matched'), (miss, 'missed')):
        if mask.any():
            print(f'  {tag:<9} n={mask.sum():4d}  |mean d| median {np.median(np.abs(mean[mask])):.2e} | '
                  f'std across seeds median {np.median(std[mask]):.2e} | ratio median {np.median(tr[mask]):.3g}')
    for i in np.argsort(-np.abs(mean)):
        if not miss[i]:
            break
        print(f'  {names[i]:<30} mean d {mean[i]:+.3e}  std {std[i]:.1e}  ratio {tr[i]:9.3g}  '
              f'same sign in {np.mean(np.sign(D[:, i]) == np.sign(mean[i])):.0%} of seeds')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='turbulence_ggd_mass_L6_2b')
    ap.add_argument('--seeds', type=int, nargs='+', default=[900])
    ap.add_argument('--root', type=Path, default=HERE,
                    help='folder holding experiments/ and saved_results/ (default: turbulence/)')
    ap.add_argument('--full', type=int, default=1, help='number of seeds that also get the Ito term (slowest part)')
    a = ap.parse_args()
    runs = []
    for i, s in enumerate(a.seeds):
        runs.append(analyse(a.label, s, a.root, with_ito=i < a.full))
        report(runs[-1])
    if len(runs) > 1:
        across_seeds(runs)
