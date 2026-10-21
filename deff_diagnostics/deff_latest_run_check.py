"""Energy- vs denoiser-form d_eff on the LATEST turbulence runs (noL6psi_noSm30, moment-mode
Theta_reg), split into its parts.  One seed per call.  See deff_diagnostics/README.md.

Checks, in order (all printed; small summary CSV + per-point CSV written to --outdir):
  [grid]   t_reg (saved in the Theta_reg file) vs the run's fine t; reconstruct_t_reg(t, 1) vs t_reg;
           lengths of theta_t / dH / barphi_e vs t.
  [logZ]   energy_d_eff's log Z = H_bound(theta_k) + Theta.m  (pipeline) vs a log Z built from
           Theta_reg ALONE, H(p0) + Theta.m - int Theta.dm  (consistent). Their difference
           delta(t) = int (Theta_reg - theta_k).mdot dt mixes the per-step theta (in dH) with the
           smoothed Theta_reg; it reaches d_eff only as the x-independent shift -d delta/ds.
  [prec]   max|Theta| and float32-vs-float64 score Theta.grad phi at a few nodes.
  [deff]   denoiser float64 (n_mc A, B) and float32 (B); energy for log Z in {bound (h_bound_at),
           thetareg (h_from_theta_reg)} x noise in {fresh per node, common to all nodes} x n_mc {A, B}
           -- per node: mean over points of (energy - denoiser f64 at B), its sd over points, sign fraction.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
sys.path.insert(0, str(ROOT / 'data'))
from filters_bank import return_Filters
from potentials_builder import get_1d_potentials
from utils import split_periodize_reshape, normalize
from utils_entropy import standard_gaussian_entropy
from ortho_wavelet.ReadyToUseWavelets import DefineWavelet
from data_loader import load_turbulence_1d
from effective_dimension import (reconstruct_t_reg, h_bound_at, h_from_theta_reg, extract_rare_typical,
                                 denoiser_d_eff, energy_d_eff, alpha_of_t, tau_of_t, s_of_t,
                                 local_poly_derivative, grad_phi_projected)

p = argparse.ArgumentParser()
p.add_argument('--run', default='latest', choices=['latest', 'final'],
               help='latest: noL6psi_noSm30 reg1eX (nt=60000, n1=8500, seeds 900-904); '
                    'final: turbulence_final (nt=55000, n1=10000, seeds 900/901/903/904/905, L_6 free)')
p.add_argument('--arm', default='reg1e-4', choices=['reg1e-4', 'reg1e-2'], help='only used for --run latest')
p.add_argument('--seed', type=int, required=True)
p.add_argument('--theta', default='neg', choices=['neg', 'free'],
               help='--run latest only: neg = newest lam1e-3 L_6<0 solve; free = lam1e-3 unconstrained')
p.add_argument('--lam', type=float, default=None, help='Theta_reg lambda; default 1e-3 (latest) or 5e-4 (final)')
p.add_argument('--ridge', type=float, default=None, help='Theta_reg ridge; default = from arm (latest) or 1e-4 (final)')
p.add_argument('--n_per_cat', type=int, default=20, help='typical and rare points each')
p.add_argument('--n_eval', type=int, default=160, help='eval nodes, uniform in s = log(tau)/2')
p.add_argument('--n_mc', type=int, nargs=2, default=[64, 256])
p.add_argument('--point_chunk', type=int, default=10)
p.add_argument('--outdir', default=str(ROOT / 'deff_diagnostics' / 'results'))
args = p.parse_args()

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TURB = ROOT / 'turbulence'
SCR = Path(os.environ['SCRATCH']) / 'MGD-for-Maximum-Entropy-Generation/turbulence'

if args.run == 'latest':
    RIDGE = args.ridge if args.ridge is not None else {'reg1e-4': 1e-4, 'reg1e-2': 1e-2}[args.arm]
    LAM = args.lam if args.lam is not None else 1e-3
    GROUP = ('turbulencesynth_M256_J8_Q3_sigma3.5_nt60000_n1_8500_lam0.0_thomas_nsub1_moment_f64_dedupfilt1e-08'
             f'_terms54d17f3b_zfloor_two_phase_noL6psi_noSm30_{args.arm}')
    CONFIG = ('turbulencesynth_M256_J8_Q3_sigma3.5_nt60000_n1_8500_lam0.0_thomas_nsub1_moment_f64_dedupfilt1e-08'
              f'_seed_{args.seed}_terms54d17f3b_zfloor_two_phase_noL6psi_noSm30_{args.arm}_20261001_1911')
    exp_dir = TURB / 'experiments' / GROUP / CONFIG
    if args.theta == 'neg':
        cand = sorted((SCR / 'theta_reg_l6neg' / CONFIG).glob(f'lam{LAM:g}_ridge{RIDGE:g}_moment_mask_neg0.01*.pt'),
                      key=lambda q: q.stat().st_mtime)
        theta_file = cand[-1]
    else:
        theta_file = SCR / 'theta_reg_lamsweep_xseed' / CONFIG / f'lam{LAM:g}_ridge{RIDGE:g}_moment_mask.pt'
    tag = f'{args.arm}_seed{args.seed}_{args.theta}'
else:                                                                       # final: n1=10000, nt=55000, L_6 free
    RIDGE = args.ridge if args.ridge is not None else 1e-4
    LAM = args.lam if args.lam is not None else 5e-4
    GROUP = ('turbulencesynth_M256_J8_Q3_sigma3.5_nt55000_n1_10000_lam0.0_thomas_nsub1_moment_f64_dedupfilt1e-08'
             '_terms54d17f3b_turbulence_final')
    pat = ('turbulencesynth_M256_J8_Q3_sigma3.5_nt55000_n1_10000_lam0.0_thomas_nsub1_moment_f64_dedupfilt1e-08'
           f'_seed_{args.seed}_terms54d17f3b_turbulence_final_*')
    cands = sorted((TURB / 'experiments' / GROUP).glob(pat))
    if not cands:
        raise FileNotFoundError(f'no final-run folder for seed {args.seed} under {TURB / "experiments" / GROUP}')
    exp_dir = cands[-1]
    CONFIG = exp_dir.name
    theta_file = TURB / 'saved_results' / 'theta_reg_lamsweep' / CONFIG / f'lam{LAM:g}_ridge{RIDGE:g}_moment_mask.pt'
    tag = f'final_seed{args.seed}_lam{LAM:g}'

cfg = json.loads((exp_dir / 'config.json').read_text())
out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
t_start = time.time()


def saved(kind):
    q = TURB / 'saved_results' / kind / CONFIG
    for c in (q.with_name(q.name + '.pt'), q):
        if c.exists():
            return torch.load(c, map_location='cpu', weights_only=False)
    raise FileNotFoundError(q)


# ------------------------------------------------------------------ inputs
sol = torch.load(theta_file, map_location='cpu', weights_only=False)
Theta = sol['Theta_reg'].double()
t_reg = np.asarray(sol['t_reg'], dtype=np.float64)
theta_k = saved('lagrange_multipliers').double()
dH = saved('entropy_bounds').double().numpy()
t = saved('sampling_times')
t = t.double().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float64)
aux = torch.load(TURB / 'saved_results' / 'aux_moments' / f'{CONFIG}_aux_moments.pt', map_location='cpu')
barphi_e = aux['barphi_e'].double()
print(f'{tag}: Theta_reg {tuple(Theta.shape)} from {theta_file.name}\n'
      f'  theta_k {tuple(theta_k.shape)}, dH {dH.shape}, t {t.shape}, barphi_e {tuple(barphi_e.shape)}')

# ------------------------------------------------------------------ [grid]
print('\n[grid]')
print(f'  len(t)-1 = {len(t) - 1};  theta_k {len(theta_k)}, dH {len(dH)}, barphi_e {len(barphi_e)} '
      f'(all should equal len(t)-1, aligned with t[1:])')
t1 = t[1:]
idx = np.searchsorted(t1, t_reg)
idx = np.clip(idx, 0, len(t1) - 1)
off = np.abs(t1[idx] - t_reg)
print(f'  t_reg: {len(t_reg)} nodes, t_reg[0]={t_reg[0]:.10f} = t[{idx[0] + 1}], t_reg[-1]={t_reg[-1]:.10f} = t[{idx[-1] + 1}]'
      f'  (t[-1]={t[-1]:.10f});  max |t_reg - t[1:][idx]| = {off.max():.2e}')
rec = reconstruct_t_reg(t, n_subsample=1)
print(f'  reconstruct_t_reg(t, 1): {len(rec)} nodes vs saved {len(t_reg)}'
      + (f', max |diff| {np.abs(rec - t_reg).max():.2e}' if len(rec) == len(t_reg) else '  (LENGTH DIFFERS)'))
assert off.max() < 1e-12, 't_reg is not a subset of t[1:] -- stop, alignment assumption broken'
m = barphi_e[idx]                       # m_t at t_reg (exact nodes)
th_k = theta_k[idx]                     # per-step theta at t_reg
d = int(np.prod(cfg['M'])) * cfg['channels']

# ------------------------------------------------------------------ [logZ]
print('\n[logZ]  bound (theta_k in the integral) minus Theta_reg version: shift of energy-form d_eff (x-independent)')
H_pipe = h_bound_at(t, dH, t_reg, d)                       # MGD bound: per-step theta_k inside the integral
H_cons = h_from_theta_reg(Theta, m, d)                     # Theta_reg inside the integral
H_cons += H_pipe[0] - H_cons[0]                            # same constant at the first node (irrelevant for d/ds)
logZ_pipe = H_pipe + (Theta * m).sum(1).numpy()
logZ_cons = H_cons + (Theta * m).sum(1).numpy()
delta = logZ_pipe - logZ_cons
s_all, tau_all = s_of_t(t_reg), tau_of_t(t_reg)
shift_dense = -local_poly_derivative(s_all, delta, half_window=25)
# also the integrand, to see which coefficients drive it: (Theta - theta_k).mdot per step
mdot = np.gradient(m.numpy(), t_reg, axis=0)
mix = ((Theta.numpy() - th_k.numpy()) * mdot)
bands = [(1e2, np.inf), (1, 1e2), (1e-2, 1), (1e-4, 1e-2), (1e-6, 1e-4), (0, 1e-6)]
rows = []
for lo, hi in bands:
    mk = (tau_all >= lo) & (tau_all < hi)
    if mk.any():
        rows.append(dict(tau_band=f'[{lo:g},{hi:g})', nodes=int(mk.sum()),
                         delta_end=delta[mk][-1], shift_median=np.median(shift_dense[mk]),
                         shift_absmax=np.abs(shift_dense[mk]).max()))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.3g}'))
top = np.argsort(-np.abs(mix).sum(0))[:8]
print(f'  coefficients carrying most of int |(Theta - theta_k).mdot| dt: {top.tolist()} '
      f'(share {np.abs(mix[:, top]).sum() / np.abs(mix).sum():.2f})')

# ------------------------------------------------------------------ data, points, potentials
W = DefineWavelet('Db', m=3, device=dev)
Data = split_periodize_reshape(load_turbulence_1d().to(dev), cfg['subseries_len'])
for _ in range(int(np.log2(cfg['subseries_len'] / cfg['target_len']))):
    Data = W.decompose(Data)[1]
x1 = normalize(Data[:cfg['n1']]).to(dev)
idx_typ, idx_rare, _ = extract_rare_typical(Data[:cfg['n1']], criterion='percentile', percentile=95)
rng = np.random.default_rng(0)
pts = np.concatenate([rng.choice(idx_typ, args.n_per_cat, replace=False), rng.choice(idx_rare, args.n_per_cat, replace=False)])
cat = np.array(['typical'] * args.n_per_cat + ['rare'] * args.n_per_cat)
x_eval = x1[pts]

filters, filters_Phi = return_Filters(cfg['M'], cfg['J'], 1, device=dev, include_phi=True)
filters_Q = return_Filters(cfg['M'], cfg['J'], cfg['Q'], device=dev)
pots = get_1d_potentials(cfg['terms'], cfg['J'], filters, cfg['Q'], scalar_param=None, parallel=False,
                         filters_Q=filters_Q, filters_Phi=filters_Phi,
                         deduplicate_filters=cfg.get('deduplicate_filters', False), dedup_tol=cfg.get('dedup_tol', 1e-12))
for name, pot in pots.items():
    pot.to(dev)
    if hasattr(pot, 'is_fitted'):
        pots[name] = type(pot).load_fixed_parameters(exp_dir / 'fitted_potentials' / f'{name}.pt', pot.filters, map_location=dev)
        pots[name].to(dev)
r = sum(q.num_coefficients for q in pots.values())
assert r == Theta.shape[1], (r, Theta.shape)

# ------------------------------------------------------------------ [prec]
print('\n[prec]  score Theta.grad phi(sin a y): float32 contraction (old denoiser) vs float64 where accepted (new)')
g = torch.Generator(device=dev).manual_seed(0)
for tau_q in (1e2, 1.0, 1e-2, 1e-4, 1e-6, tau_all.min()):
    i = int(np.argmin(np.abs(np.log(tau_all) - np.log(tau_q))))
    a = alpha_of_t(t_reg[i])
    xi = torch.randn((16,) + tuple(x_eval.shape[1:]), generator=g, device=dev)
    sy = np.sin(a) * x_eval[:16] + np.cos(a) * xi
    s32 = grad_phi_projected(pots, sy, Theta[i].to(dev, torch.float32)).double()
    s64 = grad_phi_projected(pots, sy, Theta[i].to(dev), float64=True)
    print(f'  tau={tau_all[i]:.2e}: max|Theta| {Theta[i].abs().max():.2e};  rel |s32 - s64| {(s32 - s64).norm() / s64.norm():.2e}')

# ------------------------------------------------------------------ [deff]
q = np.linspace(s_all.min(), s_all.max(), args.n_eval)
ev = np.unique([int(np.argmin(np.abs(s_all - v))) for v in q])
te, Te, me = t_reg[ev], Theta[ev], m[ev]
H_by = {'bound': H_pipe[ev], 'thetareg': H_cons[ev]}
pred_shift = -local_poly_derivative(s_of_t(te), delta[ev])
A, B = args.n_mc
print(f'\n[deff]  {len(ev)} eval nodes uniform in s, {len(pts)} points, n_mc {A}, {B}; '
      f'energy: log Z in (bound, thetareg) x noise in (fresh, common)')

res = {}
for c0 in range(0, len(pts), args.point_chunk):
    xs = x_eval[c0:c0 + args.point_chunk]
    for nmc in (A, B):
        gd = torch.Generator(device=dev).manual_seed(1000 * args.seed + nmc)
        res.setdefault(f'den64_{nmc}', []).append(denoiser_d_eff(xs, Te, te, pots, n_mc=nmc, generator=gd))
        for noise, common in (('fresh', False), ('common', True)):
            for lz, H in H_by.items():
                ge = torch.Generator(device=dev).manual_seed(2000 * args.seed + nmc + common)   # same xi for both log Z
                res.setdefault(f'en_{lz}_{noise}_{nmc}', []).append(
                    energy_d_eff(xs, Te, te, pots, me, H, d, n_mc=nmc, generator=ge, common_noise=common))
    gd = torch.Generator(device=dev).manual_seed(1000 * args.seed + B)
    res.setdefault(f'den32_{B}', []).append(denoiser_d_eff(xs, Te, te, pots, n_mc=B, generator=gd, float64=False))
    print(f'  points {c0}-{c0 + len(xs) - 1} done, {time.time() - t_start:.0f} s')
res = {k: np.concatenate(v, axis=1) for k, v in res.items()}       # (n_eval, n_points)

chk = max(np.abs((res[f'en_bound_{nz}_{n}'] - res[f'en_thetareg_{nz}_{n}']).mean(1) - pred_shift).max()
          for nz in ('fresh', 'common') for n in (A, B))
print(f'  check: energy(bound) - energy(thetareg) = predicted -d delta/ds to {chk:.2e} (same xi; should be ~0)')

per_pt = []
for j, i in enumerate(ev):
    for k in range(len(pts)):
        per_pt.append(dict(seed=args.seed, t=te[j], tau=tau_all[i], point=int(pts[k]), category=cat[k],
                           **{key: float(v[j, k]) for key, v in res.items()}))
pd.DataFrame(per_pt).to_csv(out / f'deff_points_{tag}.csv', index=False)

ref = res[f'den64_{B}']                    # reference: denoiser, float64, largest n_mc
summ = []
for j, i in enumerate(ev):
    row = dict(seed=args.seed, t=te[j], tau=tau_all[i], pred_logZ_shift=pred_shift[j], den_mean=ref[j].mean(),
               den_mc_sd=(res[f'den64_{A}'][j] - ref[j]).std(ddof=1),
               den32_minus_64=np.abs(res[f'den32_{B}'][j] - ref[j]).mean())
    for key in res:
        if key.startswith('en_'):
            gap = res[key][j] - ref[j]
            row[f'gap_{key[3:]}_mean'] = gap.mean()
            row[f'gap_{key[3:]}_sdx'] = gap.std(ddof=1)
            row[f'gap_{key[3:]}_fracneg'] = (gap < 0).mean()
    summ.append(row)
summ = pd.DataFrame(summ)
summ.to_csv(out / f'deff_summary_{tag}.csv', index=False)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
summ['band'] = pd.cut(summ.tau, [0, 1e-6, 1e-4, 1e-2, 1, 1e2, np.inf], right=False)
fmt = lambda v: f'{v:.3g}'
print('\nper tau band, medians over nodes.  den = denoiser d_eff (f64, n_mc B); den32_minus_64 = precision effect')
print(summ.groupby('band', observed=True)[['den_mean', 'den_mc_sd', 'den32_minus_64', 'pred_logZ_shift']].median().to_string(float_format=fmt))
for stat, what in (('mean', 'MEAN over points of (energy - denoiser)'), ('sdx', 'SD over points of (energy - denoiser)')):
    print(f'\n{what}:')
    cols = [f'gap_{lz}_{nz}_{n}_{stat}' for lz in H_by for nz in ('fresh', 'common') for n in (A, B)]
    t_ = summ.groupby('band', observed=True)[cols].median()
    t_.columns = [c[4:-len(stat) - 1] for c in cols]
    print(t_.to_string(float_format=fmt))
print(f'\nreading guide: sd shrinking ~2x from n_mc {A} -> {B}, or under common noise = MC noise; '
      f'bound vs thetareg mean differ by pred_logZ_shift by construction -- the question is which one is ~0; '
      f'a mean that stays nonzero, seed-consistent, at large n_mc with common noise = model (heat-equation) residual.')
print(f'total {time.time() - t_start:.0f} s -> {out}/deff_summary_{tag}.csv')
