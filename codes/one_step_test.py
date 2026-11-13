"""Part 1 of the terminal-phase test: does ONE noise step + ONE corrector pass, applied to the saved final walkers
of a finished run, reproduce the moment errors seen at the end of that run?  (Nothing here changes the SDE code.)

Hypothesis (2026-11-11): the error at the end of a run is what the single corrector pass of the last step leaves behind.

For one run (turbulence or jets; label + seed) the script loads, without refitting anything:
  * the run's fitted potentials (experiments/<group>/<config>/fitted_potentials) and the final samples (saved_results/samples),
  * the target moments of the last time e = barphi_e[-1] and the generated moments p = barphi_p[-1] (saved_results/aux_moments),
  * the last step h of the time grid (saved_results/sampling_times) and sigma, ridge, solve precision from config.json.
and then, with the SDE class itself (class OneStep below only replaces the interpolant by the fixed target e, so the real
SDE.compute_theta / compute_G / _live_potentials / Jacobi scaling + ridge / compute_grad_phi_projected are what runs):

  y  = x_final + sqrt(2 h) sigma * N(0, 1)          (the predictor drift over one step is neglected)
  th = SDE.compute_theta(y)                          (Gram at the noised walkers, Jacobi scaling, ridge, live mask)
  x' = y + (grad phi(y))^T th ;   residual = phi_bar(x') - e          (~10 independent noise draws)

Experiments (all start from the saved final walkers, target held fixed):
  0  reproduction: h = last step, ridge as in the run, 1 pass: mean (bias) and spread over the draws vs the observed error p - e
  a  step size h x {1, 1/4, 1/16, 1/64}: measured exponent of |mean residual| in h
  b  1, 2, 3 corrector passes (chord: Gram of y reused, as SDE._extra_corrector, unguarded)
  c  ridge {1e-2, 1e-4, 1e-6}
  d  repeated steps (noise + corrector, target fixed) at h/16 and h/64: residual after 1, 10, 100, 300 steps
  e  (turbulence only, --edge_scales) sigmoid edge width sw of the scalar potentials x {2, 4}: statistic scale and targets recomputed
     from the data; Ito term h sigma^2 <Laplacian phi>/|e| at sw x {0.5, 1, 2, 4}. Best effort: a failure here does not stop the rest.

Output: a text report on stdout and <out>/part1_<tag>.npz (+ part1_<tag>.csv: one row per statistic for experiment 0).
Needs a GPU for the real sizes (Gram of r ~ 300 statistics on 10000 x 256 samples), a dev job is enough: see codes/launch_one_step_test.sh.

Usage (repo root on Jean Zay):
    python codes/one_step_test.py --kind turbulence --label turbulence_nested_M3_bicw_allmass_scat4 --seed 900 \
        --config _nt55000_ --out terminal_phase_test/part1_x --tag turb_scat4
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE, HERE / 'filters', HERE / 'potentials', ROOT / 'data', ROOT / 'turbulence', ROOT / 'jets'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sde_routines import SDE                                           # noqa: E402
from filters_bank import return_Filters                                # noqa: E402
from potentials_builder import get_1d_potentials                       # noqa: E402

SCALAR = ('Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK', 'Scalar_psi_gaussianK_imag', 'Scalar_morlet_gaussianK_imag')
# the five statistics of the turbulence runs that miss by 1.6 to 3.7 % (all above the target)
KNOWN_TURB = ('Scalar_psi_gaussianK[34]', 'Scalar_psi_gaussianK[78]', 'Scalar_psi_gaussianK[56]',
              'Scalar_psi_gaussianK[35]', 'Scalar_morlet_gaussianK[8]')


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# the harness: the real SDE with a fixed target
# ----------------------------------------------------------------------------------------------------------------------
class OneStep(SDE):
    """SDE whose corrector target is a fixed vector of moments (self.target) instead of the moments of an interpolant.
    Only __init__ (no x_0 / x_1 / interpolants), compute_interpolant and compute_rhs_constraint_correction are replaced."""

    def __init__(self, potentials, n1, M, sigma, device, batch_size, ridge, solve_float64=True):
        torch.nn.Module.__init__(self)
        self.potentials = potentials
        self.signal_dim = 1
        self.original_signal_shape = (n1, 1, M)
        self.batch_size = batch_size
        self.device = device
        self.sigma = sigma
        self.regularization = ridge
        self.solve_float64 = solve_float64
        sizes = [p.num_coefficients for p in potentials.values()]
        self.num_potentials = sum(sizes)
        self.indices_potentials = np.cumsum([0] + sizes)
        self._live_floor = torch.cat([
            torch.full((p.num_coefficients,), float(getattr(p, 'near_empty_tol', 0.0) or 0.0))
            for p in potentials.values()]).to(device)
        self.cond_every = 0
        self._cond_log = {'eta': [], 'theta': []}
        self._cond_spec = {'eta': [], 'theta': []}
        self.cond_path = None
        self.cond_flush_rows = 200
        self.extra_corrector_max = 0
        self.target = None

    def compute_interpolant(self, k):
        return None

    def compute_rhs_constraint_correction(self, x_k, I_k):
        self._mom_I = None
        self._mom_x = self.compute_moments(x_k)
        return self.target - self._mom_x.mean(0)


def solve_theta(G, rhs, ridge, live):
    """The solve of SDE.compute_theta (float64, Jacobi scaling, symmetrisation, ridge, live mask), copied line by line so that
    one Gram G(y) can be re-solved for several ridges / residuals. Checked against the real compute_theta on every draw."""
    def _s(Gl, r):
        Gd, rd = Gl.double(), r.double()
        D12 = torch.diag(Gd) ** 0.5
        Gn = Gd / (D12[:, None] * D12[None, :])
        Gn = (Gn + Gn.T) / 2
        Gn += ridge * torch.eye(Gn.shape[-1], dtype=Gn.dtype, device=Gn.device)
        th = torch.linalg.solve(Gn, rd / D12)
        return (th / D12).to(r.dtype)
    if live.all():
        return _s(G, rhs)
    th = torch.zeros(G.shape[0], dtype=G.dtype, device=G.device)
    th[live] = _s(G[live][:, live], rhs[live])
    return th


def noise_like(x, h, sigma, gen):
    return (2.0 * h) ** 0.5 * sigma * torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)


def eval_variants(H, x0, h, gen, ridges, max_passes):
    """One noise step of size h from x0, then the corrector with every (ridge, number of chord passes).
    Returns ({(ridge, passes): residual phi_bar(x') - target as float64 numpy}, max relative difference of my solve vs compute_theta)."""
    y = x0 + noise_like(x0, h, H.sigma, gen)
    H.regularization = ridges[0]
    th_real, rhs = H.compute_theta(y, 0)
    G, live = H._G_theta, H._live_theta
    out, chk = {}, 0.0
    for ridge in ridges:
        th = solve_theta(G, rhs, ridge, live)
        if ridge == ridges[0]:
            chk = float((th - th_real).abs().max() / th_real.abs().max().clamp_min(1e-300))
        x = y + H.compute_grad_phi_projected(y, th)
        mom = H.compute_moments(x).mean(0)
        out[(ridge, 1)] = (mom - H.target).double().cpu().numpy()
        for p in range(2, max_passes + 1):
            r = H.target - mom
            x = x + H.compute_grad_phi_projected(x, solve_theta(G, r, ridge, live))
            mom = H.compute_moments(x).mean(0)
            out[(ridge, p)] = (mom - H.target).double().cpu().numpy()
    return out, chk


def run_chain(H, x0, h, gen, steps, checkpoints, ridge):
    """steps x (noise of size h, one corrector pass with the real compute_theta), target held fixed."""
    x = x0.clone()
    H.regularization = ridge
    res = {}
    for s in range(1, steps + 1):
        y = x + noise_like(x, h, H.sigma, gen)
        th, _ = H.compute_theta(y, s)
        x = y + H.compute_grad_phi_projected(y, th)
        if s in checkpoints:
            res[s] = (H.compute_moments(x).mean(0) - H.target).double().cpu().numpy()
    return res


# ----------------------------------------------------------------------------------------------------------------------
# loading a finished run
# ----------------------------------------------------------------------------------------------------------------------
def newest(pattern, base):
    files = sorted(base.glob(pattern), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f'nothing matches {base}/{pattern}')
    return files[-1]


def decode_scalar(pt):
    J, K, mw = int(pt.J), int(pt.K), pt.mass_windows
    out = {}
    for i, f in enumerate(pt.active_flat.tolist()):
        k, j = divmod(f, J)
        w, mass = (k, False) if k < K else (mw[k - K], True)
        edges = [0.0] + pt.cuts[j].tolist() + [float('inf')]
        out[i] = f'ch{j} win{w} {"mass" if mass else "mom"} [{edges[w]:.2g},{edges[w + 1]:.2g}) {float(pt.pi[j, w]):.1%}'
    return out


def load_run(kind, label, seed, config, root, device):
    res = root / 'saved_results'
    exps = sorted((root / 'experiments').glob(f'*/*_seed_{seed}_*_{label}_2*'), key=lambda f: f.stat().st_mtime)
    exps = [x for x in exps if (res / 'aux_moments' / f'{x.name}_aux_moments.pt').exists()]
    if config:
        exps = [x for x in exps if all(c in x.name for c in config)]
    if not exps:
        raise FileNotFoundError(f'no finished run of {label} seed {seed} {config} under {root}')
    exp = exps[-1]
    a = json.loads((exp / 'config.json').read_text())
    cfg = a.get('config_name', exp.name)
    M, J, Q, terms = int(a.get('M', a.get('target_len', 256))), int(a['J']), int(a['Q']), list(a['terms'])
    wav = a.get('wav_type', 'morlet')
    log(f'run: {cfg}\n  n1={a["n1"]} M={M} J={J} Q={Q} nt={a["nt"]} sigma={a["sigma"]} ridge={a.get("regularization")} terms={terms}')

    aux = torch.load(res / 'aux_moments' / f'{cfg}_aux_moments.pt', map_location='cpu')
    e = aux['barphi_e'][-1].double().numpy()
    p = aux['barphi_p'][-1].double().numpy()
    del aux
    xs = torch.load(newest(f'{cfg}*', res / 'samples'), map_location='cpu')
    if xs.dim() == 2:
        xs = xs[:, None]
    t = np.asarray(torch.load(newest(f'{cfg}*', res / 'sampling_times'), map_location='cpu'), float).ravel()
    h_last = float(t[-1] - t[-2])

    filters, filters_phi = return_Filters(M, J, 1, include_phi=True, wav_type=wav)
    filters_Q = return_Filters(M, J, Q, wav_type=wav) if kind == 'turbulence' else None
    ggd = {k: v for k, v in dict(tail_quantile=a.get('ggd_tail_quantile'), num_regions=a.get('ggd_num_regions'),
                                 tail_alpha_bounds=a.get('ggd_tail_alpha_bounds'),
                                 tail_mass=a.get('ggd_tail_mass') or None, all_masses=a.get('ggd_all_masses') or None,
                                 boundary_method=a.get('ggd_boundary_method')).items() if v is not None}
    pots = get_1d_potentials(terms, J, filters, Q, filters_Q=filters_Q, filters_Phi=filters_phi, scalar_param=None,
                             parallel=False, deduplicate_filters=bool(a.get('deduplicate_filters', False)),
                             dedup_tol=float(a.get('dedup_tol', 1e-12)), ggd_kwargs=ggd)
    for name in list(pots):                           # the run's fitted states (incl. drops), no refit
        state = exp / 'fitted_potentials' / f'{name}.pt'
        if hasattr(pots[name], 'load_fixed_parameters') and state.exists():
            pots[name] = type(pots[name]).load_fixed_parameters(state, pots[name].filters, map_location=device)
        if hasattr(pots[name], 'to'):
            pots[name].to(device)
    names, meta = [], []
    for nm, pt in pots.items():
        dec = decode_scalar(pt) if nm in SCALAR else {}
        for i in range(pt.num_coefficients):
            names.append(f'{nm}[{i}]')
            meta.append(dec.get(i, ''))
    if len(names) != len(e):
        raise SystemExit(f'{len(names)} statistics rebuilt vs {len(e)} in aux_moments: terms/order mismatch')
    return dict(kind=kind, cfg=cfg, a=a, M=M, pots=pots, xs=xs.to(device), e=e, p=p, h_last=h_last, names=names, meta=meta,
                sigma=float(a['sigma']), n1=int(a['n1']), ridge=float(a.get('regularization', 1e-4)),
                batch=int(a.get('batch_size', 5000)), f64=bool(a.get('solve_float64', True)))


# ----------------------------------------------------------------------------------------------------------------------
# reporting helpers
# ----------------------------------------------------------------------------------------------------------------------
def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0])


def show_list(R, n_top=10):
    """indices to print: the known turbulence misses (if present) then the statistics with the largest observed error."""
    idx = [R['names'].index(n) for n in KNOWN_TURB if n in R['names']] if R.get('kind') == 'turbulence' else []
    extra = []
    for i in np.argsort(-np.abs(R['p'] - R['e']) / np.abs(R['e']).clip(1e-300)):
        if int(i) not in idx:
            extra.append(int(i))
        if len(extra) >= n_top:
            break
    return idx + extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['turbulence', 'jets'], required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--seed', type=int, default=900)
    ap.add_argument('--config', nargs='+', default=None, help='substrings the run folder name must contain, e.g. _nt55000_')
    ap.add_argument('--root', type=Path, default=None, help='folder with experiments/ and saved_results/ (default turbulence/ or jets/)')
    ap.add_argument('--out', type=Path, required=True, help='output folder (created; must be new)')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--draws', type=int, default=10)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--h_mults', type=float, nargs='+', default=[1.0, 0.25, 1 / 16, 1 / 64])
    ap.add_argument('--ridges', type=float, nargs='+', default=[1e-4, 1e-2, 1e-6], help='first = the run ridge')
    ap.add_argument('--chain_steps', type=int, default=300)
    ap.add_argument('--chain_h', type=float, nargs='+', default=[1.0, 0.25, 1 / 16, 1 / 64])
    ap.add_argument('--edge_scales', type=float, nargs='*', default=[], help='turbulence only, e.g. 2 4')
    ap.add_argument('--skip', nargs='*', default=[], choices=['a', 'b', 'c', 'd', 'e', 'f'])
    ap.add_argument('--seed_draw', type=int, default=12345)
    ap.add_argument('--batch', type=int, default=None,
                    help='batch size of the Gram / moment passes (default: the run\'s; 5000 needs an 80 GB GPU, use e.g. 1000 on a 16 GB V100; '
                         'only the float32 summation order changes)')
    args = ap.parse_args()

    root = args.root or (ROOT / ('turbulence' if args.kind == 'turbulence' else 'jets'))
    args.out.mkdir(parents=True, exist_ok=False)
    dev = torch.device(args.device)
    t00 = time.time()
    R = load_run(args.kind, args.label, args.seed, args.config, root, dev)
    if args.batch:
        R['batch'] = args.batch
    names, e, p = R['names'], R['e'], R['p']
    ridges = [R['ridge']] + [r for r in args.ridges if abs(r - R['ridge']) > 1e-12]
    H = OneStep(R['pots'], R['n1'], R['M'], R['sigma'], dev, R['batch'], R['ridge'], R['f64'])
    H.target = torch.tensor(e, dtype=R['xs'].dtype, device=dev)
    x0 = R['xs']
    h = R['h_last']
    log(f'last step h = {h:.3e}, sigma = {R["sigma"]}, noise per sample {(2 * h) ** 0.5 * R["sigma"]:.3e}, ridge(s) {ridges}, device {dev}, '
        f'{len(names)} statistics, {args.draws} draws')
    mom0 = H.compute_moments(x0).mean(0).double().cpu().numpy()
    log(f'check: moments of the saved samples vs aux barphi_p: median rel {np.median(np.abs(mom0 - p) / np.abs(p).clip(1e-300)):.1e} '
        f'max {np.max(np.abs(mom0 - p) / np.abs(p).clip(1e-300)):.1e}')

    d_obs = (p - e)                                             # observed error, absolute units
    rel = lambda v: v / np.abs(e).clip(1e-300)
    out = {'names': np.array(names), 'e': e, 'p': p, 'h_last': h}

    # ---- experiments 0, a, b, c : fresh Gram per (h, draw) ----
    allres = {}                                                 # (h_mult, ridge, passes) -> (draws, r)
    chk_max = 0.0
    h_list = [1.0] if 'a' in args.skip else args.h_mults
    for hm in h_list:
        gen = torch.Generator(device=dev); gen.manual_seed(args.seed_draw)
        rs = ridges if hm == 1.0 else ridges[:1]
        mp = 3 if (hm == 1.0 and 'b' not in args.skip) else 1
        if hm == 1.0 and 'c' in args.skip:
            rs = ridges[:1]
        t0 = time.time()
        acc = {}
        for d in range(args.draws):
            o, chk = eval_variants(H, x0, h * hm, gen, rs, mp)
            chk_max = max(chk_max, chk)
            for k, v in o.items():
                acc.setdefault(k, []).append(v)
        for (rg, ps), v in acc.items():
            allres[(hm, rg, ps)] = np.stack(v)
        log(f'  h x {hm:.4g}: {args.draws} draws in {time.time() - t0:.0f} s')
    log(f'check: my solve vs SDE.compute_theta, max relative difference over the draws: {chk_max:.1e}')

    # ---- experiment 0: reproduction ----
    base = allres[(1.0, ridges[0], 1)]                          # (draws, r) residual in absolute units
    mean0, std0 = base.mean(0), base.std(0, ddof=1) if args.draws > 1 else np.zeros_like(base[0])
    out['res0'] = base
    sem = std0 / max(args.draws, 1) ** 0.5
    sig = np.abs(d_obs) > 1e-5 * np.abs(e)                      # above the float32 floor of the moments
    ratio = np.abs(mean0[sig]) / np.abs(d_obs[sig])
    rms0 = np.sqrt((base ** 2).mean(0))
    big = np.abs(rel(d_obs)) > 1e-3
    log('\n=== 0. reproduction: one noise step of the last size + one corrector pass, target = saved last target ===')
    log(f'{"statistic":<44}{"where":<34}{"observed":>10}{"1-step mean":>12}{"std":>10}{"mean/obs":>9}{"sign":>5}   (relative to |e|)')
    for i in show_list(R):
        s = '=' if np.sign(mean0[i]) == np.sign(d_obs[i]) else 'x'
        log(f'{names[i]:<44}{R["meta"][i][:33]:<34}{rel(d_obs)[i]:10.2e}{rel(mean0)[i]:12.2e}{rel(std0)[i]:10.1e}{mean0[i] / d_obs[i]:9.2f}{s:>5}')
    log(f'all statistics with |observed error| > 1e-5 |e| (n={sig.sum()}): median |mean residual| / |observed| = {np.median(ratio):.2f}; '
        f'Spearman(|mean residual|, |observed|) = {spearman(np.abs(mean0[sig]), np.abs(d_obs[sig])):+.2f}; '
        f'rms-over-draws version: median ratio {np.median(rms0[sig] / np.abs(d_obs[sig])):.2f}, Spearman {spearman(rms0[sig], np.abs(d_obs[sig])):+.2f}')
    if big.any():
        sg = np.sign(mean0[big]) == np.sign(d_obs[big])
        clear = np.abs(mean0[big]) > 2 * sem[big]
        log(f'statistics with observed rel error > 1e-3 (n={big.sum()}): sign of the mean residual equals the observed sign for '
            f'{sg.mean():.0%} of them ({(sg & clear).sum()} of {clear.sum()} with |mean| > 2 SEM); median |mean|/|observed| = '
            f'{np.median(np.abs(mean0[big]) / np.abs(d_obs[big])):.2f}')
    known = [names.index(n) for n in KNOWN_TURB if n in names] if args.kind == 'turbulence' else []
    if known:
        log(f'the five known turbulence statistics: observed rel {[f"{x:.1e}" for x in rel(d_obs)[known]]} vs one-step mean '
            f'{[f"{x:.1e}" for x in rel(mean0)[known]]}')
    with open(args.out / f'part1_{args.tag}.csv', 'w') as fh:
        fh.write('index,name,where,target_e,observed_p_minus_e,onestep_mean,onestep_std,ratio_mean_over_observed\n')
        for i in range(len(names)):
            fh.write(f'{i},{names[i]},"{R["meta"][i]}",{e[i]:.6e},{d_obs[i]:.6e},{mean0[i]:.6e},{std0[i]:.6e},{mean0[i] / d_obs[i] if d_obs[i] else float("nan"):.4f}\n')

    # ---- experiment a: h scaling ----
    if 'a' not in args.skip:
        log('\n=== a. step size: mean residual (relative to |e|) after one step of h x mult ===')
        log('  CAVEAT: the start is the saved final walkers, which still carry their own residual; for small h this old residual (not the noise) can set the')
        log('  level. The clean h-dependence is the settled level of experiment d.')
        ms = [m for m in h_list if (m, ridges[0], 1) in allres]
        M_ = np.stack([allres[(m, ridges[0], 1)].mean(0) for m in ms])          # (nh, r)
        S_ = np.stack([allres[(m, ridges[0], 1)].std(0, ddof=1) / args.draws ** 0.5 for m in ms])
        log(f'{"statistic":<44}' + ''.join(f'{"h x " + format(m, ".3g"):>11}' for m in ms) + f'{"exponent":>10}')
        lh = np.log(np.array(ms) * h)
        good = np.all(np.abs(M_) > 3 * S_, axis=0)
        expo = np.full(M_.shape[1], np.nan)
        for i in np.nonzero(good)[0]:
            expo[i] = np.polyfit(lh, np.log(np.abs(M_[:, i])), 1)[0]
        for i in show_list(R):
            log(f'{names[i]:<44}' + ''.join(f'{x:11.2e}' for x in M_[:, i] / abs(e[i])) + f'{expo[i]:10.2f}')
        log(f'statistics with |mean| > 3 SEM at every h: {good.sum()} of {len(good)}; exponent median {np.nanmedian(expo):.2f}, '
            f'quartiles {np.nanpercentile(expo, 25):.2f} .. {np.nanpercentile(expo, 75):.2f}' if good.any() else
            'no statistic above 3 SEM at every h: no exponent')
        out['a_mult'] = np.array(ms); out['a_mean'] = M_; out['a_exponent'] = expo

    # ---- experiment b: passes ----
    if 'b' not in args.skip:
        log('\n=== b. number of corrector passes (chord, Gram of y reused), h = last step ===')
        log(f'{"statistic":<44}{"observed":>10}' + ''.join(f'{"mean, " + str(k) + " pass":>14}' for k in (1, 2, 3)))
        for i in show_list(R):
            log(f'{names[i]:<44}{rel(d_obs)[i]:10.2e}' + ''.join(f'{rel(allres[(1.0, ridges[0], k)].mean(0))[i]:14.2e}' for k in (1, 2, 3)))
        for k in (1, 2, 3):
            r_ = allres[(1.0, ridges[0], k)]
            log(f'  {k} pass(es): median |mean|/|e| {np.median(np.abs(rel(r_.mean(0)))):.2e}, median rms-over-draws/|e| '
                f'{np.median(np.sqrt((r_ ** 2).mean(0)) / np.abs(e).clip(1e-300)):.2e}, max rms/|e| {np.max(np.sqrt((r_ ** 2).mean(0)) / np.abs(e).clip(1e-300)):.2e}')
        out['b_res'] = np.stack([allres[(1.0, ridges[0], k)] for k in (1, 2, 3)])

    # ---- experiment c: ridge ----
    if 'c' not in args.skip:
        log('\n=== c. ridge, one pass, h = last step ===')
        log(f'{"statistic":<44}{"observed":>10}' + ''.join(f'{"ridge " + format(r, ".0e"):>13}' for r in ridges))
        for i in show_list(R):
            log(f'{names[i]:<44}{rel(d_obs)[i]:10.2e}' + ''.join(f'{rel(allres[(1.0, r, 1)].mean(0))[i]:13.2e}' for r in ridges))
        for r in ridges:
            r_ = allres[(1.0, r, 1)]
            log(f'  ridge {r:.0e}: median rms/|e| {np.median(np.sqrt((r_ ** 2).mean(0)) / np.abs(e).clip(1e-300)):.2e}, '
                f'median |mean|/|e| {np.median(np.abs(rel(r_.mean(0)))):.2e}, max |mean|/|e| {np.max(np.abs(rel(r_.mean(0)))):.2e}')
        out['c_ridges'] = np.array(ridges); out['c_res'] = np.stack([allres[(1.0, r, 1)] for r in ridges])

    # ---- experiment d: repeated steps ----
    if 'd' not in args.skip:
        cps = sorted({s for s in (1, 10, 100, 200, 300) if s <= args.chain_steps} | {args.chain_steps})
        settled = [s for s in cps if s >= min(100, args.chain_steps)]      # checkpoints used as "settled" samples
        log(f'\n=== d. repeated steps (noise + one corrector pass each, target fixed), checkpoints {cps}, settled = {settled} ===')
        lvl_rms, lvl_mean, hs = [], [], []
        for hm in args.chain_h:
            gen = torch.Generator(device=dev); gen.manual_seed(args.seed_draw + 7)
            t0 = time.time()
            ch = run_chain(H, x0, h * hm, gen, args.chain_steps, set(cps), ridges[0])
            log(f'h x {hm:.4g} (h = {h * hm:.2e}): {args.chain_steps} steps in {time.time() - t0:.0f} s')
            log(f'{"statistic":<44}' + ''.join(f'{"step " + str(s):>12}' for s in cps))
            for i in show_list(R):
                log(f'{names[i]:<44}' + ''.join(f'{rel(ch[s])[i]:12.2e}' for s in cps))
            log('  median |res|/|e|: ' + '  '.join(f'{s}: {np.median(np.abs(rel(ch[s]))):.2e}' for s in cps)
                + '   max: ' + '  '.join(f'{s}: {np.max(np.abs(rel(ch[s]))):.2e}' for s in cps))
            out[f'd_h{hm:.5g}'] = np.stack([ch[s] for s in cps])
            S_ = np.stack([ch[s] for s in settled])
            lvl_rms.append(np.sqrt((S_ ** 2).mean(0))); lvl_mean.append(S_.mean(0)); hs.append(h * hm)
        out['d_checkpoints'] = np.array(cps)
        if len(hs) >= 3:
            LR = np.stack(lvl_rms) / np.abs(e)[None, :]                       # (nh, r) settled rms level, relative
            LM = np.stack(lvl_mean) / np.abs(e)[None, :]
            expo = np.array([np.polyfit(np.log(hs), np.log(np.maximum(LR[:, i], 1e-30)), 1)[0] for i in range(LR.shape[1])])
            log('\n  settled level (rms over the settled checkpoints, relative to |e|) and exponent of its dependence on h:')
            log(f'{"statistic":<44}' + ''.join(f'{"h x " + format(m, ".3g"):>11}' for m in args.chain_h) + f'{"exponent":>10}')
            for i in show_list(R):
                log(f'{names[i]:<44}' + ''.join(f'{LR[k, i]:11.2e}' for k in range(len(hs))) + f'{expo[i]:10.2f}')
            big_ = LR[0] > 1e-5
            log(f'  statistics with settled level > 1e-5 at the largest h (n={big_.sum()}): exponent median {np.median(expo[big_]):.2f}, '
                f'quartiles {np.percentile(expo[big_], 25):.2f} .. {np.percentile(expo[big_], 75):.2f}')
            out['d_settled_rms'] = LR; out['d_settled_mean'] = LM; out['d_exponent'] = expo; out['d_h'] = np.array(hs)

    # ---- experiment f: float32 evaluation floor of the moments ----
    if 'f' not in args.skip:
        try:
            float_floor(H, x0, e, d_obs, R, out)
        except Exception as ex:
            import traceback
            log(f'\n=== f. FAILED ({type(ex).__name__}: {ex}); the other experiments are unaffected ===')
            traceback.print_exc()

    # ---- experiment e: scalar edges (turbulence only) ----
    if args.kind == 'turbulence' and args.edge_scales and 'e' not in args.skip:
        try:
            edge_experiment(args, R, H, dev, h, out)
        except Exception as ex:                                   # best effort
            import traceback
            log(f'\n=== e. FAILED ({type(ex).__name__}: {ex}); the other experiments are unaffected ===')
            traceback.print_exc()

    np.savez(args.out / f'part1_{args.tag}.npz', **{k: v for k, v in out.items()})
    log(f'\nsaved {args.out}/part1_{args.tag}.npz and .csv; total {time.time() - t00:.0f} s')


def float_floor(H, x0, e, d_obs, R, out):
    """Experiment f: how much of the observed error can be the float32 evaluation noise of the moments themselves?
    (i) |phi_bar_f32(x) - phi_bar_f64(x)| on the saved samples; (ii) |phi_bar_f32(x(1+eps z)) - phi_bar_f32(x)|, eps = float32 resolution
    (6e-8): the change of the float32 result under a perturbation of the input at roundoff level. Both in units of |e|."""
    names = R['names']
    m32 = H.compute_moments(x0).mean(0).double()
    g = torch.Generator(device=x0.device); g.manual_seed(1)
    pert = x0 * (1 + 6e-8 * torch.randn(x0.shape, generator=g, device=x0.device, dtype=x0.dtype))
    d_pert = (H.compute_moments(pert).mean(0).double() - m32).abs().cpu().numpy()
    try:
        m64 = H.compute_moments(x0.double()).mean(0).double()
        d_64 = (m32 - m64).abs().cpu().numpy()
    except Exception as ex:
        log(f'  (float64 evaluation not possible: {type(ex).__name__}: {ex})')
        d_64 = None
    ae = np.abs(e).clip(1e-300)
    obs = np.abs(d_obs)
    log('\n=== f. float32 evaluation floor of the moments on the saved final samples (relative to |e|) ===')
    log(f'{"statistic":<44}{"observed":>10}{"f32 vs f64":>12}{"roundoff pert":>15}')
    for i in show_list(R):
        log(f'{names[i]:<44}{obs[i] / ae[i]:10.2e}' + (f'{d_64[i] / ae[i]:12.2e}' if d_64 is not None else f'{"n/a":>12}') + f'{d_pert[i] / ae[i]:15.2e}')
    fl = d_64 if d_64 is not None else d_pert
    sig_ = obs > 1e-5 * ae
    log(f'  statistics with observed error > 1e-5 |e| (n={sig_.sum()}): median observed {np.median(obs[sig_] / ae[sig_]):.2e}, median f32-vs-f64 '
        f'{np.median(fl[sig_] / ae[sig_]):.2e}, median roundoff-perturbation {np.median(d_pert[sig_] / ae[sig_]):.2e}')
    log(f'  fraction of those whose observed error is below 3x the f32 floor: {np.mean(obs[sig_] < 3 * fl[sig_]):.0%}; '
        f'Spearman(observed, f32 floor) = {spearman(obs[sig_], fl[sig_]):+.2f}')
    out['f_f32_vs_f64'] = d_64 if d_64 is not None else np.array([]); out['f_roundoff_pert'] = d_pert


def edge_experiment(args, R, H, dev, h, out):
    """Edge width of the scalar potentials x scale: statistic scale and targets recomputed from the DATA, same one-step protocol."""
    from check_fit_offline import build_data
    from moment_error_variance import scalar_laplacian
    a = R['a']
    x1, _ = build_data(int(a.get('subseries_len', 512)), R['M'], R['n1'], 1)
    x1 = x1.to(dev)
    log('\n=== e. scalar edge width sw x scale (targets and stat_scale recomputed from the data; walkers = saved final samples) ===')
    log('  protocol: target = phi_bar(data) with the modified potentials, start = saved final walkers, one noise step of the last size + one pass;')
    log('  the scale-1 row is the reference of THIS protocol (its start is not matched to the data, so only ratios between rows are meaningful)')
    scales = [1.0] + [s for s in args.edge_scales if s != 1.0]
    summ = {}
    for f in scales:
        pots = {}
        for nm, pt in R['pots'].items():
            if nm in SCALAR:
                q = copy.deepcopy(pt)
                q.sw = pt.sw * f
                q._compute_stat_scale(x1[:2000].to(torch.float32))
                pots[nm] = q
            else:
                pots[nm] = pt
        He = OneStep(pots, R['n1'], R['M'], R['sigma'], dev, R['batch'], R['ridge'], R['f64'])
        He.target = He.compute_moments(x1).mean(0)
        gen = torch.Generator(device=dev); gen.manual_seed(args.seed_draw)
        rs = []
        for d in range(args.draws):
            o, _ = eval_variants(He, R['xs'], h, gen, [R['ridge']], 1)
            rs.append(o[(R['ridge'], 1)])
        rs = np.stack(rs)
        tgt = He.target.double().cpu().numpy()
        summ[f] = (rs.mean(0) / np.abs(tgt).clip(1e-300), rs.std(0, ddof=1) / np.abs(tgt).clip(1e-300))
        out[f'e_mean_sw{f:g}'] = summ[f][0]
    names = R['names']
    log(f'{"statistic":<44}' + ''.join(f'{"sw x " + format(f, "g"):>12}' for f in scales))
    for i in show_list(R):
        log(f'{names[i]:<44}' + ''.join(f'{summ[f][0][i]:12.2e}' for f in scales))
    # Ito term vs edge width
    log('\n  Ito term I = h sigma^2 <Laplacian phi> / |e| on the saved final samples (|I| of the scalar statistics):')
    log(f'{"statistic":<44}' + ''.join(f'{"sw x " + format(f, "g"):>12}' for f in [0.5] + scales))
    ito = {}
    xs_cpu = R['xs'][:2000].detach().cpu().double()               # scalar_laplacian is a CPU / float64 routine
    x1_cpu = x1[:2000].cpu()
    for f in [0.5] + scales:
        cols, off = [], 0
        for nm, pt in R['pots'].items():
            n = pt.num_coefficients
            if nm in SCALAR:
                q = copy.deepcopy(pt).to('cpu')
                q.sw = q.sw * f
                q._compute_stat_scale(x1_cpu)
                lap = scalar_laplacian(q, xs_cpu).mean(0)
                phi_bar = q.forward(x1_cpu).double().mean(0).numpy()
                cols.append((off, n, h * R['sigma'] ** 2 * lap / np.abs(phi_bar).clip(1e-300)))
            off += n
        v = np.full(len(names), np.nan)
        for o_, n_, arr in cols:
            v[o_:o_ + n_] = arr
        ito[f] = v
        out[f'e_ito_sw{f:g}'] = v
    for i in show_list(R):
        log(f'{names[i]:<44}' + ''.join(f'{abs(ito[f][i]):12.2e}' for f in [0.5] + scales))


if __name__ == '__main__':
    main()
