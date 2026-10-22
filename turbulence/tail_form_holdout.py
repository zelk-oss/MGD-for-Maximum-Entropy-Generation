"""Held-out comparison of candidate tail forms for Scalar_GGD_KRegion (offline, CPU).

For one wavelet channel, the potential's outer region imposes a single statistic and
MGD solves its multiplier; the tail SHAPE is fixed by the fit before MGD. This script
compares candidate outer-region forms by solving the per-channel 1-D MaxEnt dual
(moments of half A of the data) and scoring the resulting density on held-out half B.

Candidates (cuts at fixed quantiles of half A; hard windows, unlike the smooth sigmoid
windows of the production potential):
  K4_f02  cuts q.5/.9/.99, alpha by truncated-GGD MLE with scale floor 0.02*median
          (current production floor)
  K4      same, scale floor 0.001*median
  A       extra cut at q.999 (two outer exponents), floor 0.001
  B       one outer region, statistic |z|^alpha * (1 + log(|z|/u))^gamma (second-order,
          log-corrected stretched exponential; gamma = 0 is exactly K4), alpha and gamma
          by truncated MLE on A
  C       one outer region, statistic (log(|z|/z0))^2, z0 = median|z| of A (log-normal,
          Castaing-like tail; no tail-fitted shape parameter)
  D       K4 plus a MASS statistic <w_K> on the outer region, alpha by truncated MLE
          without scale floor. With one multiplier per region, theta_K must set both the
          region's mass and its shape; the extra statistic decouples them, so the MaxEnt
          outer density is the truncated-GGD fit itself.
  BD      B plus the same mass statistic.

Splits: windows are cut into contiguous blocks so bursts are not split across halves.

The per-channel solve treats channels as independent: justified for the near-orthogonal
Morlet Q=1 bank, only indicative for the overcomplete Q=3 bank.

Usage (repo root):
    TURBULENCE_1D_DATA_PATH=$PWD/data/data_files/turbulence_1d_period.pt \
        python turbulence/tail_form_holdout.py [--n_splits 20] [--sanity_only]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from potentials.potentials_1d import Scalar_GGD_KRegion   # noqa: E402

QS_EVAL = np.array([0.99, 0.999, 0.9999, 0.99995])
RESID_TOL = 1e-6
TAIL_MASS_TOL = 1e-6      # model mass beyond 2*max|z| must be negligible vs 5e-5 quantile
CANDIDATES = ('K4_f02', 'K4', 'A', 'B', 'C', 'D', 'BD')


# ── GGD fit with a configurable scale floor (production code, only the floor changed) ──
def _ggd_with_floor(floor_frac):
    class _G(Scalar_GGD_KRegion):
        @staticmethod
        def _scale_floor(ah, floor_frac=floor_frac, min_floor=1e-10):
            med = float(np.median(ah)) if ah.size else 0.0
            return max(floor_frac * med, min_floor)
    return _G


GGD_F02, GGD_F001, GGD_F0 = _ggd_with_floor(0.02), _ggd_with_floor(0.001), _ggd_with_floor(1e-12)


def fit_alpha(a, lo, hi, G, fast, rng, max_n=200_000, bounds=(0.2, 8.0)):
    seg = a[(a >= lo) & (a < hi)]
    if seg.size < 5:
        return 1.0
    if seg.size > max_n:
        seg = rng.choice(seg, max_n, replace=False)
    fn = G._fit_ggd_truncated_fast if fast else G._fit_ggd_truncated
    return float(fn(seg, lo, hi, bounds)[0])


# ── 1-D MaxEnt dual, damped Newton ──
def make_grid(a, n=60_000):
    lo = max(float(a[a > 0].min()), 1e-14)
    return np.concatenate([[0.0], np.geomspace(lo, 20.0 * float(a.max()), n)])


def solve_dual(Bg, g, md, maxit=300):
    """min_t  t.md + log Z(t),  Z = int exp(-t.B).  Bg, md normalized so md = 1.
    Returns (t, density on grid, max relative moment residual, Hessian)."""
    K = Bg.shape[0]

    def evaluate(t):
        V = t @ Bg
        c = V.min()
        w = np.exp(-(V - c))
        Zn = np.trapezoid(w, g)
        p = w / Zn
        E = np.trapezoid(Bg * p, g, axis=1)
        return t @ md + np.log(Zn) - c, E, p

    best = None
    for t in (np.zeros(K), np.ones(K)):
        f, E, p = evaluate(t)
        for _ in range(maxit):
            grad = md - E
            if np.abs(grad).max() < 1e-12:
                break
            C = np.trapezoid(Bg[:, None, :] * Bg[None, :, :] * p, g, axis=2) - np.outer(E, E)
            mu = 1e-12 * max(np.trace(C), 1e-300)
            try:
                d = np.linalg.solve(C + mu * np.eye(K), -grad)
            except np.linalg.LinAlgError:
                break
            step, ok = 1.0, False
            for _ls in range(60):
                tn = t + step * d
                fn, En, pn = evaluate(tn)
                if np.isfinite(fn) and fn <= f + 1e-4 * step * grad @ d:
                    t, f, E, p, ok = tn, fn, En, pn, True
                    break
                step *= 0.5
            if not ok:
                break
        res = np.abs(md - E).max()
        if best is None or res < best[2]:
            C = np.trapezoid(Bg[:, None, :] * Bg[None, :, :] * p, g, axis=2) - np.outer(E, E)
            best = (t, p, res, C)
        if res < RESID_TOL:
            break
    return best


def density_summary(g, p, a_tail_B, a_B, u):
    cdf = np.concatenate([[0.0], np.cumsum((p[1:] + p[:-1]) / 2 * np.diff(g))])
    cdf /= cdf[-1]
    q_model = np.interp(QS_EVAL, cdf, g)
    q_data = np.quantile(a_B, QS_EVAL)
    logp = np.log(np.maximum(np.interp(a_tail_B, g, p), 1e-300))
    tail_mass = 1.0 - np.interp(2.0 * a_B.max(), g, cdf)
    return q_model / q_data, float(logp.mean()), float(tail_mass)


# ── candidate statistics ──
def build_stats(cand, aA, rng):
    """Return list of (lo, hi, func) and the fitted tail shape parameter."""
    q50, q90, q99, q999 = np.quantile(aA, [0.5, 0.9, 0.99, 0.999])
    inner_edges = [(0.0, q50), (q50, q90), (q90, q99)]
    inner = [(lo, hi, ('pow', fit_alpha(aA, lo, hi, GGD_F001, True, rng))) for lo, hi in inner_edges]
    if cand == 'K4_f02':
        a_out = fit_alpha(aA, q99, np.inf, GGD_F02, False, rng)
        return inner + [(q99, np.inf, ('pow', a_out))], a_out
    if cand == 'K4':
        a_out = fit_alpha(aA, q99, np.inf, GGD_F001, False, rng)
        return inner + [(q99, np.inf, ('pow', a_out))], a_out
    if cand == 'A':
        a1 = fit_alpha(aA, q99, q999, GGD_F001, False, rng)
        a2 = fit_alpha(aA, q999, np.inf, GGD_F001, False, rng)
        return inner + [(q99, q999, ('pow', a1)), (q999, np.inf, ('pow', a2))], a2
    if cand == 'B':
        a0 = fit_alpha(aA, q99, np.inf, GGD_F001, False, rng)
        alpha, gamma = fit_alpha_gamma(aA[aA >= q99], q99, a0)
        return inner + [(q99, np.inf, ('powlog', (q99, alpha, gamma)))], gamma
    if cand == 'C':
        z0 = float(np.median(aA))
        return inner + [(q99, np.inf, ('lognorm', z0))], z0
    if cand == 'D':
        a_out = fit_alpha(aA, q99, np.inf, GGD_F0, False, rng, bounds=(0.05, 8.0))
        return inner + [(q99, np.inf, ('pow', a_out)), (q99, np.inf, ('ind', None))], a_out
    if cand == 'BD':
        a0 = fit_alpha(aA, q99, np.inf, GGD_F001, False, rng)
        alpha, gamma = fit_alpha_gamma(aA[aA >= q99], q99, a0)
        return inner + [(q99, np.inf, ('powlog', (q99, alpha, gamma))),
                        (q99, np.inf, ('ind', None))], gamma
    raise ValueError(cand)


def eval_func(func, x):
    kind, par = func
    if kind == 'pow':
        return x ** par
    if kind == 'powlog':
        u, alpha, gamma = par
        return x ** alpha * (1.0 + np.log(np.maximum(x, u) / u)) ** gamma
    if kind == 'lognorm':
        return np.log(np.maximum(x, 1e-300) / par) ** 2
    if kind == 'ind':
        return np.ones_like(x)
    raise ValueError(kind)


def fit_alpha_gamma(t, u, a0):
    """Truncated MLE of p(z) ~ exp(-th * z^a * (1+log(z/u))^gam) on [u, inf).
    a > 0 keeps it normalizable for any gam; gam = 0 is the K4 outer region."""
    gg = np.geomspace(u, 20.0 * t.max(), 40_000)
    lt, lg = 1.0 + np.log(t / u), 1.0 + np.log(gg / u)
    xt, xg = t / u, gg / u                                   # scale-free, better conditioned

    def nll(p):
        th, a, gam = np.exp(p[0]), np.exp(p[1]), p[2]
        if not (0.05 <= a <= 8.0 and -5.0 <= gam <= 5.0):
            return 1e300
        Vg = th * xg ** a * lg ** gam
        c = Vg.min()
        logZ = np.log(np.trapezoid(np.exp(-(Vg - c)), gg)) - c
        return th * (xt ** a * lt ** gam).sum() + t.size * logZ

    best = None
    for g0 in (-1.0, 0.0, 1.0):
        r = minimize(nll, [0.0, np.log(a0), g0], method='Nelder-Mead',
                     options=dict(maxiter=6000, xatol=1e-8, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    return float(np.exp(best.x[1])), float(best.x[2])


def run_candidate(cand, aA, aB, rng):
    stats, shape = build_stats(cand, aA, rng)
    g = make_grid(np.concatenate([aA, aB]))
    K = len(stats)
    Bg = np.zeros((K, g.size))
    md = np.zeros(K)
    for k, (lo, hi, func) in enumerate(stats):
        mg = (g >= lo) & (g < hi)
        Bg[k, mg] = eval_func(func, g[mg])
        mA = (aA >= lo) & (aA < hi)
        md[k] = eval_func(func, aA[mA]).sum() / aA.size
    scale = np.where(np.abs(md) > 0, np.abs(md), 1.0)
    Bg /= scale[:, None]
    md = md / scale
    t, p, res, C = solve_dual(Bg, g, md)
    ev = np.linalg.eigvalsh(C)
    cond = float(ev.max() / max(ev.min(), 1e-300))
    u = np.quantile(aA, 0.99)
    ratios, ll, tail_mass = density_summary(g, p, aB[aB >= u], aB, u)
    return dict(ratios=ratios, ll=ll, res=float(res), cond=cond, shape=shape, tail_mass=tail_mass)


# ── data ──
def load_coefficients():
    from utils import split_periodize_reshape
    from ortho_wavelet.ReadyToUseWavelets import DefineWavelet
    from data.data_loader import load_turbulence_1d
    from filters_bank import return_Filters
    W = DefineWavelet('Db', m=3, device='cpu')
    D = split_periodize_reshape(load_turbulence_1d().cpu(), 512)
    D = W.decompose(D)[1]                                   # 512 -> 256, as run_SDE
    x1 = D[:10000]
    Dn = ((D - x1.mean()) / x1.std()).to(torch.float32)    # x1's affine map, as run_SDE
    out = {}
    for Q in (1, 3):
        f = return_Filters(256, 8, Q)
        out[Q] = torch.fft.ifft(f * torch.fft.fft(Dn)).real.numpy()[:, :-1]   # drop phi
    return out


def block_split(n_windows, n_blocks, rng):
    blocks = np.array_split(np.arange(n_windows), n_blocks)
    order = rng.permutation(n_blocks)
    A = np.concatenate([blocks[i] for i in order[: n_blocks // 2]])
    B = np.concatenate([blocks[i] for i in order[n_blocks // 2:]])
    return A, B


def sanity_test():
    rng = np.random.default_rng(0)
    n, a_true = 400_000, 0.4
    a = rng.gamma(1.0 / a_true, 1.0, n) ** (1.0 / a_true)
    g = make_grid(a)
    Bg = (g ** a_true)[None, :]
    md = np.array([(a ** a_true).mean()])
    Bg, md = Bg / md[:, None], md / md
    t, p, res, _ = solve_dual(Bg, g, md)
    cdf = np.concatenate([[0.0], np.cumsum((p[1:] + p[:-1]) / 2 * np.diff(g))])
    cdf /= cdf[-1]
    ratio = np.interp([0.99, 0.999, 0.9999], cdf, g) / np.quantile(a, [0.99, 0.999, 0.9999])
    ok = res < RESID_TOL and np.all(np.abs(ratio - 1) < 0.03)
    print(f'SANITY  GGD(0.4), one region: ratios {np.round(ratio, 4)}  resid {res:.1e}  '
          f'-> {"PASS" if ok else "FAIL"}')
    return ok


def main(n_splits, n_blocks, channels, candidates=CANDIDATES):
    if not sanity_test():
        raise SystemExit('sanity test failed: solver output not interpretable')
    Z = load_coefficients()
    rng = np.random.default_rng(1)
    rows = []
    for Q, ch in channels:
        z = Z[Q][:, ch]                                       # (windows, M)
        print(f'\n=== Q={Q} ch={ch} ===', flush=True)
        for s in range(n_splits):
            iA, iB = block_split(z.shape[0], n_blocks, rng)
            aA, aB = np.abs(z[iA]).ravel(), np.abs(z[iB]).ravel()
            for cand in candidates:
                r = run_candidate(cand, aA, aB, rng)
                rows.append(dict(Q=Q, ch=ch, split=s, cand=cand, **r))
        summarize([r for r in rows if r['Q'] == Q and r['ch'] == ch], candidates)
    out = HERE / 'figures' / 'data' / 'tail_form_holdout.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as fh:
        fh.write('Q,ch,split,cand,r99,r999,r9999,r99995,tail_ll,resid,cond,shape,tail_mass\n')
        for r in rows:
            fh.write(f"{r['Q']},{r['ch']},{r['split']},{r['cand']},"
                     + ','.join(f'{v:.6g}' for v in r['ratios'])
                     + f",{r['ll']:.6g},{r['res']:.3g},{r['cond']:.3g},{r['shape']:.6g},"
                       f"{r['tail_mass']:.3g}\n")
    print(f'\nper-split rows: {out}')


def summarize(rows, candidates=CANDIDATES):
    best_ll = {}
    for s in {r['split'] for r in rows}:
        ok = [r for r in rows if r['split'] == s and r['res'] < RESID_TOL]
        if ok:
            best_ll[s] = max(r['ll'] for r in ok)
    print(f"{'cand':>7} {'conv':>5} | {'q.99':>13} {'q.999':>13} {'q.9999':>13} {'q.99995':>13} | "
          f"{'dLL tail':>9} {'log10 cond':>10} | {'shape med [IQR]':>22}")
    for cand in candidates:
        rs = [r for r in rows if r['cand'] == cand]
        ok = [r for r in rs if r['res'] < RESID_TOL and r['tail_mass'] < TAIL_MASS_TOL]
        if not ok:
            print(f'{cand:>7} {0:>2}/{len(rs):<2} | no converged split')
            continue
        R = np.array([r['ratios'] for r in ok])
        dll = np.array([r['ll'] - best_ll.get(r['split'], r['ll']) for r in ok])
        cond = np.log10([r['cond'] for r in ok])
        sh = np.array([r['shape'] for r in ok])

        def mi(x):
            q1, m, q3 = np.percentile(x, [25, 50, 75])
            return f'{m:6.3f}[{q3 - q1:5.3f}]'
        print(f'{cand:>7} {len(ok):>2}/{len(rs):<2} | {mi(R[:, 0]):>13} {mi(R[:, 1]):>13} '
              f'{mi(R[:, 2]):>13} {mi(R[:, 3]):>13} | {np.median(dll):9.4f} {np.median(cond):10.1f} | '
              f'{np.median(sh):9.4f} [{np.subtract(*np.percentile(sh, [75, 25])):.4f}]')
    print('  ratio = model/held-out quantile, median [IQR] over splits; '
          'dLL = mean tail log-lik minus best candidate in the same split (0 = best)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_splits', type=int, default=20)
    ap.add_argument('--n_blocks', type=int, default=20)
    ap.add_argument('--sanity_only', action='store_true')
    ap.add_argument('--candidates', nargs='+', default=list(CANDIDATES), choices=CANDIDATES)
    a = ap.parse_args()
    if a.sanity_only:
        sys.exit(0 if sanity_test() else 1)
    main(a.n_splits, a.n_blocks, [(1, 0), (1, 7), (3, 0), (3, 3), (3, 7)], tuple(a.candidates))
