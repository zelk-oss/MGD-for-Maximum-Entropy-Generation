"""Gaussian dimension sweep: is the MGD log p chi-squared? (README.md in this folder)

White Gaussian data x ~ N(0, I_M): log p(x) = -|x|^2/2 + const, so y = -2 (log p - const) = |x|^2 ~ chi2_M.
The model is log p_model(x) = Theta . phi(x) (CODE sign), Theta = Theta_reg at the last node, phi the
order-2 scattering statistics phi_l(x) = mean_u |x * psi_l(u)|^2. log Z is not estimated: the model
log p is centred, y = -2 (lp - mean lp) + M, and compared with chi2_M (mean M, variance 2M).

Exact for this family (Parseval, unnormalised FFT):
    Theta . phi(x) = (1/M) sum_w g(w) |xhat_w|^2,   g(w) = sum_l Theta_l |psihat_l(w)|^2 / M,
so every model is a circulant Gaussian with precision -2 g_eff(w), g_eff(w) = (g(w) + g(-w)) / 2 (x is
real), and the truth is g_eff = -1/2. Hence, with no sampling:
    KL(truth || model) = 1/2 sum_w (-2 g_eff - 1 - log(-2 g_eff))         (inf if some g_eff >= 0)
and the best Theta of the family (maxent: model moments = true moments) minimises the convex dual
    f(Theta) = -1/2 sum_w log(-2 g_eff(w)) - Theta . c,   c_l = (1/M) sum_w |psihat_l(w)|^2 = E_truth[phi_l].
KL(truth || best) is the floor set by the wavelet bank (Morlet Q = 1 does not tile the spectrum flat);
KL(model) - KL(best) is what MGD + Theta_reg add.

Per M, lam and seed (fresh N(0, I) test set; the data x1 and the model samples are scored too):
chi2 test of y (variance / 2M, KS distance to chi2_M), slope and correlation of lp against |x|^2
(target slope -1/2), exact KL, and the seed agreement of lp (as krylov_stopping/turb_d64_analysis.py).

Output (small): <root>/results/summary.json, chi2_lam<lam>.pdf, spectral_weight.pdf, dimension.pdf.

Usage (Jean Zay, from finish.sh):
  python gauss_dim_sweep/analyse.py --root gauss_dim_sweep --ms 4 16 64 --lams 1e-4 1e-3 1e-2 --ridge 1e-4
"""
import argparse
import contextlib
import io
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'codes', ROOT / 'data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from codes.filters_bank import return_Filters                       # noqa: E402
from codes.potentials_builder import get_1d_potentials              # noqa: E402
from codes.resolve_theta_reg import out_path                        # noqa: E402
from codes.utils import normalize                                   # noqa: E402
from data.synthetic_data_generator import load_synthetic_data       # noqa: E402

TERM = 'Scattering_Second_Order_Q1'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--root', type=Path, required=True, help='run_task.sh OUT (experiments/, saved_results/, results/)')
    p.add_argument('--ms', type=int, nargs='+', default=[4, 16, 64])
    p.add_argument('--lams', type=float, nargs='+', default=[1e-4, 1e-3, 1e-2])
    p.add_argument('--ridge', type=float, default=1e-4)
    p.add_argument('--label', default='dimsweep_2ph')
    p.add_argument('--n_test', type=int, default=20000, help='fresh N(0, I_M) samples')
    p.add_argument('--test_seed', type=int, default=12345)
    return p.parse_args()


# ------------------------------------------------------------------ inputs
def find_runs(root, M, label):
    """[(seed, config, system path)] for this M, sorted by seed."""
    J = int(math.log2(M)) - 1
    runs = []
    for f in (root / 'results' / 'reg_system').glob(f'gaussiansynth_*_M{M}_J{J}_*{label}*.pt'):
        runs.append((int(re.search(r'seed_(\d+)', f.stem).group(1)), f.stem, f))
    return sorted(runs)


def theta_end(root, cfg, system, lam, ridge):
    """Theta_reg at the last node (closest to t = 1) and its time."""
    s = torch.load(system, map_location='cpu', weights_only=False, mmap=True)
    mask = 'system' if 'live' in s else 'none'                     # what --mask auto chose
    f = out_path(root / 'results' / 'theta_reg', cfg, lam, ridge, 'moment', 'uniform', mask)
    d = torch.load(f, map_location='cpu', weights_only=False)
    return d['Theta_reg'][-1].double(), float(np.asarray(d['t_reg'])[-1])


def data_x1(cfg, M):
    """x1 exactly as run_SDE.py made it (--data_seed, normalize)."""
    n1 = int(re.search(r'n1_(\d+)', cfg).group(1))
    dseed = int(re.search(r'dseed(\d+)', cfg).group(1))
    with contextlib.redirect_stdout(io.StringIO()):
        raw = load_synthetic_data('mrw', n1, M + 1, lam=0.0, H=0.5, seed=dseed)
    return normalize(raw.diff(dim=-1)).double()


def potential(M):
    J = int(math.log2(M)) - 1
    filters, filters_Phi = return_Filters(M, J, 1, device='cpu', include_phi=True)
    with contextlib.redirect_stdout(io.StringIO()):
        pot = get_1d_potentials([TERM], J, filters, 1, filters_Q=return_Filters(M, J, 1, device='cpu'),
                                filters_Phi=filters_Phi, scalar_param=None, parallel=False)[TERM]
    psi2 = filters[0].double().abs() ** 2                          # (r, M): |psihat_l(w)|^2
    return (lambda x: pot.forward(x.float()).double()), psi2


# ------------------------------------------------------------------ exact Gaussian quantities
def g_eff(theta, psi2):
    """(M,) symmetrised spectral weight of Theta . phi; truth = -1/2."""
    M = psi2.shape[1]
    A = psi2 / M
    A = (A + A[:, (-torch.arange(M)) % M]) / 2
    return theta @ A


def kl_truth(g):
    q = -2 * g
    return float('inf') if (q <= 0).any() else float(0.5 * (q - 1 - torch.log(q)).sum())


def best_in_family(psi2):
    """Maxent Theta of the family for the true law (convex dual, see the docstring)."""
    M = psi2.shape[1]
    A = psi2 / M
    A = ((A + A[:, (-torch.arange(M)) % M]) / 2).numpy()
    c = psi2.sum(1).numpy() / M
    th0 = np.linalg.lstsq(A.T, -0.5 * np.ones(M), rcond=None)[0]   # g = -1/2 in least squares

    def f(th):
        g = th @ A
        if (g >= 0).any():
            return np.inf, np.zeros_like(th)
        return -0.5 * np.log(-2 * g).sum() - th @ c, -0.5 * (A / g).sum(1) - c

    if (th0 @ A >= 0).any():                                       # infeasible start: all-negative Theta
        th0 = -np.ones(A.shape[0])                                 # gives g < 0 wherever the bank covers w
    res = optimize.minimize(f, th0, jac=True, method='L-BFGS-B')
    return torch.as_tensor(res.x), res


# ------------------------------------------------------------------ metrics
def chi2_metrics(lp, x, M):
    """lp (n,) model log p, x (n, 1, M) the points."""
    y = -2 * (lp - lp.mean()) + M
    r2 = x.reshape(len(x), -1).pow(2).sum(1)
    slope, _, corr, _, _ = stats.linregress(r2.numpy(), lp.numpy())
    return dict(var_ratio=float(y.var() / (2 * M)), ks=float(stats.kstest(y.numpy(), stats.chi2(M).cdf).statistic),
                slope=float(slope), corr_r2=float(-corr))


def seed_agreement(L):
    """L (n, S): lp of every seed on the same points."""
    S = L.shape[1]
    pairs = [(a, b) for a in range(S) for b in range(a + 1, S)]
    if not pairs:
        return {}
    pc = [np.corrcoef(L[:, a], L[:, b])[0, 1] for a, b in pairs]
    sc = [stats.spearmanr(L[:, a], L[:, b]).correlation for a, b in pairs]
    Lc = L - L.mean(0)                                             # each seed's own (unknown) log Z removed
    return dict(pearson_median=float(np.median(pc)), pearson_min=float(np.min(pc)),
                spearman_median=float(np.median(sc)),
                spread_ratio=float(np.median(Lc.std(1)) / Lc.mean(1).std()),
                max_spread_nats=float(Lc.std(1).max()))


def main():
    args = parse_args()
    res_dir = args.root / 'results'
    res_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(args.test_seed)
    summary, store = {}, {}

    for M in args.ms:
        runs = find_runs(args.root, M, args.label)
        if not runs:
            print(f'M {M}: no runs found, skipped')
            continue
        phi, psi2 = potential(M)
        x_test = torch.randn(args.n_test, 1, M, generator=gen, dtype=torch.float64)
        x1 = data_x1(runs[0][1], M)
        P_test, P_x1 = phi(x_test), phi(x1)

        # Parseval identity used for g: Theta . phi = (1/M) sum_w g |xhat|^2 (checked, any Theta)
        th = torch.randn(psi2.shape[0], dtype=torch.float64)
        xh2 = torch.fft.fft(x_test[:200, 0]).abs() ** 2
        assert torch.allclose(P_test[:200] @ th, xh2 @ g_eff(th, psi2) / M, rtol=1e-4, atol=1e-6), 'Parseval check'

        th_best, opt = best_in_family(psi2)
        g_best = g_eff(th_best, psi2)
        best = dict(theta=th_best.tolist(), kl=kl_truth(g_best), converged=bool(opt.success),
                    g_min=float(g_best.min()), g_max=float(g_best.max()),
                    test=chi2_metrics(P_test @ th_best, x_test, M))
        xs_path = res_dir / f'xseed_M{M}.pt'
        xseed = torch.load(xs_path, weights_only=False)['selected'] if xs_path.exists() else None
        ent = dict(seeds=[s for s, _, _ in runs], r=int(psi2.shape[0]), n1=int(len(x1)),
                   truth_test=chi2_metrics(-0.5 * x_test.reshape(args.n_test, -1).pow(2).sum(1), x_test, M),
                   family_best=best, xseed_selected=xseed, lam={})
        print(f'\n######## M {M}: r {psi2.shape[0]}, seeds {ent["seeds"]}, family-best KL {best["kl"]:.3g} nats '
              f'(r/2n1 = {psi2.shape[0] / (2 * len(x1)):.2g}), xseed {xseed}')
        for lam in args.lams:
            per, L_test, L_x1, L_xt, R2_xt, gs = {}, [], [], [], [], []
            for seed, cfg, system in runs:
                th, t_end = theta_end(args.root, cfg, system, lam, args.ridge)
                g = g_eff(th, psi2)
                xt = torch.load(args.root / 'saved_results' / 'samples' / f'{cfg}.pt', map_location='cpu').double()
                lp_test, lp_x1, lp_xt = P_test @ th, P_x1 @ th, phi(xt) @ th
                per[seed] = dict(t_end=t_end, theta=th.tolist(), kl=kl_truth(g), kl_minus_best=kl_truth(g) - best['kl'],
                                 g_min=float(g.min()), g_max=float(g.max()),
                                 test=chi2_metrics(lp_test, x_test, M), data=chi2_metrics(lp_x1, x1, M),
                                 samples_minus_data_mean_lp=float(lp_xt.mean() - lp_x1.mean()),
                                 samples_over_data_sd_lp=float(lp_xt.std() / lp_x1.std()))
                L_test.append(lp_test.numpy())
                L_x1.append(lp_x1.numpy())
                L_xt.append(lp_xt.numpy())
                R2_xt.append(xt.reshape(len(xt), -1).pow(2).sum(1).numpy())
                gs.append(g.numpy())
            L_test, L_x1, L_xt = np.stack(L_test, 1), np.stack(L_x1, 1), np.stack(L_xt, 1)
            ent['lam'][f'{lam:g}'] = dict(per_seed=per, seed_agreement=seed_agreement(L_test),
                                          seed_agreement_data=seed_agreement(L_x1))
            # every set is shifted by the same per-seed constant, the mean of log p_theta on the data x1
            # (log Z unknown): a shift of the samples relative to the data stays visible
            store[(M, lam)] = dict(L=L_test, sets={'fresh N(0, I) test set': L_test, 'original data x1': L_x1,
                                                   'MGD samples': L_xt},
                                   shift=L_x1.mean(0), r2_xt=np.stack(R2_xt, 1), g=np.stack(gs))
            print(f'  lam {lam:g}: ' + ' | '.join(
                f'seed {s}: KL {v["kl"]:.3g}, var/2M {v["test"]["var_ratio"]:.3f}, KS {v["test"]["ks"]:.3f}, '
                f'slope {v["test"]["slope"]:.3f}' for s, v in per.items()))
            print(f'           seeds: {ent["lam"][f"{lam:g}"]["seed_agreement"]}')
        store[M] = dict(g_best=g_best.numpy(), x_test=x_test, best_lp=(P_test @ th_best).numpy(),
                        r2_x1=x1.reshape(len(x1), -1).pow(2).sum(1).numpy())
        summary[str(M)] = ent

    (res_dir / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(f'\n-> {res_dir / "summary.json"}')
    ms = [M for M in args.ms if M in store]
    if ms:
        plots(ms, args.lams, store, summary, res_dir)


# ------------------------------------------------------------------ figures
def y_of(L, shift, M):
    """y = -2 (log p_theta - c) + M per seed (columns), c = the seed's mean log p_theta on the data x1."""
    return -2 * (L - shift) + M


def plots(ms, lams, store, summary, res_dir):
    for lam in lams:                         # chi2: one row per M; columns test set / data / samples, then QQ
        names = list(store[(ms[0], lam)]['sets'])
        fig, ax = plt.subplots(len(ms), len(names) + 1, figsize=(4.2 * (len(names) + 1), 3.2 * len(ms)), squeeze=False)
        for i, M in enumerate(ms):
            st = store[(M, lam)]
            r2_of = {names[0]: store[M]['x_test'].reshape(len(st['L']), -1).pow(2).sum(1).numpy(),
                     names[1]: store[M]['r2_x1'], names[2]: st['r2_xt'].ravel()}   # |x|^2 of the same points
            grid = np.linspace(0, np.quantile(r2_of[names[0]], 0.999) * 1.3, 300)
            bins = np.linspace(0, grid[-1], 60)
            for c, name in enumerate(names):
                a = ax[i, c]
                a.hist(r2_of[name], bins, density=True, color='0.8', label='|x|$^2$ of these points')
                Y = y_of(st['sets'][name], st['shift'], M)
                for j, s in enumerate(summary[str(M)]['seeds']):
                    a.hist(Y[:, j], bins, density=True, histtype='step', lw=1, label=f'seed {s}')
                a.plot(grid, stats.chi2(M).pdf(grid), 'k', lw=1.5, label=f'$\\chi^2_{{{M}}}$')
                a.set_title(f'M = {M}: {name}', fontsize=10)
                if i == 0 and c == 0:
                    a.legend(fontsize=7)
                Ysort = np.sort(Y, 0)
                ref = stats.chi2(M).ppf((np.arange(len(Y)) + 0.5) / len(Y))
                for j in range(Y.shape[1]):
                    ax[i, -1].plot(ref, Ysort[:, j], color=f'C{c}', lw=0.8, label=name if j == 0 else None)
            ax[i, -1].plot(ref, ref, 'k--', lw=0.8)
            ax[i, -1].set_xlabel(f'$\\chi^2_{{{M}}}$ quantile')
            ax[i, -1].set_ylabel('y quantile')
            ax[i, -1].set_title('QQ (one line per seed)', fontsize=10)
            if i == 0:
                ax[i, -1].legend(fontsize=7)
        fig.suptitle(f'log p_theta = Theta_reg(t_end) . phi(x), lam = {lam:g};  '
                     f'y = -2 (log p_theta - mean over the data x1) + M, per seed')
        fig.tight_layout()
        fig.savefig(res_dir / f'chi2_lam{lam:g}.pdf')
        plt.close(fig)

    fig, ax = plt.subplots(1, len(ms), figsize=(4.5 * len(ms), 3.5), squeeze=False)
    for i, M in enumerate(ms):                                     # spectral weight g_eff(w) vs -1/2
        w = np.arange(M // 2 + 1) / M
        for k, lam in enumerate(lams):
            for j, g in enumerate(store[(M, lam)]['g']):
                ax[0, i].plot(w, g[:M // 2 + 1], color=f'C{k}', lw=0.8, alpha=0.7, label=f'lam {lam:g}' if j == 0 else None)
        ax[0, i].plot(w, store[M]['g_best'][:M // 2 + 1], 'k--', lw=1.2, label='best in family')
        ax[0, i].axhline(-0.5, color='0.5', lw=0.8, ls=':', label='truth -1/2')
        ax[0, i].set_xlabel('frequency w / M')
        ax[0, i].set_title(f'M = {M}: g_eff(w)')
        ax[0, i].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(res_dir / 'spectral_weight.pdf')
    plt.close(fig)

    names = list(store[(ms[0], lams[0])]['sets'])
    fig, ax = plt.subplots(len(names), len(lams), figsize=(4.5 * len(lams), 3.3 * len(names)), squeeze=False)
    z = np.linspace(-4, 6, 300)
    for r, name in enumerate(names):                               # standardised y for every M, one row per set
        for k, lam in enumerate(lams):
            for i, M in enumerate(ms):
                st = store[(M, lam)]
                y = y_of(st['sets'][name], st['shift'], M).ravel()  # seeds pooled
                ax[r, k].hist((y - M) / math.sqrt(2 * M), np.linspace(-4, 6, 60), density=True, histtype='step',
                              color=f'C{i}', label=f'M = {M}')
                ax[r, k].plot(z, stats.chi2(M).pdf(M + z * math.sqrt(2 * M)) * math.sqrt(2 * M), color=f'C{i}', ls='--', lw=0.8)
            ax[r, k].plot(z, stats.norm.pdf(z), 'k:', lw=0.8, label='N(0, 1)')
            ax[r, k].set_xlabel('(y - M) / sqrt(2M)')
            ax[r, k].set_title(f'{name}, lam {lam:g} (dashed: $\\chi^2_M$)', fontsize=10)
            ax[r, k].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(res_dir / 'dimension.pdf')
    plt.close(fig)
    print(f'-> {res_dir}/chi2_lam*.pdf, spectral_weight.pdf, dimension.pdf')


if __name__ == '__main__':
    main()
