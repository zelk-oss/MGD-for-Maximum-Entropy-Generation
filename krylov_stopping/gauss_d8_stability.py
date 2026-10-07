"""Robustness of ridge vs early stopping (jac_minres) on the d = 8 Gaussian with near-copy statistics (2026-10-07).

phi_ij = x_i x_j plus copies psi_ij = s(x_i) s(x_j), s(u) = u + delta u^3 (delta = perturbation strength), r = 72.
Truth: theta_phi = C_t^-1 part, theta_psi = 0 (population). eta: ridge 1e-7 for every method (so methods differ only
in the theta rule); G accumulated in float64 except in the 'f32' condition.
Conditions (no retuning): delta in {0.1, 0.01, 0.001} (float64 G) and delta = 0.01 with float32 G.
Methods: direct (lam 0), ridge 1e-4 / 1e-2 / 1, ito_cg, jac_minres. N = M = 1e4, sigma^2 = 1, nt = 1000, 10 seeds.
Per run: three theta estimates at the end (last step, mean of the last 50 steps, Theta_reg for lam in {1e-3, 1e-2} x
data ridge in {0, 1e-4, 1e-2}); stability R_k / R_base (SDEKrylov monitoring); samples' KL, held-out statistics.
Usage (repo root):
  smoke (local):   python krylov_stopping/gauss_d8_stability.py --out DIR --smoke
  JZ array task:   python krylov_stopping/gauss_d8_stability.py --out DIR --task $SLURM_ARRAY_TASK_ID --n_tasks 10 --no_combine
  combine:         python krylov_stopping/gauss_d8_stability.py --out DIR --combine_only
"""
# %% imports and configuration
import argparse, contextlib, io, itertools, math, os, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().parent
for p in (ROOT, ROOT / 'codes', ROOT / 'data', ROOT / 'krylov_stopping'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from codes.sde_krylov import SDEKrylov                                               # noqa: E402
from codes.resolve_theta_reg import build_inputs                                    # noqa: E402
from codes import lam_select_cv as lsc                                              # noqa: E402
from gauss_d8_krylov import Distorted, target_cov, pop_cov, sym_from_phi, Theta_mgd, kl_gauss  # noqa: E402
from lam_selection_criterion.gauss_d8_lamtest import Quadratic, interp_second_moments          # noqa: E402

METHODS = {'direct': dict(regularization=0.0), 'ridge1e-4': dict(regularization=1e-4),
           'ridge1e-2': dict(regularization=1e-2), 'ridge1': dict(regularization=1.0),
           'ito_cg': dict(regularization=0.0, theta_solver='ito_cg'),
           'jac_minres': dict(regularization=0.0, theta_solver='jac_minres')}
CONDITIONS = {'d0.1': (0.1, True), 'd0.01': (0.01, True), 'd0.001': (0.001, True), 'd0.01_f32': (0.01, False)}
REG = [(lam, rr) for lam in (1e-3, 1e-2) for rr in (0.0, 1e-4, 1e-2)]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--d', type=int, default=8)
    p.add_argument('--N', type=int, default=10000)
    p.add_argument('--sigma2', type=float, default=1.0)
    p.add_argument('--nt', type=int, default=1000)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(900, 910)))
    p.add_argument('--conditions', nargs='+', default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument('--methods', nargs='+', default=list(METHODS), choices=list(METHODS))
    p.add_argument('--task', type=int, default=None)
    p.add_argument('--n_tasks', type=int, default=None)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--combine_only', action='store_true')
    p.add_argument('--no_combine', action='store_true')
    p.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or None)
    a = p.parse_args(argv)
    if a.smoke:
        a.N, a.nt, a.seeds = 200, 100, a.seeds[:2]
    return a


# %% one run
def held_out(x):
    """Statistics NOT among the constraints (truth: Gaussian 4th-order / extreme values)."""
    z = x.reshape(x.shape[0], -1).double()
    return torch.stack([(z[:, 0] ** 4).mean(), (z[:, 0] ** 2 * z[:, 1] ** 2).mean(),
                        (z.pow(2).sum(1) ** 2).mean(), z.abs().amax(1).mean()])


def run_one(args, C1, eps, g64, seed, method):
    d, N = args.d, args.N
    g = torch.Generator().manual_seed(seed)
    x1 = (torch.randn(N, d, generator=g, dtype=torch.float64) @ torch.linalg.cholesky(
        torch.as_tensor(C1)).T).float().reshape(N, 1, d)
    t = np.linspace(0.0, 1.0, args.nt + 1)
    pot = Quadratic(d)
    iu = pot.iu
    torch.manual_seed(seed + 10_000)
    kw = dict(METHODS[method]); solver = kw.pop('theta_solver', 'ridge')
    with contextlib.redirect_stdout(io.StringIO()):
        sde = SDEKrylov(x1, N, N, t, math.sqrt(args.sigma2), {'quad': pot, 'dup': Distorted(d, eps)}, N,
                        device='cpu', interpolant='Cos', potentials_save_dir=args.out / 'unused_potentials_dir',
                        solve_float64=True, theta_solver=solver, eta_regularization=1e-7, g64=g64,
                        monitor=True, diagnostics=False, **kw)
    sysp = args.out / 'systems' / f'{method}_{eps:g}_{int(g64)}_{seed}.pt'
    sysp.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        xT, phi_I, phi_x, eta, theta, dH, _, _ = sde.forward_regularised(
            lam=0.0, reg_solver='thomas', reg_mode='moment', reg_system_path=sysp, solve_reg=False)
    wall = time.perf_counter() - t0
    nt, h = args.nt, np.diff(t)
    theta = theta.double()                                         # (nt, 72) code sign, row k targets t_{k+1}
    system = torch.load(sysp, map_location='cpu', weights_only=False)
    sysp.unlink()
    est = {'last': theta[-1], 'mean50': theta[-50:].mean(0)}
    t_reg = system['t'][1:].double().numpy()                       # resolve drops the first node, as forward_regularised
    for lam, rr in REG:
        a_ = SimpleNamespace(schedule='uniform', ridge=rr, dim=None, n_walkers=None, sigma=None, mask='auto',
                             results_root=None, live_floor=0.0)
        inp = build_inputs(system, 'gauss', a_, 'moment')
        Th, _ = lsc.solve(inp, lam)
        est[f'reg_lam{lam:g}_ridge{rr:g}'] = Th[-1].double()
    t_end = {'last': 1.0, 'mean50': 1.0, **{k: float(system['t'][-1]) for k in est if k.startswith('reg')}}

    # stability: R_k / R_base, R_base = h s2 tr(C_t^-1) / (2d) + r / (N d)
    mon = sde.monitor_log()
    S = mon['step']
    R = S[:, 1].numpy()
    Cp = pop_cov(C1, t[1:])
    trinv = torch.linalg.inv(Cp).diagonal(dim1=1, dim2=2).sum(1).numpy()
    r = sde.num_potentials
    R_base = h * args.sigma2 * trinv / (2 * d) + r / (N * d)
    ratio = R / R_base
    half = len(ratio) // 2
    slope = np.polyfit(np.arange(half, len(ratio)), np.log(np.maximum(R[half:], 1e-300)), 1)[0]
    mis = mon['mis']
    S_emp = interp_second_moments(sde.x_0, sde.x_1, [1.0])[0]
    C_x = sym_from_phi(phi_x[-1:, :36], iu, d)[0]
    return dict(method=method, eps=eps, g64=g64, seed=seed, est={k: v for k, v in est.items()}, t_end=t_end,
                ratio=ratio, R=R, R_base=R_base, j=S[:, 2].numpy(), t=t[1:], ratio_median=float(np.median(ratio)),
                ratio_p99=float(np.percentile(ratio, 99)), ratio_max=float(ratio.max()),
                spikes=int((ratio > 10).sum()), slope_logR=float(slope),
                mis_std=float(mis[:, 1].std()) if len(mis) else float('nan'),
                mis_max=float(mis[:, 1].max()) if len(mis) else float('nan'), mis_r=int(mis[0, 2]) if len(mis) else 0,
                weak=mon['eig'][:, [0] + list(range(5, 15))].numpy() if len(mon['eig']) else None,
                held_out=held_out(xT).numpy(), KL_pop=kl_gauss(C_x, torch.as_tensor(C1)), KL_emp=kl_gauss(C_x, S_emp),
                S_int=float((h * dH[:nt].double().numpy()).sum()),
                S_exact_emp=0.5 * float(torch.logdet(S_emp) - torch.logdet(interp_second_moments(sde.x_0, sde.x_1, [0.0])[0])),
                wall_per_step=wall / nt)


def run_point(args, cond, seed):
    eps, g64 = CONDITIONS[cond]
    C1 = target_cov(args.d, 0)
    out = args.out / 'runs'
    out.mkdir(parents=True, exist_ok=True)
    for m in args.methods:
        f = out / f'{cond}_{m}_seed{seed}.pt'
        if f.exists():
            continue
        t0 = time.time()
        res = run_one(args, C1, eps, g64, seed, m)
        res['cond'] = cond
        torch.save(res, f)
        print(f'{f.name}: {time.time() - t0:.0f} s, R/R_base median {res["ratio_median"]:.2f} max {res["ratio_max"]:.2f}, '
              f'spikes {res["spikes"]}, KL {res["KL_pop"]:.2e}', flush=True)


# %% combine: tables
def load(out):
    R = [torch.load(f, weights_only=False) for f in sorted((out / 'runs').glob('*.pt'))]
    by = {}
    for r in R:
        by.setdefault(r['cond'], {}).setdefault(r['method'], []).append(r)
    return by


def ms(v):
    v = np.asarray(v, dtype=float)
    return f'{np.nanmean(v):.3g} ± {np.nanstd(v, ddof=1) / np.sqrt(len(v)):.2g}' if len(v) > 1 else f'{v[0]:.3g}'


def est_metrics(runs, key, C1):
    """Cross-seed spread (split, sum: trace of the variance over the 36 pairs), bias of the seed mean, copies size."""
    T = torch.stack([r['est'][key] for r in runs])                 # (seeds, 72), code sign
    phi, psi = T[:, :36], T[:, 36:]
    a = np.pi * runs[0]['t_end'][key] / 2
    P = torch.linalg.inv(torch.as_tensor(np.cos(a) ** 2 * np.eye(8) + np.sin(a) ** 2 * C1))
    iu = Quadratic(8).iu
    sum_mean = Theta_mgd((phi + psi).mean(0, keepdim=True), iu, 8)[0]
    copies = [float(Theta_mgd(p[None], iu, 8)[0].norm() / P.norm()) for p in psi]
    return dict(split_var=float((phi - psi).var(0).sum()), sum_var=float((phi + psi).var(0).sum()),
                bias=float((sum_mean - P).norm() / P.norm()), copies=copies)


def tables(by, C1):
    lines = []
    keys = ['last', 'mean50'] + [f'reg_lam{lam:g}_ridge{rr:g}' for lam, rr in REG]
    for cond in [c for c in CONDITIONS if c in by]:
        lines += [f'\n## condition {cond}\n',
                  '| method | estimate | split var | sum var | bias (seed mean) | copies size | KL x1e3 | entropy bias | mean j | j=0 |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for m in [m for m in METHODS if m in by[cond]]:
            runs = by[cond][m]
            jj = np.concatenate([r['j'] for r in runs])
            jtxt = (f'{np.mean(jj[jj >= 0]):.2f} | {np.mean(jj[jj >= 0] == 0):.0%}' if (jj >= 0).any() else '- | -')
            for k in keys:
                e = est_metrics(runs, k, C1)
                lines.append(f'| {m} | {k} | {e["split_var"]:.3g} | {e["sum_var"]:.3g} | {e["bias"]:.3g} | {ms(e["copies"])} | '
                             f'{ms([1e3 * r["KL_pop"] for r in runs])} | {ms([r["S_int"] - r["S_exact_emp"] for r in runs])} | {jtxt} |')
    lines += ['\n## stability (R_k / R_base over the run; mean ± SE over seeds)\n',
              '| condition | method | median | p99 | max | spikes (>10) | slope log R (last 50%) | mismatch std | mismatch max | held-out std over seeds |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for cond in [c for c in CONDITIONS if c in by]:
        for m in [m for m in METHODS if m in by[cond]]:
            runs = by[cond][m]
            ho = np.std(np.stack([r['held_out'] for r in runs]), 0, ddof=1)
            lines.append(f'| {cond} | {m} | {ms([r["ratio_median"] for r in runs])} | {ms([r["ratio_p99"] for r in runs])} | '
                         f'{ms([r["ratio_max"] for r in runs])} | {ms([r["spikes"] for r in runs])} | '
                         f'{ms([r["slope_logR"] for r in runs])} | {ms([r["mis_std"] for r in runs])} | '
                         f'{ms([r["mis_max"] for r in runs])} | {" / ".join(f"{v:.3g}" for v in ho)} |')
    return '\n'.join(lines)


# %% combine: plots
def plots(by, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    col = {'direct': '#0b0b0b', 'ridge1e-4': '#8fb8ea', 'ridge1e-2': '#2a78d6', 'ridge1': '#1a4f8f',
           'ito_cg': '#eb6834', 'jac_minres': '#1baf7a'}
    conds = [c for c in CONDITIONS if c in by]
    for key, ylab, fn in (('ratio', 'R_k / R_base', 'stability_R_ratio.png'), ('j', 'j_k (θ iterations)', 'stability_j.png')):
        fig, ax = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.4), sharey=True, squeeze=False)
        for a, c in zip(ax[0], conds):
            for m in [m for m in METHODS if m in by[c]]:
                y = np.mean([r[key] for r in by[c][m]], 0)
                if key == 'j' and not (y >= 0).any():
                    continue
                w = 10
                ys = np.convolve(y, np.ones(w) / w, mode='valid')
                a.plot(by[c][m][0]['t'][w - 1:], ys, color=col[m], lw=1.4, label=m)
            a.set_title(c); a.set_xlabel('t'); a.grid(alpha=0.25, lw=0.5)
            if key == 'ratio':
                a.set_yscale('log'); a.axhline(1, color='0.4', lw=0.8, ls=':')
        ax[0, 0].set_ylabel(ylab)
        h_, l_ = ax[0, 0].get_legend_handles_labels()
        fig.tight_layout(); fig.legend(h_, l_, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=8)
        fig.savefig(out / fn, dpi=130, bbox_inches='tight'); plt.close(fig)
    keys = ['last', 'mean50', 'reg_lam0.001_ridge0', 'reg_lam0.001_ridge0.0001', 'reg_lam0.01_ridge0.01']
    fig, ax = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.6), sharey=True, squeeze=False)
    C1 = target_cov(8, 0)
    for a, c in zip(ax[0], conds):
        ms_ = [m for m in METHODS if m in by[c]]
        for i, k in enumerate(keys):
            v = [est_metrics(by[c][m], k, C1)['split_var'] for m in ms_]
            a.bar(np.arange(len(ms_)) + (i - 2) * 0.16, v, 0.16, label=k)
        a.set_xticks(range(len(ms_))); a.set_xticklabels(ms_, rotation=45, ha='right', fontsize=8)
        a.set_yscale('log'); a.set_title(c); a.grid(alpha=0.25, lw=0.5, axis='y')
    ax[0, 0].set_ylabel('cross-seed variance of θ_φ − θ_ψ (trace)')
    h_, l_ = ax[0, 0].get_legend_handles_labels()
    fig.tight_layout(); fig.legend(h_, l_, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=8)
    fig.savefig(out / 'split_variance.png', dpi=130, bbox_inches='tight'); plt.close(fig)


# %% main
def main(argv=None):
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.combine_only:
        grid = list(itertools.product(args.conditions, args.seeds))
        todo = grid if args.task is None else grid[args.task::args.n_tasks or len(grid)]
        for cond, seed in todo:
            print(f'=== {cond} seed {seed}', flush=True)
            run_point(args, cond, seed)
    if not args.no_combine:
        by = load(args.out)
        txt = tables(by, target_cov(args.d, 0))
        print(txt)
        (args.out / 'results.md').write_text('# gauss_d8_stability\n' + txt + '\n')
        plots(by, args.out)
        print(f'wrote {args.out}/results.md and plots')


if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    main()
