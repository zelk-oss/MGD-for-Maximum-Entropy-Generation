"""MALA on p_reg ∝ exp(Theta_reg(1) . phi_c), phi_c = (x, x^2/2, x^3/3, x^4/4), for beta 0.1 and 0.9, runs 0-2
(seeds 42-44), from two starts: 'mgd' = N walkers drawn from the final MGD cloud x_final, 'gauss' = N(0,1).

lam: --lam A = lam_A of rule A (lam_selection_A.pt); --lam D = lam_D = argmin over the saved lam grid of the moment
mismatch D(lam) = (E_reg[x^a] - m)^T Cov*^-1 (E_reg[x^a] - m), m = moments of the run's data (redrawn exactly as
bimodal_ensemble_reg.py: np seed 42+k, bimodal_utils.bimodal), Cov* = Cov(x^a) under the true density.

Plain MALA (Mala.py), N independent chains (one per walker), fixed step h. h is chosen per run by a short pilot
from the MGD start: the largest h in H_GRID with acceptance >= ACC_MIN; the same h is used for both starts.

Output, one file per (beta, run, start): <beta dir>/mala/run_<k>_<start>[_lamD][_<label>].pt with per-step
acceptance, moments E[x^a] and P(x < x_b) of the cloud, log-spaced cloud snapshots, a thinned trace of N_TRACE
walkers, accepted moves per walker, the final state (cloud, h, RNG state), the exact p_reg references, and metadata.

Usage, from the repo root:
    python bimodal_experiment/mala_reg.py [--lam A|D] [--steps 5000] [--n_walkers 100000] [--device cuda] [--label x]
"""
import argparse
import contextlib
import io
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from Mala import Mala_Sampler
from bimodal_exact import GRID, X_B, log_z_quad, moments, p_below, theta_star, to_user_basis

BM = Path(__file__).resolve().parent
RES = BM / 'results_bim_theta_beta'
SEPT = RES / 'bim_theta_beta_sweep_20260924_152418'                     # Theta_reg(1) over lam, lam_A
STARTS = {0.1: RES / 'mala_start_beta0.1_1123' / 'beta_0.1000',          # x_final of the reruns (same seeds,
          0.9: RES / 'mala_start_beta0.9_1123' / 'beta_0.9000'}          # theta identical to the Sept runs)
RUNS = (0, 1, 2)
H_GRID = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
ACC_MIN, PILOT_STEPS = 0.5, 200
N_TRACE, TRACE_EVERY = 200, 10
N_DATA = 1_000_000

p = argparse.ArgumentParser()
p.add_argument('--lam', choices=['A', 'D'], default='A')
p.add_argument('--steps', type=int, default=5000)
p.add_argument('--n_walkers', type=int, default=100_000)
p.add_argument('--device', default='cpu')
p.add_argument('--label', default='', help='appended to the output file names (smoke tests)')
p.add_argument('--betas', type=float, nargs='+', default=[0.1, 0.9])
p.add_argument('--continue_steps', type=int, default=0,
               help='> 0: continue the saved chains (cloud + RNG state of run_<k>_<start>[_lamD].pt) for this many '
                    'steps instead of starting new ones; writes run_<k>_<start>[_lamD][_<label>]_cont.pt')
args = p.parse_args()
N, S, dev = args.n_walkers, args.continue_steps or args.steps, torch.device(args.device)
SNAP_STEPS = sorted({0, *np.unique(np.round(np.logspace(1, np.log10(S), 11)).astype(int)).tolist()})
A = torch.arange(1, 5, dtype=torch.float64, device=dev)
LAM_SUFFIX = '_lamD' if args.lam == 'D' else ''
SUFFIX = LAM_SUFFIX + (f'_{args.label}' if args.label else '')
def _git(*cmd):
    try:                                                     # no git on the Jean Zay compute nodes
        return subprocess.run(['git', '-C', str(BM), *cmd], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return 'unavailable'


git = _git('rev-parse', '--short', 'HEAD')
dirty = _git('status', '--porcelain', 'Mala.py', 'mala_reg.py', 'bimodal_exact.py')


def data_moments(beta, k):
    """E[x^a] of run k's data, redrawn exactly as bimodal_ensemble_reg.estimate_theta_full."""
    from bimodal_utils import bimodal
    torch.manual_seed(42 + k)
    np.random.seed(42 + k)
    with contextlib.redirect_stdout(io.StringIO()):
        x1 = bimodal(N_DATA, beta)
    return torch.as_tensor((x1[:, None] ** np.arange(1, 5)).mean(0))


def lam_index_D(Th, m, beta):
    """argmin over the lam grid of the moment mismatch D; non-normalisable Theta -> inf."""
    C_inv = torch.linalg.inv(moments(theta_star(beta))[1])
    D = [float((d := moments(th)[0] - m) @ C_inv @ d) if th[3] < 0 else float('inf') for th in Th]
    return int(np.argmin(D)), D


def step(score, energy, x, h):
    with contextlib.redirect_stdout(io.StringIO()):          # Mala.py prints the acceptance at every call
        return Mala_Sampler(score, energy, x, 1, h)


def run_chain(score, energy, x, h, steps):
    acc = torch.empty(steps)
    mom = torch.empty(steps + 1, 4, dtype=torch.float64)
    pb = torch.empty(steps + 1, dtype=torch.float64)
    n_acc = torch.zeros(len(x), dtype=torch.int32, device=x.device)
    snaps, trace = {}, []
    for i in range(steps + 1):
        if i > 0:
            x_new = step(score, energy, x, h)
            moved = x_new != x                               # continuous proposal: equal <=> rejected
            acc[i - 1], n_acc = moved.double().mean().cpu(), n_acc + moved
            x = x_new
        mom[i] = (x[:, None] ** A).mean(0).cpu()
        pb[i] = (x < X_B).double().mean().cpu()
        if i in SNAP_STEPS:
            snaps[i] = x.float().cpu()
        if i % TRACE_EVERY == 0:
            trace.append(x[:N_TRACE].float().cpu())
    return x.cpu(), acc, mom, pb, n_acc.cpu(), snaps, torch.stack(trace)


for beta in args.betas:
    sel = torch.load(SEPT / f'beta_{beta:.4f}' / 'lam_selection_A.pt', map_location='cpu', weights_only=False)
    out_dir = STARTS[beta] / 'mala'
    out_dir.mkdir(exist_ok=True)
    for k in RUNS:
        if args.continue_steps:                                              # resume the saved chains
            for start in ('mgd', 'gauss'):
                base = out_dir / f'run_{k:03d}_{start}{LAM_SUFFIX}.pt'
                r = torch.load(base, map_location='cpu', weights_only=False)
                th_dev = r['theta_code'].to(dev)
                energy = lambda x: -(x[:, None] ** A / A) @ th_dev
                score = lambda x: -(x[:, None] ** (A - 1)) @ th_dev
                torch.set_rng_state(r['rng_state'])
                if dev.type == 'cuda' and r['cuda_rng_state'] is not None:
                    torch.cuda.set_rng_state(r['cuda_rng_state'], dev)
                t0 = time.time()
                x, acc, mom, pb, n_acc, snaps, trace = run_chain(score, energy, r['x_final'].to(dev), r['h'], S)
                print(f'beta {beta} run {k} {start:5s} continued +{S} steps: P(x<x_b) {pb[0]:.4f} -> {pb[-1]:.4f} '
                      f'(exact {r["exact"]["p_below"]:.4f})  acc {acc.mean():.3f}  {time.time() - t0:.0f} s', flush=True)
                torch.save({**{key: r[key] for key in ('beta', 'run', 'data_seed', 'start', 'lam', 'lam_rule', 'theta_code',
                                                       'theta_user', 'x_b', 'h', 'n_walkers', 'exact', 'grid')},
                            'continues': base.name, 'step_offset': r['steps'], 'steps': S, 'device': str(dev),
                            'acc_rate': acc, 'moments': mom, 'p_below': pb, 'snapshots': snaps, 'trace': trace,
                            'trace_every': TRACE_EVERY, 'n_accepted': n_acc, 'x_final': x,
                            'rng_state': torch.get_rng_state(),
                            'cuda_rng_state': torch.cuda.get_rng_state(dev) if dev.type == 'cuda' else None,
                            'git': git, 'uncommitted': dirty, 'script': Path(__file__).read_text()},
                           out_dir / f'run_{k:03d}_{start}{SUFFIX}_cont.pt')
            continue
        Th = sel['theta_final_reg'][k].double()                              # (n_lam, 4), Theta_reg(1) per lam
        m, D_scan = None, None
        if args.lam == 'A':
            lam = sel['lam_selected']
        else:
            m = data_moments(beta, k)
            i_D, D_scan = lam_index_D(Th, m, beta)
            lam = sel['lams'][i_D]
        theta = Th[sel['lams'].index(lam)]                                  # code basis
        assert theta[3] < 0, f'beta {beta} run {k}: Theta_reg(1) not normalisable'
        th_dev = theta.to(dev)
        energy = lambda x: -(x[:, None] ** A / A) @ th_dev                  # p_reg ∝ exp(-energy)
        score = lambda x: -(x[:, None] ** (A - 1)) @ th_dev                 # d energy / dx
        E_ex, C_ex = moments(theta)
        exact = {'E': E_ex, 'cov': C_ex, 'p_below': p_below(theta), 'log_z': float(log_z_quad(theta)[0])}

        x_final = torch.load(STARTS[beta] / 'mgd' / f'run_{k:03d}.pt', weights_only=False)['x_final'].double().reshape(-1)
        g = torch.Generator().manual_seed(1000 + k)
        idx = torch.randperm(len(x_final), generator=g)[:N]
        inits = {'mgd': x_final[idx], 'gauss': torch.randn(N, generator=g, dtype=torch.float64)}

        torch.manual_seed(2000 + k)
        pilot = []
        for h in H_GRID:                                                    # pilot from the MGD start
            *_, a_ = run_chain(score, energy, inits['mgd'].to(dev), h, PILOT_STEPS)[:2]
            pilot.append((h, float(a_[PILOT_STEPS // 2:].mean())))
        ok = [h for h, a_ in pilot if a_ >= ACC_MIN]
        h = max(ok) if ok else min(H_GRID)
        print(f'beta {beta} run {k}: lam_{args.lam} {lam:g}  pilot ' + '  '.join(f'h={h_:g}:{a_:.2f}' for h_, a_ in pilot)
              + f'  -> h = {h:g}', flush=True)

        for s_id, (start, x0) in enumerate(inits.items()):
            seed = 3000 + 10 * k + s_id
            torch.manual_seed(seed)
            t0 = time.time()
            x, acc, mom, pb, n_acc, snaps, trace = run_chain(score, energy, x0.to(dev), h, S)
            se = x.std() / N ** 0.5
            print(f'   {start:5s}: acc {acc.mean():.3f}  P(x<x_b) {pb[0]:.4f} -> {pb[-1]:.4f} (exact {exact["p_below"]:.4f})'
                  f'  E[x] {mom[-1, 0]:.4f}±{se:.4f} (exact {E_ex[0]:.4f})  {time.time() - t0:.0f} s', flush=True)
            torch.save({
                'beta': beta, 'run': k, 'data_seed': 42 + k, 'start': start, 'lam': lam, 'lam_rule': args.lam,
                'data_moments': m, 'D_scan': D_scan, 'lams': sel['lams'],
                'theta_code': theta, 'theta_user': to_user_basis(theta), 'x_b': X_B,
                'h': h, 'pilot_h_acc': pilot, 'acc_min': ACC_MIN, 'pilot_steps': PILOT_STEPS,
                'n_walkers': N, 'steps': S, 'chain_seed': seed, 'subsample_idx': idx if start == 'mgd' else None,
                'x_start': x0.float(), 'device': str(dev),
                'acc_rate': acc, 'moments': mom, 'p_below': pb,                 # per step (moments: E[x^a], a = 1..4)
                'snapshots': snaps, 'trace': trace, 'trace_every': TRACE_EVERY, 'n_accepted': n_acc,
                'x_final': x, 'rng_state': torch.get_rng_state(),               # to continue the chain
                'cuda_rng_state': torch.cuda.get_rng_state(dev) if dev.type == 'cuda' else None,
                'exact': exact, 'grid': (float(GRID[0]), float(GRID[-1]), len(GRID)),
                'git': git, 'uncommitted': dirty, 'script': Path(__file__).read_text(),
            }, out_dir / f'run_{k:03d}_{start}{SUFFIX}.pt')
