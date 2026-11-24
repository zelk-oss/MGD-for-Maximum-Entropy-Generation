"""Newton refit of the bimodal model with MALA estimates of the moments (test C2), on Jean Zay.

For beta 0.1 and 0.9, runs 0-2: start at theta_0 = Theta_reg(1; lam_A); at every iteration
  1. MALA on p_theta (persistent chains, started from the final MGD walkers): BURN steps to re-equilibrate after
     the last theta change, then AVG steps with a snapshot every SNAP_EVERY steps;
  2. E_theta[phi_c] and Cov_theta(phi_c) from the pooled snapshots, phi_c = (x, x^2/2, x^3/3, x^4/4);
  3. Newton step theta <- theta - Cov^-1 (E - m_c), halved only if theta_4 would become >= 0.
The exact quantities (quadrature, bimodal_exact.py) are recorded alongside as the referee: exact moment mismatch D,
KL(p* || p_theta), distance to theta_ML (exact Newton, test C1). Data moments m come from the MALA lam_D files of
mala_reg.py (same runs).
Fixed MALA step per beta from the pilot of mala_reg.py: h = 1 (beta 0.1, acceptance 0.62), 0.1 (beta 0.9, 0.58).

Output: one file per (beta, run, N) in <beta dir>/newton_mala/ (iterates + diagnostics + final cloud), and a small
results_bim_theta_beta/newton_mala_summary.pt without the clouds.
Usage, from bimodal_experiment/:
    python newton_mala.py --device cuda [--n_walkers 100000 1000000] [--iters 5] [--burn 3000] [--avg 1000] [--label x]
"""
import argparse
import contextlib
import io
import time
from pathlib import Path

import torch

from Mala import Mala_Sampler
from bimodal_exact import A, kl_exact, moments, newton, theta_star

RES = Path(__file__).resolve().parent / 'results_bim_theta_beta'
SEPT = RES / 'bim_theta_beta_sweep_20260924_152418'
STARTS = {0.1: RES / 'mala_start_beta0.1_1123' / 'beta_0.1000', 0.9: RES / 'mala_start_beta0.9_1123' / 'beta_0.9000'}
H = {0.1: 1.0, 0.9: 0.1}
RUNS, SNAP_EVERY = (0, 1, 2), 50

p = argparse.ArgumentParser()
p.add_argument('--n_walkers', type=int, nargs='+', default=[100_000, 1_000_000])
p.add_argument('--iters', type=int, default=5)
p.add_argument('--burn', type=int, default=3000)
p.add_argument('--avg', type=int, default=1000)
p.add_argument('--device', default='cpu')
p.add_argument('--label', default='')
args = p.parse_args()
dev = torch.device(args.device)
Ad = A.to(dev)
SUFFIX = f'_{args.label}' if args.label else ''


def mala_block(theta, x, h, steps, snap_every=None):
    """`steps` MALA steps on p_theta from x; returns x, acceptance rate, pooled E and Cov of phi_c over snapshots."""
    th = theta.to(dev)
    energy = lambda y: -(y[:, None] ** Ad / Ad) @ th
    score = lambda y: -(y[:, None] ** (Ad - 1)) @ th
    moved, snaps = torch.zeros((), device=dev), []
    for i in range(1, steps + 1):
        with contextlib.redirect_stdout(io.StringIO()):          # Mala.py prints at every call
            x_new = Mala_Sampler(score, energy, x, 1, h)
        moved += (x_new != x).double().mean()
        x = x_new
        if snap_every and i % snap_every == 0:
            snaps.append(x)
    if not snaps:
        return x, float(moved) / steps, None, None
    phi = torch.cat(snaps)[:, None] ** Ad / Ad                   # (N * n_snap, 4)
    E = phi.mean(0)
    C = (phi - E).T @ (phi - E) / (len(phi) - 1)
    return x, float(moved) / steps, E.cpu(), C.cpu()


summary = []
for beta in (0.1, 0.9):
    sel = torch.load(SEPT / f'beta_{beta:.4f}' / 'lam_selection_A.pt', map_location='cpu', weights_only=False)
    C_star_inv = torch.linalg.inv(moments(theta_star(beta))[1])          # D metric, monomial basis (as the notebook)
    out_dir = STARTS[beta] / 'newton_mala'
    out_dir.mkdir(exist_ok=True)
    for k in RUNS:
        m = torch.load(STARTS[beta] / 'mala' / f'run_{k:03d}_mgd_lamD.pt', map_location='cpu',
                       weights_only=False)['data_moments'].double()
        m_c = m / A
        theta0 = sel['theta_final_reg'][k, sel['lams'].index(sel['lam_selected'])].double()
        theta_ml = newton(theta0, m)[-1]                                 # C1, exact
        x_mgd = torch.load(STARTS[beta] / 'mgd' / f'run_{k:03d}.pt', weights_only=False)['x_final'].double().reshape(-1)
        D_ex = lambda th: float((d := moments(th)[0] - m) @ C_star_inv @ d)
        for N in args.n_walkers:
            torch.manual_seed(5000 + 10 * k + len(str(N)))
            g = torch.Generator().manual_seed(1000 + k)
            x = x_mgd[torch.randperm(len(x_mgd), generator=g)[:N]].to(dev)
            theta, rec, t0 = theta0.clone(), [], time.time()
            for it in range(args.iters + 1):
                x, acc_b, _, _ = mala_block(theta, x, H[beta], args.burn)
                x, acc_a, E, C = mala_block(theta, x, H[beta], args.avg, SNAP_EVERY)
                E_mono = E * A                                           # E[x^a] from E[phi_c]
                d = E_mono - m
                r = {'iter': it, 'theta': theta.clone(), 'E_mala': E, 'cov_mala': C, 'acc': (acc_b, acc_a),
                     'D_mala': float(d @ C_star_inv @ d), 'D_exact': D_ex(theta),
                     'KL': float(kl_exact(theta_star(beta), theta)[0]),
                     'dist_theta_ml': float((theta - theta_ml).abs().max())}
                rec.append(r)
                print(f'beta {beta} run {k} N {N:.0e} it {it}: D mala {r["D_mala"]:.2e} exact {r["D_exact"]:.2e}  '
                      f'KL {r["KL"]:.2e}  |theta-theta_ML| {r["dist_theta_ml"]:.1e}  acc {acc_a:.2f}  '
                      f'{time.time() - t0:.0f} s', flush=True)
                if it == args.iters:
                    break
                step = torch.linalg.solve(C, E - m_c)
                eps = 1.0
                while theta[3] - eps * step[3] >= 0:
                    eps /= 2
                theta = theta - eps * step
                rec[-1]['eps'] = eps
            res = {'beta': beta, 'run': k, 'n_walkers': N, 'h': H[beta], 'burn': args.burn, 'avg': args.avg,
                   'snap_every': SNAP_EVERY, 'data_moments': m, 'theta0': theta0, 'theta_ml': theta_ml,
                   'kl_theta_ml': float(kl_exact(theta_star(beta), theta_ml)[0]), 'iterates': rec}
            torch.save({**res, 'x_final': x.float().cpu(), 'rng_state': torch.get_rng_state()},
                       out_dir / f'run_{k:03d}_N{N}{SUFFIX}.pt')
            summary.append(res)
torch.save(summary, RES / f'newton_mala_summary{SUFFIX}.pt')
print(f'saved {len(summary)} entries')
