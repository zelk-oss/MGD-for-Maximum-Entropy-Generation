"""Choose lam for the regularised estimator of the bimodal beta runs with rule A (cross-dataset).

Rule A (lam_selection_criterion/bimodal_lamtest.py, add_xseed_rules, decided 2026-11-18), here in
its cross-DATASET form: the K runs of one beta each have their own dataset x1 (seed 42 + k) and
their own SDE noise, so the data term of run b is an independent noisy copy of the data term of
run a. For every run a and every lam of the grid:

    L_ab(lam) = sum_{k: t_k >= t_end} w_k [ 1/2 Theta_a,k(lam)^T Mr_b,k Theta_a,k(lam) - b_b,k^T Theta_a,k(lam) ]

with Mr = sym(M) + ridge * diag(sym(M)) and w_k the trapezoid weights in t of the t_reg grid
(thomas_batched.data_term / ridged / codes.sde_routines.trapezoid_node_weights, imported, not
re-implemented). Pooled loss P(lam) = sum_a sum_{b != a} L_ab(lam); lam_A = argmin P over the grid,
lam = 0 included. By linearity of L in (Mr_b, b_b) (ridged() is linear in M),
    sum_{b != a} L_ab = L(Theta_a; Msum - M_a, bsum - b_a),
so the cost is O(K), not O(K^2). Only the t >= t_end nodes of M, b and Theta are kept in memory.

Also stored: per-run argmins (argmin_lam sum_{b != a} L_ab), a leave-one-run-out jackknife of lam_A
(P without every pair that contains run c, from the column sums sum_{a != c} L_ac, computed with
the second moments sum_a Theta_a Theta_a^T, again O(K)), and the oracle check of select_lambda.py
(Fisher-weighted energy MSE at t = 1 against theta*, uses the truth, reporting only).

Solver: the same batched float64 block-Thomas as select_lambda.py (codes/lam_selection.py), i.e. the
scalar port's 'legacy' energy weights (data weight a_k = 1 per node, smoothness lam/dt^2 per cell),
same ridge in the solve and in the score.

Output: <dir>/lam_selection_A.pt (lam_selection.pt is never touched; with --max_runs N the file is
lam_selection_A_maxruns<N>.pt). An existing output is not overwritten without --overwrite.

Usage (from bimodal_experiment/):
    python select_lambda_A.py results_bim_theta_beta/<sweep>/beta_0.5000 [...] \
        [--lams 0 1e-8 ...] [--t_end 0.99] [--ridge 0] [--chunk 5] [--threads N] [--max_runs N]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import sde_routines_scalar_reg  # noqa: F401  (puts the project root, codes/ and data/ on sys.path)
from lam_selection import solve_regularised_thomas_batched
from codes.sde_routines import trapezoid_node_weights

_LSC = Path(__file__).resolve().parent.parent / 'lam_selection_criterion'
if str(_LSC) not in sys.path:
    sys.path.append(str(_LSC))
import thomas_batched as tb  # noqa: E402  (data_term, ridged: the rule-A loss of bimodal_lamtest.py)

DEFAULT_LAMS = [0, 1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5,
                1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1]
N_CHECK = 3            # runs whose pooled row / column sums are checked against explicit pair sums


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('dirs', nargs='+', type=Path, help='beta results dirs')
    p.add_argument('--lams', nargs='+', type=float, default=DEFAULT_LAMS)
    p.add_argument('--t_end', type=float, default=0.99, help='score on the nodes t_reg >= t_end')
    p.add_argument('--ridge', type=float, default=0.0, help='M_k += ridge * diag(M_k), solve and score')
    p.add_argument('--chunk', type=int, default=5, help='runs solved together')
    p.add_argument('--threads', type=int,
                   default=int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or None)
    p.add_argument('--max_runs', type=int, default=None, help='smoke test: only the first N runs')
    p.add_argument('--overwrite', action='store_true', help='replace an existing output file')
    return p.parse_args()


def energy_forms(delta, S):
    """Mean over runs of delta^T S delta; delta (K, ..., r), S (r, r). As select_lambda.py."""
    return torch.einsum('k...a,ab,k...b->...', delta, S, delta) / delta.shape[0]


def edge_of(lam, lams):
    """Where lam sits on the grid: 'zero' (no regularisation), 'min_pos' / 'max' (grid edges) or ''."""
    pos = [l_ for l_ in lams if l_ > 0]
    if lam == 0.0:
        return 'zero'
    if lam == max(lams):
        return 'max'
    if pos and lam == min(pos):
        return 'min_pos'
    return ''


def out_path(d, max_runs):
    return d / ('lam_selection_A.pt' if max_runs is None else f'lam_selection_A_maxruns{max_runs}.pt')


def select(d, lams, ridge, chunk, t_end, max_runs):
    exp = torch.load(d / 'experiment_K_runs.pt', map_location='cpu', weights_only=False)
    runs = list(exp['runs'])
    if max_runs is not None:
        runs = runs[:max_runs]
    files = [d / 'reg_system' / f'run_{k:03d}.pt' for k in runs]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f'[{d}] systems missing for aggregated runs: {missing}')
    if len(files) != exp['config']['K']:
        print(f'[{d}] WARNING: {len(files)} runs of K={exp["config"]["K"]}')
    K, L = len(files), len(lams)
    if K < 2:
        raise SystemExit(f'[{d}] rule A needs at least 2 runs, got {K}')
    target = exp['target_theta_phi'].double()
    cov_phi = exp['cov_phi'].double()

    # Node alignment. The solver returns Theta on every system node t[0..n-1]; as forward_regularised
    # and select_lambda.py we drop row 0, so t_reg = t[1:] and Theta_reg[j] sits at t[j + 1]. The data
    # term of node t[j + 1] is (M[j + 1], b[j + 1]) of the SAME system, i.e. Theta row and (M, b) row
    # carry the same system index (as in bimodal_lamtest.add_xseed_rules, where Theta and S['M'],
    # S['b'] share the node index of system['t'], with no row dropped there). We therefore select the
    # scoring nodes as system indices i = j + 1 for j with t_reg[j] >= t_end, take Theta[..., i, :],
    # M[i], b[i], and the trapezoid weight of t_reg[j] (weights of the full t_reg grid, restricted to
    # the window, as bimodal_lamtest does with trapezoid_node_weights(t)[nodes]).
    t = j_tail = i_tail = w_tail = None
    th_tail, M_tail, b_tail, finals, resid = [], [], [], [], []
    t0 = time.time()
    for i0 in range(0, K, chunk):
        systems = [torch.load(f, map_location='cpu', weights_only=False) for f in files[i0:i0 + chunk]]
        if t is None:
            t = systems[0]['t'].double()
            t_reg = t[1:].numpy()
            j_tail = np.flatnonzero(t_reg >= t_end)
            if len(j_tail) == 0:
                raise SystemExit(f'[{d}] no node with t_reg >= {t_end}')
            i_tail = torch.as_tensor(j_tail + 1)                        # system index of each t_reg node
            w_tail = trapezoid_node_weights(t_reg)[j_tail]
            assert torch.equal(t[i_tail], torch.as_tensor(t_reg[j_tail]))
        for s, f in zip(systems, files[i0:i0 + chunk]):
            assert torch.equal(s['t'].double(), t), f'{f}: time grid differs from run 0'
        stack = lambda key: torch.stack([s[key] for s in systems])
        Th, res = solve_regularised_thomas_batched(t.numpy(), stack('M'), stack('G'), stack('b'),
                                                   stack('c'), lams, ridge=ridge)   # (R, L, n, r)
        M_tail.append(stack('M')[:, i_tail].double())                   # (R, n_tail, r, r)
        b_tail.append(stack('b')[:, i_tail].double())                   # (R, n_tail, r)
        del systems
        resid.append(res)
        finals.append(Th[:, :, -1].clone())                             # (R, L, r), theta at t = 1
        th_tail.append(Th[:, :, i_tail].clone())                        # (R, L, n_tail, r)
        del Th
        print(f'[{d.name}] runs {i0 + 1}-{min(i0 + chunk, K)}/{K} solved ({time.time() - t0:.0f} s)')

    Th_t = torch.cat(th_tail)                                           # (K, L, n_tail, r)
    M_t, b_t = torch.cat(M_tail), torch.cat(b_tail)                     # (K, n_tail, r, r), (K, n_tail, r)
    resid = torch.cat(resid)                                            # (K, L)
    n_tail = len(j_tail)
    nodes = np.arange(n_tail)
    term = lambda M, b, Th: tb.data_term({'M': M, 'b': b, 'ridge': ridge}, Th, nodes, w_tail)  # -> (L,)

    # row sums R_a = sum_{b != a} L_ab = L(Theta_a; Msum - M_a, bsum - b_a)
    Msum, bsum = M_t.sum(0), b_t.sum(0)
    row = torch.stack([term(Msum - M_t[a], bsum - b_t[a], Th_t[a]) for a in range(K)])    # (K, L)
    # column sums C_c = sum_{a != c} L_ac = sum_k w_k [1/2 <Mr_c,k, Q_k - Th_c Th_c^T> - b_c,k . (T1_k - Th_c)]
    Q = torch.einsum('alki,alkj->lkij', Th_t, Th_t)                     # (L, n_tail, r, r)
    T1 = Th_t.sum(0)                                                    # (L, n_tail, r)
    wt = torch.as_tensor(w_tail)
    col = []
    for c in range(K):
        Mr_c = tb.ridged(M_t[c], ridge)                                 # (n_tail, r, r)
        Qc = Q - torch.einsum('lki,lkj->lkij', Th_t[c], Th_t[c])
        quad = 0.5 * torch.einsum('kij,lkij->lk', Mr_c, Qc)
        lin = torch.einsum('ki,lki->lk', b_t[c], T1 - Th_t[c])
        col.append(((quad - lin) * wt).sum(-1))
    col = torch.stack(col)                                              # (K, L)

    # check against explicit pair sums with data_term, for the first N_CHECK runs
    chk = 0.0
    for a in range(min(N_CHECK, K)):
        r_ex = sum(term(M_t[b], b_t[b], Th_t[a]) for b in range(K) if b != a)
        c_ex = sum(term(M_t[a], b_t[a], Th_t[b]) for b in range(K) if b != a)
        for x, y in ((row[a], r_ex), (col[a], c_ex)):
            chk = max(chk, float((x - y).abs().max() / y.abs().max().clamp_min(1e-300)))

    P = row.sum(0)                                                      # (L,) pooled loss
    chk = max(chk, float((P - col.sum(0)).abs().max() / P.abs().max().clamp_min(1e-300)))  # sum rows = sum cols
    lam_A = lams[int(torch.argmin(P))]
    per_run = [lams[int(torch.argmin(row[a]))] for a in range(K)]
    jack = [lams[int(torch.argmin(P - row[c] - col[c]))] for c in range(K)] if K >= 3 else []

    # oracle check (truth), as select_lambda.py
    theta_final_reg = torch.cat(finals)                                 # (K, L, r)
    mgd = exp['theta_final_mgd'].double()[:K] if max_runs is not None else exp['theta_final_mgd'].double()
    err_reg, err_mgd = theta_final_reg - target, mgd - target
    dev_reg = theta_final_reg - theta_final_reg.mean(0)
    mse_E = energy_forms(err_reg, cov_phi)

    dt = np.diff(t.numpy())
    out = {
        'lams': list(lams), 'ridge': ridge, 't_end': t_end, 'n_nodes': n_tail,
        't_tail': torch.as_tensor(t_reg[j_tail]), 'w_tail': torch.as_tensor(w_tail),
        'pooled_loss': P, 'row_loss': row, 'col_loss': col,
        'lam_selected': lam_A, 'edge': edge_of(lam_A, lams),
        'per_run_argmin': per_run, 'jackknife': jack, 'check_max_rel_diff': chk,
        'lam_oracle': lams[int(torch.argmin(mse_E))],
        'mse_energy_reg_centered': mse_E,
        'var_energy_reg_centered': energy_forms(dev_reg, cov_phi) * K / (K - 1),
        'mse_energy_mgd_centered': energy_forms(err_mgd, cov_phi),
        'cr_bound_energy_centered': exp.get('cr_bound_energy_centered'),
        'theta_final_reg': theta_final_reg, 'target_theta_phi': target,
        'config': exp['config'], 'runs': runs, 'max_runs': max_runs, 'solve_residual': resid,
        'solver_weights': 'legacy (scalar port): data a_k = 1 per node, smoothness lam/dt^2 per cell',
        'grid_dt_rel_spread': float(dt.max() / dt.min() - 1),
    }
    dest = out_path(d, max_runs)
    tmp = dest.with_name(dest.name + '.tmp')
    torch.save(out, tmp)
    tmp.replace(dest)
    print(f'[{d.name}] wrote {dest}')
    return out


def report(d, o):
    jk = o['jackknife']
    print(f"\n=== {d}  beta={o['config']['beta']}  K={o['theta_final_reg'].shape[0]}  "
          f"t_end={o['t_end']} ({o['n_nodes']} nodes)  ridge={o['ridge']}")
    print(f"solver weights: {o['solver_weights']}; grid dt max/min - 1 = {o['grid_dt_rel_spread']:.1e}")
    print(f"lam_A = {o['lam_selected']:g}" + (f"   [grid edge: {o['edge']}]" if o['edge'] else '')
          + f"   lam_oracle = {o['lam_oracle']:g}   max solve residual {o['solve_residual'].max():.1e}"
          f"   pooled-sum check max rel diff {o['check_max_rel_diff']:.1e}")
    if jk:
        vals, cnt = np.unique(jk, return_counts=True)
        print(f"jackknife (leave one run out): min {min(jk):g} max {max(jk):g}; "
              + ', '.join(f'{v:g} x{c}' for v, c in zip(vals, cnt)))
    vals, cnt = np.unique(o['per_run_argmin'], return_counts=True)
    print('per-run argmins: ' + ', '.join(f'{v:g} x{c}' for v, c in zip(vals, cnt)))
    P0 = float(o['pooled_loss'][o['lams'].index(0.0)])
    print(f"{'lam':>8} | {'P - P(0)':>12} | MSE E (centered) | Var E (centered)")
    for i, lam in enumerate(o['lams']):
        print(f"{lam:8.0e} | {float(o['pooled_loss'][i]) - P0:12.4e} | {o['mse_energy_reg_centered'][i]:.3e}"
              f"        | {o['var_energy_reg_centered'][i]:.3e}")
    print(f"{'MGD':>8} | {'':12s} | {o['mse_energy_mgd_centered']:.3e}")


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    lams = sorted(set(float(x) for x in args.lams) | {0.0})     # lam = 0 always in the argmin
    for d in args.dirs:
        dest = out_path(d, args.max_runs)
        if dest.exists() and not args.overwrite:
            print(f'[{d}] {dest.name} exists, skipped (--overwrite to replace)')
            continue
        report(d, select(d, lams, args.ridge, args.chunk, args.t_end, args.max_runs))


if __name__ == '__main__':
    main()
