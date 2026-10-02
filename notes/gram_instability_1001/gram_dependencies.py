"""Near-linear dependencies between statistics, from a run's SAVED corrector Grams (2026-10-01).

    python notes/gram_instability_1001/gram_dependencies.py SYSTEM.pt --root turbulence \
        [--drop L_6_psi] [--nodes 25] [--tol 1e-4]

The system's M_k is G(y_k), the raw Gram of the corrector at node k (--save_reg_system).
At --nodes times, evenly spaced, this takes the Jacobi-scaled live block (unit diagonal,
before the ridge, live set = the run's own mask read from the saved theta_t), optionally
WITHOUT the statistics of the --drop families (a "what if" on the same walkers), and reports:

  (a) the 5 smallest eigenvalues, with the 3 largest components (name and weight) of each
      eigenvector -- which combination of statistics each weak direction is;
  (b) a pivoted Cholesky pass: statistics are taken greedily, each time the one with the
      most information not already carried by those taken. A statistic's residual is
      1 - R^2 of its gradient regressed on the gradients already taken (unit diagonal, so
      it is the squared sine of the angle to their span). Statistics still below --tol when
      everything above it is taken are "dependent": printed with the taken statistics that
      reproduce them (largest regression weights).

Summary over the nodes: every statistic flagged as dependent, at how many nodes, its median
residual and its usual partners. Reads only the --nodes matrices (mmap): login node is fine.
Which member of a dependent group gets flagged depends on the greedy order (it can change
from node to node): read the flagged statistic TOGETHER with its partners as one group.
Removal is NOT automatic: one statistic of each dependency has to be chosen by hand.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for d in (REPO, REPO / 'codes', REPO / 'data'):
    sys.path.insert(0, str(d))

ap = argparse.ArgumentParser()
ap.add_argument('system', type=Path)
ap.add_argument('--root', type=Path, default=Path('turbulence'))
ap.add_argument('--drop', nargs='*', default=[], help='families to remove, e.g. L_6_psi')
ap.add_argument('--nodes', type=int, default=25)
ap.add_argument('--tol', type=float, default=1e-4, help='residual below which a statistic is dependent')
ap.add_argument('--labels', type=Path, default=None, help='(testing) torch file with a list of names')
a = ap.parse_args()

S = torch.load(a.system, map_location='cpu', weights_only=False, mmap=True)
config = a.system.name[:-len('.pt')]
r, t = int(S['num_potentials']), S['t'].double().numpy()

labels = None
if a.labels is not None:
    labels = list(torch.load(a.labels))
else:
    try:
        from codes.find_duplicate_statistics import statistic_labels
        labels, src = statistic_labels(a.root, config)
    except Exception as exc:
        print(f'(no names: {type(exc).__name__}: {exc})')
if labels is None or len(labels) != r:
    print(f'(names unavailable or wrong count; using indices)')
    labels = [f'#{i}' for i in range(r)]
family = np.array([l.split('[')[0] for l in labels])
drop = np.isin(family, a.drop)
if a.drop:
    print(f'dropping {int(drop.sum())} statistics: families {a.drop}')

theta_path = a.root / 'saved_results' / 'lagrange_multipliers' / f'{config}.pt'
time_path = a.root / 'saved_results' / 'sampling_times' / f'{config}.pt'
theta_t = t_fine = None
if theta_path.exists() and time_path.exists():
    theta_t = torch.load(theta_path, map_location='cpu')
    t_fine = np.asarray(torch.load(time_path, map_location='cpu'), dtype=np.float64)
    print('live set per node: the run\'s own mask (saved theta_t == 0)')
else:
    print('live set per node: diag(M_k) > 0 (no saved theta_t found)')


def pivoted_cholesky(A, tol):
    """Greedy: returns (taken indices in order, residual of every index when the pass stops)."""
    n = A.shape[0]
    d = torch.diagonal(A).clone()
    L = torch.zeros(n, n, dtype=A.dtype)
    taken, free = [], torch.ones(n, dtype=torch.bool)
    for i in range(n):
        res = torch.where(free, d, torch.full_like(d, -1.0))
        p = int(res.argmax())
        if float(res[p]) < tol:
            break
        L[:, i] = (A[:, p] - L[:, :i] @ L[p, :i]) / d[p].sqrt()
        d = d - L[:, i] ** 2
        free[p] = False
        taken.append(p)
    return taken, d.clamp_min(0.0), free


idx_nodes = sorted(set(int(i) for i in np.linspace(0, len(t) - 1, a.nodes)))
flag_nodes = defaultdict(list)          # global stat -> [(t, residual, partners)]
weak = []
for k in idx_nodes:
    Mk = S['M'][k].double()
    live = torch.diagonal(Mk) > 0
    if theta_t is not None:
        j = int(np.clip(np.searchsorted(t_fine[1:], t[k]), 0, len(t_fine) - 2))
        live &= theta_t[j] != 0
    live &= ~torch.as_tensor(drop)
    li = live.nonzero().flatten()
    X = Mk[li][:, li]
    dd = torch.diagonal(X).sqrt()
    A = X / (dd[:, None] * dd[None, :]); A = (A + A.T) / 2

    mu, U = torch.linalg.eigh(A)
    comp = []
    for m in range(min(5, len(mu))):
        top = U[:, m].abs().argsort(descending=True)[:3]
        comp.append((float(mu[m]), [(int(li[i]), float(U[i, m])) for i in top]))
    weak.append((k, t[k], float(mu[-1]), comp))

    taken, res, free = pivoted_cholesky(A, a.tol)
    tk = torch.as_tensor(taken)
    for i in free.nonzero().flatten().tolist():
        beta = torch.linalg.lstsq(A[tk][:, tk], A[tk, i:i + 1]).solution.flatten()
        top = beta.abs().argsort(descending=True)[:3]
        partners = [(int(li[tk[q]]), float(beta[q])) for q in top]
        flag_nodes[int(li[i])].append((t[k], float(res[i]), partners))

print(f'\n{config}\nr = {r}, {len(idx_nodes)} nodes (t = {t[idx_nodes[0]]:.4f} .. {t[idx_nodes[-1]]:.4f}), '
      f'tol = {a.tol:g}' + (f', without {a.drop}' if a.drop else ''))

print('\n(a) weakest directions: eigenvalue  [weight x statistic, 3 largest]   (median over nodes of each eigenvalue first)')
mus = np.array([[c[0] for c in w[3]] for w in weak])
print('   median mu_1..mu_5: ' + '  '.join(f'{v:.2e}' for v in np.median(mus, 0))
      + f'   median mu_max: {np.median([w[2] for w in weak]):.2e}')
for k, tk_, mmax, comp in weak[::max(1, len(weak) // 6)]:
    kap = f'{mmax / comp[0][0]:.2e}' if comp[0][0] > 0 else 'undefined (mu_1 <= 0: below the precision of G)'
    print(f'   t = {tk_:.5f}   kappa = {kap}')
    for m, (mu_m, top) in enumerate(comp[:3]):
        print(f'      mu_{m + 1} = {mu_m:9.2e}   ' + '   '.join(f'{w:+.2f} {labels[g]}' for g, w in top))

print(f'\n(b) dependent statistics (residual < {a.tol:g} after pivoted Cholesky), over {len(idx_nodes)} nodes:')
if not flag_nodes:
    print('   none')
for g, rows in sorted(flag_nodes.items(), key=lambda kv: -len(kv[1])):
    res = np.median([r_[1] for r_ in rows])
    pc = Counter(labels[p] for r_ in rows for p, _ in r_[2][:2])
    print(f'   {labels[g]:42s} ({g:3d})  at {len(rows):2d}/{len(idx_nodes)} nodes, median residual {res:.1e}; '
          f'reproduced mostly by: ' + ', '.join(f'{n} ({c})' for n, c in pc.most_common(3)))
