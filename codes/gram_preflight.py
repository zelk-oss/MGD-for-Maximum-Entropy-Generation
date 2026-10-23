"""Pre-launch collinearity check across potential families.

Scalar_GGD_KRegion.prune_collinear only looks inside one family. A statistic of one family
can still be a near-copy of a combination of statistics of another (2026-10-01: Scalar_morlet
ch 8 region 3 ~ Scalar_psi ch 21 region 3, 1 - |corr| = 6.7e-6, found by hand and removed with
--drop_stats). This does it automatically, before the SDE, so the model set is fixed
and every seed of a model runs the same statistics.

Method. G = E[grad phi_i . grad phi_j] on the data x1 (the Gram the SDE solves against, at t = 1),
accumulated in float64, turned into a correlation matrix C. The statistics are visited in a
fixed ORDER (by family, then by position). Statistic i is dropped if, given the statistics already
kept, its residual fraction 1 - c_i^T C_kk^{-1} c_i is below TOL (the share of its gradient not
explained by the kept ones; the squared R-diagonal entry of an unpivoted QR of the unit-norm
gradient columns, so the same scale as prune_collinear's sqrt(cond_tol) threshold).
Because the order follows the nesting of the models (L_6, Scalar_morlet, Scalar_psi, ...), the
drops among the statistics of a smaller model are identical in the larger ones.

Limits. The Gram is evaluated on the data only, not along the transport (a dependency that only
appears at some t is not seen). The gradients are float32, so residual fractions below ~1e-7 are
noise: TOL should stay well above that. Only families with a drop_columns method (the region
potentials) can lose a statistic; for any other family the check raises instead of ignoring it.
"""
import json
from pathlib import Path

import numpy as np
import torch

# nesting order: M2 = L_6 + Scalar_morlet, M3 adds Scalar_psi. Families not listed come after, in
# the order of the potentials dict.
DEFAULT_ORDER = ('L_6', 'Scalar_morlet_gaussianK', 'Scalar_psi_gaussianK')


def fit_and_save(potentials, x, save_dir):
    """Fit every potential that has a fit and a saved state, as SDE.__init__ does (it then finds the
    saved states and loads them instead of refitting)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, pot in potentials.items():
        if not hasattr(pot, 'fit'):
            continue
        state = save_dir / f'{name}.pt'
        if hasattr(pot, 'is_fitted') and hasattr(pot, 'load_fixed_parameters') and state.exists():
            potentials[name] = type(pot).load_fixed_parameters(state, pot.filters, map_location=x.device)
            continue
        pot.fit(x)
        if hasattr(pot, 'is_fitted'):
            pot.save_fixed_parameters(state)


def stat_labels(potentials):
    """[(family, position in family, readable label)] in the column order of the potentials dict."""
    out = []
    for name, pot in potentials.items():
        flat = getattr(pot, 'active_flat', None)
        for i in range(pot.num_coefficients):
            if flat is not None:
                f = int(flat[i])
                lab = f'{name}[ch{f % pot.J},reg{f // pot.J}]'
            else:
                lab = f'{name}[{i}]'
            out.append((name, i, lab))
    return out


def gradient_gram(potentials, x, batch=250):
    """G = mean over samples of sum_n grad phi_i(n) grad phi_j(n), float64, shape (r, r)."""
    G, n = None, 0
    with torch.no_grad():
        for b in range(0, x.shape[0], batch):
            xb = x[b:b + batch]
            g = torch.cat([p.grad(xb).reshape(xb.shape[0], p.num_coefficients, -1)
                           for p in potentials.values()], dim=1).double()
            Gb = torch.einsum('brt,bst->rs', g, g)
            G = Gb if G is None else G + Gb
            n += xb.shape[0]
    return (G / n).cpu().numpy()


def ordered_elimination(G, order, tol):
    """Visit the statistics in `order` (list of column indices). Returns (dropped, residual, like):
    dropped = columns removed; residual[i] = residual fraction given the kept statistics before
    it (nan for a statistic with a zero gradient, which is skipped and not dropped); like[i] = the
    three kept statistics with the largest |regression coefficient| for a dropped i."""
    diag = np.diag(G)
    dead = diag <= 1e-30 * max(diag.max(), 1e-300)
    d = np.sqrt(np.where(dead, 1.0, diag))
    C = G / np.outer(d, d)
    resid = np.full(G.shape[0], np.nan)
    kept, dropped, like = [], [], {}
    for i in order:
        if dead[i]:
            continue
        if not kept:
            resid[i] = 1.0
            kept.append(i)
            continue
        c = C[kept, i]
        beta = np.linalg.solve(C[np.ix_(kept, kept)], c)
        resid[i] = 1.0 - float(c @ beta)
        if resid[i] < tol:
            dropped.append(i)
            top = np.argsort(-np.abs(beta))[:3]
            like[i] = [(kept[t], float(beta[t])) for t in top]
        else:
            kept.append(i)
    return dropped, resid, like


def run_gram_preflight(potentials, x, tol, logger, save_dir, order=DEFAULT_ORDER, n_samples=2000):
    """Fit, find the cross-family near-copies, drop them and re-save the fitted states (so the SDE
    and the analysis load the dropped model). Returns the report dict, also written to
    <save_dir>/../gram_preflight.json."""
    save_dir = Path(save_dir)
    fit_and_save(potentials, x, save_dir)
    step = max(x.shape[0] // n_samples, 1)
    xs = x[::step][:n_samples]
    labels = stat_labels(potentials)
    names = list(potentials)
    rank = {n: (order.index(n) if n in order else len(order) + names.index(n)) for n in names}
    visit = sorted(range(len(labels)), key=lambda c: (rank[labels[c][0]], labels[c][1]))

    G = gradient_gram(potentials, xs)
    dropped, resid, like = ordered_elimination(G, visit, tol)
    n_dead = int(np.isnan(resid).sum())
    logger.info('Gram pre-flight: %d statistics, %d samples, TOL %g, visiting %s; %d with zero gradient '
                '(skipped), %d to drop', len(labels), xs.shape[0], tol, sorted(names, key=rank.get),
                n_dead, len(dropped))

    by_family = {}
    report_drops = []
    for c in dropped:
        fam, pos, lab = labels[c]
        by_family.setdefault(fam, []).append(pos)
        sim = [(labels[k][2], round(b, 4)) for k, b in like[c]]
        report_drops.append({'stat': lab, 'residual': float(resid[c]), 'like': sim})
        logger.info('  DROP %s  residual %.3e  ~ %s', lab, resid[c], sim)
    for fam, pos in by_family.items():
        if not hasattr(potentials[fam], 'drop_columns'):
            raise NotImplementedError(f'Gram pre-flight wants to drop {len(pos)} statistic(s) of {fam}, '
                                      f'which has no drop_columns (only the region potentials do)')
    for fam, pos in by_family.items():
        potentials[fam].drop_columns(pos)
        potentials[fam].save_fixed_parameters(save_dir / f'{fam}.pt')

    kept = [c for c in visit if not np.isnan(resid[c]) and c not in set(dropped)]
    margin = sorted((float(resid[c]), labels[c][2]) for c in kept if resid[c] < 1.0)[:8]
    logger.info('  smallest residual fractions among the KEPT statistics (margin over TOL): %s',
                [(f'{r:.2e}', l) for r, l in margin])
    report = {'tol': tol, 'n_samples': int(xs.shape[0]), 'n_stats_before': len(labels),
              'n_dropped': len(dropped), 'n_zero_gradient': n_dead, 'dropped': report_drops,
              'smallest_kept_residuals': margin}
    (save_dir.parent / 'gram_preflight.json').write_text(json.dumps(report, indent=1))
    return report
