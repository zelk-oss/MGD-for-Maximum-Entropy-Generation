"""Theta_reg with upper bounds on some coordinates: Theta_{k,i} <= u_{k,i} (2026-10-08).

Use: force the L_6 coefficients of Theta_reg negative at every node, Theta_{k,i} <= -eps_{k,i},
so that exp(Theta . phi) is confined along every direction the L_6 bank sees (theta_sign_constraint/).

The time-regularised energy E(Theta) = 1/2 Theta^T A Theta - f^T Theta (A SPD, the block-tridiagonal
system of SDE._solve_regularised_thomas) is minimised under the bounds. Three loops (method=):

  'pnewton' (default) projected Newton with backtracking: start from the free solve clipped to the
      bounds; each round holds the entries at their bound whose gradient does not point below it,
      solves exactly for the rest (one pinned Thomas solve), steps towards that solution, clips back
      onto the bounds and keeps the step only if E decreases. Every iterate is feasible, E decreases
      every round, it cannot cycle. Synthetic tests: same solution as the exact loop / L-BFGS-B.
  'exact' plain primal active set (add every violated entry, release every pinned entry whose
      gradient points below the bound, all at once). On turbulence it OSCILLATED: job 781019
      (2026-10-08), seed 900, rounds 1-6 each added and released 40-50 thousand entries, pinned count
      bouncing 72-162 thousand; if it stops at max_iter its last iterate can violate the bounds.
      (On synthetic systems it converges, 14-18 rounds at n = 2000-4000.)
  'add_only' never releases: feasible and fast, but holds entries that would rather sit below their
      bound; on a synthetic n = 400 system its energy was 64% worse than the optimum.

eps_from_sigma: eps_{k,i} = eps_nats / sd_{k,i}, sd_{k,i} = sqrt(Sigma_k[i, i]) (the spread of phi_i on
the walkers at node k, from a moment-mode system). Then the bound means "Theta_i phi_i varies by at
least eps_nats nats (in the confining direction) over one standard deviation of phi_i": the same small
weight in log p at every scale and every time, whatever the units of phi_i (the raw L_6 coefficients differ
by ~7 orders of magnitude between scales).
"""
import numpy as np
import torch

from codes.sde_routines import SDE


def eps_from_sigma(Sigma, cols, eps_nats, sd_floor=1e-300):
    """(n, len(cols)) eps_{k,i} = eps_nats / sqrt(Sigma_k[i,i]), float64."""
    cols = torch.as_tensor(cols, dtype=torch.long)
    sd = torch.stack([torch.diagonal(S.double())[cols] for S in Sigma]).clamp_min(0).sqrt()
    return eps_nats / sd.clamp_min(sd_floor)


def solve_upper_bounded(solver_self, t, M, C, b, v, lam, cols, upper, ridge=0.0, mode='moment',
                        dim=None, schedule='uniform', live=None, max_iter=30, tol_viol=1e-9,
                        tol_grad=1e-9, method='pnewton', release=None, log=print):
    """Minimise the Theta_reg energy with Theta[:, cols] <= upper (n, len(cols)).

    method: 'pnewton' (default) projected Newton with backtracking, see solve_pnewton: every iterate
            feasible, energy decreases every round, cannot cycle;
            'exact'   primal active set (adds and releases all at once; oscillated on turbulence);
            'add_only' never releases (fast, feasible, but can be far from the optimum).
    release: legacy switch (True = 'exact', False = 'add_only'); overrides method if given.
    Returns (Theta (n, r) float64, info dict). Arguments as SDE._solve_regularised_thomas (called
    unbound with solver_self). Bounded entries must not be dead in `live`. info['converged'] False
    means the returned Theta is not certified optimal (and, for 'exact', may violate bounds).
    """
    if release is not None:
        method = 'exact' if release else 'add_only'
    if method == 'pnewton':
        return solve_pnewton(solver_self, t, M, C, b, v, lam, cols, upper, ridge=ridge, mode=mode, dim=dim,
                             schedule=schedule, live=live, max_iter=max_iter, tol_grad=tol_grad, log=log)
    release = method == 'exact'
    t = np.asarray(t, dtype=np.float64)
    n, r = len(t), solver_self.num_potentials
    cols = torch.as_tensor(cols, dtype=torch.long)
    U, bounded = _bounds(n, r, cols, upper, live)
    pinned = torch.zeros((n, r), dtype=torch.bool)
    history = []
    converged = False
    for it in range(max_iter):
        pin = (pinned, torch.where(pinned, U, torch.zeros_like(U))) if bool(pinned.any()) else None
        Theta = SDE._solve_regularised_thomas(solver_self, t, M, C, b, v, lam, ridge=ridge, mode=mode,
                                              dim=dim, schedule=schedule, live=live, pin=pin).cpu()
        viol = bounded & ~pinned & (Theta > U + tol_viol * U.abs())
        if pin is not None:
            g = solver_self.last_pinned_grad.cpu()
            s = solver_self.last_pinned_scale.cpu().clamp_min(1e-300)
            wants_out = pinned & (g > tol_grad * s)            # would sit below its bound if freed
        else:
            wants_out = torch.zeros_like(pinned)
        rel = wants_out if release else torch.zeros_like(wants_out)
        history.append(dict(iter=it, pinned=int(pinned.sum()), added=int(viol.sum()), released=int(rel.sum()),
                            wants_out=int(wants_out.sum()),
                            residual=float(solver_self.last_reg_residual),
                            max_violation=float((Theta - U)[bounded & ~pinned].max()) if bool((bounded & ~pinned).any()) else float('-inf')))
        log(f'  active set iter {it}: pinned {history[-1]["pinned"]}, +{history[-1]["added"]} violated, '
            f'-{history[-1]["released"]} released ({history[-1]["wants_out"]} pinned would go below their bound), '
            f'residual {history[-1]["residual"]:.2e}')
        if not bool(viol.any()) and not bool(rel.any()):
            converged = True
            break
        pinned = (pinned | viol) & ~rel
    info = dict(method=method, converged=converged, iterations=len(history), history=history,
                pinned=pinned[:, cols].clone(), release=release, n_wants_out=history[-1]['wants_out'],
                n_pinned=int(pinned.sum()), max_theta_bounded=float(Theta[:, cols].max()),
                max_excess=float((Theta - U)[bounded].max()))
    return Theta, info


def _bounds(n, r, cols, upper, live):
    U = torch.full((n, r), float('inf'), dtype=torch.float64)
    U[:, cols] = torch.as_tensor(upper, dtype=torch.float64).reshape(n, len(cols))
    bounded = torch.zeros((n, r), dtype=torch.bool)
    bounded[:, cols] = True
    bounded &= torch.isfinite(U)
    if live is not None and bool((~torch.as_tensor(np.asarray(live), dtype=torch.bool).reshape(n, r) & bounded).any()):
        raise ValueError('a bounded coordinate is dead at some node: not supported')
    return U, bounded


def solve_pnewton(solver_self, t, M, C, b, v, lam, cols, upper, ridge=0.0, mode='moment', dim=None,
                  schedule='uniform', live=None, max_iter=30, tol_grad=1e-9,
                  alphas=(1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125), log=print):
    """Projected Newton with backtracking (Bertsekas 1982) for the bounded Theta_reg energy.

    x_0 = the free solve clipped to the bounds (feasible). Each round:
      hold  = bounded entries at their bound whose energy gradient does not point below the bound;
      x_N   = the exact minimiser with `hold` pinned at the bound (one pinned Thomas solve);
      x(a)  = min(x + a (x_N - x), U): the step, projected back onto the bounds;
      take the first a in `alphas` with E(x(a)) < E(x) (all candidates scored in one pass).
    Every iterate is feasible and E decreases every round, so it cannot cycle (the plain active set
    of job 781019 did). Stops when the full step needs no clipping and the hold set is unchanged
    (then the KKT conditions hold), or when no candidate lowers E (reported, converged=False).
    Cost per round: one Thomas solve + one energy pass over the nodes.
    """
    t = np.asarray(t, dtype=np.float64)
    n, r = len(t), solver_self.num_potentials
    cols = torch.as_tensor(cols, dtype=torch.long)
    U, bounded = _bounds(n, r, cols, upper, live)
    eg = lambda Xs: SDE._reg_energy_grad(solver_self, t, M, C, b, v, lam, Xs, ridge=ridge, mode=mode,
                                         dim=dim, schedule=schedule)
    clip = lambda X: torch.where(bounded, torch.minimum(X, U), X)
    solve = lambda pin: SDE._solve_regularised_thomas(solver_self, t, M, C, b, v, lam, ridge=ridge, mode=mode,
                                                      dim=dim, schedule=schedule, live=live, pin=pin).cpu()
    free0 = solve(None)
    n_viol0 = int((bounded & (free0 > U)).sum())
    x = clip(free0)
    (E, E_free, _), (g, _, g0) = eg([x, free0, torch.zeros_like(x)])     # one pass
    g = g.cpu()
    fscale = float(g0.abs().max())                                     # max |f| (g(0) = -f), for tolerances
    tol = tol_grad * fscale
    log(f'  pnewton start: free solve has {n_viol0} entries above the bound; clipped start E = {E:.10g} '
        f'(free optimum {E_free:.10g})')
    history, converged, hold_prev = [], False, None
    for it in range(max_iter):
        at = bounded & (x >= U)
        hold = at & (g <= tol)                                         # stays at the bound
        xN = solve((hold, torch.where(hold, U, torch.zeros_like(U))) if bool(hold.any()) else None)
        d = xN - x
        cands = [clip(x + a * d) for a in alphas]
        Es, gs = eg(cands)
        j = next((i for i, Ec in enumerate(Es) if Ec < E - 1e-14 * abs(E)), None)
        clipped_full = int((bounded & (x + d > U)).sum())
        rec = dict(iter=it, hold=int(hold.sum()), at_bound=int(at.sum()), clipped_full_step=clipped_full,
                   alpha=None if j is None else alphas[j], E=E if j is None else Es[j],
                   residual=float(solver_self.last_reg_residual))
        history.append(rec)
        if j is None:
            log(f'  pnewton iter {it}: hold {rec["hold"]}, no step lowers E -> stop (E = {E:.10g})')
            converged = clipped_full == 0 and hold_prev is not None and bool((hold == hold_prev).all())
            break
        x, E, g = cands[j], Es[j], gs[j].cpu()
        log(f'  pnewton iter {it}: hold {rec["hold"]} (at bound {rec["at_bound"]}), full step clips '
            f'{clipped_full}, step {alphas[j]:g}, E = {E:.10g}, residual {rec["residual"]:.2e}')
        new_hold = bounded & (x >= U) & (g <= tol)
        if j == 0 and clipped_full == 0 and bool((new_hold == hold).all()):
            converged = True
            break
        hold_prev = hold
    at = bounded & (x >= U)
    free_b = bounded & ~at
    kkt_free = float(g[free_b].abs().max()) / fscale if bool(free_b.any()) else 0.0
    kkt_bound = float(g[at].clamp_min(0).max()) / fscale if bool(at.any()) else 0.0
    info = dict(method='pnewton', converged=converged, iterations=len(history), history=history,
                pinned=at[:, cols].clone(), n_pinned=int(at.sum()), n_wants_out=int((at & (g > tol)).sum()),
                n_violated_free_solve=n_viol0, E=E, E_free=E_free,
                kkt_free=kkt_free, kkt_bound=kkt_bound, release=None,
                max_theta_bounded=float(x[:, cols].max()), max_excess=float((x - U)[bounded].max()))
    log(f'  pnewton done: converged {converged}, {len(history)} rounds, {info["n_pinned"]} at the bound, '
        f'KKT residual free {kkt_free:.1e} / at bound {kkt_bound:.1e} (relative to max|f|)')
    return x, info
