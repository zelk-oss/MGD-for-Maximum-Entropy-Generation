"""Theta_reg with upper bounds on some coordinates: Theta_{k,i} <= u_{k,i} (2026-10-08).

Use: force the L_6 coefficients of Theta_reg negative at every node, Theta_{k,i} <= -eps_{k,i},
so that exp(Theta . phi) is confined along every direction the L_6 bank sees (theta_sign_constraint/).

The time-regularised energy E(Theta) = 1/2 Theta^T A Theta - f^T Theta (A SPD, the block-tridiagonal
system of SDE._solve_regularised_thomas) is minimised under the bounds with a primal active-set loop:

    pinned = {}                                    (entries held at their bound)
    repeat:
        Theta = argmin E with Theta = u on pinned  (one block-Thomas solve, pin=...)
        add    every free bounded entry with Theta > u               (violated)
        remove every pinned entry with dE/dTheta = (A Theta - f) > 0 (E would decrease by moving
               it below its bound, into the allowed region: the bound is not active there)
    until nothing changes -> the KKT conditions of the bounded problem hold.

For a convex quadratic with box constraints this ends after a few solves in practice; it is capped
at max_iter and the result says whether it converged.

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
                        tol_grad=1e-9, log=print):
    """Minimise the Theta_reg energy with Theta[:, cols] <= upper (n, len(cols)).

    Returns (Theta (n, r) float64, info dict). Arguments as SDE._solve_regularised_thomas
    (called unbound with solver_self). Bounded entries must not be dead in `live`.
    """
    t = np.asarray(t, dtype=np.float64)
    n, r = len(t), solver_self.num_potentials
    cols = torch.as_tensor(cols, dtype=torch.long)
    U = torch.full((n, r), float('inf'), dtype=torch.float64)
    U[:, cols] = torch.as_tensor(upper, dtype=torch.float64).reshape(n, len(cols))
    bounded = torch.zeros((n, r), dtype=torch.bool)
    bounded[:, cols] = True
    if live is not None and bool((~torch.as_tensor(np.asarray(live), dtype=torch.bool).reshape(n, r) & bounded).any()):
        raise ValueError('a bounded coordinate is dead at some node: not supported')
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
            release = pinned & (g > tol_grad * s)
        else:
            release = torch.zeros_like(pinned)
        history.append(dict(iter=it, pinned=int(pinned.sum()), added=int(viol.sum()), released=int(release.sum()),
                            residual=float(solver_self.last_reg_residual),
                            max_violation=float((Theta - U)[bounded & ~pinned].max()) if bool((bounded & ~pinned).any()) else float('-inf')))
        log(f'  active set iter {it}: pinned {history[-1]["pinned"]}, +{history[-1]["added"]} violated, '
            f'-{history[-1]["released"]} released, residual {history[-1]["residual"]:.2e}')
        if not bool(viol.any()) and not bool(release.any()):
            converged = True
            break
        pinned = (pinned | viol) & ~release
    info = dict(converged=converged, iterations=len(history), history=history, pinned=pinned[:, cols].clone(),
                n_pinned=int(pinned.sum()), max_theta_bounded=float(Theta[:, cols].max()),
                max_excess=float((Theta - U)[bounded].max()))
    return Theta, info
