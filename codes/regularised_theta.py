"""Regularised theta_t for MGD via dual score matching (Guth et al., 2025).

Replaces the pointwise estimators theta_k (solved independently at each noise
level) by a single regularised trajectory: the dual time-score term couples the
levels through the feature Gram, lowering variance. Setting that coupling to zero
returns your original M_k theta_k = b_k.

YOU SUPPLY (all aligned on the same grid t; use a MODEST grid, ~20-100 levels,
NOT your 2000-step sampling grid -- the solve is on n_t * r unknowns):

    potentials : your *fitted* potentials dict. Reuse ``Solver.potentials`` after
                 building the sampler, so phi here is identical to the phi behind
                 your theta. (SDE.__init__ runs self.fit(x_1); a fresh
                 get_1d_potentials(...) is NOT fitted -> "must be fit_reference'd".)
    t          : (n_t,) strictly increasing grid in (0, 1].
    X_levels   : sequence, X_levels[k] = walkers at t_k, shape (B, C, T).
    s_levels   : sequence, s_levels[k] = per-walker target (X_t - X_0)/t at t_k --
                 the same residual you reduce to build b_t.
    M_levels   : (n_t, r, r) space metric E[grad phi grad phi^T] from your solve.
    b_levels   : (n_t, r)    second member from your solve (M_k theta_k = b_k).
    dim        : ambient dimension d = C * T.

RETURNS  Theta_reg : (n_t, r);  Theta_reg[-1] is the regularised theta at t = 1.
"""

import torch


# --------------------------------------------------------------------------- #
# Features: per-realisation potentials, concatenated in potentials order.      #
# --------------------------------------------------------------------------- #
def features(potentials, X):
    """Evaluate all potentials on a batch and concatenate.

    Parameters
    ----------
    potentials : dict[str, object]
        Fitted potential objects; each ``p.forward(X)`` returns per-realisation
        coefficients of shape (B, p.num_coefficients).
    X : torch.Tensor, shape (B, C, T)
        Walkers at one noise level.

    Returns
    -------
    torch.Tensor, shape (B, r)
        Per-realisation feature vectors, r = sum of num_coefficients.
    """
    outs = [p.forward(X) for p in potentials.values()]
    return torch.cat([o.reshape(X.shape[0], -1) for o in outs], dim=1)


# --------------------------------------------------------------------------- #
# Time-score pieces: G_t = E[phi phi^T] and c_t = E[phi tau_t], values only.    #
# --------------------------------------------------------------------------- #
def time_terms(potentials, X_levels, s_levels, t, dim):
    """Build the time metric G_t and time second member c_t for every level.

    tau_t = d/(2t) - ||X_t - X_0||^2 / (2 t^2) = d/(2t) - 1/2 ||s_t||^2, with
    s_t = (X_t - X_0)/t the same target used for b_t. No autodiff is needed.

    Returns
    -------
    G : torch.Tensor, shape (n_t, r, r)
    c : torch.Tensor, shape (n_t, r)
    """
    G, c = [], []
    for k in range(len(X_levels)):
        Phi = features(potentials, X_levels[k])              # (B, r)
        B = Phi.shape[0]
        G.append(torch.einsum('bi,bj->ij', Phi, Phi) / B)
        s2 = s_levels[k].reshape(B, -1).pow(2).sum(1)        # ||s_t||^2  (B,)
        tau = dim / (2.0 * float(t[k])) - 0.5 * s2           # time-score target
        c.append(torch.einsum('bi,b->i', Phi, tau) / B)
    return torch.stack(G), torch.stack(c)


# --------------------------------------------------------------------------- #
# Block-tridiagonal solve of the dual-score Euler-Lagrange system.             #
# --------------------------------------------------------------------------- #
def solve_regularised_theta(t, M, G, b, c, dim):
    """Solve A Theta = f, A symmetric positive-definite block-tridiagonal.

    Discrete energy (trapezoid node weights h_k for the space integral; the time
    term lives on cells [t_k, t_{k+1}] with backward difference):

        L = sum_k (h_k/d)  ( 1/2 theta_k^T M_k theta_k - b_k^T theta_k )
          + sum_k (t_k/d^2)[ ||theta_{k+1}-theta_k||^2_{G_k}/(2 dt_k)
                             - c_k^T (theta_{k+1}-theta_k) ].

    Off-diagonal blocks are -(t_k/dt_k) G_k. Dropping the time term (G=0) gives
    back the pointwise M_k theta_k = b_k.
    """
    nt, p, _ = M.shape
    dt = t[1:] - t[:-1]
    assert torch.all(dt > 0), "t must be strictly increasing"

    h = torch.empty(nt, dtype=M.dtype, device=M.device)      # trapezoid weights
    h[0], h[-1] = dt[0] / 2, dt[-1] / 2
    h[1:-1] = (dt[:-1] + dt[1:]) / 2

    A = torch.zeros(nt, p, nt, p, dtype=M.dtype, device=M.device)
    f = torch.zeros(nt, p, dtype=M.dtype, device=M.device)

    for k in range(nt):                                      # space (DSM) term
        A[k, :, k, :] += (h[k] / dim) * M[k]
        f[k] += (h[k] / dim) * b[k]

    for k in range(nt - 1):                                  # time (TSM) term
        a = t[k] / (dim ** 2 * dt[k])                        # 1/2 ||d theta||^2_{G_k}
        g = t[k] / (dim ** 2)                                # linear c_k term
        A[k, :, k, :] += a * G[k]
        A[k + 1, :, k + 1, :] += a * G[k]
        A[k, :, k + 1, :] -= a * G[k]
        A[k + 1, :, k, :] -= a * G[k]
        f[k] -= g * c[k]
        f[k + 1] += g * c[k]

    Theta = torch.linalg.solve(A.reshape(nt * p, nt * p), f.reshape(nt * p))
    return Theta.reshape(nt, p)
    # For very large n_t * r, swap the dense solve for a block-Thomas sweep
    # (O(n_t * r^3)); the matrix is already block-tridiagonal.


# --------------------------------------------------------------------------- #
# One call: glue the pieces together.                                          #
# --------------------------------------------------------------------------- #
def regularise(potentials, t, X_levels, s_levels, M_levels, b_levels, dim):
    """Build G_t, c_t from the fitted potentials and solve for the trajectory."""
    G, c = time_terms(potentials, X_levels, s_levels, t, dim)
    return solve_regularised_theta(t, M_levels, G, b_levels, c, dim)


# --------------------------------------------------------------------------- #
# Usage (adapt names to your script).                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # 1) Reuse the potentials your sampler already fitted on x1.
    #    Solver = SDE(x1, potentials, theta, nt, delta_t, batch_size, device=device)
    #    potentials = Solver.potentials          # fit_reference already done

    # 2) Pick a MODEST regularisation grid and gather the per-level quantities.
    #    t        : (n_t,) increasing in (0, 1]      (subsample your fine grid)
    #    X_levels : [walkers at t_k]   each (B, C, T)
    #    s_levels : [(X_t - X_0)/t]    each (B, C, T)  -- same residual as b_t
    #    M_levels : (n_t, r, r)        from your pointwise solve
    #    b_levels : (n_t, r)           from your pointwise solve
    #    dim      = x1.shape[-2] * x1.shape[-1]

    # 3) Sanity check before solving: feature width must equal the model's r.
    #    Phi0 = features(potentials, X_levels[0])
    #    assert Phi0.shape == (X_levels[0].shape[0], M_levels.shape[1]), Phi0.shape

    # 4) Solve.
    #    Theta_reg = regularise(potentials, t, X_levels, s_levels,
    #                           M_levels, b_levels, dim)
    #    print("Regularised theta:", Theta_reg.shape)   # (n_t, r); theta_1 = [-1]
    pass
