# Theta_reg with negative L_6 coefficients (2026-10-08)

## Why
`L_6` (mean of |W_j x|^6 per wavelet filter j) is the only 6th-order statistic in the noSm30 runs,
so it alone decides whether p_Θ ∝ exp(Θ·φ) is confined at large amplitude. The pure-tone test
(TEST 2026-10-07 (B) in `turbulence/turb_theta_reg_rare_typical_sep_noL6psi_noSm30.ipynb`) and the
two-arm plot of Θ_reg (λ = 1e-3) showed:
- the fine filters (0-3) are negative in every seed;
- filter 4 (period 32) is positive in 9 of 10 seeds (both arms);
- filters 5-7 have both signs, sizes 10-300 in raw units;
- along a tone at a filter with a positive coefficient, log p grows like a^6: p_Θ is not normalisable.

So Θ_reg is solved again with the bound Θ_{k,i} ≤ −ε_{k,i} on all 9 L_6 columns at every node.

## How
- `codes/sde_routines.py`, `SDE._solve_regularised_thomas(..., pin=(mask, values))`: holds chosen
  entries at given values (same mechanism as the dead-statistic mask, which holds them at 0), with
  the exact coupling of the free entries to them. `pin=None` is bit-identical to before (tested).
- `codes/theta_reg_bounds.py`: the loop that enforces the bounds. Default = projected Newton: start
  from the free solve clipped to the bounds; each round hold the entries at their bound that want to
  stay there, refit the rest exactly (one pinned Thomas solve), step towards it, clip back onto the
  bounds, keep the step only if the fit improves. Every round is feasible and better than the last,
  so it cannot cycle.
  - History (2026-10-08): the first version was the plain active set (pin every violated entry,
    release every pinned entry that wants to go lower, all at once). On turbulence (job 781019,
    cancelled) it oscillated: 40-50 thousand entries added and released every round. An "add only"
    variant converges but on a synthetic system its fit was 64% worse than the optimum. Both are
    still available (`--neg_method exact | add_only`) for comparison.
  - `SDE._reg_energy_grad`: energy and gradient of the same system, for the step acceptance.
- `codes/resolve_theta_reg.py --neg_cols 0:9 --neg_eps_nats EPS [--neg_t_min T]`: output
  `lam<λ>_ridge<ridge>_moment_mask_neg<EPS>.pt`, with `neg` (bounds, iterations, which entries ended
  pinned, convergence) next to `Theta_reg`.
- `launch_l6neg.sh`: one array task per seed of one arm (usage in the header).

## Choice of ε
ε is set per statistic and per node in units that make sense for log p:
ε_{k,i} = EPS / sd_k(φ_i), sd = spread of φ_i on the walkers at node k (from the saved Σ_k). So a
pinned coefficient changes log p by about EPS nats per standard deviation of its statistic, whatever
its raw units (the raw L_6 coefficients go from ~1e7 at the finest filter to ~1e1-1e3 at the coarse ones).

Default EPS = 1e-2, deliberately much smaller than the seed noise, NOT the noise size:
- the goal is only the sign (confinement). With EPS of the size of the noise, the bound would push
  the coarse coefficients about one standard error below zero: that adds information the data do
  not give ("clearly negative") instead of "zero or slightly negative";
- a pinned coefficient then moves log p of a data signal by ~EPS = 0.01 nats, negligible next to the
  seed spread of log p (~0.8 nats for typical signals, reg1e-2 arm, Theta_reg λ = 1e-3);
- any EPS > 0 makes the coefficient strictly negative; since L_6 grows like a^6 it still wins at large
  amplitude, only later for smaller EPS.
It is a launcher knob (EPS); the files carry it in their name.

What it does not fix: the mid-amplitude bumps of log p on fine/medium tones (TEST 10-07 (B)) happen
at filters that are already negative.

## Tests
`python theta_sign_constraint/tests/test_bounds.py` (local, ~1 min): pin=None bit-identical to HEAD;
pinned solve = dense elimination (1e-14); exact active set = L-BFGS-B on the dense energy (max rel
diff 5e-7) and KKT; inactive bounds = free solve; add-only feasible but worse (3e-3 and 0.64 relative
energy gap); energy helper = dense; projected Newton = exact solution (energy and Θ identical) in 3
and 9 rounds, KKT.
