# Early-stopped Krylov solves instead of the ridge (2026-10-06)

Test bed for replacing the per-step ridge `(G + lam diag G) x = b` of MGD's predictor (eta) and
corrector (theta) by CG / MINRES started at 0 and stopped when the residual reaches the noise
level of `b`. First on the centred Gaussian d = 8 (truth known), later on turbulence.

## Files

| file | what |
|---|---|
| `codes/krylov.py` | `cg`, `minres` (Paige-Saunders), stopping rules `ito_stop` / `residual_stop`, noise weights `tol_sigma` / `tol_weight` (diag, full, Schaefer-Strimmer shrink toward the diagonal) |
| `codes/sde_krylov.py` | `SDEKrylov(SDE)`: options `theta_solver` (`ridge`, `ito_cg`, `tol_minres`), `eta_solver` (`ridge`, `tol_minres`), `tol_weight`. **`codes/sde_routines.py` is not modified**; with both solvers `ridge` the class calls the parent's methods unchanged |
| `tests/test_krylov.py` | solvers vs scipy (iterate by iterate) and `torch.linalg.solve`, MINRES residual monotone + recurrence = explicit residual, ito statistic = A-norm error, shrinkage vs the naive formula |
| `tests/test_sde_krylov.py` | ridge path bit-identical to `SDE` (lam 1e-4 and 0), per-sample mdot mean = parent's mdot, every Krylov stop meets its rule, `forward_regularised` refused |
| `gauss_d8_krylov.py` | the experiment (`# %%` cells), tables -> `results.md`, plots -> `figures/` |
| `launch_gauss_d8_krylov.sh` | Jean Zay array job, 20 tasks = (N, sigma2, seed) |

Tests (from the repo root, seconds, plain asserts):
`python krylov_stopping/tests/test_krylov.py && python krylov_stopping/tests/test_sde_krylov.py`

## Rules (code units)

The code solves `G(y) theta_raw = rhs`, `rhs = phibar(I_{k+1}) - phibar(y_k) = -rho_k` in **moment
units** and stores `theta_code = theta_raw / (h sigma^2) = -theta_MGD`. So the residual of the raw
system is the moment-unit residual `h sigma^2 u` directly.

- `ito_cg` (theta): CG on the Jacobi-scaled system; stop at the first j with
  `N / (2 h sigma^2) * res^T G^-1 res <= chi2_{r_live, 0.95}` (one Cholesky per step).
- `tol_minres` (theta): MINRES on `W^-1/2 G W^-1/2`, `W = Cov_s[phi(I^s)]/M_eff + Cov_i[phi(y^i)]/N`;
  stop at `||W^-1/2 res||^2 <= eps2`, `eps2 = chi2_{r, 0.95}` (full, shrink) or
  `r + 2 sqrt(2) ||R||_F` (diag).
- `tol_minres` (eta): same with `W_eta = Cov_s[mdot^s]/M_eff`, `mdot^s = <grad phi(I^s), Idot^s>`.
- `max_iter = r_live`; `var_floor = 1e-6 x median variance` (also an eigenvalue floor for full/shrink).

## Experiment design (agreed 2026-10-06)

C_1 = Q diag(logspace(-1, 0, 8)) Q^T, Q random (data_seed 0). N in {1e3, 1e4}, M = N, sigma^2 in
{1, 10}, nt = 1000 uniform (h sigma^2 in {1e-3, 1e-2}, N h sigma^2 in {1, 10, 10, 100}), seeds
900-904. Per seed, x_0, x_1, the walkers and the noise are the same draws for every method.
Truths: empirical S_t (whitened residual vs threshold, calibration) and population C_t (theta
error, KL, entropy integral).

## Assumptions / choices made in the code

- r = 36: the existing `Quadratic` has no linear terms.
- Float32 data with `solve_float64=True` for every method (as `gauss_d8_lamtest.py`).
- "direct" = ridge path with `regularization = 0` (no separate code path).
- eta weight in the rows `ito_cg` and `direct+eta_tol`: full.
- Diagnostics (j = 0 values of both statistics, per-step G(y) and Sigma_tol) are computed for
  every method, outside the timed call.
- Time averages skip t < 0.05 (start-up transient: the walkers start from their own N(0, I)
  draws, not the target's empirical S_0); calibration uses t >= 0.1.
- `stop_fn` returns `(stop, statistic)`, not just `stop`, so the solver can record the history.

## Results

Full run (JZ job 679591): **RESULTS_gauss_d8.md**.

## Smoke findings (local, N = 200, nt = 100, seed 900 -- code checks, not results)

- Calibration with the direct solve, z = (N h s2 / 2) e^T G' e, e = theta - theta*(S_t):
  h sigma^2 = 1e-2: mean 32.7, var 56; h sigma^2 = 1e-3 (sigma^2 = 0.1): mean 35.6, var 75.8
  (chi2_36: 36, 72). No sign/scale error; the deficit at 1e-2 is O(h sigma^2) (whitening with
  G(x) instead of G(y) does not remove it: 33.4). Hypothesis, not checked: the noise that enters
  b also moves G'(y), so G' theta* absorbs a fraction ~ h sigma^2 lambda(C^-1) of it.
- Direct solve entropy integral is 0.52 nats above the empirical exact value; the uniform-grid
  quadrature alone is exact to 1e-4, so this comes from theta . mdot itself (size / scaling to be
  read off the full run).
