# Gaussian d = 8: ridge vs early-stopped Krylov -- results (Jean Zay job 679591, 2026-10-06)

Full tables (every method x config, seed mean ± std over 5 seeds): `results/gauss_d8_krylov/results.md`.
Plots: `results/gauss_d8_krylov/figures/` (`j_theta`, `j_eta`, `theta_err_pop`, `theta_err_emp`).
Setup: README.md. Configs: N in {1e3, 1e4}, sigma^2 in {1, 10}, nt = 1000, M = N, r = 36.
Run cost: 5-6 s per run at N = 1e3, 20-23 s at N = 1e4.

## Bottom line

At d = 8, early stopping brings no gain on any metric. `ito_cg` is as good as the direct solve
(same KL up to 1.1x, except 1.26x at N h sigma^2 = 100). The three `tol_minres` variants are clearly
worse: KL 3x (full, shrink) to 9-19x (diag) the direct solve's. The ridge level does not matter
here: lam = 1e-7 and 1e-4 give the direct solve's numbers, and lam = 1e-2 changes theta by about 1%.
This problem is well conditioned, so it cannot show the case where regularisation is needed
(turbulence: near-duplicate statistics).

## What each metric says

**KL at t = 1, against the population C_1** (seed mean). The direct solve sits exactly at the
sampling floor of the target, d(d+1)/(4M) = 0.018 (M = 1e3) / 0.0018 (M = 1e4).

| method | N=1e3 s2=1 | N=1e3 s2=10 | N=1e4 s2=1 | N=1e4 s2=10 |
|---|---|---|---|---|
| direct (= ridge 1e-7, 1e-4, 1e-2; = direct+eta_tol) | 0.0184 | 0.0183 | 0.00178 | 0.00178 |
| ito_cg | 0.0182 | 0.0201 | 0.00182 | 0.00225 |
| tol_full | 0.0521 | 0.0546 | 0.00542 | 0.00532 |
| tol_shrink | 0.0561 | 0.0502 | 0.00499 | 0.00521 |
| tol_diag | 0.208 | 0.166 | 0.0274 | 0.0341 |

**Moment error against the population: does the early-stopping offset stay within the statistical
error?** No for the tol rules. The offset they add is uncorrelated with the target's own error:
sqrt(err_tol^2 - err_direct^2) equals their error against the empirical target to within 10%
(e.g. tol_full at N = 1e3, s2 = 1: 0.128 vs 0.115). So the two errors add up in quadrature; they do
not cancel, as predicted. The offset is 1.2x (diag), 1.2-1.4x (shrink) and 1.4-1.6x (full) the
statistical error. This matches the size of the rule's own tolerance: with M = N, Sigma_tol is twice
the target's sampling variance, and the Ito diffusion pushes the walkers out to the edge of the
allowed band before a correction brings them back. That is the sawtooth in `theta_err_pop.png`:
theta = 0, so the error is exactly 1, for many steps, then one large correcting step.
`ito_cg` stays within the statistical error: 0.0872 vs 0.0874, 0.026 vs 0.0261.

**Does each rule do what it claims (whitened residual against the empirical target)?**
- tol_full: never above its chi^2 threshold; the time average is 0.38-0.44 of it.
- tol_shrink: above it in 6% / 4% / 0.6% / 0.2% of steps (it stops on its own shrunk weight, the
  check uses the unshrunk one).
- tol_diag: above the full chi^2 threshold in 30-34% of steps, but meets its own diag criterion
  (0.40-0.44). The diag threshold r + 2 sqrt(2)||R||_F is loose because the 36 statistics are
  strongly correlated, so the diagonal weight misses correlated residual directions.
- ito_cg: 0.002-0.016 of the threshold, i.e. it corrects far below the target's noise.

**Iteration counts.**
- ito_cg: j_theta = 0.7 / 1.1 / 1.15 / 1.5 on average. It rises in integer steps (1 -> 2 -> 3) for
  t > 0.6-0.8, where C_t becomes anisotropic (`j_theta.png`).
- j = 0 in 31% / 1% / 1% / 0% of steps. This confirms your prediction: the j = 0 statistic is
  r + N h sigma^2 tr(C^-1)/2 (measured 44.0 / 111.6 / 116.1 / 812, predicted 44.1 / 117.0 / 116.5 /
  840.8), and N h sigma^2 tr(C^-1)/(2r) = 0.23 / 2.2 / 2.2 / 22.
- tol rules: j_theta = 0 in 97% / 74-79% / 91-92% / 40-44% of steps.
- eta (tol_minres): 1-2 iterations, at most 3, j = 0 in 1-5%. Switching eta to early stopping
  (direct+eta_tol vs direct) changes nothing measurable: KL, moments and theta are equal to 3 digits.
  The entropy integral moves by 0.05 nats at N = 1e3, s2 = 1, within the seed scatter.
- No Krylov solve ever hit max_iter.

**theta error.** The direct solve's per-step theta error depends only on N h sigma^2: 2.19 / 0.66 /
0.69 / 0.22, i.e. a factor sqrt(10) per decade of N h sigma^2. That is the Ito noise of b, as the
calibration says. It is large, yet it does not hurt the walkers: it is exactly the response that
removes the injected noise from the moments (direct: moment error against the empirical target
2e-5 to 6e-4). Early stopping removes part of this response, so the noise stays in the walkers:
ito_cg has 0.0009-0.009, still far below the target error. ito_cg's theta error is 2-10% above the
direct one at N h sigma^2 >= 10. The tol rules' "lower" error at N h sigma^2 = 1 (1.9 vs 2.2) comes
from theta = 0, which has error exactly 1.

**Entropy integral minus exact** (direct, empirical truth): 0.067 / 0.236 / 0.020 / 0.187 nats.
It fits 18 h sigma^2 + 50 / N to within 0.004 on all four configs. The h sigma^2 part is
time-discretisation bias; the 1/N part is probably the bias of theta from solving with the noisy
G'(y) (not checked). The smoke run (N = 200, s2 = 1, nt = 100) is close to this fit (0.52 vs 0.44
predicted). ito_cg adds 0.01-0.05; the tol rules add 0.06-0.6.

**Wall time.** The Krylov call costs +0.6-1.9 ms per step at N = 1e3 (+35-110%) and +1.5-3.5 ms at
N = 1e4 (+15-35%). The extra cost is the covariances for Sigma_tol and the Cholesky, not the 1-3
iterations. No speed gain is possible at r = 36.

## Calibration of the Ito noise model (direct solve, t >= 0.1)

| config | h sigma^2 | mean z (36) | var z (72) |
|---|---|---|---|
| N=1e3 s2=1 | 1e-3 | 35.85 | 71.3 |
| N=1e4 s2=1 | 1e-3 | 35.78 | 71.2 |
| N=1e3 s2=10 | 1e-2 | 33.09 | 62.6 |
| N=1e4 s2=10 | 1e-2 | 35.91 | 88.7 |
| smoke N=200 s2=1, nt=100 | 1e-2 | 32.72 | 56 |

At h sigma^2 = 1e-3 the model holds to 0.6% in mean and 1% in variance. At 1e-2 there are two
opposite effects, fitted on these 3 points: mean = 0.91 x 36 + 3.1e-4 N (the fit predicts 32.84
for the smoke run vs 32.72 measured). The 9% deficit is an O(h sigma^2) effect. The additive term
grows with N and appears only at s2 = 10. A deterministic error in b of order h sigma^2 would give
this, because its Ito-whitened weight is N (h sigma^2)^3, so 1000x larger at s2 = 10 than at s2 = 1
for the same h. A candidate is the residual the linearised corrector (21) leaves behind: your own
measurement is 0.9% at h sigma^2 = 1e-2. This is a hypothesis, not checked; it would also explain
the extra variance (88.7).

## What this does and does not answer

- Answered: the stopping rules are calibrated, their thresholds behave as derived, and early
  stopping of eta is harmless.
- Answered: at d = 8, the per-step theta "noise" is functional (it cancels the Ito noise on the
  moments), so a rule that treats it as noise to suppress (tol_*) lets the walkers drift by about
  its tolerance.
- Not answered: whether ito_cg beats the ridge when the Gram matrix is ill-conditioned. Here there
  is nothing to regularise. A d = 8 variant with deliberately near-duplicate statistics (e.g.
  phi plus slightly perturbed copies) would test that before turbulence.
