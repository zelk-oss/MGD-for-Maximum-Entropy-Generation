# Large cancelling θ from near-identical statistics: toy model and turbulence (status 2026-10-06)

What we know about pairs (or small groups) of statistics whose gradients are almost the same, and why MGD
gives them huge coefficients of opposite sign. Part A is the controlled d = 8 Gaussian toy; part B collects
the turbulence findings of 09-28 to 10-06. Sources are given for every turbulence number (most are from
analyses run on Jean Zay; I could not re-run them for this note).

---

## 1. The mechanism in one paragraph

At every step MGD solves Ĝθ = b, Ĝ = mean over walkers of ∇φ ∇φ^T. If two statistics φ_a and φ_b have almost
the same gradient, Ĝ is nearly singular along the direction v = (+1 on a, −1 on b) (the "split" direction).
The walkers only feel θ_a ∇φ_a + θ_b ∇φ_b ≈ (θ_a + θ_b)∇φ_a, so the **sum** θ_a + θ_b is well determined and the
**difference** θ_a − θ_b is not. The step noise in b is amplified along v by 1/μ_v (μ_v = curvature of Ĝ along
v), so θ_a and θ_b become large, of opposite sign, and change sign from step to step and from seed to seed,
while their sum and the samples stay fine. The two statistics "compete" for the same job.

---

## 2. Part A — toy model (Gaussian d = 8, krylov_stopping/)

**Setup.** 36 statistics φ_ij = x_i x_j plus 36 near-copies ψ_ij = s(x_i)s(x_j), s(u) = u + δu³, δ = perturbation
strength ∈ {0.1, 0.01}. So θ has 72 coefficients, in 36 competing pairs (one per (i, j)). The walkers stay
Gaussian, so the truth is known: θ_φ = C_t^{-1} part, θ_ψ = 0. N = 1e4 walkers, σ² = 1, 1000 steps, 5 seeds.

**How singular.** Jacobi-scaled condition number of Ĝ at the target: 1.0e3 (δ = 0.1), 8.5e4 (δ = 0.01); without
copies 7.4. Along the split vs the sum direction of a pair (δ = 0.01, medians over the 36 pairs):

| direction | v^T M v (data curvature) | v^T Σ v (covariance of values) | ratio Σ/M |
|---|---|---|---|
| sum φ + ψ | 1.87 | 0.37 | 0.20 |
| split φ − ψ | 5.8e-4 | 1.1e-4 | 0.17 |

(M = Gram of gradients, Σ = covariance of the statistic values; both computed at the target by Monte Carlo.)
φ − ψ = −δ(x_i³x_j + x_i x_j³) + O(δ²): small for typical x, but growing like δ·x⁴ in the tails.

**Size of the competing coefficients** ("copies size" = ‖Θ_ψ‖_F / ‖C_t^{-1}‖_F, truth 0, time average, per-step θ):

| method | δ = 0.01 | δ = 0.1 |
|---|---|---|
| direct solve | 18.6 | 1.56 |
| ridge λ = 1e-4 / 1e-2 / 1 | 15.2 / 3.1 / 0.53 | 1.55 / 1.20 / 0.28 |
| early-stopped CG (ito_cg) | 7.4 | 0.70 |

- The copies size of the direct solve grows 12× when δ drops 10×, consistent with a noise std ∝ 1/√μ_v ∝ 1/δ.
- The samples are unaffected: KL(samples ‖ target) = 1.78e-3 for direct, every ridge up to 1e-1, and ito_cg.
- The entropy estimate cannot see the split: it uses θ·ṁ, and the two statistics of a pair have almost the
  same ṁ, so only the sum enters.
- The direct solve's per-step θ is noise-dominated even without copies (relative θ error ∝ 1/√(Nhσ²)).

**Variance.** Along an eigen-direction u of Ĝ with eigenvalue μ_u, the per-step θ noise has variance
≈ 2/(N h σ² μ_u) (Itô noise of b, calibrated: mean 71.2, variance 138 vs expected 72, 144 with 72 statistics).
The split directions have μ ~ 3000× smaller than the sum directions at δ = 0.01, so their θ noise variance is
~3000× larger.

**What controls it.**
- A data ridge (Ĝ + λ diag Ĝ) is selective: diag Ĝ is O(1) along the split while Ĝ itself is O(δ²), so relative
  to the data the ridge is ~3000× stronger along the split than along normal directions.
- Time smoothing (Θ_reg) is **not** blind to the split: its weight Σ is small along the split, but so is the data
  curvature M, by the same factor (ratio 0.17 vs 0.20). Smoothing along the split is only 2–3× weaker than elsewhere
  (generalised eigenvalues Σu = μMu: median 0.10 for split-like directions vs 0.34 for the rest, δ = 0.01).
  What makes the split hard is that its noise is ~3000× larger, so it would need a stronger λ than the rest.
  Prediction, not yet run: Θ_reg alone reduces the split but only at λ larger than the best λ for the rest; the
  data ridge does it selectively.
- Early-stopped CG (ito_cg) keeps a large split by construction: its stopping rule weights residuals by 1/μ, so it
  insists on fitting the near-singular directions. It protects the moments, not θ.

**Possible systematic part.** The target moments come from M finite samples, so the copies' empirical moments are
slightly inconsistent with a Gaussian. Matching them exactly can require a nonzero, seed-dependent θ along the
split even without step noise. Not measured (the exact empirical truth for θ_ψ is unknown).

---

## 3. Part B — turbulence

### 3.1 The competing pairs found

Cross-bank near-copies: the same filter appears in two filter banks (Q = 1 filters vs the Q = 3 ψ bank;
`deduplicate_filters` only compares inside the Q = 3 bank), so statistics of different families nearly coincide.
Source: notes/mask_moment_dupes_1001/ (check_cross_bank.py), job 382751, memory 10-01.

| pair / group | correlation | coefficients seen |
|---|---|---|
| L_6[7] ~ L_6_psi[20] | 0.99999954 | θ ≈ ±5e5, sign flipping between seeds (θ_t[−1], Sep runs) |
| L_6[8] ~ L_6_psi[21] | — | — |
| Scalar_morlet[30] ~ Scalar_psi[77] | 0.99999334 | θ ≈ ∓1.2e4 per seed, seed sum −339..+458 (θ_t[−1]); Θ_reg (λ = 0, ridge 1e-6): ∓1e5 |
| L_6_psi[18] / [19] | — | large, seed-dependent |
| Scat4_Real_Q1[3,5,6,10,11,12] | — | θ 1e8–5e9, sign flips |

(— = not recorded.) Families are blocks of the θ vector, one coefficient per statistic: L_6 has 9 coefficients
(one per filter, mean |x⋆ψ_j|^6); the full Sep set had 281 statistics, the noSm30 set 258; Scalar_psi had 73 on
the data in the 09-30 note. Other family sizes are not recorded here.

### 3.2 How singular

- Smallest eigen-direction of the Jacobi-scaled Ĝ ≈ 0.71·[stat 7] − 0.71·[stat 29] (at t = 0.9985, the L_6 pair):
  exactly the split direction of section 1. μ of the pairs ~1e-7 to 6e-6; κ(θ system) median 2.5e7 (t < 0.1) to
  1.5e8 (t > 0.999), job 382722.
- Caveat: compute_G accumulates in float32, so μ ≲ 1e-7 is at the rounding floor; κ above ~1e7–1e8 is not measurable
  (μ_min ≤ 0 at 324 of 6000 steps, 5%, after a rerun; a first reading of "half the steps" was wrong). Order of magnitude consistent with float64 1 − |corr| of the L_6 pair
  (4e-7..2.4e-6).
- For comparison with the toy: the toy's split/sum curvature ratio at δ = 0.01 is ~3e-4; the turbulence pairs are
  ~100–3000× more singular. By the toy's scaling (noise std ∝ 1/√μ), their split noise would be ~10–50× larger
  relative to the sum — an extrapolation, not a measurement.

### 3.3 What it does to log p (Sep full set, seed-by-seed)

- log p = θ·φ(x) splits into family contributions that cancel in pairs: L_6 +2.88e4 vs L_6_psi −2.83e4;
  Scalar_morlet +2.1e5 vs Scalar_psi −2.14e5; family standard deviations 1e5–8e5 against a total of 1.4e3.
  July runs (no cross-bank copies): no cancellation, family std ~1e3 ~ total. Source: memory 10-01.
- Seed reproducibility with θ_t[−1]: correlation of log p between seeds −0.50..+0.43 (Sep) vs 0.956..0.982 (July);
  excess kurtosis 32–455 (Sep) vs 0.27–0.49 (July).

### 3.4 Removing the copies was not enough

- Run with the copies removed (noL6psi_noSm30, `--drop_stats` + no L_6_psi): κ 2–4e4 for t > 0.1, 5.5e5 for t < 0.1,
  μ_min > 0 always.
- But θ_t[−1] is still not reproducible across seeds (corr −0.47..0.37, kurtosis 14–141). The disagreement now sits
  in the next-weakest directions: Scalar_morlet[29], Scalar_psi[54], [76], [32] (channel 20, regions 1–3) and
  L_2_lowpass, all sign-flipping per seed. One new weak pair, Scalar_psi[54] (ch 20 r 2) + Scalar_morlet[29]
  (ch 7 r 3), dominates the smallest eigen-direction at 78–93% of steps; not checked whether it is a near-copy.
- Diagnosis: the per-step θ noise at the tiny final step (h = 7.6e-9 at the end, variance ∝ 1/h) is dumped mostly
  into the weakest directions. The diagnostic average of the last 1000 steps gave corr 0.967–0.984 (diagnostic
  only; averaging the last steps is not a valid estimator because steps are not independent).
- So in turbulence two things combine: near-copies make some directions very weak, and the end-of-run step size
  makes the noise large; the noise goes where the matrix is weakest.

### 3.5 With the time-regularised Θ_reg (λ = 1e-3, chosen by the cross-seed rule)

- log p seed correlation median 0.998 (ridge 1e-2) / 0.999 (ridge 1e-4); worst 0.981 / 0.996; Spearman 1.
- Remaining disagreement concentrated on the most extreme rare signals: top 20 all rare; top seed spread 600–700
  nats (ridge 1e-2) and 381 nats (ridge 1e-4). This fits the tail argument of section 2 (a split coefficient times
  a near-copy difference that grows in the tails), but it has not been shown that these are split directions.

### 3.6 A different competition (not near-copies)

The noise band found 10-06 is a competition between families of different order, not a near-copy pair: Scat4_Real
(∝ amplitude⁴) positive vs L_6 (∝ amplitude⁶) negative; their balance puts a bump of probability at mid amplitude.
Present already in the sampler's per-step θ, kept by Θ_reg. Listed here only to keep the two effects separate.

---

## 4. Summary

- Near-identical statistics make the split direction nearly singular; the sum of the competing coefficients is
  well determined, their difference is pure amplified noise (plus possibly a small systematic part).
- Samples and entropy are blind to it; θ itself and log p in the tails are not.
- Toy: the direct solve's split is 18.6× the true θ at δ = 0.01 and scales like 1/δ; a data ridge suppresses it
  selectively, early-stopped CG only 2.5×, time smoothing should act but needs a large λ (to be tested).
- Turbulence: cross-bank copies (Q = 1 vs Q = 3 filters) gave cancelling families and non-reproducible log p;
  removing them improved conditioning 1000× but per-step θ stayed non-reproducible because the end-of-run noise
  then goes into the next-weakest directions; Θ_reg at λ = 1e-3 restores seed agreement except on extreme signals.

## 5. Open

1. Θ_reg × data ridge on the toy (planned): does time smoothing alone remove the split, and at which λ?
2. Whether the residual Θ_reg disagreement on extreme turbulence signals comes from split directions (decompose the
   seed spread of log p by eigen-direction of Ĝ).
3. Principled cross-bank deduplication (drop a Q = 1 filter when it is within 1e-2 of a kept ψ filter: removes
   Q = 1 #7, #8) instead of `--drop_stats`; check the psi[54]/Morlet[29] pair.
4. float32 accumulation of Ĝ hides μ below ~1e-7.
