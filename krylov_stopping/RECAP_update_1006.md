# Update to the early-stopping recap (2026-10-06, later the same day)

Changes to the recap you already have. Everything not listed here is unchanged. Section numbers refer to the
recap; the old "§5 Open questions" is now §6.

## 0. Renaming

The duplicate perturbation is now called **perturbation strength δ** (was ε), so ε² refers only to the
stopping threshold of `tol_minres`. Everywhere in §4 read δ for ε: s(u) = u + δu³, δ ∈ {0.1, 0.01},
condition number 1.0e3 (δ = 0.1) and 8.5e4 (δ = 0.01). The code flag is still `--eps`.

## 1. §3 (Experiment 1), conclusion — replaced

Conclusion of experiment 1: nothing to regularise; tol rules harmful; ito_cg nearly harmless but not free:
with θ effects isolated (see §4 below, attribution) it raises KL by 10% / 26% at σ² = 10 (N = 1e3 / 1e4,
significant) and the entropy bias by +0.006 to +0.05 nats.

## 2. §4 "Other facts" — one line corrected

- η early stopping: no effect on θ, KL or moments; it does shift the entropy estimate (−0.024 nats, see
  Interpretation item 2). (Was: "still no effect".)

## 3. §4 Interpretation, item 2 — replaced (the "ito_cg lands in the unbiased window" claim is WITHDRAWN)

**The near-zero entropy bias is not evidence that any method gets θ right.** Paired entropy-bias shifts vs the
direct solve at N = 1e4, σ² = 1 (mean ± SE, 5 seeds):

| shift vs direct | no duplicates (Exp. 1) | δ = 0.01 | δ = 0.1 |
|---|---|---|---|
| direct bias itself | +0.0196 | +0.0309 | +0.0320 |
| ridge 1e-2 | −0.0251 ± 0.0002 | −0.0222 | −0.0262 |
| direct θ + early-stopped η (`direct+eta_tol`) | −0.0040 ± 0.0001 | −0.0243 ± 0.0008 | −0.0250 ± 0.0009 |
| ito_cg (early-stopped θ AND η) | +0.0015 ± 0.0002 | −0.0254 ± 0.0008 | −0.0266 ± 0.0008 |
| early-stopped θ alone (ito_cg − direct+eta_tol) | +0.0055 ± 0.0001 | −0.0010 ± 0.0001 | −0.0016 ± 0.0002 |

- The direct solve has a generic positive bias (+0.020 without duplicates; the duplicates raise it to +0.031).
  Ridge 1e-2 shifts it by a fixed ≈ −0.025 with or without duplicates: its "unbiased window" is a negative
  shrinkage bias cancelling the generic positive bias, not θ being right.
- Without duplicates, ito_cg has the direct solve's bias (+0.021): it is not unbiased.
- With duplicates, ito_cg's shift comes almost entirely from early-stopping **η** (the `ito_cg` method also uses
  tol_minres for η); early-stopping θ contributes −0.001 to −0.0016. It is a cancellation, produced by the
  η rule, present only with duplicates.
- The ridge window did not move between δ = 0.01 and 0.1 (λ = 1e-2 in both).

**Attribution θ vs η, all metrics** (the `ito_cg` method early-stops both solves; paired, 5 seeds):
- η rule alone (`direct+eta_tol` − `direct`): no measurable effect on copies size, θ_φ error, sum error, KL or
  moments (e.g. copies +0.0125 ± 0.002 on a scale of 18.6); it changes only the entropy (−0.024 with
  duplicates, −0.004 to −0.05 without).
- Early-stopped θ alone (`ito_cg` − `direct+eta_tol`): all the θ gains (copies −11.2 ± 0.07 at δ = 0.01,
  −0.86 ± 0.007 at δ = 0.1; θ_φ error −11.6 / −1.16; sum error −0.09 / −0.13); entropy −0.001 / −0.002.
- Costs of early-stopped θ alone, Exp. 1 (no duplicates): entropy bias **increases** (+0.0055 ± 0.0001 at
  N = 1e4 σ² = 1; +0.021 at N = 1e3 σ² = 1; +0.051 at N = 1e3 σ² = 10); KL increases at σ² = 10:
  +1.84 ± 0.7 ×1e-3 (+10%, N = 1e3) and +0.47 ± 0.1 ×1e-3 (+26%, N = 1e4), i.e. the ratios 1.10 / 1.26 in §3
  are real at 2.5 and 4.7 standard errors; moment tracking vs the empirical target looser by 0.0009–0.009
  (by design).
- Not separated: ridge rows apply λ to both η and θ; no run isolates the θ ridge.

## 4. §4 Honest summary — replaced

- ito_cg (early-stopped θ): a tuning-free regulariser that leaves the samples unchanged and reduces the
  cancelling split 2.2–2.5× vs the direct solve, but does not remove it; on the split it beats ridge 1e-2 at
  δ = 0.1 and loses at δ = 0.01. Present it as "comparable to a reasonable ridge, without choosing λ".
  These gains are attributable to early-stopping θ (the η rule contributes nothing to them). Costs without
  duplicates: entropy bias +0.006 to +0.05 nats and KL +10–26% at σ² = 10.
- The entropy integral cannot discriminate the methods here: its near-zero values are cancellations of a
  generic positive bias (+0.02 nats) by shrinkage (ridge) or by the η stopping rule (ito_cg rows).
- tol_minres rules: a negative result, see §5.

## 5. NEW §5 — Negative result: tolerance-based stopping (tol_minres) fails

`tol_minres` (diag, full and shrink weights) was the recommended default for θ going in. It fails in both
experiments, and the failure has one cause: **a tolerance set by the target's statistical error is the wrong
scale for a per-step solve.** The scale mismatch had been flagged in advance as an open issue in the notes and
was underweighted.

**What happens.**
- The corrector is switched off (j = 0, θ = 0) in 40–97% of steps (Exp. 1) and ~92% of steps (Exp. 2).
- With θ = 0, nothing compensates the Itô drift, so the walkers' moments drift systematically. When the drift
  exceeds the tolerance, one large correction resets them: a sawtooth in time.
- The drift does not cancel the target's own sampling error. Against the population, the two errors add in
  quadrature: the extra moment error is 1.2–1.6× the statistical error. KL: 3× the direct solve (full,
  shrink) to 9–19× (diag) in Exp. 1, and 4–20× in Exp. 2.
- The rule does what it claims, but that is the problem: tol_full never exceeds its χ² threshold, it just
  uses the whole allowance at every step.

**Why: the two scales.**
- Tolerance: Σ_tol ≈ Cov(φ)/M + Cov(φ)/N, a band of relative width ~ 1/√N per moment (M = N). It is the right
  scale for judging the **end result**, because the target itself is only known to that precision.
- Per step: the uncorrected drift moves a second moment by ~ 2hσ²/C. It is deterministic, so it accumulates
  over steps instead of averaging out.
- So the rule allows ~ 1/(√N hσ²) uncorrected steps before acting. Measured gap between corrections,
  1/(fraction of steps with j > 0), against this prediction:

  | config (Exp. 1) | N = 1e3, hσ² = 1e-3 | 1e3, 1e-2 | 1e4, 1e-3 | 1e4, 1e-2 |
  |---|---|---|---|---|
  | predicted 1/(√N hσ²) | 31.6 | 3.2 | 10.0 | 1.0 |
  | tol_full, measured | 33.3 | 3.9 | 11.4 | 1.7 |
  | tol_diag, measured | 41.7 | 4.7 | 13.0 | 1.8 |
  | tol_shrink, measured | 38.5 | 4.3 | 11.5 | 1.7 |

  Exp. 2 (N = 1e4, hσ² = 1e-3, r = 72, predicted 10): 12–15 for all three weights at both δ. The
  measured/predicted ratio is 1.05–1.8 over a 30-fold range of the prediction; the O(1) prefactor is not derived.

**Lesson.** The per-step stopping tolerance must be the per-step noise of the right-hand side (the Itô scale,
~ √(hσ²/N) in moment units, as `ito_cg` uses), not the end-to-end statistical error (~ 1/√M). An error that is
acceptable once, at the end, is not acceptable at every step, because accepted per-step residuals are
systematic and accumulate. The target's statistical error is the scale for evaluating the final samples, not
for deciding whether to correct at a given step.

## 6. §6 Open questions, item 1 — replaced

1. A metric that tests θ itself, not a cancellation-prone sum: the entropy integral mixes a generic positive
   bias with shrinkage (§4 item 2). Candidates: the split (copies size) and θ_φ error at larger Nhσ², where
   per-step noise no longer dominates; or the entropy bias relative to the no-duplicate baseline of the same
   method. Then: does the best ridge λ move with the problem while ito_cg needs no change? Not tested.
