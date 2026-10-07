# Early-stopped Krylov solvers instead of a ridge in MGD — full recap (2026-10-06)

Self-contained summary of what was implemented, how, the experiments run, all results, and the open
questions. Written to be read without access to the code. Paper notation: Moment Guided Diffusion
(Lempereur et al., arXiv:2602.17211); equation numbers refer to it.

---

## 0. The question

At every time step, MGD solves two linear systems with the Gram matrix of the potential gradients. The code
regularises them with a fixed ridge λ, which has to be tuned (and on turbulence data it sometimes had to be
chosen large just to keep the matrix from being singular). **Question: is there a regularisation that does
not need tuning?** Candidate: solve iteratively (conjugate gradients / MINRES) from θ = 0 and stop early,
when the remaining mismatch is at the level of the noise in the right-hand side ("discrepancy principle"),
with the noise level computed rather than chosen.

---

## 1. The existing solver (codes/sde_routines.py, NOT modified)

### 1.1 One time step (`SDE.iteration_step_projection`)

N walkers x_k^i, step h_k = t_{k+1} − t_k, volatility σ, r statistics φ = (φ_1..φ_r), M interpolant samples
I_t^s = cos(πt/2) x_0^s + sin(πt/2) x_1^s (Cos interpolant). x_0^s (Gaussian noise) is drawn **once** at
initialisation and x_1^s (data) is fixed, so the target moments m̂_t = mean_s φ(I_t^s) are one fixed smooth
path in t (their sampling error does not average out over steps).

1. **Predictor** (eq. 18): solve Ĝ(x_k) η = ṁ̂_{t_k}, with
   - Ĝ(x) = (1/N) Σ_i ∇φ(x^i) ∇φ(x^i)^T (r × r Gram matrix of the gradients);
   - ṁ̂_t = (1/M) Σ_s ∇φ(I_t^s)·İ_t^s, the pathwise estimate of d/dt E[φ(I_t)] (computed at the end of the
     previous step and cached).
   Then y_k^i = x_k^i + h η^T∇φ(x_k^i) + √(2h) σ ξ_k^i.
2. **Corrector** (eqs. 19, 21): the code solves
   Ĝ(y_k) θ_raw = rhs_k,  rhs_k = m̂_{t_{k+1}} − mean_i φ(y_k^i) = −ρ_k   (moment units, NO division by hσ²),
   moves x_{k+1}^i = y_k^i + θ_raw^T ∇φ(y_k^i), and stores θ_code = θ_raw / (hσ²) = −θ_MGD.
   Equivalent to the paper's form Ĝ'θ = b, b = ρ/(hσ²), x_{k+1} = y − hσ²θ^T∇φ(y); consequence: the residual
   of the raw system, Ĝ'θ_raw − rhs, is directly the moment-unit residual hσ²·u, u = Ĝ'θ − b.
3. **Entropy increment**: dH_k = −θ_code · ṁ̂_{t_{k+1}} (= +θ_MGD·ṁ); the entropy estimate is Σ_k h_k dH_k.

### 1.2 How each solve is done today (`compute_eta` :1280, `compute_theta` :1381, inner `_solve`)

1. **Dead potentials** (`_live_potentials`): statistics with diag(Ĝ) = 0 (or below a per-potential floor)
   are removed; their coefficient is set to 0 (MGD eq. 47). Non-finite Ĝ or rhs raises an error.
2. **Jacobi scaling**: D = diag(Ĝ), A = D^{-1/2} Ĝ D^{-1/2} (unit diagonal), symmetrised.
3. **Ridge**: solve (A + λI) z = D^{-1/2} rhs with `torch.linalg.solve`, θ = D^{-1/2} z.
   In original coordinates: (Ĝ + λ diag Ĝ) θ = rhs. λ = `regularization` (default 1e-4); the same λ for η
   and θ. λ = 0 is the plain direct solve. Optional `solve_float64` upcasts the solve.
4. `_log_cond` (diagnostic only): every k-th step logs the eigen-spectrum of A, the projections of rhs on
   the eigenvectors and the rhs-noise variance along them.

A separate time-regularised estimator Θ_reg (`forward_regularised`, block-Thomas solve of a quadratic energy
over all steps) builds on these per-step quantities; it was left untouched.

---

## 2. What was implemented

Design choice (user): a **parallel implementation**; `sde_routines.py` is untouched.

| file | content |
|---|---|
| `codes/krylov.py` | solvers `cg`, `minres`; stopping rules; noise weights |
| `codes/sde_krylov.py` | `class SDEKrylov(SDE)`: overrides `compute_eta`, `compute_theta`, `compute_rhs_dt_phi_I_t`; `forward_regularised` refuses non-ridge solvers |
| `krylov_stopping/tests/` | unit tests (all pass) |
| `krylov_stopping/gauss_d8_krylov.py` | experiment driver (both experiments), tables, plots |
| `krylov_stopping/neardup_regularisation.ipynb` | final figures for the near-duplicate experiment |

### 2.1 Solvers (`codes/krylov.py`)

- `cg(matvec, b, stop_fn, max_iter)`: Hestenes–Stiefel conjugate gradients, x_0 = 0.
- `minres(matvec, b, stop_fn, max_iter)`: Paige–Saunders (1975) MINRES, unpreconditioned, ported from
  scipy's structure. The residual vector is maintained by the recurrence r_j = s_j² r_{j−1} − φ̄_j c_j v_{j+1}
  (no extra matrix product); its norm is non-increasing.
- `matvec` is a closure (explicit Ĝ now; a matrix-free Ĝv = (1/N)Σ∇φ(y^i)[∇φ(y^i)^T v] would need a JVP the
  potentials do not expose yet). Warm start: hook only (`x0` raises NotImplementedError).
- `stop_fn(x, residual, j) -> (stop, statistic)` is evaluated at every iterate **including j = 0** (x = 0,
  residual = b), so a step can return θ = 0. Returns (x, j, history of the statistic).
- Default `max_iter` = number of live statistics.

Tests: iterates equal scipy's to ≤ 4e-13 over the first 8–12 iterations (later, both implementations drift
apart at rounding level, as finite-precision Lanczos does; residual norms keep matching); error vs
`torch.linalg.solve` 1e-14 (condition 1e3), 2e-10 / 7e-10 (condition 1e8); MINRES residual recurrence =
explicit b − Ax to 1e-11 and monotone (also on an indefinite matrix); the Itô statistic below equals the
squared A-norm error to 6e-14; `SDEKrylov` with ridge solvers is **bit-identical** to `SDE` (λ = 1e-4 and 0).

### 2.2 Stopping rules for θ (corrector)

All residuals below are in moment units (residual of the raw system Ĝ'θ_raw = rhs).

**(A) `ito_cg`** — noise = the Itô fluctuation injected by the predictor.
- Model: given x_k, the noise term of mean_i φ(y^i) is √(2hσ²)·mean_i ∇φ(x^i)·ξ^i, covariance (2hσ²/N)Ĝ;
  in b-units Cov(b | x) ≈ 2Ĝ/(N h σ²). **No free noise parameter**: N, h, σ, Ĝ are known.
- Solver: CG on the Jacobi-scaled system A z = D^{-1/2} rhs.
- Statistic: N/(2hσ²) · res^T Ĝ'^{-1} res  (= (Nhσ²/2) u^T Ĝ'^{-1} u in b-units), computed exactly from one
  Cholesky factorisation of A per step (res^T Ĝ^{-1} res = r_s^T A^{-1} r_s for the scaled residual r_s).
  It equals the squared A-norm of the CG error, the quantity CG minimises.
- Stop at the first j with statistic ≤ χ²_{r_live, 1−α}, α = 0.05.
- Remark derived during the work: at j = 0 the statistic is not pure noise. ρ also contains the
  deterministic Itô drift hσ²E[Δφ] = hσ²Ĝθ*, so E[statistic at j=0] ≈ r + Nhσ²·θ*^TĜθ*/2 = r + Nhσ² tr(C_t^{-1})/2
  for the Gaussian. Hence it stops at j = 0 when Nhσ² tr(C^{-1})/(2r) ≲ 1 (signal at noise level).

**(B) `tol_minres`** — noise = uncertainty of the target and of the ensemble.
- Σ_tol = Cov_s[φ(I^s_{t_{k+1}})]/M_eff + Cov_i[φ(y_k^i)]/N (unbiased sample covariances; M_eff = M).
- Weight W = Σ_tol in three variants: `diag` (diagonal only); `full`; `shrink` (Schäfer–Strimmer shrinkage
  toward the diagonal, optimal intensity, applied to each covariance separately, then summed).
  Variance floor: variances below 1e-6 × median variance are raised to it (full/shrink also clamp
  eigenvalues at that floor).
- Change of variables G̃ = W^{-1/2}Ĝ'W^{-1/2}, b̃ = W^{-1/2} rhs, θ = W^{-1/2}θ̃; MINRES on G̃θ̃ = b̃.
- Stop when ‖W^{-1/2} res‖² ≤ ε²: full/shrink ε² = χ²_{r,1−α}; diag ε² = r + κ√2‖R‖_F (R = correlation
  matrix of Σ_tol, κ = 2; mean + κ std of the whitened squared norm).

### 2.3 Stopping rule for η (predictor): `tol_minres` only

- Residual v = Ĝη − ṁ̂ (moment-rate units); W_η = Cov_s[ṁ̂^s]/M_eff where ṁ̂^s = ∇φ(I_t^s)·İ_t^s are the
  per-sample values of the pathwise estimator (the subclass computes and keeps them; the parent only kept
  the mean). MINRES on the W_η-scaled system, stop at ‖W_η^{-1/2} v‖² ≤ ε² (same thresholds as above).

### 2.4 Logging and what is not changed

Per step: iteration counts, final statistic and threshold, j = 0 values of both θ statistics for every
method (also ridge), wall time of the whole compute_eta/compute_theta call (diagnostics excluded),
optionally the raw Ĝ(y_k) and unshrunk Σ_tol. `_log_cond` still works on the Krylov path.
`forward_regularised` raises unless both solvers are ridge (early stopping has no fixed quadratic energy).

---

## 3. Experiment 1 — centred Gaussian, d = 8, well-conditioned

### Setup
- Statistics φ_ij(x) = x_i x_j, i ≤ j: r = 36 (no linear terms).
- Target C_1 = Q diag(eigenvalues log-spaced in [0.1, 1]) Q^T, Q a random rotation (fixed seed).
- Truth (MGD App. E): walkers stay N(0, C_t), C_t = cos²(πt/2) I + sin²(πt/2) C_1; θ^T∇φ(x) = Θx with
  Θ_ii = 2θ_ii, Θ_ij = θ_ij; Θ_MGD = C_t^{-1} (code sign: Θ_code = −C_t^{-1}); entropy change ½ log det C_1.
  Two truths used: population C_t (estimation quality) and empirical Ŝ_t = second moments of the frozen
  interpolant samples (what the corrector actually tracks; used to check the rules).
- Grid: N ∈ {1e3, 1e4}, M = N, σ² ∈ {1, 10}, nt = 1000 uniform steps → hσ² ∈ {1e-3, 1e-2},
  Nhσ² ∈ {1, 10, 10, 100}. (σ² = 100 was excluded: at hσ² = 0.1 the linearised corrector breaks down; the
  user's single-step check gave 33% variance error. For Gaussian + quadratic statistics MGD's solution does
  not depend on σ, so varying σ is legitimate here only.)
- 5 seeds; within a seed x_0, x_1, initial walkers and noise draws are identical for all methods.
- Float32 data, solves in float64. Time averages over t ≥ 0.05; calibration over t ≥ 0.1.
- 9 methods: ridge λ ∈ {1e-7, 1e-4, 1e-2}; direct (λ = 0); direct θ + tol_minres η; ito_cg θ + tol_minres η
  (full W); tol_minres θ and η with W diag / full / shrink.
- Run on Jean Zay (CPU work), 5–23 s per run.

### Results
- **Ridge level irrelevant**: λ ≤ 1e-2 gives the direct solve's numbers (Jacobi-scaled condition number of
  Ĝ: 1.0 at t = 0, 7.4 at t = 1).
- **KL(N(0, Ĉ_1) ‖ N(0, C_1))**: direct = 0.0184 / 0.0183 / 0.00178 / 0.00178 = exactly the target's own
  sampling floor d(d+1)/(4M). ito_cg / direct = 0.99, 1.10, 1.02, 1.26. tol_full and tol_shrink ≈ 3×;
  tol_diag 9–19×.
- **tol rules** stop at j = 0 in 40–97% of steps (corrector off). Their extra moment error against the
  population adds in quadrature with the target's error (√(err_tol² − err_direct²) equals their error vs the
  empirical target within 10%): the offset does not cancel the target error; it is 1.2–1.6× the statistical
  error, matching the size of their own tolerance (with M = N, Σ_tol = 2× the target's sampling variance;
  the Itô diffusion pushes walkers to the edge of the band, then a large correction: sawtooth in time).
- **Rule checks**: tol_full never exceeds its χ² threshold; tol_diag exceeds the full-W threshold in ~1/3 of
  steps (the diagonal weight misses correlations between the 36 statistics); ito_cg's residual is
  0.002–0.016 of the threshold.
- **ito_cg iterations**: mean 0.7 / 1.1 / 1.15 / 1.5; j = 0 in 31% / 1% / 1% / 0% of steps, matching the
  prediction (Nhσ² tr(C^{-1})/(2r) = 0.23 / 2.2 / 2.2 / 22). Measured mean of the j = 0 statistic 44.0 /
  111.6 / 116.1 / 812 vs predicted r + Nhσ² tr(Ŝ^{-1})/2 = 44.1 / 117.0 / 116.5 / 840.8.
- **η early stopping** (1–3 MINRES iterations): no measurable effect.
- **θ error of the direct solve** depends only on Nhσ² (2.19 / 0.66 / 0.69 / 0.22, factor √10 per decade):
  it is the Itô noise of b. It does not hurt the samples: it is exactly the corrector's response that
  removes the injected noise from the moments. Removing it (early stopping) leaves the noise in the walkers.
- **Calibration** of the Itô model (direct solve, z = (Nhσ²/2) e^T Ĝ' e, e = θ − θ*(Ŝ_t), expect mean r = 36,
  var 2r = 72): hσ² = 1e-3: 35.85 / 71.3 and 35.78 / 71.2 ✓. hσ² = 1e-2: N = 1e3 33.09 / 62.6, N = 1e4
  35.91 / 88.7 (smoke N = 200: 32.72 / 56). Fit at 1e-2: mean ≈ 0.91×36 + 3.1e-4·N. Hypothesis (not
  checked): a multiplicative O(hσ²) effect plus a deterministic O(hσ²) error in b (e.g. the residual the
  linearised corrector leaves, ~0.9% at hσ² = 1e-2 per the user's check) whose whitened weight ∝ N(hσ²)³.
  Whitening with Ĝ(x) instead of Ĝ(y) does not remove the deficit (33.4).
- **Entropy integral bias** of the direct solve (empirical truth): 0.067 / 0.236 / 0.020 / 0.187 nats; fits
  18·hσ² + 50/N within 0.004 (uniform-grid quadrature alone is exact to 1e-4).
- **Cost**: Krylov call +15–110% per step, from building Σ_tol and the Cholesky, not from the 1–3 iterations.

Conclusion of experiment 1: nothing to regularise; tol rules harmful; ito_cg nearly harmless but not free:
with θ effects isolated (§4, attribution) it raises KL by 10% / 26% at σ² = 10 (N = 1e3 / 1e4, significant)
and the entropy bias by +0.006 to +0.05 nats.

---

## 4. Experiment 2 — the same Gaussian plus near-duplicate statistics (ill-conditioned)

### Setup
- Added 36 near-copies ψ_ij(x) = s(x_i) s(x_j), s(u) = u + δu³ (r = 72), δ = **perturbation strength** (called δ,
  not ε, to avoid a clash with the stopping threshold ε²; the code flag is still `--eps`). Their gradients are within O(δ)
  of ∇φ_ij, so Ĝ is nearly singular: Jacobi-scaled condition number at C_1 = 1.0e3 (δ = 0.1) and
  8.5e4 (δ = 0.01). Mimics turbulence's near-duplicate statistics (e.g. Q = 1 vs ψ channels).
- Truth: the walkers stay N(0, C_t) (the Gaussian satisfies all constraints and is max-ent under the subset),
  so θ_φ = C_t^{-1} part, **θ_ψ = 0** at population level (also η_ψ = 0). With finite M the empirical target
  moments of ψ are not exactly Gaussian-consistent, so the exact empirical-truth θ_ψ is small but nonzero
  and unknown (the calibration for r = 72 used θ*_ψ = 0 as an approximation).
- One config: N = 1e4, M = N, σ² = 1, nt = 1000 (hσ² = 1e-3, Nhσ² = 10); δ ∈ {0.1, 0.01}; 5 seeds.
- Methods: first run the same 9; second run a dense ridge sweep λ ∈ {1e-3, 3e-3, 3e-2, 1e-1, 3e-1, 1}
  (total ridge λ: 1e-7, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1).
- New metrics: "copies size" = ‖Θ_ψ‖_F / ‖C_t^{-1}‖_F (Θ_ψ built from θ_ψ like Θ; truth 0), time-averaged;
  θ_φ error ‖Θ_φ − C_t^{-1}‖_F/‖C_t^{-1}‖_F; sum error with Θ_φ + Θ_ψ.

### Results (seed means; ± = std over seeds unless stated)

| δ = 0.01 | copies size | θ_φ err | sum err | entropy − exact (emp.) | KL ×1e3 |
|---|---|---|---|---|---|
| direct (= ridge 1e-7) | 18.6 | 19.4 | 1.07 | +0.031 | 1.776 |
| ridge 1e-4 | 15.2 | 15.9 | 0.99 | +0.028 | 1.776 |
| ridge 1e-3 | 8.45 | 8.89 | 0.81 | +0.022 | 1.776 |
| ridge 3e-3 | 5.43 | 5.72 | 0.74 | +0.018 | 1.776 |
| ridge 1e-2 | 3.11 | 3.28 | 0.69 | +0.009 | 1.777 |
| ridge 3e-2 | 1.81 | 1.91 | 0.67 | −0.016 | 1.777 |
| ridge 1e-1 | 1.01 | 1.06 | 0.64 | −0.097 | 1.777 |
| ridge 3e-1 | 0.64 | 0.65 | 0.59 | −0.293 | 1.801 |
| ridge 1 | 0.53 | 0.46 | 0.49 | −0.723 | 2.106 |
| **ito_cg** | 7.41 | 7.87 | 0.98 | **+0.0055** | 1.779 |

| δ = 0.1 | copies size | θ_φ err | sum err | entropy − exact (emp.) | KL ×1e3 |
|---|---|---|---|---|---|
| direct | 1.56 | 2.39 | 1.01 | +0.032 | 1.776 |
| ridge 1e-3 | 1.50 | 2.31 | 0.99 | +0.029 | 1.776 |
| ridge 1e-2 | 1.20 | 1.89 | 0.89 | +0.006 | 1.775 |
| ridge 3e-2 | 0.92 | 1.48 | 0.79 | −0.034 | 1.774 |
| ridge 1e-1 | 0.62 | 1.01 | 0.66 | −0.139 | 1.781 |
| ridge 1 | 0.28 | 0.46 | 0.44 | −0.769 | 2.442 |
| **ito_cg** | 0.70 | 1.23 | 0.88 | **+0.0054** | 1.777 |

Other facts:
- ito_cg: mean 4.7 (δ = 0.01) / 2.7 (δ = 0.1) CG iterations; never hits max_iter; +3 ms/step (+11%).
- tol rules: corrector off (j = 0) in ~92% of steps; small copies size (0.2–0.6) only because θ = 0; KL 4–20×.
- η early stopping: no effect on θ, KL or moments; it does shift the entropy estimate (−0.024 nats, see
  Interpretation item 2).
- Calibration with r = 72 (approx. truth): mean 71.2, var 138 (expect 72, 144).
- Paired differences (same seeds; mean ± standard error over 5 seeds):
  - ito_cg − direct: copies −11.2 ± 0.07 (δ = 0.01), −0.86 ± 0.007 (δ = 0.1); entropy bias −0.025 (5/5).
  - ito_cg − ridge 1e-2 (copies): +4.3 ± 0.03 at δ = 0.01 (ridge better 5/5); −0.50 ± 0.006 at δ = 0.1
    (ito better 5/5).
  - |entropy bias| − |entropy bias of ito_cg|: ridge 1e-2: +0.002 ± 0.0015 (δ = 0.01), +0.0001 ± 0.0009
    (δ = 0.1), i.e. tie; ridge 1e-3 or 3e-2 (one factor 3 away): 3–25× more bias than ito_cg.
  - On copies size, θ_φ error and sum error, the best ridge in the sweep is λ = 1 at both δ and beats
    ito_cg in 5/5 seeds; but λ = 1 costs −0.72/−0.77 nats of entropy bias and +19–37% KL.

### Interpretation

1. **Both regularisers beat the direct solve** without affecting the samples (KL unchanged).
2. **CORRECTED (cancellation check, same day): the near-zero entropy bias is NOT evidence that any method
   gets θ right.** Paired entropy-bias shifts vs the direct solve at N = 1e4, σ² = 1 (mean ± SE, 5 seeds):

   | shift vs direct | no duplicates (Exp. 1) | δ = 0.01 | δ = 0.1 |
   |---|---|---|---|
   | direct bias itself | +0.0196 | +0.0309 | +0.0320 |
   | ridge 1e-2 | −0.0251 ± 0.0002 | −0.0222 | −0.0262 |
   | direct θ + early-stopped η (`direct+eta_tol`) | −0.0040 ± 0.0001 | −0.0243 ± 0.0008 | −0.0250 ± 0.0009 |
   | ito_cg (early-stopped θ AND η) | +0.0015 ± 0.0002 | −0.0254 ± 0.0008 | −0.0266 ± 0.0008 |
   | early-stopped θ alone (ito_cg − direct+eta_tol) | +0.0055 ± 0.0001 | −0.0010 ± 0.0001 | −0.0016 ± 0.0002 |

   - The direct solve has a generic positive bias (+0.020 without duplicates; the duplicates raise it to
     +0.031). Ridge 1e-2 shifts it by a fixed ≈ −0.025 with or without duplicates: its "unbiased window" is a
     negative shrinkage bias cancelling the generic positive bias, not θ being right.
   - Without duplicates, ito_cg has the direct solve's bias (+0.021): it is not unbiased.
   - With duplicates, ito_cg's shift comes almost entirely from early-stopping **η** (the ito_cg method uses
     tol_minres for η); early-stopping θ contributes −0.001 to −0.0016. So "ito_cg lands in the unbiased
     window without tuning" is withdrawn: it is a cancellation, produced by the η rule, present only with
     duplicates.
   - The ridge window did not move between δ = 0.01 and 0.1 (λ = 1e-2 in both).

   **Attribution θ vs η, all metrics** (the `ito_cg` method early-stops both solves; paired, 5 seeds):
   - η rule alone (`direct+eta_tol` − `direct`): no measurable effect on copies size, θ_φ error, sum error,
     KL or moments (e.g. copies +0.0125 ± 0.002 on a scale of 18.6); it changes only the entropy
     (−0.024 with duplicates, −0.004 to −0.05 without).
   - Early-stopped θ alone (`ito_cg` − `direct+eta_tol`): all the θ gains (copies −11.2 ± 0.07 at δ = 0.01,
     −0.86 ± 0.007 at δ = 0.1; θ_φ error −11.6 / −1.16; sum error −0.09 / −0.13); entropy −0.001 / −0.002.
   - Costs of early-stopped θ alone, Exp. 1 (no duplicates): entropy bias **increases** (+0.0055 ± 0.0001 at
     N = 1e4 σ² = 1; +0.021 at N = 1e3 σ² = 1; +0.051 at N = 1e3 σ² = 10); KL increases at σ² = 10:
     +1.84 ± 0.7 ×1e-3 (+10%, N = 1e3) and +0.47 ± 0.1 ×1e-3 (+26%, N = 1e4) — the ratios 1.10 / 1.26
     reported in §3 are real at 2.5 and 4.7 standard errors; moment tracking vs the empirical target looser by
     0.0009–0.009 (by design).
   - Not separated: ridge rows apply λ to both η and θ; no run isolates the θ ridge.
3. **Per-step θ error is minimised by heavy shrinkage** (λ = 1), because at Nhσ² = 10 the per-step θ is
   dominated by step noise; shrinking θ toward 0 removes noise faster than it adds bias. That is not a useful
   optimum (entropy and samples degrade).
4. **ito_cg does NOT fix the cancelling-pairs problem.** Its copies size is 7.4× (δ = 0.01) and 0.7×
   (δ = 0.1) the true θ, where the truth is 0. The entropy cannot see this: it uses θ·ṁ, and a statistic and
   its copy have almost the same ṁ, so only the sum θ_φ + θ_ψ enters; the cancelling split contributes at
   O(δ). "Entropy unbiased" checks the sum, "copies size" checks the split.
5. **Why ito_cg keeps the split — by construction.** In the eigenbasis of Ĝ (eigenvalues μ_i), the noise
   of b along eigenvector i has variance ∝ μ_i, so the Itô statistic Σ_i (residual_i)²/μ_i weights
   near-singular directions heavily: the rule insists on fitting the original-minus-copy direction, which is
   exactly where solving amplifies noise into θ (θ noise variance ∝ 1/μ_i). The rule protects the
   **moments** (samples), not **θ**. A rule aimed at θ would have to accept a larger residual along
   near-singular directions — a different rule, not a different threshold.

### Honest summary
- ito_cg (early-stopped θ): a tuning-free regulariser that leaves the samples unchanged and reduces the
  cancelling split 2.2–2.5× vs the direct solve, but does not remove it; on the split it beats ridge 1e-2 at
  δ = 0.1 and loses at δ = 0.01. Present it as "comparable to a reasonable ridge, without choosing λ".
  These gains are attributable to early-stopping θ (the η rule contributes nothing to them). Costs without
  duplicates: entropy bias +0.006 to +0.05 nats and KL +10–26% at σ² = 10 (§4, attribution).
- The entropy integral cannot discriminate the methods here: its near-zero values are cancellations of a
  generic positive bias (+0.02 nats) by shrinkage (ridge) or by the η stopping rule (ito_cg rows).
- tol_minres rules: a negative result, see §5.

---

## 5. Negative result: tolerance-based stopping (tol_minres) fails

`tol_minres` (diag, full and shrink weights) was the recommended default for θ going in. It fails in both
experiments, and the failure has one cause: **a tolerance set by the target's statistical error is the wrong
scale for a per-step solve.** The scale mismatch had been flagged in advance as an open issue in the notes
and was underweighted.

**What happens.**
- The corrector is switched off (j = 0, θ = 0) in 40–97% of steps (Exp. 1) and ~92% of steps (Exp. 2).
- With θ = 0, nothing compensates the Itô drift, so the walkers' moments drift systematically. When the drift
  exceeds the tolerance, one large correction resets them: a sawtooth in time (spikes in `theta_err_pop.png`).
- The drift does not cancel the target's own sampling error. Against the population, the two errors add in
  quadrature: the extra moment error is 1.2–1.6× the statistical error. KL: 3× the direct solve (full,
  shrink) to 9–19× (diag) in Exp. 1, and 4–20× in Exp. 2.
- The rule does what it claims, but that is the problem. tol_full never exceeds its χ² threshold. It just
  uses the whole allowance at every step.

**Why: the two scales.**
- Tolerance: Σ_tol ≈ Cov(φ)/M + Cov(φ)/N, a band of relative width ~ 1/√N per moment (M = N). It is the
  right scale for judging the **end result**, because the target itself is only known to that precision.
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

  Exp. 2 (N = 1e4, hσ² = 1e-3, r = 72, predicted 10): 12–15 for all three weights at both δ.
  The measured/predicted ratio is 1.05–1.8 over a 30-fold range of the prediction; the O(1) prefactor is not
  derived.

**Lesson.** The per-step stopping tolerance must be the per-step noise of the right-hand side (the Itô scale,
~ √(hσ²/N) in moment units, as `ito_cg` uses), not the end-to-end statistical error (~ 1/√M). An error that
is acceptable once, at the end, is not acceptable at every step, because accepted per-step residuals are
systematic and accumulate. The target's statistical error is the scale for evaluating the final samples, not
for deciding whether to correct at a given step.

---

## 6. Open questions / next steps

1. A metric that tests θ itself, not a cancellation-prone sum: the entropy integral mixes a generic
   positive bias with shrinkage (see §4 item 2). Candidates: the split (copies size) and θ_φ error at
   larger Nhσ², where per-step noise no longer dominates; or the entropy bias relative to the
   no-duplicate baseline of the same method. Then: does the best ridge λ move with the problem while
   ito_cg needs no change? Not tested.
2. A stopping rule targeting θ accuracy (accept residual along small-μ directions) to suppress the
   cancelling split — design needed.
3. Exactly singular Ĝ (the turbulence case where a large ridge was needed): the ito statistic uses a
   Cholesky factorisation and currently fails; it should be evaluated on the range of Ĝ (pseudo-inverse).
   CG itself handles singular consistent systems.
4. α (5%) is the only setting of ito_cg; 95% vs 99% changes the χ² threshold by ~10% (r = 72); sensitivity
   not measured.
5. Matrix-free operator for large r (needs a JVP of the potentials); CG error-norm estimate without Cholesky
   (Strakoš–Tichý: ‖e_j‖²_A = b^TA^{-1}b − Σ_{i<j} γ_i‖r_i‖²).
6. Calibration deviation at hσ² = 1e-2 (mean −8% at N = 1e3, variance +23% at N = 1e4): cause unverified.
7. Bimodal 1-D benchmark (φ = x^a/a, a = 1..4) is also ill-conditioned (Jacobi-scaled condition number at
   the target: 20 / 60 / 176 / 1.7e3 for β = 0.1 / 0.5 / 0.9 / 1.9), from the law concentrating on two
   peaks rather than from duplicate statistics; with r = 4 early stopping is coarse (j ∈ 0..4). Not run.
