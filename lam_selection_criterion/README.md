# Known-truth test of the λ choice for Theta_reg: Gaussian, d = 8

`gauss_d8_lamtest.py` runs the **real** MGD pipeline (the same code as the turbulence runs:
`SDE.forward_regularised` → saved moment-mode system → `resolve_theta_reg.build_inputs` →
`SDE._solve_regularised_thomas`, and `lam_select_cv.lam_path` for the current λ rule) on a
problem where the exact θ(t) is known at every t, and scores every λ rule against it, **at the
end of the transport** (where Θ is used for log p) and in time windows.

## What Guth et al. propose, what we do, and why they differ

Guth, Kadkhodaie & Simoncelli, *Learning normalized image densities via dual score matching*,
arXiv:2506.05310 (v3, Jan 2026). Checked verbatim on 2026-10-02.

| | Guth et al. (paper) | Ours (moment mode) |
|---|---|---|
| object learned | one network U_θ(y, t) for **all** noise levels t (t = noise **variance**, y = x + N(0, t Id)) | a free vector Θ_k per time node; nothing ties nodes together except the time term |
| "space" term | denoising score matching, eq. 3: per-sample regression of ∇_y U on (y − x)/t, using the **pair** (x, y) | the MGD corrector equation M_k Θ = b_k (moment mismatch / (h σ²)): **no pairs** (walkers are independent of the noise Z), only moments |
| "time" term | time score matching, eq. 5: per-sample regression of ∂_t U on d/(2t) − ‖y − x‖²/(2t²) (eq. 4) | normal equation of time-score matching on the walker law, Σ_w Θ' = ṁ (moments only) |
| why the time term is there | DSM fixes U only up to an additive constant that can differ **between modes**; TSM fixes the energy levels across modes and keeps the normalisation constant across t ("mass is conserved through the diffusion") | to **reduce the noise** of the per-step Θ_k: the data term alone already determines Θ_k at each node (if G is invertible) |
| smoothness in t | comes from the network (shared across t) | comes **only** from the time term |
| relative weight | E_t[(t/d)·DSM + (t/d)²·TSM], p(t) ∝ 1/t (eqs. 7–8); weights chosen so each term is unit-less; "we found no significant improvement from tuning a tradeoff hyperparameter" (one remark, **no ablation in the paper**, not even with/without TSM) | λ is free. Two weightings: `uniform` (plain quadrature in t, ours) and `guth` (Guth's weights mapped to our Cos time; **the mapping is ours**, notes/guth_reg_audit_0930) |
| choice of the weight | none (fixed by units) | **ours**: held-out loss on even/odd nodes + amplitude veto (`codes/lam_select_cv.py`), not from the paper |
| normalisation | log Z fixed at t_max (Gaussian end), eq. 9 | log Z / entropy from integrating dH from t = 0 (noise end); the time term has a free c_t and does not normalise |

Why the "no tuning" remark may not transfer: in Guth's setting both terms are satisfied by the
same true energy (the model is the exact noisy density), so at the optimum the weight does not
matter; the terms identify *different* things (DSM: ∇_y U; TSM: levels across modes). In ours
the data term already identifies Θ, the time term is a smoother, and λ is a genuine
bias–variance knob. In addition, at finite σ the walker law is not in the exponential family,
so the two terms aim at slightly different θ (0.13–3.6 % in the 1-D bimodal). In **this**
Gaussian test the walker law *is* in the family, so both terms are exact: this is the
favourable case for the time term (and for Guth's "no tuning").

## The rules compared (per seed)

| rule | weights | λ | from |
|---|---|---|---|
| `lam0` | – | 0: the per-step θ_t (with the run's ridge) | the MGD run itself |
| `guth_lam1` | `guth` | 1, no selection | Guth's recipe, transferred through our mapping |
| `uniform_cv_eo` | `uniform` | even/odd held-out loss + veto | current `lam_select_cv.lam_path`, unchanged |
| `uniform_cv_block` | `uniform` | block held-out loss + same veto | new (D10) |
| `guth_cv_eo`, `guth_cv_block` | `guth` | as above | to see whether selecting λ helps Guth's weights |
| `*_oracle_end`, `*_oracle_mid` | both | best λ against the truth, at the end / for 0.05 ≤ t ≤ 0.95 | reference only (needs the truth) |

## Own decisions (not said by you or by the paper), with the argument against each

- **D0. Gaussian case only for now** (you asked for d = 8 Gaussian). The misspecified case
  (rotated product of 1-D bimodals, x² and x⁴ statistics) is not written yet.
- **D1. Run the real pipeline, not a re-implementation.** Against: the SDE's per-step Python
  overhead makes the run slower (minutes per seed). Holds: the question is whether *our* code
  and *our* rule are right; the truth is analytic and independent of the code.
- **D2. Statistics φ_ij = x_i x_j, i ≤ j (r = 36), no linear terms.** Against: no near-copies,
  no thin regions, walker law exactly Gaussian, so this cannot reproduce the turbulence
  difficulties and is the easiest case for the time term. Holds for a first, *necessary* test:
  a rule that fails here is wrong; one that passes still needs a misspecified case (proposed
  next: rotated product of 1-D bimodals with x², x⁴).
- **D3. Data covariance C_ij = ρ^|i−j|, ρ = 0.9 (stationary, like a signal), tr C / d = 1.**
  Against: arbitrary; the spread of eigenvalues (~0.07 … 5.9 here) sets how ill-conditioned G
  is (at ρ = 0.9, d = 8: eigenvalues 0.055 … 6.2, checked). Holds; `--rho` changes it. Unit trace matches the global normalisation of the turbulence data.
- **D4. The same data x1 for every seed** (`--data_seed`); the seed changes Z, the walkers and
  the SDE noise. Against: the across-seed check then cannot see the error from sampling x1.
  Holds: that error is not something the time regularisation can fix, and the turbulence seeds
  share x1 too.
- **D5. Truth = θ of N(0, Σ̂_t), Σ̂_t the empirical second-moment matrix of this seed's
  interpolant samples** (exactly what the corrector targets): θ_ii = −½ (Σ̂_t⁻¹)_ii,
  θ_ij = −(Σ̂_t⁻¹)_ij (code sign, p ∝ exp(+θ·φ)). Against: the population Σ_t is "the" truth.
  Holds: the difference is a finite-sample error common to every λ and every rule, which would
  only add a floor; the end-point error against the population is printed too.
- **D6. Error = local KL, ½ eᵀ F e with F the exact Fisher information at θ*.** For this family
  it is tr((Δ Σ̂)²), Δ the symmetric matrix of e (Δ_ii = e_ii, Δ_ij = e_ij / 2), in nats.
  Against: second order only; for large errors (λ = 0 near the end) it is not the true KL, and
  Θ need not even define a normalisable density. Holds for ranking rules; the relative version
  √(KL(e) / KL(θ*)) is printed for comparison with the 1-D table (KL(θ*) = d/4).
- **D7. Primary score at the last node (1 − t = 5e-5), plus time-weighted window averages**
  [0.05, 0.95] (as the 1-D validation), [0.95, 0.99), [0.99, 0.999), [0.999, end]. Against: one
  node is a noisy score. Holds with 5 seeds; the windows show where each rule wins or loses.
- **D8. Grid, σ, n, ridge as the noSm30 turbulence runs** (two_phase nt 60000, n_bulk 10000,
  t_switch 0.9, gap_end 5e-5, σ = 3.5, n = 8500, regularization 1e-4, solve_float64). Against:
  d = 8 vs 256 changes the noise per statistic and Guth's weights (they contain d); not every
  dimensionless number can match. Holds for what produces the step-to-step noise (grid, σ, n).
- **D9. "Guth's recipe" = our `guth` weights with multiplier λ = 1.** Against: the mapping from
  Guth's (y, t) to our Cos time and our moment terms is our derivation; their DSM/TSM are
  per-sample regressions with paired data, ours are moment equations. Holds as the closest
  transfer, but the label means "Guth's relative weights", not "Guth's estimator".
- **D10. Block cross-validation (new).** Nodes are cut into blocks of B; two folds hold out
  alternate blocks (both folds used). Held-out nodes stay in the solve with zero data weight
  (M_k = 0, b_k = 0), so Θ there is the model's own prediction from the time term, instead of a
  linear interpolation. Score = the data term on the held-out nodes with trapezoid weights, as
  `lam_select_cv`. B = max(20, ⌈10 τ_int⌉), τ_int from the residual about a moving average
  (the estimator usable on turbulence, where the truth is unknown). Against: (i) λ = 0 has no
  prediction for held-out nodes, so block CV cannot pick λ = 0 (the grid has 1e-6 instead);
  (ii) long held-out blocks favour larger λ (the time term carries the prediction) — intended,
  it is what we rely on at t = 1; (iii) the factor 10 is arbitrary (`--block` overrides).
  Holds.
- **D11. The held-out loss uses trapezoid weights h_k for both weightings**, as `lam_select_cv`.
  Against: for `guth` the energy's data weight is h_k w_D(t); scoring with h_k measures a
  different objective. Holds for comparability with the current code; noted.
- **D12. λ grid 0, 1e-6 … 1e3 (decades), the same for both weightings.** Against: coarse (the
  oracle is resolved to ± half a decade). Holds for a first pass; warnings when a rule picks a
  grid edge.
- **D13. Noise diagnostics from the exact residual θ_t − θ*:** autocorrelation (lags 1–50) and
  τ_int per window, plus E[z²] of the noise model Cov = 2 M⁻¹/(n h σ²). Also the moving-average
  τ_int estimator, to check it against the exact one (it is the one proposed for turbulence).

## How to run

Smoke (seconds, local): see the script docstring. Full (Jean Zay), one array task per seed:

    bash lam_selection_criterion/launch_gauss_d8.sh                  # seeds 900-904
    SEEDS="900" bash lam_selection_criterion/launch_gauss_d8.sh      # one seed first, to time it

Outputs, all in `results/gauss_d8/` (override with `OUT=`): `seed_<s>.pt` (~6 MB each),
`args.json` (the problem settings), slurm logs, and the figure written by the combine step.
The ~0.6 GB system per seed is written to `$SCRATCH/.../lam_selection_criterion/systems` while
the seed runs and deleted afterwards (unless `--keep_system`). Combined table, on a login node:

    python lam_selection_criterion/gauss_d8_lamtest.py --out lam_selection_criterion/results/gauss_d8 --combine_only

More seeds: launch again with the same `OUT` and new `SEEDS`; the combine step pools every
`seed_*.pt` in `OUT`. Different settings (`EXTRA="--rho 0.8"`, ...): use a new `OUT`; the script
refuses to put seeds with different settings in the same folder.
