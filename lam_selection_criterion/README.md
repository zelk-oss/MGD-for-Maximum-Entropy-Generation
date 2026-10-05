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

## Result of the Gaussian test (2026-10-05, nt 3000 two-phase, n 5000, σ 1, 5 seeds, job 617402)

Uniform weights + even/odd CV picked λ = 1 in 5/5 seeds, median end-point KL equal to the uniform
oracle's (0.0036 nats; 1.9× the max-likelihood floor of this dataset against the population). But
the per-step noise was **independent** (τ_int 0.6–1.2, E[z²] 0.95–0.99): the favourable case for
even/odd CV, not the turbulence one. Hence the bimodal test below.

---

# Known-truth test of the λ choice: the paper's 1-D bimodal benchmark

`bimodal_lamtest.py`: same pipeline and same rules as the Gaussian test, on the benchmark of the
draft's Figure 1, log p\*(x) = −β x⁴ + 5β x² + β x/2, φ_a = x^a / a (a = 1..4), at the paper's
settings (n = 10⁶, σ = 10, nt = 5·10⁴). At σ = 10 the 1-D validation of 09-30 found the per-step
noise correlated in time (lag-1 0.18–0.99) and even/odd CV picking λ 100–1000× too small, so this is
the regime the turbulence runs are expected to be in. `analyse_bimodal_lamtest.ipynb` shows what
each rule does to θ(t), to the final θ̂, to the density and to the weight of each well.

Files: `bimodal_lamtest.py` (one run = one dataset at one β), `thomas_batched.py` (the moment-mode
Thomas solve for all λ / CV masks in one sweep, r small only), `check_thomas_batched.py` (checks it
against `SDE._solve_regularised_thomas`; PASS 2026-10-05, see its docstring for the tolerances),
`launch_bimodal.sh`, `analyse_bimodal_lamtest.ipynb`.

## Own decisions, with the argument against each

- **B1. Shared SDE (`codes/sde_routines.py`, moment mode) instead of the bimodal port**
  (`bimodal_experiment/sde_routines_scalar_reg.py`). For: it is the turbulence pipeline and the
  draft's eqs. 5–6; the port still has the legacy time term. Against: results are not directly
  comparable with the June/September bimodal runs, and the shared SDE's cost at n = 10⁶ is unmeasured.
- **B2. Two-phase grid** (n_bulk = nt/6, t_switch 0.9, gap_end 5·10⁻⁵, the turbulence shape) instead
  of the paper's grid. For: the λ question is about the turbulence grid. Against: the paper's
  Figure 1 numbers come from another grid.
- **B3. Truth.** End point: θ\* and the exact D_KL(p_θ\* ‖ p_θ̂) by quadrature on [−10, 10]
  (8001 points, as `kl_bimodal.ipynb`); floor = max-likelihood fit of each x1. Along the run:
  θ_ME(m_t), the max-ent fit of the interpolant's empirical moments, by Newton on the same grid.
  Against: MGD's θ_t targets the Fokker–Planck projection, not θ_ME (0.13 % apart at σ = 10 in the
  09-30 check), so "noise" along the run includes that slow difference.
- **B4. Nodes where θ_ME does not exist on R are excluded** (θ₄ ≥ 0; a β = 1.9 interpolant had them
  for t in [0.20, 0.36]: skewed, nearly Gaussian kurtosis, the known non-existence case of quartic
  max-ent). They are dropped from every θ_ME-based number; the noise uses the longest valid stretch
  per window. Against: the mid-run windows then miss part of [0.05, 0.95).
- **B5. Block CV at fixed lengths B = 20, 200, 2000 nodes** instead of an automatic B from an
  estimated τ_int. For: the Gaussian test showed the moving-average τ_int estimator reads far too
  low; fixed B shows directly how long the blocks must be. Against: 2000 nodes is ~4 % of the grid,
  coarse near t = 1 where the grid is dense in log(1 − t), not in t.
- **B6. Oracle at the end against θ\*** (the paper's metric) rather than against θ_ME(t_end).
  Against: it then also "fits" the sampling error of the dataset, which no λ can remove; both
  numbers are reported (`end_true`, `end_sample`).
- **B7. 3 β (0.1, 0.9, 1.9) × 10 runs**, each run its own dataset (seed k) as the paper's ensemble.
  Against: 10 runs give a rough mean; the paper used 100.
- **B8. λ grid 10⁻⁶ … 10³ in decades** (uniform-weight units; not comparable with the legacy grid of
  `bimodal_sweep.sh`). Against: one point per decade; edges are flagged when a rule picks one.
- **B9. Ridge 10⁻⁴ in the run and the solves** (turbulence reg1e-4 arm).

- **B10. Cross-seed CV** (added 2026-10-05, after the first result below). Every dataset is run
  with 3 SDE seeds; each seed's Θ(λ) is scored with the data term of the other seeds (independent
  noise), pooled over the 6 ordered pairs, one λ per dataset and weighting. No assumption on how the
  noise of θ_t is correlated in time within a run (user's requirement). Scored on the whole
  transport and on t ≥ 0.99 / 0.999 (user: compare both; the end matters for log p, the whole
  transport for log Z and the entropy). Against: the seeds also draw different Z, so their targets
  differ early in the transport (stored as `xseed_target_gap`, vanishes at t = 1); it needs ≥ 2 runs
  per dataset (the turbulence arms have 5 seeds on the same data).

## First result (job 619951, one seed per dataset, 2026-10-05)

Error D_KL(p_θ\* ‖ p_θ̂) at t = 1, mean over 10 datasets (floor = max-likelihood fit, 1.7–2.0e-6 = r/2n):

| rule | β = 0.1 | β = 0.9 | β = 1.9 |
|---|---|---|---|
| oracle, uniform / Guth | 2.1e-6 / 2.3e-6 | 5.9e-6 / 4.5e-6 | 2.0e-4 / 1.1e-4 |
| even/odd CV, uniform (current) | 3.6e-6 | 4.0e-5 (λ = 1e-6 = grid edge, 10/10) | 0.020 (λ = 1e-6, 10/10) |
| block CV B = 200, uniform / Guth | 3.6e-6 / 2.3e-6 | 1.4e-5 / 1.5e-5 | 6.6e-3 / 3.1e-3 |
| Guth, λ = 1 | 1.0e-4 | 5.9e-4 | 1.1e-3 |

Noise of θ_t: correlated over most of the transport (τ_int ≈ 60–175 steps in [0.1, 0.9), growing with
β; E[z²] 2–20), independent only for t ≥ 0.99 (τ_int ≈ 1, E[z²] ≈ 0.85). Even/odd CV fails as soon as
the wells deepen; no tested rule is reliable at β = 1.9 (best: Guth block CV, 27× the oracle), hence B10.

## How to run

Smoke (seconds, local): see the script docstring. Full (Jean Zay):

    bash lam_selection_criterion/launch_bimodal.sh      # 10 tasks (dataset k): 3 betas x 3 seeds each, ~63 min
                                                        # (dev QoS: at most 10 submitted jobs per user)

Outputs in `results/bimodal_xseed/beta_<b>/run_<k>_s<s>.pt` (~1–2 MB each, incl. the Θ(t) curves on
~3000 nodes for the notebook), `args.json`, slurm logs. The systems (~8 MB each) are kept on
`$SCRATCH/.../systems_bimodal_xseed`, so new λ rules need no new SDE run. The first result (one seed
per dataset, no cross-seed rule) is in `results/bimodal/`. Combined table, on a login node:

    python lam_selection_criterion/bimodal_lamtest.py --out lam_selection_criterion/results/bimodal_xseed --combine_only

Then run `analyse_bimodal_lamtest.ipynb` on Jean Zay, next to the results (send it there with
`rsync -avhR lam_selection_criterion/analyse_bimodal_lamtest.ipynb jz:<project path>/`; the push
script skips `.ipynb`). It finds the repo root by itself and reads `results/bimodal_xseed` by default.
