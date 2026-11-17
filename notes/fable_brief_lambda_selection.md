# Brief: how should λ of the time-regularised Θ be chosen?

Written 2026-11-13. Facts are marked **[measured]**, **[derived]** (from the code, not measured) or **[inferred]** (my reading, untested).
You know how the moment-guided diffusion (MGD) SDE works; the code pointers are at the end. **This is a theoretical question**: is the rule we use to
choose λ the right one for what we do with Θ, and if not, what should replace it? An answer that can be checked with data we already have is preferred
(each new SDE run of the turbulence model costs about 18 h of H100).

## 1. What Θ_reg is and what it is used for

MGD transports n1 walkers from noise (t = 0) to the maximum-entropy model (t = 1) on a grid of n nodes. At every node k the corrector gives a per-step
multiplier θ_k (r components, one per statistic φ_i), the minimiser of `1/2 θ^T M_k θ - b_k^T θ`, with
`M_k = E_w[∇φ ∇φ^T]` (walkers at the node) and `b_k = (m_{t_k} - φ̄(y_k)) / (h σ²)` (moment target of the interpolant minus the walkers' moments). **[derived]**
θ_k is noisy (finite n1, one SDE path), so we replace it by a curve Θ(t) that minimises (moment mode, uniform weights; code sign, `θ_code = -θ_MGD`)

    E(Θ; λ) = Σ_k a_k [ 1/2 Θ_k^T M_k Θ_k - b_k^T Θ_k ]                         (data term, a_k = trapezoid weights in t)
            + λ Σ_cells dt [ 1/2 Θ'^T C Θ' - ṁ^T Θ' ],   Θ' = (Θ_{k+1} - Θ_k)/dt   (time term)

with `C = (Σ_k + Σ_{k+1})/2`, `Σ_k = Cov_w(φ)` (walkers), `ṁ = d/dt E[φ(I_t)]` (pathwise, from the interpolant). The time term is the normal equation of time-score
matching on the walker law, `Σ_w Θ' = ṁ`, so it is not a plain smoothness penalty: its own minimiser is a Θ whose derivative obeys the moment dynamics. One
scalar λ for all r statistics. At λ = 0, Θ_k = θ_k (per-node solves); λ → ∞ forces Θ' = Σ^{-1} ṁ everywhere, with the level set by the data term.
Solved exactly (block-Thomas, float64) for each λ. **[derived]** (`codes/sde_routines.py:88` `reg_energy_weights`, `:1114` `_solve_regularised_thomas`)

Θ_reg is used twice **[derived]**:
- **log p of signals at t = 1**: `-log p(x) = -Θ(1)·φ(x) + log Z`. Only Θ near t = 1 matters for the *differences* of log p between signals.
- **log Z and the entropy bound**: integrals of Θ over the whole transport (from the noise end).

## 2. The rule in use: cross-seed CV

For one model ("arm") we run S = 4 or 5 MGD runs on **the same data and potentials, differing only in the random seed** (independent walker noise; same
data x1). Seed A's system is solved for each λ on a decade grid (here 0, 1e-6, …, 100) → Θ_A(λ). Θ_A(λ) is scored with **seed B's data term**:

    L_AB(λ) = Σ_k w_k [ 1/2 Θ_A,k^T (M_B,k + ridge·diag M_B,k) Θ_A,k - b_B,k^T Θ_A,k ]     (statistics dead in B at node k set to 0)

summed over all S(S-1) ordered pairs, minimised over the grid. Three windows: whole transport, t ≥ 0.99, t ≥ 0.999. **Decision (2026-10-05): the λ of the
t ≥ 0.99 window is used for both log p and log Z/entropy** (one method for the whole paper). (`lam_selection_criterion/turb_xseed.py`)

The argument we rely on **[derived]**: B's noise is independent of Θ_A, so up to a λ-independent constant, `E[L_AB(λ)]` is the error of Θ_A(λ) in B's data
metric, with no assumption on how the noise of one run is correlated in time. It was chosen over an even/odd node split and over block CV because the
per-step noise is strongly correlated in time in the bulk of the transport (see §3).

## 3. Evidence so far

**Bimodal 1-D benchmark with known truth [measured]** (10 datasets × 3 seeds per β, n = 10⁶, σ = 10; metric D_KL(p_θ* ‖ p_θ̂) at t = 1, ratio to the best
λ of the grid for the same run, median over 30 runs; pass ≤ 10×, flag > 3×):

| rule | β = 0.1 | β = 0.9 | β = 1.9 (deep wells) |
|---|---|---|---|
| cross-seed CV, t ≥ 0.99 (**chosen**) | 1.2× | 1.9× | 117× |
| cross-seed CV, whole transport | 1.8× | 11.7× | 140× |
| even/odd CV | 1.8× | 11.7× | 140× |
| block CV, 200 nodes | 1.8× | 1.9× | 50× |
| Guth et al. weights, λ = 1 (no data) | 72× | 244× | 9.4× |

- Noise of θ_t along t (same benchmark): integrated autocorrelation time 62 / 152 / 175 nodes on t ∈ [0.1, 0.9) (β = 0.1 / 0.9 / 1.9), about 1 for t ≥ 0.99.
- The slow error of θ_t on [0.1, 0.9] is 92–100 % **common to the seeds** (β = 0.9, 1.9); at t ≥ 0.99 about 1/3 = independent. At β = 1.9 every rule's final θ
  is biased to the same side for all 30 runs; small-well mass: truth 5.0 %, chosen rule 9.5 %, raw θ_t 55 %. We read this as an MGD bias shared by the seeds,
  invisible to any rule that only compares runs, and accepted it.
- The best λ for the end (t → 1) is smaller than the best λ for the bulk (oracle tuned on 0.05 ≤ t ≤ 0.95 is much worse at t = 1).
- log Z / entropy were **never validated**: only the KL at t = 1 was.

**Turbulence [measured]** (M = 256, nt 55000, two_phase grid ending at 1-t = 5e-5, step h = 7.6e-9 at the end, n1 = 10000, σ = 3.5, ridge 1e-4):
- noSm30 arm (r = 258), ridge 1e-2: λ = 1e-3 in all three windows; log p across seeds: median correlation 0.998 at λ = 1e-3, -0.20 at λ = 0.
- nested_M3 (5 seeds, finer grid): selected 1e-4 (whole), 3e-5 (t ≥ 0.99), 1e-5 (t ≥ 0.999). Pooled loss minus min, t ≥ 0.99:
  `0: 865, 1e-7: 16.3, 1e-6: 5.04, 1e-5: 0.639, 3e-5: 0, 1e-4: 1.1, 1e-3: 54.9, 1e-2: 2.17e3`. Each of the 20 pairs picked 3e-5.
- ggd_mass_final (4 seeds): selected 1e-4 in all windows, but the curve is **not unimodal**, t ≥ 0.99:
  `0: 1.14e3, 1e-6: 4.73e3, 1e-5: 46.1, 1e-4: 0, 5e-4: 394, 1e-3: 40.5, 1e-2: 3.07e4`. λ = 1e-6 is worse than λ = 0, and 5e-4 worse than 1e-3. Unexplained.
- Current case (B+Scat4, r = 341, 5 seeds): 9 of 10 λ solved for every seed, λ = 100 missing for one seed. Each solve costs 30–120 min on CPU.

## 4. The questions

1. **Is cross-seed CV the right estimator, and of what?** The seeds share the data x1 and the potentials. E[L_AB] estimates the error of Θ_A against
   the *expected per-step θ* of the MGD process (what seed B's data term targets), not against the max-ent θ*. Is that the right target here, given the
   shared bias of §3? Is the claim "no assumption on time correlation" correct, given that b_B and M_B both come from seed B's walkers and that M_B ≠ M_A?
2. **Is the data metric M the right metric for what we use?** M = E_w[∇φ ∇φ^T] is a score (Fisher-type) metric. An error δΘ at t = 1 changes log p of a
   signal by `δΘ·(φ(x) - E φ)`, whose variance over signals is `δΘ^T Cov_data(φ) δΘ`, not `δΘ^T M δΘ`. Should the score at t = 1 use Cov(φ) (or the KL
   between p_Θ and p_Θ+δΘ, i.e. the Fisher metric Cov_p(φ)) instead of M? What would the CV loss look like in that metric with the same independence argument?
3. **One λ for log p and log Z.** log p needs Θ near t = 1; log Z integrates Θ over the transport. The bimodal test says the end wants less smoothing
   than the bulk. Is one λ defensible? Would a λ(t) (e.g. one per phase of the grid, or λ scaled with the step size / noise level) be better motivated?
   How would one choose λ for log Z without a known truth?
4. **One scalar λ for all r statistics.** The time term is in the Σ metric, the data term in the M metric; the statistics have very different noise
   (fine-scale scalar windows vs L_6 vs Scat4). Is a single λ justified, or should λ be per family (and how would cross-seed CV choose several λ)?
5. **The grid and the minimum.** A decade grid, argmin, a grid-edge warning. Is E[L(λ)] expected to be unimodal in λ? What could make the measured curve
   non-unimodal (ggd_mass_final), and does that point at a problem with the loss or with the solves? Given an interior minimum with the loss rising on both
   sides by large amounts (as in nested_M3), is it safe to skip λ values far from it (here λ = 100 when the minimum is near 1e-4)?
6. **Uncertainty of the chosen λ.** Pooled sum vs per-pair argmin; how to attach an uncertainty to λ and propagate it to log p and to the entropy bound.
7. **Alternatives** you would rank above ours, and why: block CV, Guth et al.'s fixed weights (λ = 1, no tuning), a GCV/SURE-type criterion from one run,
   a Bayesian evidence (λ as the precision of a Gaussian smoothness prior), a discrepancy principle with a noise model, or something else.

## 5. What I would like back

- A verdict on the rule (right, right for log p only, wrong), with the conditions under which it holds.
- For each question, the correct statement and its reason (short derivations welcome).
- If you propose a change: the exact loss/criterion, what it assumes, and **a test that can be run on data we already have** (saved systems of 4–5 seeds per
  arm on SCRATCH: per node M_k, b_k, Σ_k, ṁ_k, live mask; Θ(λ) for the grid above; the bimodal benchmark has the exact truth).

## 6. Files (to attach)
See the list given with this brief.
