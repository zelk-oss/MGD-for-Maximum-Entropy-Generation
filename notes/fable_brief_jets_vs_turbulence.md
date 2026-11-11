# Brief: moment-matching error in the MGD runs, Lagrangian turbulence vs Eulerian jets — find a strategy that fixes it with fewer time steps

Written 2026-11-11. Facts are marked **[measured]**, **[derived]** (from the code, not measured) or **[inferred]** (my reading, untested).
You know how the moment-guided diffusion (MGD) SDE works; code pointers and the files to read are at the end. A companion brief,
`notes/fable_brief_moment_mismatch.md`, goes deeper into the turbulence case (scalar windows, Itô term); this one compares it with the jets.

## 1. Goal

We sample 1-D signals (M = 256 samples) from a maximum-entropy model defined by moment constraints `E[phi_n(x)] = phi_bar_n(data)`;
the SDE transports n1 = 10000 walkers from noise (t = 0) to the model (t = 1) and enforces the moments along the way.
A run costs about 1.2 s per step (one H100) and uses 40000 to 55000 steps (13 to 19 h). **We want to run with far fewer steps, and still match the moments.**
Two experiments show a moment mismatch, with different symptoms. The question: is there one strategy (a change of the potentials, the predictor, the corrector,
the solver or the time grid) that fixes both, and which of the possible strategies is the winning one when nt must stay small?

## 2. The two experiments side by side

| | **Lagrangian turbulence** ("B+Scat4") | **Eulerian jets** ("jets_full") |
|---|---|---|
| data | 1-D turbulence velocity record, subseries 512 → M = 256 | Modane Eulerian velocity record, windows 512 → M = 256, non-consecutive windows (`--n1_spread`) |
| signals / walkers | n1 = 10000, M = 256, J = 8 | n1 = 10000, M = 256, J = 8, Q = 1 |
| sigma, regularisation, solver | sigma 3.5, ridge 1e-4 on the Jacobi-normalised Gram, lam 0, float64, Thomas | same |
| interpolant | Cos | Cos |
| time grid (`two_phase`, `codes/time_schedules.py`) | nt 55000: 5000 uniform steps on [0, 0.9], then 50000 steps with h ∝ (1−t) down to 1−t = **5e-5** (h/(1−t) = 1.5e-4) | nt 40000: 5000 uniform steps, then 35000 steps with h ∝ (1−t) down to 1−t = **7.5e-4** (h/(1−t) = 1.4e-4, from a Nyquist criterion) |
| potentials | L_6 (9), Scalar_psi_gaussianK (132 stats, 22 channels), Scalar_morlet_gaussianK (60, 9 ch), Scat4 Mod2 Real Q1 (84), Scat4 Mod2 Imag Q1 (56): **r = 341** | L_6 (9), Scat 1st order (8), Scat 2nd order (9), **Scat 3rd order Real (44) and Imag (44)**, Scat4 Mod2 Real Q1 (84), Scat4 Mod2 Imag Q1 (56), Scalar_morlet_gaussianK (45): **r = 299** |
| scalar windows | K = 4 sigmoid windows, likelihood boundaries, tail q.99, tail mass, constrained BIC C = 1, `trans_frac` 0.1 | same recipe (variant B), 9 channels, 0 pruned |
| cost | 1.22 s/step, 18.7 h | 1.17 s/step, about 13 h |
| seeds / reference | seed 900 (one seed analysed per run; the Scat4 reference run) | seeds 500 to 504 |
| a no-scalar contrast exists | no | yes (`jets_noscalar`) |

So the SDE, the solver settings, sigma, n1, M, J and the step density in the tail are the same. What differs: the data, the potential set (jets add Scat 1, 2, 3 and drop the scalar psi bank), and the end of the grid.

## 3. What is not matched

Error measure: relative error at the last step `rel = |p − e| / |e|` (p generated moment, e target moment from the interpolant at the last time).
Because some targets are tiny, I also use an "own-scale" error `|p − e| / median_family |e|`. Neither is a significance test: **no z-score (error over sampling noise) exists yet for the jets**
(for turbulence it does).

### 3a. Turbulence (one reference run)  [measured]
5 of 341 statistics have rel > 1 % (1.6 to 3.7 %), the rest have a median of 1e-7. All five are scalar statistics of the three finest channels
(psi ch 0 and 1, morlet ch 0), tied to one cut per channel (the q.99 cut: body moment, tail moment, tail mass). The error decreases steadily with channel index
(psi body moment: 3.7 %, 1.7 %, 0.51 %, 0.16 % for ch 0, 1, 2, 3), and the generated moment is above the target in all 20 worst statistics (systematic sign).
Scat4 Mod2 Real/Imag: matched (max own-scale error 9e-4 / 5e-5). L_6: 1e-6.
Likely cause (derived + measured correlations): a missing Itô term, `h sigma^2 <Laplacian phi>`, in the predictor, absorbed by a one-step Gauss-Newton corrector, for sigmoid edges 10× narrower than the step noise;
Spearman(|Itô term|, error) = +0.74. Details in the companion brief.

### 3b. Jets, `jets_full`, 5 seeds, 299 statistics each  [measured]
Raw count: **28 to 35 statistics per seed have rel > 1 %** (and 73 to 86 have rel > 0.1 %). Per family (average over seeds; own-scale error in units of the family median |e|):

| family | n | median \|e\| | median rel | rel > 1 % | own-scale > 1 % | own-scale > 0.1 % | max own-scale |
|---|---|---|---|---|---|---|---|
| Scalar morlet | 45 | 0.45 | 1.6e-5 | 0 | 0 | 0 | 1.5e-4 |
| Scat 1st order | 8 | 5.6e-2 | 5.9e-6 | 0 | 0 | 0 | 5.4e-5 |
| Scat 2nd order | 9 | 5.5e-3 | 7.6e-6 | 0 | 0 | 0 | 1.3e-4 |
| L_6 | 9 | 3.3e-6 | 1.1e-4 | 0 | 0.8 | 1.6 | 3.2e-2 |
| Scat 3rd order Real | 44 | 4.2e-6 | 9.8e-4 | 1.6 | **7.4** | **25.8** | 0.48 |
| Scat 3rd order Imag | 44 | 4.9e-5 | 9.8e-5 | 6.2 | 0 | 0.6 | 1.3e-3 |
| Scat4 Mod2 Real Q1 | 84 | 8.0e-7 | 8.3e-5 | 0 | 2.2 | 12.8 | 9.1e-2 |
| Scat4 Mod2 Imag Q1 | 56 | 2.7e-9 | **8.1e-3** | **24.4** | **24.2** | **51.6** | 0.42 |

Observations:
- **The scalar morlet bank, the 1st and 2nd order scattering are matched to better than 1e-3.** The scalar potentials are NOT where the jets fail.
- **The failure sits in the odd/small higher-order scattering statistics:** about 24 of the 56 Scat4 Mod2 Imag statistics are off by more than 1 % of their family scale, up to 42 %; about 7 of 44 Scat 3rd-order Real, up to 48 %.
- **Third-order Imag has 6 raw misses per seed but they are not real:** their targets are ≈ 1e-10 of the family scale (effectively zero for a nearly symmetric signal), and the absolute error is ≈ 1e-11 of the scale.
- **Same indices fail in every seed:** 22 statistics have rel > 1 % in at least 4 of 5 seeds; Mod2-Imag `[5, 11, 19, 21, 30, 34]` fail in all 5 (mean raw error 24 to 73 %). But **the sign of the error is not consistent across seeds**
  (e.g. Mod2-Imag `[34]`: relative error +0.67, −1.33, −1.02, −0.34, +0.32 in seeds 500 to 504, with targets −1.3e-10, −6.8e-11, +8.8e-11, −7.9e-11, −2.3e-10, so the targets differ by seed too).
  sign(p − e) = sign(e) in only 41 to 62 % of the Mod2-Imag statistics (chance level). This is unlike turbulence, where every large error has the same sign.
- **There is an absolute error floor:** within the Mod2-Imag family, the median absolute error is 1.6 to 3.1e-11 per seed, against a median target of 2.7e-9 (1 %), and the smallest targets are ~1e-10 (error of the same order as the target). The Spearman correlation of |e| with the
  absolute error is +0.5 to +0.7, and with the relative error −0.6 to −0.7.
- **Contrast on the same statistic family:** Scat4 Mod2 Imag Q1 (identical definition, 56 statistics) has a median |e| of 2.0e-8 in turbulence and 2.7e-9 in the jets (7× smaller), but its median relative error is **4.9e-7 in turbulence and 8.1e-3 in the jets** (4 orders of magnitude);
  the absolute error floor is about 1e-14 in turbulence against 3e-11 in the jets. So the same SDE and the same potential can be matched to ~1e-6 on one data set and not on the other.
- **The scalar potentials help a little:** `jets_noscalar` has 35 to 50 statistics above 1 % per seed against 28 to 35 for `jets_full`.
- **Along the run** the peaks are huge (relative errors of 1e2 to 1e4 for statistics whose target crosses zero), then they come down. The final error is what is left.

### 3c. A related failure in the turbulence σ × nt sweep  [measured]
A check of the entropy bound runs the turbulence model at sigma 1, 2, 3.5, 6 and nt 55000 and 27500 (windows frozen). Job 54939 (**sigma 6, nt 27500**) crashed at step 15694 of 27500 with `FloatingPointError: non-finite Gram/rhs` (NaN in walkers or potentials),
after 4 to 6 scalar potentials turned "dead" (empty region) in the preceding steps. By my estimate the step is at 1−t ≈ 2e-3. sigma²/nt is 1.3e-3 for this run; the sigma 6, nt 55000 job (6.5e-4) is still running and may do the same.
Other runs of the sweep are still running (results due on 2026-11-11 evening and 2026-11-12); the prediction to test is that the turbulence error scales as `h sigma^2` (nt 27500 at sigma 3.5: about 8 statistics above 1 %).

## 4. What is known about causes

Turbulence (scalar windows):
- **[measured]** a bias (flat in n1), not Gram conditioning, not the ridge (1e-4 vs 1e-2 identical), not the number of walkers.
- **[derived]** the predictor has no Itô term; the one-step Gauss-Newton corrector must absorb `h sigma^2 <Laplacian phi>`, which is large for sigmoid edges narrower than the per-step noise; the iterated corrector (`_extra_corrector`) cut the misses from 15 to 4 on an older fit at +69 % cost (dropped).
- **[inferred]** the constrained-BIC fit bounds the window width by the one-step noise but not the sigmoid edge, which is 0.1× the window.

Jets (scattering statistics) — all untested:
- **[derived]** the scalar-window mechanism does not apply: the scalar potentials are matched. The Mod2/Third-order statistics are smooth polynomial-type functions of x (quartic in the filtered signal), so their Laplacian is smooth and the Itô term is not obviously large;
  but it was never computed (`turbulence/moment_error_variance.scalar_laplacian` handles only the scalar potentials).
- **[derived]** `_live_potentials` marks a statistic dead only when diag(G) ≤ `_live_floor`; the floor is 0 for the scattering potentials (`near_empty_tol` exists only for Scalar_GGD_KRegion), so "dead statistics" does not explain the jets misses.
- **[inferred]** candidate 1, *conditioning and the ridge*: the 56 Mod2-Imag statistics (products of filtered quantities) are probably strongly collinear. The corrector solves `(G_normalised + 1e-4 I) theta = rhs`; a component of the residual along a direction with eigenvalue ≲ 1e-4 is only corrected by the fraction `mu/(mu + 1e-4)`.
  The runs saved the per-step conditioning log (`saved_results/aux_moments/*_cond.pt`: min/max eigenvalues, spectra). Not read for the jets. In turbulence the ridge was ruled out for the scalar statistics, not for the scattering ones.
- **[inferred]** candidate 2, *noise floor of an odd statistic*: the targets are ~1e-10 to 1e-9 and the absolute error is a few 1e-11; the error has random sign across seeds, which looks like a stochastic residual (step noise, finite n1) rather than a deterministic bias.
  A z-score would tell whether the target itself is above its sampling noise.
- **[inferred]** candidate 3, *nonlinearity of the one-step corrector* on the quartic statistics, plus the step noise: the corrector displacement is of the order of the step noise; for a quartic function the linearisation error is second order in the displacement.
- **[inferred]** candidate 4, *the grid ends at 1−t = 7.5e-4 in the jets*, vs 5e-5 in turbulence: the last states may be less corrected. (The tail step density is the same in both.)

Possible common mechanism: an O(h) bias of the one-step predictor-corrector scheme with the Euler-Maruyama noise (Itô term plus the second-order remainder of the corrector), which is large where the statistic has large curvature relative to the step noise, and which shows up as a bias in turbulence and as a residual of random sign in the jets.
This is a hypothesis; the experiments in section 6 are meant to discriminate it.

## 5. Candidate strategies to evaluate (rank them by expected gain, code cost and risk)

1. **Scalar potentials** (turbulence only): a per-channel edge width tied to the one-step noise, or edges smooth in log|z|. Does nothing for the jets.
2. **Itô term (second-order drift correction) in the predictor right-hand side, for all potentials**: needs `<Laplacian phi_n>` on the walkers; for the scattering statistics an exact or Hutchinson-estimated Laplacian. This is the one lever that would apply to both experiments.
3. **Corrector**: iterated or damped Newton restricted to the statistics still off (`_extra_corrector` exists), or a second-order (Heun/midpoint) predictor-corrector. Cost per step vs gain in steps.
4. **Solver**: smaller or per-block ridge, or a ridge relative to the smallest meaningful eigenvalue; a truncated-SVD solve; check whether the Jacobi normalisation hides near-copies. Tests candidate 1 for the jets.
5. **Time grid**: spend steps where the bias accrues (the turbulence error peaks near 1−t of 1e-3 to 5e-4 and recovers); jets: extend `gap_end` below 7.5e-4.
6. **Statistic set / normalisation**: for the odd (time-reversal) statistics, whether the targets can be matched at all when they are within sampling noise of zero; use a z-based acceptance instead of a relative error.
7. Something not listed.

Constraints: the other statistics are matched to ~1e-7 (turbulence) and ~1e-5 (jets) with the current setup, so a change must not degrade them; the cost per step is ~1.2 s at r ≈ 300 (so a 1.7 h dev job gives nt ≈ 5000); all runs happen on one GPU, at most 2 h per test job, 20 h for a production job.

## 6. Experiments that would discriminate (cheap first)

- **Jets conditioning log** (`*_cond.pt`, already saved): number of Jacobi-scaled eigenvalues below 1e-4 near the end, and the weight of the residual on those directions. Tests candidate 1. CPU, minutes.
- **Jets z-scores** (error over the sampling noise of the target, as in `turbulence/moment_error_variance.py`). Tests candidate 2: are the Mod2-Imag and Third-order targets above their sampling noise? CPU, minutes.
- **Generic Itô term for all potentials** (Laplacian by autograd or Hutchinson) on the jets, then Spearman(|I|, error) as in turbulence. Tests the common mechanism.
- **Jets nt dependence**: one or two jets runs at nt 20000 (6.5 h) for comparison with nt 40000: does the Mod2-Imag error scale with h? No jets nt sweep exists yet.
- **Turbulence σ × nt sweep** (running): the O(h) scaling.
- **Short test runs (nt ≈ 5000, 1.7 h, dev QoS)** for any proposed fix, comparing the two worst families (turbulence psi ch 0 and 1, jets Mod2 Imag).

## 7. Open questions
- Are the failing jets statistics statistically distinguishable from zero in the data? (z-scores)
- Why is the same Scat4 Mod2 Imag family matched to 1e-6 in turbulence and 1e-2 in the jets, with the same SDE? Data magnitude (7× smaller targets), the different potential set (Scat 3 added in the jets), conditioning, or the grid end?
- Is the Itô term the common cause, or two different problems (scalar edges in turbulence, conditioning/noise floor in the jets)?
- How many steps do the matched statistics really need? No completed low-nt run of either model exists.

## 8. Files to attach

Core (both experiments):
- `codes/sde_routines.py` — `iteration_step_projection` (l. 1332: predictor, corrector, normalisation of theta), `compute_eta` (1435), `compute_theta` (1536: Jacobi scaling and ridge, l. 1573 to 1581), `_extra_corrector` (1589), `_live_potentials` (1789), `compute_rhs_dt_phi_I_t` (1830), `compute_G`, `forward_regularised` (618)
- `codes/time_schedules.py` — `two_phase_schedule` (l. 57)
- `codes/potentials/potentials_1d.py` — scalar: `Scalar_GGD_KRegion` (l. 1129 to the end); scattering: `Scattering_Third_Order_Real_1d` (l. 247), `..._Imag_1d` (306), `Scattering_Fourth_Order_Mod2_Real_1d` (638), `..._Imag_1d` (745) with their `grad`
- `codes/potentials_builder.py` — which filters and in which order the statistics are concatenated
- `turbulence/launch/turbulence_nested_M3_bicw.sh` and `jets/launch_jet_SDE.sh` — exact configurations
- `turbulence/moment_error_variance.py` — error definitions, scalar Itô-term computation

Data of the analysis:
- `turbulence/moment_errors/*.csv` + `report_*.txt`; `jets/moment_errors/*.csv` + `report.txt` (one row per statistic: target, generated, signed/relative error, peak along the run, decoded channel and window for the scalar ones)
- `notes/fable_brief_moment_mismatch.md` (the turbulence brief)

Optional: `jets/moment_errors_jets.py`, `jets/run_jet_SDE.py` (flags), `codes/utils_experiment.py` (how the potentials are built and fitted), memory note `moment_misfit_bias_test_1023.md` (history of the turbulence investigation), the conditioning logs of the jets (`jets/saved_results/aux_moments/*_cond.pt`, on JZ).
