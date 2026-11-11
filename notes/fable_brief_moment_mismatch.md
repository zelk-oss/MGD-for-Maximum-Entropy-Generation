# Brief: reducing the moment-matching error of 5 statistics in the turbulence MGD runs

Written 2026-11-11. Facts are marked **[measured]**, **[derived]** (from the code, not measured) or **[inferred]** (my reading, untested).
You know how the moment-guided diffusion (MGD) SDE works; the code pointers are at the end.

## 1. Goal and constraint

We sample 1-D turbulence velocity signals (Modane data, M = 256 samples) with a maximum-entropy model defined by moment constraints
`E[phi_n(x)] = phi_bar_n(data)`. The SDE transports walkers from noise (t = 0) to the model (t = 1) and enforces the moments along the way.

The run uses 55000 time steps (about 18.7 h on one H100). We want to run with far fewer steps. Almost all of the 341 moments are matched to
1e-7 relative (median), but a few statistics in the finest channels are not, and they alone would force a much larger number of steps.
**The aim is to remove the mismatch on those statistics, so that nt can be reduced.** The moments must stay matched (we use them to attribute the
model's limits), so dropping the statistics is not the answer. I bet on a small change to the scalar potentials, but it may be something else
(predictor, corrector, time grid).

## 2. The experiment

- **Data / signals:** subseries 512, `target_len` M = 256, J = 8 octaves, Q = 3, n1 = 10000 walkers/samples, sigma = 3.5 (noise `sqrt(2h) * sigma`).
- **Model ("B+Scat4"):** potentials `L_6` (9 stats), `Scalar_psi_gaussianK` (132 stats, 22 channels after deduplication of the Q = 3 bank),
  `Scalar_morlet_gaussianK` (60 stats, 9 channels; I read this as the Q = 1 bank plus low-pass, check `potentials_builder.py`),
  `Scattering_Fourth_Order_Mod2_Real_Q1` (84), `..._Imag_Q1` (56). Total r = 341. Launcher arm: `bicw_allmass_scat4` of
  `turbulence/launch/turbulence_nested_M3_bicw.sh`.
- **Time grid:** `two_phase`: 5000 uniform steps on [0, 0.9], then 50000 steps with h proportional to (1-t) down to 1-t = 5e-5
  (`codes/time_schedules.py`). Interpolant `Cos`, schedule exponent 2. The step at the end is h = 7.6e-9.
- **Solver:** time-regularised system (regularisation 1e-4, Thomas, float64) for the Theta_reg part; the per-step corrector uses `regularization` 1e-4 ridge on the
  normalised Gram matrix.
- **Reference run:** seed 900, nt 55000, `experiments/.../seed_900_..._20261101_1714` (SDE time 67398 s).

## 3. The scalar potentials (where I suspect the fix is)

`Scalar_GGD_KRegion` (`codes/potentials/potentials_1d.py:1129`). For each wavelet channel j, with z_j the coefficient and `|z|_smooth = sqrt(z^2 + eps_j)`:
- The range of |z| is split into K = 4 windows by cuts `c_j[m]`. The window indicator is built from sigmoids
  `g_m = sigmoid(-(|z| - c_m) / sw_m)` and `w_0 = g_0`, `w_k = g_k - g_{k-1}`, `w_{K-1} = 1 - g_{K-2}` (`_windows_from_sigmoids`, l. 1757).
- Edge width `sw_m = trans_frac * max(window width, cut)`, `trans_frac = 0.1` (l. 1617). So the edge is 10 % of the cut.
- Statistics per channel and window k: the moment `<w_k |z|^alpha_k>` (slot k < K), and with `tail_mass` and `all_masses` the plain mass `<w_m>` (slots K...).
  `alpha_k` is the exponent of a generalised-Gaussian fit in the window.
- Cuts are fitted by a constrained BIC (`--ggd_bic_min_window C --ggd_noise_ref_gap G`; here C = 1, G = 1e-2): no window narrower than C times the noise one SDE
  step adds to the channel at 1-t = G. A forced tail cut at the 0.99 quantile. Windows that the BIC merges become placeholder cuts (1e-8, 1.5e-8), so the finest
  channels end with Keff = 2 or 3: one body window (about 99 % of the data) and one tail window (1 %).
- Fitted state of the reference run: `turbulence/experiments/.../seed_900_..._20261101_1714/fitted_potentials/Scalar_{psi,morlet}_gaussianK.pt`.
  Windows are frozen across the sweep runs (`--potentials_from`).

## 4. What is not matched [measured]

Relative error |p - e| / |e| at t = 1 over the 341 statistics (p = generated moment, e = target), seed 900, nt 55000, sigma 3.5.
All of the 5 below are in the finest channels, and in every one the generated moment is **above** the target.

| statistic | channel | part of the distribution | window in \|Wx\| | share of data | rel error | z |
|---|---|---|---|---|---|---|
| `psi[34]` | psi ch 0 | body, moment | [~0, 1.01e-3) | 99.0 % | 3.70 % | 9.7 |
| `psi[78]` | psi ch 0 | tail (top 1 %), mass | [1.01e-3, inf) | 1.0 % | 3.50 % | 1.2 |
| `psi[56]` | psi ch 0 | tail, moment (alpha = 0.17) | [1.01e-3, inf) | 1.0 % | 3.16 % | 1.0 |
| `psi[35]` | psi ch 1 | body, moment | [~0, 9.25e-4) | 99.0 % | 1.72 % | 4.7 |
| `morlet[8]` | morlet ch 0 | body, moment | [1e-8, 3.52e-3) | 96.0 % | 1.63 % | 4.4 |

(`z` = error over the sampling standard error of two independent samples; z of order 1 means the gap is within sampling noise, but the walkers are not independent of the data,
so this only rules out "too few samples".) `psi[78]`, `psi[56]` (and `psi[34]`) are the same cut of channel 0, so there are about 3 independent problems, one per channel's q.99 cut.

Next-worst matched statistics (the error falls with channel index; same kind of statistic in each row):

| | ch 0 | ch 1 | ch 2 | ch 3 |
|---|---|---|---|---|
| psi body moment | 3.70 % | 1.72 % | 0.51 % | 0.16 % |
| psi tail mass (top 1 %) | 3.50 % | 0.67 % | 0.064 % | |

Per family (median / max): L_6 8.5e-8 / 1.1e-5, psi 9.5e-8 / 3.7e-2, morlet 8.8e-8 / 1.6e-2, Scat4 Real 2.1e-7 / 1.0e-3, Scat4 Imag 4.0e-7 / 1.9e-4.
Of the 341, 13 have an error above 1e-3 and 5 above 1 %. All 20 worst have the generated moment above the target.
The files: `turbulence/moment_errors/*.csv` (one row per statistic) and `report_*.txt`.

Along the run, the 5 peak much higher before recovering: `psi[78]`, `psi[56]` reach 224 % around 1-t = 5e-4 and come back to 3.5 %.

## 5. What we know about the cause

**[measured]** It is a bias, not sampling variance: with n1 = 1000, 3000, 10000 (frozen fit) the number of misses and the mean error stay flat (15, 15, 16; mean 0.13, 0.14, 0.12 on an older fit).
**[measured]** Not Gram conditioning (kappa improved by 1000x, same 11 misses on an older fit), not the ridge (1e-4 vs 1e-2 identical misses), not the number of walkers.
**[measured]** Itô-term diagnostic (`turbulence/moment_error_variance.py`): with `I_n(h) = h sigma^2 <Laplacian_x phi_n> / |e_n|`, the Spearman correlation of |I_n| with the error across
the 192 scalar statistics is +0.74 (p = 7e-35); the median |I| of the missed ones is 18 against 1.3e-4 for the matched ones at 1-t = 1e-2; `sign(p - e) = sign(I)` for 5 of 5.
I(1e-2) is 15 to 130 times the moment itself. The Laplacian was checked against finite differences (max rel err 1e-4).
**[measured]** The window width is not the discriminator: the 20 narrowest windows (ch 3 to 9, e.g. `psi[4]` ch 8 [0, 6.5e-4)) are all matched; the 5 misses have width ranks 7, 11, 33, 133, 155 of 192.
**[derived]** Mechanism: the predictor right-hand side is the moment velocity `d/dt phi_bar(I_t)` with no Itô term. After the noisy step the walkers' moments are off by about `h sigma^2 <Laplacian phi>`,
and the single Gauss-Newton corrector must absorb it. For a sigmoid edge of width s, `Laplacian phi ~ 1/s^2`. In the finest channels the per-step noise in z is larger than the edge width
(sw = 1e-4 for psi ch 0, against the noise of a step which I estimate to be of order 1e-3 near 1-t = 1e-2; **[inferred]**, not yet measured for this fit).
**[inferred]** The window-width constraint of the BIC fit (C x noise at 1-t = 1e-2) is on the window, but the sigmoid *edge* is `0.1 x` the window, so the edge is up to 10x
narrower than the one-step noise. The Itô error scales as 1/sw^2, so this is where I would act first.
**[measured, older fit]** An iterated corrector (chord passes with G(y_k) until the residual is within one standard error; `SDE._extra_corrector`, flag `--extra_corrector_max`) cut the misses from 15 to 4,
at +69 % cost at M = 128; it was dropped for cost. So the one-step Gauss-Newton linearisation of the corrector is part of the story too: the corrector displacement is of the
order of the step noise, which is larger than the edge, where phi is strongly nonlinear. I cannot separate "missing Itô term in the predictor" from "linearisation error of the corrector"; they are
two sides of the same thing and an answer should say which one it removes.

Two clean falsifiable predictions, being tested by running sweeps (jobs on JZ, results due 2026-11-11 evening and 2026-11-12):
- The error should scale as `h sigma^2`: halving nt (55000 to 27500) doubles it, and sigma 3.5 to 6 multiplies it by 2.9. Prediction for nt 27500, sigma 3.5: 8 statistics above 1 %
  (the 5 plus `psi[79]`, `psi[57]`, `psi[36]`). At nt about 6000 about 11 would be above 1 %.
- Planned: measure the per-channel step noise against the cut and the edge width (`turbulence/window_vs_step_noise.py` does this for the width; it needs a version for the edge `sw`).

## 6. Candidate levers (what I would like you to evaluate)

1. **Scalar potentials only (my bet):** a per-channel `trans_frac`, or an edge width tied to the one-step noise, `sw >= c x noise(z_j) at the reference gap`, for the cuts in channels 0 to 2;
   edges that are smooth in log|z| rather than in |z|; a different position of the tail cut in the finest channels. This keeps the model and changes the statistic definition.
   What it costs: coarser resolution at that cut. Check whether the *definition* of the statistic can be changed without changing the maximum-entropy family in a way that hurts the other results.
2. **Itô term in the predictor right-hand side** for the scalar statistics (needs `<Laplacian phi>`, available in `moment_error_variance.scalar_laplacian`; noisy for windows narrower than the step).
3. **Corrector:** damped/iterated/line-searched Newton restricted to the statistics that still fail; or a different update that is exact for the sigmoid (the statistic is a smooth function of z).
4. **Time grid:** more steps only where the bias accrues (the error peaks near 1-t of 1e-3 to 5e-4 and recovers).
5. Something I have not thought of.

Please keep in mind: the other 330 statistics are matched to median 1e-7 with the current setup, so any change must not degrade them; the cost per step is about 1.2 s at M = 256 (341 potentials, n1 = 10000).

## 7. Open questions
- Is the per-channel step noise over edge width really the discriminator between ch 0/1 and ch 3 to 9? (Error vs channel index is monotone, which supports it, but noise per channel is not yet measured for this fit.)
- Is the generated moment always above the target because of a signed effect of the Itô term, or of the linearisation? (All 20 worst are positive; in older fits the sign agreement with I was only 56 to 73 %.)
- How many steps do the other statistics really need? No completed low-nt run of this model exists yet.

## 8. Files (to attach)
See the list given with this brief.
