# Part 1 report: does the last step explain the final moment error? (2026-11-12)

Job 65029 (H100 dev, COMPLETED), harness `codes/one_step_test.py`, logs and data in `terminal_phase_test/part1/`.
Runs analysed (seed 900 / 500, no refit, saved final walkers and saved last target): turbulence B+Scat4 (r = 341), turbulence B without Scat4 (r = 201), `jets_full` (r = 299).
20 independent noise draws per point. **Measured** unless marked **[inferred]**.

## What was done
Starting from the saved final walkers, one step = noise `sqrt(2h) sigma` + one corrector pass of the real `SDE.compute_theta` (Gram at the noised walkers, Jacobi scaling, ridge 1e-4, float64, live mask), target = the run's last saved target.
The predictor drift over one step is neglected. My re-implemented solve equals `compute_theta` exactly (difference 0.0) on every draw. The saved samples reproduce `aux_moments` (difference 0.0).
Last step of the runs: turbulence h = 7.6e-9 (noise 4.3e-4 per sample), jets h = 1.05e-7 (noise 1.6e-3 per sample, 14x larger, because the jets grid stops at 1-t = 7.5e-4).

## Verdict on the gate
| | reproduction at the run's own h | does the error depend on h as the hypothesis says? | verdict |
|---|---|---|---|
| turbulence B+Scat4 | ratio one-step / observed = 1.00 (n = 77), Spearman +1.00, 13 of 13 signs | settled level ~ h^1.7 to 2.1 (median 1.73) | **passes** |
| turbulence B (no Scat4) | ratio 1.00 (n = 45), Spearman +1.00, 12 of 12 signs | median exponent 1.79 | **passes** |
| jets_full | ranking reproduced (Spearman +0.94, n = 251); magnitude: median ratio 0.52 of the mean residual, 2.28 for the rms over draws; sign agreement 70 % (n = 73) | settled level does NOT depend on h (median exponent -0.01) | **reproduces size, fails the mechanism test** |

## Experiment 0 (reproduction), turbulence B+Scat4, relative to |e|
| statistic | where | observed | one-step mean | std | ratio |
|---|---|---|---|---|---|
| psi[34] | ch0 body moment | 3.70e-2 | 3.71e-2 | 2.3e-4 | 1.00 |
| psi[78] | ch0 tail mass | 3.50e-2 | 3.50e-2 | 4.5e-4 | 1.00 |
| psi[56] | ch0 tail moment | 3.16e-2 | 3.16e-2 | 4.2e-4 | 1.00 |
| psi[35] | ch1 body moment | 1.72e-2 | 1.73e-2 | 1.1e-4 | 1.00 |
| morlet[8] | ch0 win 1 moment | 1.63e-2 | 1.62e-2 | 8.8e-5 | 1.00 |
| psi[79], psi[57], psi[36] | ch1 tail, ch2 body | 6.7e-3, 6.0e-3, 5.1e-3 | same to 1 to 2 % | | 0.99 to 1.01 |
| Mod2_Real[27], [26], [24] | | 1.0e-3, 8.5e-4, 6.9e-4 | same to 1 % | | 1.00 |
Without Scat4 the same five give 3.69, 3.53, 3.19, 1.72, 1.62 e-2: the Scat4 potentials play no role.
Caution: if the final error is a stationary level of the last step, one more identical step reproduces it, so this agreement alone is not a proof of the mechanism; the h-dependence below is.

## Experiment d (repeated steps, target fixed): settled level vs h, turbulence B+Scat4
Rms over steps 100, 200, 300, relative to |e|; chains of 300 steps at h x {1, 1/4, 1/16, 1/64} (h = 7.6e-9 x mult).
| statistic | h x 1 | h x 1/4 | h x 1/16 | h x 1/64 | exponent |
|---|---|---|---|---|---|
| psi[34] | 3.71e-2 | 3.93e-3 | 3.05e-4 | 1.69e-5 | 1.85 |
| psi[78] | 3.49e-2 | 1.61e-3 | 9.2e-5 | 7.1e-6 | 2.05 |
| psi[56] | 3.15e-2 | 1.43e-3 | 8.4e-5 | 6.5e-6 | 2.04 |
| psi[35] | 1.71e-2 | 1.93e-3 | 1.40e-4 | 6.9e-6 | 1.88 |
| morlet[8] | 1.62e-2 | 3.03e-3 | 4.14e-4 | 4.0e-5 | 1.44 |
| Mod2_Real[27] | 1.00e-3 | 8.2e-5 | 5.8e-6 | 4.4e-7 | 1.87 |
Median exponent over the 76 statistics with a settled level > 1e-5: 1.73 (quartiles 1.36 to 1.94); without Scat4: 1.79 (1.48 to 1.95). The tail-window statistics give 2.0 to 2.1.
Settling is fast: at h/4, psi[34] is 7.3e-3 after 1 step, 4.3e-3 after 10, 3.9e-3 after 300. About 10 steps are enough.
Consequence **[inferred from these levels]**: all 13 statistics above 1e-3 fall below ~4e-4 at h/16 of the last planned step (the worst, morlet[8], 4.1e-4); h/4 is not enough (psi[34] 3.9e-3). A terminal stretch of a few hundred steps at ~h/16 costs < 1 % of nt.
Prediction for the finished nt 27500 sweep runs (not yet analysed): their last step is 2x larger, so psi[34] should be ~3.7e-2 x 2^1.85 = 13 % and psi[78] ~14 % (exponent law, extrapolated by 1 factor of 2).

## Experiments b, c, a, e, f, turbulence B+Scat4
- **b, chord passes at the run's h** (psi[34]): 3.71e-2, 1.36e-2, 4.55e-3 for 1, 2, 3 passes (about x0.37 per pass; the worst rms over all statistics 3.7e-2, 1.5e-2, 6.8e-3). Reaching 1e-3 would take about 5 to 6 passes per statistic; shrinking h is cheaper.
- **c, ridge 1e-2 / 1e-4 / 1e-6**: psi[34] 3.88e-2 / 3.71e-2 / 3.71e-2. The ridge is not the cause.
- **a, one step from the final walkers at h x mult**: exponent median 0.86 (86 statistics). Confounded: the start still carries the old residual (see the caveat in the log); use experiment d.
- **e, scalar edge width sw x {1, 2, 4}** (targets and stat scale recomputed from the data, protocol differs from the saved run, so only ratios matter): psi[78] 3.66e-2, 6.07e-2, 2.44e-2; psi[56] 3.3e-2, 6.6e-2, 3.2e-2; psi[35] 1.9e-2, 2.3e-2, 3.4e-2; psi[36] 5.6e-3, 1.3e-2, 2.7e-2; morlet[51] 1.8e-3, 5.8e-3, 1.1e-2; only morlet[8] improves (1.7e-2, 1.6e-2, 9.1e-3).
  The Itô term `h sigma^2 <Laplacian phi>/|e|` does not shrink with a wider edge: body statistics unchanged (psi[34] 0.68 at every sw), tail windows grow (psi[78]: 0.044, 0.089, 0.27, 0.18 for sw x 0.5, 1, 2, 4). **Widening the edges does not help and mostly hurts.**
  **[inferred]**: for a smooth sigmoid, `<Laplacian phi> = int p'' phi` depends on the density slope at the cut, not on the edge width, while a wider edge reaches into higher density.
- **f, float32 evaluation floor**: f32 vs f64 on the saved samples ~1e-7 of |e|, against observed errors 1e-3 to 4e-2. Not roundoff.

## Jets (jets_full seed 500)
- **Experiment d**: the settled level does not fall with h: median exponent -0.01 (quartiles -0.11 to 0.07, n = 283). Median |res|/|e| is 1e-4 to 3e-4 at every h and checkpoint, with maxima 0.2 to 1.35 and random sign (Mod2_Imag[34]: 1.37, -1.15, 0.91, -0.24, -0.36 at steps 1, 10, 100, 200, 300 for h x 1).
- **b**: more passes do not help: median rms 2.7e-4, 4.8e-4, 4.9e-4 for 1, 2, 3 passes.
- **c**: ridge changes nothing in the median (rms 2.7e-4, 3.0e-4, 2.9e-4 for ridge 1e-4, 1e-2, 1e-6).
- **f**: float32 evaluation floor 4e-7 median (3 % of statistics have an observed error below 3x the floor); exception: Scat3 Imag, where float32 and float64 results differ by 95 % (unexplained: dtype handling of that potential in float64, or real cancellation) and a roundoff-level perturbation changes `Third_Order_Imag[16]` by 1.9e-2.
- **Conclusion**: the jets error is not set by the last step size, not by the ridge, not by float32 evaluation (except possibly Scat3 Imag). It behaves like a random-sign residual of the corrector that does not shrink with the noise.
  **[inferred, untested]**: the Gram is accumulated in float32 and an ill-conditioned direction turns a ~1e-7 error into an O(1) error of theta; or the corrector is hypersensitive along near-null directions. Next tests: read the saved conditioning logs (`jets/saved_results/aux_moments/*_cond.pt`), and a Gram-precision test (solve with G(y) and with G(y) perturbed at 1e-7).

## What this means for Part 2
- For **turbulence-type** errors (scalar windows narrower than the step noise) the hypothesis is confirmed and the fix is predicted to be cheap: end the run with ~100 to 300 steps at ~h/16 of the last planned step.
- A final iterated corrector (2c) is less efficient than shrinking h (about 5 to 6 passes for 1e-3), and it does nothing for the jets.
- Widening the scalar edges (2d) is not supported by these data.
- A terminal phase will **not** fix the jets as the experiment stands; the jets need the conditioning / Gram-precision diagnosis first.

## Not checked
One seed per case; the predictor drift over the last step is neglected; the chains hold the target fixed (the real target moves, by a negligible amount at these h); experiment e uses a different protocol; the nt 27500 runs are not analysed yet; no S&P, Gaussian-spectrum or other examples were run in Part 1.
