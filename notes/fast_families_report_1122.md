# Faster L_6, GGD (Scalar_psi / Scalar_morlet), Scat3 Real/Imag: candidates, validation, timing (2026-11-22/23)

Tags: **[measured]** = jobs 575821 (validation, M = 256, full model r = 457, H100, batch 5000, 200-step loop at n1 = 5000) and 575827 (per-operation tables of the ORIGINAL code at M = 1024, batch 5000, GGD fitted states of the reference run loaded at M = 1024); **[derived]** = from the code or exact algebra; **[estimate]** = extrapolation.

Code: `grad_speed/fast_families.py` (new versions, not used by production; `codes/` unchanged), `validate_families_fast.py`, `bench_family_ops.py`. Nothing committed, nothing is the default. Same protocol as `notes/scat4_fast_report_1122.md` §2. Raw numbers: `grad_speed/results_families_fast/validate_families_M256_families_validate_M256.json`, `grad_speed/results_bench_family_ops/family_ops_M1024_families_bench_M1024.json`.

## 1. Result

- Per call at M = 256 [measured]: Scat3 forward 3.10 → 0.89 ms, GGD (psi) Jacobian 14.98 → 11.05 ms (eager) or 3.78 ms (compiled), GGD (morlet) Jacobian 6.58 → 4.79 / 1.75 ms. L_6 gets no gain.
- Recommended eager set: Scat3 J/A/V, GGD R. Not recommended: L_6 R (slower).
- Per step (one batch of 5000 walkers = 3 forward + 3 Jacobian + 2 grad(x, v) calls) the eager saving of the families is **≈ 38 ms** (GGD psi 13.8, GGD morlet 6.3, Scat3 Real 9.5, Scat3 Imag 8.1, L_6 −0.5) [derived from measured per-call times]. With the Scat4 eager set (≈ 80 ms, same formula) the sum is ≈ 118 ms, and the measured loop is **0.393 → 0.268 s/step (−32 %, −124 ms)**: the per-call savings carry over into the loop.
- Compiled GGD (R+E) would add ≈ 45 + 19 ms per step on paper, but the compiled loop is **0.282 s/step, slower than eager (0.268)** [measured]. The per-call compiled savings do not appear in the loop; cause not identified (not measured: compile time, recompiles, launch overheads inside the loop). **Do not adopt E for GGD on this evidence.**
- Accuracy: the new versions are within the float64-baseline spread in the downstream quantities and the loop (§4). GGD is the least comfortable class: error against float64 up to 2.8 times the original (eager), 4.4 times (compiled).

## 2. Per-operation cost of the originals at M = 1024 [measured, ms per call, batch 5000]

| Class | forward | Jacobian | grad(x, v) | per step (3F + 3J + 2V) | peak GiB (Jacobian) |
|---|---|---|---|---|---|
| L_6 (K = 9) | 1.43 | 3.24 | 2.91 | 19.8 | 1.2 |
| Scalar_psi GGD (K = 132, 25 channels) | 14.27 | 58.97 | 39.46 | 298.6 | 12.6 |
| Scalar_morlet GGD (K = 60) | 5.99 | 25.44 | 16.58 | 127.5 | 5.7 |
| Scat3 Real (K = 44) | 12.07 | 19.19 | 6.39 | 106.6 | 5.9 |
| Scat3 Imag, no low-pass (K = 36) | 12.07 | 15.95 | 6.52 | 97.1 | 4.1 |

Sum 650 ms per batch of 5000, **1.30 s per step of 10000 walkers** for these five classes alone (the Scat4 classes are in `notes/scat4_fast_report_1122.md`). Scalar_psi is 46 % of it.

Where the time goes (share of the call):
- GGD: pointwise operations. The derivative array D_k (all slots, (B, 154, M)) is 28 % of the Jacobian and 42 % of grad(x, v); the sigmoid derivative `dg` 10–14 %; the sigmoids and windows 10–28 % of the forward. FFTs are the minority (ifft 17 % + fft 12 % of the Jacobian). So the gain has to come from fusing the pointwise part (E), not from the FFTs.
- Scat3 forward: the (B, 9, 9, M) tensors, `conv` (ifft) 37 %, `cprod` 22 %, `prod` 15 %, `mean` 9 % of 12.07 ms; this is what the Parseval form J removes.
- Scat3 Jacobian: the two inverse FFTs (phase and modulus term) 25 % plus the output sum 14 %.
- Scat3 grad(x, v): already small (6.4 ms), spread over FFTs.
- L_6: `D = 2p Wx |Wx|^(2p−2)` 35–39 %, rest FFTs. Tiny in absolute terms (≤ 3.3 ms).

## 3. Candidates

Per-call times in ms, batch 5000, M = 256 [measured]; error ratio = max over states of e_k(new)/e_k(orig) against a float64 reference, max over statistics [measured].

### Scat3 J: forward by Parseval

- Math: mean_t Re/Im[conj(W_l x) (ψ_l * |W_i x|)] = (1/M²) Σ_k |ψ_l(k)|² conj X(k) Â_i(k), with X = fft(x), A_i = |W_i x|; one batched GEMM instead of the (B, 9, 9, M) ifft. Exact.
- Error ratio: Real 1.31, Imag 1.97 (median 1.00 / 0.99).
- Time: 3.10 → 0.89 (both). Per step 3 × 2.2 = 6.6 ms for each of Real and Imag.
- Verdict: **adopt**.

### Scat3 A: Jacobian with phase and modulus terms added in Fourier space

- Math: out = Re ifft(Z)/M with Z = conj(ψ_j) fft(ph_j W_k) + |ψ_k|² Â_j (Real); Imag: Re ifft(Zp + i Zm), using Re W* and −Im of the modulus term as one real part. The weight Re x2 (Real) or −Im x2 (Imag) from irfft(rfft(x) · Hermitian weight): exact for real x.
- Error ratio: Real 1.17, Imag 1.09.
- Time: Real 4.87 → 4.15, Imag 4.09 → 3.84. Per step 2.2 / 0.8 ms. Peak memory goes UP (1.78 → 2.25, 1.34 → 1.90 GiB).
- Verdict: **marginal gain, adopt only together with J/V** (negligible; reject if the memory increase matters at M = 1024).

### Scat3 V: grad(x, v) with channel sums in Fourier space, one inverse FFT

- Math: Σ over channels of the two terms taken in Fourier space, one ifft of length M. Exact.
- Error: no separate ratio recorded for grad(x, v) in the summary (the errors stored are the original's only, ~1e-5 for the GGD case; see §6).
- Time: Real 1.79 → 1.43, Imag 1.82 → 1.46. Per step 2 × 0.36 = 0.7 ms.
- Verdict: **negligible gain; adopt only as part of the set**.

### GGD R: rfft for the coefficients and the adjoint

- Math: z = Re ifft(ψ X) = irfft(rfft(x) ψ_s), ψ_s = (ψ(k) + conj ψ(−k))/2; adjoint Re ifft(fft(D) conj ψ) = irfft(rfft(D) conj ψ_s) with 1/(M · stat_scale) folded into the filter. Exact for real x, D. Undecimated bank only (asserted).
- Error ratio (R): Scalar_psi forward 1.87, Jacobian 2.82; Scalar_morlet forward 2.35, Jacobian 1.74. grad(x, v) is the same as the original within a few percent (psi, t = 0.5: 1.8e-5 / 4.0e-5 / 2.2e-5 original vs 2.1e-5 / 3.4e-5 / 2.1e-5 new, three probe vectors). The medians are 0.98–1.03: the ratio is driven by a few statistics.
- Time: psi forward 3.72 → 3.53, Jacobian 14.98 → 11.05, grad(x, v) 10.10 → 9.38; morlet 1.63 → 1.55, 6.58 → 4.79, 4.39 → 4.07. Per step 13.8 (psi) + 6.3 (morlet) ms. Memory unchanged (3.15 → 3.27 GiB Jacobian psi).
- Verdict: **adopt (small gain)**; the Jacobian is where it helps.

### GGD E: torch.compile of the pointwise part

- Math: none changed; the sigmoids, windows, powers, means and derivatives are fused.
- Error ratio: forward up to 4.25 (psi) / 4.43 (morlet), Jacobian up to 3.31 / 3.11; 0.8–5 % of the statistics have a ratio > 3.
- Time per call: psi forward 3.72 → 0.71, Jacobian 14.98 → 3.78, grad(x, v) 10.10 → 2.11; morlet 1.63 → 0.32, 6.58 → 1.75, 4.39 → 1.03; Jacobian peak memory unchanged (3.3 GiB), forward and grad(x, v) lower.
- In the loop: 0.282 vs 0.268 s/step eager, so no gain there (and the error is larger).
- Verdict: **unsure, not adopted**. The per-call gain is large and GGD is the biggest target at M = 1024 (46 % of the family time for Scalar_psi, §2), so it is worth understanding why the loop does not benefit before dropping it (compile time and recompiles inside the loop, or batch shapes differing from the benchmark).

### L_6 R: adjoint by irfft

- Math: D = 2p Wx |Wx|^(2p−2) is complex, so only the Hermitian part of Y = fft(D) conj ψ is kept before irfft (exact).
- Error ratio ≤ 1.09. Time: Jacobian 0.97 → 1.13 ms (slower); per step −0.5 ms.
- Verdict: **reject** (same reason as A2 for Scat4: the gather and conjugation cost more than the half-length FFT saves). L_6 is 6 ms of the 393 ms per step; nothing to gain.

## 4. Downstream and loop [measured, M = 256, fitted states of the reference run seed 900]

"new" = all new versions together (families eager set + Scat4 eager set); "compiled" = with GGD E and Scat4 E; "f64" = the original code with the Scat4 block in float64.

| Quantity | new vs orig | compiled vs orig | f64 vs orig |
|---|---|---|---|
| G, relative (t = 0.5 / 0.99) | 2.2e-7 / 3.1e-7 | 2.2e-7 / 3.4e-7 | 1.9e-7 / 3.7e-7 |
| eta, relative (t = 0.5 / 0.99) | 3.5e-5 / 5.5e-5 | 4.3e-5 / 4.9e-5 | 7.2e-5 / 4.5e-5 |
| theta difference / sampling noise of theta (t = 0.5 / 0.99) | 4.6e-5 / 2.8e-5 | 3.9e-5 / 3.8e-5 | 2.8e-5 / 5.9e-5 |
| statistics with difference > 10 % of their noise | 0 / 0 | 0 / 0 | 0 / 0 |
| loop: walkers after 200 steps, relative | 3.2e-3 | 3.2e-3 | 3.2e-3 |
| loop: max moment difference over all steps (standard errors) | 2.76 | 2.56 | 2.74 |
| loop: max moment difference at the last step (standard errors) | 1.55 | 0.87 | 2.06 |
| loop: seconds per step (n1 = 5000) | 0.268 | 0.282 | 0.801 |

The original is 0.393 s/step. Sampling noise of theta (difference of two half-batch solutions, relative): 1.0 at t = 0.5, 1.8 at t = 0.99; the new-vs-orig theta differences are 4 orders of magnitude below it.

Reading: the new code is as close to the original as the original is to itself with the Scat4 block in float64. After ~100 steps all runs differ by O(1) standard error of the moments from the dynamics amplifying rounding (same as for Scat4). The compiled run is not worse than eager on these checks.

## 5. What is not done / open

- **M = 1024: only the original code was benchmarked.** The fast versions have not been run at M = 1024 (no timing, no validation, no memory). Estimate, applying the M = 256 ratios (new/orig per step: psi 0.82, morlet 0.81, Scat3 0.65 / 0.68; each class from §1) to the §2 table: eager saving ≈ 146 ms per batch (psi 54, morlet 24, Scat3 37 + 31), ≈ 0.29 s per step of 10000 walkers [estimate]. Whether FFT/pointwise proportions are the same at M = 1024 is unchecked.
- Compiled GGD in the loop: no gain, cause unknown (§3). Not measured: start-up and recompile times.
- Scat3 V and A: error ratio of grad(x, v) not tabulated in the summary above; the Jacobian ratios are (≤ 1.17).
- The "other uses" check (is anything of the changed objects used outside the call?) was done in the code design (all new functions are local to the call; p.`_fast_cache` keyed on the identity of the filters and GGD parameters), but it has not been re-verified in this report against `codes/` callers for the families; do that before adoption (as was done for Scat4: `compute_grad_potentials`, `gram_preflight.gradient_gram`, `compute_G`, `compute_rhs_dt_phi_I_t`).
- GGD limits: undecimated bank only (asserted in the code).
- n1 = 200 divergence (also in float64) unexplained; smoke 6 hang unexplained. Not tested: other J, F + A1 for Scat4 together.
- Scat4 candidates E, F, A1 applied to GGD/Scat3: Scat3 J is the same Parseval idea as Scat4 J; GGD R is the same idea as Scat4 F (irfft of an exactly real quantity). For Scat4 E the per-call gain was also the largest (`notes/scat4_fast_report_1122.md`), and the loop with both E versions is slower than eager (0.282 vs 0.268 s/step): the E results of both reports need a loop check before use.

## 6. Verdict summary

| Candidate | Per step at M = 256 | Error vs orig | Verdict |
|---|---|---|---|
| Scat3 J (forward Parseval) | 6.6 ms | ≤ 2.0 | adopt |
| Scat3 A (Jacobian) | 3.0 ms, + memory | ≤ 1.2 | marginal |
| Scat3 V (grad(x, v)) | 0.7 ms | not tabulated | marginal |
| GGD R (rfft/irfft) | 20 ms | ≤ 2.8 | adopt |
| GGD E (compile) | 44 ms on paper, none in loop | ≤ 4.4 | unsure, do not adopt |
| L_6 R | −0.5 ms | ≤ 1.1 | reject |

Nothing becomes the default without your approval.
