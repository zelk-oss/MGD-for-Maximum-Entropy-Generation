# Project status — MGD for Maximum Entropy Generation

Last updated: 2026-11-23. Source of truth for Claude Desktop (see CLAUDE.md).
The information comes from the README, code, launchers and notes in the repo. Where it cannot be derived: "TO COMPLETE".

## 1. Goal and theoretical idea

- The repo is the implementation (and research extension) of the paper *MGD: Moment Guided Diffusion for Maximum
  Entropy Generation* (Lempereur, Cuvelle-Magar, Coeurdoux, Mallat, Vanden-Eijnden; arXiv 2602.17211, Feb. 2026).
- Idea: sample from the maximum-entropy distribution constrained by a set of moments `m` estimated on a dataset, in
  high dimension. Instead of MCMC at equilibrium (slow), MGD solves in finite time a McKean–Vlasov-type SDE (stochastic
  interpolant) that drives the moments of the "walkers" toward the target values. The Lagrange multipliers `Θ_t` depend
  on time and are obtained by solving a linear system in the moments at every step.
- Moments used: wavelet scattering statistics (orders 1–4, Real/Imag parts), per-channel marginal scalar potentials
  (GGD, "Scalar_psi", "Scalar_morlet"), the confinement potential `L_6`.
- The trajectory also gives a lower bound on the entropy (`entropy_bound`; convention: it is a lower bound) and, with
  `Θ_reg` (time-regularised multipliers, parameter λ), an estimate of log p.
- Datasets: finance (S&P 500, daily series), turbulence (1D Eulerian/Lagrangian, 2D Navier–Stokes), jets (Modane
  recording), cosmic web (Quijote 2D), synthetic fBm/MRW, 1D bimodal density (test with known truth).

## 2. Repo structure

- `codes/` — shared code: `utils.py` (SDE utilities, `save_results`/`load_results`), `utils_experiment.py`
  (experiment names/configurations, `run_experiment`), `potentials/` (1D, 2D, conditional, scalar),
  `sde_routines*.py`, `resolve_theta_reg.py` (solves Θ_reg offline on saved systems; options `--c_shrink`,
  `--pin_l6`, `--neg_cols`, `--drop_last`), `lam_selection.py`.
- Entry points (argparse CLI): `turbulence/run_SDE.py`, `jets/run_jet_SDE.py`, `gaussian_experiment/run_SDE.py`
  (synthetic fBm/MRW), `cosmic_web/run_SDE.py`, `turbulence_2d/run_SDE.py`, `snp_experiment/run_SDE.py`.
- One folder per experiment (each with `.py` scripts, notebooks and `launch/`): `turbulence/`, `turbulence_2d/`,
  `jets/`, `cosmic_web/`, `snp_experiment/`, `gaussian_experiment/`, `bimodal_experiment/`.
- Cross-cutting analysis folders: `lam_selection_criterion/` (choice of λ, cross-seed rule), `kl_groups/` (D_k per
  group of statistics), `theta_sign_constraint/` (sign of L_6), `terminal_phase_test/`, `grad_speed/` (fast versions
  of the potentials), `gauss_scat4_test/`, `gauss_spectrum/`, `gauss_dim_sweep/`, `deff_diagnostics/`,
  `moment_reproduction/`, `nested_models/`, `whitening_test/`, `krylov_stopping/`, `fixed_potentials/`,
  `lagr_vs_old_turbulence/` (comparison of the new vs old Lagrangian dataset).
- `data/` (data and loaders; `data/data_files/` protected), `saved_results/` (results, protected), `notebooks/`
  (analysis notebooks, not run on the cluster), `notes/` (technical notes, PDF/tex, reports).
- Results of each run: `*/experiments/<config_name>/` with `config.json`, `logs/run.log`, `figures/`; data in
  `saved_results/{samples,lagrange_multipliers,lagrange_multipliers_regularised,entropy_bounds,sampling_times}/`.
- Configuration name: `build_config_name()` in `codes/utils_experiment.py`
  (`<jetsynth|turbulencesynth|gaussiansynth>_..._M<M>_J<J>_Q<Q>_sigma<σ>_nt<nt>_n1_<n1>_lam<λ>_seed_<seed>_terms<hash>...`).
- Compute: everything on Jean Zay (SLURM); this machine has only a CPU. Smoke tests run on JZ, not locally.

## 3. Numerical experiments

General command: `bash <launcher>` from the repo root on Jean Zay, settings as environment variables
(`GPU=v100|h100`, `QOS=`, `TIME=`, `SEEDS=`, `LABEL=` ...). Read the launcher header before use.

### 3.1 1D turbulence (final model: `bicw_allmass_scat34d`)
- Configuration: M=256 (subseries 512, coarse-grained to 256), J=8, Q=3, σ=3.5, nt=27500 + terminal phase,
  n1=10000, seeds 900–904; 457 statistics: L_6, Scalar_psi, Scalar_morlet, Scat3 Real/Imag, Scat4 Mod2 Real (diag) and Imag.
- Launch: `ONLY=bicw_allmass_scat34d SEEDS="900 ..." bash turbulence/launch/turbulence_nested_M3_bicw.sh`
  (other arms: `bicw`, `bicw_allmass`, `bicw_allmass_scat4`, `bicw_rich`).
- Results: `turbulence/experiments/`, `turbulence/saved_results/`, comparisons in `turbulence/cross_scale_scat34.ipynb`
  and `turbulence/sample_categories/`.
- Outcome (2026-11-18, seed 900): the scat34d model was judged the best; fraction within 2σ close to B+Scat4;
  tail accelerations within ±2σ of the data down to ~1e-6. Oscillating events ("packets") under-produced (0.84 model/data).
  Imposed moments reproduced (max z 0.04 SE).

### 3.2 Maximum-entropy σ × nt scan (1D turbulence)
- Question: does the model reach the maximum-entropy distribution? Bound vs σ, extrapolated to h→0.
- Launch: same launcher, `ONLY=bicw_allmass_scat34d` (and B+Scat4) with `SIGMA=`, `NT=`, `N_BULK=`, `LABEL=maxent`,
  `POTENTIALS_FROM=<fitted_potentials>`, `TERMINAL_PHASE=true`, QoS t3 (≤20 h).
- Results: `turbulence/maxent_bound/{table,extrapolation}.csv`, `turbulence/maxent_bound.ipynb`,
  scripts `turbulence/maxent_bound_analysis.py`, `viz_maxent_bound.py`, `maxent_bound_pooled_fit.py`.
- Outcome: see section 4.

### 3.3 1D bimodal density (known truth)
- K-run MGD vs time-regularised estimator (REG); energy variance vs Cramér–Rao; λ chosen after the run.
- Launch: `bimodal_experiment/launch/bimodal_sweep.sh`, then `bimodal_experiment/launch/select_lambda_A.sh`.
- Results: `bimodal_experiment/results_bim_theta_beta/`; notebook `bimodal_experiment/kl_bimodal.ipynb`
  (LAM='A', saved outputs NOT updated: to be re-run on JZ), figures `KL_bimodal_{paper,truth}_lamA.png`.
- Outcome: see section 4 (rule A).

### 3.4 Jets (Modane)
- Launch: `ARM=full|noscalar|scat2|scat2_scalar bash jets/launch_jet_SDE.sh`; λ and the L_6<0 chain are done offline.
- Results: `jets/saved_results/`, `jets/experiments/`; analysis `jets/logp_jets.py`, `jets/moment_errors_jets.py`.
- Outcome: L_6<0 on `jets_full` with 5 seeds and λ=1e-2 run (2026-11-05); 28–35 statistics/seed with error >1%
  (Scat3 Real and Mod2 Imag) as of 2026-11-11. Final state and `logp_jets.py` wired in: TO COMPLETE.

### 3.5 S&P 500
- Daily series (d=6062) as the single sample, 100 particles, J=6, Q=1; nested models M0–M3.
- Launch: `MODEL=M0|M1|M2|M3 bash snp_experiment/launch/snp_models.sh`; chain with L_6:
  `bash snp_experiment/launch/snp_chain_l6.sh`.
- Outcome: arms without L_6 completed; chain with L_6 not launched (from notes of 2026-11-04): current state TO COMPLETE.

### 3.6 2D cosmic web (Quijote) and 2D turbulence (Navier–Stokes)
- Launch: `bash cosmic_web/launch/cosmic_web.sh` (arms `base`, `scat4`), `turbulence_2d/launch/turb2d.sh`;
  then Θ_reg (`*_theta_reg.sh`) and choice of λ (`*_xseed.sh`). Σ²=5.5, `two_phase` time grid.
- `scat4` production (r=4294, N=200, nt=5000, seeds 900–904, label `scat4_np200_nt5000`): launched on 2026-11-18
  (cosmic job 391581, 2D turbulence job 391585). Outcome and analysis with diagonal/shrunk C: TO COMPLETE
  (the result is not recorded in the repo).
- Validation of shrunk C (r=324, 2414): see section 4.

### 3.7 Synthetic fBm / MRW (`gaussian_experiment/`)
- Data generated by `data/synthetic_data_generator.py`; launch `bash gaussian_experiment/launch_SDE.sh`
  (parameters M, seeds 300–350, n1=5000, nt=2000, σ=1). Used to calibrate the bound on Gaussians (M=32/64).
- Overall outcome: TO COMPLETE (scattered notes in `gauss_*`).

### 3.8 Other targeted tests
- `kl_groups/`: D_k per group of statistics (`launch_kl_groups.sh`); `terminal_phase_test/`; `theta_sign_constraint/`
  (`l6_eps_scan.py`, `explore_l6_eps_scan.ipynb`); `lam_selection_criterion/` (`turb_xseed.py`, rule A/B);
  `grad_speed/` (benchmarks and fast versions of Scat4 and the other families).

## 4. Key results and important numbers

- **Entropy bound vs σ² and nt (1D turbulence, 2026-11-19/20).** For σ ≥ 2 the bound depends only on x = σ²/nt
  (within <1 nat). H_bound at nt=27500, σ=3.5: B+Scat4 −995.58, scat34d −1002.19 nat (H(p0)=363.25, d=256).
  Combined fit of all σ as g(x), x→0: g(0) ≈ −1170 ± 6 nat (both models) on 11-19. On 11-20 the σ²=0.1 runs
  (nt=27500) give H_bound −1219.96 (σ=0.316), −1180.05 (σ=0.5), −1081.49 (σ=1.58), −1026.39 (σ=2.35): the value at
  σ=0.316 is 50 nat below the previous estimate, so −1170 was NOT the limit. Not converged in nt. The nt=85000 runs
  (jobs 454513/15/16/17) were expected around 12:00 on 11-20: outcome TO COMPLETE.
- **Rule for choosing λ (2026-11-18): rule A** = cross-seed CV, metric M, t ≥ 0.99, uniform weights.
  Bimodal (job 382786, combine 391219), KL/best for A vs B vs C (Guth λ=1) at β=0.1/0.9/1.9: A 1.24/1.87/117×,
  B 1.55/2.74/131×, C 72/244/9.4×. B is never better than A. At β=1.9 A and B choose 1e-5 versus the oracle 1e-2
  (common bias). Turbulence B+Scat4 (job 380172): A 1e-3 (window 0.99).
- **Bimodal, September sweep re-selected with cross-dataset rule A (2026-11-19, job 426369):** λ_A from 3e-4 (β=0.1)
  to 1e-6 (β=1.9); differs from the old z² rule only at β=0.1, 1.1, 1.5.
- **Sign of L_6 (11/2026):** standard method `pinpos` with ε=5e-3 for Lagrangian turbulence (decision 2026-11-22;
  before that `pinall` ε=1e-2, 11-18). Valid only for local quantities (ranking of −log p, families); NOT for the
  absolute log Z (the far "bump" grows ∝ ε⁻²). Check on 5 seeds: 20 pairs out of 22 with L_6/scalar energy ratio < 0.1.
- **Diagonal/shrunk C in Θ_reg (2026-11-17/18):** full C with N=100 walkers is unusable (energy std 1e3–1e5
  nat); the shrunk form "shrG" (Gaussian plug-in, α* ≈ 0.04–0.15) is consistent between 2D turbulence and cosmic web
  (d_ref/floor 1.3–1.9 at N=100, r=324); pure diagonal has an N-independent bias that grows with λ.
  Choice of how to approximate C: the user decided "shrinking C is the way".
- **Speed, Scat4 (2026-11-22, `notes/scat4_fast_report_1122.md`):** exact versions of Scat4 save ~157 ms/step
  (21%) at M=256 and ~620 ms at M=1024; with `torch.compile` ~220/~880 ms. Errors ≤ 1.6× the original against float64.
  Nothing has become the default yet.
- **Speed, other families (2026-11-22/23, `notes/fast_families_report_1122.md`; job 575821, M=256, n1=5000, 200 steps,
  full model r=457):** the loop goes from 0.393 to 0.268 s/step (−32%) with all fast versions in eager mode; with
  `torch.compile` 0.282 s/step (not faster than eager in the loop; cause not identified). Errors vs float64: Scat3 and
  Scat4 1.0–2.0× the original, GGD (Scalar_psi/morlet) up to 4.4× with compile (absolute ~3e-7). G, eta, theta:
  differences negligible compared with the noise. Loop: walkers 3.2e-3 relative, moments up to ~2.7 SE, same as the
  float64 baseline (2.7 SE). Eager savings of the families ~38 ms/step (batch 5000) at M=256: Scat3 forward by Parseval
  3.10→0.89 ms, GGD rfft/irfft ~20 ms/step; L_6 irfft is slower (rejected). GGD with compile: much faster per call but
  error up to 4.4× and no gain in the loop. M=1024 bench (job 575827) = original code only: 1.30 s/step for the 5
  families, Scalar_psi 46%, pointwise operations dominate (D_k 28–42%), not the FFTs. Eager estimate at M=1024
  ≈ 0.29 s/step (not measured: the fast versions have never been run at M=1024).
- **Fast vs original, long runs (2026-11-23; scat34d, seed 900, n1=5000, nt=27500; runs 608990/624368 M=128,
  624370/624373 M=64; comparison jobs 628852/628871, `grad_speed/compare_fast_orig_run.py`, results in
  `grad_speed/results_fast_vs_orig_M*/`):** s/step M=64 0.071 → 0.053 (−26%), M=128 0.159 → 0.113 (−29%).
  Moments at t=1: same |z| distribution (max 32.0 vs 32.1 SE at M=64, 43.6 vs 43.9 at M=128; the original itself
  misses 4–7 moments by > 3 SE), fast − original ≤ 0.45 SE. θ_t: fast − original ≤ 1.1 × the step-to-step fluctuation
  of θ_t (50-step window, M=64); near t=1 θ_t fluctuates by ~100% per step in both. Verdict: equivalent.
  Scaling with M (orig, n1=5000): ×2.24 (64→128), ×2.4 (128→256), driven by r growing with J (255/347/457) in the
  n1·r·M Jacobian/projection terms and the n1·r²·M Gram.
- **Signal-length benchmark (2026-11-20, M=256, J=8):** Jacobian 58%, elementwise kernels 56%, FFT 17%.
- **Reference timings M=256:** M3 4.0 h, bicw 4.0 h, allmass 4.6 h, Scat4 18.2 h; B+Scat4 0.575 s/step, scat34d 0.747 s/step.
- **New vs old Lagrangian dataset (2026-11-20):** they agree on pdf, energy per scale, event categories;
  kurtosis at Q=3 is 2–13× larger in the new one at the finest scales (uncertainty: a single event weighs 10–29%).

## 5. Decisions made and why

- **Final turbulence model = `scat34d`** (L_6, scalars, Scat3 Re/Im, Scat4 diag). Why: the comparison with B+Scat4
  shows no loss and it costs less than full Scat4. Log p is studied after the choice of λ. (2026-11-18)
- **λ with rule A** (not B, not Guth λ=1). Why: the B curve is flat and does not resolve λ; Guth λ=1 fails for weak β. (2026-11-18)
- **New Lagrangian dataset for all runs from now on** (`lagr_dataset_1_32k.npz`); loader and `run_SDE` not yet updated.
  Coarse-graining only once is acceptable; M=1024 is preferred if the cost is sustainable. (2026-11-20)
- **Sign of L_6: pinpos, ε=5e-3** for Lagrangian turbulence; discussion with collaborators pending. (2026-11-22)
- **Shrink C** (not N ≥ r, impossible at r=4294). (2026-11-18)
- **No use of `prepost` on JZ**, not even for CPU-only jobs: always a GPU dev QoS. (2026-11-17)
- **W&B is not used** in the code and must not be added unless asked.
- **nt=85k as the main nt** in the scan (not 90k). (2026-11-20)
- **The fast versions of Scat4/families do not become the default without explicit approval** (separate flags, no change to production).
- **Signal length M=512 = the 1024-sample Lagrangian signals coarse-grained once (Db3) (2026-11-23).** Why: one coarse
  graining discards 0.48 % of the acceleration variance, against a 1.47 % sampling error of that variance over 10000
  signals (ratio 0.32; `lagr_vs_old_turbulence/compare_lib.coarse_loss_exact`); the acceleration spectrum peaks at
  f = 0.0068 cycles/sample, > 5 octaves below the cut at 0.25. Caveat: kurtosis of the finest wavelet channel (ξ=0.245)
  shifts by ~2 SE. M=1024 timing test no longer needed for this decision.
- **Fast versions made the standard in `codes/potentials/potentials_1d.py` (2026-11-23)** for all instances of Scat3
  Real/Imag, Scat4 Mod2 Real/Imag and Scalar_GGD_KRegion (undecimated); original code kept only where the fast one
  does not apply (complex filter bank for Scat4 Imag and Scat4 grad(x, v); DWT bank for GGD). L_6 unchanged. No
  per-variant tests (user's decision); smoke job 636650. The GGD fit (BIC windows, pruning) now also uses the new
  code: a fresh fit may resolve near-ties differently (windows, r) than an old run with the same settings; accepted
  by the user. Runs with `POTENTIALS_FROM` keep the old fit exactly.
- **Fast eager versions accepted as equivalent (2026-11-23):** Scat4 J/F/I/A1, Scat3 J/A/V, GGD R (L_6 unchanged), as in
  `grad_speed/run_SDE_fast.py`. Not accepted: `torch.compile`; chunked-bmm Gram not tested in the long runs. How to
  adopt them (wrapper vs flag in `codes/`) and the test at the production M still to decide.

## 6. Open hypotheses and known problems

- The entropy bound is not converged in nt and σ: the h→0 limit remains uncertain (estimates range from −1170 to beyond
  −1220 nat). Linearity in h and noise not measured (a single seed).
- Rule A: a bias common to all seeds is not detectable (e.g. bimodal β=1.9, λ chosen 3 decades below the oracle);
  M versus the Σ metric not tested at r=341 with known truth.
- The far bump of log Z with ε in `pinpos`: absolute log Z not reliable.
- Runs with nt=2000 blow up at step 1274 (step size, not code): do not use them for timings/physics.
- Accuracy of shrunk C at the production N/r (0.047) not measured; the J=3 validation (N=57 vs N=1539) is ready but
  its outcome is not recorded: TO COMPLETE.
- Is Scat3 informative for Eulerian jets? Open question (2026-11-04).
- Oscillating events ("packets") under-produced by the model in 1D turbulence.
- `grad_speed/` fast versions: long-run equivalence shown at M=64/128 (see §4). Not measured: fast versions at the
  production M (M=1024 never run with them), why `torch.compile` gives no gain; the "other uses" check against
  `codes/` callers is needed only if they become a flag in `codes/`. Only seed 900 and turbulence scat34d tested.
- Smoke-test divergence at n1=200: also with the original code at M=128 (smoke 607137, NaN at step 17): not caused by
  the fast versions; cause not identified, does not appear at n1=5000.
- Some smoke tests of the large models with `n1 ≤ 200` produce NaN (absurd step): not indicative of the physics.
- Names such as `cross_scale_scat34_local.ipynb` etc.: duplicate/local notebooks to be tidied. TO COMPLETE (list).
- Many files are untracked in git (see `git status`); the last commit cited in the notes is 60b4986 (2026-11-18), not
  pushed to a remote.

## 7. Next steps

- Max-ent scan: nt=85k results (σ² = 0.1, 0.25, 2.5, 5.5) pulled 2026-11-23, notebook re-run (scat34d bound, nt 27.5k → 85k:
  σ=0.316 −1219.96 → −1220.23; σ=0.5 −1180.05 → −1186.48; σ=1.58 −1081.49 → −1133.93; σ=2.35 −1026.39 → −1099.35).
  ON HOLD (2026-11-23). Proposed plateau test at fixed x = σ²/nt (step error cancels if it depends on x only):
  7 scat34d runs, ~31 GPU-h, not launched. Current estimate from existing runs: plateau from σ ≈ 1.5–2, max-ent ≈ −1163 ± 6 nats.
- **M=512 timing test (production M decided; status 2026-11-23 evening).** Loader done: `load_turbulence_1d` in
  `data/data_loader.py` reads the Lagrangian npz when `TURBULENCE_1D_DATA_PATH` ends in `.npz`: 10000 random rows
  (`default_rng(0)`, without replacement), one row per trajectory (`selection_i` distinct, asserted, so never two
  components of one particle); components split 3280/3389/3331; held-out 22768 = complement of the same draw (no
  function yet). Used by `run_SDE.py` and `bench_signal_length.py` unchanged (SUBSERIES_LEN=1024, 1 window/row).
  Uncommitted; pushed to JZ by the user with rsync (`jz_push` refuses `data/`).
  The npz upload to JZ was still in progress (105.9 of 268.96 MB): check the size = 268961292 B before using it.
  Meanwhile, timing proxy on the OLD file: step 1 = pre-flight M=512 J=9 (job 638301, H100 dev, `..._scat34d_bench_old`,
  `turbulence/experiments/`), then step 2 = `grad_speed/launch_bench_signal_length.sh` with its `fitted_potentials`
  (~1-1.5 GPU-h, to be approved). Estimate ~1.4 s/step fast at 10000 walkers; budget 2.33 s/step at nt 27500.
  Status (2026-11-24, from notes; JZ unreachable that morning): pre-flight 638301 submitted, outcome NOT recorded
  (check `jz_status sacct 638301` / its log); the timing benchmark itself was NOT launched (awaiting user approval).
  Afterwards: pre-flight on the new npz; repeat the benchmark only if r differs noticeably.
  Open: the 10000-of-32768 fit with 22768 held out was never explicitly confirmed by the user.
- `grad_speed`: one fast-vs-original long-run pair at M=512 (after the timing test); decide how to adopt the
  fast versions (done: standard in `codes/`); commit `codes/potentials/potentials_1d.py`, `data/data_loader.py`
  and `grad_speed/` (all uncommitted).
- **TO DELETE on JZ SCRATCH (kept for now, user 2026-11-23): regularised systems of the fast-vs-original runs, 88 GB**,
  under `$SCRATCH/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system/`:
  `turbulence_nested_M3_bicw_allmass_scat34d_speedcheck_M128_nt27500_n15000` (26.7 GB),
  `..._pf_fast_M128_n15000` (26.7 GB), `..._orig_M64_n15000` (14.5 GB), `..._fast_M64_n15000` (14.5 GB),
  `..._speedcheck_M128_n15000` (5.8 GB, failed run 607992). Needed only to compare Θ_reg fast vs original.
- Wire in the new Lagrangian dataset (`data/data_loader.py`, `turbulence/run_SDE.py`) and launch the run with
  `pinpos` ε=5e-3; choose N and length (1024 vs coarse-graining).
- Apply rule A to scat34d and to the non-unimodal arms; compare the log p chain.
- Read the results of the shrunk-C production (cosmic web and 2D turbulence) and of the J=3 validation.
- Re-run `kl_bimodal.ipynb` on JZ to update the outputs (rule A).
- `bimodal_experiment/mala_newton_bimodal.ipynb` (untracked; MALA/Newton on the bimodal density) was in a feedback round on
  2026-11-23: comments are collected first and applied in one go on the user's "go" (one execution); state of the
  comments: TO COMPLETE.
- Claude Desktop access (set up 2026-11-22): this file exists, but the MCP "filesystem" server is NOT configured. Node/npx
  is missing in both WSL and Windows; to install on Windows (Desktop runs npx directly) or in WSL (launched via `wsl.exe`),
  then write the MCP config. Repo path from Windows: `\\wsl.localhost\Ubuntu-22.04\home\chiaraz\phd\MGD-for-Maximum-Entropy-Generation`.
  Whether this progressed on 2026-11-23: unknown.
- Complete S&P with L_6 (chain `snp_chain_l6.sh`) and the wiring of `logp_jets.py`.
