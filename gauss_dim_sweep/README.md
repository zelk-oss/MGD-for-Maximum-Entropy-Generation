# Gaussian dimension sweep: is the MGD log p chi-squared?

**Question.** For white Gaussian data x ~ N(0, I_M), log p(x) = −|x|²/2 + const, so
y = −2 (log p − const) = |x|² follows χ²_M (mean M, variance 2M). We build the MGD log p exactly as for
turbulence (log p = Θ_reg(t_end) · φ(x), code sign) and check, for M = 4, 16, 64:
1. whether its distribution follows χ²_M, and how that changes with M;
2. whether 4 seeds give the same log p.

## Settings (decided 2026-10-13)

| | |
|---|---|
| data | fGn, H = 0.5, intermittency 0 (white noise), `gaussian_experiment/run_SDE.py`, n1 = 5000, normalised |
| seeds | 900–903; **x1 shared** (`--data_seed 0`), the seed changes only the SDE noise (as turbulence) |
| statistics | `Scattering_Second_Order_Q1` only: φ_l = mean_u \|x ⋆ ψ_l\|², Q = 1, J = log2(M) − 1 → r = 2, 4, 6 |
| grid | two_phase, nt 10000, n_bulk 2000, t_switch 0.9, gap_end 5e-5 (as the krylov d = 64 runs); σ 3.5 |
| per-step solve | ridge 1e-4, float64 |
| Θ_reg | moment mode, uniform weights, Thomas, data ridge 1e-4; λ ∈ {1e-4, 1e-3, 1e-2} |
| λ choice | the saved system is a few MB (r ≤ 6), so it is kept and the paper's cross-seed CV (`lam_selection_criterion/turb_xseed.py`, line `end0.99`) picks λ afterwards; all three λ are analysed anyway |
| log Z | not estimated: log p is centred, y = −2 (lp − mean lp) + M |

## What `analyse.py` computes (exact where possible)

With order-2 statistics, Θ · φ(x) = (1/M) Σ_ω g(ω) |x̂_ω|² (Parseval, asserted in the code), so every model
is a circulant Gaussian with precision −2 g_eff(ω), and the truth is g_eff ≡ −½. This gives:
- **exact KL(truth ‖ model)** = ½ Σ_ω (−2g − 1 − log(−2g)), no sampling;
- **best in family**: the maxent Θ for the true law (convex dual). Its KL is the floor set by the
  wavelet bank: Morlet Q = 1 does not tile the spectrum flat, so **even a perfect MGD run is not exactly
  χ²_M** (smoke test: floor 0.24 nats at M = 4, 0.26 at M = 16; analytic, independent of n1 and the run).
  `kl_minus_best` is what MGD + Θ_reg add;
- on a fresh N(0, I) test set (20000), the data x1 and the model samples: var(y)/2M (target 1), KS distance
  to χ²_M, slope of log p against |x|² (target −½), correlation;
- seed agreement of log p (Pearson/Spearman between seed pairs, seed spread / signal spread), as
  `krylov_stopping/turb_d64_analysis.py`.

Figures: `chi2_lam<λ>.pdf` (histogram and QQ against χ²_M, per M), `spectral_weight.pdf` (g_eff(ω) per seed
and λ, best in family, −½), `dimension.pdf` (standardised y for the three M on one plot).

## Files

- `run_task.sh SEED`: every M for one seed: run_SDE, then `codes/resolve_theta_reg.py` for every λ.
- `finish.sh`: cross-seed λ choice per M, then `analyse.py`.
- `launch.sh`: SLURM (dev QoS: 4 array tasks + 1 dependent job, under the cap of 10).
- Outputs: `experiments/`, `saved_results/` (samples, θ_t, …) and `results/` (`reg_system/`,
  `theta_reg/`, `xseed_M<M>.pt`, `summary.json`, PDFs, `logs/`). To pull: `results/summary.json`,
  `results/*.pdf`, `results/xseed_*.pt` (small).

## How to run

    DRY=1 bash gauss_dim_sweep/launch.sh          # print the plan
    SEEDS="900" bash gauss_dim_sweep/launch.sh    # timing run (cost not measured yet)
    bash gauss_dim_sweep/launch.sh                # seeds 900-903 (seed 900 reloads, not rerun)

Smoke test (local, 2026-10-13): `OUT=<tmp> MS="4 16" N1=100 NT=40 NBULK=20`, `run_task.sh 900`, `901`,
then `finish.sh`: the whole chain runs; the regenerated x1 matches both runs' target moments to 1e-5.

## Code changes this needed

- `data/synthetic_data_generator.py`: `load_synthetic_data(..., seed=None)`. The generator called
  `np.random.seed(None)` before every batch, so **fGn/MRW data was never reproducible**, whatever
  `--seed` (true for every earlier `gaussian_experiment` run). With `seed`, batch i is drawn under
  `seed + i` (identical for 1 or 4 workers; the parallel branch now keeps batch order). Default unchanged.
- `gaussian_experiment/run_SDE.py`: `--data_seed` (name tag `dseed<N>`), `--schedule two_phase|adaptive`
  (the default `power` grid is identical to the old one, checked), diagnostic figures can no longer fail
  a run whose results are saved (`hist_plot` breaks at M = 4).
