# Gaussian with the data spectrum: MGD log p (2026-11-03)

**Question.** What does the MGD log p look like when the data are exactly Gaussian, with the power
spectrum of (a) the Lagrangian turbulence data and (b) the jets (Modane) data? The earlier
`gauss_dim_sweep/` did this with a WHITE spectrum, which is not the data's.

**Data.** `--gauss_spectrum` (`codes/gauss_spectrum.py`), in both `turbulence/run_SDE.py` and
`jets/run_jet_SDE.py`: after the usual windowing / coarse-graining / normalisation, x1 is replaced
by n1 draws of ONE stationary Gaussian with spectrum S(w) = mean_n |FFT(x1_n)(w)|^2 (data seed 0,
same x1 for every run seed). Not per-signal phase randomisation (that keeps each window's energy:
a mixture of Gaussians). S is saved to `<exp_dir>/gauss_spectrum.pt`; the exact log density is
`true_log_p(x, S)` = -1/2 sum_w |x_hat|^2 / S - 1/2 sum_w log(2 pi S / M).

**Model.** `Scattering_Second_Order_Q1` only (9 statistics, Q = 1 Morlet, J = 8, M = 256), as
`turbulence_final_gauss_scat2`, so the model is a stationary Gaussian too.

| | turbulence | jets |
|---|---|---|
| launcher | `turbulence/launch/turbulence_gauss_spectrum.sh` | `ARM=gauss_spectrum jets/launch_jet_SDE.sh` |
| seeds | 900-904 | 500-504 |
| n1, nt | 10000, 10000 (two_phase, n_bulk 2000, t_switch 0.9) | same |
| gap_end | 5e-5 | 7.5e-4 |
| ridge (per step) | 1e-4 | 1e-4 |
| GPU | V100, 1 h | V100, 1 h |

Theta_reg offline from the saved systems, same lambda / ridge for both data sets (lambda 5e-4,
ridge 1e-4 as `turbulence_final_gauss_scat2`). Then log p = Theta_reg(t_end) . phi(x) on model
samples and on x1, compared with the exact log p.

Entropy check (2026-11-06): `turbulence/logz_gauss_benchmark.py`, per seed:
(1) MGD H_bound(t) vs the exact H(p_t) of the Gaussian interpolant marginal (`codes/utils_entropy.compute_gaussian_entropy`,
upstream-corrected version); (2) log Z exact (Fourier) vs the MGD bound H_bound + theta_end . m(t_end), theta_end =
Theta_reg[-1] or theta_t[-1]; (3) E[-log p] = log Z - mean(Theta_reg . phi) on data / generated, each log Z, vs H(p_1).
Outputs `logz_benchmark.csv`, `path_seed*.npz`. Sigma sweep: `launch_sigma_sweep.sh` (nt = max(2000, 300 sigma^2),
36 runs: SDE + Theta_reg per task, then `finish_sigma_sweep.sh` -> benchmark per group + `sigma_sweep_summary.py`).

Smoke-tested locally 2026-11-03 (n1 40, nt 20): both entry points run, S saved, name tag `gaussspec`.
