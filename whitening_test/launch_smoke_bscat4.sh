#!/bin/bash
# Smoke test (2026-11-13) of --whiten_alpha on real data, B+Scat4 model (turbulence, M = 256), tiny sizes:
#  1. whitening_test/check_whiten_potentials.py: phi~(y) = phi(x) and grad_y = C^(a/2) grad_x for all 5 potential
#     families with the frozen fitted windows, + negative control (Scat4 both layers whitened must fail)
#  2. run_SDE.py with the options of the B+Scat4 sweep runs (turbulence_nested_M3_bicw_allmass_scat4_pf), frozen windows
#     (POTENTIALS_FROM = the 11-01 reference run), n1 200, nt 50. Not informative on the science.
# Usage (repo root): LABEL=whiten_bscat4_smoke ALPHA=0.75 bash whitening_test/launch_smoke_bscat4.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
LABEL="${LABEL:?set LABEL (must contain smoke)}"
ALPHA="${ALPHA:?set ALPHA (whitening exponent, e.g. 0.75)}"
# SCHED two_phase (as the real runs; at nt 50 its tail is unstable for B+Scat4 with or without whitening: smokes 148892/149211)
#       uniform  (power exponent 1, ends at t = 0.98: tests the whole pipeline incl. saving samples in x)
SCHED="${SCHED:-two_phase}"
SIGMA="${SIGMA:-2}"
case "${SCHED}" in
  two_phase) SCHED_ARGS="--schedule two_phase --schedule_exponent 2 --n_bulk 10 --t_switch 0.9 --gap_end 5e-5" ;;
  uniform)   SCHED_ARGS="--schedule power --schedule_exponent 1" ;;
  *) echo "Error: SCHED must be two_phase or uniform"; exit 1 ;;
esac
PF="${PROJECT_DIR}/turbulence/experiments/turbulencesynth_M256_J8_Q3_sigma3.5_nt55000_n1_10000_lam0.0_thomas_nsub1_moment_f64_dedupfilt1e-08_terms9296e719_turbulence_nested_M3_bicw_allmass_scat4/turbulencesynth_M256_J8_Q3_sigma3.5_nt55000_n1_10000_lam0.0_thomas_nsub1_moment_f64_dedupfilt1e-08_seed_900_terms9296e719_turbulence_nested_M3_bicw_allmass_scat4_20261101_1714/fitted_potentials"
OUT="${HERE}/smoke/${LABEL}_a${ALPHA}_${SCHED}/turbulence"
LOGS="${HERE}/smoke/logs"
mkdir -p "${LOGS}" "${OUT}/reg_system"
for d in samples lagrange_multipliers lagrange_multipliers_regularised entropy_bounds sampling_times aux_moments; do
    mkdir -p "${OUT}/saved_results/${d}"
done
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${LABEL}
#SBATCH -A wbg@h100
#SBATCH -C h100
#SBATCH --partition=gpu_p6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:15:00
#SBATCH --output=${LOGS}/slurm_%x_%j.log
#SBATCH --qos=qos_gpu_h100-dev
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
set -x
python -u whitening_test/check_whiten_potentials.py --potentials_from ${PF} --alpha ${ALPHA} --n 200 || exit 1
python -u turbulence/run_SDE.py --seed 900 --n1 200 --subseries_len 512 --target_len 256 --nt 50 --J 8 --Q 3 \
 --sigma ${SIGMA} --interpolant Cos ${SCHED_ARGS} \
 --regularization 1e-4 --lam 0.0 --n_subsample 1 --reg_solver thomas --reg_ridge 0.0 --solve_float64 --cond_every 10 \
 --terms L_6 Scalar_psi_gaussianK Scalar_morlet_gaussianK Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1 \
 --deduplicate_filters --dedup_tol 1e-8 --ggd_tail_quantile 0.99 --ggd_tail_mass --ggd_all_masses \
 --ggd_bic_min_window 1.0 --ggd_noise_ref_gap 0.01 --drop_stats Scalar_morlet_gaussianK:8:1 --gram_preflight 1e-6 \
 --potentials_from ${PF} --whiten_alpha ${ALPHA} \
 --save_reg_system --reg_system_dir ${OUT}/reg_system --skip_reg_solve \
 --batch_size 200 --time_limit_min 10 --outdir ${OUT} --label ${LABEL}
EOT
