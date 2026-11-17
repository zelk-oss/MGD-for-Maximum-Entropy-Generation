#!/bin/bash
# Whitened B+Scat4 (2026-11-13): does MGD in whitened coordinates (alpha 0.75) give an nt-converged entropy bound and good
# moments with few steps? ONE H100 dev array, one task per nt (uniform grid, ends at t = 1 - 1/nt), sigma 3.5, seed 900,
# n1 1e4, frozen windows of the 11-01 reference (same as the unwhitened sigma x nt sweep runs, which used two_phase nt 27.5k/55k).
# Options identical to whitening_test/launch_smoke_bscat4.sh (smoke 149898 passed) except n1, nt, sigma, batch, time.
# Read: bound(nt 4000) - bound(nt 1000) in x (H(x) = H(y) + shift in run.log), vs unwhitened -995.6 / -1022.5 (sigma 3.5),
#       per-statistic moment errors (turbulence/moment_error_variance.py), stability.
# Regularised systems (~0.9 MB/step) -> $SCRATCH/.../whitening_test/reg_system/bscat4.
# Usage (Jean Zay, repo root): bash whitening_test/launch_bscat4_whiten_dev.sh   (DRY=1: print only)
# overrides: NTS ("1000 4000"), ALPHA (0.75), SIGMA (3.5), N1 (10000), SEED (900), TIME1 (02:00:00), TL (guard minutes, 110)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
NTS=(${NTS:-1000 4000})
ALPHA="${ALPHA:-0.75}"
SIGMA="${SIGMA:-3.5}"
N1="${N1:-10000}"
SEED="${SEED:-900}"
TIME1="${TIME1:-02:00:00}"
TL="${TL:-110}"
PF="${PROJECT_DIR}/turbulence/experiments/turbulencesynth_M256_J8_Q3_sigma3.5_nt55000_n1_10000_lam0.0_thomas_nsub1_moment_f64_dedupfilt1e-08_terms9296e719_turbulence_nested_M3_bicw_allmass_scat4/turbulencesynth_M256_J8_Q3_sigma3.5_nt55000_n1_10000_lam0.0_thomas_nsub1_moment_f64_dedupfilt1e-08_seed_900_terms9296e719_turbulence_nested_M3_bicw_allmass_scat4_20261101_1714/fitted_potentials"
OUT="${HERE}/bscat4/turbulence"
LOGS="${HERE}/bscat4/logs"
TAG="${ALPHA//./p}"
echo "whitened B+Scat4 dev array: nt ${NTS[*]} (uniform), alpha ${ALPHA}, sigma ${SIGMA}, n1 ${N1}, seed ${SEED} -> ${OUT}"
if [ -n "${DRY:-}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${LOGS}"
for d in samples lagrange_multipliers lagrange_multipliers_regularised entropy_bounds sampling_times aux_moments; do
    mkdir -p "${OUT}/saved_results/${d}"
done

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=wh_bscat4
#SBATCH -A wbg@h100
#SBATCH -C h100
#SBATCH --partition=gpu_p6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-$(( ${#NTS[@]} - 1 ))
#SBATCH --time=${TIME1}
#SBATCH --output=${LOGS}/slurm_%x_%A_%a.log
#SBATCH --qos=qos_gpu_h100-dev
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NTS=( ${NTS[@]} )
NT=\${NTS[\${SLURM_ARRAY_TASK_ID}]}
REGDIR="\${SCRATCH:?SCRATCH not set}/MGD-for-Maximum-Entropy-Generation/whitening_test/reg_system/bscat4"
mkdir -p "\${REGDIR}"
python -u turbulence/run_SDE.py --seed ${SEED} --n1 ${N1} --subseries_len 512 --target_len 256 --nt \${NT} --J 8 --Q 3 \
 --sigma ${SIGMA} --interpolant Cos --schedule power --schedule_exponent 1 \
 --regularization 1e-4 --lam 0.0 --n_subsample 1 --reg_solver thomas --reg_ridge 0.0 --solve_float64 --cond_every 10 \
 --terms L_6 Scalar_psi_gaussianK Scalar_morlet_gaussianK Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1 \
 --deduplicate_filters --dedup_tol 1e-8 --ggd_tail_quantile 0.99 --ggd_tail_mass --ggd_all_masses \
 --ggd_bic_min_window 1.0 --ggd_noise_ref_gap 0.01 --drop_stats Scalar_morlet_gaussianK:8:1 --gram_preflight 1e-6 \
 --potentials_from ${PF} --whiten_alpha ${ALPHA} \
 --save_reg_system --reg_system_dir \${REGDIR} --skip_reg_solve \
 --batch_size 5000 --time_limit_min ${TL} --outdir ${OUT} --label wh_bscat4_a${TAG}_unif_nt\${NT}
EOT
