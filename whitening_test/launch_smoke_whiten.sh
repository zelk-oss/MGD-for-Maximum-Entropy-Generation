#!/bin/bash
# Smoke test (2026-11-13) of --whiten_alpha on the Gaussian-with-data-spectrum run (turbulence, M = 256, Scat2 Q1):
# tiny sizes, then Theta_reg and the entropy benchmark on the result, so the whole whitened path (run_SDE ->
# resolve_theta_reg -> logz_gauss_benchmark with whitened filters/data) is exercised. Not informative on the science.
# Usage (repo root): LABEL=whiten_smoke ALPHA=0.5 [SCHED=two_phase|uniform] bash whitening_test/launch_smoke_whiten.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
LABEL="${LABEL:?set LABEL (must contain smoke)}"
ALPHA="${ALPHA:?set ALPHA (whitening exponent, e.g. 0.5)}"
SCHED="${SCHED:-two_phase}"
case "${SCHED}" in
  two_phase) SCHED_ARGS="--schedule two_phase --schedule_exponent 2 --n_bulk 10 --t_switch 0.9" ;;
  uniform)   SCHED_ARGS="--schedule power --schedule_exponent 1" ;;
  *) echo "Error: SCHED must be two_phase or uniform"; exit 1 ;;
esac
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
python -u turbulence/run_SDE.py --seed 900 --n1 200 --subseries_len 512 --target_len 256 --nt 50 --J 8 \
 --sigma 2 --interpolant Cos --regularization 1e-4 --lam 0.0 --n_subsample 1 \
 --reg_solver thomas --reg_ridge 0.0 --time_limit_min 10 --outdir ${OUT} \
 --terms Scattering_Second_Order_Q1 --label ${LABEL} --batch_size 200 --solve_float64 --cond_every 10 \
 ${SCHED_ARGS} --gauss_spectrum --whiten_alpha ${ALPHA} \
 --save_reg_system --reg_system_dir ${OUT}/reg_system --skip_reg_solve \
 --Q 3 --gap_end 5e-5 || exit 1
SYS=\$(ls ${OUT}/reg_system/*${LABEL}*.pt)
python -u codes/resolve_theta_reg.py "\${SYS}" --lams 5e-4 --ridge 1e-4 --schedule uniform --mask auto --diagnose 0 \
    --outdir ${OUT}/saved_results/lagrange_multipliers_regularised || exit 1
G=\$(ls -d ${OUT}/experiments/*/)
python -u turbulence/logz_gauss_benchmark.py --group_dir "\${G}" --results_root ${OUT} \
    --theta_reg_dir ${OUT}/saved_results/lagrange_multipliers_regularised --lam 5e-4 --ridge 1e-4 \
    --outdir ${OUT}/analysis
EOT
