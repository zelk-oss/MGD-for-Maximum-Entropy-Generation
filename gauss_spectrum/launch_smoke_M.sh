#!/bin/bash
# Smoke test (2026-11-12) of the M scan: the Gaussian-with-data-spectrum run at M = 32 / 64 (--target_len, J = log2 M),
# tiny sizes (settings of run_sigma_task.sh, turbulence). Checks that coarse-graining 512 -> M, the J = log2 M filter bank
# (Q 3) and the rest of the pipeline run, and prints the number of filters / statistics.
# Usage (repo root): LABEL=mscan_smoke MLEN=32 JSC=5 bash gauss_spectrum/launch_smoke_M.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
LABEL="${LABEL:?set LABEL (must contain smoke)}"
MLEN="${MLEN:?set MLEN (32 or 64)}"
JSC="${JSC:?set JSC (5 for 32, 6 for 64)}"
OUT="${HERE}/smoke/turbulence_M${MLEN}"
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
python -u turbulence/run_SDE.py --seed 900 --n1 200 --subseries_len 512 --target_len ${MLEN} --nt 50 --J ${JSC} \
 --sigma 1 --interpolant Cos --schedule_exponent 2 --regularization 1e-4 --lam 0.0 --n_subsample 1 \
 --reg_solver thomas --reg_ridge 0.0 --time_limit_min 10 --outdir ${OUT} \
 --terms Scattering_Second_Order_Q1 --label ${LABEL} --batch_size 200 --solve_float64 --cond_every 10 \
 --schedule two_phase --n_bulk 10 --t_switch 0.9 --gauss_spectrum \
 --save_reg_system --reg_system_dir ${OUT}/reg_system --skip_reg_solve \
 --Q 3 --gap_end 5e-5
EOT
