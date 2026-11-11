#!/bin/bash
# Smoke test (2026-11-10) of the jets --gauss_spectrum fix: n1 Gaussian draws may exceed the number of Modane windows.
# Settings of run_sigma_task.sh (jets) at tiny size. To hit the "n1 > windows" branch inside the smoke limits (n1 <= 200),
# the windows are made long: SUBLEN 262144 -> ~98 windows in the record, so n1 200 > 98 (S from 98 windows, 200 draws).
# Usage (repo root): LABEL=gsfix_smoke bash gauss_spectrum/launch_smoke_jets.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
LABEL="${LABEL:?set LABEL (must contain smoke)}"
SUBLEN="${SUBLEN:-262144}"
OUT="${HERE}/smoke/jets"
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
python -u jets/run_jet_SDE.py --seed 500 --n1 200 --subseries_len ${SUBLEN} --target_len 256 --nt 50 --J 8 \
 --sigma 1 --interpolant Cos --schedule_exponent 2 --regularization 1e-4 --lam 0.0 --n_subsample 1 \
 --reg_solver thomas --reg_ridge 0.0 --time_limit_min 10 --outdir ${OUT} \
 --terms Scattering_Second_Order_Q1 --label ${LABEL} --batch_size 200 --solve_float64 --cond_every 10 \
 --schedule two_phase --n_bulk 10 --t_switch 0.9 --gauss_spectrum \
 --save_reg_system --reg_system_dir ${OUT}/reg_system --skip_reg_solve \
 --Q 1 --gap_end 7.5e-4 --dataset modane --deduplicate_filters --dedup_tol 1e-8
EOT
