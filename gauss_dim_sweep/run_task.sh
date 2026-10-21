#!/bin/bash
# One seed of the dimension sweep (README.md): for each M in MS, an MGD run on white Gaussian
# data (fGn H = 0.5, intermittency 0) with order-2 scattering statistics only, then Theta_reg
# re-solved from the saved (MB-sized) system for every lam in LAMS.
# Called by launch.sh (one array task per seed) and by hand for the local smoke test.
#
# Usage: bash gauss_dim_sweep/run_task.sh SEED
#   env: OUT (base dir: experiments/, saved_results/, results/; default this folder),
#        MS ("4 16 64"), N1 (5000), NT (10000), NBULK (2000), SIGMA (3.5), RIDGE (1e-4),
#        LAMS ("1e-4 1e-3 1e-2"), DATA_SEED (0), LABEL (dimsweep_2ph), EXTRA (more run_SDE args)
set -euo pipefail

SEED="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
OUT="${OUT:-${HERE}}"
MS=(${MS:-4 16 64})
N1="${N1:-5000}"
NT="${NT:-10000}"
NBULK="${NBULK:-2000}"
SIGMA="${SIGMA:-3.5}"
RIDGE="${RIDGE:-1e-4}"
LAMS=(${LAMS:-1e-4 1e-3 1e-2})
DATA_SEED="${DATA_SEED:-0}"
LABEL="${LABEL:-dimsweep_2ph}"
EXTRA="${EXTRA:-}"

cd "${PROJECT_DIR}"
# save_results_theta_reg writes into these without creating them (after the SDE: a missing one loses the run)
for d in samples lagrange_multipliers lagrange_multipliers_regularised entropy_bounds sampling_times aux_moments; do
    mkdir -p "${OUT}/saved_results/${d}"
done
for M in "${MS[@]}"; do
    J=$(python -c "import math; print(int(math.log2(${M})) - 1)")
    echo "======== seed ${SEED}, M ${M}, J ${J} ($(date))"
    python -u gaussian_experiment/run_SDE.py \
        --M "${M}" --J "${J}" --Q 1 --terms Scattering_Second_Order_Q1 \
        --hurst 0.5 --intermittency 0.0 --n1 "${N1}" --data_seed "${DATA_SEED}" --seed "${SEED}" \
        --nt "${NT}" --schedule two_phase --n_bulk "${NBULK}" --t_switch 0.9 --gap_end 5e-5 \
        --sigma "${SIGMA}" --regularization "${RIDGE}" --solve_float64 \
        --reg_solver thomas --n_subsample 1 --reg_mode moment --reg_ridge "${RIDGE}" --lam 1e-3 \
        --save_reg_system --reg_system_dir "${OUT}/results/reg_system" \
        --outdir "${OUT}" --label "${LABEL}" ${EXTRA}
    shopt -s nullglob
    SYS=( "${OUT}"/results/reg_system/gaussiansynth_*_M${M}_J${J}_*seed_${SEED}_*${LABEL}*.pt )
    shopt -u nullglob
    if [ ${#SYS[@]} -ne 1 ]; then echo "Error: expected 1 system for M ${M} seed ${SEED}, found ${#SYS[@]}"; exit 1; fi
    python -u codes/resolve_theta_reg.py "${SYS[0]}" --lams "${LAMS[@]}" --ridge "${RIDGE}" \
        --schedule uniform --mask auto --diagnose 0 --outdir "${OUT}/results/theta_reg"
done
echo "======== seed ${SEED} done ($(date))"
