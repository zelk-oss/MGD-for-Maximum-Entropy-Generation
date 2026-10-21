#!/bin/bash
# Phase 2 of the dimension sweep (README.md), after every seed's run_task.sh: for each M, the
# cross-seed lam choice (lam_selection_criterion/turb_xseed.py, the paper's rule) on that M's
# saved systems, then analyse.py on everything (small outputs in ${OUT}/results/).
#
# Usage: bash gauss_dim_sweep/finish.sh      env: OUT, MS, RIDGE, LAMS, LABEL as run_task.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
OUT="${OUT:-${HERE}}"
MS=(${MS:-4 16 64})
RIDGE="${RIDGE:-1e-4}"
LAMS=(${LAMS:-1e-4 1e-3 1e-2})
LABEL="${LABEL:-dimsweep_2ph}"

cd "${PROJECT_DIR}"
for M in "${MS[@]}"; do
    J=$(python -c "import math; print(int(math.log2(${M})) - 1)")
    shopt -s nullglob
    SYS=( "${OUT}"/results/reg_system/gaussiansynth_*_M${M}_J${J}_*${LABEL}*.pt )
    shopt -u nullglob
    echo "======== M ${M}: ${#SYS[@]} seeds"
    if [ ${#SYS[@]} -lt 2 ]; then echo "  fewer than 2 seeds: no cross-seed choice for M ${M}"; continue; fi
    python -u lam_selection_criterion/turb_xseed.py --systems "${SYS[@]}" \
        --sweep_dir "${OUT}/results/theta_reg" --lams "${LAMS[@]}" --ridge "${RIDGE}" \
        --arm "M${M}" --out "${OUT}/results"
done
python -u gauss_dim_sweep/analyse.py --root "${OUT}" --ms "${MS[@]}" --lams "${LAMS[@]}" \
    --ridge "${RIDGE}" --label "${LABEL}"
