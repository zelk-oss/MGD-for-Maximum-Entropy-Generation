#!/bin/bash
# Phase 2 of the sigma sweep: entropy / log Z benchmark for every (dataset, sigma) group, then the summary figures.
# Usage: bash gauss_spectrum/finish_sigma_sweep.sh      env: OUT (default gauss_spectrum/sigma_sweep)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
OUT="${OUT:-${HERE}/sigma_sweep}"
cd "${PROJECT_DIR}"
for DS in turbulence jets; do
    for G in "${OUT}/${DS}"/experiments/*/; do
        [ -d "${G}" ] || continue
        A="${OUT}/analysis/${DS}/$(basename "${G}")"
        mkdir -p "${A}"
        echo "======== ${DS} $(basename "${G}")"
        python -u turbulence/logz_gauss_benchmark.py --group_dir "${G}" --results_root "${OUT}/${DS}" \
            --theta_reg_dir "${OUT}/${DS}/saved_results/lagrange_multipliers_regularised" \
            --lam 5e-4 --ridge 1e-4 --outdir "${A}" > "${A}/benchmark.log" 2>&1 \
            || echo "  FAIL (see ${A}/benchmark.log)"
        sed -n '/^=== /,$p' "${A}/benchmark.log"; grep "FAIL" "${A}/benchmark.log"
    done
done
python -u gauss_spectrum/sigma_sweep_summary.py --root "${OUT}"
