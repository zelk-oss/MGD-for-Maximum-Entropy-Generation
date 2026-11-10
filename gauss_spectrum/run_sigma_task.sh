#!/bin/bash
# One run of the sigma sweep (gauss_spectrum/README.md, 2026-11-06): MGD (Scattering_Second_Order_Q1 only, M = 256)
# on the Gaussian with the turbulence (Lagrangian) or Modane (Eulerian) spectrum, then Theta_reg from the saved system.
# Same settings as turbulence/launch/turbulence_gauss_spectrum.sh / ARM=gauss_spectrum jets/launch_jet_SDE.sh except:
#   sigma   argument
#   nt      max(2000, 300 sigma^2): constant h sigma^2 (sigma 10 -> 30000), floor 2000 for the transport
#   n_bulk  nt / 5 (same grid shape as the existing runs: 2000 of 10000 steps on [0, 0.9])
#   outdir  ${OUT}/<dataset> (experiments/, saved_results/, reg_system/), label sigsweep_s<sigma>
#
# Usage: bash gauss_spectrum/run_sigma_task.sh <turbulence|jets> SIGMA SEED
#   env: OUT (default gauss_spectrum/sigma_sweep), TIMESTAMP, TIME_LIMIT_MIN (55),
#        N1 (10000), NT (overrides the rule), NBULK, LABEL (default sigsweep_s<sigma>; must differ between two runs
#        with the same sigma and seed in one OUT, or the reg-system lookup below finds both -- launch_nt_sweep.sh)
set -euo pipefail

DS="$1"; SIGMA="$2"; SEED="$3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
OUT="${OUT:-${HERE}/sigma_sweep}"
DSOUT="${OUT}/${DS}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M)}"
TIME_LIMIT_MIN="${TIME_LIMIT_MIN:-55}"
N1="${N1:-10000}"
NT_RULE=$(python -c "print(max(2000, round(300 * ${SIGMA} ** 2)))")
NT="${NT:-${NT_RULE}}"
NBULK="${NBULK:-$(( NT / 5 ))}"
TAG="${SIGMA//./p}"
LABEL="${LABEL:-sigsweep_s${TAG}}"
REGDIR="${DSOUT}/reg_system"
mkdir -p "${REGDIR}"
for d in samples lagrange_multipliers lagrange_multipliers_regularised entropy_bounds sampling_times aux_moments; do
    mkdir -p "${DSOUT}/saved_results/${d}"
done
cd "${PROJECT_DIR}"

COMMON="--seed ${SEED} --timestamp ${TIMESTAMP} --n1 ${N1} --subseries_len 512 --target_len 256 --nt ${NT} --J 8 \
 --sigma ${SIGMA} --interpolant Cos --schedule_exponent 2 --regularization 1e-4 --lam 0.0 --n_subsample 1 \
 --reg_solver thomas --reg_ridge 0.0 --time_limit_min ${TIME_LIMIT_MIN} --outdir ${DSOUT} \
 --terms Scattering_Second_Order_Q1 --label ${LABEL} --batch_size 5000 --solve_float64 --cond_every 10 \
 --schedule two_phase --n_bulk ${NBULK} --t_switch 0.9 --gauss_spectrum \
 --save_reg_system --reg_system_dir ${REGDIR} --skip_reg_solve"

echo "======== ${DS} sigma ${SIGMA} seed ${SEED}: nt ${NT} (rule ${NT_RULE}), n_bulk ${NBULK}, n1 ${N1} ($(date))"
case "${DS}" in
    turbulence) python -u turbulence/run_SDE.py ${COMMON} --Q 3 --gap_end 5e-5 ;;
    jets)       python -u jets/run_jet_SDE.py ${COMMON} --Q 1 --gap_end 7.5e-4 --dataset modane \
                    --deduplicate_filters --dedup_tol 1e-8 ;;
    *) echo "unknown dataset ${DS}"; exit 1 ;;
esac

shopt -s nullglob
SYS=( "${REGDIR}"/*seed_${SEED}_*${LABEL}*.pt )
shopt -u nullglob
if [ ${#SYS[@]} -ne 1 ]; then echo "Error: expected 1 reg system for seed ${SEED} ${LABEL}, found ${#SYS[@]}"; exit 1; fi
python -u codes/resolve_theta_reg.py "${SYS[0]}" --lams 5e-4 --ridge 1e-4 --schedule uniform --mask auto --diagnose 0 \
    --outdir "${DSOUT}/saved_results/lagrange_multipliers_regularised"
echo "======== done ($(date))"
