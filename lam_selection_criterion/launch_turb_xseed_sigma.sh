#!/bin/bash
# Phase 2 only of the cross-seed lam choice, with rule B (turb_xseed.py --sigma_rule, 2026-11-18):
# rule A (M_B metric, unchanged) + rule B (Sigma_A metric, t >= ENDS) + the M_B check + jackknife,
# from EXISTING phase-1 Theta files (no new solve). Output: ${OUT}/xseed_<ARM>_sigma.pt + slurm log.
# Cost measured for rule A alone: 282 s for 4 seeds (ggd_mass_final, job 1747112); rule B reads in
# addition each seed's Sigma on the t >= min(ENDS) nodes (memory-mapped). One GPU dev job (the GPU is idle:
# CPU work; billed GPU x elapsed). Never prepost (user decision 2026-11-17).
#
# Usage (repo root on Jean Zay):
#   DRY=1 GPU=h100 ARM=turbulence_nested_M3_bicw_allmass_scat4 RIDGE=1e-4 \
#       SWEEP_DIR=$SCRATCH/MGD-for-Maximum-Entropy-Generation/turbulence/theta_reg_lamsweep_xseed/bscat4 \
#       LAMS="0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1 10" bash lam_selection_criterion/launch_turb_xseed_sigma.sh
# settings: GPU=h100|v100 (required), ARM, RIDGE (required), SWEEP_DIR, LAMS, ENDS (default "0.99 0.999"),
#   REG_SYSTEM_DIR (default .../turbulence/reg_system/<ARM>), SYSTEM_PATTERN (default *<ARM>*.pt),
#   SEEDS (optional, e.g. "900 901": keep only these seeds' systems), TIME (default 01:00:00), CPUS, OUT, LABEL.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"

if [ -z "${ARM}" ] || [ -z "${RIDGE}" ] || [ -z "${GPU}" ]; then
    echo "Set GPU (h100|v100), ARM and RIDGE"; exit 1
fi
REG_SYSTEM_DIR="${REG_SYSTEM_DIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system/${ARM}}"
SYSTEM_PATTERN="${SYSTEM_PATTERN:-*${ARM}*.pt}"
SWEEP_DIR="${SWEEP_DIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/theta_reg_lamsweep_xseed}"
LAM_LIST=(${LAMS:-0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1 10})
END_LIST=(${ENDS:-0.99 0.999})
OUT="${OUT:-${HERE}/results/turb_xseed}"
TIME="${TIME:-01:00:00}"
LABEL="${LABEL:-${ARM}}"

case "${GPU}" in
    h100) CPUS="${CPUS:-24}"; HEADER="#SBATCH -A wbg@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH -C h100"
          MODULES="module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0" ;;
    v100) CPUS="${CPUS:-10}"; HEADER="#SBATCH -A wbg@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH -C v100-32g"
          MODULES="module purge
module load pytorch-gpu/py3/2.8.0" ;;
    *) echo "GPU must be h100 or v100"; exit 1 ;;
esac

shopt -s nullglob
ALL=( "${REG_SYSTEM_DIR}"/${SYSTEM_PATTERN} )
shopt -u nullglob
FILES=()
if [ -n "${SEEDS}" ]; then
    for S in ${SEEDS}; do
        for F in "${ALL[@]}"; do [[ "${F##*/}" == *_seed_${S}_* ]] && FILES+=( "${F}" ); done
    done
else
    FILES=( "${ALL[@]}" )
fi
echo "arm ${ARM}, ridge ${RIDGE}: ${#FILES[@]} systems in ${REG_SYSTEM_DIR} matching ${SYSTEM_PATTERN}${SEEDS:+ (seeds ${SEEDS})}"
printf '  %s\n' "${FILES[@]##*/}"
echo "lams ${LAM_LIST[*]} | ends ${END_LIST[*]} | sweep ${SWEEP_DIR} | out ${OUT} | ${GPU} dev QoS, ${CPUS} CPUs, ${TIME}"
if [ ${#FILES[@]} -lt 2 ]; then echo "Error: need at least 2 seeds"; exit 1; fi
if [ -n "${DRY}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${OUT}"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=xseed_sigma_${LABEL}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${OUT}/slurm_%x_%j.log
${HEADER}
${MODULES}
cd "${PROJECT_DIR}"
python -u lam_selection_criterion/turb_xseed.py --systems ${FILES[@]} --sweep_dir "${SWEEP_DIR}" \
    --lams ${LAM_LIST[@]} --ends ${END_LIST[@]} --ridge ${RIDGE} --arm ${LABEL} --out "${OUT}" --sigma_rule
EOT
