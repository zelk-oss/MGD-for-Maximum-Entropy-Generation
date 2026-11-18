#!/bin/bash
# Phase 1 only of the cross-seed lam choice (2026-11-18): solve Theta_reg(lam) for EXTRA lam values on the
# EXISTING saved systems of one arm, one array task per seed, on the H100/V100 dev QoS (the t3 default of
# launch_turb_xseed.sh is not accepted by the guard). Already-solved (system, lam) pairs are skipped by
# codes/resolve_theta_reg.py. Writes only Theta files in SWEEP_DIR (next to the existing ones, same naming).
# Then score with launch_turb_xseed_sigma.sh using the union of the old and new LAMS.
# Cost NOT measured: one Thomas solve ~5-6 min (inferred, launcher header of launch_turb_xseed.sh) + loading the
# ~49 GB system per task; CPU/RAM-bound, the GPU is idle (billed GPU x elapsed).
#
# Usage (repo root on Jean Zay):
#   DRY=1 GPU=h100 ARM=turbulence_nested_M3_bicw_allmass_scat4 RIDGE=1e-4 LAMS="1e-7 1e-8" \
#       SWEEP_DIR=$SCRATCH/MGD-for-Maximum-Entropy-Generation/turbulence/theta_reg_lamsweep_xseed/bscat4 \
#       bash lam_selection_criterion/launch_turb_solve_lams.sh
# settings: GPU=h100|v100, ARM, RIDGE, LAMS, SWEEP_DIR (required); REG_SYSTEM_DIR (default .../reg_system/<ARM>),
#   SYSTEM_PATTERN, SEEDS (optional filter), MMAP (default 1: --mmap, big systems), TIME (default 01:30:00),
#   CPUS, OUT (slurm logs), LABEL.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
SWEEP_DIR="${SWEEP_DIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/theta_reg_lamsweep_xseed/${SWEEP_SUB}}"
for v in GPU ARM RIDGE LAMS SWEEP_SUB; do
    if [ -z "${!v}" ]; then echo "Set ${v}"; exit 1; fi
done
REG_SYSTEM_DIR="${REG_SYSTEM_DIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system/${ARM}}"
SYSTEM_PATTERN="${SYSTEM_PATTERN:-*${ARM}*.pt}"
OUT="${OUT:-${HERE}/results/turb_solve_lams}"
TIME="${TIME:-01:30:00}"
LABEL="${LABEL:-${ARM}}"
MMAP_ARG=""; [ "${MMAP:-1}" = "1" ] && MMAP_ARG="--mmap"
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
echo "arm ${ARM}, ridge ${RIDGE}: ${#FILES[@]} systems in ${REG_SYSTEM_DIR}${SEEDS:+ (seeds ${SEEDS})}"
printf '  %s\n' "${FILES[@]##*/}"
echo "new lams ${LAMS} | sweep ${SWEEP_DIR} | logs ${OUT} | ${GPU} dev QoS, ${CPUS} CPUs, ${TIME}, ${MMAP_ARG:-no mmap}"
if [ ${#FILES[@]} -lt 1 ]; then echo "Error: no system found"; exit 1; fi
if [ -n "${DRY}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${OUT}" "${SWEEP_DIR}"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=solve_lams_${LABEL}
#SBATCH --array=0-$(( ${#FILES[@]} - 1 ))
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${OUT}/slurm_%x_%A_%a.log
${HEADER}
${MODULES}
cd "${PROJECT_DIR}"
FILES=( ${FILES[@]} )
python -u codes/resolve_theta_reg.py "\${FILES[\${SLURM_ARRAY_TASK_ID}]}" --lams ${LAMS} \
    --ridge ${RIDGE} --schedule uniform --mask auto --diagnose 0 --outdir "${SWEEP_DIR}" ${MMAP_ARG}
EOT
