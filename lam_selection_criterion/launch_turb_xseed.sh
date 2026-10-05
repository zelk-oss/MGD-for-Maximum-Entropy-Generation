#!/bin/bash
# Cross-seed choice of lam for one turbulence arm (lam_selection_criterion/turb_xseed.py).
#   phase 1: array, one task per seed: codes/resolve_theta_reg.py solves the seed's saved
#            moment-mode system for every lam in LAMS (uniform weights, the run's ridge, saved
#            live mask) -> Theta files in SWEEP_DIR (~124 MB each at n 60000, r 258: on $SCRATCH).
#            Already-solved (system, lam) pairs are skipped, so a killed task can be resubmitted.
#   phase 2: one task, starts when phase 1 succeeded (afterok): scores every seed's Theta(lam) with
#            the other seeds' data terms, pools the pairs, prints the selected lam (whole transport,
#            t >= 0.99, t >= 0.999) -> ${OUT}/xseed_<ARM>.pt + slurm log.
# Cost NOT measured: phase 1 ~ LAMS x one Thomas solve (5-6 min each, inferred 2026-10-01, never
# timed) + loading the ~32 GB system; RAM ~ system + Thomas factor (~65 GB) -> CPUS=24 (~120 GB).
# Default queue: H100 t3 (QOS=""), because the dev QoS allows only 10 submitted jobs per user.
#
# Usage (from the repo root on Jean Zay), one launch per arm; first a dry run to see the files:
#   DRY=1 ARM=noL6psi_noSm30_reg1e-4 RIDGE=1e-4 bash lam_selection_criterion/launch_turb_xseed.sh
#   ARM=noL6psi_noSm30_reg1e-4 RIDGE=1e-4 bash lam_selection_criterion/launch_turb_xseed.sh
#   ARM=noL6psi_noSm30_reg1e-2 RIDGE=1e-2 bash lam_selection_criterion/launch_turb_xseed.sh
# overrides: SYSTEM_PATTERN (default *<ARM>*.pt), LAMS, QOS, TIME1/TIME2, CPUS, SWEEP_DIR, OUT

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"

if [ -z "${ARM}" ] || [ -z "${RIDGE}" ]; then
    echo "Set ARM (label part of the system file names) and RIDGE (the arm's --regularization)"; exit 1
fi
REG_SYSTEM_DIR="${REG_SYSTEM_DIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system}"
SYSTEM_PATTERN="${SYSTEM_PATTERN:-*${ARM}*.pt}"
LAM_LIST=(${LAMS:-0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1 10 100})
SWEEP_DIR="${SWEEP_DIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/theta_reg_lamsweep_xseed}"
OUT="${OUT:-${HERE}/results/turb_xseed}"
QOS="${QOS-}"
TIME1="${TIME1:-08:00:00}"
TIME2="${TIME2:-02:00:00}"
CPUS="${CPUS:-24}"

shopt -s nullglob
FILES=( "${REG_SYSTEM_DIR}"/${SYSTEM_PATTERN} )
shopt -u nullglob
echo "arm ${ARM}, ridge ${RIDGE}: ${#FILES[@]} systems matching ${REG_SYSTEM_DIR}/${SYSTEM_PATTERN}"
printf '  %s\n' "${FILES[@]##*/}"
echo "lams ${LAM_LIST[*]} | sweep -> ${SWEEP_DIR} | summary -> ${OUT} | QoS ${QOS:-default}, ${CPUS} CPUs, phase 1 ${TIME1}, phase 2 ${TIME2}"
if [ ${#FILES[@]} -lt 2 ]; then echo "Error: need at least 2 seeds"; exit 1; fi
if [ -n "${DRY}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${SWEEP_DIR}" "${OUT}"

HEADER="#SBATCH -A wbg@h100
#SBATCH --partition=gpu_p6
${QOS:+#SBATCH --qos=${QOS}}
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread"
SETUP="module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
cd \"${PROJECT_DIR}\""

JOB1=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=xseed_solve_${ARM}
#SBATCH --array=0-$(( ${#FILES[@]} - 1 ))
#SBATCH --time=${TIME1}
#SBATCH --output=${OUT}/slurm_%x_%A_%a.log
${HEADER}
${SETUP}
FILES=( ${FILES[@]} )
python -u codes/resolve_theta_reg.py "\${FILES[\${SLURM_ARRAY_TASK_ID}]}" --lams ${LAM_LIST[@]} \
    --ridge ${RIDGE} --schedule uniform --mask auto --diagnose 0 --outdir "${SWEEP_DIR}"
EOT
)
if [ -z "${JOB1}" ]; then echo "Error: phase-1 sbatch returned no job id"; exit 1; fi

JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} <<EOT
#!/bin/bash
#SBATCH --job-name=xseed_score_${ARM}
#SBATCH --time=${TIME2}
#SBATCH --output=${OUT}/slurm_%x_%j.log
${HEADER}
${SETUP}
python -u lam_selection_criterion/turb_xseed.py --systems ${FILES[@]} --sweep_dir "${SWEEP_DIR}" \
    --lams ${LAM_LIST[@]} --ridge ${RIDGE} --arm ${ARM} --out "${OUT}"
EOT
)
if [ -z "${JOB2}" ]; then echo "Error: phase-2 sbatch returned no job id (phase 1 ${JOB1} is submitted)"; exit 1; fi
echo "Submitted phase 1 ${JOB1} (array of ${#FILES[@]}) and phase 2 ${JOB2} (after phase 1 succeeds)"
