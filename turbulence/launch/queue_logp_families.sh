#!/bin/bash
# Offline Theta_reg (L_6 < 0, lam 5e-4, ridge 1e-4, as the turbulence_final analyses) for
# one arm, then turbulence/logp_family_contrib.py on it (2026-10-22).
#
#   1. theta_sign_constraint/launch_l6neg.sh on the arm's saved reg system, chained behind
#      the SDE job (DEPEND, aftercorr) when given, else submitted at once;
#   2. a small H100 dev job (afterok on 1) that prints the per-family log p table and writes
#      turbulence/figures/logp_families_<LABEL>_p<P>.png, output in
#      turbulence/figures/data/logp_<LABEL>.txt.
# Needs the reg system of the arm on $SCRATCH (SAVE_REG_SYSTEM=true in the launchers).
#
# Usage (repo root on Jean Zay):
#   DEPEND=<SDE job id> LABEL=turbulence_ggd_mass_L6_2b_scat4 bash turbulence/launch/queue_logp_families.sh
#   LABEL=turbulence_final_coarse_noL2_scat4 bash turbulence/launch/queue_logp_families.sh   # finished run
# overrides: COLS (L_6 columns of theta, default 0:8 = M=128 J=7; M=256 J=8 is 0:9), RIDGE,
#   LAMS (list, default "5e-4" = the turbulence_final choice; e.g. LAMS="1e-4 5e-4 1e-3" solves all
#   three and prints one table per lam: with a single seed lam cannot be picked by cross-seed
#   agreement, so this shows whether the family table depends on it), SEEDS (default 900),
#   DRY=1 (l6neg dry run only)

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TURB_DIR="$(cd "${LAUNCH_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${TURB_DIR}/.." && pwd)"
LOG_DIR="${TURB_DIR}/saved_results/logs"
mkdir -p "${LOG_DIR}" "${TURB_DIR}/figures/data"

: "${LABEL:?LABEL (experiment label, e.g. turbulence_ggd_mass_L6_2b_scat4) is required}"
SEEDS="${SEEDS:-900}"
LAMS="${LAMS:-5e-4}"
RIDGE="${RIDGE:-1e-4}"
COLS="${COLS:-0:8}"
REG_DIR="${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system/${LABEL}"

OUT=$(DEPEND="${DEPEND-}" SEEDS="${SEEDS}" ARM="${LABEL}" RIDGE="${RIDGE}" LAMS="${LAMS}" COLS="${COLS}" \
      REG_SYSTEM_DIR="${REG_DIR}" DRY="${DRY-}" bash "${PROJECT_DIR}/theta_sign_constraint/launch_l6neg.sh" 2>&1)
echo "${OUT}"
[ -n "${DRY}" ] && exit 0
L6_ID=$(echo "${OUT}" | sed -n 's/^Submitted \([0-9]*\) .*/\1/p' | tail -1)
if [ -z "${L6_ID}" ]; then echo "Error: could not read the L_6<0 job id"; exit 1; fi

OUTTXT="${TURB_DIR}/figures/data/logp_${LABEL}.txt"
POST_ID=$(sbatch --parsable --dependency=afterok:${L6_ID} <<EOT
#!/bin/bash
#SBATCH --job-name=logp_${LABEL}
#SBATCH -A wbg@h100
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=${LOG_DIR}/slurm_%x_%j.log
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
: > "${OUTTXT}"
for L in ${LAMS}; do
    echo "################ lam \${L}" >> "${OUTTXT}"
    python turbulence/logp_family_contrib.py --label ${LABEL} --seed ${SEEDS%% *} --lam \${L} --ridge ${RIDGE} >> "${OUTTXT}" 2>&1
done
echo "written ${OUTTXT}"
EOT
)
if [ -z "${POST_ID}" ]; then echo "Error: sbatch returned no job id for the log p job"; exit 1; fi
echo "Queued log p families ${POST_ID} after L_6<0 job ${L6_ID} -> ${OUTTXT}"
