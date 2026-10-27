#!/bin/bash
# Offline Theta_reg (unconstrained, lam 5e-4, ridge 1e-4) of the FINISHED seeds of one run label, then
# nested_models/analyse_nested.py on them, chained (afterok) in one command. Run from anywhere on Jean Zay.
#
#   bash nested_models/launch_analysis.sh turbulence_nested_M2                  # seeds 900-904: what is not finished is skipped
#   SEEDS="900 901 902" bash nested_models/launch_analysis.sh turbulence_nested_M2
#   NO_RESOLVE=1 bash nested_models/launch_analysis.sh turbulence_final_gauss_scat2   # M1: its Theta_reg already exists
#
# Safe to repeat while the runs finish: resolve_lamsweep.sh skips (system, lam) pairs already solved and
# analyse_nested.py keeps the .npz it already wrote (FORCE=1 to redo).
# resolve_lamsweep.sh sizes its array by the system files present when it is called, so call this again
# after more seeds have finished. Dev QoS allows 10 submitted tasks at a time (array tasks included).
LABEL="${1:?run label required, e.g. turbulence_nested_M2}"
SEEDS="${SEEDS:-900 901 902 903 904}"
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PROJECT_DIR="$PWD"
LOG_DIR="${PROJECT_DIR}/turbulence/saved_results/logs"
mkdir -p "${LOG_DIR}"

DEP=""
if [ -z "${NO_RESOLVE}" ]; then
    OUT=$(SYSTEM_PATTERN="${LABEL}/*.pt" LAMS=5e-4 RIDGE=1e-4 bash turbulence/launch/resolve_lamsweep.sh 2>&1)
    echo "${OUT}" | tail -6
    RID=$(echo "${OUT}" | sed -n 's/^Submitted array job \([0-9]*\).*/\1/p' | tail -1)
    if [ -z "${RID}" ]; then echo "No Theta_reg job submitted (no finished system of ${LABEL} yet?): nothing to analyse."; exit 1; fi
    DEP="--dependency=afterok:${RID}"
fi

AID=$(sbatch --parsable ${DEP} <<EOT
#!/bin/bash
#SBATCH --job-name=nested_analyse
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
python nested_models/analyse_nested.py --labels ${LABEL} --seeds ${SEEDS} ${FORCE:+--force}
EOT
)
if [ -z "${AID}" ]; then echo "Error: sbatch returned no job id for the analysis"; exit 1; fi
echo "Queued analysis ${AID}${RID:+ after Theta_reg ${RID}}  (log: ${LOG_DIR}/slurm_nested_analyse_${AID}.log)"
