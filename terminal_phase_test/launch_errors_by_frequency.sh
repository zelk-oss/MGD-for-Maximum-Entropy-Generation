#!/bin/bash
# terminal_phase_test/errors_by_frequency.py as a GPU dev job (CPU work; user rule: never prepost), 2026-11-25.
# Needs the csv of launch_terminal_errors_1118.sh for every (label, seed) in ERRORS_DIR.
# Usage (Jean Zay, repo root): GPU=h100 LABELS="L1 L2 ..." ERRORS_DIR=<dir> OUT=<dir> \
#   bash terminal_phase_test/launch_errors_by_frequency.sh      overrides: SEEDS ("900 901 902"), TIME (00:15:00), CPUS (24)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
: "${GPU:?set GPU=v100 or GPU=h100}"; : "${LABELS:?}"; : "${ERRORS_DIR:?}"; : "${OUT:?}"
SEEDS="${SEEDS:-900 901 902}"
TIME="${TIME:-00:15:00}"
CPUS="${CPUS:-24}"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE="" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=errfreq
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${LOGS}/slurm_%x_%j.log

module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
export OMP_NUM_THREADS=${CPUS}
cd "${PROJECT_DIR}"
python -u terminal_phase_test/errors_by_frequency.py --labels ${LABELS} --seeds ${SEEDS} --errors_dir "${ERRORS_DIR}" --out "${OUT}"
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
