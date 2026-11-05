#!/bin/bash
# moment_reproduction as a SLURM array on H100 (default; GPU=v100 for V100), default QoS: one task per model of
# run_moment_reproduction.py.
# Results: moment_reproduction/results/<group>/<model>.pkl (load in the notebook with load_results()).
# Logs: moment_reproduction/results/logs/.
#
# Usage (repo root on Jean Zay):
#   bash moment_reproduction/launch_moment_reproduction.sh              # every model in GROUPS
#   TASKS=6-9 bash moment_reproduction/launch_moment_reproduction.sh    # some (indices from --list)
#   TIME=04:00:00 bash ...                                              # longer, if 2 h is too short
#   QOS=qos_gpu_h100-dev TASKS=0-4 bash ...                             # dev QoS (max 10 submitted jobs per user)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
N=$(python3 "${HERE}/run_moment_reproduction.py" --list | wc -l)     # --list imports nothing heavy
TASKS="${TASKS:-0-$(( N - 1 ))}"
QOS="${QOS-}"            # default QoS (2026-11-02: dev QoS allows 10 submitted jobs per user, usually taken)
GPU="${GPU:-h100}"       # GPU=v100: gpu_p13, 10 CPUs (V100 share per GPU), 4 h (V100 slower, runtime not measured)
case "${GPU}" in
    h100) ACCOUNT="wbg@h100"; PARTITION="gpu_p6";  CONSTRAINT="h100"; MODULE_PRE="arch/h100"
          TIME="${TIME:-02:00:00}"; CPUS="${CPUS:-24}" ;;
    v100) ACCOUNT="wbg@v100"; PARTITION="gpu_p13"; CONSTRAINT="v100"; MODULE_PRE=""
          TIME="${TIME:-04:00:00}"; CPUS="${CPUS:-10}" ;;
    *) echo "GPU must be h100 or v100"; exit 1 ;;
esac
LOG_DIR="${HERE}/results/logs"
mkdir -p "${LOG_DIR}"

echo "tasks ${TASKS} of ${N}; ${GPU}, QoS ${QOS:-default}, ${TIME}, ${CPUS} CPUs"
JOB=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=moment_repro
#SBATCH --array=${TASKS}%10
#SBATCH -A ${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH -C ${CONSTRAINT}
${QOS:+#SBATCH --qos=${QOS}}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/slurm_%x_%A_%a.log
module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
python -u moment_reproduction/run_moment_reproduction.py --task \${SLURM_ARRAY_TASK_ID}
EOT
)
if [ -z "${JOB}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOB}"
