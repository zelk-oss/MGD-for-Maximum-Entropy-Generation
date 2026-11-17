#!/bin/bash
# Scat4 Mod2 full-Jacobian gather vs the old dense code (grad_speed/check_scat4_gather.py, 2026-11-17).
# Tiny sizes (B 16 for the comparison, B 500 for the timing): one short dev GPU job, billed 1 GPU x elapsed.
#
# Usage (Jean Zay, repo root):  GPU=v100 bash grad_speed/launch_check_scat4_gather.sh
# overrides: GPU (v100|h100, mandatory), OUT (grad_speed/results), M (256), J (8), B (16), B_TIME (500),
#            TIME (00:15:00), CPUS (10)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
: "${GPU:?set GPU=v100 or GPU=h100}"
OUT="${OUT:-${PROJECT_DIR}/grad_speed/results}"
M="${M:-256}"
J="${J:-8}"
B="${B:-16}"
B_TIME="${B_TIME:-500}"
TIME="${TIME:-00:15:00}"
CPUS="${CPUS:-10}"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE="" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"
echo "scat4 gather check: M ${M}, J ${J}, B ${B}, B_time ${B_TIME}, ${GPU} dev (${QOS}), ${TIME}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=scat4_gather
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
python -u grad_speed/check_scat4_gather.py --M ${M} --J ${J} --B ${B} --B_time ${B_TIME}
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
