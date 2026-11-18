#!/bin/bash
# 2D port of the 1D speed-ups vs the frozen pre-port code (grad_speed/check_2d_port.py, 2026-11-17): equality check
# (B 2, float32 + float64) then timing old vs new (float32, B_TIME 10 = 2D run batch). One short dev GPU job, billed
# 1 GPU x elapsed. H100: the OLD Scat4 code builds (B, 21, 21, 21, 128, 128) complex intermediates (~12 GB at B 10).
#
# Usage (Jean Zay, repo root):  GPU=h100 bash grad_speed/launch_check_2d_port.sh
# overrides: GPU (v100|h100, mandatory), OUT (grad_speed/results), M (128), J (5), L (4), B (2), B_TIME (10),
#            TIME (00:15:00), CPUS (24 on h100, 10 on v100)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
: "${GPU:?set GPU=v100 or GPU=h100}"
OUT="${OUT:-${PROJECT_DIR}/grad_speed/results}"
M="${M:-128}"
J="${J:-5}"
L="${L:-4}"
B="${B:-2}"
B_TIME="${B_TIME:-10}"
TIME="${TIME:-00:15:00}"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE=""; CPUS="${CPUS:-10}" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100"; CPUS="${CPUS:-24}" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"
echo "check_2d_port.py: M ${M}, J ${J}, L ${L}, B ${B}, ${GPU} dev (${QOS}), ${TIME}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=check_2d_port
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
python -u grad_speed/check_2d_port.py --M ${M} --J ${J} --L ${L} --B ${B} --B_time ${B_TIME}
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
