#!/bin/bash
# Scat4 Mod2 forward, kept-triples vs dense (grad_speed/check_forward_cost.py, 2026-11-18): wall time and peak GPU
# memory, 1D (batch 5000) and 2D (batches 10, 20). One short dev GPU job, billed 1 GPU x elapsed. H100: the OLD 2D
# forward builds (B, 21, 21, 21, 128, 128) complex intermediates (~12 GB at B 10, ~24 GB at B 20).
#
# Usage (Jean Zay, repo root):  GPU=h100 bash grad_speed/launch_check_forward_cost.sh
# overrides: GPU (v100|h100, mandatory), OUT (grad_speed/results_forward_cost), B1D (5000), B2D ("10 20"), REPS (10),
#            EXTRA (extra args, e.g. "--skip_2d"), TIME (00:15:00), CPUS (24 on h100, 10 on v100)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
: "${GPU:?set GPU=v100 or GPU=h100}"
OUT="${OUT:-${PROJECT_DIR}/grad_speed/results_forward_cost}"
B1D="${B1D:-5000}"
B2D="${B2D:-10 20}"
REPS="${REPS:-10}"
EXTRA="${EXTRA:-}"
TIME="${TIME:-00:15:00}"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE=""; CPUS="${CPUS:-10}" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100"; CPUS="${CPUS:-24}" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"
echo "check_forward_cost.py: B1D ${B1D}, B2D ${B2D}, REPS ${REPS} ${EXTRA}, ${GPU} dev (${QOS}), ${TIME}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=check_fwd_cost
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
python -u grad_speed/check_forward_cost.py --B1d ${B1D} --B2d ${B2D} --reps ${REPS} ${EXTRA}
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
