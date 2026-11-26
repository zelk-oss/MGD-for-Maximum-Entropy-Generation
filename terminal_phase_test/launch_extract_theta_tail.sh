#!/bin/bash
# Extract for explore_theta_tail_m1024.ipynb (terminal_phase_test/extract_theta_tail.py), 2026-11-25.
# CPU work as a GPU dev job (user rule: never prepost; billed 1 GPU x elapsed).
# Usage (Jean Zay, repo root): GPU=h100 CONFIGS="C900 C901 C902" REG_DIR=<system folder> NAMES=<csv> OUT=<file.npz> \
#   bash terminal_phase_test/launch_extract_theta_tail.sh        overrides: LAM (1e-4), TIME (00:15:00), CPUS (24)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
: "${GPU:?set GPU=v100 or GPU=h100}"; : "${CONFIGS:?}"; : "${REG_DIR:?}"; : "${NAMES:?}"; : "${OUT:?}"
LAM="${LAM:-1e-4}"
TIME="${TIME:-00:15:00}"
CPUS="${CPUS:-24}"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE="" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
LOGS="$(dirname "${OUT}")/logs"
mkdir -p "${LOGS}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=thtail
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
python -u terminal_phase_test/extract_theta_tail.py --configs ${CONFIGS} --reg_dir "${REG_DIR}" \
    --theta_reg_dir terminal_phase_test/theta_reg_insys_m1024 --lam ${LAM} --names "${NAMES}" --out "${OUT}"
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
