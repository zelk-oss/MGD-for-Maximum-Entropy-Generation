#!/bin/bash
# (a) of the whitening test: spectral range kappa of each data set (whitening_test/kappa_spectra.py).
# CPU-only work (FFTs of <= 10^4 signals), run as a short dev GPU job because prepost queues for hours (11-11).
# Billed 1 GPU x elapsed (expected a few minutes).
#
# Usage (Jean Zay, repo root):  GPU=v100 bash whitening_test/launch_kappa.sh
# overrides: GPU (v100|h100, mandatory), OUT (whitening_test/kappa), N1 (10000), TIME (00:20:00), CPUS (10)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
: "${GPU:?set GPU=v100 or GPU=h100}"
OUT="${OUT:-${PROJECT_DIR}/whitening_test/kappa}"
N1="${N1:-10000}"
TIME="${TIME:-00:20:00}"
CPUS="${CPUS:-10}"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE="" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"
echo "kappa spectra: n1 ${N1}, out ${OUT}, ${GPU} dev (${QOS}), ${CPUS} CPUs, ${TIME}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=kappa
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
python -u whitening_test/kappa_spectra.py --out ${OUT} --n1 ${N1}
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
