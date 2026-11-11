#!/bin/bash
# Per-statistic moment error at t=1 of the finished jets (Eulerian, Modane) runs (jets/moment_errors_jets.py), CPU analysis
# run as a short GPU dev job (the free prepost partition queues for hours, 11-11). Reads only saved_results/aux_moments and
# experiments/.../{config.json,fitted_potentials}. One CSV per run + a text report. Billed 1 GPU x elapsed (minutes).
#
# Usage (Jean Zay, repo root):
#   GPU=v100 bash jets/launch_moment_errors.sh                            # jets_full, seeds 500-504
#   GPU=v100 LABELS="jets_full jets_noscalar" bash jets/launch_moment_errors.sh
# overrides: GPU (v100|h100, mandatory), LABELS (jets_full), SEEDS ("500 501 502 503 504"), OUT (jets/moment_errors), TIME (00:40:00), CPUS (10)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
: "${GPU:?set GPU=v100 or GPU=h100}"
LABELS="${LABELS:-jets_full}"
SEEDS="${SEEDS:-500 501 502 503 504}"
OUT="${OUT:-${PROJECT_DIR}/jets/moment_errors}"
TIME="${TIME:-00:40:00}"
CPUS="${CPUS:-10}"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE="" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"
echo "jets moment errors: labels ${LABELS}, seeds ${SEEDS}, out ${OUT}, ${GPU} dev (${QOS}), ${CPUS} CPUs, ${TIME}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=momerr_jets
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
python -u jets/moment_errors_jets.py --labels ${LABELS} --seeds ${SEEDS} --dump ${OUT} > ${OUT}/report.txt 2>&1
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
