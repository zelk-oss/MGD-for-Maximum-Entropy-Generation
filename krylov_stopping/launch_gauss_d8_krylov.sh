#!/bin/bash
# Ridge vs early-stopped Krylov, Gaussian d = 8 (gauss_d8_krylov.py). Grid (N, sigma2, seed) =
# 2 x 2 x 5 = 20 points, run as 10 array tasks: qos_gpu_h100-dev allows at most 10 SUBMITTED jobs
# per user (MaxSubmitPU, counts running + pending, array tasks included), so other dev jobs of yours
# still queued or running make this submission fail too. Task k runs points k and k + 10, i.e. one
# N = 1e3 and one N = 1e4 point, each with the 9 methods one after the other.
# CPU work on the H100 allocation (as launch_gauss_d8.sh: gpu_p6 jobs must take a GPU, the code
# never uses it; wall time per step is measured on CPU with CPUS threads).
# Cost NOT measured at full size (smoke only: N=200, nt=100 ran at ~10 ms/step). Rough guess per
# task: 10-30 min for the N = 1e4 point (9 x 1000 steps) + a few minutes for the N = 1e3 point.
# Writes ${OUT}/runs/<config>_<method>_seed<s>.pt (~0.2 MB each) + slurm logs. Finished runs are
# skipped, so a resubmit with the same OUT only redoes what is missing.
# When all tasks are done, on a login node:
#   python krylov_stopping/gauss_d8_krylov.py --out ${OUT} --combine_only
#
# Usage (from anywhere): bash krylov_stopping/launch_gauss_d8_krylov.sh
#   overrides: OUT=... QOS="" (default t3 queue) TIME=02:00:00 CPUS=8 EXTRA="--methods direct ito_cg"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"

OUT="${OUT:-${HERE}/results/gauss_d8_krylov}"
QOS="${QOS-qos_gpu_h100-dev}"
TIME="${TIME:-02:00:00}"
CPUS="${CPUS:-8}"
EXTRA="${EXTRA:-}"
N_TASKS=10                      # <= 10: dev QoS MaxSubmitPU; covers the 20 default grid points

mkdir -p "${OUT}"
echo "${N_TASKS} tasks -> ${OUT} | QoS ${QOS:-default}, time ${TIME}, ${CPUS} CPUs ${EXTRA:+| extra: ${EXTRA}}"

JOBID=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=gauss_d8_krylov
#SBATCH -A wbg@h100
#SBATCH --array=0-$(( N_TASKS - 1 ))
#SBATCH --partition=gpu_p6
${QOS:+#SBATCH --qos=${QOS}}
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${OUT}/slurm_%x_%A_%a.log

module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
python krylov_stopping/gauss_d8_krylov.py --out "${OUT}" --task "\${SLURM_ARRAY_TASK_ID}" --n_tasks ${N_TASKS} --no_combine ${EXTRA}
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted array job ${JOBID}; when done: python krylov_stopping/gauss_d8_krylov.py --out ${OUT} --combine_only"
