#!/bin/bash
# Known-truth lam test (README.md in this folder): one array task per seed, CPU work on the
# H100 allocation (as launch/resolve_lamsweep.sh: gpu_p6 jobs must take a GPU, the code never
# uses it). Per-seed cost NOT measured at full size (smoke only): rough estimate 30-60 min
# (SDE 60000 steps at n = 8500, d = 8, then ~70 Thomas solves of 60000 nodes, r = 36).
# Writes ${OUT}/seed_<s>.pt (~6 MB each) + slurm logs; the ~0.6 GB system file per seed goes to
# ${SYSTEM_DIR} on $SCRATCH and is deleted after use.
# More seeds later: launch again with the same OUT and new SEEDS; the combined table pools every
# seed in OUT. Other settings (EXTRA="--rho 0.8", ...): use a new OUT (the script refuses to mix).
# When all tasks are done, on a login node:
#   python lam_selection_criterion/gauss_d8_lamtest.py --out ${OUT} --combine_only
#
# Usage (from anywhere): bash lam_selection_criterion/launch_gauss_d8.sh
#   overrides: SEEDS="900 901" OUT=... QOS="" (default t3 queue) TIME=04:00:00 CPUS=8 EXTRA="--rho 0.8"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"

SEEDS=(${SEEDS:-900 901 902 903 904})
OUT="${OUT:-${HERE}/results/gauss_d8}"
SYSTEM_DIR="${SYSTEM_DIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/lam_selection_criterion/systems}"
QOS="${QOS-qos_gpu_h100-dev}"
TIME="${TIME:-02:00:00}"
CPUS="${CPUS:-8}"
EXTRA="${EXTRA:-}"

mkdir -p "${OUT}" "${SYSTEM_DIR}"
echo "seeds ${SEEDS[*]} -> ${OUT} | QoS ${QOS:-default}, time ${TIME}, ${CPUS} CPUs ${EXTRA:+| extra: ${EXTRA}}"

JOBID=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=gauss_d8_lamtest
#SBATCH -A wbg@h100
#SBATCH --array=0-$(( ${#SEEDS[@]} - 1 ))
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
SEEDS=( ${SEEDS[@]} )
python lam_selection_criterion/gauss_d8_lamtest.py --out "${OUT}" --system_dir "${SYSTEM_DIR}" \
    --seeds "\${SEEDS[\${SLURM_ARRAY_TASK_ID}]}" --no_combine ${EXTRA}
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted array job ${JOBID}; when done: python lam_selection_criterion/gauss_d8_lamtest.py --out ${OUT} --combine_only"
