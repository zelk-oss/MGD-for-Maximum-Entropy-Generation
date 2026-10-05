#!/bin/bash
# Bimodal known-truth lam test (bimodal_lamtest.py; README.md in this folder): one array task per
# run index k, which does run k for every beta in turn (so K tasks: dev QoS allows at most 10
# submitted jobs per user, each array task counting as one -- sacctmgr, 2026-10-05).
# H100 dev QoS, 2 h per task. The SDE runs on the GPU; the lam rules on the CPU (thomas_batched).
# Measured (job 619951, one seed per dataset): ~7 min per MGD run (SDE 6.2 min, rules ~50 s).
# Since 2026-10-05 every dataset is run with 3 SDE seeds (cross-seed CV, --seeds_per_data), so a
# task = 3 betas x 3 seeds ~ 63 min.
# Writes ${OUT}/beta_<b>/run_<k>_s<s>.pt (~1-2 MB each) + slurm logs; the regularised systems are
# KEPT in ${SYSTEM_DIR} on $SCRATCH (~8 MB each, 90 in all), so new lam rules can be tested without
# rerunning the SDE. The replicas of a dataset are saved together when its last seed is done; a
# task killed at its time limit keeps the betas it finished; launching again skips saved datasets.
# More runs/betas later: same OUT (the script refuses other settings: new OUT for those).
# When all tasks are done, on a login node:
#   python lam_selection_criterion/bimodal_lamtest.py --out ${OUT} --combine_only
#
# Usage (from anywhere): bash lam_selection_criterion/launch_bimodal.sh
#   overrides: BETAS="0.1 0.9 1.9" K=10 RUN0=0 OUT=... QOS="" (default t3 queue) TIME=02:00:00 CPUS=8 EXTRA="--sigma 2"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"

BETAS=(${BETAS:-0.1 0.9 1.9})
K="${K:-10}"                       # runs per beta: k = RUN0 .. RUN0+K-1 (dataset and SDE seed k) = array tasks
RUN0="${RUN0:-0}"
OUT="${OUT:-${HERE}/results/bimodal_xseed}"
SYSTEM_DIR="${SYSTEM_DIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/lam_selection_criterion/systems_bimodal_xseed}"
QOS="${QOS-qos_gpu_h100-dev}"
TIME="${TIME:-02:00:00}"
CPUS="${CPUS:-8}"
EXTRA="${EXTRA:-}"

mkdir -p "${OUT}" "${SYSTEM_DIR}"
echo "${K} tasks (runs ${RUN0}..$(( RUN0 + K - 1 ))), each doing betas ${BETAS[*]} in turn -> ${OUT} | QoS ${QOS:-default}, time ${TIME}, ${CPUS} CPUs ${EXTRA:+| extra: ${EXTRA}}"

JOBID=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=bimodal_lamtest
#SBATCH -A wbg@h100
#SBATCH --array=0-$(( K - 1 ))
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
RUN=\$(( ${RUN0} + SLURM_ARRAY_TASK_ID ))
for BETA in ${BETAS[@]}; do
    python -u lam_selection_criterion/bimodal_lamtest.py --out "${OUT}" --system_dir "${SYSTEM_DIR}" \
        --beta "\${BETA}" --runs "\${RUN}" --no_combine --keep_system ${EXTRA}
done
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted array job ${JOBID}; when done: python lam_selection_criterion/bimodal_lamtest.py --out ${OUT} --combine_only"
