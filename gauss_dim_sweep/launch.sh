#!/bin/bash
# Gaussian dimension sweep (README.md): MGD + Theta_reg on white Gaussian data, M = 4, 16, 64, 4 seeds.
#   phase 1: array, one task per seed; each task runs run_task.sh, i.e. every M in turn (SDE, then
#            Theta_reg for every lam from the saved system, a few MB at r <= 6).
#   phase 2: one task after phase 1 succeeded (afterok): finish.sh = cross-seed lam choice per M +
#            analyse.py -> results/summary.json and the PDFs.
# Dev QoS by default: 4 + 1 jobs, under its cap of 10 submitted jobs (array tasks count).
# Cost NOT measured at full size (nt 10000, n1 5000, r <= 6): first time one seed (SEEDS="900").
#
# Usage (Jean Zay, from the repo root):
#   DRY=1 bash gauss_dim_sweep/launch.sh                 # print the plan, submit nothing
#   SEEDS="900" bash gauss_dim_sweep/launch.sh           # timing run (phase 2 is skipped: < 2 seeds)
#   bash gauss_dim_sweep/launch.sh                       # seeds 900-903
# overrides: SEEDS, MS, N1, NT, NBULK, LAMS, OUT, QOS ("" = default t3 queue), TIME1, TIME2, CPUS

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"

SEEDS=(${SEEDS:-900 901 902 903})
export MS="${MS:-4 16 64}" N1="${N1:-5000}" NT="${NT:-10000}" NBULK="${NBULK:-2000}"
export LAMS="${LAMS:-1e-4 1e-3 1e-2}" OUT="${OUT:-${HERE}}"
QOS="${QOS-qos_gpu_h100-dev}"
TIME1="${TIME1:-02:00:00}"
TIME2="${TIME2:-00:30:00}"
CPUS="${CPUS:-8}"
LOGS="${OUT}/results/logs"

echo "seeds ${SEEDS[*]} | M ${MS} | n1 ${N1}, nt ${NT} (n_bulk ${NBULK}) | lams ${LAMS} | -> ${OUT}"
echo "QoS ${QOS:-default}, ${CPUS} CPUs, phase 1 ${TIME1} per seed, phase 2 ${TIME2} | logs ${LOGS}"
if [ -n "${DRY}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${LOGS}"

HEADER="#SBATCH -A wbg@h100
#SBATCH --partition=gpu_p6
${QOS:+#SBATCH --qos=${QOS}}
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread"
SETUP="module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
cd \"${PROJECT_DIR}\"
export MS=\"${MS}\" N1=\"${N1}\" NT=\"${NT}\" NBULK=\"${NBULK}\" LAMS=\"${LAMS}\" OUT=\"${OUT}\""

JOB1=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=gauss_dim
#SBATCH --array=0-$(( ${#SEEDS[@]} - 1 ))
#SBATCH --time=${TIME1}
#SBATCH --output=${LOGS}/slurm_%x_%A_%a.log
${HEADER}
${SETUP}
SEEDS=( ${SEEDS[@]} )
bash gauss_dim_sweep/run_task.sh "\${SEEDS[\${SLURM_ARRAY_TASK_ID}]}"
EOT
)
if [ -z "${JOB1}" ]; then echo "Error: phase-1 sbatch returned no job id"; exit 1; fi
if [ ${#SEEDS[@]} -lt 2 ]; then
    echo "Submitted phase 1 ${JOB1} (1 seed: no phase 2). Timing: grep 'SDE integration finished' ${LOGS}/slurm_gauss_dim_${JOB1}_0.log"
    exit 0
fi

JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} <<EOT
#!/bin/bash
#SBATCH --job-name=gauss_dim_finish
#SBATCH --time=${TIME2}
#SBATCH --output=${LOGS}/slurm_%x_%j.log
${HEADER}
${SETUP}
bash gauss_dim_sweep/finish.sh
EOT
)
if [ -z "${JOB2}" ]; then echo "Error: phase-2 sbatch returned no job id (phase 1 ${JOB1} is submitted)"; exit 1; fi
echo "Submitted phase 1 ${JOB1} (array of ${#SEEDS[@]}) and phase 2 ${JOB2} (after phase 1 succeeds)"
