#!/bin/bash
# Open-loop synthesis test of Theta_reg on the bimodal benchmark (bimodal_openloop.py; 2026-11-13).
# One array task per beta: 3 closed-loop MGD runs (systems kept) + Theta(lam) + rules C0-C4 + <= 6 open-loop runs
# (check (ii) + C0-C4, duplicates skipped). Cost NOT measured: ~7 min per MGD run on H100 (job 619951) -> ~1 h per task.
# Writes ${OUT}/beta_<b>/openloop_run<k>.pt + slurm logs; systems (~8 MB each) in ${SYSTEM_DIR} on $SCRATCH.
#
# Usage (repo root on Jean Zay):
#   full:   GPU=h100 bash lam_selection_criterion/launch_bimodal_openloop.sh
#   smoke:  GPU=h100 LABEL=smoke1 NT=50 N_BULK=10 N1=200 LAMS="0 1e-3 1" BETAS="0.9" bash lam_selection_criterion/launch_bimodal_openloop.sh
#   check (i) (exact replay):  same as smoke + REPLAY=1
# overrides: BETAS RUN NT N_BULK N1 LAMS LABEL REPLAY OUT SYSTEM_DIR GPU (h100|v100) TIME CPUS

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"

BETAS=(${BETAS:-0.1 0.9 1.9})
RUN="${RUN:-0}"
NT="${NT:-50000}"
N_BULK="${N_BULK:-8334}"
N1="${N1:-1000000}"
LAMS="${LAMS:-0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1 10 100 1000}"
LABEL="${LABEL:-}"
REPLAY="${REPLAY:-0}"
OUT="${OUT:-${HERE}/results/bimodal_openloop${LABEL:+_${LABEL}}}"
SYSTEM_DIR="${SYSTEM_DIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/lam_selection_criterion/systems_bimodal_openloop${LABEL:+_${LABEL}}}"
GPU="${GPU:-h100}"
TIME="${TIME:-02:00:00}"
CPUS="${CPUS:-8}"

if [ "${GPU}" = "h100" ]; then
HEADER="#SBATCH -A wbg@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH -C h100"
MODULES="module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0"
elif [ "${GPU}" = "v100" ]; then
HEADER="#SBATCH -A wbg@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH -C v100-32g"
MODULES="module purge
module load pytorch-gpu/py3/2.8.0"
else
    echo "GPU must be h100 or v100"; exit 1
fi
REPLAY_ARG=""; [ "${REPLAY}" = "1" ] && REPLAY_ARG="--replay_check"

mkdir -p "${OUT}" "${SYSTEM_DIR}"
echo "${#BETAS[@]} tasks (betas ${BETAS[*]}), dataset ${RUN}, nt ${NT}, n1 ${N1}, lams ${LAMS} ${REPLAY_ARG} -> ${OUT} | ${GPU} dev, ${TIME}, ${CPUS} CPUs"

JOBID=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=bimodal_openloop${LABEL:+_${LABEL}}
${HEADER}
#SBATCH --array=0-$(( ${#BETAS[@]} - 1 ))
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${OUT}/slurm_%x_%A_%a.log

${MODULES}
cd "${PROJECT_DIR}"
BETAS=( ${BETAS[@]} )
BETA=\${BETAS[\${SLURM_ARRAY_TASK_ID}]}
python -u lam_selection_criterion/bimodal_openloop.py --out "${OUT}" --system_dir "${SYSTEM_DIR}" \
    --beta "\${BETA}" --run ${RUN} --nt ${NT} --n_bulk ${N_BULK} --n1 ${N1} --lams ${LAMS} ${REPLAY_ARG}
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted array job ${JOBID}"
