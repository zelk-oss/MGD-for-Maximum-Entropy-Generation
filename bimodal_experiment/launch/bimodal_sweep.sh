#!/bin/bash
# Bimodal MGD vs REG ensemble sweep over beta, then the post-hoc lam choice.
#
# 1. Runs: GPU array with one task per (beta, chunk of RUNS_PER_TASK runs), so the
#    K runs of each beta go in parallel. Every run writes its own
#    mgd/run_<k>.pt and reg_system/run_<k>.pt (~8 MB at nt=5e4); a resubmit into
#    the same SWEEP_DIR skips runs already done.
# 2. Select: one task per beta, after ALL run tasks ended (afterany, so one failed
#    chunk doesn't block the rest -- missing runs are reported in select.log):
#    bimodal_ensemble_reg.py --aggregate -> experiment_K_runs.pt, then
#    select_lambda.py -> lam_selection.pt (data-driven lam + oracle check).
#
# Layout (same as the June 2026 sweeps):
#   bimodal_experiment/results_bim_theta_beta/bim_theta_beta_sweep_<ts>/beta_<b>/
#       mgd/  reg_system/  run_<chunk>.log  experiment_K_runs.pt  lam_selection.pt  select.log
#
# Every parameter below can be overridden from the environment, e.g. a quick test:
#   K=2 RUNS_PER_TASK=1 NT=5000 BETAS="0.5000" GPU_TIME=00:30:00 ./bimodal_sweep.sh
# Step 2 alone (e.g. another LAM_LIST, or after resubmitting failed chunks):
#   SUBMIT_RUNS=false SWEEP_DIR=<existing sweep dir> ./bimodal_sweep.sh

# ── experiment ───────────────────────────────────────
K="${K:-100}"
RUNS_PER_TASK="${RUNS_PER_TASK:-10}"   # 10 -> 10 tasks per beta; 1 -> one task per run
N1="${N1:-1000000}"
NT="${NT:-50000}"
SIGMA="${SIGMA:-10}"
N_SUBSAMPLE="${N_SUBSAMPLE:-1}"
read -r -a BETA_LIST <<< "${BETAS:-0.1000 0.3000 0.5000 0.7000 0.9000 1.1000 1.3000 1.5000 1.7000 1.9000}"
read -r -a LAM_LIST <<< "${LAMS:-0 1e-8 3e-8 1e-7 3e-7 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3}"

SUBMIT_RUNS="${SUBMIT_RUNS:-true}"
SUBMIT_SELECT="${SUBMIT_SELECT:-true}"

# ── SLURM ────────────────────────────────────────────
# V100 (June: ~11.6 min per run at n1=1e6, nt=5e4 -> ~2 h for 10 runs)
GPU_ACCOUNT="${GPU_ACCOUNT:-wbg@v100}"
GPU_CONSTRAINT="${GPU_CONSTRAINT:-v100}"
GPU_PARTITION="${GPU_PARTITION:-gpu_p13}"
GPU_CPUS="${GPU_CPUS:-10}"
GPU_TIME="${GPU_TIME:-03:30:00}"      # per task: RUNS_PER_TASK runs + margin
MODULE_PRE="${MODULE_PRE:-}"          # arch/h100 when running on gpu_p6
MODULE="${MODULE:-pytorch-gpu/py3/2.8.0}"
# select step: on the same GPU account by default (no CPU allocation needed);
# SELECT_ON=cpu uses CPU_ACCOUNT/CPU_PARTITION instead (check with idr_compuse)
SELECT_ON="${SELECT_ON:-gpu}"
SELECT_TIME="${SELECT_TIME:-01:00:00}"
CPU_ACCOUNT="${CPU_ACCOUNT:-wbg@cpu}"
CPU_PARTITION="${CPU_PARTITION:-cpu_p1}"
CPU_CPUS="${CPU_CPUS:-10}"

# ── paths ────────────────────────────────────────────
LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BM_DIR="$(cd "${LAUNCH_DIR}/.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SWEEP_DIR="${SWEEP_DIR:-${BM_DIR}/results_bim_theta_beta/bim_theta_beta_sweep_${TIMESTAMP}}"
mkdir -p "${SWEEP_DIR}"
NBETA=${#BETA_LIST[@]}
NCHUNK=$(( (K + RUNS_PER_TASK - 1) / RUNS_PER_TASK ))
NTASKS=$(( NBETA * NCHUNK ))

COMMON="--K ${K} --n1 ${N1} --nt ${NT} --sigma ${SIGMA} --n_subsample ${N_SUBSAMPLE}"

echo "Sweep dir: ${SWEEP_DIR}"
echo "  K=${K} n1=${N1} nt=${NT} sigma=${SIGMA} n_subsample=${N_SUBSAMPLE}"
echo "  betas: ${BETA_LIST[*]}"
echo "  ${NTASKS} run tasks = ${NBETA} betas x ${NCHUNK} chunks of ${RUNS_PER_TASK} runs, ${GPU_TIME} each"
echo "  lams : ${LAM_LIST[*]}"

case "${SELECT_ON}" in
    gpu) SEL_SBATCH="#SBATCH -A ${GPU_ACCOUNT}
#SBATCH -C ${GPU_CONSTRAINT}
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${GPU_CPUS}" ;;
    cpu) SEL_SBATCH="#SBATCH -A ${CPU_ACCOUNT}
#SBATCH --partition=${CPU_PARTITION}
#SBATCH --cpus-per-task=${CPU_CPUS}" ;;
    *) echo "Error: SELECT_ON must be gpu or cpu"; exit 1 ;;
esac

DEPENDENCY=""
if [ "${SUBMIT_RUNS}" = "true" ]; then
    RUNS_JOB=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=bimodal_runs
#SBATCH -A ${GPU_ACCOUNT}
#SBATCH -C ${GPU_CONSTRAINT}
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --array=0-$((NTASKS-1))
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${GPU_CPUS}
#SBATCH --time=${GPU_TIME}
#SBATCH --output=${SWEEP_DIR}/slurm_runs_%A_%a.log

set -ex
module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load ${MODULE}
cd "${BM_DIR}"
BETA_LIST=(${BETA_LIST[@]})
BETA=\${BETA_LIST[\$(( SLURM_ARRAY_TASK_ID / ${NCHUNK} ))]}
CHUNK=\$(( SLURM_ARRAY_TASK_ID % ${NCHUNK} ))
OUT="${SWEEP_DIR}/beta_\${BETA}"
mkdir -p "\${OUT}"
echo "\$(date -Iseconds) SLURM_JOB_ID=\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID} chunk \${CHUNK}" >> "\${OUT}/job_info.txt"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
srun python bimodal_ensemble_reg.py ${COMMON} --beta \${BETA} \
    --run_start \$(( CHUNK * ${RUNS_PER_TASK} )) --n_runs ${RUNS_PER_TASK} \
    --outdir "\${OUT}" > "\${OUT}/run_\$(printf %02d \${CHUNK}).log" 2>&1
EOT
)
    if [ -z "${RUNS_JOB}" ]; then
        echo "Error: run submission failed (${GPU_ACCOUNT}, ${GPU_PARTITION}, -C ${GPU_CONSTRAINT})"; exit 1
    fi
    echo "Submitted run array job ${RUNS_JOB}"
    DEPENDENCY="#SBATCH --dependency=afterany:${RUNS_JOB}"
fi

if [ "${SUBMIT_SELECT}" = "true" ]; then
    SELECT_JOB=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=bimodal_select
${SEL_SBATCH}
#SBATCH --array=0-$((NBETA-1))
#SBATCH --time=${SELECT_TIME}
#SBATCH --output=${SWEEP_DIR}/slurm_select_%A_%a.log
${DEPENDENCY}

set -ex
module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load ${MODULE}
cd "${BM_DIR}"
BETA_LIST=(${BETA_LIST[@]})
BETA=\${BETA_LIST[\${SLURM_ARRAY_TASK_ID}]}
OUT="${SWEEP_DIR}/beta_\${BETA}"
{ python bimodal_ensemble_reg.py ${COMMON} --beta \${BETA} --aggregate --outdir "\${OUT}"
  python select_lambda.py "\${OUT}" --lams ${LAM_LIST[*]}; } > "\${OUT}/select.log" 2>&1
EOT
)
    if [ -z "${SELECT_JOB}" ]; then
        echo "Error: select submission failed (SELECT_ON=${SELECT_ON})"; exit 1
    fi
    echo "Submitted select array job ${SELECT_JOB}${RUNS_JOB:+ (after all of ${RUNS_JOB})}"
fi
