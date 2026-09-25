#!/bin/bash
# Shared SLURM submission logic for the turbulence MGD sweeps.
#
# Not run directly: each experiment script in this folder (one per potential
# set, e.g. lamtune_full.sh) sets the experiment's parameters and then sources
# this file, which submits one SLURM array job with one task per seed. Keeping
# the parameters in per-experiment scripts (tracked in git, pushed with the
# code) and the mechanics here means every launched experiment has a file that
# says exactly what it was, and a fix to the mechanics lands in all of them.
#
# Required variables (set by the sourcing script):
#   EXP_NAME        short name, used for the SLURM job name and the label
#   TERMS           bash array of potential terms
#   SEED_LIST       bash array of seeds (one array task each)
#   N1 NT J Q SIGMA LAM N_SUBSAMPLE SUBSERIES_LEN TARGET_LEN
#   TIME            SLURM wall time HH:MM:SS; also sets --time_limit_min, so the
#                   SDE loop aborts cleanly if projected past 90% of it
# Optional (defaults below): LABEL, REG_SOLVER, REG_RIDGE, SAVE_REG_SYSTEM,
#   SOLVE_FLOAT64, DEDUPLICATE_FILTERS,
#   SKIP_REG_SOLVE, REG_SYSTEM_DIR, REGULARIZATION, SCHEDULE_EXPONENT,
#   INTERPOLANT, BATCH_SIZE, ACCOUNT, CONSTRAINT, PARTITION, CPUS, NGPUS, MODULE,
#   SCHEDULE_ARGS (extra time-grid flags passed verbatim, e.g.
#   "--schedule two_phase --n_bulk 10000" -- see codes/time_schedules.py; empty =
#   legacy power schedule), PROFILE (true: run under cProfile with CUDA_LAUNCH_BLOCKING=1 so GPU time is
#   charged to the Python call that launched it; stats written next to the SLURM log
#   as profile_<exp>_<jobid>_<task>.prof -- timings are for attribution, not speed)

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "_submit_turb_sweep.sh is sourced by the experiment scripts in this folder;"
    echo "run one of those instead (e.g. ./lamtune_full.sh)."
    exit 1
fi

for v in EXP_NAME N1 NT J Q SIGMA LAM N_SUBSAMPLE SUBSERIES_LEN TARGET_LEN TIME; do
    if [ -z "${!v}" ]; then echo "Error: ${v} is not set"; exit 1; fi
done
if [ ${#TERMS[@]} -eq 0 ] || [ ${#SEED_LIST[@]} -eq 0 ]; then
    echo "Error: TERMS and SEED_LIST must be non-empty arrays"; exit 1
fi

# ── defaults ─────────────────────────────────────────
LABEL="${LABEL-${EXP_NAME}}"
REG_SOLVER="${REG_SOLVER:-thomas}"
REG_RIDGE="${REG_RIDGE:-0.0}"
SAVE_REG_SYSTEM="${SAVE_REG_SYSTEM:-false}"
SKIP_REG_SOLVE="${SKIP_REG_SOLVE:-false}"
SOLVE_FLOAT64="${SOLVE_FLOAT64:-false}"
DEDUPLICATE_FILTERS="${DEDUPLICATE_FILTERS:-false}"
REGULARIZATION="${REGULARIZATION:-0.01}"
SCHEDULE_EXPONENT="${SCHEDULE_EXPONENT:-2}"
INTERPOLANT="${INTERPOLANT:-Cos}"
ACCOUNT="${ACCOUNT:-wbg@v100}"
CONSTRAINT="${CONSTRAINT:-v100}"
PARTITION="${PARTITION:-gpu_p13}"
CPUS="${CPUS:-10}"
NGPUS="${NGPUS:-1}"
MODULE="${MODULE:-pytorch-gpu/py3/2.8.0}"
MODULE_PRE="${MODULE_PRE:-}"          # loaded before MODULE, e.g. arch/h100 on gpu_p6

# ── paths ────────────────────────────────────────────
LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TURB_DIR="$(cd "${LAUNCH_DIR}/.." && pwd)"
PYTHON_SCRIPT="${TURB_DIR}/run_SDE.py"
OUTDIR="${TURB_DIR}"
LOG_DIR="${OUTDIR}/saved_results/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M")
IFS=: read -r _h _m _s <<< "${TIME}"
TIME_LIMIT_MIN=$(( 10#${_h} * 60 + 10#${_m} ))

# ── optional flags ───────────────────────────────────
EXTRA_FLAGS=""
[ -n "${LABEL}" ]      && EXTRA_FLAGS+=" --label ${LABEL}"
[ -n "${BATCH_SIZE}" ] && EXTRA_FLAGS+=" --batch_size ${BATCH_SIZE}"
[ "${SOLVE_FLOAT64}" = "true" ]       && EXTRA_FLAGS+=" --solve_float64"
[ "${DEDUPLICATE_FILTERS}" = "true" ] && EXTRA_FLAGS+=" --deduplicate_filters"
[ -n "${SCHEDULE_ARGS}" ] && EXTRA_FLAGS+=" ${SCHEDULE_ARGS}"
if [ "${SAVE_REG_SYSTEM}" = "true" ]; then
    EXTRA_FLAGS+=" --save_reg_system"
    [ -n "${REG_SYSTEM_DIR}" ] && EXTRA_FLAGS+=" --reg_system_dir ${REG_SYSTEM_DIR}"
fi
if [ "${SKIP_REG_SOLVE}" = "true" ]; then
    if [ "${SAVE_REG_SYSTEM}" != "true" ]; then
        echo "Error: SKIP_REG_SOLVE=true needs SAVE_REG_SYSTEM=true"; exit 1
    fi
    EXTRA_FLAGS+=" --skip_reg_solve"
fi

PY_CMD="python"
PROFILE_ENV=""
if [ "${PROFILE:-false}" = "true" ]; then
    PY_CMD="python -m cProfile -o ${LOG_DIR}/profile_${EXP_NAME}_\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}.prof"
    PROFILE_ENV="export CUDA_LAUNCH_BLOCKING=1"
fi

NTASKS=${#SEED_LIST[@]}
TERMS_STR="${TERMS[*]}"

echo "Submitting ${EXP_NAME}: ${NTASKS} seeds (${SEED_LIST[*]}), n1=${N1}, nt=${NT}, lam=${LAM}"
echo "  terms: ${TERMS_STR}"
echo "  time ${TIME} (time_limit_min ${TIME_LIMIT_MIN}), ${PARTITION}/${CONSTRAINT}, ${CPUS} cpus"
echo "  flags:${EXTRA_FLAGS}"

# ── submit ───────────────────────────────────────────
JOBID=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=${EXP_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH --array=0-$((NTASKS-1))
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${NGPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/slurm_%x_%A_%a.log

module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load ${MODULE}
cd "${TURB_DIR}"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
# fewer fragmentation OOMs (reserved-but-unallocated blocks become reusable)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
${PROFILE_ENV}

SEED_LIST=(${SEED_LIST[@]})
SEED_VAL=\${SEED_LIST[\${SLURM_ARRAY_TASK_ID}]}

srun ${PY_CMD} "${PYTHON_SCRIPT}" \
    --seed \${SEED_VAL} \
    --timestamp "${TIMESTAMP}" \
    --n1 ${N1} \
    --subseries_len ${SUBSERIES_LEN} \
    --target_len ${TARGET_LEN} \
    --nt ${NT} \
    --J ${J} \
    --Q ${Q} \
    --sigma ${SIGMA} \
    --interpolant ${INTERPOLANT} \
    --schedule_exponent ${SCHEDULE_EXPONENT} \
    --regularization ${REGULARIZATION} \
    --lam ${LAM} \
    --n_subsample ${N_SUBSAMPLE} \
    --reg_solver ${REG_SOLVER} \
    --reg_ridge ${REG_RIDGE} \
    --time_limit_min ${TIME_LIMIT_MIN} \
    --outdir "${OUTDIR}" \
    --terms ${TERMS_STR} \
    ${EXTRA_FLAGS}
EOT
)

if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted array job ${JOBID}"

# Ledger of every submission (on the cluster, next to the SLURM logs).
LEDGER="${LOG_DIR}/submissions.tsv"
[ -f "${LEDGER}" ] || printf 'date\tjob_id\texp_name\tscript\ttimestamp\tseeds\tn1\tnt\tlam\tterms\n' > "${LEDGER}"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Iseconds)" "${JOBID}" "${EXP_NAME}" "$(basename "$0")" "${TIMESTAMP}" \
    "${SEED_LIST[*]}" "${N1}" "${NT}" "${LAM}" "${TERMS_STR}" >> "${LEDGER}"
