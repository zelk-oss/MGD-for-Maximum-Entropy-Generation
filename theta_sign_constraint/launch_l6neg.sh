#!/bin/bash
# Theta_reg with every L_6 coefficient forced negative (theta_sign_constraint/README.md).
# One array task per seed of one arm: codes/resolve_theta_reg.py --neg_cols on the seed's saved
# moment-mode system, at the arm's chosen lam (default 1e-3), the run's ridge, saved live mask.
# Bound loop = projected Newton (codes/theta_reg_bounds.py; the plain active set oscillated in job
# 781019). Per round: one Thomas solve (measured 10-05 on these systems: 250-310 s, 640-1000 s on a
# slow node) + one energy pass (not measured); rounds: 3-9 on synthetic tests, unknown on turbulence,
# capped at MAX_ITER. A non-converged result is still feasible and is written as *_NOTCONVERGED.pt.
# TIME defaults to the 20 h t3 maximum because the round count is unknown.
# RAM as launch_turb_xseed.sh (CPUS=24, ~120 GB). Output ~124 MB per (seed, lam) on $SCRATCH.
#
# Usage (from the repo root on Jean Zay), one launch per arm; first a dry run to see the files:
#   DRY=1 ARM=noL6psi_noSm30_reg1e-4 RIDGE=1e-4 bash theta_sign_constraint/launch_l6neg.sh
#   ARM=noL6psi_noSm30_reg1e-4 RIDGE=1e-4 bash theta_sign_constraint/launch_l6neg.sh
#   ARM=noL6psi_noSm30_reg1e-2 RIDGE=1e-2 bash theta_sign_constraint/launch_l6neg.sh
# overrides: LAMS (default 1e-3), EPS (eps in nats per sd of phi, default 1e-2), COLS (default 0:9 =
#   L_6 in the noSm30 runs, terms start with L_6, 9 filters), T_MIN (0 = every node), MAX_ITER (30),
#   METHOD (pnewton | exact | add_only), SYSTEM_PATTERN, QOS, TIME, CPUS, OUTDIR
#
# Chained mode (DEPEND=<SDE array job id> SEEDS="900 901 ..."): the systems don't exist yet, so
# nothing is globbed at submit time. Task k waits for task k of job DEPEND (aftercorr: same index
# = same seed, both arrays list SEEDS in the same order), then globs
# ${REG_SYSTEM_DIR}/*seed_<SEEDS[k]>_*${ARM}*.pt and needs exactly one match. If the SDE task
# fails, the dependent task is cancelled (--kill-on-invalid-dep). Used by
# turbulence/launch/turbulence_final_noScat4.sh.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"

if [ -z "${ARM}" ] || [ -z "${RIDGE}" ]; then
    echo "Set ARM (label part of the system file names) and RIDGE (the arm's --regularization)"; exit 1
fi
REG_SYSTEM_DIR="${REG_SYSTEM_DIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system}"
SYSTEM_PATTERN="${SYSTEM_PATTERN:-*${ARM}*.pt}"
DEPEND="${DEPEND-}"
LAM_LIST=(${LAMS:-1e-3})
EPS="${EPS:-1e-2}"
COLS="${COLS:-0:9}"
T_MIN="${T_MIN:-0}"
MAX_ITER="${MAX_ITER:-30}"
METHOD="${METHOD:-pnewton}"
OUTDIR="${OUTDIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/theta_reg_l6neg}"
LOGDIR="${HERE}/results"
QOS="${QOS-}"
TIME="${TIME:-20:00:00}"
# GPU=v100 (2026-11-03): gpu_p13, whole-node CPUs by default (RAM comes with CPUs; the solve is CPU/RAM-bound).
# The H100 queue was stuck at (Priority); B+Scat4 systems are 49 GB, C's 120 GB (C does not fit a V100 node).
GPU="${GPU:-h100}"
case "${GPU}" in
    h100) ACCOUNT="wbg@h100"; PARTITION="gpu_p6";  CONSTRAINT="h100"; MODULE_PRE="arch/h100"; CPUS="${CPUS:-24}" ;;
    v100) ACCOUNT="wbg@v100"; PARTITION="gpu_p13"; CONSTRAINT="v100"; MODULE_PRE="";          CPUS="${CPUS:-40}" ;;
    *) echo "GPU must be h100 or v100"; exit 1 ;;
esac

if [ -n "${DEPEND}" ]; then
    SEED_ARR=(${SEEDS-})
    if [ ${#SEED_ARR[@]} -lt 1 ]; then echo "Error: DEPEND needs SEEDS"; exit 1; fi
    echo "arm ${ARM}, ridge ${RIDGE}: chained after job ${DEPEND} (aftercorr), seeds ${SEED_ARR[*]}"
    echo "  systems resolved at run time: ${REG_SYSTEM_DIR}/*seed_<seed>_*${ARM}*.pt"
    echo "lams ${LAM_LIST[*]} | columns ${COLS} <= -${EPS} nats/sd from t >= ${T_MIN} | ${METHOD}, <= ${MAX_ITER} rounds | -> ${OUTDIR} | logs ${LOGDIR}"
    echo "${GPU}, QoS ${QOS:-default}, ${CPUS} CPUs, ${TIME}, MAX_ITER ${MAX_ITER}"
    if [ -n "${DRY}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
    mkdir -p "${OUTDIR}" "${LOGDIR}"
    JOB=$(sbatch --parsable --dependency=aftercorr:${DEPEND} --kill-on-invalid-dep=yes <<EOT
#!/bin/bash
#SBATCH --job-name=l6neg_${ARM}
#SBATCH --array=0-$(( ${#SEED_ARR[@]} - 1 ))
#SBATCH --time=${TIME}
#SBATCH --output=${LOGDIR}/slurm_%x_%A_%a.log
#SBATCH -A ${ACCOUNT}
#SBATCH --partition=${PARTITION}
${QOS:+#SBATCH --qos=${QOS}}
#SBATCH -C ${CONSTRAINT}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
SEED_ARR=( ${SEED_ARR[@]} )
SEED=\${SEED_ARR[\${SLURM_ARRAY_TASK_ID}]}
shopt -s nullglob
FILES=( "${REG_SYSTEM_DIR}"/*seed_\${SEED}_*${ARM}*.pt )
if [ \${#FILES[@]} -ne 1 ]; then
    echo "Error: seed \${SEED}: expected 1 system file, found \${#FILES[@]}:"; printf '  %s\n' "\${FILES[@]}"; exit 1
fi
echo "seed \${SEED}: \${FILES[0]}"
python -u codes/resolve_theta_reg.py "\${FILES[0]}" --lams ${LAM_LIST[@]} \
    --ridge ${RIDGE} --schedule uniform --mask auto --diagnose 0 --outdir "${OUTDIR}" \
    --neg_cols ${COLS} --neg_eps_nats ${EPS} --neg_t_min ${T_MIN} --neg_max_iter ${MAX_ITER} --neg_method ${METHOD} \
    $( [ "${MMAP:-0}" = "1" ] && echo --mmap )
EOT
)
    if [ -z "${JOB}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
    echo "Submitted ${JOB} (array of ${#SEED_ARR[@]}, after ${DEPEND})"
    exit 0
fi

shopt -s nullglob
FILES=( "${REG_SYSTEM_DIR}"/${SYSTEM_PATTERN} )
shopt -u nullglob
echo "arm ${ARM}, ridge ${RIDGE}: ${#FILES[@]} systems matching ${REG_SYSTEM_DIR}/${SYSTEM_PATTERN}"
printf '  %s\n' "${FILES[@]##*/}"
echo "lams ${LAM_LIST[*]} | columns ${COLS} <= -${EPS} nats/sd from t >= ${T_MIN} | ${METHOD}, <= ${MAX_ITER} rounds | -> ${OUTDIR} | logs ${LOGDIR}"
echo "${GPU}, QoS ${QOS:-default}, ${CPUS} CPUs, ${TIME}, MAX_ITER ${MAX_ITER}"
if [ ${#FILES[@]} -lt 1 ]; then echo "Error: no system file"; exit 1; fi
if [ -n "${DRY}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${OUTDIR}" "${LOGDIR}"

JOB=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=l6neg_${ARM}
#SBATCH --array=0-$(( ${#FILES[@]} - 1 ))
#SBATCH --time=${TIME}
#SBATCH --output=${LOGDIR}/slurm_%x_%A_%a.log
#SBATCH -A ${ACCOUNT}
#SBATCH --partition=${PARTITION}
${QOS:+#SBATCH --qos=${QOS}}
#SBATCH -C ${CONSTRAINT}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
FILES=( ${FILES[@]} )
python -u codes/resolve_theta_reg.py "\${FILES[\${SLURM_ARRAY_TASK_ID}]}" --lams ${LAM_LIST[@]} \
    --ridge ${RIDGE} --schedule uniform --mask auto --diagnose 0 --outdir "${OUTDIR}" \
    --neg_cols ${COLS} --neg_eps_nats ${EPS} --neg_t_min ${T_MIN} --neg_max_iter ${MAX_ITER} --neg_method ${METHOD} \
    $( [ "${MMAP:-0}" = "1" ] && echo --mmap )
EOT
)
if [ -z "${JOB}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOB} (array of ${#FILES[@]})"
