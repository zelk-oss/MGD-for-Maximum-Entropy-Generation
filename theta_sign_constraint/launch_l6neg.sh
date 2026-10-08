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

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"

if [ -z "${ARM}" ] || [ -z "${RIDGE}" ]; then
    echo "Set ARM (label part of the system file names) and RIDGE (the arm's --regularization)"; exit 1
fi
REG_SYSTEM_DIR="${REG_SYSTEM_DIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system}"
SYSTEM_PATTERN="${SYSTEM_PATTERN:-*${ARM}*.pt}"
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
CPUS="${CPUS:-24}"

shopt -s nullglob
FILES=( "${REG_SYSTEM_DIR}"/${SYSTEM_PATTERN} )
shopt -u nullglob
echo "arm ${ARM}, ridge ${RIDGE}: ${#FILES[@]} systems matching ${REG_SYSTEM_DIR}/${SYSTEM_PATTERN}"
printf '  %s\n' "${FILES[@]##*/}"
echo "lams ${LAM_LIST[*]} | columns ${COLS} <= -${EPS} nats/sd from t >= ${T_MIN} | ${METHOD}, <= ${MAX_ITER} rounds | -> ${OUTDIR} | logs ${LOGDIR}"
echo "QoS ${QOS:-default}, ${CPUS} CPUs, ${TIME}"
if [ ${#FILES[@]} -lt 1 ]; then echo "Error: no system file"; exit 1; fi
if [ -n "${DRY}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${OUTDIR}" "${LOGDIR}"

JOB=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=l6neg_${ARM}
#SBATCH --array=0-$(( ${#FILES[@]} - 1 ))
#SBATCH --time=${TIME}
#SBATCH --output=${LOGDIR}/slurm_%x_%A_%a.log
#SBATCH -A wbg@h100
#SBATCH --partition=gpu_p6
${QOS:+#SBATCH --qos=${QOS}}
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
FILES=( ${FILES[@]} )
python -u codes/resolve_theta_reg.py "\${FILES[\${SLURM_ARRAY_TASK_ID}]}" --lams ${LAM_LIST[@]} \
    --ridge ${RIDGE} --schedule uniform --mask auto --diagnose 0 --outdir "${OUTDIR}" \
    --neg_cols ${COLS} --neg_eps_nats ${EPS} --neg_t_min ${T_MIN} --neg_max_iter ${MAX_ITER} --neg_method ${METHOD}
EOT
)
if [ -z "${JOB}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOB} (array of ${#FILES[@]})"
