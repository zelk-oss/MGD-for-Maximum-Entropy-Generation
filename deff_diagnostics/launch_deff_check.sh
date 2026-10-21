#!/bin/bash
# deff_diagnostics/deff_latest_run_check.py on 5 seeds of one arm, one array task per seed (dev QoS:
# 5 tasks <= the 10-job cap). Cost NOT measured: rough guess 20-60 min per seed on one H100 at the
# defaults (40 points x 160 nodes; denoiser 64+256+256(f32), energy 4 variants x (64+256) MC draws).
# Check the first task's "points ... done, N s" lines; if it will not fit in TIME (dev max 2 h),
# rerun with EXTRA="--n_eval 100" or EXTRA="--n_mc 32 128".
#
# Usage (repo root on Jean Zay):
#   DRY=1 bash deff_diagnostics/launch_deff_check.sh
#   RUN=final bash deff_diagnostics/launch_deff_check.sh                             # turbulence_final, L_6 free, lam 5e-4
#   RUN=latest ARM=reg1e-4 THETA=neg bash deff_diagnostics/launch_deff_check.sh      # earlier noL6psi_noSm30
# overrides: RUN (latest|final), ARM (reg1e-4|reg1e-2), THETA (neg|free), LAM, RIDGE, SEEDS, EXTRA, QOS, TIME

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
RUN="${RUN:-latest}"
ARM="${ARM:-reg1e-4}"
THETA="${THETA:-neg}"
LAM="${LAM-}"
RIDGE="${RIDGE-}"
EXTRA="${EXTRA-}"
QOS="${QOS-qos_gpu_h100-dev}"
TIME="${TIME:-02:00:00}"
LOGDIR="${HERE}/results"

if [ "${RUN}" = "final" ]; then
    SEEDS=(${SEEDS:-900 901 903 904 905})
    JNAME="deff_final"
    PYARGS="--run final"
    [ -n "${LAM}" ]   && PYARGS="${PYARGS} --lam ${LAM}"
    [ -n "${RIDGE}" ] && PYARGS="${PYARGS} --ridge ${RIDGE}"
else
    SEEDS=(${SEEDS:-900 901 902 903 904})
    JNAME="deff_${ARM}_${THETA}"
    PYARGS="--run latest --arm ${ARM} --theta ${THETA}"
    [ -n "${LAM}" ]   && PYARGS="${PYARGS} --lam ${LAM}"
    [ -n "${RIDGE}" ] && PYARGS="${PYARGS} --ridge ${RIDGE}"
fi

echo "run ${RUN}, seeds ${SEEDS[*]}, pyargs '${PYARGS}', QoS ${QOS:-default}, ${TIME}, extra '${EXTRA}' -> ${LOGDIR}"
if [ -n "${DRY}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${LOGDIR}"

JOB=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=${JNAME}
#SBATCH --array=0-$(( ${#SEEDS[@]} - 1 ))
#SBATCH --time=${TIME}
#SBATCH --output=${LOGDIR}/slurm_%x_%A_%a.log
#SBATCH -A wbg@h100
#SBATCH --partition=gpu_p6
${QOS:+#SBATCH --qos=${QOS}}
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
SEEDS=( ${SEEDS[@]} )
python -u deff_diagnostics/deff_latest_run_check.py ${PYARGS} \
    --seed "\${SEEDS[\${SLURM_ARRAY_TASK_ID}]}" ${EXTRA}
EOT
)
if [ -z "${JOB}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOB} (array of ${#SEEDS[@]})"
