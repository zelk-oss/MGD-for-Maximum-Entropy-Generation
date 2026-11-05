#!/bin/bash
# Offline Theta_reg for one cosmic-web arm + L_6 sign report. One array task per seed: codes/resolve_theta_reg.py
# on the seed's saved regularised system (cosmic_web.sh saves it), for every lam in LAMS, then
# theta_sign_constraint/check_l6_sign.py on the files it wrote (columns 0:21 = the L_6 block: 5 scales x 4
# orientations + low-pass; the builder puts L_6 first). Output of the sign check is in the SLURM log.
# Unconstrained solve. If the sign is violated, the L_6<0 solve is theta_sign_constraint/launch_l6neg.sh
# (not set up for 2D yet).
#
# Usage (repo root on Jean Zay):
#   ARM=base GPU=v100 bash cosmic_web/launch/cosmic_web_theta_reg.sh                        # SDE jobs finished
#   ARM=base GPU=v100 DEPEND=<SDE array job id> bash cosmic_web/launch/cosmic_web_theta_reg.sh   # task k after SDE task k
#   DRY=1 ...                                                                                # print only
# overrides: SEEDS, LAMS, RIDGE, LABEL, GPU (mandatory), TIME, CPUS, REG_SYSTEM_DIR, OUTDIR
#
# RIDGE default 1e-7 (paper App. D.1, as S&P); turbulence used 1e-4. Not tuned for 2D -> a choice to confirm.
# The in-run ridge is a different thing: 0.0, as in the notebook.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/../.." && pwd)"

ARM="${ARM:?set ARM=base or scat4}"
GPU="${GPU:?set GPU=v100 or h100 (no default on purpose)}"
LABEL="${LABEL:-${ARM}}"
SEED_LIST=(${SEEDS:-900 901 902 903 904})
LAM_LIST=(${LAMS:-0 1e-6 1e-5 1e-4 5e-4 1e-3 1e-2 1e-1 1})
RIDGE="${RIDGE:-1e-7}"
REG_SYSTEM_DIR="${REG_SYSTEM_DIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/cosmic_web/reg_system/${LABEL}}"
OUTDIR="${OUTDIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/cosmic_web/theta_reg/${LABEL}}"
LOGDIR="${PROJECT_DIR}/cosmic_web/saved_results/logs"
DEPEND="${DEPEND-}"
TIME="${TIME:-04:00:00}"
case "${GPU}" in
    h100) ACCOUNT="wbg@h100"; PARTITION="gpu_p6";  CONSTRAINT="h100"; MODULE_PRE="arch/h100"; CPUS="${CPUS:-24}" ;;
    v100) ACCOUNT="wbg@v100"; PARTITION="gpu_p13"; CONSTRAINT="v100"; MODULE_PRE="";          CPUS="${CPUS:-10}" ;;
    *) echo "GPU must be h100 or v100"; exit 1 ;;
esac

echo "arm ${ARM} (${LABEL}), seeds ${SEED_LIST[*]}, lams ${LAM_LIST[*]}, ridge ${RIDGE}"
echo "  systems: ${REG_SYSTEM_DIR}/*seed_<seed>_*.pt  ->  ${OUTDIR}   | ${GPU}, ${CPUS} CPUs, ${TIME}, $([ -n "${DEPEND}" ] && echo "after job ${DEPEND}" || echo "no dependency")"
if [ -n "${DRY}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${OUTDIR}" "${LOGDIR}"

JOB=$(sbatch --parsable ${DEPEND:+--dependency=aftercorr:${DEPEND} --kill-on-invalid-dep=yes} <<EOT
#!/bin/bash
#SBATCH --job-name=cw_thetareg_${LABEL}
#SBATCH --array=0-$(( ${#SEED_LIST[@]} - 1 ))
#SBATCH --time=${TIME}
#SBATCH --output=${LOGDIR}/slurm_%x_%A_%a.log
#SBATCH -A ${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH -C ${CONSTRAINT}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
SEED_ARR=( ${SEED_LIST[@]} )
SEED=\${SEED_ARR[\${SLURM_ARRAY_TASK_ID}]}
shopt -s nullglob
FILES=( "${REG_SYSTEM_DIR}"/*seed_\${SEED}_*.pt )
if [ \${#FILES[@]} -ne 1 ]; then
    echo "Error: seed \${SEED}: expected 1 system file, found \${#FILES[@]}:"; printf '  %s\n' "\${FILES[@]}"; exit 1
fi
echo "seed \${SEED}: \${FILES[0]}"
python -u codes/resolve_theta_reg.py "\${FILES[0]}" --lams ${LAM_LIST[@]} \
    --ridge ${RIDGE} --schedule uniform --mask auto --diagnose 0 --outdir "${OUTDIR}"
python -u theta_sign_constraint/check_l6_sign.py --cols 0:21 "${OUTDIR}"/*seed_\${SEED}_*/lam*.pt
EOT
)
if [ -z "${JOB}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOB} (array of ${#SEED_LIST[@]})"
