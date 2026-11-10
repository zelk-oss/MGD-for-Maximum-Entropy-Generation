#!/bin/bash
# Offline Theta_reg for one S&P model + sign report. One array task per seed: codes/resolve_theta_reg.py on the seed's
# saved regularised system (snp_models.sh saves it, SKIP_REG_SOLVE), for every lam in LAMS, ridge 1e-7 (paper App. D.1),
# saved live mask; then snp_experiment/check_sign.py on the files it wrote (output in the SLURM log).
# The system is small (r ~ 143-173, ~5000 nodes: ~1-2 GB), so this is minutes per (seed, lam) on CPU.
# Unconstrained: these models have no L_6. The sign report lists, per family, how many Theta entries are > 0 (code sign,
# < 0 confines), with scat2 and the real scat4 diagonal flagged as the confining candidates. If it shows a violation we
# decide what to bound (codes/resolve_theta_reg.py --neg_cols takes a column range) -- not done here.
#
# Usage (from the repo root on Jean Zay):
#   MODEL=M0 DEPEND=<SDE array job id> bash snp_experiment/launch/snp_theta_reg.sh    # chained: task k after SDE task k
#   MODEL=M0 bash snp_experiment/launch/snp_theta_reg.sh                              # SDE jobs already finished
#   DRY=1 MODEL=M0 bash ...                                                           # print what would run
# overrides: SEEDS (900 901 902 903 904), LAMS, RIDGE (1e-7), LABEL_SUFFIX (as in snp_models.sh), GPU (v100), TIME, CPUS,
#   REG_SYSTEM_DIR, OUTDIR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/../.." && pwd)"

MODEL="${MODEL:?set MODEL=M0|M1|M2|M3}"
[ "${L6:-0}" = "1" ] && LABEL_SUFFIX="${LABEL_SUFFIX:-_L6}"      # arms with L_6 (snp_models.sh L6=1): sign check with the L_6 block
ARM="snp_${MODEL}${LABEL_SUFFIX}"
SEED_LIST=(${SEEDS:-900 901 902 903 904})
LAM_LIST=(${LAMS:-0 1e-6 1e-5 1e-4 5e-4 1e-3 1e-2 1e-1 1})
RIDGE="${RIDGE:-1e-7}"
REG_SYSTEM_DIR="${REG_SYSTEM_DIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/snp_experiment/reg_system/${ARM}}"
OUTDIR="${OUTDIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/snp_experiment/theta_reg/${ARM}}"
LOGDIR="${PROJECT_DIR}/snp_experiment/saved_results/logs"
DEPEND="${DEPEND-}"
TIME="${TIME:-02:00:00}"
GPU="${GPU:-v100}"
case "${GPU}" in
    h100) ACCOUNT="wbg@h100"; PARTITION="gpu_p6";  CONSTRAINT="h100"; MODULE_PRE="arch/h100"; CPUS="${CPUS:-24}" ;;
    v100) ACCOUNT="wbg@v100"; PARTITION="gpu_p13"; CONSTRAINT="v100"; MODULE_PRE="";          CPUS="${CPUS:-10}" ;;
    *) echo "GPU must be h100 or v100"; exit 1 ;;
esac

echo "model ${MODEL} (${ARM}), seeds ${SEED_LIST[*]}, lams ${LAM_LIST[*]}, ridge ${RIDGE}"
echo "  systems: ${REG_SYSTEM_DIR}/*seed_<seed>_*.pt  ->  ${OUTDIR}   | ${GPU}, ${CPUS} CPUs, ${TIME}, $([ -n "${DEPEND}" ] && echo "after job ${DEPEND}" || echo "no dependency")"
if [ -n "${DRY}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${OUTDIR}" "${LOGDIR}"

JOB=$(sbatch --parsable ${QOS:+--qos=${QOS}} ${DEPEND:+--dependency=aftercorr:${DEPEND} --kill-on-invalid-dep=yes} <<EOT
#!/bin/bash
#SBATCH --job-name=snp_thetareg_${ARM}
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
python -u snp_experiment/check_sign.py --model ${MODEL} --J 6 $([ "${L6:-0}" = "1" ] && echo --l6) "${OUTDIR}"/*seed_\${SEED}_*/lam*.pt
EOT
)
if [ -z "${JOB}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOB} (array of ${#SEED_LIST[@]})"
