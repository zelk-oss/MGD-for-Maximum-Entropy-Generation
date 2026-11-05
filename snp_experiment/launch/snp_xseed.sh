#!/bin/bash
# Cross-seed choice of lam for one S&P model (lam_selection_criterion/turb_xseed.py, the rule decided 10-06: xseed).
# Phase 2 only. Phase 1 = snp_theta_reg.sh (the Theta_reg solve for every lam in LAMS, one task per seed): it already
# wrote the Theta_reg of every (seed, lam), so there is NO extra solve at the chosen lam -- the selected files are the
# phase-1 files: out["theta_files"][window][config] in the result.
# One task: scores every seed's Theta(lam) with the other seeds' data terms, pools the pairs, prints the selected lam
# for the whole transport ('whole'), t >= 0.99 ('end0.99', what log p at t = 1 uses) and t >= 0.999, and writes
#   <XOUT>/xseed_<ARM>.pt   (default XOUT = snp_experiment/results/xseed)
# LAMS must be the grid used in phase 1 (same default as snp_theta_reg.sh). Needs >= 2 seeds.
#
# Usage (repo root on Jean Zay), chained after the Theta_reg array of the model (afterok: all its tasks must succeed):
#   MODEL=M0 DEPEND=<Theta_reg array job id> bash snp_experiment/launch/snp_xseed.sh
#   DRY=1 MODEL=M0 ...                       # print only
# overrides: SEEDS, LAMS, RIDGE (1e-7), LABEL_SUFFIX, GPU (v100), TIME (01:00:00), CPUS, REG_SYSTEM_DIR, OUTDIR, XOUT

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/../.." && pwd)"

MODEL="${MODEL:?set MODEL=M0|M1|M2|M3}"
ARM="snp_${MODEL}${LABEL_SUFFIX}"
SEED_LIST=(${SEEDS:-900 901 902 903 904})
LAM_LIST=(${LAMS:-0 1e-6 1e-5 1e-4 5e-4 1e-3 1e-2 1e-1 1})
RIDGE="${RIDGE:-1e-7}"
REG_SYSTEM_DIR="${REG_SYSTEM_DIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/snp_experiment/reg_system/${ARM}}"
OUTDIR="${OUTDIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/snp_experiment/theta_reg/${ARM}}"
XOUT="${XOUT:-${PROJECT_DIR}/snp_experiment/results/xseed}"
LOGDIR="${PROJECT_DIR}/snp_experiment/saved_results/logs"
DEPEND="${DEPEND-}"
TIME="${TIME:-01:00:00}"
GPU="${GPU:-v100}"
case "${GPU}" in
    h100) ACCOUNT="wbg@h100"; PARTITION="gpu_p6";  CONSTRAINT="h100"; MODULE_PRE="arch/h100"; CPUS="${CPUS:-24}" ;;
    v100) ACCOUNT="wbg@v100"; PARTITION="gpu_p13"; CONSTRAINT="v100"; MODULE_PRE="";          CPUS="${CPUS:-10}" ;;
    *) echo "GPU must be h100 or v100"; exit 1 ;;
esac
if [ ${#SEED_LIST[@]} -lt 2 ]; then echo "Error: the cross-seed rule needs at least 2 seeds"; exit 1; fi

echo "model ${MODEL} (${ARM}): xseed lam choice over seeds ${SEED_LIST[*]}, lams ${LAM_LIST[*]}, ridge ${RIDGE}"
echo "  systems ${REG_SYSTEM_DIR}/*seed_<seed>_*.pt, Theta_reg sweep ${OUTDIR} -> ${XOUT}/xseed_${ARM}.pt | ${GPU}, ${CPUS} CPUs, ${TIME}, $([ -n "${DEPEND}" ] && echo "after job ${DEPEND}" || echo "no dependency")"
if [ -n "${DRY}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${XOUT}" "${LOGDIR}"

JOB=$(sbatch --parsable ${DEPEND:+--dependency=afterok:${DEPEND} --kill-on-invalid-dep=yes} <<EOT
#!/bin/bash
#SBATCH --job-name=snp_xseed_${ARM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOGDIR}/slurm_%x_%j.log
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
shopt -s nullglob
FILES=()
for SEED in ${SEED_LIST[@]}; do FILES+=( "${REG_SYSTEM_DIR}"/*seed_\${SEED}_*.pt ); done
if [ \${#FILES[@]} -ne ${#SEED_LIST[@]} ]; then echo "Error: expected ${#SEED_LIST[@]} system files, found \${#FILES[@]}"; printf '  %s\n' "\${FILES[@]}"; exit 1; fi
echo "systems: \${FILES[@]}"
python -u lam_selection_criterion/turb_xseed.py --systems "\${FILES[@]}" --sweep_dir "${OUTDIR}" \
    --lams ${LAM_LIST[@]} --ridge ${RIDGE} --arm ${ARM} --out "${XOUT}"
EOT
)
if [ -z "${JOB}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOB}"
