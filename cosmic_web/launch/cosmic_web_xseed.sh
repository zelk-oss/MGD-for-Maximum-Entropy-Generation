#!/bin/bash
# Cross-seed choice of lam for one cosmic-web arm (lam_selection_criterion/turb_xseed.py, the xseed rule of 10-06),
# phase 2 after cosmic_web_theta_reg.sh. The Theta_reg of every (seed, lam) already exists, so the selected files are
# the phase-1 files (out["theta_files"][window][config] in xseed_<ARM>.pt). One task; needs >= 2 seeds.
# Prints the selected lam for the whole transport ('whole'), t >= 0.99 ('end0.99', what log p at t = 1 uses), t >= 0.999.
# Watch for "GRID EDGE" (selected lam at the end of LAMS).
#
# Usage (repo root on Jean Zay), chained after the Theta_reg array (afterok):
#   ARM=base GPU=v100 DEPEND=<Theta_reg array job id> bash cosmic_web/launch/cosmic_web_xseed.sh
#   DRY=1 ...                                                                  # print only
# overrides: SEEDS, LAMS (same grid as phase 1), RIDGE (same as phase 1), LABEL, GPU (mandatory), TIME, CPUS,
#   REG_SYSTEM_DIR, OUTDIR, XOUT

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
XOUT="${XOUT:-${PROJECT_DIR}/cosmic_web/results/xseed}"
LOGDIR="${PROJECT_DIR}/cosmic_web/saved_results/logs"
DEPEND="${DEPEND-}"
TIME="${TIME:-02:00:00}"
case "${GPU}" in
    h100) ACCOUNT="wbg@h100"; PARTITION="gpu_p6";  CONSTRAINT="h100"; MODULE_PRE="arch/h100"; CPUS="${CPUS:-24}" ;;
    v100) ACCOUNT="wbg@v100"; PARTITION="gpu_p13"; CONSTRAINT="v100"; MODULE_PRE="";          CPUS="${CPUS:-10}" ;;
    *) echo "GPU must be h100 or v100"; exit 1 ;;
esac
if [ ${#SEED_LIST[@]} -lt 2 ]; then echo "Error: the cross-seed rule needs at least 2 seeds"; exit 1; fi

echo "arm ${ARM} (${LABEL}): xseed lam choice over seeds ${SEED_LIST[*]}, lams ${LAM_LIST[*]}, ridge ${RIDGE}"
echo "  systems ${REG_SYSTEM_DIR}/*seed_<seed>_*.pt, Theta_reg sweep ${OUTDIR} -> ${XOUT}/xseed_${LABEL}.pt | ${GPU}, ${CPUS} CPUs, ${TIME}, $([ -n "${DEPEND}" ] && echo "after job ${DEPEND}" || echo "no dependency")"
if [ -n "${DRY}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${XOUT}" "${LOGDIR}"

JOB=$(sbatch --parsable ${DEPEND:+--dependency=afterok:${DEPEND} --kill-on-invalid-dep=yes} <<EOT
#!/bin/bash
#SBATCH --job-name=cw_xseed_${LABEL}
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
    --lams ${LAM_LIST[@]} --ridge ${RIDGE} --arm ${LABEL} --out "${XOUT}"
EOT
)
if [ -z "${JOB}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOB}"
