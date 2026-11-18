#!/bin/bash
# Diagonal / shrunk C in the Theta_reg time term, on saved turbulence_2d systems (2026-11-17). No SDE run.
# One array task per seed. Per task, for each walker count (systems in reg_system/<LABEL> for LABEL in LABELS):
#   1. codes/resolve_theta_reg.py, full C            (lams LAMS_FULL, includes 0 = no time term)
#   2. codes/resolve_theta_reg.py --c_shrink 1 -1    (1 = pure diagonal, -1 = Gaussian plug-in alpha; lams LAMS)
# then codes/compare_c_shrink.py over ALL seeds whose files exist (the last task to finish writes the full table):
# std_x[log p_arm - log p_ref] at t ~ 1 against the full-C solve of REF_LABEL (many walkers), same seed and lam.
#
# Usage (repo root on Jean Zay):
#   GPU=v100 bash turbulence_2d/launch/c_shrink.sh
#   DRY=1 ...                                                       # print only
# overrides: SEEDS (900..904), LABELS ("ho_base_real_np100 ho_base_real_np400"), REF_LABEL (ho_base_real_np400),
#   LAMS (1e-4 1e-3 1e-2 1e-1 1), LAMS_FULL (0 + LAMS), RIDGE (1e-7), TIME (01:30:00), CPUS, OUTDIR, CSV
# Billing: GPU dev QoS (never prepost), 1 GPU x elapsed per task although the solve runs on the CPU cores.
# Threads fixed at CPUS for every arm (the solve is ill-conditioned: thread count alone moves Theta by a few %).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/../.." && pwd)"

GPU="${GPU:?set GPU=v100 or h100 (no default on purpose)}"
SEED_LIST=(${SEEDS:-900 901 902 903 904})
LABEL_LIST=(${LABELS:-ho_base_real_np100 ho_base_real_np400})
REF_LABEL="${REF_LABEL:-ho_base_real_np400}"
LAM_LIST=(${LAMS:-1e-4 1e-3 1e-2 1e-1 1})
LAM_FULL_LIST=(${LAMS_FULL:-0 ${LAM_LIST[*]}})
RIDGE="${RIDGE:-1e-7}"
SYS_ROOT="${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence_2d/reg_system"
OUTDIR="${OUTDIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence_2d/theta_reg_cshrink}"
CSV="${CSV:-${PROJECT_DIR}/turbulence_2d/saved_results/c_shrink/compare_c_shrink.csv}"
LOGDIR="${PROJECT_DIR}/turbulence_2d/saved_results/logs"
TIME="${TIME:-01:30:00}"
case "${GPU}" in
    h100) ACCOUNT="wbg@h100"; PARTITION="gpu_p6";  CONSTRAINT="h100"; MODULE_PRE="arch/h100"; CPUS="${CPUS:-24}"; QOS=qos_gpu_h100-dev ;;
    v100) ACCOUNT="wbg@v100"; PARTITION="gpu_p13"; CONSTRAINT="v100"; MODULE_PRE="";          CPUS="${CPUS:-10}"; QOS=qos_gpu-dev ;;
    *) echo "GPU must be h100 or v100"; exit 1 ;;
esac

echo "c_shrink: seeds ${SEED_LIST[*]}, labels ${LABEL_LIST[*]} (ref ${REF_LABEL}), lams full ${LAM_FULL_LIST[*]} / shrink ${LAM_LIST[*]}"
echo "  systems ${SYS_ROOT}/<label>/*seed_<seed>_*.pt -> ${OUTDIR} ; table -> ${CSV} | ${GPU} ${QOS}, ${CPUS} CPUs, ${TIME}"
if [ -n "${DRY}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${OUTDIR}" "${LOGDIR}" "$(dirname "${CSV}")"

JOB=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=t2d_cshrink
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
for LBL in ${LABEL_LIST[@]}; do
    FILES=( "${SYS_ROOT}/\${LBL}"/*seed_\${SEED}_*.pt )
    if [ \${#FILES[@]} -ne 1 ]; then
        echo "Error: \${LBL} seed \${SEED}: expected 1 system file, found \${#FILES[@]}"; exit 1
    fi
    echo "== \${LBL} seed \${SEED}: \${FILES[0]}"
    python -u codes/resolve_theta_reg.py "\${FILES[0]}" --lams ${LAM_FULL_LIST[@]} --ridge ${RIDGE} \
        --schedule uniform --mask auto --diagnose 0 --mmap --threads ${CPUS} --outdir "${OUTDIR}"
    python -u codes/resolve_theta_reg.py "\${FILES[0]}" --lams ${LAM_LIST[@]} --ridge ${RIDGE} --c_shrink 1 -1 \
        --schedule uniform --mask auto --diagnose 0 --mmap --threads ${CPUS} --outdir "${OUTDIR}"
done
python -u codes/compare_c_shrink.py --theta_root "${OUTDIR}" --systems_root "${SYS_ROOT}/${REF_LABEL}" \
    --labels ${LABEL_LIST[@]} --ref_label ${REF_LABEL} --seeds ${SEED_LIST[@]} --lams ${LAM_FULL_LIST[@]} \
    --ridge ${RIDGE} --csv "${CSV}"
EOT
)
if [ -z "${JOB}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOB} (array of ${#SEED_LIST[@]})"
