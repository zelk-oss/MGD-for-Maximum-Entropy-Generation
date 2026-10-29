#!/bin/bash
# Jets (Modane record) MGD runs, set up like the turbulence runs (turbulence/launch/turbulence_nested_M3_bicw.sh):
# two-phase time grid, float64 block-Thomas Theta_reg (moment energy), Gram pre-flight, scalar windows by
# constrained likelihood + tail mass. lambda is NOT chosen here: the regularised system is saved
# (--save_reg_system, --skip_reg_solve) and lambda / the L_6<0 chain are done afterwards, offline.
#
# Two arms (ARM):
#   full      L_6, Scattering 1st/2nd/3rd (Real+Imag)/4th Mod2 (Real+Imag), Scalar_morlet_gaussianK
#   noscalar  the same without the scalar potential
# (Scalar_psi_gaussianK is not used: at Q=1 it is the same bank as Scalar_morlet_gaussianK.)
#
# Choices (2026-10-28): N1 10000, nt 40000, seeds 500-504, regularization 1e-4, SUBSERIES_LEN 512 -> 256.
# Schedule from the data (jets/data_audit/choose_schedule.py): gap_end 7.5e-4 = leftover Cos noise 1% of the
# data POWER at Nyquist (the same criterion that gives 5e-5 for turbulence); tail ratio r = 1.4e-4.
# Window bound --ggd_bic_min_window 1 (x the noise of one SDE step at 1-t = 1e-2) = 5.8e-3 on this grid.
#
# Time: TIME 14 h; --time_limit_min is derived from it and the SDE loop gets TIME minus a 30 min reserve
# (min(30 min, 10%), loop_budget_s) = 13.5 h; the guard aborts early if the projection exceeds it.
# Measured 2026-10-28 on H100 (n1 10000, nt 40000): full 1.17 s/it -> 12.98 h projected (r = 299 statistics),
# noscalar 1.07 s/it -> 11.92 h. A 10 h job aborted both arms after ~130 steps (as designed).
#
# Usage (on Jean Zay, from the repo root):
#   ARM=full PREFLIGHT_ONLY=true bash jets/launch_jet_SDE.sh   # fit + Gram check only, one seed, 30 min
#   ARM=full bash jets/launch_jet_SDE.sh
#   ARM=noscalar bash jets/launch_jet_SDE.sh
#   SEEDS="500" ARM=full bash jets/launch_jet_SDE.sh           # other seeds

# timestamp shared by all array tasks of one submission
TIMESTAMP=$(date +"%Y%m%d_%H%M")

ARM="${ARM:-full}"
SEED_LIST=(${SEEDS:-500 501 502 503 504})
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"

# ── PARAMETERS ───────────────────────────────────────
N1=10000
NT=40000
SIGMA=3.5
SUBSERIES_LEN=512
TARGET_LEN=256
J=8
Q=1
INTERPOLANT="Cos"
SCHEDULE_EXPONENT=2                       # only used by --schedule power
SCHEDULE_ARGS="--schedule two_phase --n_bulk 5000 --t_switch 0.9 --gap_end 7.5e-4"
BATCH_SIZE=5000

# Theta_reg: as turbulence (lambda after the run)
REGULARIZATION=1e-4
LAM=0.0
N_SUBSAMPLE=1
REG_SOLVER=thomas
REG_RIDGE=0.0
SOLVE_FLOAT64=true
DEDUPLICATE_FILTERS=true
DEDUP_TOL=1e-8
COND_EVERY=10
SAVE_REG_SYSTEM=true
SKIP_REG_SOLVE=true
GRAM_PREFLIGHT="${GRAM_PREFLIGHT:-1e-6}"

# scalar windows (variant B of jets/data_audit/scalar_windows)
GGD_BOUNDARY_METHOD=likelihood
GGD_TAIL_QUANTILE=0.99
GGD_BIC_MIN_WINDOW=1
GGD_NOISE_REF_GAP=1e-2

# 🔥 POTENTIALS (MAIN CONTROL)
BASE_TERMS=(
    L_6
    Scattering_First_Order
    Scattering_Second_Order
    Scattering_Third_Order_Real
    Scattering_Third_Order_Imag
    Scattering_Fourth_Order_Mod2_Real_Q1
    Scattering_Fourth_Order_Mod2_Imag_Q1
)
case "${ARM}" in
    full)     TERMS=("${BASE_TERMS[@]}" Scalar_morlet_gaussianK); SCALAR=true ;;
    noscalar) TERMS=("${BASE_TERMS[@]}");                         SCALAR=false ;;
    *) echo "unknown ARM '${ARM}'; use full | noscalar"; exit 1 ;;
esac

# ── SLURM & JEAN ZAY QOS CONFIG ──────────────────────
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
TIME="14:00:00"          # job time; loop budget = TIME - 30 min reserve = 13.5 h (loop_budget_s in codes/sde_routines.py)
CPUS=24
NGPUS=1

if [ "${PREFLIGHT_ONLY}" = "true" ]; then
    TIME="00:30:00"
    SEED_LIST=(${SEED_LIST[0]})
fi

# ── paths ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_jet_SDE.py"
OUTDIR="${SCRIPT_DIR}"
LOG_DIR="${OUTDIR}/saved_results/logs"
mkdir -p "${LOG_DIR}"

EXP_NAME="jets_${ARM}"
LABEL="${EXP_NAME}"
REG_SYSTEM_DIR="${SCRATCH:+${SCRATCH}/MGD-for-Maximum-Entropy-Generation/jets/reg_system/${EXP_NAME}}"

NTASKS=${#SEED_LIST[@]}
IFS=: read -r _h _m _s <<< "${TIME}"
TIME_LIMIT_MIN=$(( 10#${_h} * 60 + 10#${_m} ))

# ── optional flags ───────────────────────────────────
EXTRA_FLAGS=" --label ${LABEL} ${SCHEDULE_ARGS}"
[ -n "${BATCH_SIZE}" ]                 && EXTRA_FLAGS+=" --batch_size ${BATCH_SIZE}"
[ "${SOLVE_FLOAT64}" = "true" ]        && EXTRA_FLAGS+=" --solve_float64"
[ "${DEDUPLICATE_FILTERS}" = "true" ]  && EXTRA_FLAGS+=" --deduplicate_filters --dedup_tol ${DEDUP_TOL}"
[ -n "${COND_EVERY}" ]                 && EXTRA_FLAGS+=" --cond_every ${COND_EVERY}"
[ -n "${GRAM_PREFLIGHT}" ]             && EXTRA_FLAGS+=" --gram_preflight ${GRAM_PREFLIGHT}"
[ "${PREFLIGHT_ONLY}" = "true" ]       && EXTRA_FLAGS+=" --preflight_only"
if [ "${SCALAR}" = "true" ]; then
    EXTRA_FLAGS+=" --ggd_boundary_method ${GGD_BOUNDARY_METHOD} --ggd_tail_quantile ${GGD_TAIL_QUANTILE} --ggd_tail_mass"
    EXTRA_FLAGS+=" --ggd_bic_min_window ${GGD_BIC_MIN_WINDOW} --ggd_noise_ref_gap ${GGD_NOISE_REF_GAP}"
fi
if [ "${SAVE_REG_SYSTEM}" = "true" ]; then
    EXTRA_FLAGS+=" --save_reg_system"
    [ -n "${REG_SYSTEM_DIR}" ] && EXTRA_FLAGS+=" --reg_system_dir ${REG_SYSTEM_DIR}"
fi
[ "${SKIP_REG_SOLVE}" = "true" ]       && EXTRA_FLAGS+=" --skip_reg_solve"

# ── terms → string ───────────────────────────────────
TERMS_STR="${TERMS[*]}"

echo "Submitting ${EXP_NAME}: ${NTASKS} seeds (${SEED_LIST[*]}), n1=${N1}, nt=${NT}, lam=${LAM}, regularization=${REGULARIZATION}"
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
module load arch/h100
module load pytorch-gpu/py3/2.8.0
cd "${SCRIPT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED_LIST=(${SEED_LIST[@]})
SEED_VAL=\${SEED_LIST[\${SLURM_ARRAY_TASK_ID}]}

srun python "${PYTHON_SCRIPT}" \
    --seed \${SEED_VAL} \
    --timestamp "${TIMESTAMP}" \
    --dataset modane \
    --n1 ${N1} \
    --nt ${NT} \
    --subseries_len ${SUBSERIES_LEN} \
    --target_len ${TARGET_LEN} \
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

# Ledger of every submission (next to the SLURM logs).
LEDGER="${LOG_DIR}/submissions.tsv"
[ -f "${LEDGER}" ] || printf 'date\tjob_id\texp_name\tscript\ttimestamp\tseeds\tn1\tnt\tlam\tterms\n' > "${LEDGER}"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Iseconds)" "${JOBID}" "${EXP_NAME}" "$(basename "$0")" "${TIMESTAMP}" \
    "${SEED_LIST[*]}" "${N1}" "${NT}" "${LAM}" "${TERMS_STR}" >> "${LEDGER}"
