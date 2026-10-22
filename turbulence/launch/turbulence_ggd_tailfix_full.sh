#!/bin/bash
# Full-resolution counterpart of turbulence_ggd_tailfix.sh (2026-10-20).
#
# At M=128 the tail fix (last cut forced to q_0.99 of |z_j|, outer alpha in [0.5, 3])
# turned morletQ1_noL6 from a blow-up (ch=0: q0.9999=10.4, m2=30.6) into an essentially
# matched arm (q0.9999=0.98, m2=1.89) with ONE Scalar_morlet_gaussianK potential, no
# L_6, no Scat4. Open question: does the same fix hold at M=256 Q=1, where the finest
# band is more intermittent? If yes, the simplest model with the fix is the solution
# at full resolution; if no, the fix helps but is not sufficient at this resolution.
#
# Identical to turbulence_scalar_orthog_full.sh (M=256, J=8, n1=10000, nt=55000) with:
#   TERMS                      Scalar_morlet_gaussianK only (optional L_6 variant)
#   --ggd_tail_quantile 0.99   outermost region cut forced to q_0.99 of |z_j|
#   --ggd_tail_alpha_bounds 0.5 3   outer-region alpha restricted (keeps the tail
#                                   statistic both sensitive and well-estimated)
# The region FORM is fitted on x1 (n1 windows), as in every other run.
#
# Usage (on Jean Zay, from the repo root):
#   bash turbulence/launch/turbulence_ggd_tailfix_full.sh                     # noL6 only
#   ONLY='tailfix_noL6_full tailfix_L6_full' bash ...                         # both arms

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEED_LIST=(${SEEDS:-900})
ONLY="${ONLY:-tailfix_noL6_full}"

# ── full-resolution knobs (match turbulence_scalar_orthog_full.sh) ──
N1=10000
SUBSERIES_LEN=512
TARGET_LEN=256
J=8
Q=1
SIGMA=3.5
NT=55000
SCHEDULE_ARGS="--schedule two_phase --n_bulk 5000 --t_switch 0.9 --gap_end 5e-5"
BATCH_SIZE=5000
REGULARIZATION=1e-4
COND_EVERY=10

SOLVE_FLOAT64=true
DEDUPLICATE_FILTERS=false        # morlet Q=1 at M=256 has no near-copies

LAM=0.0
N_SUBSAMPLE=1
REG_SOLVER=thomas
REG_RIDGE=0.0
SAVE_REG_SYSTEM=true
SKIP_REG_SOLVE=true

# ── the tail fix ──
GGD_TAIL_QUANTILE=0.99
GGD_TAIL_ALPHA_BOUNDS="0.5 3"

# scalar_orthog_full took ~1h10 per arm (M=256, 1 fam); same here
TIME="02:00:00"
CPUS=24
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
MODULE_PRE="arch/h100"

submit() {
    local arm="$1"; shift 1
    EXP_NAME="turbulence_ggd_${arm}"
    LABEL="${EXP_NAME}"
    TERMS=("$@")
    WAV_TYPE="morlet"
    REG_SYSTEM_DIR="${SCRATCH:+${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system/${EXP_NAME}}"
    echo "============================================================"
    echo "Arm ${arm}: TERMS = ${TERMS[*]}"
    echo "============================================================"
    source "${LAUNCH_DIR}/_submit_turb_sweep.sh"
}

for arm in ${ONLY}; do
    case "${arm}" in
        tailfix_noL6_full) submit tailfix_noL6_full     Scalar_morlet_gaussianK ;;
        tailfix_L6_full)   submit tailfix_L6_full   L_6 Scalar_morlet_gaussianK ;;
        *) echo "unknown arm '${arm}'; use tailfix_noL6_full | tailfix_L6_full"; exit 1 ;;
    esac
done
