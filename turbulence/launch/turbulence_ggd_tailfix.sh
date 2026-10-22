#!/bin/bash
# Tail fix for Scalar_GGD_KRegion (2026-10-20).
#
# compute_marginal_agreement.py on the scalar_orthog arms: the simplest model
# (morletQ1_noL6: Scalar_morlet_gaussianK alone, M=128) has fine-band tails that are
# far too heavy (ch=0: q0.9999_ratio 10.4, m2_ratio 30.6); with L_6 they become too
# narrow. Beyond the last region cut the model log-density is -theta_K |z|^alpha_K,
# i.e. one (theta, alpha) pair sets the whole tail, and both the cut and alpha_K come
# from a bulk-dominated likelihood fit on few tail points.
#
# Same arm as morletQ1_noL6 (same data, seed, schedule, terms) with:
#   --ggd_tail_quantile 0.99           last cut fixed at q_0.99 of |z_j|
#   --ggd_tail_alpha_bounds 0.5 3      alpha of the outer region kept in [0.5, 3]
# The region form is fitted on x1 only (n1 windows), as in every other run, so the
# result is a statement about the model at this n1.
# Control: turbulence_scalar_orthog.sh arm morletQ1_noL6 (seed 900, already run).
# Success: q0.999_ratio in [0.9, 1.1] on ch 0-2 and no blow-up without L_6.
#
# Usage (from the repo root on Jean Zay):
#   bash turbulence/launch/turbulence_ggd_tailfix.sh
#   ONLY='tailfix_L6' bash ...          # optional variant with L_6

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEED_LIST=(${SEEDS:-900})
ONLY="${ONLY:-tailfix_noL6}"

# ── same knobs as turbulence_scalar_orthog.sh ───
N1=3000
SUBSERIES_LEN=512
TARGET_LEN=128
J=7
Q=1
SIGMA=3.5
NT=15000
SCHEDULE_ARGS="--schedule two_phase --n_bulk 1000 --t_switch 0.9 --gap_end 5e-5"
BATCH_SIZE=3000
REGULARIZATION=1e-4
COND_EVERY=10

SOLVE_FLOAT64=true
DEDUPLICATE_FILTERS=false

LAM=0.0
N_SUBSAMPLE=1
REG_SOLVER=thomas
REG_RIDGE=0.0
SAVE_REG_SYSTEM=true
SKIP_REG_SOLVE=true

# ── the tail fix ───
GGD_TAIL_QUANTILE=0.99
GGD_TAIL_ALPHA_BOUNDS="0.5 3"

# the control arm took ~4 min
TIME="00:30:00"
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
        tailfix_noL6) submit tailfix_noL6     Scalar_morlet_gaussianK ;;
        tailfix_L6)   submit tailfix_L6   L_6 Scalar_morlet_gaussianK ;;
        *) echo "unknown arm '${arm}'; use tailfix_noL6 | tailfix_L6"; exit 1 ;;
    esac
done
