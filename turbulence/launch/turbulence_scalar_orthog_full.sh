#!/bin/bash
# Full-resolution counterpart of turbulence_scalar_orthog.sh (2026-10-20).
#
# The coarse 4-arm test (M=128, nt=15000) showed scalar GGD-K per channel
# recovers per-channel marginals on the model's own bank. In turbulence_final_noScat4
# (M=256, nt=55000) the fine-band Q=1 marginals are visibly bad even on their own
# bank (data tails to +-0.4, synth to +-0.15 on j=0). Two confounds:
#   (a) resolution: M=256 keeps fine-scale intermittency the M=128 Db3 smoothed out
#   (b) model: noScat4 has 4 families on 2 banks (Scalar_morlet + L_6 + L_2_lowpass
#       on Q=1 + Scalar_psi on Q=3); the coarse test had 1 family on 1 bank.
#
# This launcher removes (b) and holds the test at (a): 4 arms with 1 family + optional
# L_6, M=256 full resolution matching turbulence_final_noScat4 on n1, nt, schedule, dedup.
#   morletQ1_L6_full      : L_6 + Scalar_morlet_gaussianK, morlet Q=1
#   morletQ1_noL6_full    :        Scalar_morlet_gaussianK, morlet Q=1
#   shannon_L6_full       : L_6 + Scalar_morlet_gaussianK, shannon Q=1
#   shannon_noL6_full     :        Scalar_morlet_gaussianK, shannon Q=1
# Marginals on own bank at M=256 should decide: if bad, scalar GGD-K is intrinsically
# insufficient at this resolution regardless of orthogonality -> need Scat4 or K>>4.
# If good, noScat4's bad marginals are a symptom of model complexity (2 banks competing),
# not of scalar potentials being too few.
#
# Time budget: coarse took ~3.5 min / arm at M=128 nt=15000. Scaling to M=256 nt=55000
# (~4x data, ~3.7x steps) => ~45-60 min / arm. TIME 02:00:00 for safety.
#
# Usage (from the repo root on Jean Zay):
#   bash turbulence/launch/turbulence_scalar_orthog_full.sh
#   ONLY='shannon_L6_full shannon_noL6_full' bash ...
#   SEEDS='900 901' bash ...

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEED_LIST=(${SEEDS:-900})
ONLY="${ONLY:-morletQ1_L6_full morletQ1_noL6_full shannon_L6_full shannon_noL6_full}"

# ── full-resolution knobs (match turbulence_final_noScat4) ──
N1=10000
SUBSERIES_LEN=512
TARGET_LEN=256
J=8
Q=1                     # single bank only; Scalar_psi (which uses filters_Q) NOT in TERMS
SIGMA=3.5
NT=55000
SCHEDULE_ARGS="--schedule two_phase --n_bulk 5000 --t_switch 0.9 --gap_end 5e-5"
BATCH_SIZE=5000
REGULARIZATION=1e-4
COND_EVERY=10

SOLVE_FLOAT64=true
DEDUPLICATE_FILTERS=false        # shannon disjoint by construction; morlet Q=1 negligible overlap

LAM=0.0
N_SUBSAMPLE=1
REG_SOLVER=thomas
REG_RIDGE=0.0
SAVE_REG_SYSTEM=true
SKIP_REG_SOLVE=true

TIME="02:00:00"
CPUS=24
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
MODULE_PRE="arch/h100"

submit() {
    local arm="$1"; local wav="$2"; shift 2
    EXP_NAME="turbulence_scalar_orthog_${arm}"
    LABEL="${EXP_NAME}"
    TERMS=("$@")
    WAV_TYPE="${wav}"
    REG_SYSTEM_DIR="${SCRATCH:+${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system/${EXP_NAME}}"
    echo "============================================================"
    echo "Arm ${arm}: wav_type=${wav}, TERMS = ${TERMS[*]}"
    echo "============================================================"
    source "${LAUNCH_DIR}/_submit_turb_sweep.sh"
}

for arm in ${ONLY}; do
    case "${arm}" in
        morletQ1_L6_full)   submit morletQ1_L6_full   morlet  L_6 Scalar_morlet_gaussianK ;;
        morletQ1_noL6_full) submit morletQ1_noL6_full morlet      Scalar_morlet_gaussianK ;;
        shannon_L6_full)    submit shannon_L6_full    shannon L_6 Scalar_morlet_gaussianK ;;
        shannon_noL6_full)  submit shannon_noL6_full  shannon     Scalar_morlet_gaussianK ;;
        *) echo "unknown arm '${arm}'; use *_L6_full | *_noL6_full with morletQ1 or shannon"; exit 1 ;;
    esac
done
