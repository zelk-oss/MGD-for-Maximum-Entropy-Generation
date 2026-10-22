#!/bin/bash
# Scalar-only sanity check (2026-10-20): can per-channel scalar moments alone reproduce
# the fine-band wavelet marginals, if we switch to an orthogonal filter bank?
#
# Context: in turbulence_final_noScat4 (= turbulence_final minus Scat4) the generated
# wavelet-coefficient histograms on the fine bands (j=0,1,2 of hist_plot Morlet Q=3)
# have tails 2-4x narrower than the data, even though
#   - the Scalar_psi/morlet_gaussianK moments on those channels ARE matched (<= 10^-8),
#   - L_6 is matched too,
#   - only 10/124 moments > 0.01, all in the CORE/intermediate regions of the finest
#     Q=3 channels (f_peak ~0.25-0.49), not in the tail regions.
# Scat4 fixes it indirectly via inter-scale correlations. We want to know whether the
# problem is a MaxEnt limit of "K=3-4 scalar moments per channel", or whether it goes
# away with an orthogonal bank (disjoint Fourier support, no near-copies).
#
# Four arms, scalar potentials only (no Scat4, no L_2_lowpass, no Scalar_psi Q=3 bank):
#   morletQ1_L6     : L_6 + Scalar_morlet_gaussianK, morlet Q=1
#   morletQ1_noL6   :        Scalar_morlet_gaussianK, morlet Q=1
#   shannon_L6      : L_6 + Scalar_morlet_gaussianK, shannon Q=1 (orthogonal in Fourier)
#   shannon_noL6    :        Scalar_morlet_gaussianK, shannon Q=1
# Scalar_morlet_gaussianK uses the Q=1 bank (`filters`), so switching the bank via
# --wav_type substitutes it entirely; same binding for L_6.
# Rationale for including/excluding L_6:
#   L_6 is a 6th-order constraint per channel, tail-dominated. With it we test "L_6 does
#   the tail work the scalar_* does not". Without, we isolate the scalar potential alone.
#
# Coarse setup, same as turbulence_final_coarse: target_len=128, J=7, n1=3000, nt=15000,
# two_phase schedule. 1 seed per arm (900) suffices for a first screen; add seeds later
# if an arm looks promising.
#
# Expected cost: ~1-1.5 h per arm on H100 (fewer stats than turbulence_final_coarse_full,
# which took ~1 h). TIME 03:00:00 for safety.
#
# Dedup OFF: shannon has disjoint Fourier support (no near-copies by construction);
# morlet Q=1 has at most one near-duplicate (filter 7 vs 8), negligible for a diagnostic.
#
# Usage (from the repo root on Jean Zay):
#   bash turbulence/launch/turbulence_scalar_orthog.sh                       # all four
#   ONLY='shannon_L6 shannon_noL6' bash ...turbulence_scalar_orthog.sh       # subset
#   SEEDS='900 901' bash ...turbulence_scalar_orthog.sh                      # extra seeds

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEED_LIST=(${SEEDS:-900})
ONLY="${ONLY:-morletQ1_L6 morletQ1_noL6 shannon_L6 shannon_noL6}"

# ── shared knobs (same as turbulence_final_coarse) ───
N1=3000
SUBSERIES_LEN=512
TARGET_LEN=128
J=7
Q=1                     # single bank only; Scalar_psi (which uses filters_Q) is NOT in TERMS
SIGMA=3.5
NT=15000
SCHEDULE_ARGS="--schedule two_phase --n_bulk 1000 --t_switch 0.9 --gap_end 5e-5"
BATCH_SIZE=3000
REGULARIZATION=1e-4
COND_EVERY=10

SOLVE_FLOAT64=true
DEDUPLICATE_FILTERS=false        # shannon disjoint by construction; morlet Q=1 few/no copies

LAM=0.0
N_SUBSAMPLE=1
REG_SOLVER=thomas
REG_RIDGE=0.0
SAVE_REG_SYSTEM=true
SKIP_REG_SOLVE=true

TIME="03:00:00"
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
        morletQ1_L6)   submit morletQ1_L6   morlet  L_6 Scalar_morlet_gaussianK ;;
        morletQ1_noL6) submit morletQ1_noL6 morlet      Scalar_morlet_gaussianK ;;
        shannon_L6)    submit shannon_L6    shannon L_6 Scalar_morlet_gaussianK ;;
        shannon_noL6)  submit shannon_noL6  shannon     Scalar_morlet_gaussianK ;;
        *) echo "unknown arm '${arm}'; use morletQ1_L6 | morletQ1_noL6 | shannon_L6 | shannon_noL6"; exit 1 ;;
    esac
done
