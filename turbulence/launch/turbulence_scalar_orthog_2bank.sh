#!/bin/bash
# Coarse (M=128) + two-bank counterpart of turbulence_scalar_orthog.sh (2026-10-20).
#
# Fills the fourth cell of the 2x2 design (resolution x #banks):
#   M=128, 1 bank  (Q=1 only)  = turbulence_scalar_orthog.sh         (code matchate)
#   M=256, 1 bank  (Q=1 only)  = turbulence_scalar_orthog_full.sh    (running)
#   M=128, 2 banks (Q=1 + Q=3) = THIS LAUNCHER                       (to run)
#   M=256, 2 banks             = turbulence_final_noScat4            (code brutte)
#
# If 2 banks matter even at M=128 -> the multi-bank competition on the Thomas solver
# with ridge is the limiting factor; adding Q=3 to the model dilutes the per-band
# constraints enough to lose the fine-band tails.
# If 2 banks don't matter at M=128 -> the problem of noScat4 is resolution, not banks:
# at M=256 fine-scale intermittency needs more constraints (K>>4 or Scat4) regardless.
#
#   morletQ1Q3_L6_2b    : L_6 + Scalar_morlet_gaussianK + Scalar_psi_gaussianK, morlet
#   morletQ1Q3_noL6_2b  :        Scalar_morlet_gaussianK + Scalar_psi_gaussianK, morlet
#   shannonQ1Q3_L6_2b   : L_6 + Scalar_morlet_gaussianK + Scalar_psi_gaussianK, shannon
#   shannonQ1Q3_noL6_2b :        Scalar_morlet_gaussianK + Scalar_psi_gaussianK, shannon
# (Scalar_morlet_gaussianK uses the Q=1 bank 'filters'; Scalar_psi_gaussianK uses the
# Q=3 bank 'filters_Q'; both derived from the same --wav_type.)
#
# Dedup OFF deliberately: for shannon Q=3 the bands are disjoint (no copies anyway);
# for morlet Q=3 keeping the near-copies lets us test the "two-bank effect" clean of
# any dedup confound; the ridge 1e-4 keeps the Gram tractable.
#
# Time: 1 extra family than scalar_orthog (which took ~3.5 min / arm), ~5-6 min / arm.
# TIME 00:30:00 dev QoS.
#
# Usage (from the repo root on Jean Zay):
#   bash turbulence/launch/turbulence_scalar_orthog_2bank.sh
#   ONLY='shannonQ1Q3_L6_2b shannonQ1Q3_noL6_2b' bash ...

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEED_LIST=(${SEEDS:-900})
ONLY="${ONLY:-morletQ1Q3_L6_2b morletQ1Q3_noL6_2b shannonQ1Q3_L6_2b shannonQ1Q3_noL6_2b}"

# ── coarse knobs (match turbulence_scalar_orthog.sh except Q=3) ──
N1=3000
SUBSERIES_LEN=512
TARGET_LEN=128
J=7
Q=3                     # Q=3 for filters_Q (used by Scalar_psi_gaussianK)
SIGMA=3.5
NT=15000
SCHEDULE_ARGS="--schedule two_phase --n_bulk 1000 --t_switch 0.9 --gap_end 5e-5"
BATCH_SIZE=3000
REGULARIZATION=1e-4
COND_EVERY=10

SOLVE_FLOAT64=true
DEDUPLICATE_FILTERS=true         # shannon Q=3 at J=7 produces zero-norm sub-bands
DEDUP_TOL=1e-8                   # below the FFT resolution 1/128; dedup drops them.
                                 # For morlet the drop is a no-op (bank well-defined).

LAM=0.0
N_SUBSAMPLE=1
REG_SOLVER=thomas
REG_RIDGE=0.0
SAVE_REG_SYSTEM=true
SKIP_REG_SOLVE=true

TIME="00:30:00"
CPUS=24
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
MODULE_PRE="arch/h100"
QOS="qos_gpu_h100-dev"

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
        morletQ1Q3_L6_2b)    submit morletQ1Q3_L6_2b    morlet  L_6 Scalar_morlet_gaussianK Scalar_psi_gaussianK ;;
        morletQ1Q3_noL6_2b)  submit morletQ1Q3_noL6_2b  morlet      Scalar_morlet_gaussianK Scalar_psi_gaussianK ;;
        shannonQ1Q3_L6_2b)   submit shannonQ1Q3_L6_2b   shannon L_6 Scalar_morlet_gaussianK Scalar_psi_gaussianK ;;
        shannonQ1Q3_noL6_2b) submit shannonQ1Q3_noL6_2b shannon     Scalar_morlet_gaussianK Scalar_psi_gaussianK ;;
        *) echo "unknown arm '${arm}'"; exit 1 ;;
    esac
done
