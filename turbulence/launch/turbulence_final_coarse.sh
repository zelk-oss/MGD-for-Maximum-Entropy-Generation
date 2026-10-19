#!/bin/bash
# Fast coarse-grained diagnostic (2026-10-19): is L_2_lowpass still needed on the
# turbulence_final setup minus Scattering_Fourth_Order? Compare two 1-seed arms side by side.
#
# Baseline (26-09) found: without BOTH L_2_lowpass AND Scalar_morlet the marginal goes bimodal.
# With only Scalar_morlet removed the physics was clearly damaged; with only L_2_lowpass removed
# the physics was indistinguishable from full. BUT that test ran at reg1e-4 on a different setup
# (L_6_psi still in, no drop_stats, no dedup_tol, no eps_quantile, n1 8500, nt 60k). We want to
# repeat the single-term removal on the CURRENT pipeline, which is why this diagnostic exists.
#
# Shared with turbulence_final: Q=3 wavelet bank, ridge 1e-4, two_phase schedule, dedup 1e-8,
# solve_float64. L_6_psi kept out (as in turbulence_final). Scat4 removed from BOTH arms (we
# only want the L_2_lowpass contrast, not an interaction with Scat4; turbulence_final_noScat4.sh
# already covers that one).
#
# Coarse graining: target_len = 128 (not 64) on purpose. L_2_lowpass's filter is the J-th scale
# of the Q=3 bank: at target_len = 64 with J = 6 its support covers ~half the signal, i.e. it
# collapses onto plain variance and the test would trivially say "L_2_lowpass redundant". At
# target_len = 128 with J = 7 the lowpass filter is still ~2 Fourier bins wide, so it keeps a
# distinct band to constrain.
#
# Both arms share:
#   target_len 128, J 7, Q 3, sigma 3.5
#   n1 3000, nt 15000, two_phase with n_bulk 1000, t_switch 0.9, gap_end 5e-5
#   seed 900, ridge 1e-4, lam 0.0, --skip_reg_solve (no Theta_reg; the diagnostic lives in the
#   samples: marginal, spectrum, flat6, wav_KS; theta_t is enough for moment matching).
#
# WITH_SCAT4=true appends Scattering_Fourth_Order_Mod2_{Real,Imag}_Q1 to BOTH arms and tags the
# label '_scat4', so the saved files don't collide with the no-Scat4 version. Default: no Scat4.
#
# Time: estimated 1-2 h per arm without Scat4; +30-60% with Scat4 (its gradient builds complex
# (B, 8, 8, 9, 256) tensors ~1.1 MB/sample, 6 held at once -> ~20 GB at batch 3000, fits H100).
# 04:00:00 asked for safety; default QoS (not dev, which caps at 2 h).
#
# Usage (from the repo root on Jean Zay):
#   bash turbulence/launch/turbulence_final_coarse.sh                            # both arms, no Scat4
#   WITH_SCAT4=true bash turbulence/launch/turbulence_final_coarse.sh            # both arms, with Scat4
#   ONLY=noL2 WITH_SCAT4=true bash turbulence/launch/turbulence_final_coarse.sh  # one arm only (full|noL2)

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WITH_SCAT4="${WITH_SCAT4:-false}"
SCAT4_TAG=""
SCAT4_TERMS=()
if [ "${WITH_SCAT4}" = "true" ]; then
    SCAT4_TAG="_scat4"
    SCAT4_TERMS=(Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1)
fi

FULL=(L_6 L_2_lowpass Scalar_psi_gaussianK Scalar_morlet_gaussianK "${SCAT4_TERMS[@]}")
NO_L2=(L_6                  Scalar_psi_gaussianK Scalar_morlet_gaussianK "${SCAT4_TERMS[@]}")

SEED_LIST=(${SEEDS:-900})
ONLY="${ONLY:-full noL2}"

# ── shared knobs ─────────────────────────────────────
N1=3000
SUBSERIES_LEN=512
TARGET_LEN=128
J=7
Q=3
SIGMA=3.5
NT=15000
SCHEDULE_ARGS="--schedule two_phase --n_bulk 1000 --t_switch 0.9 --gap_end 5e-5"
BATCH_SIZE=3000
REGULARIZATION=1e-4
COND_EVERY=10

SOLVE_FLOAT64=true
DEDUPLICATE_FILTERS=true
DEDUP_TOL=1e-8

LAM=0.0
N_SUBSAMPLE=1
REG_SOLVER=thomas
REG_RIDGE=0.0
SAVE_REG_SYSTEM=true
SKIP_REG_SOLVE=true

TIME="04:00:00"
CPUS=24
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
MODULE_PRE="arch/h100"

# ── submit helper: picks TERMS and labels; sources the shared launcher ──
submit() {
    local label="$1"; shift
    EXP_NAME="turbulence_final_coarse_${label}${SCAT4_TAG}"
    LABEL="${EXP_NAME}"
    TERMS=("$@")
    REG_SYSTEM_DIR="${SCRATCH:+${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system/${EXP_NAME}}"
    echo "============================================================"
    echo "Arm ${label}: TERMS = ${TERMS[*]}"
    echo "============================================================"
    source "${LAUNCH_DIR}/_submit_turb_sweep.sh"
}

for arm in ${ONLY}; do
    case "${arm}" in
        full) submit full  "${FULL[@]}"  ;;
        noL2) submit noL2  "${NO_L2[@]}" ;;
        *) echo "unknown arm '${arm}'; use full | noL2"; exit 1 ;;
    esac
done
