#!/bin/bash
# Outer-region mass statistic for Scalar_*_gaussianK, coarse two-bank arm (2026-10-21).
#
# Same configuration as turbulence_scalar_orthog_2bank.sh (morlet, M=128, J=7, Q=3,
# n1=3000, nt=15000, dedup 1e-8, seed 900), with:
#   --ggd_tail_quantile 0.99   outermost cut at q_0.99 of |z_j|
#   --ggd_tail_mass            + <w_K(|z|)> per channel; outer alpha fitted without
#                               scale floor (codes/potentials/potentials_1d.py)
# Controls (already run, production fit): turbulence_scalar_orthog_morletQ1Q3_{noL6,L6}_2b.
#
# What it tests: turbulence/check_fit_offline.py gives the per-channel MaxEnt target with
# channels solved independently. This run includes the coupling between channels (Q=3 is
# overcomplete). Same q.9999 here as offline -> coupling negligible; lower on Q=3 only ->
# coupling matters there.
#
# Usage (repo root on Jean Zay):
#   bash turbulence/launch/turbulence_ggd_mass_2bank.sh               # both mass arms
#   ONLY='mass_noL6_2b' bash turbulence/launch/turbulence_ggd_mass_2bank.sh
#   ONLY='tailq_noL6_2b' bash turbulence/launch/turbulence_ggd_mass_2bank.sh
#   ONLY='tailq_L6_2b'   bash turbulence/launch/turbulence_ggd_mass_2bank.sh
#   ONLY='mass_L6_1b'    bash turbulence/launch/turbulence_ggd_mass_2bank.sh
#   GPU=h100 ONLY=...    bash ...      # default GPU is v100 (gpu_p13)
#
# mass_L6_1b: ONE bank (Q=1, no dedup, as turbulence_scalar_orthog.sh arm morletQ1_L6, its
# control), L_6 + Scalar_morlet + mass. Here the offline per-channel MaxEnt target
# (check_fit_offline.py --with_l6) is exact: q.9999 1.03 / 1.05 / 0.97 on ch 0-2. In the
# two-bank run MGD reached 0.70 / 0.83 / 1.03. Reaching ~1 here -> the gap comes from the
# coupling with the Q=3 bank; staying at ~0.7 -> it comes from the transport.
#
# tailq_L6_2b: tail_quantile only, with L_6 -> against mass_L6_2b it isolates the mass
# statistic in the configuration we keep (L_6 is required: mass_noL6_2b blew up).
#
# tailq_noL6_2b: tail_quantile only, NO mass statistic. Against the control it measures
# what forcing the last cut does alone (it also revives regions the BIC left dead:
# Scalar_psi dead slots 25 -> 10); against mass_noL6_2b it isolates the mass statistic.

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEED_LIST=(${SEEDS:-900})
ONLY="${ONLY:-mass_noL6_2b mass_L6_2b}"

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

GGD_TAIL_QUANTILE=0.99
GGD_TAIL_MASS=true

# Hardware. The morlet 2b controls and the first mass arms ran on H100 in ~11 min; a
# V100 is ~2-3x slower on this workload (float64 solves are fine on V100), so 1 h.
GPU="${GPU:-v100}"
case "${GPU}" in
    v100) TIME="01:00:00"; CPUS=10; ACCOUNT="wbg@v100"; CONSTRAINT="v100"
          PARTITION="gpu_p13"; MODULE_PRE="" ;;
    h100) TIME="00:30:00"; CPUS=24; ACCOUNT="wbg@h100"; CONSTRAINT="h100"
          PARTITION="gpu_p6"; MODULE_PRE="arch/h100" ;;
    *) echo "GPU must be v100 or h100"; exit 1 ;;
esac

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
        mass_noL6_2b)  Q=3; DEDUPLICATE_FILTERS=true;  GGD_TAIL_MASS=true;  submit mass_noL6_2b      Scalar_morlet_gaussianK Scalar_psi_gaussianK ;;
        mass_L6_2b)    Q=3; DEDUPLICATE_FILTERS=true;  GGD_TAIL_MASS=true;  submit mass_L6_2b    L_6 Scalar_morlet_gaussianK Scalar_psi_gaussianK ;;
        tailq_noL6_2b) Q=3; DEDUPLICATE_FILTERS=true;  GGD_TAIL_MASS=false; submit tailq_noL6_2b     Scalar_morlet_gaussianK Scalar_psi_gaussianK ;;
        tailq_L6_2b)   Q=3; DEDUPLICATE_FILTERS=true;  GGD_TAIL_MASS=false; submit tailq_L6_2b   L_6 Scalar_morlet_gaussianK Scalar_psi_gaussianK ;;
        mass_L6_1b)    Q=1; DEDUPLICATE_FILTERS=false; GGD_TAIL_MASS=true;  submit mass_L6_1b    L_6 Scalar_morlet_gaussianK ;;
        mass_L6_2b_drop)
            # mass_L6_2b minus the 15 windows narrower than 10x the noise of the last SDE step
            # (turbulence/window_vs_step_noise.py on mass_L6_2b seed 900, 2026-10-22); the fit
            # is deterministic on x1, so the (channel, window) pairs are the same.
            DROP_STATS="$(printf 'Scalar_psi_gaussianK:%s ' 0:0 1:0 2:0 3:0 4:0 5:0 6:0 0:1 1:1 2:1 3:1 0:2)"
            DROP_STATS+="$(printf 'Scalar_morlet_gaussianK:%s ' 0:0 0:1 1:0)"
            Q=3; DEDUPLICATE_FILTERS=true;  GGD_TAIL_MASS=true;  submit mass_L6_2b_drop L_6 Scalar_morlet_gaussianK Scalar_psi_gaussianK
            DROP_STATS="" ;;
        mass_L6_2b_merge)
            # mass_L6_2b with windows narrower than 10x the noise of the last SDE step merged
            # in the fit (Scalar_GGD_KRegion min_window); mass_L6_2b_drop removed them instead
            # and left the bulk of the finest channels unconstrained (Q=3 ch0 q.9999 1.77).
            GGD_MIN_WINDOW_NOISE=10
            Q=3; DEDUPLICATE_FILTERS=true;  GGD_TAIL_MASS=true;  submit mass_L6_2b_merge L_6 Scalar_morlet_gaussianK Scalar_psi_gaussianK
            GGD_MIN_WINDOW_NOISE="" ;;
        mass_L6_2b_scat4)
            # mass_L6_2b + Scat4 (Real/Imag Q1). Control without the fix:
            # turbulence_final_coarse_noL2_scat4 (turbulence_final_coarse.sh, same terms),
            # which took ~35 min on H100 (Scat4 is the slow part).
            TIME0="${TIME}"; [ "${GPU}" = "h100" ] && TIME="01:30:00" || TIME="03:00:00"
            Q=3; DEDUPLICATE_FILTERS=true;  GGD_TAIL_MASS=true
            submit mass_L6_2b_scat4 L_6 Scalar_morlet_gaussianK Scalar_psi_gaussianK \
                   Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1
            TIME="${TIME0}"
            if [ "${POST_CHECKS:-true}" = "true" ]; then
                DEPEND="${JOBID}" ARM=mass_L6_2b_scat4 LABEL=turbulence_ggd_mass_L6_2b_scat4 \
                CONTROLS="mass_L6_2b coarse_noL2_scat4" PAIRS="Scat4" \
                bash "${LAUNCH_DIR}/queue_post_checks.sh"
            fi ;;
        *) echo "unknown arm '${arm}'; use mass_noL6_2b | mass_L6_2b | tailq_noL6_2b | tailq_L6_2b | mass_L6_1b | mass_L6_2b_drop | mass_L6_2b_merge | mass_L6_2b_scat4"; exit 1 ;;
    esac
done
