#!/bin/bash
# M3 (turbulence_nested_M3.sh) with the windows of the scalar potentials fitted by constrained BIC
# (2026-10-27): no window narrower than BIC_C x the noise one SDE step adds to the channel at
# 1 - t = NOISE_REF_GAP (--ggd_bic_min_window, codes/potentials/potentials_1d.py). Narrower windows
# are not held by the SDE corrector (the 10-36% missed moments; psi[6] of mass_L6_2b, [0, 8.1e-5),
# noise > width even at the last step).
#
# Two arms, everything else exactly as M3 (data, schedule, regularisation, tail fix, DROP_STATS,
# Gram pre-flight):
#   bicw          A: constrained windows, usual statistics (power per window + outer-window mass)
#   bicw_allmass  B: A + the mass of every inner window (--ggd_all_masses)
#   bicw_rich     C (2026-10-28): B + up to 6 windows per channel (--ggd_num_regions 6), a second
#                 forced tail cut at q.999 (--ggd_tail_quantile 0.99 0.999), and the same marginal
#                 statistics on the imaginary parts (Scalar_*_gaussianK_imag)
# Reference: turbulence_nested_M3 (unconstrained windows, 5 seeds). Question: with constrained
# windows and more marginal statistics, are the marginals recovered without Scat4? C: does the
# cross-scale gap to the Scat4 run keep closing with even more one-point statistics?
#
# Time (measured, seed 900): M3 4.2 h (r=160), A 4.0 h, B 4.6 h (r=201). C: unmeasured, r from the
# pre-flight; RICH_TIME sets its wall time (the run aborts cleanly at 90% of it). NT / N_BULK
# override the schedule for a short timing job (the config name carries nt, so no clash).
#
# Usage (on Jean Zay, from the repo root):
#   bash turbulence/launch/turbulence_nested_M3_bicw.sh                    # both arms, seed 900
#   ONLY=bicw_allmass SEEDS="900 901 902 903 904" bash ...                  # one arm, 5 seeds
#   PREFLIGHT_ONLY=true bash ...                                            # fit + Gram check only
#   BIC_C=2 bash ...                                                        # wider bound (label gets _C2)

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEED_LIST=(${SEEDS:-900})
ONLY="${ONLY:-bicw bicw_allmass}"
BIC_C="${BIC_C:-1}"
NOISE_REF_GAP="${NOISE_REF_GAP:-1e-2}"
BASE_TERMS=(L_6 Scalar_psi_gaussianK Scalar_morlet_gaussianK)
RICH_TIME="${RICH_TIME:-16:00:00}"

# ── as turbulence_nested_M3.sh ──
N1=10000
SUBSERIES_LEN=512
TARGET_LEN=256
J=8
Q=3
SIGMA=3.5
NT="${NT:-55000}"
SCHEDULE_ARGS="--schedule two_phase --n_bulk ${N_BULK:-5000} --t_switch 0.9 --gap_end 5e-5"
BATCH_SIZE=5000
REGULARIZATION=1e-4
BASE_DROP_STATS="Scalar_morlet_gaussianK:8:1"
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

GGD_TAIL_MASS=true

GRAM_PREFLIGHT="${GRAM_PREFLIGHT:-1e-6}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"

# ── the change: constrained BIC windows ──
GGD_BIC_MIN_WINDOW="${BIC_C}"
GGD_NOISE_REF_GAP="${NOISE_REF_GAP}"

CPUS=24
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
MODULE_PRE="arch/h100"

TAG=""
[ "${BIC_C}" != "1" ] && TAG+="_C${BIC_C}"
[ "${NOISE_REF_GAP}" != "1e-2" ] && TAG+="_gap${NOISE_REF_GAP}"

for arm in ${ONLY}; do
    TERMS=("${BASE_TERMS[@]}"); GGD_TAIL_QUANTILE=0.99; GGD_NUM_REGIONS=""; DROP_STATS="${BASE_DROP_STATS}"
    case "${arm}" in
        bicw)         GGD_ALL_MASSES="";     TIME="06:00:00" ;;
        bicw_allmass) GGD_ALL_MASSES=true;   TIME="10:00:00" ;;
        bicw_rich)    GGD_ALL_MASSES=true;   TIME="${RICH_TIME}"
                      GGD_NUM_REGIONS=6;     GGD_TAIL_QUANTILE="0.99 0.999"
                      TERMS+=(Scalar_psi_gaussianK_imag Scalar_morlet_gaussianK_imag)
                      # morlet ch 8 window 1 was a near-copy for the 4-window cuts; with 6 windows
                      # the slot need not exist (--drop_stats fails then): the Gram pre-flight
                      # removes near-copies instead
                      DROP_STATS="" ;;
        *) echo "unknown arm '${arm}'; use bicw | bicw_allmass | bicw_rich"; exit 1 ;;
    esac
    [ "${PREFLIGHT_ONLY}" = "true" ] && TIME="00:30:00"
    EXP_NAME="turbulence_nested_M3_${arm}${TAG}"
    LABEL="${EXP_NAME}"
    REG_SYSTEM_DIR="${SCRATCH:+${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system/${EXP_NAME}}"
    echo "============================================================"
    echo "Arm ${arm}: windows >= ${BIC_C} x step noise at 1-t=${NOISE_REF_GAP}, all masses: ${GGD_ALL_MASSES:-no}"
    source "${LAUNCH_DIR}/_submit_turb_sweep.sh"
done
