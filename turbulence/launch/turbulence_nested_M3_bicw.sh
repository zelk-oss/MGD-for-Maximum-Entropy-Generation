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
# pre-flight; RICH_TIME sets its wall time (the SDE loop aborts cleanly once its projection exceeds RICH_TIME
# minus min(30 min, 10%), codes/sde_routines.py:loop_budget_s; 18 h -> 17h30). NT / N_BULK
# override the schedule for a short timing job (the config name carries nt, so no clash).
#
# Usage (on Jean Zay, from the repo root):
#   bash turbulence/launch/turbulence_nested_M3_bicw.sh                    # both arms, seed 900
#   ONLY=bicw_allmass SEEDS="900 901 902 903 904" bash ...                  # one arm, 5 seeds
#   PREFLIGHT_ONLY=true bash ...                                            # fit + Gram check only
#   BIC_C=2 bash ...                                                        # wider bound (label gets _C2)
#   ONLY=bicw_rich RICH_TIME=18:00:00 CPUS=48 NGPUS=2 bash ...              # model C, full grid (~240 GB RAM)
#   ONLY=bicw_allmass_scat4 NT=6000 N_BULK=1000 SCAT4_TIME=02:00:00 QOS=qos_gpu_h100-dev bash ...  # B+Scat4 timing
#   ONLY=bicw_allmass_scat4 SEEDS="900 901 902 903 904" bash ...            # B+Scat4, 5 seeds (~18.6 h each)
#   ONLY=bicw_allmass_scat4 SIGMA=6 NT=27500 N_BULK=2500 POTENTIALS_FROM=<run>/fitted_potentials bash ...
#                                     # sigma x nt check of the entropy bound (2026-11-10), windows frozen

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEED_LIST=(${SEEDS:-900})
ONLY="${ONLY:-bicw bicw_allmass}"
BIC_C="${BIC_C:-1}"
NOISE_REF_GAP="${NOISE_REF_GAP:-1e-2}"
BASE_TERMS=(L_6 Scalar_psi_gaussianK Scalar_morlet_gaussianK)
RICH_TIME="${RICH_TIME:-16:00:00}"
# bicw_allmass_scat4: mass_final (same Scat4 terms) took 18.2 h SDE. This arm, 2026-11-01: 1.22 s/it
# steady to step 4300, then a node slowdown aborted job 1812104 (NT=52000) at 1.5 h; guard window widened
# (sde_routines._check_time_budget) -> default NT=55000, ~18.6 h. (+ Scat3 Real/Imag on Q=3: 2.03 s/it
# = 31 h at 55000; dropped.)
SCAT4_TIME="${SCAT4_TIME:-20:00:00}"

# ── as turbulence_nested_M3.sh ──
N1=10000
SUBSERIES_LEN=512
TARGET_LEN=256
J=8
Q=3
SIGMA="${SIGMA:-3.5}"
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
# Theta_reg system stored every N_SUBSAMPLE steps (block average). bicw_rich at 1: r=522 x 55000 steps
# x (M + Sigma) float32 = 120 GB host RAM -> OOM (job 1670470, 2026-10-29); N_SUBSAMPLE=2 halves it.
N_SUBSAMPLE="${N_SUBSAMPLE:-1}"
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

# host RAM = 5000 MB per requested CPU (sacct 1670470: 24 CPUs -> ReqMem 120000M). bicw_rich at
# N_SUBSAMPLE=1 needs ~120 GB for the Theta_reg system alone: CPUS=48 -> ~240 GB.
CPUS="${CPUS:-24}"
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
MODULE_PRE="arch/h100"

TAG=""
[ "${BIC_C}" != "1" ] && TAG+="_C${BIC_C}"
[ "${NOISE_REF_GAP}" != "1e-2" ] && TAG+="_gap${NOISE_REF_GAP}"
# POTENTIALS_FROM (fitted_potentials/ dir of another run): reuse its windows instead of refitting. Needed when
# sigma or nt change: the constrained-BIC window bound uses the run's own step noise, so a refit would change the
# statistics (and the max-ent model). Label gets _pf so these runs are told apart from fitted ones.
[ -n "${POTENTIALS_FROM:-}" ] && TAG+="_pf"

for arm in ${ONLY}; do
    TERMS=("${BASE_TERMS[@]}"); GGD_TAIL_QUANTILE=0.99; GGD_NUM_REGIONS=""; DROP_STATS="${BASE_DROP_STATS}"
    case "${arm}" in
        bicw)         GGD_ALL_MASSES="";     TIME="06:00:00" ;;
        bicw_allmass) GGD_ALL_MASSES=true;   TIME="10:00:00" ;;
        bicw_allmass_scat4)
                      # B + Scat4 (same two terms as turbulence_ggd_mass_final): the Scat4 model with
                      # matched moments. B vs this = Scat4 alone; mass_final vs this = windows + masses
                      GGD_ALL_MASSES=true;   TIME="${SCAT4_TIME}"
                      TERMS+=(Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1) ;;
        bicw_rich)    GGD_ALL_MASSES=true;   TIME="${RICH_TIME}"
                      GGD_NUM_REGIONS=6;     GGD_TAIL_QUANTILE="0.99 0.999"
                      TERMS+=(Scalar_psi_gaussianK_imag Scalar_morlet_gaussianK_imag)
                      # morlet ch 8 window 1 was a near-copy for the 4-window cuts; with 6 windows
                      # the slot need not exist (--drop_stats fails then): the Gram pre-flight
                      # removes near-copies instead
                      DROP_STATS="" ;;
        *) echo "unknown arm '${arm}'; use bicw | bicw_allmass | bicw_allmass_scat4 | bicw_rich"; exit 1 ;;
    esac
    [ "${PREFLIGHT_ONLY}" = "true" ] && TIME="00:30:00"
    EXP_NAME="turbulence_nested_M3_${arm}${TAG}"
    LABEL="${EXP_NAME}"
    REG_SYSTEM_DIR="${SCRATCH:+${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system/${EXP_NAME}}"
    echo "============================================================"
    echo "Arm ${arm}: windows >= ${BIC_C} x step noise at 1-t=${NOISE_REF_GAP}, all masses: ${GGD_ALL_MASSES:-no}"
    source "${LAUNCH_DIR}/_submit_turb_sweep.sh"
done
