#!/bin/bash
# turbulence_final (M=256, n1=10000, 2 banks, L_6 + Scalar + Scat4) + the scalar fix chosen on
# the coarse runs (2026-10-22):
#   --ggd_tail_quantile 0.99   last cut at q.99 of |z_j|
#   --ggd_tail_mass            <w_K> per channel, outer alpha without scale floor
# Everything else identical to turbulence_final.sh. Controls: final (same without the fix),
# mass_noScat4 (same fix, no Scat4), noScat4.
#
# TIME RISK. turbulence_final projected ~17.9 h of a 20 h job (SDE loop, 99.8 % of it). The mass
# statistic adds ~27 columns to a system of 266 (+10 %); an earlier +18 % in cost per step for a
# similar increase (eps_abs_floor_bug note, 09-28, rough) would put this at ~21 h, above the 19h30
# in-run guard of _submit_turb_sweep.sh (I have not checked what the guard saves when it fires).
# Calibrate before submitting: mass_noScat4 and noScat4 differ only by the fix, so
#   projected = 17.9 h x (SDE time of mass_noScat4 / SDE time of noScat4)
# grep "SDE integration finished" turbulence/experiments/*/*/logs/run.log   (both runs)
# If projected > ~19 h, shorten the bulk, not the tail: the tail keeps n_tail = NT - N_BULK = 50000
# steps (r = 1.52e-4, as turbulence_final), e.g. NT=52000 N_BULK=2000 (bulk step 4.5e-4 instead of
# 1.8e-4: 2.5x coarser bulk, tail untouched). Or lower n1, which also scales the cost linearly:
#   NT=52000 N_BULK=2000 bash turbulence/launch/turbulence_ggd_mass_final.sh
#
# Usage (on Jean Zay, repo root):
#   SEEDS="901 902 903 904" bash turbulence/launch/turbulence_ggd_mass_final.sh
#       chain per seed: SDE -> Theta_reg L_6<0 (lam 5e-4) -> log p per family; + cross-seed lam check
#       (2026-10-30; post checks are seed-900 only and off for other seeds)
#   bash turbulence/launch/turbulence_ggd_mass_final.sh                   # seed 900: + post checks + log p chain
#   CHAIN_LOGP=false bash ...      # post checks only (the L_6<0 Theta_reg solve of M=256 is long)
#   POST_CHECKS=false bash ...     # neither

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="turbulence_ggd_mass_final"
LABEL="${EXP_NAME}"
TERMS=(L_6 Scalar_psi_gaussianK Scalar_morlet_gaussianK
       Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1)
SEED_LIST=(${SEEDS:-900})

# ── as turbulence_final.sh ──
N1=10000
SUBSERIES_LEN=512
TARGET_LEN=256
J=8
Q=3
SIGMA=3.5
NT="${NT:-55000}"
N_BULK="${N_BULK:-5000}"
SCHEDULE_ARGS="--schedule two_phase --n_bulk ${N_BULK} --t_switch 0.9 --gap_end 5e-5"
BATCH_SIZE=5000
REGULARIZATION=1e-4
# the fit is deterministic on x1 and mass_noScat4 got past its fit with this drop, so the slot is active
DROP_STATS="Scalar_morlet_gaussianK:8:1"
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
REG_SYSTEM_DIR="${SCRATCH:+${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system/${EXP_NAME}}"

# ── the fix ──
GGD_TAIL_QUANTILE=0.99
GGD_TAIL_MASS=true

TIME="20:00:00"
CPUS=24
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
MODULE_PRE="arch/h100"

source "${LAUNCH_DIR}/_submit_turb_sweep.sh"

# post checks read seed 900's outputs only: on by default only when seed 900 is in this launch
case " ${SEED_LIST[*]} " in *" 900 "*) POST_DEFAULT=true ;; *) POST_DEFAULT=false ;; esac
if [ "${POST_CHECKS:-${POST_DEFAULT}}" = "true" ]; then
    DEPEND="${JOBID}" ARM=mass_final LABEL="${EXP_NAME}" CONTROLS="final mass_noScat4" \
    PAIRS="mass \+ Scat4 at M=256|Scat4 on top of mass, M=256" \
        bash "${LAUNCH_DIR}/queue_post_checks.sh"
fi

# Theta_reg with L_6<0 (lam 5e-4, ridge 1e-4, L_6 columns 0:9 at J=8) + per-family log p.
# M=256 solves are slower than the M=128 ones: 250-310 s per round on the turbulence_final systems.
# (independent of POST_CHECKS since 2026-10-30)
if [ "${CHAIN_LOGP:-true}" = "true" ]; then
    DEPEND="${JOBID}" SEEDS="${SEED_LIST[*]}" LABEL="${EXP_NAME}" COLS=0:9 TIME=06:00:00 \
        bash "${LAUNCH_DIR}/queue_logp_families.sh"
fi

# Cross-seed lambda check (lam_selection_criterion/turb_xseed.py), chained per seed. Only a CHECK: the L_6<0
# solve above uses lam 5e-4 for every seed, so they stay comparable with seed 900, whose reg system was deleted
# (2026-10-28) and cannot be re-solved at another lam. Needs >= 2 seeds in this launch; 5e-4 is in the grid.
if [ "${CHAIN_XSEED:-true}" = "true" ] && [ ${#SEED_LIST[@]} -ge 2 ]; then
    DEPEND="${JOBID}" SEEDS="${SEED_LIST[*]}" ARM="${EXP_NAME}" RIDGE="${REGULARIZATION}" \
    REG_SYSTEM_DIR="${REG_SYSTEM_DIR}" LAMS="${XSEED_LAMS:-0 1e-6 1e-5 1e-4 5e-4 1e-3 1e-2 1e-1 1}" \
    OUT="${LAUNCH_DIR}/../../lam_selection_criterion/results/turb_xseed" \
        bash "${LAUNCH_DIR}/../../lam_selection_criterion/launch_turb_xseed.sh"
fi
