#!/bin/bash
# Mass fix at full resolution without Scat4 (2026-10-22).
#
# turbulence_final_noScat4 (M=256, n1=10000, 2 banks, L_6 + Scalar_morlet + Scalar_psi)
# + the fix chosen on the coarse runs (mass_L6_2b, turbulence_ggd_mass_2bank.sh):
#   --ggd_tail_quantile 0.99   last cut at q.99 of |z_j|
#   --ggd_tail_mass            <w_K> per channel, outer alpha without scale floor
# Question: at higher resolution and with more particles (M 128 -> 256, n1 3000 -> 10000,
# finer tail steps), do the marginals get better than on the coarse run? Controls:
# noScat4 (same run without the fix) and final (with Scat4, without the fix).
#
# Identical to turbulence_ggd_tailfix_noScat4.sh except the fix (mass, no alpha bounds).
# DROP_STATS kept as in noScat4 for comparability: if the new fit leaves morlet ch 8
# window 1 inactive, the fit stops with "drop_slots ... are not active" in the first minutes.
#
# Usage (on Jean Zay, from the repo root):
#   bash turbulence/launch/turbulence_ggd_mass_noScat4.sh            # + post checks
#   POST_CHECKS=false bash turbulence/launch/turbulence_ggd_mass_noScat4.sh

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="turbulence_ggd_mass_noScat4"
LABEL="${EXP_NAME}"
TERMS=(L_6 Scalar_psi_gaussianK Scalar_morlet_gaussianK)
SEED_LIST=(${SEEDS:-900})

# ── match turbulence_final_noScat4 ──
N1=10000
SUBSERIES_LEN=512
TARGET_LEN=256
J=8
Q=3
SIGMA=3.5
NT=55000
SCHEDULE_ARGS="--schedule two_phase --n_bulk 5000 --t_switch 0.9 --gap_end 5e-5"
BATCH_SIZE=5000
REGULARIZATION=1e-4
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

# ── the fix (propagates to every Scalar_*_gaussianK in TERMS) ──
GGD_TAIL_QUANTILE=0.99
GGD_TAIL_MASS=true

# noScat4 took ~3.7 h; the mass statistic adds ~1 statistic per channel
TIME="06:00:00"
CPUS=24
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
MODULE_PRE="arch/h100"

source "${LAUNCH_DIR}/_submit_turb_sweep.sh"

if [ "${POST_CHECKS:-true}" = "true" ]; then
    DEPEND="${JOBID}" ARM=mass_noScat4 LABEL="${EXP_NAME}" CONTROLS="noScat4 final" \
    PAIRS="mass at M=256" bash "${LAUNCH_DIR}/queue_post_checks.sh"
fi
