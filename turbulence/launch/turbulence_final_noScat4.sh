#!/bin/bash
# Ablation of the final turbulence runs (2026-10-16): no fourth-order scattering terms.
#
# Question: do the typical / rare events of the model change when the order-4 terms
# (Scattering_Fourth_Order_Mod2_Real_Q1, Scattering_Fourth_Order_Mod2_Imag_Q1) are removed?
#
# Identical to turbulence_final.sh (n1 10000, nt 55000 two_phase n_bulk 5000, ridge 1e-4,
# drop Scalar_morlet_gaussianK:8:1, dedup 1e-8) except:
#   TERMS   the two Scattering_Fourth_Order_* terms removed. Removing columns cannot create a
#           new near-copy pair (pairwise corr of the remaining stats is unchanged) and cannot
#           raise the Gram condition number (eigenvalue interlacing), so the turbulence_final
#           pre-flight (preflight_dedupe_data.py) still covers this set.
#   SEEDS   900-904 by default: the same seeds as turbulence_final, so the comparison is paired.
#   reg_system saved to its own SCRATCH subfolder (turbulence_final_noScat4).
#
# CHAIN_L6NEG=true (default) also queues the offline Theta_reg with L_6 < 0
# (theta_sign_constraint/launch_l6neg.sh) at the same settings as the final 30-seed run
# (job 1062453): lam 5e-4, ridge 1e-4, eps 1e-2 nats/sd, columns 0:9 (L_6 is still the first
# term, 9 filters). Task k of that array starts only when task k (same seed) of this SDE array
# has finished OK (aftercorr); if an SDE task fails, its Theta_reg task is cancelled.
#
# Time: fewer terms than turbulence_final (r = 265 there), so <= its ~17.9 h; Theta_reg L_6 < 0
# took 3.5-8.5 h per seed at r = 265.
#
# Usage (on Jean Zay, from the repo root):
#   bash turbulence/launch/turbulence_final_noScat4.sh
#   CHAIN_L6NEG=false bash turbulence/launch/turbulence_final_noScat4.sh   # SDE only

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="turbulence_final_noScat4"
TERMS=(L_6 Scalar_psi_gaussianK Scalar_morlet_gaussianK)
SEED_LIST=(${SEEDS:-900 901 902 903 904})

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

TIME="20:00:00"
CPUS=24
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
MODULE_PRE="arch/h100"

source "${LAUNCH_DIR}/_submit_turb_sweep.sh"

if [ "${CHAIN_L6NEG:-true}" = "true" ]; then
    echo "Queueing Theta_reg L_6 < 0 after job ${JOBID} (task by task)"
    DEPEND="${JOBID}" SEEDS="${SEED_LIST[*]}" ARM="${EXP_NAME}" RIDGE=1e-4 LAMS=5e-4 EPS=1e-2 COLS=0:9 \
    REG_SYSTEM_DIR="${REG_SYSTEM_DIR}" \
        bash "${LAUNCH_DIR}/../../theta_sign_constraint/launch_l6neg.sh"
fi
