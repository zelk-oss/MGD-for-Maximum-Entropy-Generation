#!/bin/bash
# Full turbulence_final + tail fix (2026-10-20).
#
# Closes the 2x2 cube:
#       no tail fix        tail fix
#   no Scat4   noScat4         tailfix_noScat4
#   + Scat4    final           tailfix_final   (THIS)
#
# Answers both:
#   (a) does Scat4 still help once the tail fix is in place?
#   (b) does the tail fix still help once Scat4 is in place?
#
# Identical to turbulence_final.sh except:
#   --ggd_tail_quantile 0.99 and --ggd_tail_alpha_bounds 0.5 3 propagated to BOTH
#   Scalar_morlet_gaussianK and Scalar_psi_gaussianK via potentials_builder's
#   ggd_kwargs. Single seed by default (900), paired with turbulence_final seed 900.
#
# Time: ~17.9 h as turbulence_final (same r = 266, same nt, same n1). TIME 20:00:00.
#
# Usage (on Jean Zay, from the repo root):
#   bash turbulence/launch/turbulence_ggd_tailfix_final.sh
#   SEEDS='900 901' bash ...

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="turbulence_ggd_tailfix_final"
TERMS=(L_6 Scalar_psi_gaussianK Scalar_morlet_gaussianK
       Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1)
SEED_LIST=(${SEEDS:-900})

# ── match turbulence_final.sh ──
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

# ── the tail fix (propagates to every Scalar_*_gaussianK in TERMS) ──
GGD_TAIL_QUANTILE=0.99
GGD_TAIL_ALPHA_BOUNDS="0.5 3"

TIME="20:00:00"
CPUS=24
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
MODULE_PRE="arch/h100"

source "${LAUNCH_DIR}/_submit_turb_sweep.sh"
