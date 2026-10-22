#!/bin/bash
# Tail-fix counterpart of turbulence_final_noScat4 (2026-10-20).
#
# turbulence_final_noScat4 (M=256, 2 banks, L_6 + Scalar_morlet + Scalar_psi, no Scat4)
# had ch=0 Q=1 q0.9999=0.50 and m2=0.67: tails too narrow. Adding Scat4 (the 'final' arm)
# brought ch=0 Q=1 to q0.9999=0.86 and m2=0.90. This launcher asks: does the tail fix
# applied to both Scalar_* potentials (last cut at q_0.99 of |z_j|, outer alpha in
# [0.5, 3]) match or exceed what Scat4 gives, with no Scat4 terms?
#
# Identical to turbulence_final_noScat4.sh except:
#   --ggd_tail_quantile 0.99 and --ggd_tail_alpha_bounds 0.5 3 propagated to BOTH
#   Scalar_morlet_gaussianK (Q=1 bank) and Scalar_psi_gaussianK (Q=3 bank).
#   CHAIN_L6NEG off by default (not relevant here).
#   Single seed by default (900), as in all first-screen tail-fix runs.
#
# Usage (on Jean Zay, from the repo root):
#   bash turbulence/launch/turbulence_ggd_tailfix_noScat4.sh
#   SEEDS='900 901' bash ...

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="turbulence_ggd_tailfix_noScat4"
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

# ── the tail fix (propagates to every Scalar_*_gaussianK in TERMS) ──
GGD_TAIL_QUANTILE=0.99
GGD_TAIL_ALPHA_BOUNDS="0.5 3"

# noScat4 took ~3.7 h; same here (no extra terms)
TIME="05:00:00"
CPUS=24
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
MODULE_PRE="arch/h100"

source "${LAUNCH_DIR}/_submit_turb_sweep.sh"
