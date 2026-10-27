#!/bin/bash
# Nested-model hierarchy, side arm M2-Db4 (2026-10-23): M2 with the scalar potential on the critically
# decimated, periodic, orthonormal Daubechies-4 DWT (codes/ortho_wavelet/periodic_dwt.py) instead of the
# Morlet bank: d = 256 coefficients (J = 8 detail scales + the approximation), one set of scalar statistics per
# scale. Compared with M2 (turbulence_nested_M2.sh). Plan: ~/.claude/plans/hashed-noodling-llama.md
#
# IDENTICAL to turbulence_nested_M2.sh (data, schedule, regularisation, scalar fix, pre-flight, seeds, float64
# solve, reg system, L_6 on the Q=1 Morlet bank) except:
#   Scalar_dwt_gaussianK + --dwt_order 4   instead of Scalar_morlet_gaussianK
#   no DROP_STATS                          the M2 slot (Scalar_morlet:8:1) is a Morlet statistic, absent here
# Note: L_6 stays on the Morlet bank (user's choice), so the model is not a strict product over the DWT
# coefficients; the scalar part alone is.
#
# Pre-flight: expected to drop nothing (the DWT Gram is block-diagonal across scales); a drop would be a red
# flag, read gram_preflight.json. Time: unmeasured (M2 is ~1.5 h per seed); 3 h request, re-estimate from the
# first minutes of the log.
#
# Usage (on Jean Zay, from the repo root):
#   PREFLIGHT_ONLY=true bash turbulence/launch/turbulence_nested_M2_db4.sh   # fit + Gram check, no SDE
#   bash turbulence/launch/turbulence_nested_M2_db4.sh                       # 5 seeds, 900-904

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="turbulence_nested_M2_db4"
LABEL="${EXP_NAME}"
TERMS=(L_6 Scalar_dwt_gaussianK)
DWT_ORDER=4
SEED_LIST=(${SEEDS:-900 901 902 903 904})

# ── as turbulence_final.sh, Q=1 ──
N1=10000
SUBSERIES_LEN=512
TARGET_LEN=256
J=8
Q=1
SIGMA=3.5
NT=55000
SCHEDULE_ARGS="--schedule two_phase --n_bulk 5000 --t_switch 0.9 --gap_end 5e-5"
BATCH_SIZE=5000
REGULARIZATION=1e-4
COND_EVERY=10

SOLVE_FLOAT64=true
DEDUPLICATE_FILTERS=false        # Q=1 Morlet at M=256 has no near-copies (turbulence_ggd_tailfix_full.sh)

LAM=0.0
N_SUBSAMPLE=1
REG_SOLVER=thomas
REG_RIDGE=0.0
SAVE_REG_SYSTEM=true
SKIP_REG_SOLVE=true
REG_SYSTEM_DIR="${SCRATCH:+${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system/${EXP_NAME}}"

# ── the scalar fix ──
GGD_TAIL_QUANTILE=0.99
GGD_TAIL_MASS=true

# ── Gram pre-flight across families ──
GRAM_PREFLIGHT="${GRAM_PREFLIGHT:-1e-6}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"

TIME="03:00:00"
[ "${PREFLIGHT_ONLY}" = "true" ] && TIME="00:30:00"
CPUS=24
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
MODULE_PRE="arch/h100"

source "${LAUNCH_DIR}/_submit_turb_sweep.sh"
