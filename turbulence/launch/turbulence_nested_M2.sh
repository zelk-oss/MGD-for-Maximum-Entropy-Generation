#!/bin/bash
# Nested-model hierarchy, model M2 (2026-10-23): scalar potentials on the Q=1 Morlet bank + L_6.
#   M1 (Scattering_Second_Order_Q1, Gaussian)  ⊂  M2 (this)  ⊂  M3 (+ Q=3 scalar, turbulence_nested_M3.sh)
# Plan: ~/.claude/plans/hashed-noodling-llama.md
#
# Same data, schedule and regularisation as turbulence_final (M=256, n1=10000, nt=55000, two_phase),
# with the scalar fix (--ggd_tail_quantile 0.99 --ggd_tail_mass, no alpha bounds). No scattering,
# no L_2_lowpass, no Q=3 bank.
#
# GRAM_PREFLIGHT (codes/gram_preflight.py): before the SDE, fit the potentials and drop every
# statistic explained to within TOL by the ones before it (order L_6, Scalar_morlet, Scalar_psi),
# so M2's statistics are the same ones M3 starts from. Report: <run>/gram_preflight.json.
# DROP_STATS morlet ch 8 region 1 is kept as in M3 / turbulence_final, so M2 is nested in M3 (that
# slot is a near-copy of a Scalar_psi one, found by hand on 10-01; in M3 it is dropped here, not there).
#
# Time. tailfix_L6_full (M=256, Q=1, L_6 + Scalar_morlet, r=45): SDE 4487 s. The mass slot adds ~1
# statistic per channel (r ~ 54): ~1.5 h per seed expected, not measured. 3 h wall time.
#
# Usage (on Jean Zay, from the repo root):
#   PREFLIGHT_ONLY=true bash turbulence/launch/turbulence_nested_M2.sh   # fit + Gram check, no SDE (minutes)
#   bash turbulence/launch/turbulence_nested_M2.sh                       # 5 seeds, 900-904
#   SEEDS="900" bash ...                                                 # one seed

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="turbulence_nested_M2"
LABEL="${EXP_NAME}"
TERMS=(L_6 Scalar_morlet_gaussianK)
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
DROP_STATS="Scalar_morlet_gaussianK:8:1"
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
