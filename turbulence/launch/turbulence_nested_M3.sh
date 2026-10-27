#!/bin/bash
# Nested-model hierarchy, model M3 (2026-10-23): M2 + scalar potentials on the Q=3 Morlet bank.
#   M1 (Scattering_Second_Order_Q1, Gaussian)  ⊂  M2 (turbulence_nested_M2.sh)  ⊂  M3 (this)
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
# Time. ggd_mass_noScat4 (the same model, seed 900, job 1447814): SDE 15069 s = 4.2 h (r=160). 6 h wall time.
# Seed 900 of that run was fitted WITHOUT the pre-flight: if the pre-flight drops anything here, it is
# not the same model and seed 900 reruns with the others (hence 900-904 by default).
#
# Usage (on Jean Zay, from the repo root):
#   PREFLIGHT_ONLY=true bash turbulence/launch/turbulence_nested_M3.sh   # fit + Gram check, no SDE (minutes)
#   bash turbulence/launch/turbulence_nested_M3.sh                       # 5 seeds, 900-904
#   SEEDS="900" bash ...                                                 # one seed
#   PREFLIGHT_ONLY=true SEEDS=900 LABEL_SUFFIX=_pfcheck bash ...         # refit + Gram check in a new folder

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="turbulence_nested_M3${LABEL_SUFFIX}"   # LABEL_SUFFIX: new run folder (else an identical
LABEL="${EXP_NAME}"                              # config reuses the finished run and skips the fit)
TERMS=(L_6 Scalar_psi_gaussianK Scalar_morlet_gaussianK)
SEED_LIST=(${SEEDS:-900 901 902 903 904})

# ── as turbulence_final.sh, Q=1 ──
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

# ── the scalar fix ──
GGD_TAIL_QUANTILE=0.99
GGD_TAIL_MASS=true

# ── Gram pre-flight across families ──
GRAM_PREFLIGHT="${GRAM_PREFLIGHT:-1e-6}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"

TIME="06:00:00"
[ "${PREFLIGHT_ONLY}" = "true" ] && TIME="00:30:00"
CPUS=24
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
MODULE_PRE="arch/h100"

source "${LAUNCH_DIR}/_submit_turb_sweep.sh"
