#!/bin/bash
# Full-size single-seed runs: does moment matching near t -> 1 get fixed, and by what?
#
# The lamtune_* and July runs match moments to ~1e-3 for t < 0.95 but fail in
# [0.95, 1]. Suspects: the per-step ridge (--regularization) and the time grid near
# t = 1 (--schedule_exponent). One seed each, at production size (n1=8500, NT=40000):
#
#                        reg 1e-2                          reg 1e-4
#   schedule 2   lamtune_full_n1_8500 (exists, 5 seeds)   long_full_sched2_reg1e-4
#   schedule 3   long_full_sched3_reg1e-2                 long_<set>_sched3_reg1e-4  (all 4 sets)
#
# -> full-set 2x2 (schedule effect, ridge effect, interaction) plus a first run of
#    every potential set at the expected production setting (sched 3, reg 1e-4).
# All: --solve_float64, --deduplicate_filters, system saved for offline lam tuning
# (single seed: residual-whiteness criterion only). ~18 h each on H100.
#
# Compare afterwards (from turbulence/):
#   python compare_moment_matching.py --labels lamtune_full_n1_8500 long_full_sched2_reg1e-4 \
#       long_full_sched3_reg1e-2 long_full_sched3_reg1e-4

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FULL=(L_6 L_6_psi L_2_lowpass Scalar_psi_gaussianK Scalar_morlet_gaussianK
      Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1)
NO_L2=(L_6 L_6_psi Scalar_psi_gaussianK Scalar_morlet_gaussianK
       Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1)
NO_MORLET=(L_6 L_6_psi L_2_lowpass Scalar_psi_gaussianK
           Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1)
NO_BOTH=(L_6 L_6_psi Scalar_psi_gaussianK
         Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1)

submit() {   # submit <set name> <regularization> <schedule_exponent> <terms...>
  local set=$1 reg=$2 sched=$3; shift 3
  (
    EXP_NAME="long_${set}_sched${sched}_reg${reg}"
    TERMS=("$@")
    REGULARIZATION=${reg}
    SCHEDULE_EXPONENT=${sched}
    SEED_LIST=(900)

    N1=8500
    SUBSERIES_LEN=512
    TARGET_LEN=256
    J=8
    Q=3
    SIGMA=3.5
    NT=40000
    BATCH_SIZE=5000

    SOLVE_FLOAT64=true
    DEDUPLICATE_FILTERS=true

    LAM=0.0
    N_SUBSAMPLE=1
    REG_SOLVER=thomas
    REG_RIDGE=0.0
    SAVE_REG_SYSTEM=true
    SKIP_REG_SOLVE=true
    REG_SYSTEM_DIR="${SCRATCH:+${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system}"

    TIME="20:00:00"
    CPUS=24
    ACCOUNT="wbg@h100"
    CONSTRAINT="h100"
    PARTITION="gpu_p6"
    MODULE_PRE="arch/h100"

    source "${LAUNCH_DIR}/_submit_turb_sweep.sh"
  )
}

# expected production setting, every potential set
submit full                   1e-4 3 "${FULL[@]}"
submit noL2lowpass            1e-4 3 "${NO_L2[@]}"
submit noScalarMorlet         1e-4 3 "${NO_MORLET[@]}"
submit noL2lowpass_noScalarMorlet 1e-4 3 "${NO_BOTH[@]}"
# the rest of the full-set 2x2
submit full                   1e-2 3 "${FULL[@]}"
submit full                   1e-4 2 "${FULL[@]}"
