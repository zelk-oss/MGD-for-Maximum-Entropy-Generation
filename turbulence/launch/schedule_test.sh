#!/bin/bash
# Do the new time grids remove the late jump in moment error?
#
# 2026-09-25 (compare_runs.ipynb, theta check): with the power schedules the moment
# error jumps to ~0.3 once h/(1-t) passes ~1.5e-3 (sched2 at 1-t ~ 7e-4, sched3 at
# ~1.8e-4) and the samples come out noisy (high-frequency power up to 1e4x the data).
# Only sched3 + ridge 1e-2 avoided it, and it still misses the 1% target (final mean
# rel. error 3e-2). Two new grids (codes/time_schedules.py), both float64, both
# ending at 1 - t = 5e-5:
#
#   two_phase  10k uniform steps on [0, 0.9], then 50k steps at constant
#              h/(1-t) = 1.5e-4 (~13x below sched3 at the end). 60k steps.
#   adaptive   h from the previous step's moment error (tol 1e-3 on
#              |e-p| / max_t|e|), h/(1-t) capped at 1e-3; at most 60k steps. Step
#              count, hence runtime, is not known in advance: the loop stops cleanly
#              at 90% of TIME if it has not reached the end (logged as a WARNING).
#
# Both at ridge 1e-4, the setting that jumped with the power schedules: if the jump
# is gone here, it was the step size. Everything else as long_full_sched*_reg1e-4.
#
# Budget: 1.57 s/step at n1=8500 before the 2026-09-25 savings (reuse of G(y_k) and
# of the moment velocity, codes/sde_routines.py); ~1.05 s/step expected after them
# (NOT yet measured) -> 60k steps ~17.5 h. If it is slower, two_phase aborts after
# ~30 steps via --time_limit_min (projected > 90% of TIME); adaptive stops cleanly at
# 90% of TIME. Host RAM: the saved regularised system is ~0.9 MB/step (r=233), ~52 GB.
#
# Compare afterwards: add the two labels to RUNS in compare_runs.ipynb (moment
# matching, theta check, physics), next to long_full_sched3_reg1e-4 / _reg1e-2.

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FULL=(L_6 L_6_psi L_2_lowpass Scalar_psi_gaussianK Scalar_morlet_gaussianK
      Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1)

submit() {   # submit <label> <nt> <schedule flags...>
  local label=$1 nt=$2; shift 2
  (
    EXP_NAME="${label}"
    SCHEDULE_ARGS="$*"
    TERMS=("${FULL[@]}")
    REGULARIZATION=1e-4
    SEED_LIST=(900)

    N1=8500
    SUBSERIES_LEN=512
    TARGET_LEN=256
    J=8
    Q=3
    SIGMA=3.5
    NT=${nt}
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

submit schedtest_two_phase_reg1e-4 60000 --schedule two_phase --n_bulk 10000 --t_switch 0.9 --gap_end 5e-5
submit schedtest_adaptive_reg1e-4  60000 --schedule adaptive --adapt_tol 1e-3 --adapt_ratio_max 1e-3 --gap_end 5e-5
