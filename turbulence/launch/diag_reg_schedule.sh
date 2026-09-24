#!/bin/bash
# Diagnostic 2x2: does moment matching near t -> 1 improve with a smaller per-step
# ridge (--regularization) and/or more time steps near t = 1 (--schedule_exponent)?
#
# Motivation: the lamtune_full_{nt33000,n1_8500} runs (and the July batch) match
# moments to ~1e-3 for t < 0.95 but fail in [0.95, 1] (compare_moment_matching.py).
# Suspects: the 0.01 ridge only partially corrects near-degenerate directions of G,
# which degenerate further near t = 1, and the step size there.
#
#   regularization  {1e-2, 1e-4}  x  schedule_exponent  {2, 3}
#
# All four: full potential set, n1=2000, NT=40000, 512 -> 256 windows, one seed,
# --solve_float64 (float32 per-step solves are already ~5e-4 off at ridge 1e-4 with
# near-duplicate statistics) and --deduplicate_filters (drops the exact-copy
# filters_Q channels 22, 23 at M=256, J=8, Q=3). Theta_reg solved in-run
# (thomas, full grid) -- lam is irrelevant for moment matching.
# Expected ~0.38 s/step on H100 (1.9e-4 s per particle per step) -> ~4.5 h each.
#
# regularization and schedule_exponent are NOT in the run name, so each combo gets
# its own label; compare afterwards with
#   python compare_moment_matching.py --labels diag_reg1e-2_sched2 diag_reg1e-4_sched2 \
#       diag_reg1e-2_sched3 diag_reg1e-4_sched3

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for REG in 1e-2 1e-4; do
  for SCHED in 2 3; do
    (
      EXP_NAME="diag_reg${REG}_sched${SCHED}"
      REGULARIZATION=${REG}
      SCHEDULE_EXPONENT=${SCHED}

      TERMS=(
          L_6
          L_6_psi
          L_2_lowpass
          Scalar_psi_gaussianK
          Scalar_morlet_gaussianK
          Scattering_Fourth_Order_Mod2_Real_Q1
          Scattering_Fourth_Order_Mod2_Imag_Q1
      )
      SEED_LIST=(900)

      N1=2000
      SUBSERIES_LEN=512
      TARGET_LEN=256
      J=8
      Q=3
      SIGMA=3.5
      NT=40000
      BATCH_SIZE=2000

      SOLVE_FLOAT64=true
      DEDUPLICATE_FILTERS=true

      LAM=1e-6
      N_SUBSAMPLE=1
      REG_SOLVER=thomas
      REG_RIDGE=1e-6
      SAVE_REG_SYSTEM=false
      SKIP_REG_SOLVE=false

      TIME="06:00:00"
      CPUS=24
      ACCOUNT="wbg@h100"
      CONSTRAINT="h100"
      PARTITION="gpu_p6"
      MODULE_PRE="arch/h100"

      source "${LAUNCH_DIR}/_submit_turb_sweep.sh"
    )
  done
done
