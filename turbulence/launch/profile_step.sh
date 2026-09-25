#!/bin/bash
# Where does the ~1.6 s per SDE step go at production size (n1=8500)?
#
# Motivation: forward_regularised computes the Gram matrix three times per step --
# G(x_k) for the predictor, G(y_k) for the corrector, and G(y_k) AGAIN for the
# regularised-system accumulation (Mk = compute_G(Xt), Xt = y_k) -- plus repeated
# moment evaluations on the interpolant. Before caching any of it, measure what
# each part costs.
#
# Same settings as the long_full_* runs (full set, n1=8500, batch 5000, float64
# solves, dedup filters) but only NT=400 steps, run under cProfile with
# CUDA_LAUNCH_BLOCKING=1 (PROFILE=true) so GPU time lands on the Python call that
# launched it. Blocking launches slow the run down: use the profile for the SHARES
# of time, not the absolute s/step.
#
# Read the result (from turbulence/):
#   python launch/read_profile.py saved_results/logs/profile_profile_step_<jobid>_0.prof

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="profile_step"
PROFILE=true
TERMS=(L_6 L_6_psi L_2_lowpass Scalar_psi_gaussianK Scalar_morlet_gaussianK
       Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1)
SEED_LIST=(900)

N1=8500
SUBSERIES_LEN=512
TARGET_LEN=256
J=8
Q=3
SIGMA=3.5
NT=400
BATCH_SIZE=5000
REGULARIZATION=1e-2
SCHEDULE_EXPONENT=3

SOLVE_FLOAT64=true
DEDUPLICATE_FILTERS=true

LAM=0.0
N_SUBSAMPLE=1
REG_SOLVER=thomas
REG_RIDGE=0.0
SAVE_REG_SYSTEM=false
SKIP_REG_SOLVE=false

TIME="01:00:00"
CPUS=24
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
MODULE_PRE="arch/h100"

source "${LAUNCH_DIR}/_submit_turb_sweep.sh"
