#!/bin/bash
# FINAL turbulence runs for the paper (2026-10-13): 30 seeds, n1 = 10000.
#
# Identical to the 2026-10-01 zfloor_two_phase_noL6psi_noSm30_reg1e-4 runs (zfloor_test.sh)
# except:
#   N1      8500  -> 10000
#   NT      60000 -> 55000, N_BULK 10000 -> 5000: the tail keeps n_tail = 50000 steps, so its
#           geometric ratio r = 1.520e-4 is bit-for-bit the Oct 1 one (codes/time_schedules.py,
#           two_phase_schedule); only the bulk [0, 0.9] is coarser (h = 1.8e-4).
#   DROP    Scalar_morlet_gaussianK:8:3 -> :8:1. The cross-bank near-copy (Scalar_morlet
#           channel 8 = Scalar_psi channel 21, same coarsest filter in the Q=1 and Q=3 banks)
#           sits in whichever GGD region the n1-dependent cuts align: at n1 = 10000 it is
#           region 1 (corr 0.99964). Checked with turbulence/preflight_dedupe_data.py at
#           n1 = 10000: with L_6_psi omitted and this drop, r = 266, cond = 1.1e5,
#           0 pairs above corr 0.999, 0 null directions below 1e-6.
#   reg_system saved to its own SCRATCH subfolder.
#
# Time: Oct 1 baseline (10 seeds, n1 8500, nt 60000, COND_EVERY=10) 16.63 h +- 70 s; scaled
# linearly in n1 and nt -> ~17.9 h, under the 19h30 loop budget of the 20 h job. If a node is
# slower, _check_time_budget aborts after ~130 steps (minutes) and logs the measured s/it.
# Saved system: r^2 x 8 B x 2 (M_k, G_k) ~ 1.1 MB/step -> ~62 GB per seed (host RAM + SCRATCH).
#
# Theta_reg (lam 1e-3, ridge 1e-4, moment/uniform, L_6 < 0) is solved offline afterwards from
# the saved systems (codes/resolve_theta_reg.py), hence SKIP_REG_SOLVE=true / LAM=0.0 here.
#
# Usage (on Jean Zay, from turbulence/launch/):
#   step 1:  bash turbulence_final.sh                                  # seed 900 only
#            -> after ~15 min check the SLURM log: fit OK, r = 266, steady s/it <= ~1.17
#   step 2:  SEEDS="$(seq -s ' ' 901 929)" bash turbulence_final.sh    # the other 29 seeds

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="turbulence_final"
TERMS=(L_6 Scalar_psi_gaussianK Scalar_morlet_gaussianK
       Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1)
SEED_LIST=(${SEEDS:-900})

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
REG_SYSTEM_DIR="${SCRATCH:+${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system/turbulence_final}"

TIME="20:00:00"
CPUS=24
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
MODULE_PRE="arch/h100"

source "${LAUNCH_DIR}/_submit_turb_sweep.sh"
