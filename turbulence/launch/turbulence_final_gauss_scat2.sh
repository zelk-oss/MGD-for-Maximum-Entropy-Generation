#!/bin/bash
# Gaussian baseline for the final turbulence runs (2026-10-21): second-order scattering only.
#
# Question: is there a concentration phenomenon for the Gaussian model, i.e. how wide is the
# distribution of log p compared with the full model (turbulence_final, 30 seeds)?
#
# Identical to turbulence_final.sh (n1 10000, nt 55000 two_phase n_bulk 5000,
# M = 256, J = 8, sigma 3.5, reg system saved, Theta_reg solved offline) except:
#   RIDGE   Gram Tikhonov 1e-6 instead of 1e-4 (user's choice, 2026-10-21): 9 well-separated
#           statistics, so a small ridge suffices and biases theta_t less.
#   TERMS   Scattering_Second_Order_Q1 only: E|x * psi_j|^2 on the Q = 1 Morlet bank
#           (8 band-pass + 1 low-pass = 9 statistics, the bank of the order-4 terms). The
#           energy is quadratic in x, so the maxent model is a stationary Gaussian.
#           Not the Q = 3 bank (Scattering_Second_Order): it has exact duplicate channels
#           (|psi_20|^2 and |psi_21|^2 at cosine 1.0) and Scattering_* skip --deduplicate_filters,
#           so its Gram would be singular. The Q = 1 bank has max cosine 0.24 between the
#           |psi_j|^2 and its Littlewood-Paley sum is >= 1.5e-3 at every frequency, so every
#           Fourier mode is confined (normalizable Gaussian).
#   no L_6  L_6 is a sixth moment: it would make the model non-Gaussian, which defeats the
#           purpose. It is also not needed for confinement (see above).
#   no DROP_STATS / dedup: both only act on the Scalar_* / channel-list terms, absent here.
#   SEEDS   900-904 by default: the same seeds as turbulence_final, so the comparison is paired.
#   V100    gpu_p13, 5 h wall time (shorter queue; 9 statistics vs r = 266, expected much
#           faster than the final runs). Speed on V100 not measured: check the steady s/it in
#           the SLURM log after ~15 min; _check_time_budget aborts cleanly if over budget.
#
# Theta_reg is solved offline afterwards from the saved systems (codes/resolve_theta_reg.py,
# no --neg_cols: there is no L_6 to constrain).
#
# Usage (on Jean Zay, from the repo root):
#   bash turbulence/launch/turbulence_final_gauss_scat2.sh
#   SEEDS="900" bash turbulence/launch/turbulence_final_gauss_scat2.sh   # one seed first

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="turbulence_final_gauss_scat2"
TERMS=(Scattering_Second_Order_Q1)
SEED_LIST=(${SEEDS:-900 901 902 903 904})

N1=10000
SUBSERIES_LEN=512
TARGET_LEN=256
J=8
Q=3
SIGMA=3.5
NT=55000
SCHEDULE_ARGS="--schedule two_phase --n_bulk 5000 --t_switch 0.9 --gap_end 5e-5"
BATCH_SIZE=5000
REGULARIZATION=1e-6
COND_EVERY=10

SOLVE_FLOAT64=true

LAM=0.0
N_SUBSAMPLE=1
REG_SOLVER=thomas
REG_RIDGE=0.0
SAVE_REG_SYSTEM=true
SKIP_REG_SOLVE=true
REG_SYSTEM_DIR="${SCRATCH:+${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system/${EXP_NAME}}"

TIME="05:00:00"
CPUS=10
ACCOUNT="wbg@v100"
CONSTRAINT="v100"
PARTITION="gpu_p13"

source "${LAUNCH_DIR}/_submit_turb_sweep.sh"
