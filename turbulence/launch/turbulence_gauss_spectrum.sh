#!/bin/bash
# Gaussian with the turbulence spectrum (gauss_spectrum/README.md, 2026-11-03).
#
# x1 = n1 draws of the stationary Gaussian whose power spectrum is the turbulence spectrum averaged
# over the n1 real windows (--gauss_spectrum, codes/gauss_spectrum.py; data seed fixed at 0, so all
# seeds see the same x1, as for the real data). The earlier gauss_dim_sweep used WHITE noise instead.
#
# Same as turbulence_final_gauss_scat2.sh (Scattering_Second_Order_Q1 only, M = 256, J = 8, n1 10000,
# sigma 3.5, seeds 900-904, reg system saved, Theta_reg offline) except:
#   data            Gaussian with the data spectrum, not the real data
#   NT              10000 (user, 2026-11-03: enough for moment matching at r = 9), n_bulk 2000 as gauss_dim_sweep
#   REGULARIZATION  1e-4 as turbulence_final and the jets runs (gauss_scat2 used 1e-6)
#   TIME            1 h (gauss_scat2: 23.7 min for nt 55000 on V100, 25.8 ms/it)
#
# Usage (on Jean Zay, from the repo root):
#   bash turbulence/launch/turbulence_gauss_spectrum.sh
#   SEEDS="900" bash turbulence/launch/turbulence_gauss_spectrum.sh   # one seed first

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="turbulence_gauss_spectrum"
TERMS=(Scattering_Second_Order_Q1)
SEED_LIST=(${SEEDS:-900 901 902 903 904})
GAUSS_SPECTRUM=true

N1=10000
SUBSERIES_LEN=512
TARGET_LEN=256
J=8
Q=3
SIGMA=3.5
NT=10000
SCHEDULE_ARGS="--schedule two_phase --n_bulk 2000 --t_switch 0.9 --gap_end 5e-5"
BATCH_SIZE=5000
REGULARIZATION=1e-4
COND_EVERY=10

SOLVE_FLOAT64=true

LAM=0.0
N_SUBSAMPLE=1
REG_SOLVER=thomas
REG_RIDGE=0.0
SAVE_REG_SYSTEM=true
SKIP_REG_SOLVE=true
REG_SYSTEM_DIR="${SCRATCH:+${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system/${EXP_NAME}}"

TIME="01:00:00"
CPUS=10
ACCOUNT="wbg@v100"
CONSTRAINT="v100"
PARTITION="gpu_p13"

source "${LAUNCH_DIR}/_submit_turb_sweep.sh"
