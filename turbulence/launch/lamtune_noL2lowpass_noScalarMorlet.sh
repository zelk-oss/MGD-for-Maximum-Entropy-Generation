#!/bin/bash
# lam-tuning runs, full set WITHOUT L_2_lowpass AND Scalar_morlet_gaussianK.
# Same protocol as lamtune_full.sh (see its header); only TERMS/EXP_NAME differ.

EXP_NAME="lamtune_noL2lowpass_noScalarMorlet"

TERMS=(
    L_6
    L_6_psi
    Scalar_psi_gaussianK
    Scattering_Fourth_Order_Mod2_Real_Q1
    Scattering_Fourth_Order_Mod2_Imag_Q1
)

SEED_LIST=({900..904})

N1=10000
SUBSERIES_LEN=512
TARGET_LEN=256

J=8
Q=3
SIGMA=3.5
NT=40000
REGULARIZATION=0.01
# Potential evaluations are chunked in batches of BATCH_SIZE samples (results
# unchanged). The order-4 scattering gradient builds complex (B, 8, 8, 9, 256)
# tensors (~1.1 MB/sample) and Scattering_Fourth_Order_Mod2_Imag.grad holds ~6
# at once: peak ~6.6 MB/sample. 5000 -> ~33 GiB on an 80 GB H100. (A 16 GB V100
# needs <= 1000: 2500 OOM'd there, job 111309.)
BATCH_SIZE=5000

LAM=0.0
N_SUBSAMPLE=1
REG_SOLVER=thomas
REG_RIDGE=0.0
SAVE_REG_SYSTEM=true
SKIP_REG_SOLVE=true
REG_SYSTEM_DIR="${SCRATCH:+${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system}"

# Fewest potentials -> fastest of the four; tighten after the first run
TIME="20:00:00"
CPUS=24                  # a quarter of an H100 node (4 GPUs); host RAM scales with it
# H100 partition: the 2026-07-28 reference batch ran on H100 (0.96 s/it at n1=5000);
# a 16 GB V100 measured 6.7 s/it at n1=10000 (job 112348) -> 74 h, unusable.
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
MODULE_PRE="arch/h100"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_submit_turb_sweep.sh"
