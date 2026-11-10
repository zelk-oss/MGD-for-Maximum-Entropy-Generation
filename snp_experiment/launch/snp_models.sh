#!/bin/bash
# MGD on the daily S&P 500 log-returns (MGD paper Sec. 6.3): the whole series (d = 6062) is ONE sample, 100 particles,
# J = 6, Q = 1, repo Morlet bank (as the other experiments), uniform time grid,
# regularisation 1e-7 (paper App. D.1), float64 solves.
# Four nested models, one launch per model (MODEL=M0|M1|M2|M3):
#   M0  scat1 + scat2 + scat3 (Re+Im) + scat4 (Re+Im)
#   M1  M0 + scalar Morlet (Scalar_morlet_gaussianK, Q=1 bank)
#   M2  scat1 + scat2 + scalar
#   M3  scat1 + scat2
# L6=1 adds the L_6 confinement potential (L2p_norm(3), J+1 = 7 columns, FIRST in Theta) and the default label suffix _L6
# (arms snp_M0_L6, snp_M1_L6: the Eulerian-turbulence construction; Theta_reg then needs the L_6<0 gate, snp_chain_l6.sh).
# Without L6: no L_6 (the first four arms, unconstrained). No Q=3 bank, no raw-signal marginal. scat2 keeps the low-pass (Morel / MGD convention).
# Conventions = Morel's scat_spectra on this series at J=6 (checked by running their analyze(): 1+6+7+27+56 complex
# coefficients): scat3 = Scattering_Third_Order_Real/Imag (27: diagonal j=k and low-pass column included; its Imag
# low-pass columns are identically zero and are switched off by the live/dead mechanism), scat4 WITH the real
# diagonal (the *_diag terms, 56 real + 20 imag at J=6; the codebase default without it has 35).
# SCAT4_ENV=mod2 (default): |W|^2 envelope as in all the other experiments (Mod2_Real_Q1_diag, Mod2_Imag_Q1);
# SCAT4_ENV=mod: the paper's / Morel's |W| envelope (Real_Q1_diag, Imag_Q1).
#
# Usage (on Jean Zay, from the repo root; the pkl must be in data/data_files/):
#   MODEL=M3 bash snp_experiment/launch/snp_models.sh                     # 5 seeds, 900-904, V100 (GPU=h100 for H100)
#   MODEL=M0 SEEDS="900" NT=2000 bash snp_experiment/launch/snp_models.sh
# then, per model, the Theta_reg solve + sign report: snp_experiment/launch/snp_theta_reg.sh
#   LABEL_SUFFIX=_test ...                                                # new run folder
# NT: SDE steps. sigma = 3.5 (sigma^2 = 12.25, the turbulence value; paper S&P: sigma^2 <= 5.5 with 2500 steps, +~200 steps per
# unit of sigma^2) -> 5000 by default; check moment matching and rerun with more if not converged. TIME is NOT measured.

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWEEP_DIR="$(cd "${LAUNCH_DIR}/.." && pwd)"         # the shared submit script runs <SWEEP_DIR>/run_SDE.py

MODEL="${MODEL:?set MODEL=M0|M1|M2|M3}"
SCAT4_ENV="${SCAT4_ENV:-mod2}"

SCAT1="Scattering_First_Order"
SCAT2="Scattering_Second_Order_Q1"
SCAT3="Scattering_Third_Order_Real Scattering_Third_Order_Imag_nolp"   # Imag without its J identically-zero low-pass columns
case "${SCAT4_ENV}" in
    mod2) SCAT4="Scattering_Fourth_Order_Mod2_Real_Q1_diag Scattering_Fourth_Order_Mod2_Imag_Q1" ;;
    mod)  SCAT4="Scattering_Fourth_Order_Real_Q1_diag Scattering_Fourth_Order_Imag_Q1" ;;
    *) echo "SCAT4_ENV must be mod2 or mod"; exit 1 ;;
esac
SCALAR="Scalar_morlet_gaussianK"
case "${MODEL}" in
    M0) TERMS=(${SCAT1} ${SCAT2} ${SCAT3} ${SCAT4}) ;;
    M1) TERMS=(${SCAT1} ${SCAT2} ${SCAT3} ${SCAT4} ${SCALAR}) ;;
    M2) TERMS=(${SCAT1} ${SCAT2} ${SCALAR}) ;;
    M3) TERMS=(${SCAT1} ${SCAT2}) ;;
    M4) TERMS=(${SCAT2}) ;;       # scat2 only (Scattering_Second_Order_Q1), as the earlier Gaussian-spectrum runs
    *) echo "MODEL must be M0|M1|M2|M3|M4"; exit 1 ;;
esac

# GAUSS=1: x1 = ONE Gaussian series with the Welch spectrum of the S&P (Hann segments WELCH_NPERSEG, default 512),
# data seed 0 (same x1 for all seeds); label suffix _gauss (arm snp_M4_gauss). Use with MODEL=M4.
if [ "${GAUSS:-0}" = "1" ]; then
    GAUSS_SPECTRUM=true; WELCH_NPERSEG="${WELCH_NPERSEG:-512}"; LABEL_SUFFIX="${LABEL_SUFFIX:-_gauss}"
fi

if [ "${L6:-0}" = "1" ]; then TERMS=(L_6 "${TERMS[@]}"); LABEL_SUFFIX="${LABEL_SUFFIX:-_L6}"; fi
EXP_NAME="snp_${MODEL}${LABEL_SUFFIX}"
LABEL="${EXP_NAME}"
SEED_LIST=(${SEEDS:-900 901 902 903 904})

# ── data / model ──
N1=1                       # the whole series is one sample
SUBSERIES_LEN=6062         # ignored by snp_experiment/run_SDE.py (M = d = 6062); required by the shared launcher
TARGET_LEN=6062
J=6
Q=1
WAV_TYPE=morlet           # as the other experiments (the MGD paper text says Morlet; Morel's tutorial uses battle_lemarie)
N_PARTICLES=100
SIGMA=3.5                  # sigma (not sigma^2), same as the turbulence runs
NT="${NT:-5000}"
SCHEDULE_EXPONENT=1        # --schedule power, exponent 1 = uniform grid (paper); no SCHEDULE_ARGS
REGULARIZATION=1e-7        # paper App. D.1: Gram diagonal normalised to 1 (Jacobi scaling) + delta Id
SOLVE_FLOAT64=true         # needed at a ridge this small
COND_EVERY=10

LAM=0.0
N_SUBSAMPLE=1
REG_SOLVER=thomas
REG_RIDGE=0.0
SAVE_REG_SYSTEM=true       # Theta_reg is solved afterwards from the saved system (as turbulence_ggd_mass_final.sh)
SKIP_REG_SOLVE=true
REG_SYSTEM_DIR="${SCRATCH:+${SCRATCH}/MGD-for-Maximum-Entropy-Generation/snp_experiment/reg_system/${EXP_NAME}}"

# ── scalar potential: as turbulence_ggd_mass_final.sh (tail fix), no DROP_STATS (that slot was turbulence-specific) ──
GGD_TAIL_QUANTILE=0.99
GGD_TAIL_MASS=true

# ── cluster (GPU=v100 | h100, always explicit) ──
GPU="${GPU:-v100}"
TIME="${TIME:-08:00:00}"      # NOT measured; the in-run guard stops cleanly if projected past 90% of it
if [ "${GPU}" = "v100" ]; then
    CPUS=10; ACCOUNT="wbg@v100"; CONSTRAINT="v100"; PARTITION="gpu_p13"; MODULE_PRE=""
elif [ "${GPU}" = "h100" ]; then
    CPUS=24; ACCOUNT="wbg@h100"; CONSTRAINT="h100"; PARTITION="gpu_p6"; MODULE_PRE="arch/h100"
else
    echo "GPU must be v100 or h100"; exit 1
fi

source "${LAUNCH_DIR}/../../turbulence/launch/_submit_turb_sweep.sh"
