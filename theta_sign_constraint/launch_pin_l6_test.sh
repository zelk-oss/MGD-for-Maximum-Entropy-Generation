#!/bin/bash
# Pinned-L_6 test on B+Scat4 (2026-11-17), theta_sign_constraint/pin_l6_test.py: Theta_reg with the L_6 coefficients
# fixed at -eps (all of them / only the positive ones) vs the free fit and the existing projected-Newton L_6<0 file:
# energy cost, -log p ranking on data and generated signals, per-family theta change, radial scan for far bumps.
# The SDE transport is not touched; only the offline Theta_reg is re-solved from the saved systems.
# Thomas solves on CPU, potentials/log p on the GPU. GPU dev QoS only (never prepost: user 2026-11-17), GPU=v100|h100
# mandatory; billed GPU x elapsed even while the solve runs on CPU. One array task per seed.
# RAM: system 49 GB (mmap) + Thomas factor; the unconstrained solve of these systems peaked at 103-144 GB (job 1864057).
# RAM comes with CPUs: H100 24 CPUs, V100 10 CPUs (no extra billing); 10 V100 CPUs is too little for the real systems.
# Output: full Theta_reg per arm -> $SCRATCH/.../turbulence/theta_reg_pinl6/<config>/ (4 files x ~124 MB per seed);
#         summary.json + logp_scan.npz (a few MB) -> theta_sign_constraint/results/pin_l6_test/<LABEL>/seed<N>/
#
# Usage (Jean Zay, repo root):
#   GPU=h100 LABEL=pinl6_bscat4 bash theta_sign_constraint/launch_pin_l6_test.sh          # 5 seeds 900-904
#   GPU=v100 SMOKE=1 LABEL=pinl6_smoke bash theta_sign_constraint/launch_pin_l6_test.sh   # 15 min, small whitened system
#   DRY=1 ...                                                                             # print the files, submit nothing
# overrides: SEEDS, LAM (3e-5), RIDGE (1e-4), EPS (1e-2), COLS (0:9), CPUS, TIME (02:00:00), BUDGET (90 min of solves)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
LABEL="${LABEL:?set LABEL}"
GPU="${GPU:?set GPU=v100 or GPU=h100}"
case "${GPU}" in
    h100) ACCOUNT="wbg@h100"; PARTITION="gpu_p6";  CONSTRAINT="h100"; QOS="qos_gpu_h100-dev"; MODULE_PRE="arch/h100"; CPUS="${CPUS:-24}" ;;
    v100) ACCOUNT="wbg@v100"; PARTITION="gpu_p13"; CONSTRAINT="v100"; QOS="qos_gpu-dev";      MODULE_PRE="";          CPUS="${CPUS:-10}" ;;
    *) echo "GPU must be v100 or h100"; exit 1 ;;
esac
SEEDS=(${SEEDS:-900 901 902 903 904})
LAM="${LAM:-3e-5}"
RIDGE="${RIDGE:-1e-4}"
EPS="${EPS:-1e-2}"
COLS="${COLS:-0:9}"
TIME="${TIME:-02:00:00}"
BUDGET="${BUDGET:-90}"
N_SCAN="${N_SCAN:-200}"
ARM="turbulence_nested_M3_bicw_allmass_scat4"
SCR="\${SCRATCH:?SCRATCH not set}/MGD-for-Maximum-Entropy-Generation"
SYS_DIR="${SCR}/turbulence/reg_system/${ARM}"
THETA_OUT="${SCR}/turbulence/theta_reg_pinl6"
EXP_ROOT="${PROJECT_DIR}/turbulence/experiments"
SAMPLES="${PROJECT_DIR}/turbulence/saved_results/samples"
FREE_DIR="${PROJECT_DIR}/turbulence/saved_results/theta_reg_lamsweep"
PNEWTON_DIR="${SCR}/turbulence/theta_reg_l6neg"
OUT="${HERE}/results/pin_l6_test/${LABEL}"
if [ -n "${SMOKE:-}" ]; then                 # small whitened B+Scat4 run (nt 1000, same terms/windows): code paths only
    SEEDS=(900); TIME="00:15:00"; BUDGET=8; N_SCAN=20
    WH="${PROJECT_DIR}/whitening_test/bscat4/turbulence"
    SYS_DIR="${SCR}/whitening_test/reg_system/bscat4"; ARM="wh_bscat4_a0p75_unif_nt1000"
    EXP_ROOT="${WH}/experiments"; SAMPLES="${WH}/saved_results/samples"; FREE_DIR=""; PNEWTON_DIR=""
    THETA_OUT="${SCR}/turbulence/theta_reg_pinl6_smoke"
fi
LOGS="${OUT}/logs"
echo "pinned-L_6 test ${LABEL}: seeds ${SEEDS[*]} | ${ARM} | lam ${LAM} ridge ${RIDGE} eps ${EPS} cols ${COLS}"
echo "  ${GPU} (${QOS}), ${CPUS} CPUs, ${TIME} | Theta -> ${THETA_OUT} | results -> ${OUT}"
if [ -n "${DRY:-}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${LOGS}"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${LABEL}
#SBATCH -A ${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH -C ${CONSTRAINT}
#SBATCH --qos=${QOS}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --array=0-$(( ${#SEEDS[@]} - 1 ))
#SBATCH --time=${TIME}
#SBATCH --output=${LOGS}/slurm_%x_%A_%a.log
module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
SEEDS=( ${SEEDS[@]} )
SEED=\${SEEDS[\${SLURM_ARRAY_TASK_ID}]}
shopt -s nullglob
one() { if [ \$# -ne 1 ]; then echo "Error: expected 1 match, found \$#: \$*" >&2; exit 1; fi; echo "\$1"; }
SYSTEM=\$(one ${SYS_DIR}/*seed_\${SEED}_*${ARM}*.pt) || exit 1
CONFIG=\$(basename "\${SYSTEM}" .pt)
EXP=\$(one ${EXP_ROOT}/*/\${CONFIG}) || exit 1
EXTRA=""
[ -n "${FREE_DIR}" ] && EXTRA="\${EXTRA} --free_dir ${FREE_DIR}"
[ -n "${PNEWTON_DIR}" ] && EXTRA="\${EXTRA} --pnewton_dir ${PNEWTON_DIR}"
echo "seed \${SEED}: \${SYSTEM}"
echo "  exp \${EXP}"
python -u theta_sign_constraint/pin_l6_test.py --system "\${SYSTEM}" --exp_dir "\${EXP}" \
    --samples "${SAMPLES}/\${CONFIG}.pt" \${EXTRA} \
    --lam ${LAM} --ridge ${RIDGE} --eps ${EPS} --cols ${COLS} --n_scan ${N_SCAN} \
    --theta_out "${THETA_OUT}" --out "${OUT}/seed\${SEED}" --time_budget_min ${BUDGET} --mmap --threads ${CPUS}
EOT
