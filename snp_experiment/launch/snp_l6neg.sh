#!/bin/bash
# L_6 sign gate + (only if violated) Theta_reg with every L_6 coefficient <= -EPS, for one S&P arm with L_6.
# Chained after the xseed job of the arm (afterok). One array task per seed. Each task:
#   1. python snp_experiment/l6_gate.py: counts L_6 entries with Theta > 0 in the unconstrained Theta_reg of ALL seeds at
#      the cross-seed lambda (all nodes) -> results/gate/gate_<arm>.json, line "GATE <arm>: VIOLATED|OK";
#   2. OK       -> nothing more to do (the unconstrained Theta_reg of snp_theta_reg.sh is the result);
#      VIOLATED -> codes/resolve_theta_reg.py --neg_cols 0:7 (projected Newton, theta_sign_constraint/README.md) on the
#                  seed's saved system at that lambda -> <OUTDIR_NEG>/<config>/lam<lam>_ridge..._neg<eps>.pt
#                  (or *_NOTCONVERGED_<time>.pt: still feasible, as in the turbulence runs).
# The whole model is gated at once, so all seeds get the same kind of Theta.
#
# Usage (repo root on Jean Zay):
#   MODEL=M0 DEPEND=<xseed job id> bash snp_experiment/launch/snp_l6neg.sh
# overrides: LABEL_SUFFIX (_L6), SEEDS, RIDGE (1e-7), EPS (1e-2 nats per sd of phi, as turbulence), MAX_ITER (20),
#   WINDOW (end0.99), COLS (0:7), GPU (v100), TIME (04:00:00), CPUS, REG_SYSTEM_DIR, OUTDIR_NEG

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/../.." && pwd)"

MODEL="${MODEL:?set MODEL=M0|M1}"
ARM="snp_${MODEL}${LABEL_SUFFIX:-_L6}"
SEED_LIST=(${SEEDS:-900 901 902 903 904})
RIDGE="${RIDGE:-1e-7}"
EPS="${EPS:-1e-2}"
MAX_ITER="${MAX_ITER:-20}"
WINDOW="${WINDOW:-end0.99}"
COLS="${COLS:-0:7}"
REG_SYSTEM_DIR="${REG_SYSTEM_DIR:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/snp_experiment/reg_system/${ARM}}"
OUTDIR_NEG="${OUTDIR_NEG:-${SCRATCH}/MGD-for-Maximum-Entropy-Generation/snp_experiment/theta_reg_l6neg/${ARM}}"
LOGDIR="${PROJECT_DIR}/snp_experiment/saved_results/logs"
DEPEND="${DEPEND-}"
TIME="${TIME:-04:00:00}"
GPU="${GPU:-v100}"
case "${GPU}" in
    h100) ACCOUNT="wbg@h100"; PARTITION="gpu_p6";  CONSTRAINT="h100"; MODULE_PRE="arch/h100"; CPUS="${CPUS:-24}" ;;
    v100) ACCOUNT="wbg@v100"; PARTITION="gpu_p13"; CONSTRAINT="v100"; MODULE_PRE="";          CPUS="${CPUS:-10}" ;;
    *) echo "GPU must be h100 or v100"; exit 1 ;;
esac

echo "arm ${ARM}: L_6 gate (cols ${COLS}, ${WINDOW}) then, if violated, L_6<0 (eps ${EPS}, <= ${MAX_ITER} rounds), seeds ${SEED_LIST[*]}"
echo "  systems ${REG_SYSTEM_DIR} -> ${OUTDIR_NEG} | ${GPU}, ${CPUS} CPUs, ${TIME}, $([ -n "${DEPEND}" ] && echo "after job ${DEPEND}" || echo "no dependency")"
if [ -n "${DRY}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${OUTDIR_NEG}" "${LOGDIR}"

JOB=$(sbatch --parsable ${DEPEND:+--dependency=afterok:${DEPEND} --kill-on-invalid-dep=yes} <<EOT
#!/bin/bash
#SBATCH --job-name=snp_l6neg_${ARM}
#SBATCH --array=0-$(( ${#SEED_LIST[@]} - 1 ))
#SBATCH --time=${TIME}
#SBATCH --output=${LOGDIR}/slurm_%x_%A_%a.log
#SBATCH -A ${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH -C ${CONSTRAINT}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
SEED_ARR=( ${SEED_LIST[@]} )
SEED=\${SEED_ARR[\${SLURM_ARRAY_TASK_ID}]}
python -u snp_experiment/l6_gate.py --arm ${ARM} --cols ${COLS} --window ${WINDOW} || exit 1
GATE=snp_experiment/results/gate/gate_${ARM}.json
if [ "\$(python -c "import json;print(int(json.load(open('\${GATE}'))['violated']))")" != "1" ]; then
    echo "L_6 gate OK: the unconstrained Theta_reg is the result, no L_6<0 solve"; exit 0
fi
LAM=\$(python -c "import json;print(json.load(open('\${GATE}'))['lam'])")
shopt -s nullglob
FILES=( "${REG_SYSTEM_DIR}"/*seed_\${SEED}_*.pt )
if [ \${#FILES[@]} -ne 1 ]; then echo "Error: seed \${SEED}: expected 1 system file, found \${#FILES[@]}"; exit 1; fi
echo "L_6 gate VIOLATED: seed \${SEED}, lam \${LAM}, L_6<0 solve on \${FILES[0]}"
python -u codes/resolve_theta_reg.py "\${FILES[0]}" --lams \${LAM} --ridge ${RIDGE} --schedule uniform --mask auto \
    --diagnose 0 --outdir "${OUTDIR_NEG}" --neg_cols ${COLS} --neg_eps_nats ${EPS} --neg_t_min 0 \
    --neg_max_iter ${MAX_ITER} --neg_method pnewton
EOT
)
if [ -z "${JOB}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOB} (array of ${#SEED_LIST[@]})"
