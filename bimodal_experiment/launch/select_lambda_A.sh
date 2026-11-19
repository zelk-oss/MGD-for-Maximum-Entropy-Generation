#!/bin/bash
# Rule-A (cross-dataset) lam choice for an existing bimodal sweep: select_lambda_A.py on every
# beta dir, one array task per beta, as a dev-QoS job (no computation on login nodes).
# Reads <SWEEP_DIR>/beta_<b>/{experiment_K_runs.pt, reg_system/run_*.pt}; writes
# <SWEEP_DIR>/beta_<b>/lam_selection_A.pt (lam_selection_A_maxruns<N>.pt with MAX_RUNS) and the log
# <SWEEP_DIR>/beta_<b>/select_A.log (select_A_maxruns<N>.log); slurm logs in SWEEP_DIR.
# lam_selection.pt is never touched; an existing lam_selection_A*.pt is skipped unless OVERWRITE=1.
# One sbatch call, no dependency chaining; deletes/moves nothing.
#
# Options (environment):
#   SWEEP_DIR  required, the sweep dir (bimodal_experiment/results_bim_theta_beta/bim_theta_beta_sweep_<ts>)
#   GPU        required, v100 | h100 (dev QoS: qos_gpu-dev | qos_gpu_h100-dev)
#   BETAS      default "0.1000 0.3000 ... 1.9000" (as bimodal_sweep.sh), at most 10 (dev QoS cap)
#   LAMS       default "0 1e-10 3e-10 1e-9 ... 3e-1 1" (half decades; 0 is always added)
#   T_END      default 0.99 (score on t_reg >= T_END)
#   RIDGE      default 0
#   CHUNK      default 5 (runs solved together)
#   MAX_RUNS   smoke test: only the first N runs (empty = all)
#   OVERWRITE  1 to replace an existing output
#   TIME       default 01:00:00, at most 02:00:00
#   CPUS       default 10 (v100, max 10) / 24 (h100, max 24): more would be billed as extra GPUs
#   LABEL      job name suffix, default selectA (put "smoke" in it for a smoke test)
# Example (from the repo root on JZ):
#   GPU=v100 SWEEP_DIR=bimodal_experiment/results_bim_theta_beta/bim_theta_beta_sweep_<ts> \
#       bash bimodal_experiment/launch/select_lambda_A.sh
# Smoke: GPU=v100 SWEEP_DIR=... BETAS="0.5000" MAX_RUNS=5 LAMS="0 1e-6 1e-3" TIME=00:15:00 LABEL=selectA_smoke \
#       bash bimodal_experiment/launch/select_lambda_A.sh

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BM_DIR="$(cd "${LAUNCH_DIR}/.." && pwd)"
if [ -z "${SWEEP_DIR}" ] || [ -z "${GPU}" ]; then echo "Set SWEEP_DIR and GPU (v100|h100)"; exit 1; fi
SWEEP_DIR="$(cd "${SWEEP_DIR}" && pwd)" || { echo "SWEEP_DIR not found"; exit 1; }

read -r -a BETA_LIST <<< "${BETAS:-0.1000 0.3000 0.5000 0.7000 0.9000 1.1000 1.3000 1.5000 1.7000 1.9000}"
read -r -a LAM_LIST <<< "${LAMS:-0 1e-10 3e-10 1e-9 3e-9 1e-8 3e-8 1e-7 3e-7 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1}"
T_END="${T_END:-0.99}"
RIDGE="${RIDGE:-0}"
CHUNK="${CHUNK:-5}"
TIME="${TIME:-01:00:00}"
LABEL="${LABEL:-selectA}"
MODULE="${MODULE:-pytorch-gpu/py3/2.8.0}"
NBETA=${#BETA_LIST[@]}
if [ "${NBETA}" -gt 10 ]; then echo "At most 10 betas per submission (dev QoS cap), got ${NBETA}"; exit 1; fi

IFS=: read -r HH MM SS <<< "${TIME}"
if [ $(( 10#${HH} * 3600 + 10#${MM} * 60 + 10#${SS:-0} )) -gt 7200 ]; then echo "TIME > 02:00:00"; exit 1; fi

case "${GPU}" in
    v100) CPUS="${CPUS:-10}"; MAXC=10; HEADER="#SBATCH -A wbg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-dev
#SBATCH -C v100"
          MODULES="module purge
module load ${MODULE}" ;;
    h100) CPUS="${CPUS:-24}"; MAXC=24; HEADER="#SBATCH -A wbg@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH -C h100"
          MODULES="module purge
module load arch/h100
module load ${MODULE}" ;;
    *) echo "GPU must be v100 or h100"; exit 1 ;;
esac
if [ "${CPUS}" -gt "${MAXC}" ]; then echo "CPUS=${CPUS} > ${MAXC} for ${GPU} (billed as extra GPUs)"; exit 1; fi

EXTRA="--t_end ${T_END} --ridge ${RIDGE} --chunk ${CHUNK}"
SUFFIX=""
if [ -n "${MAX_RUNS}" ]; then EXTRA="${EXTRA} --max_runs ${MAX_RUNS}"; SUFFIX="_maxruns${MAX_RUNS}"; fi
if [ "${OVERWRITE}" = "1" ]; then EXTRA="${EXTRA} --overwrite"; fi

echo "rule A select: ${SWEEP_DIR}"
echo "  betas: ${BETA_LIST[*]}  (${NBETA} array tasks, ${GPU} dev QoS, ${CPUS} CPUs, ${TIME})"
echo "  lams : ${LAM_LIST[*]}"
echo "  ${EXTRA}"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=bimodal_${LABEL}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --array=0-$((NBETA-1))
#SBATCH --output=${SWEEP_DIR}/slurm_%x_%A_%a.log
${HEADER}
${MODULES}
set -x
cd "${BM_DIR}"
BETA_LIST=(${BETA_LIST[@]})
BETA=\${BETA_LIST[\${SLURM_ARRAY_TASK_ID}]}
OUT="${SWEEP_DIR}/beta_\${BETA}"
python -u select_lambda_A.py "\${OUT}" --lams ${LAM_LIST[*]} ${EXTRA} > "\${OUT}/select_A${SUFFIX}.log" 2>&1
EOT
