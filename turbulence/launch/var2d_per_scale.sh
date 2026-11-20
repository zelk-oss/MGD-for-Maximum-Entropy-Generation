#!/bin/bash
# Var/2d of theta.phi per wavelet scale for Scalar_psi and Scalar_morlet (B+Scat4, Theta_reg lam 1e-3 = rule A),
# on the data and on the model samples (2026-11-20). turbulence/var2d_per_scale/var2d_per_scale.py.
# CPU work as a GPU dev job (user rule 11-17: never prepost; billed 1 GPU x elapsed).
# Usage (Jean Zay, repo root):  GPU=v100 bash turbulence/launch/var2d_per_scale.sh
# overrides: GPU (v100|h100, mandatory), TIME (00:20:00), CPUS (10 on v100, 24 on h100), RUN_LABEL, SEED, LAM, N_MAX, OUT

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${LAUNCH_DIR}/../.." && pwd)"
: "${GPU:?set GPU=v100 or GPU=h100}"
TIME="${TIME:-00:20:00}"
SCRIPT="${SCRIPT:-var2d_per_scale}"      # or var2d_scat4_triples (Scat4 per (j, j', l))
RUN_LABEL="${RUN_LABEL:-turbulence_nested_M3_bicw_allmass_scat4}"
SEED="${SEED:-900}"
LAM="${LAM:-1e-3}"
OUT="${OUT:-${PROJECT_DIR}/turbulence/var2d_per_scale/results}"
EXTRA=""
[ -n "${N_MAX:-}" ] && EXTRA="--n_max ${N_MAX}"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE=""; CPUS="${CPUS:-10}" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100"; CPUS="${CPUS:-24}" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"
echo "var2d_per_scale: ${RUN_LABEL} seed ${SEED} lam ${LAM} -> ${OUT}; ${GPU} dev (${QOS}), ${CPUS} CPUs, ${TIME}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=var2d
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${LOGS}/slurm_%x_%j.log

module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
export OMP_NUM_THREADS=${CPUS}
cd "${PROJECT_DIR}"
python -u turbulence/var2d_per_scale/${SCRIPT}.py --label ${RUN_LABEL} --seed ${SEED} --lam ${LAM} --out "${OUT}" ${EXTRA}
echo "=== done \$(date)"
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
