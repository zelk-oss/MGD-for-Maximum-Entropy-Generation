#!/bin/bash
# Theta_reg(1) with vs without the terminal-phase nodes (terminal_phase_test/compare_theta_droplast.py), 2026-11-25.
# CPU work as a GPU dev job (user rule: never prepost; billed 1 GPU x elapsed).
# Usage (Jean Zay, repo root): GPU=h100 SYSTEM=<system .pt> ALL=<Theta file> DROP=<Theta file> [NAMES=<csv>] OUT=<dir> \
#   bash terminal_phase_test/launch_compare_theta_droplast.sh
# overrides: TIME (00:15:00), CPUS (24)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
: "${GPU:?set GPU=v100 or GPU=h100}"
: "${OUT:?}"
# ARGS (2026-11-25): raw script arguments instead of SYSTEM/ALL/DROP, e.g. the cross-seed mode
# ARGS="--cross_systems S900 S901 S902 --cross_all A900 A901 A902 --cross_drop B900 B901 B902"
[ -n "${ARGS:-}" ] || { : "${SYSTEM:?}"; : "${ALL:?}"; : "${DROP:?}"; }
TIME="${TIME:-00:15:00}"
CPUS="${CPUS:-24}"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE="" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"
EXTRA=""; [ -n "${NAMES:-}" ] && EXTRA="--names ${NAMES}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=thdropcmp
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
python -u terminal_phase_test/compare_theta_droplast.py ${ARGS:---system "${SYSTEM}" --all "${ALL}" --drop "${DROP}" ${EXTRA}} --out "${OUT}"
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
