#!/bin/bash
# A/B + mechanics check of the terminal phase / final projection (codes/ab_check_terminal.py): the frozen old SDE module vs the new one, tiny
# synthetic problem, flags off must be bit-identical. Dev GPU job, a couple of minutes. Reads nothing from results, writes only under OUT.
#
# Usage (Jean Zay, repo root):
#   GPU=h100 TAG=ab1 NEWMOD=sde_routines_new_candidate bash codes/launch_ab_check_terminal.sh     # candidate copy, before the shared file is pushed
#   GPU=h100 TAG=ab2 bash codes/launch_ab_check_terminal.sh                                       # the shared sde_routines.py itself
# overrides: GPU (v100|h100, mandatory), TAG (mandatory; output folder terminal_phase_test/<TAG> must not exist), NEWMOD (sde_routines), TIME (00:20:00)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
: "${GPU:?set GPU=v100 or GPU=h100}"
: "${TAG:?set TAG}"
NEWMOD="${NEWMOD:-sde_routines}"
TIME="${TIME:-00:20:00}"
OUT="${OUT:-${PROJECT_DIR}/terminal_phase_test/${TAG}}"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE="" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
if [ -e "${OUT}" ]; then echo "Error: ${OUT} exists already (use a new TAG; nothing is overwritten)"; exit 1; fi
mkdir -p "${OUT}"
echo "A/B check: new module ${NEWMOD}, out ${OUT}, ${GPU} dev (${QOS}), ${TIME}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=abcheck_${TAG}
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${OUT}/slurm_%x_%j.log

module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
python -u codes/ab_check_terminal.py ${NEWMOD} > ${OUT}/ab_check.log.txt 2>&1
echo "exit code \$?" >> ${OUT}/ab_check.log.txt
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
