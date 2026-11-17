#!/bin/bash
# H_G = D_KL(p || q_G) per group of potentials (kl_groups/): cache the per-term energies Theta_k phi_k on the generated
# signals (GPU), then compute H for all groups (CPU, same job). Reads existing samples / Theta_reg / fitted potentials;
# writes only under OUT = kl_groups/results/<TAG>. Dev GPU job.
#
# Usage (Jean Zay, repo root):
#   GPU=v100 TAG=smoke N_MAX=200 DATASETS="fbm:300 lagrangian:900" bash kl_groups/launch_kl_groups.sh
#   GPU=v100 TAG=run1 DATASETS="fbm:300,301,302,303,304,305,306,307,308,309 lagrangian:900,901,902,903,904" bash ...
# DATASETS = space-separated NAME:SEEDS (comma separated); NAME in lagrangian jets snp cosmic fbm.
# overrides: GPU (v100|h100, mandatory), TAG (mandatory, new folder), N_MAX (keep the first N_MAX samples, smoke only),
#   N_BOOT (200), TIME (01:00:00), CPUS (10).
# A dataset that fails does not stop the others (its log is <OUT>/<NAME>.log.txt).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
: "${DATASETS:?set DATASETS, see the header}"
: "${GPU:?set GPU=v100 or GPU=h100}"
: "${TAG:?set TAG (output folder kl_groups/results/<TAG>)}"
OUT="${PROJECT_DIR}/kl_groups/results/${TAG}"
TIME="${TIME:-01:00:00}"
CPUS="${CPUS:-10}"
N_BOOT="${N_BOOT:-200}"
NMAX=""; [ -n "${N_MAX:-}" ] && NMAX="--n_max ${N_MAX}"
# ON_DATA=1: energies on the TRAINING windows (cache_energies --on_data, lagrangian only), no compute_H (not samples of p)
ONDATA=""; [ -n "${ON_DATA:-}" ] && ONDATA="--on_data"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE="" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
if [ -e "${OUT}" ]; then echo "Error: ${OUT} exists already (use a new TAG; nothing is overwritten)"; exit 1; fi
mkdir -p "${OUT}/slurm"

CMDS=""
for D in ${DATASETS}; do
  IFS=: read -r NAME SEEDS <<< "${D}"
  if [ -n "${FROM_TAG:-}" ]; then      # FROM_TAG=run1: no energy caching, compute_H on results/<FROM_TAG>/E_*.npz -> <TAG>
    CMDS+="python -u kl_groups/compute_H.py --dataset ${NAME} --n_boot ${N_BOOT} --in_dir ${PROJECT_DIR}/kl_groups/results/${FROM_TAG} > ${OUT}/${NAME}.log.txt 2>&1 || echo FAILED ${D}"$'\n'
  elif [ -n "${ONDATA}" ]; then
    CMDS+="python -u kl_groups/cache_energies.py --dataset ${NAME} --seeds ${SEEDS//,/ } ${NMAX} --on_data > ${OUT}/${NAME}_train.log.txt 2>&1 || echo FAILED ${D}"$'\n'
  else
    CMDS+="( python -u kl_groups/cache_energies.py --dataset ${NAME} --seeds ${SEEDS//,/ } ${NMAX} && python -u kl_groups/compute_H.py --dataset ${NAME} --n_boot ${N_BOOT} ) > ${OUT}/${NAME}.log.txt 2>&1 || echo FAILED ${D}"$'\n'
  fi
done
echo "kl_groups: ${DATASETS}, out ${OUT}, ${GPU} dev (${QOS}), ${CPUS} CPUs, ${TIME}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=klgroups_${TAG}
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${OUT}/slurm/slurm_%x_%j.log

module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
export OMP_NUM_THREADS=${CPUS}
export KL_RESULTS="${OUT}"
cd "${PROJECT_DIR}"
${CMDS}
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
