#!/bin/bash
# Combine step of the bimodal lam test (bimodal_lamtest.py --combine_only) as a short dev job
# (no computation on login nodes): pools every run_*.pt of ${OUT}/beta_*, prints the tables
# (incl. the rule A / B / C comparison, 2026-11-18) to the slurm log in ${OUT}. Writes nothing else.
# Usage: GPU=h100|v100 OUT=<results dir> bash lam_selection_criterion/launch_bimodal_combine.sh   (TIME, CPUS, LABEL)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
if [ -z "${OUT}" ] || [ -z "${GPU}" ]; then echo "Set OUT and GPU (h100|v100)"; exit 1; fi
TIME="${TIME:-00:30:00}"
LABEL="${LABEL:-combine}"
case "${GPU}" in
    h100) CPUS="${CPUS:-8}"; HEADER="#SBATCH -A wbg@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH -C h100"
          MODULES="module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0" ;;
    v100) CPUS="${CPUS:-8}"; HEADER="#SBATCH -A wbg@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH -C v100-32g"
          MODULES="module purge
module load pytorch-gpu/py3/2.8.0" ;;
    *) echo "GPU must be h100 or v100"; exit 1 ;;
esac
echo "combine ${OUT} | ${GPU} dev QoS, ${CPUS} CPUs, ${TIME}"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=bimodal_${LABEL}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${OUT}/slurm_%x_%j.log
${HEADER}
${MODULES}
cd "${PROJECT_DIR}"
python -u lam_selection_criterion/bimodal_lamtest.py --out "${OUT}" --combine_only
EOT
