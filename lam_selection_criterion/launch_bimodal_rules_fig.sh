#!/bin/bash
# Figures + table of the A/B/C lam-rule comparison on the bimodal runs (bimodal_rules_comparison.py, the script
# version of bimodal_rules_comparison.ipynb) as a short GPU dev job (CPU work; billed GPU x elapsed).
# Reads ${RES}/beta_*/run_*_s*.pt, writes only 2 PNGs + the slurm log (with the table) into ${OUT} (a NEW folder).
# Usage: GPU=h100|v100 RES=<bimodal results dir> OUT=<new figure dir> bash lam_selection_criterion/launch_bimodal_rules_fig.sh
#   (TIME default 00:20:00, CPUS, LABEL)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
if [ -z "${RES}" ] || [ -z "${OUT}" ] || [ -z "${GPU}" ]; then echo "Set GPU (h100|v100), RES and OUT"; exit 1; fi
TIME="${TIME:-00:20:00}"
LABEL="${LABEL:-rules_fig}"
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
echo "rules figures: ${RES} -> ${OUT} | ${GPU} dev QoS, ${CPUS} CPUs, ${TIME}"
mkdir -p "${OUT}"

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
MPLBACKEND=Agg BM_XSEED_DIR="${RES}" BM_FIG_DIR="${OUT}" python -u lam_selection_criterion/bimodal_rules_comparison.py
EOT
