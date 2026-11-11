#!/bin/bash
# Generalised spectrum of C relative to M (codes/c_spectrum_gen.py) on saved regularised systems, CPU only,
# on the free prepost partition. One task, several systems one after the other.
#
# Usage (Jean Zay, from the repo root):
#   SYSTEMS="<abs path 1.pt> <abs path 2.pt> ..." LABEL=<name> bash codes/launch_c_spectrum_gen.sh
# overrides: TIME (default 01:00:00), CPUS (10), ACCOUNT (wbg@v100, the account that prepost accepted on 11-11),
#   EXTRA (extra arguments of c_spectrum_gen.py, e.g. "--t_targets 0.1 0.5 0.9" or "--ridge 1e-6").
# Output: <repo>/codes/results/c_spectrum_gen/ (npz + png per system) and the table in the SLURM log.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"

: "${SYSTEMS:?set SYSTEMS to the absolute paths of the system .pt files}"
: "${LABEL:?set LABEL}"
TIME="${TIME:-01:00:00}"
CPUS="${CPUS:-10}"
ACCOUNT="${ACCOUNT:-wbg@v100}"
LOG_DIR="${HERE}/saved_results/logs"
mkdir -p "${LOG_DIR}" "${HERE}/results/c_spectrum_gen"

echo "C spectrum (generalised): ${SYSTEMS}, prepost, account ${ACCOUNT}, ${CPUS} CPUs, ${TIME}"

JOBID=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=cspecgen_${LABEL}
#SBATCH -A ${ACCOUNT}
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/slurm_%x_%j.log

module purge
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
python -u "${HERE}/c_spectrum_gen.py" ${SYSTEMS} --out "${HERE}/results/c_spectrum_gen" ${EXTRA}
EOT
)

if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
