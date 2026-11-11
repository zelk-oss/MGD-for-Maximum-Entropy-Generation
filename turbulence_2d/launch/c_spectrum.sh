#!/bin/bash
# Spectrum of the time-coupling matrix C on saved regularised systems (turbulence_2d/c_spectrum_t2d.py), CPU only,
# on the free prepost partition. One task. Prints, per cell, rank / condition number / weight of mdot and of the
# solution on the weak directions of C (see the script's docstring).
#
# Usage (Jean Zay, from the repo root):
#   LABEL=timing_unif_real_np1300 bash turbulence_2d/launch/c_spectrum.sh
#   SYSTEM_DIR=<dir with the system .pt files> bash ...      # default: $SCRATCH/.../turbulence_2d/reg_system/<LABEL>
# overrides: LABEL (mandatory), SYSTEM_DIR, TIME, CPUS
# Account: prepost needs one but is not billed (IDRIS docs). wbg@cpu is refused (no CPU allocation, 11-11); the
# default below is a GPU-type account of the project, UNTESTED until the first run. ACCOUNT=wbg@h100 to change.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T2_DIR="$(cd "${HERE}/.." && pwd)"
PROJECT_DIR="$(cd "${T2_DIR}/.." && pwd)"

: "${LABEL:?set LABEL, e.g. timing_unif_real_np1300}"
SYSTEM_DIR="${SYSTEM_DIR:-${SCRATCH:?SCRATCH is not set}/MGD-for-Maximum-Entropy-Generation/turbulence_2d/reg_system/${LABEL}}"
TIME="${TIME:-00:30:00}"
CPUS="${CPUS:-10}"
ACCOUNT="${ACCOUNT:-wbg@v100}"
LOG_DIR="${T2_DIR}/saved_results/logs"
mkdir -p "${LOG_DIR}" "${T2_DIR}/results/c_spectrum"

echo "C spectrum: systems in ${SYSTEM_DIR}, prepost, account ${ACCOUNT}, ${CPUS} CPUs, ${TIME}"

JOBID=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=cspec_${LABEL}
#SBATCH -A ${ACCOUNT}
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/slurm_%x_%j.log

module purge
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
shopt -s nullglob
FILES=( "${SYSTEM_DIR}"/*.pt )
if [ \${#FILES[@]} -lt 1 ]; then echo "Error: no system file in ${SYSTEM_DIR}"; exit 1; fi
echo "systems: \${FILES[@]}"
python -u "${T2_DIR}/c_spectrum_t2d.py" "\${FILES[@]}" --out "${T2_DIR}/results/c_spectrum"
EOT
)

if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
