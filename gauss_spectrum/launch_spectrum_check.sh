#!/bin/bash
# Frequency-by-frequency check of the MGD sample entropy vs the exact max-ent (gauss_spectrum/spectrum_check.py), CPU only,
# on the free prepost partition. One task: reads the nt-sweep samples (n1 1e5) of every sigma for one nt, both datasets.
#
# Usage (Jean Zay, repo root):
#   bash gauss_spectrum/launch_spectrum_check.sh            # nt 50000
# overrides: NT, OUT (default gauss_spectrum/nt_sweep), TIME, CPUS, ACCOUNT,
#   PLOT_ONLY=1 (redraw spectrum_check.png from the saved npz, no recomputation; 2026-11-11 after a label bug broke the figure)
# Account / partition: prepost (not charged). 2026-11-11: -A wbg@cpu REFUSED by sbatch ("Invalid account or
# account/partition combination": the project has no CPU allocation) -> default the GPU account wbg@v100.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
NT="${NT:-50000}"
OUT="${OUT:-${HERE}/nt_sweep}"
TIME="${TIME:-00:30:00}"
CPUS="${CPUS:-10}"
ACCOUNT="${ACCOUNT:-wbg@v100}"
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"
echo "spectrum check: ${OUT}, nt ${NT}, prepost, ${CPUS} CPUs, ${TIME}"

JOBID=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=speccheck_${NT}
#SBATCH -A ${ACCOUNT}
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${LOGS}/slurm_%x_%j.log

module purge
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
python -u gauss_spectrum/spectrum_check.py --root "${OUT}" --nt ${NT} --outdir "${OUT}/${SUBDIR:-spectrum_check}" ${PLOT_ONLY:+--plot_only}
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
