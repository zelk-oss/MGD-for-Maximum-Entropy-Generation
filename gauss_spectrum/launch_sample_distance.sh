#!/bin/bash
# Calibration of a sample-based sigma-convergence criterion on the Gaussian M-scan samples (sample_distance_calibration.py):
# self-test first, then M = 32 and 64 (sigma 1/2/5 at nt 1.6M, nt 400k controls, noise floors from half-sample splits).
# One H100 dev job (reads ~12 sample files of 1-3 MB, pairwise distances on 5000 x 5000 points: minutes).
# Usage (Jean Zay, repo root): bash gauss_spectrum/launch_sample_distance.sh
# overrides: OUT (default gauss_spectrum/m_scan), TIME (default 00:30:00)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
OUT="${OUT:-${HERE}/m_scan}"
TIME="${TIME:-00:30:00}"
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"

JOBID=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=sampdist
#SBATCH -A wbg@h100
#SBATCH -C h100
#SBATCH --partition=gpu_p6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=${TIME}
#SBATCH --output=${LOGS}/slurm_%x_%j.log
#SBATCH --qos=qos_gpu_h100-dev
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
echo "======== selftest"
python -u gauss_spectrum/sample_distance_calibration.py --selftest
echo "======== M scan samples"
python -u gauss_spectrum/sample_distance_calibration.py --root "${OUT}" --Ms 32 64 --outdir "${OUT}/sample_distance"
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
