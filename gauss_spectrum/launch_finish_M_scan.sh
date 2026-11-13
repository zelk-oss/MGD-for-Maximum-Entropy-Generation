#!/bin/bash
# Analysis of the M scan (launch_M_scan_dev.sh), CPU only, free prepost partition, ONE task:
#   for M in 32 64: entropy benchmark of every run (finish_sigma_sweep.sh -> m_scan/M<M>/analysis, summary/endpoint.csv),
#   then the figure gauss_spectrum/entropy_vs_sigma_lagrangian_M<M>.png (bound / true value / MGD entropy vs sigma, one curve per nt).
# Run it after the dev array AND the t3 nt 1.6M array have finished (the benchmark reads each run's Theta_reg file).
#
# Usage (Jean Zay, repo root): bash gauss_spectrum/launch_finish_M_scan.sh
# GPU=h100: run on the H100 dev QoS instead of prepost (prepost can pend for a long time; costs GPU x elapsed).
# overrides: MS (default "32 64"), OUT (default gauss_spectrum/m_scan), TIME, CPUS, ACCOUNT, ZOOM ("LO HI", default "-15 45")
# Account/partition as launch_spectrum_check.sh (prepost, project account wbg@v100).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
MS=(${MS:-32 64})
OUT="${OUT:-${HERE}/m_scan}"
TIME="${TIME:-01:30:00}"
CPUS="${CPUS:-10}"
ACCOUNT="${ACCOUNT:-wbg@v100}"
ZOOM="${ZOOM:--15 45}"
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"
echo "M scan analysis: M ${MS[*]}, ${OUT}, prepost, ${CPUS} CPUs, ${TIME}"

if [ "${GPU:-}" = "h100" ]; then
    # 2026-11-12: prepost was pending for a long time -> H100 dev (<= 2 h, 1 GPU, billed GPU x elapsed)
    HEAD="#SBATCH -A wbg@h100
#SBATCH -C h100
#SBATCH --partition=gpu_p6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --qos=qos_gpu_h100-dev"
    PRE="module load arch/h100"
else
    HEAD="#SBATCH -A ${ACCOUNT}
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread"
    PRE=""
fi

JOBID=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=mscan_fin
${HEAD}
#SBATCH --time=${TIME}
#SBATCH --output=${LOGS}/slurm_%x_%j.log

module purge
${PRE}
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
for M in ${MS[@]}; do
    echo "======== M \${M}"
    OUT="${OUT}/M\${M}" bash gauss_spectrum/finish_sigma_sweep.sh
    python -u gauss_spectrum/entropy_vs_sigma.py --csv "${OUT}/M\${M}/summary/endpoint.csv" --dataset turbulence \
        --zoom ${ZOOM} --title "Lagrangian turbulence spectrum, M=\${M}" \
        --out "gauss_spectrum/entropy_vs_sigma_lagrangian_M\${M}.png"
done
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
