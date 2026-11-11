#!/bin/bash
# Method A (c_floor_eps) on saved regularised systems, CPU only, prepost partition. One task.
#   MODE=check : codes/check_c_floor_eps0.py (eps = 0 vs the Theta_reg files saved before the change)
#   MODE=scan  : codes/resolve_theta_reg.py --c_floor_eps EPS over LAMS (one solve per (lam, eps))
#
# Usage (Jean Zay, from the repo root):
#   MODE=check SYSTEMS=<one abs .pt> SAVED_DIR=<abs dir of the old theta files> LAMS="1e-4 1e-2 1" LABEL=x bash codes/launch_c_floor.sh
#   MODE=scan  SYSTEMS="<abs .pt> ..." OUTDIR=<abs dir> EPS="0 5e-5 5e-4 5e-3 5e-2" LABEL=x bash codes/launch_c_floor.sh
# STRIDE=m (scan only): keep every m-th node (grid-dependence test, default 1). One stride per job.
# overrides: LAMS (default 0 1e-6 1e-5 1e-4 5e-4 1e-3 1e-2 1e-1 1), RIDGE (1e-7), TIME (10:00:00), CPUS (20),
#   ACCOUNT (wbg@v100: the account prepost accepted on 11-11; prepost is not billed per the IDRIS docs).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"

: "${MODE:?set MODE=check or scan}"
: "${SYSTEMS:?set SYSTEMS}"
: "${LABEL:?set LABEL}"
LAMS="${LAMS:-0 1e-6 1e-5 1e-4 5e-4 1e-3 1e-2 1e-1 1}"
RIDGE="${RIDGE:-1e-7}"
EPS="${EPS:-0}"
TIME="${TIME:-10:00:00}"
CPUS="${CPUS:-20}"
ACCOUNT="${ACCOUNT:-wbg@v100}"
LOG_DIR="${HERE}/saved_results/logs"
mkdir -p "${LOG_DIR}"
case "${MODE}" in
    check) : "${SAVED_DIR:?set SAVED_DIR}"
           CMD="python -u ${HERE}/check_c_floor_eps0.py ${SYSTEMS} ${SAVED_DIR} --lams ${LAMS} --ridge ${RIDGE}" ;;
    scan)  : "${OUTDIR:?set OUTDIR}"
           CMD="python -u ${HERE}/resolve_theta_reg.py ${SYSTEMS} --lams ${LAMS} --ridge ${RIDGE} --c_floor_eps ${EPS} --node_stride ${STRIDE:-1} --schedule uniform --mask auto --diagnose 0 --mmap --threads ${CPUS} --outdir ${OUTDIR}" ;;
    *) echo "Error: MODE must be check or scan"; exit 1 ;;
esac
# GPU=v100: run on a V100 dev node instead (billed 1 GPU x elapsed, <= 2 h; the solve itself uses the CPU cores).
# The prepost queue can be long; dev starts in minutes when GPUs are free.
if [ "${GPU:-}" = "v100" ]; then
    ACCOUNT="wbg@v100"; CPUS=10
    PLACE="#SBATCH --partition=gpu_p13
#SBATCH -C v100
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-dev"
else
    PLACE="#SBATCH --partition=prepost"
fi
echo "c_floor ${MODE}: ${CMD}"
echo "${GPU:-prepost}, account ${ACCOUNT}, ${CPUS} CPUs, ${TIME}"

JOBID=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=cfloor_${MODE}_${LABEL}
#SBATCH -A ${ACCOUNT}
${PLACE}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/slurm_%x_%j.log

module purge
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
${CMD}
EOT
)

if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
