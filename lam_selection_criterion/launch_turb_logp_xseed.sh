#!/bin/bash
# Cross-seed coherence of log p vs lam (turb_logp_xseed.py) as one GPU dev job; existing Theta files only.
# Usage (repo root on Jean Zay):
#   GPU=h100 ARM=turbulence_nested_M3_bicw_allmass_scat4 SEEDS="900 901 902 903 904" RIDGE=1e-4 \
#     SWEEP_DIR=$SCRATCH/MGD-for-Maximum-Entropy-Generation/turbulence/theta_reg_lamsweep_xseed/bscat4 \
#     LAMS="0 1e-6 1e-5 1e-4 1e-3 1e-2" bash lam_selection_criterion/launch_turb_logp_xseed.sh
# settings: GPU=h100|v100, ARM, SEEDS, RIDGE, SWEEP_DIR, LAMS (required); OUT, TIME (default 00:45:00), CPUS,
#   NSIG (first N signals only, smoke), LABEL (job name suffix).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
for v in GPU ARM SEEDS RIDGE SWEEP_DIR LAMS; do
    if [ -z "${!v}" ]; then echo "Set ${v}"; exit 1; fi
done
OUT="${OUT:-${HERE}/results/turb_logp_xseed}"
TIME="${TIME:-00:45:00}"
JOBNAME="logp_xseed_${LABEL:-run}"
NSIG_ARG=""; [ -n "${NSIG}" ] && NSIG_ARG="--n_signals ${NSIG}"
case "${GPU}" in
    h100) CPUS="${CPUS:-24}"; HEADER="#SBATCH -A wbg@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH -C h100"
          MODULES="module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0" ;;
    v100) CPUS="${CPUS:-10}"; HEADER="#SBATCH -A wbg@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH -C v100-32g"
          MODULES="module purge
module load pytorch-gpu/py3/2.8.0" ;;
    *) echo "GPU must be h100 or v100"; exit 1 ;;
esac
echo "log p coherence: ${ARM}, seeds ${SEEDS}, lams ${LAMS}, ridge ${RIDGE} | sweep ${SWEEP_DIR} | out ${OUT} | ${GPU} dev, ${CPUS} CPUs, ${TIME}"
if [ -n "${DRY}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${OUT}"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOBNAME}
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
python -u lam_selection_criterion/turb_logp_xseed.py --label ${ARM} --seeds ${SEEDS} --sweep_dir "${SWEEP_DIR}" \
    --lams ${LAMS} --ridge ${RIDGE} --out "${OUT}" ${NSIG_ARG}
EOT
