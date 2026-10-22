#!/bin/bash
# Queue the standard post-run checks of one arm behind its SDE job (2026-10-22).
#
# Runs, once job DEPEND has finished OK (afterok), on the arm's seed-900 outputs:
#   check_end_moments.py      statistics at t=1, arm and controls
#   window_vs_step_noise.py   (channel, window), width / step noise, error at t=1
#   window_shares.py          per window: fraction of coefficients and share of <z^2>
#   compute_marginal_agreement.py  + the report blocks matching PAIRS
#   compare_scalar_orthog_marginals.py --arms  histograms, controls next to the arm
# Everything goes to turbulence/figures/data/post_<ARM>_<DEPEND>.txt (+ the PNGs in
# turbulence/figures/). A failing step prints FAILED and the next one still runs.
#
# Called by the launchers after `source _submit_turb_sweep.sh` (JOBID set there):
#   DEPEND=${JOBID} ARM=mass_noScat4 LABEL=turbulence_ggd_mass_noScat4 \
#   CONTROLS="noScat4 final" PAIRS="mass at M=256" bash queue_post_checks.sh
# Hardware: H100 dev QoS (starts at once, max 2 h); POST_ON=cpu uses cpu_p1 instead.

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TURB_DIR="$(cd "${LAUNCH_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${TURB_DIR}/.." && pwd)"
LOG_DIR="${TURB_DIR}/saved_results/logs"
mkdir -p "${LOG_DIR}" "${TURB_DIR}/figures/data"

: "${DEPEND:?DEPEND (SDE job id) is required}"
: "${ARM:?ARM is required}"
: "${LABEL:?LABEL (experiment label) is required}"
CONTROLS="${CONTROLS:-}"
PAIRS="${PAIRS:-}"
OUT="${TURB_DIR}/figures/data/post_${ARM}_${DEPEND}.txt"

if [ "${POST_ON:-gpu}" = "cpu" ]; then
    HW="#SBATCH -A wbg@cpu
#SBATCH --partition=cpu_p1
#SBATCH --cpus-per-task=20"
    MODULE_PRE=""
else
    HW="#SBATCH -A wbg@h100
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24"
    MODULE_PRE="arch/h100"
fi

POST_ID=$(sbatch --parsable --dependency=afterok:${DEPEND} <<EOT
#!/bin/bash
#SBATCH --job-name=post_${ARM}
${HW}
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/slurm_%x_%j.log

module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"

run() { echo; echo "######## \$*"; "\$@" || echo "FAILED: \$*"; }
{
  echo "post-run checks for ${ARM} (SDE job ${DEPEND}), controls: ${CONTROLS}"
  run python turbulence/check_end_moments.py --arms ${ARM} ${CONTROLS} --top 15
  run python turbulence/window_vs_step_noise.py --label ${LABEL} --show 30
  run python turbulence/window_shares.py --arm ${ARM} --channels 0 1 2 3
  echo; echo "######## compute_marginal_agreement (report blocks: ${PAIRS})"
  python turbulence/compute_marginal_agreement.py > /dev/null 2>&1 || echo "FAILED: compute_marginal_agreement"
  grep -A 22 -E '^\[(${PAIRS})' turbulence/figures/data/axis_isolations.txt || echo "no matching report block"
  run python turbulence/compare_scalar_orthog_marginals.py --arms ${CONTROLS} ${ARM}
} > "${OUT}" 2>&1
echo "post checks written to ${OUT}"
EOT
)
if [ -z "${POST_ID}" ]; then echo "Error: post-check sbatch returned no job id"; exit 1; fi
echo "Queued post checks ${POST_ID} for ${ARM} after job ${DEPEND} -> ${OUT}"
