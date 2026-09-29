#!/bin/bash
# Offline lam sweep: re-solve Theta_reg for every lam in LAM_LIST from the
# regularised systems saved by the lamtune_*.sh runs (--save_reg_system), with
# codes/resolve_theta_reg.py. CPU only, no SDE re-run. One array task per
# system file; each task loops over all lam values (the ~24 GB system is loaded
# once per task). Already-solved (system, lam) pairs are skipped, so the grid
# can be extended later by adding values to LAM_LIST and resubmitting.
#
# Output: turbulence/saved_results/theta_reg_lamsweep/<config>/lam<lam>_ridge<ridge>[_<mode>][_interp].pt

# Which systems: glob matched inside REG_SYSTEM_DIR (e.g. one potential set)
SYSTEM_PATTERN="${SYSTEM_PATTERN:-*_lamtune_*.pt}"   # override: SYSTEM_PATTERN="..." bash resolve_lamsweep.sh
REG_SYSTEM_DIR="${SCRATCH:+${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system}"

# lam grid: 0 (the unsmoothed per-node solve, the reference lamtune_select.ipynb
# measures residuals against) plus half-decade steps over 1e-8 .. 1e-3
LAM_LIST=(${LAMS:-0 1e-8 3e-8 1e-7 3e-7 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3})   # override: LAMS="0 1e-4 ..."
# Energy solved (codes/resolve_theta_reg.py --mode): legacy (old system, wrong-sign
# time term), fixed (correct centred time term), guth (correct term + Guth et al.'s
# weights; lam is then a multiplier, 1 = theirs). fixed/guth read m from
# saved_results/aux_moments/ for systems saved before 2026-09-29; guth needs DIM.
MODE="${MODE:-legacy}"
DIM="${DIM:-}"
# Time terms from the walkers (all systems) or the interpolant samples (TIME_SOURCE=
# interpolant; systems saved from 2026-09-29 on, needs MODE fixed or guth)
TIME_SOURCE="${TIME_SOURCE:-walkers}"
# Ridge on the data term, M_k += RIDGE * diag(M_k), same for every lam (incl. the
# lam=0 reference). Needed: with RIDGE=0 every solve failed as singular (job 146575):
# the M_k/G_k blocks are exactly rank-deficient (statistics with identical gradients),
# which the in-run per-step solves never saw thanks to their own 0.01 ridge.
RIDGE="${RIDGE:-1e-6}"

# SLURM (CPU). Peak RAM ~ 2 x system size (~50 GB for nt=40000, r=272); on
# Jean Zay host memory scales with --cpus-per-task, so CPUS sets the memory too.
#
# CPU_MODE=true: a CPU partition (needs CPU hours on the project; wbg@cpu/cpu_p1
# was refused: "Invalid account or account/partition combination").
# CPU_MODE=false: run on the H100 allocation instead. The solve itself is CPU-only
# (codes/resolve_theta_reg.py never touches the GPU); a GPU is requested only
# because gpu_p6 jobs must take one. Costs ~2-3 H100-hours per system.
CPU_MODE=false
if [ "${CPU_MODE}" = "true" ]; then
    ACCOUNT="wbg@cpu"            # check: sacctmgr -n show assoc user=$USER format=account,partition
    PARTITION="cpu_p1"
    CONSTRAINT=""
    GRES=""
    CPUS=20
    MODULE_PRE=""
else
    ACCOUNT="wbg@h100"
    PARTITION="gpu_p6"
    CONSTRAINT="h100"
    GRES="gpu:1"
    CPUS=24                      # a quarter of an H100 node; host RAM scales with it
    MODULE_PRE="arch/h100"
fi
TIME="06:00:00"
MODULE="pytorch-gpu/py3/2.8.0"

# ── paths ────────────────────────────────────────────
LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TURB_DIR="$(cd "${LAUNCH_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${TURB_DIR}/.." && pwd)"
REG_SYSTEM_DIR="${REG_SYSTEM_DIR:-${TURB_DIR}/saved_results/reg_system}"
OUT_DIR="${TURB_DIR}/saved_results/theta_reg_lamsweep"
LOG_DIR="${TURB_DIR}/saved_results/logs"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

shopt -s nullglob
FILES=( "${REG_SYSTEM_DIR}"/${SYSTEM_PATTERN} )
shopt -u nullglob
if [ ${#FILES[@]} -eq 0 ]; then
    echo "No system files match ${REG_SYSTEM_DIR}/${SYSTEM_PATTERN}"; exit 1
fi
echo "Re-solving ${#FILES[@]} systems for ${#LAM_LIST[@]} lam values (ridge ${RIDGE}, mode ${MODE}${DIM:+, dim ${DIM}}, time terms: ${TIME_SOURCE}):"
printf '  %s\n' "${FILES[@]##*/}"

JOBID=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=resolve_lamsweep
#SBATCH -A ${ACCOUNT}
#SBATCH --array=0-$(( ${#FILES[@]} - 1 ))
#SBATCH --partition=${PARTITION}
${CONSTRAINT:+#SBATCH -C ${CONSTRAINT}}
${GRES:+#SBATCH --gres=${GRES}}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/slurm_%x_%A_%a.log

module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load ${MODULE}
cd "${PROJECT_DIR}"

FILES=( ${FILES[@]} )
python codes/resolve_theta_reg.py "\${FILES[\${SLURM_ARRAY_TASK_ID}]}" \
    --lams ${LAM_LIST[@]} \
    --ridge ${RIDGE} \
    --mode ${MODE} ${DIM:+--dim ${DIM}} --time_source ${TIME_SOURCE} \
    --results_root "${TURB_DIR}" \
    --diagnose 5 \
    --outdir "${OUT_DIR}"
EOT
)

if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted array job ${JOBID}"
