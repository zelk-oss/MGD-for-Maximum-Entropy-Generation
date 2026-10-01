#!/bin/bash
# Offline lam sweep: re-solve Theta_reg for every lam in LAM_LIST from the
# regularised systems saved by the lamtune_*.sh runs (--save_reg_system), with
# codes/resolve_theta_reg.py. CPU only, no SDE re-run. One array task per
# system file; each task loops over all lam values (the ~24 GB system is loaded
# once per task). Already-solved (system, lam) pairs are skipped, so the grid
# can be extended later by adding values to LAM_LIST and resubmitting.
#
# Output: turbulence/saved_results/theta_reg_lamsweep/<config>/lam<lam>_ridge<ridge>[_moment[_guthsched]][_mask].pt
#         (SELECT=1: one lam_path_ridge<ridge><tag>.pt per system with Theta for every lam)

# Which systems: glob matched inside REG_SYSTEM_DIR (e.g. one potential set)
SYSTEM_PATTERN="${SYSTEM_PATTERN:-*_lamtune_*.pt}"   # override: SYSTEM_PATTERN="..." bash resolve_lamsweep.sh
REG_SYSTEM_DIR="${SCRATCH:+${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system}"

# lam grid: 0 (the unsmoothed per-node solve, the reference lamtune_select.ipynb
# measures residuals against) plus half-decade steps over 1e-8 .. 1e-3
LAM_LIST=(${LAMS:-0 1e-8 3e-8 1e-7 3e-7 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3})   # override: LAMS="0 1e-4 ..."
# Energy solved (codes/resolve_theta_reg.py --mode; notes/guth_reg_audit_0930):
#   moment -- time term Sigma_w Theta_dot = mdot. On systems saved in legacy mode it is
#             rebuilt from saved_results/aux_moments/ (Sigma = G - m m^T, mdot = finite
#             difference of barphi_e). SCHEDULE=guth: Guth et al.'s weights (needs DIM).
#   legacy -- the pre-2026-09-30 tau-based energy (reproduction only).
# Empty MODE = the system's own mode (legacy for files saved before 2026-09-30).
MODE="${MODE:-}"
SCHEDULE="${SCHEDULE:-uniform}"
DIM="${DIM:-}"
# Pin dead potentials to 0 per node: auto (saved live mask if any) | none | system | diag | theta
# (theta: rebuilt from the run's saved theta_t, which is exactly 0 where the corrector masked)
MASK="${MASK:-auto}"
# SELECT=1: lam selection (held-out moment-matching loss + amplitude veto), one file
SELECT="${SELECT:-0}"
# Ridge on the data term, M_k += RIDGE * diag(M_k), same for every lam (incl. the
# lam=0 reference). Needed: with RIDGE=0 every solve failed as singular (job 146575):
# the M_k blocks are exactly rank-deficient (statistics with identical gradients).
# Default = the per-step ridge of the zfloor/long runs (--regularization 1e-4), with
# which lam = 0 reproduces theta_t except where potentials were masked; set it to the
# run's --regularization for other runs (the July batch used 1e-2).
RIDGE="${RIDGE:-1e-4}"

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
    CPUS="${CPUS:-24}"           # a quarter of an H100 node; host RAM scales with it
                                 # (--select at nt 60000, r 281 peaks ~80 GB: system 38 + Thomas c' 38)
    MODULE_PRE="arch/h100"
    # dev QoS: starts almost immediately, but max 2 h; QOS="" falls back to the default
    # (t3, 20 h, queued). The array is all-or-nothing per task: a task killed at the
    # time limit saves nothing, so split LAMS over two launches if 2 h is too short.
    QOS="${QOS-qos_gpu_h100-dev}"
fi
TIME="${TIME:-$( [ -n "${QOS}" ] && [[ "${QOS}" == *-dev ]] && echo 02:00:00 || echo 06:00:00 )}"
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
echo "QoS ${QOS:-default}, time ${TIME}, ${CPUS} CPUs"
echo "Re-solving ${#FILES[@]} systems for ${#LAM_LIST[@]} lam values (ridge ${RIDGE}, mode ${MODE:-from file}, schedule ${SCHEDULE}${DIM:+, dim ${DIM}}, mask ${MASK}, select ${SELECT}):"
printf '  %s\n' "${FILES[@]##*/}"

JOBID=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=resolve_lamsweep
#SBATCH -A ${ACCOUNT}
#SBATCH --array=0-$(( ${#FILES[@]} - 1 ))
#SBATCH --partition=${PARTITION}
${QOS:+#SBATCH --qos=${QOS}}
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
    ${MODE:+--mode ${MODE}} --schedule ${SCHEDULE} ${DIM:+--dim ${DIM}} \
    --mask ${MASK} $( [ "${SELECT}" = "1" ] && echo --select ) \
    --results_root "${TURB_DIR}" \
    --diagnose 5 \
    --outdir "${OUT_DIR}"
EOT
)

if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted array job ${JOBID}"
