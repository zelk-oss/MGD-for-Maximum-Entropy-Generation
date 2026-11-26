#!/bin/bash
# Theta_reg with and without the terminal-phase rows, from ONE saved system (2026-11-18).
# The run must have been launched with TERMINAL_PHASE=true TERMINAL_IN_SYSTEM=true: its system then holds the planned
# steps + the terminal steps. Two offline solves of codes/resolve_theta_reg.py on that same system (same walkers):
#   A  all nodes                      -> <OUT>/<config>/lam<lam>_ridge<ridge>_moment_mask.pt
#   B  --drop_last N (terminal rows)  -> <OUT>/<config>/lam<lam>_ridge<ridge>_moment_mask_droplastN.pt
# N = number of system nodes after the end of the planned grid (terminal_phase_test/count_terminal_rows.py, read at run
# time). Until 2026-11-25 N was len(terminal log): wrong when a terminal step is missing from the system (smoke 717357).
# CPU-only work as a GPU dev job (user rule 11-17: never prepost; billed 1 GPU x elapsed).
#
# Usage (Jean Zay, repo root):
#   GPU=h100 CONFIG=<config name> LAMS="1e-5 1e-4 1e-3" bash terminal_phase_test/launch_theta_reg_insys.sh
# overrides: GPU (v100|h100, mandatory), CONFIG (mandatory), REG_DIR (folder of the system, mandatory: the run's
# --reg_system_dir), LAMS ("1e-5 1e-4 1e-3"), RIDGE (1e-4 = the runs' --regularization), OUT
# (terminal_phase_test/theta_reg_insys), TIME (01:00:00), CPUS (24: host RAM 5 GB/CPU on H100),
# EXTRA_ARGS (2026-11-25: passed to both solves, e.g. "--pin_l6 0:9 --pin_eps_nats 5e-3 --pin_mode pos"; pinpos reuses
# the free file of the same lam in OUT if it exists, so run once without EXTRA_ARGS first)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
: "${GPU:?set GPU=v100 or GPU=h100}"
: "${CONFIG:?set CONFIG=<config name of the run>}"
: "${REG_DIR:?set REG_DIR=<the run reg_system_dir>}"
LAMS="${LAMS:-1e-5 1e-4 1e-3}"
RIDGE="${RIDGE:-1e-4}"
OUT="${OUT:-${PROJECT_DIR}/terminal_phase_test/theta_reg_insys}"
TIME="${TIME:-01:00:00}"
CPUS="${CPUS:-24}"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE="" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
SYSTEM="${REG_DIR}/${CONFIG}.pt"
TERM="${PROJECT_DIR}/turbulence/saved_results/aux_moments/${CONFIG}_terminal.pt"
[ -f "${SYSTEM}" ] || { echo "Error: no system ${SYSTEM}"; exit 1; }
[ -f "${TERM}" ]   || { echo "Error: no terminal log ${TERM}"; exit 1; }
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"
echo "theta_reg insys: ${CONFIG}, lams ${LAMS}, ridge ${RIDGE}, out ${OUT}, ${GPU} dev (${QOS}), ${CPUS} CPUs, ${TIME}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=thregins
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${LOGS}/slurm_%x_%j.log

module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
export OMP_NUM_THREADS=${CPUS}
cd "${PROJECT_DIR}"
TIMES=\$(ls -t ${PROJECT_DIR}/turbulence/saved_results/sampling_times/${CONFIG}* | head -1)
N=\$(python -u terminal_phase_test/count_terminal_rows.py "${SYSTEM}" "${TERM}" "\${TIMES}")
echo "terminal rows in the system: \${N}"
if [ "\${N}" -le 0 ]; then echo "FAILED: the run was not launched with TERMINAL_IN_SYSTEM (or has no terminal steps)"; exit 1; fi
echo "=== A: all nodes"
python -u codes/resolve_theta_reg.py "${SYSTEM}" --lams ${LAMS} --ridge ${RIDGE} --outdir "${OUT}" --mmap ${EXTRA_ARGS:-} || echo "FAILED A"
echo "=== B: --drop_last \${N}"
python -u codes/resolve_theta_reg.py "${SYSTEM}" --lams ${LAMS} --ridge ${RIDGE} --outdir "${OUT}" --mmap --drop_last \${N} ${EXTRA_ARGS:-} || echo "FAILED B"
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
