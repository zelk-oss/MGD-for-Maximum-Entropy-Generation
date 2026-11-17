#!/bin/bash
# Per-statistic errors of the two B terminal-phase runs (jobs 78221, 78222), CPU-only analysis as a short GPU dev job
# (user rule 11-17: never prepost; billed 1 GPU x elapsed, expect well under 1 GPU-h). One task, three steps per run:
#   1. terminal_phase_test/terminal_phase_errors.py : error of every statistic BEFORE (end of the planned grid, same
#      walkers) and AFTER the terminal phase, paired within the same run (reads only aux_moments + terminal.pt).
#   2. turbulence/moment_error_variance.py --dump   : the usual end-of-run table (z in SE of two samples, peak along the
#      run, Ito term), computed on the saved final samples; it also rebuilds phi on the samples and checks it against the
#      last aux row ("check:" line). Reads samples, sampling_times, fitted_potentials and the data.
# Output: ${OUT}/*.txt (reports), ${OUT}/*.csv (all statistics), ${OUT}/logs/slurm_*.log. Nothing outside OUT is written.
#
# Usage (Jean Zay, repo root):  GPU=h100 bash terminal_phase_test/launch_terminal_errors.sh
# overrides: GPU (v100|h100, mandatory), OUT (terminal_phase_test/errors), TIME (01:00:00), CPUS (10), SEED (900)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
: "${GPU:?set GPU=v100 or GPU=h100}"
SEED="${SEED:-900}"
OUT="${OUT:-${PROJECT_DIR}/terminal_phase_test/errors}"
TIME="${TIME:-01:00:00}"
CPUS="${CPUS:-10}"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE="" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"

# LABEL : sigma-substring : nt-substring  (label as in the config name, followed by _2<timestamp>)
RUNS="turbulence_nested_M3_bicw_allmass_scat4_pf_term:3.5:27500 turbulence_nested_M3_bicw_allmass_pf_term:3.5:55000"
CMDS=""
for R in ${RUNS}; do
  IFS=: read -r LABEL SIG NTV <<< "${R}"
  TAG="${LABEL}_sigma${SIG}_nt${NTV}"
  CMDS+="python -I -u terminal_phase_test/terminal_phase_errors.py --label ${LABEL} --seed ${SEED} --config _sigma${SIG}_ _nt${NTV}_ --out ${OUT} > ${OUT}/before_after_${TAG}.txt 2>&1 || echo FAILED before_after ${R}"$'\n'
  CMDS+="python -u turbulence/moment_error_variance.py --label ${LABEL} --seeds ${SEED} --config _sigma${SIG}_ _nt${NTV}_ --dump ${OUT} > ${OUT}/report_${TAG}.txt 2>&1 || echo FAILED report ${R}"$'\n'
done
echo "terminal errors: runs ${RUNS}, seed ${SEED}, out ${OUT}, ${GPU} dev (${QOS}), ${CPUS} CPUs, ${TIME}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=termerr
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
${CMDS}
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
