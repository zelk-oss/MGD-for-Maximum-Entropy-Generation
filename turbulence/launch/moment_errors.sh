#!/bin/bash
# Per-statistic moment error at t=1 of finished turbulence runs (turbulence/moment_error_variance.py --dump).
# The analysis itself is CPU only (no CUDA call), but it runs as a short GPU job on the dev QoS (<= 2 h): the free
# prepost partition queued for hours (11-11, job 58953). Billed 1 GPU x elapsed (expect well under 1 GPU-h).
# One task; runs the listed runs one after the other; writes one CSV per run (every statistic: target, generated,
# signed + relative error, z, peak along the run, Ito term) and the text report.
#
# Usage (Jean Zay, repo root):
#   GPU=v100 RUNS="turbulence_nested_M3_bicw_allmass_scat4:3.5:55000" bash turbulence/launch/moment_errors.sh
#   GPU=v100 RUNS="turbulence_nested_M3_bicw_allmass_scat4_pf:1:27500 turbulence_nested_M3_bicw_allmass_scat4_pf:3.5:27500" bash ...
# RUNS = space-separated LABEL:SIGMA:NT (sigma as in the run name: 1, 2, 3.5, 6 -> sigma1.0 ... sigma3.5; integers get ".0").
# overrides: GPU (v100|h100, mandatory), SEED (900), OUT (turbulence/moment_errors), TIME (01:30:00 for all runs), CPUS (10)
# Reads saved_results/{aux_moments,samples,sampling_times} + experiments/.../fitted_potentials.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/../.." && pwd)"
: "${RUNS:?set RUNS, e.g. RUNS=\"turbulence_nested_M3_bicw_allmass_scat4:3.5:55000\"}"
: "${GPU:?set GPU=v100 or GPU=h100}"
SEED="${SEED:-900}"
OUT="${OUT:-${PROJECT_DIR}/turbulence/moment_errors}"
TIME="${TIME:-01:30:00}"
CPUS="${CPUS:-10}"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE="" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"

# python commands, one per run
CMDS=""
for R in ${RUNS}; do
  IFS=: read -r LABEL SIG NTV <<< "${R}"
  case "${SIG}" in *.*) ;; *) SIG="${SIG}.0" ;; esac
  CMDS+="python -u turbulence/moment_error_variance.py --label ${LABEL} --seeds ${SEED} --config _sigma${SIG}_ _nt${NTV}_ --dump ${OUT} > ${OUT}/report_${LABEL}_sigma${SIG}_nt${NTV}.txt 2>&1 || echo FAILED ${R}"$'\n'
done
echo "moment errors: runs ${RUNS}, seed ${SEED}, out ${OUT}, ${GPU} dev (${QOS}), ${CPUS} CPUs, ${TIME}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=momerr
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
