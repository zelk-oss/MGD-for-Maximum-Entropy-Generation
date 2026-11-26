#!/bin/bash
# Cost of one MGD step per operation at signal length M, J scales (grad_speed/bench_signal_length.py, 2026-11-20):
# per potential family, per SDE operation (fixed states, batches 5000/2500/1000), float32 vs float64 solve,
# kernel profile, and a real 100-step loop on true walkers. One dev GPU job per (M, J) cell, billed 1 GPU x elapsed.
# Needs the fitted_potentials folder of a pre-flight run of the same (M, J) (turbulence_nested_M3_bicw.sh,
# PREFLIGHT_ONLY=true, see its usage lines).
#
# Usage (Jean Zay, repo root):
#   GPU=h100 M=1024 J=10 SUBSERIES_LEN=1024 TARGET_LEN=1024 FITTED_DIR=<run>/fitted_potentials \
#       bash grad_speed/launch_bench_signal_length.sh
# overrides: GPU (v100|h100, mandatory), M, J, SUBSERIES_LEN, TARGET_LEN, FITTED_DIR (all mandatory),
#            OUT (grad_speed/results_bench_signal_length), EXTRA (extra args, e.g. "--real_steps 0"),
#            TIME (01:30:00), CPUS (24 on h100, 10 on v100)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
: "${GPU:?set GPU=v100 or GPU=h100}"
: "${M:?set M (signal length)}"
: "${J:?set J (number of scales)}"
: "${SUBSERIES_LEN:?set SUBSERIES_LEN (window length before coarse-graining)}"
: "${TARGET_LEN:?set TARGET_LEN (= M)}"
: "${FITTED_DIR:?set FITTED_DIR (fitted_potentials folder of the pre-flight run)}"
OUT="${OUT:-${PROJECT_DIR}/grad_speed/results_bench_signal_length}"
EXTRA="${EXTRA:-}"
TIME="${TIME:-01:30:00}"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE=""; CPUS="${CPUS:-10}" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100"; CPUS="${CPUS:-24}" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"
echo "bench_signal_length.py: M ${M}, J ${J}, ${GPU} dev (${QOS}), ${TIME}, fitted ${FITTED_DIR} ${EXTRA}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=bench_M${M}_J${J}
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
python -u grad_speed/bench_signal_length.py --M ${M} --J ${J} --subseries_len ${SUBSERIES_LEN} --target_len ${TARGET_LEN} --fitted_dir "${FITTED_DIR}" --out "${OUT}" ${EXTRA}
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
