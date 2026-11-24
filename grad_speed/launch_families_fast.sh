#!/bin/bash
# Faster versions of the other families (2026-11-22): validation (grad_speed/validate_families_fast.py) or per-operation
# tables of the original code (grad_speed/bench_family_ops.py). One dev GPU job, billed 1 GPU x elapsed.
#
# Usage (Jean Zay, repo root):
#   GPU=h100 SCRIPT=validate M=256 SUBSERIES_LEN=512 TARGET_LEN=256 FITTED_DIR=<run>/fitted_potentials bash grad_speed/launch_families_fast.sh
#   GPU=h100 SCRIPT=bench M=1024 SUBSERIES_LEN=1024 TARGET_LEN=1024 FITTED_DIR=<M=256 run>/fitted_potentials LOAD_FROM_M=1 bash ...
#   GPU=h100 SCRIPT=compare M=128 J=7 SUBSERIES_LEN=256 TARGET_LEN=128 ORIG=speedcheck_M128_nt27500 FAST=pf_fast_M128 bash ...
#            (2026-11-23: two finished long runs, original vs fast code, grad_speed/compare_fast_orig_run.py)
# overrides: GPU, SCRIPT (validate|bench|compare), M, SUBSERIES_LEN, TARGET_LEN, FITTED_DIR (validate/bench), LOAD_FROM_M (0|1),
#            N1 (5000), NT (200, validate only), BATCH (5000), N_ERR (512), REPS (20), LABEL, OUT, TIME (01:00:00),
#            J, ORIG, FAST (compare only)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
: "${GPU:?set GPU=v100 or GPU=h100}"
: "${SCRIPT:?set SCRIPT=validate or SCRIPT=bench}"
: "${M:?set M}"
: "${SUBSERIES_LEN:?set SUBSERIES_LEN}"
: "${TARGET_LEN:?set TARGET_LEN}"
[ "${SCRIPT}" = "compare" ] || : "${FITTED_DIR:?set FITTED_DIR}"
N1="${N1:-5000}"
NT="${NT:-200}"
BATCH="${BATCH:-5000}"
N_ERR="${N_ERR:-512}"
REPS="${REPS:-20}"
LABEL="${LABEL:-}"
TIME="${TIME:-01:00:00}"
LOAD_ARG=""
if [ "${LOAD_FROM_M:-0}" = "1" ]; then LOAD_ARG="--load_from_M"; fi
case "${SCRIPT}" in
  validate) OUT="${OUT:-${PROJECT_DIR}/grad_speed/results_families_fast}"
            CMD="grad_speed/validate_families_fast.py --M ${M} --subseries_len ${SUBSERIES_LEN} --target_len ${TARGET_LEN} --n1 ${N1} --nt ${NT} --batch ${BATCH} --n_err ${N_ERR} --reps ${REPS} --fitted_dir ${FITTED_DIR} ${LOAD_ARG} --label ${LABEL:-none}" ;;
  bench)    OUT="${OUT:-${PROJECT_DIR}/grad_speed/results_bench_family_ops}"
            CMD="grad_speed/bench_family_ops.py --M ${M} --subseries_len ${SUBSERIES_LEN} --target_len ${TARGET_LEN} --n1 ${BATCH} --reps ${REPS} --fitted_dir ${FITTED_DIR} ${LOAD_ARG} --label ${LABEL:-none}" ;;
  compare)  : "${J:?set J}"; : "${ORIG:?set ORIG (label of the original run)}"; : "${FAST:?set FAST (label of the fast run)}"
            OUT="${OUT:-${PROJECT_DIR}/grad_speed/results_fast_vs_orig}"
            CMD="grad_speed/compare_fast_orig_run.py --M ${M} --J ${J} --subseries_len ${SUBSERIES_LEN} --target_len ${TARGET_LEN} --n1 ${N1} --orig ${ORIG} --fast ${FAST}" ;;
  *) echo "Error: SCRIPT must be validate, bench or compare"; exit 1 ;;
esac
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE=""; CPUS=10 ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100"; CPUS=24 ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"
echo "${SCRIPT}: M ${M}, n1 ${N1}, nt ${NT}, batch ${BATCH}, load_from_M ${LOAD_FROM_M:-0}, ${GPU} dev (${QOS}), ${TIME}, label ${LABEL}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=fam_${SCRIPT}
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
python -u ${CMD} --out "${OUT}"
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
