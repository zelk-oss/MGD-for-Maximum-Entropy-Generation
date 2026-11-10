#!/bin/bash
# 2D turbulence (Navier-Stokes vorticity, ns_randn4_N256_c1.mat) MGD runs: one SLURM array job, one task per seed
# (the seed only changes the initial white noise; the data are always the same first N1 maps).
# Copy of cosmic_web/launch/cosmic_web.sh; differences: data file and loader (no log), NT 3300, no DATA_IS_LOG.
#
# Paper (MGD, sec. 6.4 + App. D.3): 128 x 128 maps, sigma^2 = 5.5, nt = 3300 for 2D turbulence, r = 2392 scattering
# moments (Real parts, orders 1-4; our scat4_real is the comparable arm), 100 replicas. Our decision of 11-09 (cosmic
# web): two_phase grid + N_PARTICLES - 1 >= r so that the time-regularised Theta is defined in every cell
# (base_real r = 324 -> N_PARTICLES 400). scat4_real (r ~ 2400) would need N_PARTICLES >= ~2500: not feasible now.
#
# NOTHING IS MEASURED for this data set. Same filters and 128 x 128 as cosmic web, where base_real N = 100 nt = 3000
# took 988 s on H100. N = 400, nt = 3300: ~72 min IF the cost is linear in N (not measured) -> TIME cap 3 h.
#
# Usage (Jean Zay, from turbulence_2d/launch/); GPU is mandatory, no default:
#   ARM=base_real GPU=h100 N_PARTICLES=400 LABEL=base_real_np400 bash turb2d.sh          # seeds 900-904
# Overrides: SEEDS, N1, N_PARTICLES, NT, TIME, QOS, BATCH_SIZE, N_SUBSAMPLE, CPUS, LABEL, REG_SYSTEM_DIR, FACT,
#   N_BULK, T_SWITCH, GAP_END, REGULARIZATION, SOLVE_FLOAT64, SCHEDULE, DATA_PATH, SIGMA, J.

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T2_DIR="$(cd "${LAUNCH_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${T2_DIR}/.." && pwd)"

DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/data/data_files/ns_randn4_N256_c1.mat}"
: "${ARM:?set ARM=base_real, scat4_real, base or scat4}"
: "${GPU:?set GPU=v100 or h100 (no default on purpose)}"
[ -f "${DATA_PATH}" ] || { echo "Error: data file not found: ${DATA_PATH}"; exit 1; }

case "${ARM}" in
    # TIME defaults scaled from the cosmic-web measurements (NT 3000 -> 3300); a cap, the guard aborts cleanly.
    base_real)  NT_DEFAULT=3300; TIME_DEFAULT="03:00:00"; BATCH_DEFAULT=20; NSUB_DEFAULT=1 ;;
    scat4_real) NT_DEFAULT=3300; TIME_DEFAULT="05:00:00"; BATCH_DEFAULT=$([ "${GPU}" = "h100" ] && echo 10 || echo 5); NSUB_DEFAULT=5 ;;
    base)  NT_DEFAULT=3300; TIME_DEFAULT="03:00:00"; BATCH_DEFAULT=20; NSUB_DEFAULT=1 ;;
    scat4) NT_DEFAULT=3300; TIME_DEFAULT="20:00:00"; BATCH_DEFAULT=$([ "${GPU}" = "h100" ] && echo 10 || echo 5); NSUB_DEFAULT=10 ;;
    *) echo "Error: ARM must be base_real, scat4_real, base or scat4"; exit 1 ;;
esac
REGULARIZATION="${REGULARIZATION:-1e-7}"
SOLVE_FLOAT64_FLAG=$([ "${SOLVE_FLOAT64:-true}" = "true" ] && echo "--solve_float64")
NT="${NT:-${NT_DEFAULT}}"
TIME="${TIME:-${TIME_DEFAULT}}"
BATCH_SIZE="${BATCH_SIZE:-${BATCH_DEFAULT}}"
N_SUBSAMPLE="${N_SUBSAMPLE:-${NSUB_DEFAULT}}"
N1="${N1:-100}"
# N_PARTICLES: a multiple of N1 (data maps repeated cyclically). Default N1. With N1 = N_PARTICLES every walker has its
# own data map (1000 maps available here, unlike cosmic web's 100).
NPART_FLAG=$([ -n "${N_PARTICLES}" ] && echo "--n_particles ${N_PARTICLES}")
FACT="${FACT:-2}"
N_BULK="${N_BULK:-$(( NT / 2 ))}"
T_SWITCH="${T_SWITCH:-0.9}"
GAP_END="${GAP_END:-5e-5}"
case "${SCHEDULE:-two_phase}" in
    two_phase) SCHED_ARGS="--schedule two_phase --n_bulk ${N_BULK} --t_switch ${T_SWITCH} --gap_end ${GAP_END}" ;;
    uniform)   SCHED_ARGS="--schedule power --schedule_exponent 1" ;;
    *) echo "Error: SCHEDULE must be two_phase or uniform"; exit 1 ;;
esac
J="${J:-5}"; Q=1; L=4
SIGMA="${SIGMA:-2.345207879911715}"     # sigma^2 = 5.5
INTERPOLANT=Cos
SEED_LIST=(${SEEDS:-900 901 902 903 904})
LABEL="${LABEL:-${ARM}}"
REG_SYSTEM_DIR="${REG_SYSTEM_DIR:-${SCRATCH:?SCRATCH is not set}/MGD-for-Maximum-Entropy-Generation/turbulence_2d/reg_system/${LABEL}}"

case "${GPU}" in
    v100) ACCOUNT="wbg@v100"; CONSTRAINT="v100"; PARTITION="gpu_p13"; MODULE_PRE=""; CPUS_DEFAULT=10 ;;
    h100) ACCOUNT="wbg@h100"; CONSTRAINT="h100"; PARTITION="gpu_p6";  MODULE_PRE="arch/h100"; CPUS_DEFAULT=24 ;;
    *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
CPUS="${CPUS:-${CPUS_DEFAULT}}"
MODULE="${MODULE:-pytorch-gpu/py3/2.8.0}"

LOG_DIR="${T2_DIR}/saved_results/logs"
mkdir -p "${LOG_DIR}" "${REG_SYSTEM_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
IFS=: read -r _h _m _s <<< "${TIME}"
TIME_LIMIT_MIN=$(( 10#${_h} * 60 + 10#${_m} ))
NTASKS=${#SEED_LIST[@]}

echo "Submitting turbulence_2d ${ARM}: fact ${FACT}, J ${J}, ${SCHEDULE:-two_phase}, seeds (${SEED_LIST[*]}), n1=${N1}, particles=${N_PARTICLES:-${N1}}, nt=${NT}, batch ${BATCH_SIZE}, nsub ${N_SUBSAMPLE}, ${GPU}, time ${TIME}"
echo "  data ${DATA_PATH}"
echo "  reg systems -> ${REG_SYSTEM_DIR}"

JOBID=$(sbatch --parsable ${QOS:+--qos=${QOS}} <<EOT
#!/bin/bash
#SBATCH --job-name=t2d_${ARM}
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH --array=0-$((NTASKS-1))
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/slurm_%x_%A_%a.log

module purge
$([ "${GPU}" = "v100" ] && echo "module unload arch/h100 2>/dev/null")
${MODULE_PRE:+module load ${MODULE_PRE}}
module load ${MODULE}
module list 2>&1 | tail -n +1
python -c "import torch; (torch.ones(1, device='cuda') + 1).item()" || { echo "PyTorch build has no kernels for this GPU (${GPU}): use GPU=h100 or submit from a non-H100 login node"; exit 1; }
cd "${T2_DIR}"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED_LIST=(${SEED_LIST[@]})
SEED_VAL=\${SEED_LIST[\${SLURM_ARRAY_TASK_ID}]}

srun python "${T2_DIR}/run_SDE.py" \
    --seed \${SEED_VAL} \
    --arm ${ARM} \
    --label ${LABEL} \
    --data_path "${DATA_PATH}" \
    --fact ${FACT} \
    --n1 ${N1} ${NPART_FLAG} \
    --J ${J} --Q ${Q} --L ${L} \
    --sigma ${SIGMA} \
    --interpolant ${INTERPOLANT} \
    --nt ${NT} \
    ${SCHED_ARGS} \
    --batch_size ${BATCH_SIZE} \
    --regularization ${REGULARIZATION} ${SOLVE_FLOAT64_FLAG} \
    --lam 0.0 --n_subsample ${N_SUBSAMPLE} --reg_solver thomas --reg_ridge 0.0 \
    --save_reg_system --reg_system_dir "${REG_SYSTEM_DIR}" --skip_reg_solve \
    --time_limit_min ${TIME_LIMIT_MIN} \
    --outdir "${T2_DIR}"
EOT
)

if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted array job ${JOBID}"

LEDGER="${LOG_DIR}/submissions.tsv"
[ -f "${LEDGER}" ] || printf 'date\tjob_id\tarm\tgpu\ttimestamp\tseeds\tn1\tnpart\tnt\tbatch\tnsub\n' > "${LEDGER}"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "${JOBID}" "${ARM}" "${GPU}" \
    "${TIMESTAMP}" "${SEED_LIST[*]}" "${N1}" "${N_PARTICLES:-${N1}}" "${NT}" "${BATCH_SIZE}" "${N_SUBSAMPLE}" >> "${LEDGER}"
