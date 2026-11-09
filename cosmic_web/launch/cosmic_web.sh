#!/bin/bash
# Cosmic-web (Quijote, 2D) MGD runs: one SLURM array job, one task per seed (the seed only changes the
# initial white noise; the data are always the same first N1 maps).
#
# Two arms (ARM), settings of notebooks/mgd_sampling/mgd_2d_example.ipynb (J=5 Q=1 L=4, sigma=1, Cos interpolant,
# 100 maps of 128x128 = fact 2) with differences decided 2026-11-05: sigma^2 = 5.5 and nt = 2700 (paper), and time grid two_phase (as the turbulence
# runs), per-step ridge REGULARIZATION 1e-7 with float64 solves (the notebook had 0), and
# what the other experiments have:
#   base   L_6 confinement + scattering order 1, 2, 3 (Real and Imag)                      nt = 2700
#   scat4  base + scattering order 4 (Real and Imag, squared-modulus envelope); nothing else differs   nt = 2700
# The time-regularised system is saved (SAVE_REG_SYSTEM, in-run solve skipped); lam is chosen ex post:
#   cosmic_web_theta_reg.sh (Theta_reg for a grid of lam, per seed) -> cosmic_web_xseed.sh (cross-seed choice).
#
# NOTHING IS MEASURED YET (time, GPU memory, system size). To measure: one seed per arm on the H100 dev QoS
# (qos_gpu_h100-dev: starts at once, max 2 h, max 10 jobs; V100 dev is qos_gpu-dev):
#   DATA_PATH=... ARM=base  GPU=h100 SEEDS=900 QOS=qos_gpu_h100-dev TIME=02:00:00 LABEL=timing_base  bash cosmic_web.sh
#   DATA_PATH=... ARM=scat4 GPU=h100 SEEDS=900 QOS=qos_gpu_h100-dev TIME=02:00:00 LABEL=timing_scat4 bash cosmic_web.sh
# and read "SDE integration finished in" / s/it in the SLURM log. TIME is a cap; the SDE loop aborts cleanly
# (--time_limit_min) if projected past it.
# Estimates (not measured): r = 564 (base) / 4294 (scat4) statistics. System on disk ~ 2 float32 r x r blocks per
# node: base ~2.5 GB, scat4 ~400 GB at nsub 1, nt 2700 -> N_SUBSAMPLE averages fine steps into coarse nodes
# (SDE evolution unchanged); scat4 defaults to 10 (~40 GB host RAM per seed), a choice (unmeasured). The order-4 gradient builds
# (B, 20, 20, 21, 128, 128) complex tensors: 22 GB at B = 20 -> scat4 defaults to BATCH_SIZE 10 on H100 (5 on V100);
# base uses the notebook's 20. CPUs per task: 24 on H100, 10 on V100.
#
# Usage (Jean Zay, from cosmic_web/launch/); GPU is mandatory, no default:
#   DATA_PATH=/path/to/quijote.npy ARM=scat4_real GPU=h100 bash cosmic_web.sh        # seeds 900-904
#   (ARM = base_real, scat4_real: Real parts only; base, scat4: with the Imag terms)
#   DATA_PATH=...                  ARM=scat4 GPU=h100 bash cosmic_web.sh
# Other overrides: SEEDS, N1, NT, TIME, QOS, BATCH_SIZE, N_SUBSAMPLE, CPUS, LABEL, REG_SYSTEM_DIR, FACT,
#   N_BULK, T_SWITCH, GAP_END, REGULARIZATION, SOLVE_FLOAT64.

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CW_DIR="$(cd "${LAUNCH_DIR}/.." && pwd)"

: "${DATA_PATH:?set DATA_PATH to the Quijote .npy}"
: "${ARM:?set ARM=base_real, scat4_real, base or scat4}"
: "${GPU:?set GPU=v100 or h100 (no default on purpose)}"

case "${ARM}" in
    # Real parts only (the overnight runs). Measured on H100: base_real 0.33 s/step, scat4_real 3.77 s/step (batch 10).
    # NT 3000: base_real ~17 min, scat4_real ~3.1 h. TIME is a cap: the guard aborts if the projection > 90 % of it.
    base_real)  NT_DEFAULT=3000; TIME_DEFAULT="02:00:00"; BATCH_DEFAULT=20; NSUB_DEFAULT=1 ;;
    scat4_real) NT_DEFAULT=3000; TIME_DEFAULT="04:30:00"; BATCH_DEFAULT=$([ "${GPU}" = "h100" ] && echo 10 || echo 5); NSUB_DEFAULT=5 ;;
    base)  NT_DEFAULT=2700; TIME_DEFAULT="03:00:00"; BATCH_DEFAULT=20; NSUB_DEFAULT=1 ;;
    scat4) NT_DEFAULT=2700; TIME_DEFAULT="20:00:00"; BATCH_DEFAULT=$([ "${GPU}" = "h100" ] && echo 10 || echo 5); NSUB_DEFAULT=10 ;;
    *) echo "Error: ARM must be base_real, scat4_real, base or scat4"; exit 1 ;;
esac
# per-step ridge and float64 per-step solves (decided 2026-11-05; the notebook had ridge 0, but the model now has
# more, possibly collinear, statistics: L_6). The offline Theta_reg ridge (1e-7) is separate.
REGULARIZATION="${REGULARIZATION:-1e-7}"
SOLVE_FLOAT64_FLAG=$([ "${SOLVE_FLOAT64:-true}" = "true" ] && echo "--solve_float64")
NT="${NT:-${NT_DEFAULT}}"
TIME="${TIME:-${TIME_DEFAULT}}"
BATCH_SIZE="${BATCH_SIZE:-${BATCH_DEFAULT}}"
N_SUBSAMPLE="${N_SUBSAMPLE:-${NSUB_DEFAULT}}"
N1="${N1:-100}"
# N_PARTICLES (a multiple of N1): more walkers than data maps. Default = N1 (production). With 100 walkers the walker
# covariance C of the time-regularised solve has rank <= 99 < r = 324, so the time coupling misses ~225 directions;
# N_PARTICLES >= r makes it full rank (test: ARM=base_real N_PARTICLES=400 LABEL=base_real_np400).
NPART_FLAG=$([ -n "${N_PARTICLES}" ] && echo "--n_particles ${N_PARTICLES}")
FACT="${FACT:-2}"
# quijote.npy (100 maps 256x256, uploaded 2026-11-05) is ALREADY a log field (min -2.99, exp(min) = 0.0501), not the
# notebook's log(x - min + 0.1) (whose floor is -2.303): skip the shift+log, keep the db4 coarse graining (FACT=2).
# DATA_IS_LOG=false for a raw-density file.
DATA_IS_LOG_FLAG=$([ "${DATA_IS_LOG:-true}" = "true" ] && echo "--data_is_log")
# time grid (codes/time_schedules.py two_phase, as the turbulence runs): N_BULK uniform steps on [0, T_SWITCH], then
# NT - N_BULK steps with constant h/(1-t) down to 1 - GAP_END. Default N_BULK = NT/2 (a choice: bulk h 1.8e-3 and
# 348 steps in the last 1 % of t at NT = 1000); same cost per step as the uniform grid.
N_BULK="${N_BULK:-$(( NT / 2 ))}"
T_SWITCH="${T_SWITCH:-0.9}"
GAP_END="${GAP_END:-5e-5}"
# SCHEDULE=uniform: --schedule power --schedule_exponent 1 (the S&P grid; ends at t = 1 - 1/NT, not 1 - GAP_END)
case "${SCHEDULE:-two_phase}" in
    two_phase) SCHED_ARGS="--schedule two_phase --n_bulk ${N_BULK} --t_switch ${T_SWITCH} --gap_end ${GAP_END}" ;;
    uniform)   SCHED_ARGS="--schedule power --schedule_exponent 1" ;;
    *) echo "Error: SCHEDULE must be two_phase or uniform"; exit 1 ;;
esac
J="${J:-5}"; Q=1; L=4     # J=3 for FACT=16 (16x16 maps)
# sigma^2 = 5.5 and 2700 steps: the MGD paper's cosmic-web values (sec. 6.4, App. D.3: samples at sigma^2 = 5.5; steps {1000,1000,1000,1000,2500,2700} for sigma^2 {0.1,0.25,1,2.5,4,5.5}); the notebook had sigma = 1
SIGMA="${SIGMA:-2.345207879911715}"
INTERPOLANT=Cos
SEED_LIST=(${SEEDS:-900 901 902 903 904})
LABEL="${LABEL:-${ARM}}"
REG_SYSTEM_DIR="${REG_SYSTEM_DIR:-${SCRATCH:?SCRATCH is not set}/MGD-for-Maximum-Entropy-Generation/cosmic_web/reg_system/${LABEL}}"

case "${GPU}" in
    v100) ACCOUNT="wbg@v100"; CONSTRAINT="v100"; PARTITION="gpu_p13"; MODULE_PRE=""; CPUS_DEFAULT=10 ;;
    h100) ACCOUNT="wbg@h100"; CONSTRAINT="h100"; PARTITION="gpu_p6";  MODULE_PRE="arch/h100"; CPUS_DEFAULT=24 ;;
    *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
CPUS="${CPUS:-${CPUS_DEFAULT}}"
MODULE="${MODULE:-pytorch-gpu/py3/2.8.0}"

LOG_DIR="${CW_DIR}/saved_results/logs"
mkdir -p "${LOG_DIR}" "${REG_SYSTEM_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
IFS=: read -r _h _m _s <<< "${TIME}"
TIME_LIMIT_MIN=$(( 10#${_h} * 60 + 10#${_m} ))
NTASKS=${#SEED_LIST[@]}

echo "Submitting cosmic_web ${ARM}: fact ${FACT}, J ${J}, ${SCHEDULE:-two_phase}, seeds (${SEED_LIST[*]}), n1=${N1}, particles=${N_PARTICLES:-${N1}}, nt=${NT}, batch ${BATCH_SIZE}, nsub ${N_SUBSAMPLE}, ${GPU}, time ${TIME}"
echo "  reg systems -> ${REG_SYSTEM_DIR}"

JOBID=$(sbatch --parsable ${QOS:+--qos=${QOS}} <<EOT
#!/bin/bash
#SBATCH --job-name=cw_${ARM}
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH --array=0-$((NTASKS-1))
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/slurm_%x_%A_%a.log

module purge
# V100 submitted from an H100 login node (jzxh*): drop the H100 stack if purge kept it (job 2172641, 11-09)
$([ "${GPU}" = "v100" ] && echo "module unload arch/h100 2>/dev/null")
${MODULE_PRE:+module load ${MODULE_PRE}}
module load ${MODULE}
module list 2>&1 | tail -n +1
python -c "import torch; (torch.ones(1, device='cuda') + 1).item()" || { echo "PyTorch build has no kernels for this GPU (${GPU}): use GPU=h100 or submit from a non-H100 login node"; exit 1; }
cd "${CW_DIR}"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED_LIST=(${SEED_LIST[@]})
SEED_VAL=\${SEED_LIST[\${SLURM_ARRAY_TASK_ID}]}

srun python "${CW_DIR}/run_SDE.py" \
    --seed \${SEED_VAL} \
    --arm ${ARM} \
    --label ${LABEL} \
    --data_path "${DATA_PATH}" \
    --fact ${FACT} ${DATA_IS_LOG_FLAG} \
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
    --outdir "${CW_DIR}"
EOT
)

if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted array job ${JOBID}"

LEDGER="${LOG_DIR}/submissions.tsv"
[ -f "${LEDGER}" ] || printf 'date\tjob_id\tarm\tgpu\ttimestamp\tseeds\tn1\tnt\tbatch\tnsub\n' > "${LEDGER}"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "${JOBID}" "${ARM}" "${GPU}" \
    "${TIMESTAMP}" "${SEED_LIST[*]}" "${N1}" "${NT}" "${BATCH_SIZE}" "${N_SUBSAMPLE}" >> "${LEDGER}"
