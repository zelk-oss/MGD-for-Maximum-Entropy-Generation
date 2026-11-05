#!/bin/bash
# Cosmic-web (Quijote, 2D) MGD runs: one SLURM array job, one task per seed (the seed only changes the
# initial white noise; the data are always the same first N1 maps).
#
# Two arms (ARM), settings of notebooks/mgd_sampling/mgd_2d_example.ipynb (J=5 Q=1 L=4, sigma=1, Cos interpolant,
# ridge 0, 100 maps of 128x128 = fact 2, uniform time grid) PLUS what the other experiments have:
#   base   L_6 confinement + scattering order 1, 2, 3 (Real and Imag)                        nt = 1000
#   scat4  base + scattering order 4 (Real and Imag); nothing else differs                   nt = 2000
# The time-regularised system is saved (SAVE_REG_SYSTEM, in-run solve skipped); lam is chosen ex post:
#   cosmic_web_theta_reg.sh (Theta_reg for a grid of lam, per seed) -> cosmic_web_xseed.sh (cross-seed choice).
#
# NOTHING IS MEASURED YET (time, GPU memory, system size). To measure: one seed on the dev QoS first:
#   DATA_PATH=... ARM=base GPU=v100 SEEDS=900 QOS=qos_gpu-dev TIME=02:00:00 bash cosmic_web.sh
# and read "SDE integration finished in" / s/it in the SLURM log. TIME is a cap; the SDE loop aborts cleanly
# (--time_limit_min) if projected past it.
# Estimates (not measured): r = 564 (base) / 4294 (scat4) statistics. System on disk ~ 2 float32 r x r blocks per
# node: base ~2.5 GB, scat4 ~295 GB at nsub 1, nt 2000 -> N_SUBSAMPLE averages fine steps into coarse nodes
# (SDE evolution unchanged); scat4 defaults to 5 (~59 GB), a choice to confirm. The order-4 gradient builds
# (B, 20, 20, 21, 128, 128) complex tensors: 22 GB at B = 20 -> scat4 defaults to BATCH_SIZE 5.
#
# Usage (Jean Zay, from cosmic_web/launch/); GPU is mandatory, no default:
#   DATA_PATH=/path/to/quijote.npy ARM=base  GPU=v100 bash cosmic_web.sh             # seeds 900-904
#   DATA_PATH=...                  ARM=scat4 GPU=v100 bash cosmic_web.sh
# Other overrides: SEEDS, N1, NT, TIME, QOS, BATCH_SIZE, N_SUBSAMPLE, CPUS, LABEL, REG_SYSTEM_DIR, FACT.

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CW_DIR="$(cd "${LAUNCH_DIR}/.." && pwd)"

: "${DATA_PATH:?set DATA_PATH to the Quijote .npy}"
: "${ARM:?set ARM=base or scat4}"
: "${GPU:?set GPU=v100 or h100 (no default on purpose)}"

case "${ARM}" in
    base)  NT_DEFAULT=1000; TIME_DEFAULT="03:00:00"; BATCH_DEFAULT=20; NSUB_DEFAULT=1 ;;
    scat4) NT_DEFAULT=2000; TIME_DEFAULT="20:00:00"; BATCH_DEFAULT=5;  NSUB_DEFAULT=5 ;;
    *) echo "Error: ARM must be base or scat4"; exit 1 ;;
esac
NT="${NT:-${NT_DEFAULT}}"
TIME="${TIME:-${TIME_DEFAULT}}"
BATCH_SIZE="${BATCH_SIZE:-${BATCH_DEFAULT}}"
N_SUBSAMPLE="${N_SUBSAMPLE:-${NSUB_DEFAULT}}"
N1="${N1:-100}"
FACT="${FACT:-2}"
J=5; Q=1; L=4
SIGMA=1
INTERPOLANT=Cos
SEED_LIST=(${SEEDS:-900 901 902 903 904})
LABEL="${LABEL:-${ARM}}"
REG_SYSTEM_DIR="${REG_SYSTEM_DIR:-${SCRATCH:?SCRATCH is not set}/MGD-for-Maximum-Entropy-Generation/cosmic_web/reg_system/${LABEL}}"

case "${GPU}" in
    v100) ACCOUNT="wbg@v100"; CONSTRAINT="v100"; PARTITION="gpu_p13"; MODULE_PRE="" ;;
    h100) ACCOUNT="wbg@h100"; CONSTRAINT="h100"; PARTITION="gpu_p6";  MODULE_PRE="arch/h100" ;;
    *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
CPUS="${CPUS:-10}"
MODULE="${MODULE:-pytorch-gpu/py3/2.8.0}"

LOG_DIR="${CW_DIR}/saved_results/logs"
mkdir -p "${LOG_DIR}" "${REG_SYSTEM_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
IFS=: read -r _h _m _s <<< "${TIME}"
TIME_LIMIT_MIN=$(( 10#${_h} * 60 + 10#${_m} ))
NTASKS=${#SEED_LIST[@]}

echo "Submitting cosmic_web ${ARM}: seeds (${SEED_LIST[*]}), n1=${N1}, nt=${NT}, batch ${BATCH_SIZE}, nsub ${N_SUBSAMPLE}, ${GPU}, time ${TIME}"
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
${MODULE_PRE:+module load ${MODULE_PRE}}
module load ${MODULE}
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
    --fact ${FACT} \
    --n1 ${N1} \
    --J ${J} --Q ${Q} --L ${L} \
    --sigma ${SIGMA} \
    --interpolant ${INTERPOLANT} \
    --nt ${NT} \
    --schedule power --schedule_exponent 1 \
    --batch_size ${BATCH_SIZE} \
    --regularization 0.0 \
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
