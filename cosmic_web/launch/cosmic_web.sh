#!/bin/bash
# Cosmic-web (Quijote, 2D) MGD runs: one SLURM array job, one task per seed (the seed only changes the
# initial white noise; the data are always the same first N1 maps).
#
# Two arms (ARM), settings of notebooks/mgd_sampling/mgd_2d_example.ipynb except where noted:
#   base   the notebook as is: Scattering {Second, First, Third_Real}, J=5 Q=1 L=4, sigma=1,
#          Cos interpolant, nt = 1000, 100 maps of 128x128 (fact=2), batch 20.
#   scat4  the same + Scattering_Fourth_Order_Real, nt = 2000 (finer grid, to see whether the
#          ~2000 moments are matched; decided 2026-11-05).
#
# Nothing here is measured yet: the notebook took 11 min for `base` on an unknown GPU, `scat4` has ~7x more
# statistics and 2x more steps. TIME is a cap; the SDE loop aborts cleanly (--time_limit_min) if the projected
# time exceeds it, and says the measured s/it in run.log. Check the first minutes of the SLURM log.
#
# Usage (Jean Zay, from cosmic_web/launch/); GPU is mandatory, no default:
#   DATA_PATH=/path/to/Quijote_Fidu_15000_256.npy ARM=base  GPU=v100 bash cosmic_web.sh              # seeds 900-904
#   DATA_PATH=...                                  ARM=scat4 GPU=h100 bash cosmic_web.sh
#   SEEDS="900" ...                                                                                  # one seed first
# Other overrides: N1, NT, TIME, SEEDS, LABEL.

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CW_DIR="$(cd "${LAUNCH_DIR}/.." && pwd)"

: "${DATA_PATH:?set DATA_PATH to Quijote_Fidu_15000_256.npy}"
: "${ARM:?set ARM=base or scat4}"
: "${GPU:?set GPU=v100 or h100 (no default on purpose)}"

case "${ARM}" in
    base)  NT_DEFAULT=1000; TIME_DEFAULT="03:00:00" ;;
    scat4) NT_DEFAULT=2000; TIME_DEFAULT="20:00:00" ;;
    *) echo "Error: ARM must be base or scat4"; exit 1 ;;
esac
NT="${NT:-${NT_DEFAULT}}"
TIME="${TIME:-${TIME_DEFAULT}}"
N1="${N1:-100}"
FACT=2
J=5; Q=1; L=4
SIGMA=1
INTERPOLANT=Cos
BATCH_SIZE=20
SEED_LIST=(${SEEDS:-900 901 902 903 904})
LABEL="${LABEL:-${ARM}}"

case "${GPU}" in
    v100) ACCOUNT="wbg@v100"; CONSTRAINT="v100"; PARTITION="gpu_p13"; MODULE_PRE="" ;;
    h100) ACCOUNT="wbg@h100"; CONSTRAINT="h100"; PARTITION="gpu_p6";  MODULE_PRE="arch/h100" ;;
    *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
CPUS="${CPUS:-10}"
MODULE="${MODULE:-pytorch-gpu/py3/2.8.0}"

LOG_DIR="${CW_DIR}/saved_results/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
IFS=: read -r _h _m _s <<< "${TIME}"
TIME_LIMIT_MIN=$(( 10#${_h} * 60 + 10#${_m} ))
NTASKS=${#SEED_LIST[@]}

echo "Submitting cosmic_web ${ARM}: seeds (${SEED_LIST[*]}), n1=${N1}, nt=${NT}, ${GPU}, time ${TIME}"

JOBID=$(sbatch --parsable <<EOT
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
    --batch_size ${BATCH_SIZE} \
    --regularization 0.0 \
    --time_limit_min ${TIME_LIMIT_MIN} \
    --outdir "${CW_DIR}"
EOT
)

if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted array job ${JOBID}"

LEDGER="${LOG_DIR}/submissions.tsv"
[ -f "${LEDGER}" ] || printf 'date\tjob_id\tarm\tgpu\ttimestamp\tseeds\tn1\tnt\n' > "${LEDGER}"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "${JOBID}" "${ARM}" "${GPU}" \
    "${TIMESTAMP}" "${SEED_LIST[*]}" "${N1}" "${NT}" >> "${LEDGER}"
