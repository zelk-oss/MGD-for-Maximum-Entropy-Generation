#!/bin/bash
# (b) of the whitening test (2026-11-13): does running MGD on whitened coordinates y = C^(-alpha/2) x cut the number of
# steps needed on the Gaussian-with-Lagrangian-spectrum problem (M = 256, Scat2 Q1, exact H(p*) known)?
# Reference without whitening (alpha 0, same sigma/seed/n1, gauss_spectrum/nt_sweep): bound above H(p*) by 177 / 87 / 45 nats
# at nt 10k / 25k / 50k (two_phase).
# ONE dev array (H100 qos_gpu_h100-dev), one task per ALPHA:SCHEDULE; each task runs its nt values in turn (ascending), then
# the entropy benchmark of its runs (finish_sigma_sweep.sh) -> ${OUT}/a<alpha>_<sched>/summary/endpoint.csv.
# SCHEDULE two_phase (n_bulk nt/5, gap_end 5e-5, as the reference) | uniform (power, exponent 1: t ends at 1 - 1/nt).
# Saved regularised systems -> $SCRATCH/.../whitening_test/reg_system (regenerable).
#
# Usage (Jean Zay, repo root): bash whitening_test/launch_whiten_dev.sh        (DRY=1: print only)
# overrides: TASKS ("alpha:sched ..."), NTS_two_phase, NTS_uniform, SIGMA (2), N1 (100000), SEED (900), OUT, TIME1, TL_<nt>

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
TASKS=(${TASKS:-0.5:two_phase 0.75:two_phase 1:two_phase 0.75:uniform 1:uniform})
NTS_two_phase="${NTS_two_phase:-2000 10000 25000}"
NTS_uniform="${NTS_uniform:-2000 10000}"
SIGMA="${SIGMA:-2}"
N1="${N1:-100000}"
SEED="${SEED:-900}"
OUT="${OUT:-${HERE}/runs}"
TIME1="${TIME1:-01:30:00}"
TL_2000="${TL_2000:-6}"; TL_10000="${TL_10000:-12}"; TL_25000="${TL_25000:-25}"   # guard minutes of each SDE run
LOGS="${OUT}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
if [ "${#TASKS[@]}" -gt 10 ]; then echo "Error: ${#TASKS[@]} tasks > dev cap 10"; exit 1; fi
echo "whitening dev array: ${#TASKS[@]} tasks (${TASKS[*]}), sigma ${SIGMA}, n1 ${N1}, seed ${SEED}, nt two_phase [${NTS_two_phase}] uniform [${NTS_uniform}] -> ${OUT}"
if [ -n "${DRY:-}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${LOGS}"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=whiten_dev
#SBATCH -A wbg@h100
#SBATCH -C h100
#SBATCH --partition=gpu_p6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-$(( ${#TASKS[@]} - 1 ))
#SBATCH --time=${TIME1}
#SBATCH --output=${LOGS}/slurm_%x_%A_%a.log
#SBATCH --qos=qos_gpu_h100-dev
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TASKS=( ${TASKS[@]} )
IFS=: read -r A SCH <<< "\${TASKS[\${SLURM_ARRAY_TASK_ID}]}"
if [ "\${SCH}" = "uniform" ]; then SCHED=power; EXP=1; NTS="${NTS_uniform}"; else SCHED=two_phase; EXP=2; NTS="${NTS_two_phase}"; fi
TOUT="${OUT}/a\${A}_\${SCH}"
declare -A TL=( [2000]=${TL_2000} [10000]=${TL_10000} [25000]=${TL_25000} )
for NT in \${NTS}; do
    OUT="\${TOUT}" REGDIR="\${SCRATCH:?SCRATCH not set}/MGD-for-Maximum-Entropy-Generation/whitening_test/reg_system/a\${A}_\${SCH}" \
        TIMESTAMP="${TIMESTAMP}" N1="${N1}" NT="\${NT}" NBULK="\$(( NT / 5 ))" TIME_LIMIT_MIN="\${TL[\${NT}]}" \
        WHITEN="\${A}" SCHEDULE="\${SCHED}" SCHED_EXP="\${EXP}" LABEL="wh_a\${A//./p}_\${SCH}_nt\${NT}" \
        bash gauss_spectrum/run_sigma_task.sh turbulence "${SIGMA}" "${SEED}" \
        || echo "######## FAIL alpha \${A} \${SCH} nt \${NT}"
done
OUT="\${TOUT}" bash gauss_spectrum/finish_sigma_sweep.sh
EOT
