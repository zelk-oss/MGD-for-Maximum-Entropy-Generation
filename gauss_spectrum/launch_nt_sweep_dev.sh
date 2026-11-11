#!/bin/bash
# nt sweep (launch_nt_sweep.sh) on the H100 dev QoS in ONE submission, packed under the dev cap of 10 submitted jobs:
#   one dev array of 10 tasks (5 sigma x 2 datasets); each task runs its nt 10000 run, then its nt 25000 run
#   (estimated ~15 + ~40 min at n1 1e5, extrapolated from the 11-09 sweep: 9 ms/step at n1 1e4, x10; NOT measured).
#   The nt 50000 runs are submitted separately on t3 (launch_nt_sweep.sh NTS=50000); give that job id in AFTER so the
#   finish job (on t3: not counted in the dev cap) waits for everything.
# Same runs, labels and OUT as launch_nt_sweep.sh, so finish_sigma_sweep.sh sees all nt together.
#
# Usage (Jean Zay, repo root):
#   AFTER=25180 DRY=1 bash gauss_spectrum/launch_nt_sweep_dev.sh
#   AFTER=25180 bash gauss_spectrum/launch_nt_sweep_dev.sh
# overrides: DATASETS (default "turbulence jets"), SIGMAS, NTS (in order, run one after the other in a task), N1, SEED_TURB, SEED_JETS, OUT, TIME1 (dev max 2 h),
#            TL_<nt> (run_SDE.py time guard in minutes per nt), TIME2, AFTER (extra job ids to wait for, colon-separated)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
SIGMAS=(${SIGMAS:-0.1 0.5 1 2 5})
NTS=(${NTS:-10000 25000})
N1="${N1:-100000}"
SEED_TURB="${SEED_TURB:-900}"
SEED_JETS="${SEED_JETS:-500}"
OUT="${OUT:-${HERE}/nt_sweep}"
TIME1="${TIME1:-02:00:00}"
TL_10000="${TL_10000:-35}"           # minutes; with Theta_reg (~2-3 min per run) 35 + 70 + margin < 120
TL_25000="${TL_25000:-70}"
TIME2="${TIME2:-02:00:00}"
AFTER="${AFTER:-}"
LOGS="${OUT}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6; MODULE_PRE=arch/h100

DATASETS=(${DATASETS:-turbulence jets})   # e.g. DATASETS=jets (2026-11-10: rerun of the jets tasks only)
RUNS=()
for S in "${SIGMAS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        case "${DS}" in
            turbulence) RUNS+=("turbulence:${S}:${SEED_TURB}") ;;
            jets)       RUNS+=("jets:${S}:${SEED_JETS}") ;;
            *) echo "Error: unknown dataset ${DS}"; exit 1 ;;
        esac
    done
done
if [ "${#RUNS[@]}" -gt 10 ]; then echo "Error: ${#RUNS[@]} tasks > dev cap 10"; exit 1; fi
TLS=()
for NT in "${NTS[@]}"; do
    V="TL_${NT}"; [ -n "${!V:-}" ] || { echo "Error: set TL_${NT}"; exit 1; }
    TLS+=("${NT}:${!V}")
done
echo "dev array: ${#RUNS[@]} tasks x nt (${NTS[*]}) in turn, task time ${TIME1}, guards ${TLS[*]} min; n1 ${N1} -> ${OUT}"
echo "finish job on qos_gpu_h100-t3 after the dev array${AFTER:+ and ${AFTER}}"
if [ -n "${DRY:-}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${LOGS}"

J1=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=ntsweep_dev
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-$(( ${#RUNS[@]} - 1 ))
#SBATCH --time=${TIME1}
#SBATCH --output=${LOGS}/slurm_%x_%A_%a.log
#SBATCH --qos=qos_gpu_h100-dev
module purge
module load ${MODULE_PRE}
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUNS=( ${RUNS[@]} )
IFS=: read -r DS S SD <<< "\${RUNS[\${SLURM_ARRAY_TASK_ID}]}"
for P in ${TLS[@]}; do
    IFS=: read -r NT TL <<< "\${P}"
    OUT="${OUT}" TIMESTAMP="${TIMESTAMP}" N1="${N1}" NT="\${NT}" NBULK="\$(( NT / 5 ))" TIME_LIMIT_MIN="\${TL}" \
        LABEL="ntsweep_s\${S//./p}_nt\${NT}" bash gauss_spectrum/run_sigma_task.sh "\${DS}" "\${S}" "\${SD}" \
        || echo "######## FAIL \${DS} sigma \${S} nt \${NT} seed \${SD}"
done
EOT
)
if [ -z "${J1}" ]; then echo "Error: dev sbatch returned no job id"; exit 1; fi
echo "Submitted dev array ${J1}"

DEP="${J1}${AFTER:+:${AFTER}}"
J2=$(sbatch --parsable --dependency=afterany:${DEP} <<EOT
#!/bin/bash
#SBATCH --job-name=ntsweep_fin
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=${TIME2}
#SBATCH --output=${LOGS}/slurm_%x_%j.log
#SBATCH --qos=qos_gpu_h100-t3
module purge
module load ${MODULE_PRE}
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
OUT="${OUT}" bash gauss_spectrum/finish_sigma_sweep.sh
EOT
)
echo "Submitted finish ${J2} (t3, after ${DEP}). Logs: ${LOGS}"
