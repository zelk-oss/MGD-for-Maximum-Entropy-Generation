#!/bin/bash
# Sigma sweep on the Gaussians with the Lagrangian (turbulence) and Eulerian (Modane jets) spectra, M = 256
# (gauss_spectrum/README.md, 2026-11-06). Question: is the entropy along the transport and at t = 1 right, and does the
# violation of the MGD entropy bound seen at sigma = 3.5 change with sigma?
#   sigma   0.5 0.77 1.18 1.8 2.77 3.5 4.25 6.52 10 (8 log-spaced in [0.5, 10] + 3.5, the existing runs' value)
#   nt      max(2000, 300 sigma^2) (constant h sigma^2, sigma 10 -> 30000), n_bulk nt/5   (run_sigma_task.sh)
#   seeds   turbulence 900 901, jets 500 501
#   phase 1 one array task per sigma (9), each running its 4 (dataset, seed) runs in turn: run_sigma_task.sh (SDE + Theta_reg)
#   phase 2 after phase 1 (afterany: analyses what finished): finish_sigma_sweep.sh (benchmark + summary figures)
#   9 + 1 jobs = the dev-QoS cap of 10 submitted jobs; GPU h100 + qos_gpu_h100-dev by default (GPU=v100, QOS="" to change)
# Cost: 25.8 ms/step on V100 measured for M=256 r=9 (gauss_scat2) -> sigma 10 (30000 steps) ~ 13 min per run; TIME1 2 h.
#
# Usage (Jean Zay, repo root):
#   DRY=1 bash gauss_spectrum/launch_sigma_sweep.sh     # print the plan only
#   bash gauss_spectrum/launch_sigma_sweep.sh
# overrides: SIGMAS, SEEDS_TURB, SEEDS_JETS, OUT, TIME1, TIME2

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
SIGMAS=(${SIGMAS:-0.5 0.77 1.18 1.8 2.77 3.5 4.25 6.52 10})
SEEDS_TURB=(${SEEDS_TURB:-900 901})
SEEDS_JETS=(${SEEDS_JETS:-500 501})
OUT="${OUT:-${HERE}/sigma_sweep}"
TIME1="${TIME1:-02:00:00}"         # per sigma: 4 runs in turn (sigma 10: 30000 steps each)
TIME2="${TIME2:-01:00:00}"
LOGS="${OUT}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
# GPU: h100 (default) or v100. 2026-11-06: V100 tasks submitted from an H100 login node (jzxh*) got the sm_90-only
# PyTorch build ("no kernel image is available"); H100 matches that environment.
GPU="${GPU:-h100}"
case "${GPU}" in
    h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  MODULE_PRE=arch/h100; DEV_QOS=qos_gpu_h100-dev ;;
    v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; MODULE_PRE="";        DEV_QOS=qos_gpu-dev ;;
    *) echo "GPU must be h100 or v100"; exit 1 ;;
esac
QOS="${QOS-${DEV_QOS}}"              # dev QoS by default (max 2 h, max 10 submitted jobs incl. array tasks); QOS="" = normal queue
echo "GPU ${GPU} (${ACCOUNT}, ${PARTITION})"

# one array task per sigma, running its (dataset, seed) runs in turn: 9 tasks + 1 finish job = 10, the dev-QoS cap
RUNS=()
for SD in "${SEEDS_TURB[@]}"; do RUNS+=("turbulence:${SD}"); done
for SD in "${SEEDS_JETS[@]}"; do RUNS+=("jets:${SD}"); done
echo "${#SIGMAS[@]} tasks (one per sigma) x ${#RUNS[@]} runs each -> ${OUT}; QoS ${QOS:-default}, task time ${TIME1}"
for S in "${SIGMAS[@]}"; do
    NT=$(python3 -c "print(max(2000, round(300 * ${S} ** 2)))")
    echo "  sigma ${S}: nt ${NT}, n_bulk $(( NT / 5 ))"
done
if [ -n "${DRY:-}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${LOGS}"

JOB1=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=sigsweep
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-$(( ${#SIGMAS[@]} - 1 ))
#SBATCH --time=${TIME1}
#SBATCH --output=${LOGS}/slurm_%x_%A_%a.log
${QOS:+#SBATCH --qos=${QOS}}
module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SIGMAS=( ${SIGMAS[@]} )
S=\${SIGMAS[\${SLURM_ARRAY_TASK_ID}]}
for R in ${RUNS[@]}; do
    IFS=: read -r DS SD <<< "\${R}"
    OUT="${OUT}" TIMESTAMP="${TIMESTAMP}" bash gauss_spectrum/run_sigma_task.sh "\${DS}" "\${S}" "\${SD}" \
        || echo "######## FAIL \${DS} sigma \${S} seed \${SD}"
done
EOT
)
if [ -z "${JOB1}" ]; then echo "Error: phase-1 sbatch returned no job id"; exit 1; fi

JOB2=$(sbatch --parsable --dependency=afterany:${JOB1} <<EOT
#!/bin/bash
#SBATCH --job-name=sigsweep_fin
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=${TIME2}
#SBATCH --output=${LOGS}/slurm_%x_%j.log
${QOS:+#SBATCH --qos=${QOS}}
module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
OUT="${OUT}" bash gauss_spectrum/finish_sigma_sweep.sh
EOT
)
echo "Submitted phase 1 ${JOB1} (${#SIGMAS[@]} tasks x ${#RUNS[@]} runs), phase 2 ${JOB2} (after phase 1). Logs: ${LOGS}"
