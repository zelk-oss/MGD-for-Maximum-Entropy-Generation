#!/bin/bash
# nt-convergence sweep on the Gaussians with the Lagrangian (turbulence) and Eulerian (Modane jets) spectra, M = 256
# (agreed 2026-11-10). Question: is the violation of the MGD entropy bound (H_sigma* > H(p_1^sigma), sigma sweep of
# 11-09) a time-discretisation error? At fixed sigma the target H(p_1^sigma) is fixed and only the step h changes, so
# three nt values per sigma test the order in h: for an O(h) error the gap ratio is 2.5 (nt 10k -> 25k) and 2 (25k -> 50k).
#   sigma   0.1 0.5 1 2 5
#   nt      10000 25000 50000, two_phase grid, n_bulk nt/5 (bulk step h = 4.5/nt on [0, 0.9])
#   n1      100000
#   seeds   turbulence 900, jets 500 (one per run)
#   label   ntsweep_s<sigma>_nt<nt> (distinct per run, needed by the reg-system lookup in run_sigma_task.sh)
#   phase 1 one array per nt (5 sigma x 2 datasets = 10 tasks, one run each): run_sigma_task.sh (SDE + Theta_reg)
#   phase 2 after all of phase 1 (afterany): finish_sigma_sweep.sh with OUT = nt_sweep (benchmark + summary figures)
# Cost: NOT measured at n1 1e5. Extrapolated from 25.8 ms/step on V100 at n1 1e4 (gauss_scat2): ~0.26 s/step on V100
# at n1 1e5 -> nt 50000 ~ 3.6 h on V100. Default task times below leave a margin; check them with sacct on the old
# sweep (jobs 2072478, 2072479) and override TIME_10000 / TIME_25000 / TIME_50000 if needed.
#
# Usage (Jean Zay, repo root):
#   DRY=1 GPU=h100 bash gauss_spectrum/launch_nt_sweep.sh     # print the plan only
#   GPU=h100 bash gauss_spectrum/launch_nt_sweep.sh
# overrides: SIGMAS, NTS, N1, SEED_TURB, SEED_JETS, OUT, TIME_<nt>, TIME2, QOS, NO_FINISH=1 (skip the phase-2 job)
# dev QoS (max 2 h, max 10 submitted jobs): one nt per round, e.g.
#   NTS=10000 TIME_10000=01:00:00 QOS=qos_gpu_h100-dev NO_FINISH=1 GPU=h100 bash gauss_spectrum/launch_nt_sweep.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
SIGMAS=(${SIGMAS:-0.1 0.5 1 2 5})
NTS=(${NTS:-10000 25000 50000})
N1="${N1:-100000}"
SEED_TURB="${SEED_TURB:-900}"
SEED_JETS="${SEED_JETS:-500}"
OUT="${OUT:-${HERE}/nt_sweep}"
TIME_10000="${TIME_10000:-03:00:00}"
TIME_25000="${TIME_25000:-06:00:00}"
TIME_50000="${TIME_50000:-10:00:00}"
TIME2="${TIME2:-02:00:00}"
LOGS="${OUT}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
# GPU: h100 (default) or v100. V100 tasks submitted from an H100 login node (jzxh*) fail ("no kernel image").
GPU="${GPU:-h100}"
case "${GPU}" in
    h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  MODULE_PRE=arch/h100; T3_QOS=qos_gpu_h100-t3 ;;
    v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; MODULE_PRE="";        T3_QOS=qos_gpu-t3 ;;
    *) echo "GPU must be h100 or v100"; exit 1 ;;
esac
QOS="${QOS-${T3_QOS}}"              # t3 by default (tasks > 2 h, so not dev); QOS="" = default queue
echo "GPU ${GPU} (${ACCOUNT}, ${PARTITION}), QoS ${QOS:-default}, n1 ${N1} -> ${OUT}"

# one task = one (dataset, sigma) run at a given nt
RUNS=()
for S in "${SIGMAS[@]}"; do RUNS+=("turbulence:${S}:${SEED_TURB}" "jets:${S}:${SEED_JETS}"); done
for NT in "${NTS[@]}"; do
    TV="TIME_${NT}"; echo "  nt ${NT}: n_bulk $(( NT / 5 )), ${#RUNS[@]} tasks, task time ${!TV:-unset}"
done
for NT in "${NTS[@]}"; do
    TV="TIME_${NT}"; [ -n "${!TV:-}" ] || { echo "Error: set TIME_${NT}"; exit 1; }
done
if [ -n "${DRY:-}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${LOGS}"

JOBS=()
for NT in "${NTS[@]}"; do
    TV="TIME_${NT}"; T="${!TV}"
    # run_SDE.py's own time limit: task time minus 30 min for Theta_reg and saving
    TL=$(( $(echo "${T}" | awk -F: '{print $1*60 + $2}') - 30 ))
    J=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=ntsweep_${NT}
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-$(( ${#RUNS[@]} - 1 ))
#SBATCH --time=${T}
#SBATCH --output=${LOGS}/slurm_%x_%A_%a.log
${QOS:+#SBATCH --qos=${QOS}}
module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUNS=( ${RUNS[@]} )
IFS=: read -r DS S SD <<< "\${RUNS[\${SLURM_ARRAY_TASK_ID}]}"
OUT="${OUT}" TIMESTAMP="${TIMESTAMP}" N1="${N1}" NT="${NT}" NBULK="$(( NT / 5 ))" TIME_LIMIT_MIN="${TL}" \
    LABEL="ntsweep_s\${S//./p}_nt${NT}" bash gauss_spectrum/run_sigma_task.sh "\${DS}" "\${S}" "\${SD}" \
    || echo "######## FAIL \${DS} sigma \${S} nt ${NT} seed \${SD}"
EOT
)
    if [ -z "${J}" ]; then echo "Error: sbatch for nt ${NT} returned no job id"; exit 1; fi
    JOBS+=("${J}")
    echo "Submitted nt ${NT}: ${J} (${#RUNS[@]} tasks, ${T})"
done

if [ -n "${NO_FINISH:-}" ]; then
    # dev QoS allows 10 submitted jobs: a 10-task round + this job would be 11. Run finish_sigma_sweep.sh by hand at the end.
    echo "NO_FINISH set: no phase-2 job. When all runs are done: OUT=gauss_spectrum/nt_sweep bash gauss_spectrum/finish_sigma_sweep.sh"
    exit 0
fi
DEP=$(IFS=:; echo "${JOBS[*]}")
JOB2=$(sbatch --parsable --dependency=afterany:${DEP} <<EOT
#!/bin/bash
#SBATCH --job-name=ntsweep_fin
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
echo "Submitted phase 2 ${JOB2} (after ${DEP}). Logs: ${LOGS}"
