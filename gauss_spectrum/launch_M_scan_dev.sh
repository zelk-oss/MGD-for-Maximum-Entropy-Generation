#!/bin/bash
# M scan (2026-11-12): is the entropy bound H_sigma* <= H(gen) <= H(p*) restored when sigma^2/nt is small?
# Gaussian with the Lagrangian-turbulence spectrum coarse-grained to M = 32 and 64 (--target_len, J = log2 M),
# Scat2 Q1, n1 1e4, ONE dev array (H100 qos_gpu_h100-dev) of 6 tasks = 2 M x 3 sigma. Each task runs its nt values in turn
# (ascending, so the cheap ones are saved if the last hits its guard). Same run script as the nt sweep (run_sigma_task.sh).
# Outputs: ${OUT}/M<M>/turbulence/..., labels ntsweep_s<sigma>_nt<nt>. NO finish job (analysis is a separate step).
# The saved regularised systems (large: ~0.8 GB per nt 400000 run) go to $SCRATCH/MGD-for-Maximum-Entropy-Generation/
# gauss_spectrum/m_scan/M<M>/reg_system (purged after 30 days unused; regenerable).
# Step speed at n1 1e4 and M 32/64 is NOT measured: the guards below are guesses; read the log of the nt 25000 run first.
#
# Usage (Jean Zay, repo root): bash gauss_spectrum/launch_M_scan_dev.sh        (DRY=1: print only)
# overrides: SIGMAS, MS (pairs M:J), NTS, N1, SEED, OUT, TL_<nt> (guard minutes of the run), TIME1 (dev max 2 h), QOS

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
SIGMAS=(${SIGMAS:-1 2 5})
MS=(${MS:-32:5 64:6})
NTS=(${NTS:-25000 100000 400000})
N1="${N1:-10000}"
SEED="${SEED:-900}"
OUT="${OUT:-${HERE}/m_scan}"
TIME1="${TIME1:-02:00:00}"
QOS="${QOS:-qos_gpu_h100-dev}"       # t3 (<= 20 h, user launches): QOS=qos_gpu_h100-t3 TIME1=04:00:00 NTS=1600000 TL_1600000=190
TL_25000="${TL_25000:-12}"
TL_100000="${TL_100000:-30}"
TL_400000="${TL_400000:-60}"
LOGS="${OUT}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M")

RUNS=()
for MJ in "${MS[@]}"; do
    for S in "${SIGMAS[@]}"; do RUNS+=("${MJ}:${S}"); done
done
if [ "${#RUNS[@]}" -gt 10 ]; then echo "Error: ${#RUNS[@]} tasks > dev cap 10"; exit 1; fi
TLS=()
for NT in "${NTS[@]}"; do
    V="TL_${NT}"; [ -n "${!V:-}" ] || { echo "Error: set TL_${NT}"; exit 1; }
    TLS+=("${NT}:${!V}")
done
echo "dev array: ${#RUNS[@]} tasks (M:J:sigma ${RUNS[*]}), nt in turn (${NTS[*]}), guards ${TLS[*]} min, n1 ${N1} -> ${OUT}"
if [ -n "${DRY:-}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${LOGS}"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=mscan_dev
#SBATCH -A wbg@h100
#SBATCH -C h100
#SBATCH --partition=gpu_p6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-$(( ${#RUNS[@]} - 1 ))
#SBATCH --time=${TIME1}
#SBATCH --output=${LOGS}/slurm_%x_%A_%a.log
#SBATCH --qos=${QOS}
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUNS=( ${RUNS[@]} )
IFS=: read -r MLEN JSC S <<< "\${RUNS[\${SLURM_ARRAY_TASK_ID}]}"
for P in ${TLS[@]}; do
    IFS=: read -r NT TL <<< "\${P}"
    OUT="${OUT}/M\${MLEN}" REGDIR="\${SCRATCH:?SCRATCH not set}/MGD-for-Maximum-Entropy-Generation/gauss_spectrum/m_scan/M\${MLEN}/reg_system" TIMESTAMP="${TIMESTAMP}" N1="${N1}" NT="\${NT}" NBULK="\$(( NT / 5 ))" TIME_LIMIT_MIN="\${TL}" \
        MLEN="\${MLEN}" JSC="\${JSC}" LABEL="ntsweep_s\${S//./p}_nt\${NT}" bash gauss_spectrum/run_sigma_task.sh turbulence "\${S}" "${SEED}" \
        || echo "######## FAIL M \${MLEN} sigma \${S} nt \${NT} seed ${SEED}"
done
EOT
