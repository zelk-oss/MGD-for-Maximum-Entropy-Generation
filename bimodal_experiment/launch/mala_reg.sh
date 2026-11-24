#!/bin/bash
# MALA on the regularised bimodal density (bimodal_experiment/mala_reg.py), one GPU task for beta 0.1 and 0.9,
# runs 0-2, starts MGD walkers + N(0,1). Writes <beta dir>/mala/run_<k>_<start>[_lamD][_<LABEL>].pt next to the
# MGD walkers in results_bim_theta_beta/mala_start_beta*_1123/.
#
# Usage (repo root on Jean Zay):
#   full:   GPU=v100 LAM=D bash bimodal_experiment/launch/mala_reg.sh
#   smoke:  GPU=v100 LAM=D LABEL=smoke1 STEPS=20 N_WALKERS=1000 TIME=00:15:00 bash bimodal_experiment/launch/mala_reg.sh
#   summary of the outputs (mala_summary.py, CPU work on a GPU dev node):  GPU=v100 SUMMARY=1 TIME=00:15:00 bash ...
#   continue saved chains:  GPU=v100 LAM=A MALA_ARGS="--betas 0.9 --continue_steps 5000" bash ...
#   Newton with MALA moments (newton_mala.py, test C2):  GPU=v100 NEWTON=1 TIME=01:00:00 bash ...
#     smoke:  GPU=v100 NEWTON=1 LABEL=smoke1 NEWTON_ARGS="--n_walkers 1000 --iters 1 --burn 20 --avg 50" TIME=00:15:00 bash ...
# overrides: LAM (A|D) STEPS N_WALKERS LABEL SUMMARY (1 = mala_summary.py) NEWTON (1 = newton_mala.py) NEWTON_ARGS
#            GPU (h100|v100) TIME CPUS

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/../.." && pwd)"

LAM="${LAM:-D}"
STEPS="${STEPS:-5000}"
N_WALKERS="${N_WALKERS:-100000}"
LABEL="${LABEL:-}"
GPU="${GPU:-v100}"
TIME="${TIME:-00:30:00}"
CPUS="${CPUS:-4}"
LOG_DIR="${PROJECT_DIR}/bimodal_experiment/results_bim_theta_beta/mala_logs"

if [ "${GPU}" = "h100" ]; then
HEADER="#SBATCH -A wbg@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH -C h100"
MODULES="module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0"
elif [ "${GPU}" = "v100" ]; then
HEADER="#SBATCH -A wbg@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH -C v100-32g"
MODULES="module purge
module load pytorch-gpu/py3/2.8.0"
else
    echo "GPU must be h100 or v100"; exit 1
fi

mkdir -p "${LOG_DIR}"
echo "MALA lam ${LAM}, ${STEPS} steps, ${N_WALKERS} walkers, label '${LABEL}' | ${GPU} dev, ${TIME}, ${CPUS} CPUs"

JOBID=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=mala_reg${LABEL:+_${LABEL}}
${HEADER}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/slurm_%x_%j.log

${MODULES}
cd "${PROJECT_DIR}/bimodal_experiment"
if [ "${SUMMARY}" = "1" ]; then
    python -u mala_summary.py
elif [ "${NEWTON}" = "1" ]; then
    python -u newton_mala.py --device cuda ${NEWTON_ARGS} ${LABEL:+--label ${LABEL}}
else
    python -u mala_reg.py --lam ${LAM} --steps ${STEPS} --n_walkers ${N_WALKERS} --device cuda ${LABEL:+--label ${LABEL}} ${MALA_ARGS}
fi
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted job ${JOBID}"
