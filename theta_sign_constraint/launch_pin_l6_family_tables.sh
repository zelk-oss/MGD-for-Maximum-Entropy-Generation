#!/bin/bash
# Per-family interpretation tables of the pinned-L_6 test (theta_sign_constraint/pin_l6_family_tables.py): rare vs
# typical table + histograms (logp_family_contrib.py) and Var/2d tables (talk_figures.kl_table), one set per arm.
# Reads <results>/seed<N>/{summary.json,logp_scan.npz} written by launch_pin_l6_test.sh. One task, all seeds.
# GPU dev QoS (never prepost), 10 CPUs (no extra billing). Output: <results>/family_tables/ (csv + png, a few MB).
#
# Usage (Jean Zay, repo root):
#   GPU=v100 RESULTS=pinl6_bscat4 LABEL=pinl6_tables bash theta_sign_constraint/launch_pin_l6_family_tables.sh
#   GPU=v100 SMOKE=1 LABEL=pinl6_tables_smoke bash theta_sign_constraint/launch_pin_l6_family_tables.sh   # on pinl6_smoke3
# overrides: TIME (00:40:00), CPUS

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
LABEL="${LABEL:?set LABEL}"
GPU="${GPU:?set GPU=v100 or GPU=h100}"
case "${GPU}" in
    h100) ACCOUNT="wbg@h100"; PARTITION="gpu_p6";  CONSTRAINT="h100"; QOS="qos_gpu_h100-dev"; MODULE_PRE="arch/h100"; CPUS="${CPUS:-24}" ;;
    v100) ACCOUNT="wbg@v100"; PARTITION="gpu_p13"; CONSTRAINT="v100"; QOS="qos_gpu-dev";      MODULE_PRE="";          CPUS="${CPUS:-10}" ;;
    *) echo "GPU must be v100 or h100"; exit 1 ;;
esac
TIME="${TIME:-00:40:00}"
EXP_ROOT="turbulence/experiments"
SAMPLES="turbulence/saved_results/samples"
RESULTS="${RESULTS:-pinl6_bscat4}"
if [ -n "${SMOKE:-}" ]; then
    RESULTS="pinl6_smoke3"; TIME="00:15:00"
    EXP_ROOT="whitening_test/bscat4/turbulence/experiments"; SAMPLES="whitening_test/bscat4/turbulence/saved_results/samples"
fi
RES="theta_sign_constraint/results/pin_l6_test/${RESULTS}"
LOGS="${HERE}/results/pin_l6_test/${RESULTS}/family_tables/logs"
echo "family tables ${LABEL}: ${RES} | ${GPU} (${QOS}), ${CPUS} CPUs, ${TIME}"
if [ -n "${DRY:-}" ]; then echo "(dry run: nothing submitted)"; exit 0; fi
mkdir -p "${LOGS}"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${LABEL}
#SBATCH -A ${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH -C ${CONSTRAINT}
#SBATCH --qos=${QOS}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${LOGS}/slurm_%x_%j.log
module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"
python -u theta_sign_constraint/pin_l6_family_tables.py --results "${RES}" --exp_root "${EXP_ROOT}" --samples "${SAMPLES}"
EOT
