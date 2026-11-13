#!/bin/bash
# Part 1 of the terminal-phase test (codes/one_step_test.py): one noise step + one corrector pass on the saved final walkers of
# finished runs, compared with the errors those runs ended with. Reads saved_results/{aux_moments,samples,sampling_times} and
# experiments/.../{config.json,fitted_potentials}; writes only under OUT. Dev GPU job (the Gram of r ~ 300 statistics needs a GPU).
#
# Usage (Jean Zay, repo root):
#   GPU=v100 TAG=smoke DRAWS=2 CHAIN_STEPS=5 EDGES="2" \
#     RUNS="turbulence:turbulence_nested_M3_bicw_allmass_scat4:900:_nt55000_:turb_scat4" bash codes/launch_one_step_test.sh
#   GPU=v100 TAG=part1 RUNS="turbulence:turbulence_nested_M3_bicw_allmass_scat4:900:_nt55000_:turb_scat4 jets:jets_full:500::jets_full500" bash ...
# RUNS = space-separated KIND:LABEL:SEED:CONFIG_SUBSTRINGS(comma separated, may be empty):RUN_TAG
# overrides: GPU (v100|h100, mandatory), TAG (mandatory, names the output folder terminal_phase_test/<TAG>, which must not exist),
#   DRAWS (10), CHAIN_STEPS (300), EDGES ("2 4", turbulence runs only; empty = skip experiment e), EXTRA (more one_step_test.py flags),
#   TIME (01:30:00), CPUS (10).
# A run that fails does not stop the others (its log is <OUT>/<RUN_TAG>.log.txt).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
: "${RUNS:?set RUNS, see the header}"
: "${GPU:?set GPU=v100 or GPU=h100}"
: "${TAG:?set TAG (output folder terminal_phase_test/<TAG>)}"
DRAWS="${DRAWS:-10}"
CHAIN_STEPS="${CHAIN_STEPS:-300}"
EDGES="${EDGES:-}"
EXTRA="${EXTRA:-}"
OUT="${OUT:-${PROJECT_DIR}/terminal_phase_test/${TAG}}"
TIME="${TIME:-01:30:00}"
CPUS="${CPUS:-10}"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE="" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
if [ -e "${OUT}" ]; then echo "Error: ${OUT} exists already (use a new TAG; nothing is overwritten)"; exit 1; fi
mkdir -p "${OUT}/slurm"

CMDS=""
for R in ${RUNS}; do
  IFS=: read -r KIND LABEL SEED CFG RTAG <<< "${R}"
  CFGARGS=""; [ -n "${CFG}" ] && CFGARGS="--config ${CFG//,/ }"
  EDGEARGS=""; [ "${KIND}" = "turbulence" ] && [ -n "${EDGES}" ] && EDGEARGS="--edge_scales ${EDGES}"
  CMDS+="python -u codes/one_step_test.py --kind ${KIND} --label ${LABEL} --seed ${SEED} ${CFGARGS} --out ${OUT}/${RTAG} --tag ${RTAG} --draws ${DRAWS} --chain_steps ${CHAIN_STEPS} ${EDGEARGS} ${EXTRA} > ${OUT}/${RTAG}.log.txt 2>&1 || echo FAILED ${R}"$'\n'
done
echo "one-step test: runs ${RUNS}, out ${OUT}, ${GPU} dev (${QOS}), ${CPUS} CPUs, ${TIME}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=onestep_${TAG}
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${OUT}/slurm/slurm_%x_%j.log

module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
export OMP_NUM_THREADS=${CPUS}
cd "${PROJECT_DIR}"
${CMDS}
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
