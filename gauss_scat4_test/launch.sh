#!/bin/bash
# In-family Gaussian test of H_k and of the entropy bound (gauss_scat4_test/README.md). ONE dev task, everything in order:
#   1. make_spectrum.py: S* = max-ent Gaussian of the order-2 family with the order-2 moments of SOURCE (white or a
#      gauss_spectrum.pt), unit variance -> data in the family, truth theta = 0 on L_6 and order 4, H(p*) exact;
#   2. per seed: gaussian_experiment/run_SDE.py --spectrum_file (shared data, DATA_SEED) + saved system, then
#      codes/resolve_theta_reg.py for every lam in LAMS;
#   3. cross-seed lam (lam_selection_criterion/turb_xseed.py), check.py (entropy, theta vs truth),
#      kl_groups cache_energies.py + compute_H.py (H per group, dataset gauss4).
# Writes only under OUT = gauss_scat4_test/runs/<TAG>.
#
# Usage (Jean Zay, repo root):
#   GPU=v100 TAG=smoke2 LABEL=g4smoke SEEDS="900 901" N1=100 NBULK=10 NT=40 LAMS="1e-3 1e-2" bash gauss_scat4_test/launch.sh
#   GPU=v100 TAG=d16 PHASE=run bash gauss_scat4_test/launch.sh      # array, one dev task per seed
#   GPU=v100 TAG=d16 PHASE=finish bash gauss_scat4_test/launch.sh   # after all run tasks: xseed, check, H
# PHASE: all (default: one task, everything in order; for smoke tests), run, finish.
# overrides: GPU (v100|h100, mandatory), TAG (mandatory, new folder), M (16), J (4), SOURCE (white), SEEDS
#   ("900 901 902 903 904"), DATA_SEED (0), N1 (10000), NBULK (5000), NT (50000), SIGMA (3.5), RIDGE (1e-4),
#   LAMS ("1e-5 1e-4 1e-3 1e-2"), NEW_LAMS (finish only: extra lams re-solved from the saved systems, e.g. "1e-6 1e-1 1"),
#   LABEL (g4_M<M>), TERMS, TIME (02:00:00), CPUS (10).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
: "${GPU:?set GPU=v100 or GPU=h100}"
: "${TAG:?set TAG (output folder gauss_scat4_test/runs/<TAG>)}"
M="${M:-16}"; J="${J:-4}"; SOURCE="${SOURCE:-white}"
SEEDS="${SEEDS:-900 901 902 903 904}"; DATA_SEED="${DATA_SEED:-0}"
# NBULK = time STEPS of the bulk phase (< NT), as B+Scat4
N1="${N1:-10000}"; NBULK="${NBULK:-5000}"; NT="${NT:-50000}"; SIGMA="${SIGMA:-3.5}"; RIDGE="${RIDGE:-1e-4}"
LAMS="${LAMS:-1e-5 1e-4 1e-3 1e-2}"; LABEL="${LABEL:-g4_M${M}}"
TERMS="${TERMS:-L_6 Scattering_Second_Order_Q1 Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1}"
TIME="${TIME:-02:00:00}"; CPUS="${CPUS:-10}"
OUT="${PROJECT_DIR}/gauss_scat4_test/runs/${TAG}"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE="" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
PHASE="${PHASE:-all}"          # all = one task, everything in order (smoke); run = array, one task per seed; finish = after run
if [ "${PHASE}" = "finish" ]; then
  [ -d "${OUT}" ] || { echo "Error: PHASE=finish needs the existing ${OUT}"; exit 1; }
else
  if [ -e "${OUT}" ]; then echo "Error: ${OUT} exists already (use a new TAG; nothing is overwritten)"; exit 1; fi
fi
mkdir -p "${OUT}/slurm" "${OUT}/results"
for d in samples lagrange_multipliers lagrange_multipliers_regularised entropy_bounds sampling_times aux_moments; do
  mkdir -p "${OUT}/saved_results/${d}"
done
SPEC="${OUT}/results/spectrum.pt"

seed_cmds() {   # $1 seed, $2 spectrum file (deterministic: every task may write its own copy, identical up to round-off)
  local S="$1" SP="$2"
  echo "python -u gauss_scat4_test/make_spectrum.py --M ${M} --J ${J} --source ${SOURCE} --out ${SP} || exit 1"
  echo "echo '======== seed ${S}' \$(date)"
  echo "python -u gaussian_experiment/run_SDE.py --M ${M} --J ${J} --Q 1 --terms ${TERMS} --hurst 0.5 --intermittency 0.0 --spectrum_file ${SP} --n1 ${N1} --data_seed ${DATA_SEED} --seed ${S} --nt ${NT} --schedule two_phase --n_bulk ${NBULK} --t_switch 0.9 --gap_end 5e-5 --sigma ${SIGMA} --regularization ${RIDGE} --solve_float64 --reg_solver thomas --n_subsample 1 --reg_mode moment --reg_ridge ${RIDGE} --lam 1e-3 --save_reg_system --reg_system_dir ${OUT}/results/reg_system --outdir ${OUT} --label ${LABEL} || echo FAILED SDE seed ${S}"
  echo "SYS=\$(ls ${OUT}/results/reg_system/*_seed_${S}_*${LABEL}*.pt) && python -u codes/resolve_theta_reg.py \${SYS} --lams ${LAMS} --ridge ${RIDGE} --schedule uniform --mask auto --diagnose 0 --outdir ${OUT}/results/theta_reg || echo FAILED Theta_reg seed ${S}"
}

CMDS=""; ARRAY=""
if [ "${PHASE}" = "all" ]; then
  for S in ${SEEDS}; do CMDS+="$(seed_cmds "${S}" "${SPEC}")"$'\n'; done
elif [ "${PHASE}" = "run" ]; then
  i=0; CMDS="case \${SLURM_ARRAY_TASK_ID} in"$'\n'
  for S in ${SEEDS}; do
    CMDS+="${i})"$'\n'"$(seed_cmds "${S}" "${OUT}/results/spectrum_seed${S}.pt")"$'\n'";;"$'\n'; i=$((i + 1))
  done
  CMDS+="esac"$'\n'; ARRAY="#SBATCH --array=0-$((i - 1))"
fi
if [ "${PHASE}" != "run" ]; then
[ "${PHASE}" = "finish" ] && CMDS+="python -u gauss_scat4_test/make_spectrum.py --M ${M} --J ${J} --source ${SOURCE} --out ${SPEC} || exit 1"$'\n'
CMDS+="echo '======== finish' \$(date)"$'\n'
if [ -n "${NEW_LAMS:-}" ]; then     # extra lams solved here from the saved systems (no SDE rerun); xseed uses LAMS + NEW_LAMS
  CMDS+="for SYS in ${OUT}/results/reg_system/*${LABEL}*.pt; do python -u codes/resolve_theta_reg.py \${SYS} --lams ${NEW_LAMS} --ridge ${RIDGE} --schedule uniform --mask auto --diagnose 0 --outdir ${OUT}/results/theta_reg || echo FAILED Theta_reg \${SYS}; done"$'\n'
  LAMS="${LAMS} ${NEW_LAMS}"
fi
CMDS+="python -u lam_selection_criterion/turb_xseed.py --systems ${OUT}/results/reg_system/*${LABEL}*.pt --sweep_dir ${OUT}/results/theta_reg --lams ${LAMS} --ridge ${RIDGE} --arm ${LABEL} --out ${OUT}/results || exit 1"$'\n'
CMDS+="python -u gauss_scat4_test/check.py --root ${OUT} --arm ${LABEL} --spectrum_file ${SPEC} || echo FAILED check"$'\n'
CMDS+="export KL_RESULTS=${OUT}/results GAUSS4_ROOT=${OUT} GAUSS4_ARM=${LABEL}"$'\n'
CMDS+="python -u kl_groups/cache_energies.py --dataset gauss4 --seeds ${SEEDS} && python -u kl_groups/compute_H.py --dataset gauss4 || echo FAILED H"$'\n'
fi
echo "gauss_scat4_test PHASE=${PHASE}: M ${M} J ${J} source ${SOURCE}, seeds ${SEEDS}, n1 ${N1}, nt ${NT}, sigma ${SIGMA}, lams ${LAMS}; out ${OUT}; ${GPU} dev, ${TIME}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=g4_${TAG}_${PHASE}
${ARRAY}
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
