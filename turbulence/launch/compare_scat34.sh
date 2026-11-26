#!/bin/bash
# Compare three B-family runs (seed 900) on the cross-scale / single-scale statistics and the dynamical categories
# (2026-11-18). CPU work as a GPU dev job (user rule 11-17: never prepost; billed 1 GPU x elapsed).
#   old_B+Scat4_nt55k      turbulence_nested_M3_bicw_allmass_scat4                 (11-01, nt 55000, no terminal phase)
#   B+Scat4_nt27k_term     turbulence_nested_M3_bicw_allmass_scat4_pf_term_fast    (11-17, nt 27500 + terminal phase)
#   B+Scat3+Scat4d_nt27k   turbulence_nested_M3_bicw_allmass_scat34d_pf_term_fast  (11-17, same + Scat3 Re/Im, Scat4 diag)
# Steps (a failure of one does not stop the other):
#   1. turbulence/sample_categories/compare_categories.py -> ${OUT_CAT}  (counts.csv, cat_<category>.png)
#   2. turbulence/cross_scale_compare.py                  -> ${OUT_CS}   (summary.csv, *.png, curves.npz, cache/)
# Usage (Jean Zay, repo root):  GPU=h100 bash turbulence/launch/compare_scat34.sh
# overrides: GPU (v100|h100, mandatory), TIME (01:30:00), CPUS (24), SEED (900), OUT_CAT, OUT_CS,
#            LABELS / NAMES (space-separated, same length), SEED_LABEL + SEEDS (noise runs, same M), SKIP_CAT=1

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${LAUNCH_DIR}/../.." && pwd)"
: "${GPU:?set GPU=v100 or GPU=h100}"
TIME="${TIME:-01:30:00}"
CPUS="${CPUS:-24}"
SEED="${SEED:-900}"
OUT_CAT="${OUT_CAT:-${PROJECT_DIR}/turbulence/sample_categories/compare_scat34}"
OUT_CS="${OUT_CS:-${PROJECT_DIR}/turbulence/figures/cross_scale_scat34}"
case "${GPU}" in
  v100) ACCOUNT=wbg@v100; CONSTRAINT=v100; PARTITION=gpu_p13; QOS=qos_gpu-dev; MODULE_PRE=""; CPUS="${CPUS_V100:-10}" ;;
  h100) ACCOUNT=wbg@h100; CONSTRAINT=h100; PARTITION=gpu_p6;  QOS=qos_gpu_h100-dev; MODULE_PRE="arch/h100" ;;
  *) echo "Error: GPU must be v100 or h100"; exit 1 ;;
esac
LABELS="${LABELS:-turbulence_nested_M3_bicw_allmass_scat4 turbulence_nested_M3_bicw_allmass_scat4_pf_term_fast turbulence_nested_M3_bicw_allmass_scat34d_pf_term_fast}"
NAMES="${NAMES:-old_B+Scat4_nt55k B+Scat4_nt27k_term B+Scat3+Scat4d_nt27k}"
# run-side noise (sigma_seed) = spread over the seeds of SEED_LABEL: it must be runs of the SAME M as LABELS
# (2026-11-25; the script default turbulence_nested_M3 is M=256). SKIP_CAT=1 skips step 1.
SEED_LABEL="${SEED_LABEL:-turbulence_nested_M3}"
SEEDS="${SEEDS:-900 901 902 903 904}"
SKIP_CAT="${SKIP_CAT:-0}"
LOGS="${OUT_CS}/logs"
mkdir -p "${LOGS}" "${OUT_CAT}"
echo "compare_scat34: seed ${SEED}; categories -> ${OUT_CAT}; cross-scale -> ${OUT_CS}; ${GPU} dev (${QOS}), ${CPUS} CPUs, ${TIME}"

JOBID=$(sbatch --parsable --qos=${QOS} <<EOT
#!/bin/bash
#SBATCH --job-name=cmpscat34
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${LOGS}/slurm_%x_%j.log

module purge
${MODULE_PRE:+module load ${MODULE_PRE}}
module load pytorch-gpu/py3/2.8.0
export OMP_NUM_THREADS=${CPUS}
cd "${PROJECT_DIR}"
echo "=== 1. categories \$(date)"
[ "${SKIP_CAT}" = "1" ] || python -u turbulence/sample_categories/compare_categories.py --labels ${LABELS} --names ${NAMES} --seed ${SEED} --out "${OUT_CAT}" || echo "FAILED categories"
echo "=== 2. cross-scale \$(date)"
python -u turbulence/cross_scale_compare.py --labels ${LABELS} --names ${NAMES} --seed ${SEED} --seed_label ${SEED_LABEL} --seeds ${SEEDS} --out "${OUT_CS}" || echo "FAILED cross_scale"
echo "=== done \$(date)"
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted ${JOBID}"
