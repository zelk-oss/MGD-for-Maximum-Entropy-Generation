#!/bin/bash
# Turbulence d = 64 (coarse x3, 2000 windows from 2000 trajectories), 4 configurations = {ridge 1e-4, jac_minres} x
# {with L_6_psi (near-copies L_6[5,6] ~ L_6_psi[14,15], 1 - |corr| ~ 5e-7..1e-6), without L_6_psi}.
# x SEEDS (default 900-904) = 20 array tasks, one run each; same settings otherwise; eta ridge 1e-7 and float64 Gram for all; Theta_reg in-run with
# lam 1e-3, data ridge 1e-4 (moment mode). jac_minres runs abort on the live criteria (--monitor_abort).
# Live monitoring file: turbulence/saved_results/aux_moments/<config>_monitor.pt (flushed every 200 steps).
# Cost not measured (smoke not run): guess 10-30 min per run on H100.
# Default queue t3 (QOS=""): 20 tasks exceed the 10-job dev limit. Finished runs are reloaded, not redone.
# Usage (repo root on Jean Zay): bash krylov_stopping/launch_turb_d64.sh   overrides: SEEDS="900 901" QOS=qos_gpu_h100-dev TIME=02:00:00
#   CONFIGS="1" (subset of 0 copies_ridge, 1 copies_jacminres, 2 nopsi_ridge, 3 nopsi_jacminres)
#   NOABORT=1   (jac_minres runs without the live abort; the monitoring file is still written)
#   e.g. rerun of the 3 aborted seeds: SEEDS="901 903 904" CONFIGS="1" NOABORT=1 bash krylov_stopping/launch_turb_d64.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
QOS="${QOS-}"
SEEDS=(${SEEDS:-900 901 902 903 904})
NS=${#SEEDS[@]}
CONFIGS=(${CONFIGS:-0 1 2 3})
NC=${#CONFIGS[@]}
ABORT=$([ "${NOABORT:-0}" = 1 ] && echo "" || echo "--monitor_abort")
TIME="${TIME:-02:00:00}"
LOGDIR="${HERE}/results/turb_d64/slurm"
REGDIR="${SCRATCH}/MGD-for-Maximum-Entropy-Generation/krylov_stopping/turb_d64_reg_system"
mkdir -p "${LOGDIR}" "${REGDIR}"

JOBID=$(sbatch --parsable <<EOT
#!/bin/bash
#SBATCH --job-name=turb_d64
#SBATCH -A wbg@h100
#SBATCH --array=0-$(( NC * NS - 1 ))
#SBATCH --partition=gpu_p6
${QOS:+#SBATCH --qos=${QOS}}
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=${TIME}
#SBATCH --output=${LOGDIR}/slurm_%x_%A_%a.log

module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
cd "${PROJECT_DIR}"

FULL="L_6 L_6_psi L_2_lowpass Scalar_psi_gaussianK Scalar_morlet_gaussianK Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1"
NOPSI="L_6 L_2_lowpass Scalar_psi_gaussianK Scalar_morlet_gaussianK Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1"
SEEDS=( ${SEEDS[@]} )
CONFIGS=( ${CONFIGS[@]} )
CFG=\${CONFIGS[\$(( SLURM_ARRAY_TASK_ID / ${NS} ))]}
SEED=\${SEEDS[\$(( SLURM_ARRAY_TASK_ID % ${NS} ))]}
case \${CFG} in
  0) TERMS="\${FULL}";  SOLVER="--theta_solver ridge --regularization 1e-4";               LABEL=d64_copies_ridge1e-4 ;;
  1) TERMS="\${FULL}";  SOLVER="--theta_solver jac_minres --regularization 0 ${ABORT}"; LABEL=d64_copies_jacminres ;;
  2) TERMS="\${NOPSI}"; SOLVER="--theta_solver ridge --regularization 1e-4";               LABEL=d64_nopsi_ridge1e-4 ;;
  3) TERMS="\${NOPSI}"; SOLVER="--theta_solver jac_minres --regularization 0 ${ABORT}"; LABEL=d64_nopsi_jacminres ;;
esac
python turbulence/run_SDE.py --seed \${SEED} --n1 2000 --n1_spread --subseries_len 512 --target_len 64 --J 6 --Q 3 \
    --terms \${TERMS} --deduplicate_filters --dedup_tol 1e-8 \
    --nt 10000 --schedule two_phase --n_bulk 2000 --t_switch 0.9 --gap_end 5e-5 --sigma 3.5 \
    \${SOLVER} --eta_regularization 1e-7 --g64 --solve_float64 \
    --lam 1e-3 --reg_solver thomas --n_subsample 1 --reg_mode moment --reg_ridge 1e-4 \
    --save_reg_system --reg_system_dir "${REGDIR}" \
    --batch_size 2000 --time_limit_min 110 --label \${LABEL}
EOT
)
if [ -z "${JOBID}" ]; then echo "Error: sbatch returned no job id"; exit 1; fi
echo "Submitted array job ${JOBID} ($(( NC * NS )) runs: configurations ${CONFIGS[*]} x seeds ${SEEDS[*]}, abort flag "${ABORT}"). Logs: ${LOGDIR}"
echo "Monitoring files (live): turbulence/saved_results/aux_moments/*d64_*_monitor.pt"
