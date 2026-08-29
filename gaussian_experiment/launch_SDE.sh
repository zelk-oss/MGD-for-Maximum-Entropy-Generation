#!/bin/bash
# timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M")

# ── PARAMETERS ───────────────────────────────────────
SEED_LIST=({300..350})
N1_LIST=(5000)
NT_LIST=(60000)
SIGMA_LIST=(3.5)

# fBm/MRW data generation (replaces SUBSERIES_LEN/TARGET_LEN/coarse-graining:
# gaussian_experiment/run_SDE.py generates data directly at length M, no
# real-recording windowing/coarse-graining step is needed).
M=256
HURST_LIST=(0.5)          # Hurst exponent of the underlying fBm
INTERMITTENCY_LIST=(0.0)  # MRW intermittency (lam); 0.0 = pure fBm increments

J=8
Q=3
INTERPOLANT="Cos"
SCHEDULE_EXPONENT=2
REGULARIZATION=0.01
LAM=2e-08
N_SUBSAMPLE=300
BATCH_SIZE=""          # optional

# 🔥 POTENTIALS (MAIN CONTROL)
TERMS=(
    L_2_lowpass
    L_6
    L_6_psi
    Scattering_Fourth_Order_Mod2_Real_Q1
    Scattering_Fourth_Order_Mod2_Imag_Q1
    Scalar_psi_gaussianK
    Scalar_morlet_gaussianK
)

# ── SLURM & JEAN ZAY QOS CONFIG ──────────────────────
ACCOUNT="wbg@h100"
CONSTRAINT="h100"
PARTITION="gpu_p6"
TIME="20:00:00"
CPUS=24
NGPUS=1

# ── paths ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_SDE.py"
OUTDIR="${SCRIPT_DIR}"
mkdir -p "${OUTDIR}/saved_results/logs"

# Added HURST_LIST/INTERMITTENCY_LIST to the TOTAL calculation
TOTAL=$(( ${#N1_LIST[@]} * ${#NT_LIST[@]} * ${#SEED_LIST[@]} * ${#SIGMA_LIST[@]} \
          * ${#HURST_LIST[@]} * ${#INTERMITTENCY_LIST[@]} ))

# ── optional flags ───────────────────────────────────
BS_FLAG=""
if [ -n "${BATCH_SIZE}" ]; then
    BS_FLAG="--batch_size ${BATCH_SIZE}"
fi

# ── terms → string ───────────────────────────────────
TERMS_STR="${TERMS[@]}"

# ── submit ───────────────────────────────────────────
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=gaussian_sde_sweep
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH --array=0-$((TOTAL-1))
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${NGPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --output=${OUTDIR}/saved_results/logs/slurm_%A_%a.log

module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
cd "${SCRIPT_DIR}"

# Pass arrays into the Slurm environment
SEED_LIST=(${SEED_LIST[@]})
N1_LIST=(${N1_LIST[@]})
NT_LIST=(${NT_LIST[@]})
SIGMA_LIST=(${SIGMA_LIST[@]})
HURST_LIST=(${HURST_LIST[@]})
INTERMITTENCY_LIST=(${INTERMITTENCY_LIST[@]})

# Calculate lengths
nSEED=\${#SEED_LIST[@]}
nN1=\${#N1_LIST[@]}
nNT=\${#NT_LIST[@]}
nSIGMA=\${#SIGMA_LIST[@]}
nHURST=\${#HURST_LIST[@]}
nINTERMITTENCY=\${#INTERMITTENCY_LIST[@]}

# Grab the current array ID
IDX=\${SLURM_ARRAY_TASK_ID}

# Decode the 6D grid
SEED_INDEX=\$(( IDX % nSEED ));            IDX=\$(( IDX / nSEED ))
SIGMA_INDEX=\$(( IDX % nSIGMA ));          IDX=\$(( IDX / nSIGMA ))
NT_INDEX=\$(( IDX % nNT ));                IDX=\$(( IDX / nNT ))
N1_INDEX=\$(( IDX % nN1 ));                IDX=\$(( IDX / nN1 ))
HURST_INDEX=\$(( IDX % nHURST ));          IDX=\$(( IDX / nHURST ))
INTERMITTENCY_INDEX=\$(( IDX % nINTERMITTENCY ))

# Extract the values for this specific job
SEED_VAL=\${SEED_LIST[\$SEED_INDEX]}
SIGMA_VAL=\${SIGMA_LIST[\$SIGMA_INDEX]}
NT=\${NT_LIST[\$NT_INDEX]}
N1=\${N1_LIST[\$N1_INDEX]}
HURST_VAL=\${HURST_LIST[\$HURST_INDEX]}
INTERMITTENCY_VAL=\${INTERMITTENCY_LIST[\$INTERMITTENCY_INDEX]}

srun python "${PYTHON_SCRIPT}" \
    --seed \${SEED_VAL} \
    --timestamp "$TIMESTAMP" \
    --n1 \${N1} \
    --nt \${NT} \
    --M ${M} \
    --hurst \${HURST_VAL} \
    --intermittency \${INTERMITTENCY_VAL} \
    --J ${J} \
    --Q ${Q} \
    --sigma \${SIGMA_VAL} \
    --interpolant ${INTERPOLANT} \
    --schedule_exponent ${SCHEDULE_EXPONENT} \
    --regularization ${REGULARIZATION} \
    --lam ${LAM} \
    --n_subsample ${N_SUBSAMPLE} \
    --outdir "${OUTDIR}" \
    --terms ${TERMS_STR} \
    ${BS_FLAG}
EOT
