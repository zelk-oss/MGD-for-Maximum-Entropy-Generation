#!/bin/bash
# timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M")

# ── PARAMETERS ───────────────────────────────────────
SEED_LIST=({300..350})
N1_LIST=(5000)
NT_LIST=(2000)
SIGMA_LIST=(1.0)

# fBm/MRW data generation (replaces SUBSERIES_LEN/TARGET_LEN/coarse-graining:
# gaussian_experiment/run_SDE.py generates data directly at length M, no
# real-recording windowing/coarse-graining step is needed).
M=8
HURST_LIST=(0.5)          # Hurst exponent of the underlying fBm
INTERMITTENCY_LIST=(0.0)  # MRW intermittency (lam); 0.0 = pure fBm increments

J=3
Q=1
INTERPOLANT="Cos"
SCHEDULE_EXPONENT=2
REGULARIZATION=1e-5
LAM=1e-03
N_SUBSAMPLE=1              # 1 = no block averaging (full fine grid)
REG_SOLVER="thomas"        # NEW: 'thomas' (float64 block-Thomas) | 'dense' (legacy)
REG_RIDGE=0.0              # NEW: M_k += REG_RIDGE*diag(M_k) in the Theta_reg solve; 0 = off
BATCH_SIZE="5000"          # optional

# Human-readable label appended to the config name (same value for every
# seed in this sweep -- see --label in run_SDE.py). Leave empty for none.
LABEL="gaussian_d8_thomas"

# Whether to apply fit_micro() physical normalization to
# that supports it before fitting (--normalize_potentials in run_SDE.py,
# threaded into run_experiment's normalize_potentials arg). true/false.
NORMALIZE_POTENTIALS=false

# 🔥 POTENTIALS (MAIN CONTROL)
TERMS=(
    Scattering_Second_Order
    Scattering_Fourth_Order_Mod2_Real_Q1
    Scattering_Fourth_Order_Mod2_Imag_Q1
    )

# ── SLURM & JEAN ZAY QOS CONFIG ──────────────────────
ACCOUNT="wbg@v100"
CONSTRAINT="v100"
PARTITION="gpu_p13"
TIME="00:30:00"
CPUS=10
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

LABEL_FLAG=""
if [ -n "${LABEL}" ]; then
    LABEL_FLAG="--label ${LABEL}"
fi

NORM_FLAG=""
if [ "${NORMALIZE_POTENTIALS}" = "true" ]; then
    NORM_FLAG="--normalize_potentials"
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
    --reg_solver ${REG_SOLVER} \
    --reg_ridge ${REG_RIDGE} \
    --outdir "${OUTDIR}" \
    --terms ${TERMS_STR} \
    ${LABEL_FLAG} \
    ${NORM_FLAG} \
    ${BS_FLAG}
EOT