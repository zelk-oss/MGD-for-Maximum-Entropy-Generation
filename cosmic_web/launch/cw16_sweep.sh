#!/bin/bash
# 16x16 cosmic-web test bed (2026-11-09): why does Theta_reg fail in 2D? base_real (L_6 + orders 1-3 Real), FACT=16
# (256 -> 16, db4), J=3 (r = 148), sigma^2 = 5.5, 5 seeds each. Five configurations (one cosmic_web.sh array each):
#   cw16_np100_nt3000        100 walkers, two_phase, NT 3000     (production settings at 16x16)
#   cw16_np400_nt3000        400 walkers (each map 4x)
#   cw16_np100_nt12000       time step / 4
#   cw16_np400_nt12000       both
#   cw16_np100_nt3000_unif   uniform grid (S&P grid; ends at t = 1 - 1/NT)
# Usage (repo root on Jean Zay):  DATA_PATH=data/data_files/quijote.npy GPU=v100 bash cosmic_web/launch/cw16_sweep.sh
#   ONLY="cw16_np100_nt3000 ..." to submit a subset; TIME / QOS / SEEDS pass through to cosmic_web.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DATA_PATH:?set DATA_PATH to quijote.npy}"
: "${GPU:?set GPU=v100 or h100}"
DATA_PATH="$(realpath "${DATA_PATH}")"          # the job cd's into cosmic_web/: relative paths would break
export DATA_PATH GPU ARM=base_real FACT=16 J=3 TIME="${TIME:-01:00:00}"

submit() {   # label n_particles nt schedule
    local label=$1
    if [ -n "${ONLY:-}" ] && [[ " ${ONLY} " != *" ${label} "* ]]; then return; fi
    LABEL=${label} N_PARTICLES=$2 NT=$3 SCHEDULE=$4 bash "${HERE}/cosmic_web.sh"
}
submit cw16_np100_nt3000      100  3000 two_phase
submit cw16_np400_nt3000      400  3000 two_phase
submit cw16_np100_nt12000     100 12000 two_phase
submit cw16_np400_nt12000     400 12000 two_phase
submit cw16_np100_nt3000_unif 100  3000 uniform
