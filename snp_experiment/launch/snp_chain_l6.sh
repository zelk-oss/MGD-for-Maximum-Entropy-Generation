#!/bin/bash
# The S&P experiment WITH the L_6 confinement potential (same construction as the Eulerian turbulence runs), whole chain
# for the models M1 (full: scat1-4 + scalar + L_6) and M0 (no scalar: scat1-4 + L_6), arms snp_M1_L6, snp_M0_L6:
#   SDE (5 seeds, saved system) -> Theta_reg sweep (9 lams, unconstrained, sign report with the L_6 block)
#   -> xseed lambda choice (end0.99) -> L_6 sign gate on all seeds -> [only if violated] Theta_reg with L_6 < 0
# Everything is chained with SLURM dependencies (SDE tasks -> Theta_reg tasks aftercorr; then afterok), nothing manual.
# After it: python snp_experiment/logp_snp.py --l6 (log p per family, uses the L_6<0 Theta when the gate fired).
#
# Usage (repo root on Jean Zay):
#   bash snp_experiment/launch/snp_chain_l6.sh                    # M1 and M0
#   MODELS="M1" bash snp_experiment/launch/snp_chain_l6.sh
#   DRY=1 bash ...                                                  # print the commands, submit nothing
# overrides are passed on to the launchers: SEEDS, NT, GPU (v100), TIME

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPU="${GPU:-v100}"
export L6=1
sub() { awk '/^Submitted/{for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) {print $i; exit}}'; }

for M in ${MODELS:-M1 M0}; do
    echo "================ ${M} + L_6 ================"
    if [ -n "${DRY}" ]; then
        echo "(SDE: MODEL=${M} L6=1 snp_models.sh has no dry mode, not called)"
        for s in snp_theta_reg snp_xseed snp_l6neg; do MODEL=${M} DRY=1 bash "${HERE}/${s}.sh" | head -3; done; continue
    fi
    OUT=$(MODEL=${M} bash "${HERE}/snp_models.sh"); echo "${OUT}"
    SDE=$(echo "${OUT}" | awk '/Submitted array job/{print $4}')
    [ -n "${SDE}" ] || { echo "Error: no SDE job id for ${M}"; exit 1; }
    OUT=$(MODEL=${M} DEPEND=${SDE} bash "${HERE}/snp_theta_reg.sh"); echo "${OUT}"; TH=$(echo "${OUT}" | sub)
    [ -n "${TH}" ] || { echo "Error: no Theta_reg job id for ${M}"; exit 1; }
    OUT=$(MODEL=${M} DEPEND=${TH} bash "${HERE}/snp_xseed.sh"); echo "${OUT}"; XS=$(echo "${OUT}" | sub)
    [ -n "${XS}" ] || { echo "Error: no xseed job id for ${M}"; exit 1; }
    OUT=$(MODEL=${M} DEPEND=${XS} bash "${HERE}/snp_l6neg.sh"); echo "${OUT}"
    echo "${M}: SDE ${SDE} -> Theta_reg ${TH} -> xseed ${XS} -> gate + L_6<0 (see above)"
done
