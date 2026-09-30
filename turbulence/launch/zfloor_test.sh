#!/bin/bash
# Full potential set with the per-channel |z| floor in Scalar_GGD_KRegion and the
# near-copy filter dropped (--dedup_tol 1e-8).
#
# 2026-09-28: Scalar_GGD_KRegion used |z| -> sqrt(z^2 + 1e-6), i.e. |z| >= 1e-3, while
# the finest psi channels live at |z| ~ 1e-4 (all region cuts of channels 0-7 below
# 1e-3). Their regions were invisible to forward/grad (channel 0 saw [0, 0, 0.5, 99.5]%
# of the data per region instead of [40, 50, 8, 2]%), their statistics degenerate, and
# prune_collinear dropped 52 of 81 (schedtest_two_phase_reg1e-4). Near-zero mass
# (sparsity) and tails of the fine channels were therefore unconstrained: the suspected
# cause of the fine-scale noise (top-octave excess, sparsity ratio ~2x) and part of the
# rare-event deficit. Now eps is per channel, quantile(|z_j|, 1e-3)^2 at fit time
# (codes/potentials/potentials_1d.py, eps_quantile). Checked on 1000 data series: all 77
# statistics survive the pruning; with --dedup_tol 1e-8 (drops filter 21, a near-copy of
# filter 20) 73 of 73, condition number of the normalized gradient matrix 12 (9.6e4
# with the pair). Scalar_morlet_gaussianK gets the same fix (its channels 0-1 cores
# were blind too).
#
# First launch (2026-09-28; two_phase = job 268295): fit OK (psi 78 of 78 statistics kept,
# 281 in total, 1.09 s/step), but two_phase blew up at step 163 (t = 0.015): regions that
# are empty on the early, noise-like walkers (outer regions of the coarsest psi channel,
# Gram diagonal 1e-16..1e-8 of its data value) are no longer EXACTLY dead with the new
# floor, so the solver solved for them and occasionally produced ~100x drifts. Fix:
# Scalar_GGD_KRegion.near_empty_tol = 1e-6 -> the SDE sets a region statistic's
# coefficient to 0 at steps where its Gram diagonal is below 1e-6 of its data value
# (codes/sde_routines.py, _live_potentials). Reproduced and fixed in a 64-walker smoke
# test (no masking: non-finite at step 18; masking: finite, like the old statistic set).
#
# Same set-up as the schedtest full-set runs otherwise (n1 8500, subseries 512, ridge
# 1e-4, float64 solves, seed 900), both time grids:
#   two_phase  10k uniform steps on [0, 0.9], then NT - 10k steps at constant h/(1-t)
#   adaptive   h from the previous step's moment error, at most NT steps; stops
#              cleanly at the loop budget if it has not reached the end
#
# Budget: r goes from 233 to ~276 statistics; per-step cost estimated ~+18% (NOT
# measured): two_phase at NT=60k ~21.8 h > the 19h30 loop budget of a 20 h job.
# MEASURED on job 268295: r = 281, 1.09 s/step -> ~18.2 h at 60k steps (+ ~5 min fit),
# ~1 h under the budget. If a node is slower, _check_time_budget aborts after ~130 steps
# and logs the measured s/it; relaunch with fewer steps, e.g.
#   ONLY="zfloor_two_phase_reg1e-4" NT_TWO_PHASE=52000 bash zfloor_test.sh
# (NT_TWO_PHASE - N_BULK = steps in the tail; N_BULK can be lowered too). adaptive
# always finishes. Saved regularised system: ~1.2 MB/step at r~276 (~75 GB at 60k
# steps, host RAM and SCRATCH), vs ~0.9 MB/step at r=233.
#
# 2026-09-30 (C4 of notes/guth_reg_audit_0930 + notes/eps_conditioning_0930): thomas runs
# now default to --reg_mode moment (config tag _moment), and COND_EVERY=N logs the
# per-step Gram conditioning to saved_results/aux_moments/<config>_cond.pt:
#         ONLY="zfloor_two_phase_reg1e-4" COND_EVERY=10 bash zfloor_test.sh
#
# Usage:  bash zfloor_test.sh                                   (both runs, seed 900)
#         ONLY="zfloor_adaptive_reg1e-4" bash zfloor_test.sh      (a subset)
#         ONLY="zfloor_two_phase_reg1e-4" SEEDS="901 902 903 904" bash zfloor_test.sh
#                                   (more seeds: one SLURM array job, one task per seed)
# Compare afterwards with compare_runs.ipynb / compare_july_vs_sept.ipynb against
# schedtest_two_phase_reg1e-4 (same data and settings, old floor).

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FULL=(L_6 L_6_psi L_2_lowpass Scalar_psi_gaussianK Scalar_morlet_gaussianK
      Scattering_Fourth_Order_Mod2_Real_Q1 Scattering_Fourth_Order_Mod2_Imag_Q1)

SEEDS="${SEEDS:-900}"
NT_TWO_PHASE="${NT_TWO_PHASE:-60000}"
NT_ADAPTIVE="${NT_ADAPTIVE:-60000}"
N_BULK="${N_BULK:-10000}"
ONLY="${ONLY:-zfloor_two_phase_reg1e-4 zfloor_adaptive_reg1e-4}"

TWO_PHASE="--schedule two_phase --n_bulk ${N_BULK} --t_switch 0.9 --gap_end 5e-5"
ADAPTIVE="--schedule adaptive --adapt_tol 1e-3 --adapt_ratio_max 1e-3 --gap_end 5e-5"

submit() {   # submit <label> <nt> <schedule flags (one string)> <terms...>
  local label=$1 nt=$2 sched=$3; shift 3
  [[ " ${ONLY} " == *" ${label} "* ]] || return 0
  (
    EXP_NAME="${label}"
    SCHEDULE_ARGS="${sched}"
    TERMS=("$@")
    REGULARIZATION=1e-4
    SEED_LIST=(${SEEDS})

    N1=8500
    SUBSERIES_LEN=512
    TARGET_LEN=256
    J=8
    Q=3
    SIGMA=3.5
    NT=${nt}
    BATCH_SIZE=5000

    SOLVE_FLOAT64=true
    DEDUPLICATE_FILTERS=true
    DEDUP_TOL=1e-8

    LAM=0.0
    N_SUBSAMPLE=1
    REG_SOLVER=thomas
    REG_RIDGE=0.0
    SAVE_REG_SYSTEM=true
    SKIP_REG_SOLVE=true
    REG_SYSTEM_DIR="${SCRATCH:+${SCRATCH}/MGD-for-Maximum-Entropy-Generation/turbulence/reg_system}"

    TIME="20:00:00"
    CPUS=24
    ACCOUNT="wbg@h100"
    CONSTRAINT="h100"
    PARTITION="gpu_p6"
    MODULE_PRE="arch/h100"

    source "${LAUNCH_DIR}/_submit_turb_sweep.sh"
  )
}

submit zfloor_two_phase_reg1e-4 "${NT_TWO_PHASE}" "${TWO_PHASE}" "${FULL[@]}"
submit zfloor_adaptive_reg1e-4  "${NT_ADAPTIVE}"  "${ADAPTIVE}"  "${FULL[@]}"
