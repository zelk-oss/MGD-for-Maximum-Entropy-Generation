# Repository structure

- `codes/` — shared implementation: `utils.py` (SDE utilities, `save_results`/`load_results`, symmetrization), `utils_experiment.py` (experiment naming/config resolution, `run_experiment`), `potentials/` (`potentials_1d.py`, `potentials_2d.py`, `potentials_1d_condi.py`, `potentials_scalar.py`, `potentials_classes/`).
- `jets/run_jet_SDE.py`, `turbulence/run_SDE.py`, and `gaussian_experiment/run_SDE.py` — the experiment entry points (argparse CLIs). The last runs on fractional Brownian motion / multifractal random walk increments generated via `data/synthetic_data_generator.py` (backed by `data/standard_models/`, ported from the `conditional_mgd` project / github.com/RudyMorel/scattering_spectra) rather than on real recorded data.
- `bimodal_experiment/` — standalone 1-D experiment (moved out of `turbulence/` on 2026-09-24): K-run MGD vs time-regularised (REG) estimator on the bimodal density, energy variance vs Cramér–Rao. Own scalar SDE port (`sde_routines_scalar_reg.py`), but the REG solve uses the shared float64 Thomas from `codes/` (`codes/lam_selection.py` batches it over runs × λ). λ is chosen after the run by `select_lambda.py`; launched with `bimodal_experiment/launch/bimodal_sweep.sh`. Results in `bimodal_experiment/results_bim_theta_beta/`.
- Results land under `saved_results/` (and per-subproject copies in `jets/`, `turbulence/`, `codes/`): `samples/`, `lagrange_multipliers/`, `lagrange_multipliers_regularised/`, `entropy_bounds/`, `sampling_times/` — `<config>.pt` (PyTorch) files as of 2026-08-25; older runs may lack the `.pt` extension, `load_results` handles both.
- Each run also gets `*/experiments/<config_name>/` with `config.json` (provenance, always kept), `logs/run.log`, and `figures/` (diagnostic plots).
- `notebooks/` — analysis notebooks; not run on the cluster.

# Execution environment

All actual computation runs on Jean Zay (SLURM cluster) — this machine only
has CPU. Jobs are submitted by the launchers in `*/launch/*.sh` (and a few
`*/launch_*.sh`), run with `bash <launcher>` from the repo root on JZ with
settings as environment variables (`GPU=v100|h100`, `QOS=`, `TIME=`,
`SEEDS=`, ...); SLURM logs go to `<experiment>/saved_results/logs/`. Read the
launcher's header for its options before using it.

## Smoke tests run on JZ, not locally (since 2026-11-10)

The user wants local computing kept to a minimum. Locally: only syntax/import
checks (`python -m py_compile`). Smoke tests of the entry points or launchers
run on JZ as a 15-min dev job with tiny sizes (`n1`/`M` in the tens to low
hundreds, `nt` in the tens, one seed), through `~/scripts/jz_submit` and with
the user's approval (see the Jean Zay section of `~/.claude/CLAUDE.md`).
Never size a smoke job to be informative about the science. If it is unclear
whether a size counts as a smoke test, ask.

# Conventions

- Experiment/config naming comes from `build_config_name()` in
  `codes/utils_experiment.py`: `<jetsynth|turbulencesynth|gaussiansynth>_[Re_number<N>_][H<hurst>_intermittency<λ_mrw>_]M<M>_J<J>_Q<Q>_sigma<σ>_nt<nt>_n1_<n1>_lam<λ>_seed_<seed>_terms<hash>[_<label>][_<timestamp>]`.
  The `gaussiansynth` prefix (and `H<hurst>_intermittency<λ_mrw>` segment) triggers off `args.hurst`
  being set, the same way `jetsynth` triggers off `args.Re_number` — see `gaussian_experiment/run_SDE.py`.
- Project's Lustre path: `/lustre/fswork/projects/rech/wbg/ukv59en/MGD-for-Maximum-Entropy-Generation` (see `~/scripts/sync_common.sh`).
- W&B is not used anywhere in this codebase (checked) — don't assume it is, and don't add it without being asked.

# Don't invent

Never invent or approximate a numerical result, and never say a test or a
run passed unless you actually executed it and saw the output yourself.

# Cluster access

Since 2026-11-10 Claude reaches Jean Zay ONLY through the `~/scripts/jz_*`
wrappers, with the limits and rules in the "Jean Zay" section of
`~/.claude/CLAUDE.md`. Direct `ssh`/`scp`/`rsync`, SLURM commands, and the
user's own `push`/`pull`/`pull_results` scripts stay blocked. Lamsade stays
blocked.

When something is outside what the wrappers allow (a job above the limits,
a pull above 30 MB, a deletion, Lamsade), output the exact command in a
copy-pasteable block and stop there.

Format every command the user must run (cluster or local) so it stands out:
a bold `▶ RUN (JZ)` / `▶ RUN (local)` label line, then a fenced ```bash block
(the terminal colours fenced code; plain text cannot be coloured). Never put
a command to run inline in a sentence. The user runs it and pastes back the
output if there's something to report back.

# Local safety

- `rm -rf` and `git push --force` are blocked the same way, with no
  exceptions carved out for a specific case, even if it looks safe in the
  moment. If either is genuinely needed, explain what and why, and let the
  user run it themselves or temporarily relax the rule in
  `~/.claude/settings.json`.
- `data/data_files/`, `saved_results/` (and its per-subproject copies under
  `jets/`, `turbulence/`, `codes/`), and `*/experiments/*/figures/` hold
  compute results that took real GPU/cluster time to produce. Editing/
  writing there is blocked by permission rules (read/analysis is fine). If
  a file in one of these paths genuinely needs to change, that should
  happen by running the actual pipeline code, not by directly editing the
  output.
- Before starting substantial work, check `git status`. Uncommitted
  changes aren't protected by git history — if there's meaningful
  uncommitted work, flag it and suggest committing before proceeding.
