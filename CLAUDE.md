# Cluster access

This session has no execution access to Jean Zay or Lamsade. `ssh`, `scp`,
`rsync`, SLURM commands (`sbatch`, `scancel`, `squeue`), and the `push`/
`pull`/`pull_results` scripts in `~/scripts/` are all blocked by permission
rules — attempting them will be refused, not just prompted.

When something needs to happen on the cluster — fetching results, pushing
code, submitting or cancelling a job — output the exact command in a
copy-pasteable block and stop there. The user runs it and pastes back the
output if there's something to report back.

# Local safety

- `rm -rf` and `git push --force` are blocked the same way, with no
  exceptions carved out for a specific case, even if it looks safe in the
  moment. If either is genuinely needed, explain what and why, and let the
  user run it themselves or temporarily relax the rule in
  `~/.claude/settings.json`.
- `data/`, `saved_results/` (and its per-subproject copies under `jets/`,
  `turbulence/`, `codes/`), and `*/experiments/*/figures/` hold compute
  results that took real GPU/cluster time to produce. Editing/writing
  there is blocked by permission rules (read/analysis is fine). If a file
  in one of these paths genuinely needs to change, that should happen by
  running the actual pipeline code, not by directly editing the output.
- Before starting substantial work, check `git status`. Uncommitted
  changes aren't protected by git history — if there's meaningful
  uncommitted work, flag it and suggest committing before proceeding.
