# Regularised theta (Theta_reg): diagnosis and fix, 2026-09-29

Code commits: `e3e0bae` (sign fix + modes), `6453320` (interpolant-side time terms).
Everything below marked **measured** was run and seen; **hypothesis** was not tested directly.

## 1. Starting point

`turbulence/lamtune_select_zfloor.ipynb` (JZ outputs, `zfloor_two_phase_reg1e-4`, seeds
900-904, offline ridge 1e-6): the lam rule suggested **no lam** (windows `[0,0.2)` and
`[0.95,1]` have E[z^2] > 1 at every lam), and Theta_reg explodes on some coefficients
(coef 0 ~1e9 near t=1, coef 140 ~3e6 for t<0.25), **including at lam = 0**.

The selection rule (block z^2 of `R = Theta_reg(lam) - Theta_reg(0)`) cannot see the
explosions: z is normalised by R's own std, so a blow-up looks like more "noise";
the variance diagnostics are medians, which hide a few bad coefficients.

## 2. What was wrong (method vs our changes)

| # | Problem | Ours or Guth's? | Status |
|---|---|---|---|
| 1 | **Sign of the time term.** MGD writes p ∝ exp(+θ·φ) (dH = -θ·ṁ); Guth et al. write p ∝ exp(-U). τ = -∂ₜ log p(Xₜ\|X) was copied from Guth without flipping, so the energy targeted G Θ̇ = c instead of Cov(φ) Θ̇ = -Cov(φ, τ). | ours (bug) | **measured** (1-D Gaussian, `tau_sign_check.py`); fixed |
| 2 | **Uncentred G, c.** Guth's energy network absorbs log Zₜ; our θ·φ cannot, so the time term needs Cov(φ) and Cov(φ, τ). | ours | fixed (modes `fixed`, `guth`) |
| 3 | **τ on the walkers.** τ assumes each sample is cos(a) Z + sin(a) X with X a real data point. On the walkers X is reconstructed as (Xₜ - cos a Z)/sin a and E[τ] = 0 fails. | ours (setting) | **measured**, see §4; interpolant-side terms now saved |
| 4 | **No time weighting / free λ.** Guth: t = noise variance, log t uniform ("found best"), loss E_t[(t/d) DSM + (t/d)^2 TSM], "unit-less", *no tradeoff hyperparameter*. We used a constant λ tuned by hand. | ours | mode `guth` |
| 5 | **Ridge 1e-6 offline vs 1e-4 per step.** Per-step θₜ uses Jacobi-scaled G + 1e-4 I = M + 1e-4 diag(M). Offline 1e-6 is 100x weaker, so Theta_reg(0) ≠ θₜ and explodes. | ours | use `RIDGE=1e-4`; not yet run |
| 6 | **Near-dead potentials** are masked to 0 per step (`_live_potentials`) but not in Theta_reg. | ours | not addressed |

Why only *some* coefficients explode (**hypothesis**): all of the above only become huge in directions
the data barely pins down (near-empty regions, near-duplicate statistics).

### Guth's weights in our Cos time

Y = Xₜ / sin a = X + cot(a) Z, so Guth's noise variance is s = cot²(a), a = πt/2. Their
objective in our t (checked numerically, `guth_weights_check.py`):

- data weight   w_D(t) = 2ȧ cos a / (d sin³a)
- time weight   w_T(t) = sin a cos a / (2ȧ d²)
- effective λ   w_T / w_D = **sin⁴(πt/2) / (π² d)** ≈ 4e-4 near t = 1 for d = 256.

It fades at the **noise** end (t → 0), not at t = 1, and has no free parameter
(`lam` in mode `guth` is a multiplier; 1 = Guth's choice).

## 3. What the code does now

- `SDE._solve_regularised_thomas(..., mode='legacy'|'fixed'|'guth', m, tau_mean, dim)`
  (`codes/sde_routines.py`), weights in the module function `reg_energy_weights`.
  `legacy` = old system, **bit-identical**, still the default (old runs and the bimodal
  experiment unchanged).
- `forward_regularised` saves with the system: `m`, `tau_mean` (walkers), `meta['dim']`,
  and with `interp_time_terms=True` (default, thomas only) `G_I`, `c_I`, `m_I`,
  `tau_mean_I` built on the interpolant samples I_t = cos a x₀ + sin a x₁ (true pairs).
  φ(y_k) and φ(I_k) are now reused from the corrector instead of recomputed.
- `codes/resolve_theta_reg.py`: `--mode`, `--time_source walkers|interpolant`,
  `--results_root` (for old systems: m from `aux_moments` barphi_p), `--dim`; prints a
  "c vs m" table. Output `lam<lam>_ridge<ridge>[_<mode>][_interp].pt`.
- `turbulence/launch/resolve_lamsweep.sh`: env `MODE`, `DIM`, `RIDGE`, `LAMS`, `TIME_SOURCE`.

## 4. Tests (all local, all passed)

Run from the repo root.

| Script | What it shows | Result |
|---|---|---|
| `tau_sign_check.py` | E[φτ] vs ±Gθ̇ on a 1-D Gaussian | c = -Gθ̇ + m ∂ₜlog Z to 3 digits (the code assumed +Gθ̇) |
| `guth_weights_check.py` | closed forms of w_D, w_T, λ_eff | rel. error 1e-9 vs finite differences |
| `test_known_theta.py`, `_big.py` | error vs a known θ(t) | legacy: best 8% (λ=1e-2), 104% at λ=1, 139% at λ=100. fixed: 4.8% for λ = 1..100 (saturates, no bias) |
| `test_legacy_identical.py OLD.py` | legacy == old solver (`git show 35da82b:codes/sde_routines.py > OLD.py`) | bit-identical, 8/8 |
| `test_resolve_e2e.py OUTDIR` | resolve script, all modes, m from aux | fixed via aux == direct solve |
| `test_cache_identical.py OUTDIR` | new loop vs old recompute path, interp on/off, same RNG state | samples, θₜ, ηₜ, dH, moments, Theta_reg, M/G/b/c **bit-identical** |

Walkers vs interpolant (from `test_cache_identical.py`, B = 64, d = 64, **measured**):

| t | τ̄ walkers | cos(c, m) walkers | τ̄ interpolant |
|---|---|---|---|
| 0.19 | 323 | 1.000 | -0.9 |
| 0.58 | 77 | 1.000 | -2.2 |
| 0.84 | 21 | 0.97 | -5.7 |
| 0.98 | -33 | -0.88 | -38 |

→ on the walkers c is almost entirely the m·τ̄ part: the old time term was mostly an artefact.

Side finding: `gaussian_experiment/run_SDE.py` is not reproducible run to run even with
`--seed 0` (two identical runs differed; cause not investigated).

## 5. Before launching the full MGD batch (not done yet)

1. Push. **Disk:** system at nt = 60000, r = 281 is **57 GB/seed** (was 38), ~2.8 TB for
   50 seeds on SCRATCH. **Host RAM:** +19 GB held during the run; check the job's allocation
   (an OOM would only show at the end).
2. One seed to measure the step cost (was 1.09 s/step; the time guard aborts early if over):
   `ONLY="zfloor_two_phase_reg1e-4" SEEDS="905" bash zfloor_test.sh`
3. Then launch the batch, saving systems as before.

## 6. Offline tests on existing systems (old zfloor runs; walker terms only)

```bash
cd turbulence/launch
P="*zfloor_two_phase_reg1e-4*seed_900*.pt"
SYSTEM_PATTERN="$P" MODE=legacy RIDGE=1e-4 LAMS="0" bash resolve_lamsweep.sh   # should reproduce theta_t
SYSTEM_PATTERN="$P" MODE=fixed  RIDGE=1e-4 LAMS="0 1e-6 1e-5 1e-4 1e-3 1e-2" bash resolve_lamsweep.sh
SYSTEM_PATTERN="$P" MODE=guth   RIDGE=1e-4 DIM=256 LAMS="0 0.1 0.3 1 3 10" bash resolve_lamsweep.sh
```

Caveats: old systems lack τ̄, so `fixed`/`guth` on them cannot remove the m·τ̄ part
(check the "c vs m" table in the log); `DIM=256` inferred from `M256` with one channel,
confirm. On new runs use `TIME_SOURCE=interpolant`.

## 7. Open

- Choosing λ (or checking Guth's λ = 1) needs a new rule: the z² test is blind to
  blow-ups; add an amplitude veto / held-out loss.
- `--reg_mode` is not a `run_SDE.py` flag (in-run solve stays legacy; offline is the path).
- Bimodal experiment has the same τ sign (`sde_routines_scalar_reg.py:217`) and its own
  batched solver (`codes/lam_selection.py`): not fixed.
- Dead-potential masking in Theta_reg (item 6).
