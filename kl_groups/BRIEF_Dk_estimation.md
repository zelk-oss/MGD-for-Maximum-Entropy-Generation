# Estimating D_k = D_KL(p || q_k) for groups of potentials of a trained max-ent model, WITHOUT retraining

## Setting
- Trained maximum-entropy model of 1D signals x in R^d (Lagrangian turbulence velocity, jets, fBm, a Gaussian test):
  p(x) ∝ exp(-Σ_j θ_j·φ_j(x)), φ_j = wavelet-scattering statistics (orders 1-4, scalar "potential" banks) + a confinement
  term L_6 that is never removed. Fitted by MGD (an SDE whose drift is learned; θ = Θ_reg(t=1)).
- Group k of statistics (one family, a (j, j') pair of scales, or a single term). A_k(x) = θ_k·φ_k(x).
- q_k ∝ p·e^{A_k}: the same model with group k removed, other θ NOT refitted.
- Quantity: D_k = D_KL(p‖q_k) = log E_p[e^{A_k − E_p A_k}] = Λ(1) = Σ_{n≥2} κ_n/n!  (nats, total for the signal, not per dimension).
  It is an upper bound on I_k = S(q_k*) − S(p) (entropy gain with refit), which we do NOT want to compute: **no retraining**.

## Data available (per seed, 5 seeds; fBm 10 seeds)
- N = 10^4 generated signals x_i (fBm 1.5·10^4), from the SDE endpoint (law ρ_1 ≈ p; assumed ρ_1 = p).
- Cached per-sample, per-term energies E[i, k] = Θ_k φ_k(x_i) (code sign: log p = +Θ·φ, so A_k = −Σ_{k∈G} E[:, k]).
- θ known exactly; φ_k computable on any signal (GPU code); the trained model can be sampled at FIXED θ with a
  Langevin/MALA code (exists, not validated; sign convention to check). Θ_reg(t) on a time grid t ∈ [0, 1] is saved.

## Estimator used and what fails
D̂_k = log (1/N) Σ_i e^{a_i − ā}, a_i = A_k(x_i). Biased low (Jensen), consistent.
- Diagnostics: ESS = (Σ e^{a_i})² / Σ e^{2a_i}; drift D̂(N) − D̂(N/2) vs bootstrap std; ½Var(A_k);
  Gaussian floor σ√(2 ln N) − ln N (value of D̂ when one sample dominates; REM freezing picture).
- Reliability needs N ≫ e^{D_KL(q_k‖p)} (Chatterjee & Diaconis 2018); Gaussian A: e^{σ²/2}.
- Results (run1): for every FAMILY in the real datasets ESS ≈ 1 (one signal carries all the weight), so only
  D_k ≥ D̂_k is known; seed-to-seed spread ~ the mean. Examples (nats, mean over seeds):
  Lagrangian scalar banks D̂ ≈ 650, ½Var ≈ 5·10^3; Lagrangian Scat4 Real D̂ ≈ 50 (seeds 18-134), ½Var ≈ 11.5;
  Scat4 Imag D̂ ≈ 60 (6-140), ½Var ≈ 2.3 → A_k tails heavier than Gaussian.
  Many single terms / (j, j') pairs ARE well estimated (ESS ≥ 100).
- Validation case that works: Gaussian in-family test d = 16, true D_k(Scat4) = 0 + known finite-data value
  (10^-6 to 2·10^-3 nats); D̂_k matches the prediction (ratio 0.5-1.2), ESS ≈ N.

## Question
How can D_k (or a two-sided bracket L ≤ D_k ≤ U with useful width) be estimated for whole families, without
refitting any model?
1. Using ONLY the existing N samples and energies (e.g. tail modelling of A_k / extreme-value extrapolation of
   Λ(1), cumulant or saddle-point bounds, an upper bound to pair with the Jensen lower bound, Rao-Blackwellisation)?
2. If new sampling at FIXED θ is allowed: thermodynamic integration along θ_k → (1−t)θ_k,
   D_k = ∫_0^1 E_{p_t}[A_k − E_p A_k] dt, vs Bennett acceptance ratio / annealed importance sampling: how many
   intermediate t, how many samples per t, how to choose the schedule, how to get an error bar?
   Can the saved MGD path (Θ_reg(t), t ∈ [0, 1]) be reused as intermediate distributions?
3. Is D_k even the right quantity when it is dominated by tails with probability ~ e^{-D_KL(q_k‖p)} under p?
   Any better-behaved per-group measure of "how much this group shapes p" that needs no refit?
Please state assumptions and give formulas with exact statements (no approximations without saying so).

## Files to attach
- Code: kl_groups/kl_groups.py (estimator, ESS, drift), kl_groups/compute_H.py (groups), kl_groups/plots.py
  (flag rule, Gaussian floor), kl_groups/cache_energies.py (how E is built).
- Results tables: kl_groups/results/run1/H_lagrangian.csv, H_jets.csv, H_fbm.csv (one row per seed × group; column H = D̂_k),
  gauss_scat4_test/runs/d16/results/H_gauss4.csv + pred_H.csv + check.csv (validation case).
- Raw energies to try methods on (JZ, kl_groups/results/run1/, ~27 MB each): E_lagrangian_seed900.npz
  (N 10^4 × 341 terms; keys E, families, starts, widths, idx_<family>), and the small
  gauss_scat4_test/runs/d16/results/E_gauss4_seed900.npz (10^4 × 24, known answer).
- Figures: the notebook bar plot (one marker per seed, ½Var, Gaussian floor) and the D̂_k-vs-N diagnostic.
