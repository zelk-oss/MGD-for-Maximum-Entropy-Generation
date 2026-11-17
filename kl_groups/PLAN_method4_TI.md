# Method 4 — thermodynamic integration of Λ_G (PLAN ONLY, nothing implemented, 2026-11-14)

Exact identity: Λ_G(t) = log E_p[e^{t(A_G − E_p A_G)}], Λ_G'(t) = E_{p_t}[A_G] − E_p[A_G], p_t ∝ p e^{t A_G}
(= the model with θ_G scaled by (1−t)). Λ' is non-decreasing (Λ convex, Λ'' = Var_{p_t} A_G ≥ 0), Λ(0)=0, Λ(1)=D_G.
Grid 0 = t_0 < … < t_K = 1, ΔΛ'_i := Λ'(t_i) − Λ'(t_{i−1}):
  lower sum  Σ Λ'(t_{i−1}) (t_i − t_{i−1})  ≤  D_G  ≤  upper sum  Σ Λ'(t_i) (t_i − t_{i−1})   (EXACT for exact Λ').
Gap = Σ ΔΛ'(t_i−t_{i−1}) ≤ (max step) × (Λ'(1) − Λ'(0)). With estimated Λ' the brackets hold only up to MC error: add the sampling sd.
Caveat: if Λ'(t) → ∞ as t → t* < 1 (z_G ≥ 1, D = ∞) the upper sum is infinite and the lower sum grows without bound with the
grid: TI then only says "D = ∞"; the finite-t part Λ(t) for t < t* is still obtained.

## Validating the MALA code on d16 (codes/mala.py: Mala_Sampler(x_0, potentials, theta, n_steps, step_size, ...); unvalidated)
1. Sampler for p* at θ_trained: 10^4 chains, step size tuned to acceptance 0.5-0.6 (MALA optimum ~0.574); report acceptance, autocorrelation time.
2. Check against the exact Gaussian: sample covariance (circulant spectrum S, 16 values) vs S within sampling error; entropy of the
   fitted Gaussian vs H(p*) = 22.6743; Fourier modes |x̂_w|²/S_w vs Exp(1) / χ²_1 (KS test).
3. t = 0 consistency: E[φ] from MALA vs E[φ] from the SDE samples (E_gauss4_seed900.npz), per term, in units of the standard error.
4. Sign check: code has log p = +θ·φ, energy = −θ·φ; verify the gradient sign with a finite difference.
5. Tilted chains: p_t from θ_G·(1−t). Exact Λ'(t) of the bench (kl_groups/bench_d16_exact.py: derivative of lam_exact) is the reference
   at each t on a grid, for all 31 Scat2 subsets with z_G < 1/t (otherwise the chain is not normalisable: the tilted model does not exist).
## Cost
d16 is tiny (one signal of 16 points): 10^4 chains × 10^4 MALA steps × K = 20 grid points ≈ 2·10^9 potential evaluations of Scat2
(+ Scat4 if included: far more expensive) — minutes on one GPU, cannot be estimated on CPU here. Decorrelation: run 2-3 autocorrelation times
between recorded samples. Number of samples per t: Var_{p_t}(A_G)/(ε² ) with ε = target error on Λ' (e.g. 0.01 nat → ~10^3-10^4 effective samples).
Real data (d=256-1300): each MALA step costs a full forward+gradient of ALL potentials incl. Scat4; NOT estimated here.
## What I need from you
(a) OK to run the d16 validation as a JZ smoke/dev job (it needs the GPU potentials code); (b) which θ: trained θ of seed 900 (not stored in E npz — need
the .pt of the Lagrangian multipliers) or θ_star; (c) whether real-data TI is wanted at all given the cost.
