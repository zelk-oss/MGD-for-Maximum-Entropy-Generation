# Li et al. 2024: why small τ/τ_η is hard (2026-10-28)

## What the paper says
- The diffusion models reproduce everything "especially for τ ≳ τ_η". Below τ_η, DM-3c is "slightly smoother" than the DNS, in the acceleration PDF and in F^(p) as τ → 0.
- Table 2: the cross-correlation of the acceleration components, Σ_A, is about half the DNS value (2.4 vs 4.4 ×10⁻⁵).
- τ ≈ τ_η is where "most of the turbulent deadlocks develop" and where non-Gaussianity is strongest.
- The paper gives no explanation of why the model deviates there.

## Why it is hard (my reading, consistent with the paper)
1. **Extreme intermittency.** For τ ≲ τ_η the signal is smooth, so δ_τV ≈ a τ and the statistics are those of the acceleration. Its tails reach 60–100 σ. On our data the flatness of the one-step increment is 45 (Gaussian: 3). High moments (F^(8) ~ 10⁶) are dominated by very rare events, so the model has to give exactly the right probability to events it sees only a few times.
2. **The dissipative scale fluctuates.** The switch from inertial to dissipative happens at a local τ̃_η that depends on the local intensity: intense events become smooth at shorter times. That produces the dip in ζ(4,τ) around τ_η. It is a multi-scale effect, not a one-scale one. Multifractal models describe it, but nobody has a constructive way to generate it.
3. **Smoothing is the generator's typical error.** Missing the rare short events gives a smoother signal: thinner tails and lower F^(p) at small τ. That is the bias they see. In their DM the fine scales are built last during denoising (Fig 3), but the paper does not link this to the deviation.

For us, point 1 is the same issue as the fine tails of the Q = 3 marginals.

## Side note: can we compute τ_η from our data? (to think about)
τ_η = √(ν/ε). ν (viscosity) and ε (dissipation) are not in a single V(t). Two Lagrangian laws give

S₂(τ) = C₀ ε τ (inertial range),  ⟨a²⟩ = a₀ ε / τ_η  ⟹  τ_η = (a₀/C₀) · S₂(τ) / (τ ⟨a²⟩).

a₀ and C₀ depend on Re and are known to ~10–20 %, and at R_λ ≈ 310 the S₂ ∝ τ range is short. So the result is a check within a factor of ~1.5, less precise than the paper's τ_η = (2.1 ± 0.2)×10⁻², i.e. 9.33 samples (from the DNS ν and ε).
