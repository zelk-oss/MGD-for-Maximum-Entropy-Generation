# In-family Gaussian test of H_k and of the entropy bound (2026-11-12)

**Question.** On Gaussian data, do the order-4 terms get theta ~ 0 and H_k = D_KL(p||q_k) ~ 0, and does the MGD entropy
bound match the exact entropy, with the CURRENT Theta_reg pipeline (saved system -> lam sweep -> cross-seed lam)?
The September fBm runs could not answer it: pre-09-29 Theta_reg, and white noise is not in the order-2 family.

**Why "in-family".** The Morlet Q = 1 order-2 bank does not tile the spectrum flat (gauss_dim_sweep: KL(white || best
of the family) = 0.24-0.35 nats). For data outside the family the max-ent fit legitimately uses theta(order 4) != 0.
So the data are drawn from p* = the max-ent Gaussian of the order-2 family with the order-2 moments of a source
spectrum (`make_spectrum.py`). Then, by uniqueness of max-ent, the model with Scat2 + L_6 + Scat4 has exactly
theta(Scat2) = theta_star, theta(L_6) = theta(Scat4) = 0, and H(p*) is exact. Finite n1 leaves a small noise floor
(roughly k / (2 n1) nats for k extra statistics, approximate).

**Steps.** (1) d = 16 (M 16, J 4: Scat4 10 Re + 4 Im), source white, 5 seeds, n1 1e4, nt 5e4 two_phase, sigma 3.5,
dev GPU (`launch.sh`, one task). (2) d = 256 (J 8), source = Lagrangian gauss_spectrum.pt projected on the family,
same terms, t3 queue (user launches; launcher to write).

**Outputs** (`runs/<TAG>/results/`): `spectrum.pt` (S*, theta_star, H_star), `xseed_<label>.pt`, `check.csv`
(entropy bound vs exact, theta Scat2 vs truth, 1/2 Var of the zero-truth terms), `H_gauss4.csv` (kl_groups).
`gaussian_experiment/run_SDE.py --spectrum_file` is the only change to an entry point (default off).
