"""Analytic check of kl_groups.H_estimate (numpy, seconds, CPU).

Task convention: p = N(0, 1/s), two potentials phi_1 = phi_2 = x^2/2, s = theta_1 + theta_2. Removing term 2:
    H_2 = -0.5 log(1 - r) - r/2,   r = theta_2 / s.
Code sign: C_2 = Theta_2 phi_2 with Theta = -theta (p ~ exp(+Theta . phi)), so C_2 = -theta_2 x^2 / 2.
The weights exp(theta_2 x^2/2) have finite variance only for r < 1/2: at r = 0.5 and 0.9 the flag is expected.
"""
import numpy as np
from kl_groups import H_estimate

N, s = 10_000, 1.0
for seed in (0, 1):
    x = np.random.default_rng(100 + seed).normal(0.0, 1 / np.sqrt(s), N)
    for r in (0.1, 0.5, 0.9):
        C2 = -(r * s) * x ** 2 / 2
        o = H_estimate(C2)
        exact = -0.5 * np.log(1 - r) - r / 2
        print(f'seed {seed} r={r}: H = {o["H"]:.4f} +- {o["boot_std"]:.4f}  exact {exact:.4f}  '
              f'0.5Var = {o["var2"]:.4f} (exact r^2/4 = {r * r / 4:.4f})  ESS/N {o["ess_frac"]:.3f}  '
              f'wmax {o["wmax"]:.4f}  drift {o["drift"]:+.4f}  FLAG={o["flag"]}')
