# 1-D Gaussian check of the time-score term c = E[phi tau] used in forward_regularised.
# Z ~ N(0,1) at t=0, X ~ N(0,s^2) at t=1, X_t = cos(a) Z + sin(a) X, a = pi t/2.
# p_t = N(0, v), v = cos^2 + s^2 sin^2.  phi = x^2 (no constant feature).
# MGD convention (dH = -theta . mdot): p_t ∝ exp(+theta phi) -> theta = -1/(2v).
import numpy as np
rng = np.random.default_rng(0)
s, N, adot = 0.2, 4_000_000, np.pi / 2
for t in [0.2, 0.5, 0.8, 0.95]:
    a = np.pi * t / 2
    Z, X = rng.standard_normal(N), s * rng.standard_normal(N)
    Xt = np.cos(a) * Z + np.sin(a) * X
    tau = -adot * (np.tan(a) * (1 - Z**2) + Z * X)          # exactly the code's tau (d = 1)
    phi = Xt**2
    c = (phi * tau).mean()
    v = np.cos(a)**2 + s**2 * np.sin(a)**2
    vdot = adot * 2 * np.sin(a) * np.cos(a) * (s**2 - 1)
    G = 3 * v**2                                            # E[phi^2]
    theta_dot = vdot / (2 * v**2)                           # d/dt of -1/(2v)
    dlogZ = vdot / (2 * v)                                  # normaliser: log Z = 0.5 log(2 pi v)
    print(f"t={t:4.2f}  c={c:+.4f}  | G*theta_dot={G*theta_dot:+.4f}  "
          f"-G*theta_dot={-G*theta_dot:+.4f}  -G*theta_dot + m*dlogZ={-G*theta_dot + v*dlogZ:+.4f}")
