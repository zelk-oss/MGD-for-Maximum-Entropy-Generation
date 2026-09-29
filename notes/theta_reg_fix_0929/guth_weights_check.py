# Map Guth et al. (2025, eq. 8) to the Cos interpolant X_t = cos(a) Z + sin(a) X, a = pi t/2.
# Y = X_t / sin(a) = X + cot(a) Z  -> Guth's noise variance s = cot(a)^2 (data at s -> 0 <-> t -> 1).
# Guth: E_{s ~ 1/s}[ (s/d) DSM(s) + (s/d)^2 TSM(s) ] = int ds/s [...]. In our t:
#   data weight  w_D(t) = |ds/dt| / d
#   time weight  w_T(t) = s / (d^2 |ds/dt|)     (TSM in t: (d/ds)^2 = (dt/ds)^2 (d/dt)^2)
# Claimed closed forms: w_D = 2 adot cos a / (d sin^3 a), w_T = sin a cos a / (2 adot d^2),
#   ratio w_T / w_D = sin(a)^4 / (pi^2 d).
import numpy as np
d, adot = 256.0, np.pi / 2
t = np.array([0.1, 0.5, 0.9, 0.99, 0.9999])
a = np.pi * t / 2
s = 1 / np.tan(a) ** 2
eps = 1e-7
sdot = (1 / np.tan(np.pi * (t + eps) / 2) ** 2 - 1 / np.tan(np.pi * (t - eps) / 2) ** 2) / (2 * eps)
wD_num, wT_num = np.abs(sdot) / d, s / (d ** 2 * np.abs(sdot))
wD, wT = 2 * adot * np.cos(a) / (d * np.sin(a) ** 3), np.sin(a) * np.cos(a) / (2 * adot * d ** 2)
print("rel err w_D:", np.abs(wD / wD_num - 1).max(), " rel err w_T:", np.abs(wT / wT_num - 1).max())
print("t      :", t)
print("lam_eff = w_T/w_D:", wT / wD, " sin^4/(pi^2 d):", np.sin(a) ** 4 / (np.pi ** 2 * d))
