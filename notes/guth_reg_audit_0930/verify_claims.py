"""Independent checks (2026-09-30) of the audit claims. Cos schedule, a = pi t / 2.
C1  E_I[tau] = 0 and E_I[phi tau] = -mdot on true pairs (score-function identity).
C2  walkers independent of Z: E_w[tau] = adot d cot a, Cov_w(phi, tau) ~ 0.
C3  walkers started AT Z (x(0) = Z) still break the identity (the 'pairing fix').
C4  Guth weights: DSM_Y = sin^2 a DSM_x, and lam_eff = sin^2 a / (pi^2 d).
"""
import numpy as np

rng = np.random.default_rng(0)
adot = np.pi / 2
A = lambda t: adot * t

def tau_fn(xt, Z, t):
    a = A(t); d = Z.shape[1]
    X = (xt - np.cos(a) * Z) / np.sin(a)
    return -adot * (np.tan(a) * (d - (Z ** 2).sum(1)) + (Z * X).sum(1))

# ---------- C1, C2: d-dim Gaussian data, phi = (|x|^2/d, x_1^2, x_1 x_2) ----------
d, B, sd = 16, 400_000, 0.5
phi = lambda x: np.stack([(x ** 2).mean(1), x[:, 0] ** 2, x[:, 0] * x[:, 1]], 1)
X = sd * rng.standard_normal((B, d)); X[:, 1] = 0.6 * X[:, 0] + 0.8 * X[:, 1]
Z = rng.standard_normal((B, d))
print('C1 (true pairs) and C2 (independent walkers), d=%d, B=%d' % (d, B))
for t in (0.2, 0.5, 0.8):
    a = A(t); I = np.cos(a) * Z + np.sin(a) * X
    tau = tau_fn(I, Z, t)
    Idot = adot * (-np.sin(a) * Z + np.cos(a) * X)
    # pathwise mdot = E[grad phi . Idot]
    gp = np.stack([2 * (I * Idot).mean(1), 2 * I[:, 0] * Idot[:, 0],
                   I[:, 0] * Idot[:, 1] + I[:, 1] * Idot[:, 0]], 1)
    mdot_path = gp.mean(0)
    c_I = (phi(I) * tau[:, None]).mean(0)
    se_tau = (phi(I) * tau[:, None]).std(0) / np.sqrt(B)
    se_path = gp.std(0) / np.sqrt(B)
    # walkers: same marginal law as I_t, but independent of Z
    W = np.cos(a) * rng.standard_normal((B, d)) + np.sin(a) * X[rng.permutation(B)]
    tw = tau_fn(W, Z, t)
    cov_w = ((phi(W) - phi(W).mean(0)) * (tw - tw.mean())[:, None]).mean(0)
    print(f' t={t}: E_I[tau]={tau.mean():+.3f}  -c_I={np.round(-c_I, 4)}  mdot(path)={np.round(mdot_path, 4)}'
          f'  s.e. ratio tau/path={np.round(se_tau / se_path, 1)}')
    print(f'        walkers: E_w[tau]={tw.mean():.3f} vs adot d cot a={adot * d / np.tan(a):.3f};'
          f' Cov_w(phi,tau)={np.round(cov_w, 4)}')

# ---------- C3: walkers started at Z, exact 1-D MGD dynamics with phi = x^2 ----------
# data N(0, sd^2); v(t) = cos^2 a + sd^2 sin^2 a; p_theta = N(0, v) (exp family, exact).
# MGD SDE (code sign): dx = (2 eta x + 2 sigma^2 th x) dt + sqrt(2) sigma dW, eta = vdot/(4v),
# th = -1/(2v) (code convention p ∝ exp(+th x^2)); keeps Law(x_t) = N(0, v(t)) exactly.
print('\nC3: 1-D walkers started AT Z (x(0)=Z), phi=x^2; compare E_w[phi tau] - E_w[tau] E_w[phi] with -mdot')
B1, sd1 = 400_000, 0.5
v = lambda t: np.cos(A(t)) ** 2 + sd1 ** 2 * np.sin(A(t)) ** 2
vdot = lambda t: adot * np.sin(2 * A(t)) * (sd1 ** 2 - 1)
for sigma in (0.0, 1.0, 3.0):
    Z1 = rng.standard_normal((B1, 1)); x = Z1.copy()
    nt = 4000; ts = np.linspace(0, 1, nt + 1)
    out = {}
    for k in range(nt):
        t, h = ts[k], ts[k + 1] - ts[k]
        drift = 2 * (vdot(t) / (4 * v(t))) * x + 2 * sigma ** 2 * (-1 / (2 * v(t))) * x
        x = x + h * drift + np.sqrt(2 * h) * sigma * rng.standard_normal(x.shape)
        tn = ts[k + 1]
        if any(abs(tn - s) < 1e-9 for s in (0.2, 0.5, 0.8)):
            tw = tau_fn(x, Z1, tn); ph = x[:, 0] ** 2
            cov = ((ph - ph.mean()) * (tw - tw.mean())).mean()
            out[round(tn, 2)] = (x[:, 0].var(), v(tn), -cov, vdot(tn), tw.mean())
    for tn, (vw, vt, mc, md, tm) in out.items():
        print(f' sigma={sigma}: t={tn}: Var_w={vw:.4f} (v={vt:.4f})  -Cov_w(phi,tau)={mc:+.4f}  '
              f'mdot={md:+.4f}  E_w[tau]={tm:+.3f}')

# ---------- C4: Guth weights in Cos time ----------
print('\nC4: Guth weights')
t = np.linspace(0.05, 0.95, 7); a = A(t); dd = 256
s = 1 / np.tan(a) ** 2
e = 1e-6                                                       # numerical ds/dt (central, fine step)
dsdt = (1 / np.tan(A(t + e)) ** 2 - 1 / np.tan(A(t - e)) ** 2) / (2 * e)
dsdt_ex = -2 * adot * np.cos(a) / np.sin(a) ** 3
# DSM_Y = sin^2 a DSM_x: check the per-sample identity on random (Z, model score)
Zs, g = rng.standard_normal(5), rng.standard_normal(5)           # g = grad_x (theta.phi) at x
aa = 0.7
lhs = np.sum((-np.sin(aa) * g - np.tan(aa) * Zs) ** 2)            # |grad_y U - (y-x)/s|^2, U = -theta.phi(sin a y)
rhs = np.sin(aa) ** 2 * np.sum((g + Zs / np.cos(aa)) ** 2)
print(f' ds/dt closed form rel err {np.abs(dsdt / dsdt_ex - 1).max():.1e}; DSM_Y/DSM_x/sin^2a = {lhs / rhs:.12f}')
# weights per unit t, with p(s) ∝ 1/s and |ds/dt|:
wD_x = (s / dd) * np.sin(a) ** 2 * (1 / s) * np.abs(dsdt_ex)      # multiplies DSM_x
wT = (s / dd) ** 2 * (1 / s) * np.abs(dsdt_ex) / dsdt_ex ** 2     # multiplies (d_t U)^2 terms
wD_code = 2 * adot * np.cos(a) / (dd * np.sin(a) ** 3)
print(' lam_eff = wT/wD_x * pi^2 d / sin^2 a :', np.round(wT / wD_x * np.pi ** 2 * dd / np.sin(a) ** 2, 10))
print(' code lam_eff * pi^2 d / sin^4 a      :', np.round(wT / wD_code * np.pi ** 2 * dd / np.sin(a) ** 4, 10))
