"""Exact-Gaussian oracle test for codes/effective_dimension.py (denoiser_d_eff, energy_d_eff).

x ~ N(0, diag(sig2)) in d dims, Cos interpolant I_t = cos a Z + sin a x.  MGD's p_t is exactly
N(0, diag(v)), v_i(t) = cos^2 a + sin^2 a sig2_i, so in the code's (+) convention with phi_i = x_i^2:
    theta_i(t) = -1/(2 v_i),   m_i(t) = v_i,   log Z_t = 1/2 sum log(2 pi v_i).
Reference (c): closed form d_eff(x,tau) = sum_i (tau x_i^2 + sig2_i^2)/(sig2_i + tau)^2, cross-checked
against a brute-force numpy MC in a randomly ROTATED basis (exact Gaussian conditioning, no MGD code).
Grid: the latest turbulence run's two_phase schedule (nt 60000, n_bulk 10000, t_switch .9, gap 5e-5).
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path('/home/chiaraz/phd/MGD-for-Maximum-Entropy-Generation')
sys.path.insert(0, str(REPO / 'codes'))
from effective_dimension import (denoiser_d_eff, energy_d_eff, tau_of_t, s_of_t, alpha_of_t,
                                 h_bound_at, local_poly_derivative)
from time_schedules import two_phase_schedule
from utils_entropy import _phi

torch.set_num_threads(4)
rng = np.random.default_rng(0)

# ----------------------------------------------------------------------------- problem
d = 16
sig2 = np.logspace(-2, 1, d)                     # anisotropic: variances 1e-2 .. 10
n_x = 12
x_np = rng.standard_normal((n_x, d)) * np.sqrt(sig2)


class DiagQuad:
    """phi_i(x) = c_i x_{j_i}^2 (one coefficient per listed coordinate)."""
    def __init__(self, coords, scale=None):
        self.coords = list(coords)
        self.scale = np.ones(len(coords)) if scale is None else np.asarray(scale, float)
        self.num_coefficients = len(coords)

    def forward(self, x):
        xf = x.reshape(x.shape[0], -1)
        c = torch.as_tensor(self.scale, dtype=x.dtype, device=x.device)
        return xf[:, self.coords] ** 2 * c

    def grad(self, x, v):
        xf = x.reshape(x.shape[0], -1)
        out = torch.zeros_like(xf)
        c = torch.as_tensor(self.scale, dtype=x.dtype, device=x.device)
        v = v.to(dtype=x.dtype, device=x.device)
        out[:, self.coords] += 2 * (v * c) * xf[:, self.coords]   # index_add-free: coords unique
        return out.reshape(x.shape)


def exact(t):
    a = alpha_of_t(t)[:, None]
    v = np.cos(a) ** 2 + np.sin(a) ** 2 * sig2[None]
    return -0.5 / v, v, 0.5 * np.log(2 * np.pi * v).sum(1)          # theta, m, logZ


def oracle(x, tau):                                                 # (n_tau, n_x)
    tau = np.asarray(tau)[:, None, None]
    return ((tau * x[None] ** 2 + sig2 ** 2) / (sig2 + tau) ** 2).sum(-1)


# ----------------------------------------------------------------------------- (c) cross-check
Qr, _ = np.linalg.qr(rng.standard_normal((d, d)))
Sig = Qr @ np.diag(sig2) @ Qr.T
xr = x_np @ Qr.T
for tau in (1e-4, 1e-1, 1.0, 1e2):
    A = Sig @ np.linalg.inv(Sig + tau * np.eye(d))
    xi = rng.standard_normal((200_000, d))
    errs = []
    for k in range(3):
        y = xr[k] + np.sqrt(tau) * xi
        errs.append(((xr[k] - y @ A.T) ** 2).sum(1).mean() / tau)
    print(f'[c] tau={tau:7.0e}  brute-force rotated MC {np.round(errs, 3)}  closed form '
          f'{np.round(oracle(x_np[:3], [tau])[0], 3)}')

# ----------------------------------------------------------------------------- grid
t_fine = two_phase_schedule(60000, 10000, 0.9, 5e-5)[0].numpy()
t_dense = t_fine[2:]                                    # Theta_reg = Theta[1:] on t[1:] (nsub1)
s_dense = s_of_t(t_dense)
# evaluation nodes: N_EVAL nodes uniform in s over the dense grid's s range (what a sensible
# subsampler does), snapped to existing dense nodes
N_EVAL = 240
targets = np.linspace(s_dense.min(), s_dense.max(), N_EVAL)
idx = np.unique([int(np.argmin(np.abs(s_dense - q))) for q in targets])
t_reg = t_dense[idx]
tau = tau_of_t(t_reg)
th, m, logZ = exact(t_reg)
print(f'\ngrid: dense {len(t_dense)} nodes, tau in [{tau_of_t(t_dense[-1]):.2e}, {tau_of_t(t_dense[0]):.2e}];'
      f' eval {len(t_reg)} nodes uniform in s; ds max/min = {np.abs(np.diff(s_of_t(t_reg))).max() / np.abs(np.diff(s_of_t(t_reg))).min():.1f}')

pots = {'q': DiagQuad(range(d))}
X64 = torch.tensor(x_np).reshape(n_x, 1, d)
X32 = X64.float()
ref = oracle(x_np, tau)

def rel(a):
    return np.abs(a - ref) / ref

def summarize(name, a):
    r = rel(a)
    bands = [(1e2, 1e5), (1, 1e2), (1e-2, 1), (1e-4, 1e-2), (0, 1e-4)]
    cells = []
    for lo, hi in bands:
        mk = (tau >= lo) & (tau < hi)
        cells.append(f'{np.median(r[mk]):7.2%}/{r[mk].max():7.2%}' if mk.any() else '   -   ')
    print(f'{name:48s} ' + ' '.join(cells))

print('\nmedian/max relative error vs oracle, by tau band: '
      '[1e2,1e5) [1,1e2) [1e-2,1) [1e-4,1e-2) <1e-4')

g = torch.Generator().manual_seed(1)
TH = torch.tensor(th)
den64 = denoiser_d_eff(X64, TH, t_reg, pots, n_mc=512, generator=g)
den32 = denoiser_d_eff(X32, TH, t_reg, pots, n_mc=512, generator=g)
summarize('(a) denoiser, x float64, n_mc 512', den64)
summarize('(a) denoiser, x float32, n_mc 512', den32)

d_ = d
H_exact = logZ - (th * m).sum(1)                         # makes energy_d_eff's log Z exact
en_exact = {}
for nmc in (64, 512, 4096):
    en_exact[nmc] = energy_d_eff(X32, TH, t_reg, pots, torch.tensor(m), H_exact, d_, n_mc=nmc, generator=g)
    summarize(f'(b1) energy, EXACT logZ, n_mc {nmc}', en_exact[nmc])

# (b2) the pipeline log Z: H_bound from dH_k = -theta_k . mdot on the FINE grid, theta_k = exact
t1 = t_fine[1:]
th_f, m_f, _ = exact(t1)
a1 = alpha_of_t(t1)[:, None]
mdot_f = 2 * np.sin(a1) * np.cos(a1) * (sig2[None] - 1) * (np.pi / 2)
dH_exact = -(th_f * mdot_f).sum(1)
Hb = h_bound_at(t_fine, dH_exact, t_reg, d_)
print(f'\n(b2) |logZ_pipeline - logZ_exact| max over nodes = {np.abs(Hb + (th * m).sum(1) - logZ).max():.2e}')
en_b2 = energy_d_eff(X32, TH, t_reg, pots, torch.tensor(m), Hb, d_, n_mc=4096, generator=g)
summarize('(b2) energy, pipeline H_bound (theta_k=Theta), 4096', en_b2)

# (b3) theta_k != Theta_reg in dH (per-step theta noisier/shrunk vs the smoothed Theta_reg)
for lab, thk in [('theta_k = 0.9*Theta (ridge-like shrink)', 0.9 * th_f),
                 ('theta_k = Theta*(1+0.3*iid noise)', th_f * (1 + 0.3 * rng.standard_normal(th_f.shape)))]:
    Hb3 = h_bound_at(t_fine, -(thk * mdot_f).sum(1), t_reg, d_)
    e3 = energy_d_eff(X32, TH, t_reg, pots, torch.tensor(m), Hb3, d_, n_mc=4096, generator=g)
    summarize('(b3) ' + lab, e3)
    gap = e3 - en_b2
    print(f'      -> shift vs (b2): sd across x / |mean| = {np.median(gap.std(1) / (np.abs(gap.mean(1)) + 1e-12)):.2e}'
          f'  (x-independent if << 1);  max |mean shift| {np.abs(gap.mean(1)).max():.2f}')
    # predicted: d_eff shift = -d/ds (logZ_used - logZ_exact)
    pred = -local_poly_derivative(s_of_t(t_reg), Hb3 - Hb)
    print(f'      -> predicted -(d/ds) delta logZ vs observed mean shift: max abs diff '
          f'{np.abs(pred - gap.mean(1)).max():.3f}')

# (a') precision: near-copy pair with huge cancelling coefficients
print("\n(a') near-copy pair phi_a = x_0^2, phi_b = (1+1e-3) x_0^2, theta_a = +K, theta_b = -K/(1+1e-3)  (exact contribution 0)")
pots2 = {'q': DiagQuad(range(d)), 'pair': DiagQuad([0, 0][:1]), 'pair2': DiagQuad([0], scale=[1 + 1e-3])}
for K in (1e2, 1e4, 1e6, 1e8):
    TH2 = torch.cat([TH, torch.full((len(t_reg), 1), K, dtype=torch.float64),
                     torch.full((len(t_reg), 1), -K / (1 + 1e-3), dtype=torch.float64)], 1)
    m2 = torch.cat([torch.tensor(m), torch.tensor(m[:, :1]), torch.tensor(m[:, :1] * (1 + 1e-3))], 1)
    a32 = denoiser_d_eff(X32, TH2, t_reg, pots2, n_mc=256, generator=g)
    a64 = denoiser_d_eff(X64, TH2, t_reg, pots2, n_mc=256, generator=g)
    b32 = energy_d_eff(X32, TH2, t_reg, pots2, m2, H_exact - (TH2[:, d:] * m2[:, d:]).sum(1).numpy(), d_,
                       n_mc=256, generator=g)
    print(f'  K={K:.0e}: median rel err  denoiser f32 {np.median(rel(a32)):.2e}  f64 {np.median(rel(a64)):.2e}'
          f'   energy (x f32, theta f64) {np.median(rel(b32)):.2e}')

# common random numbers variant of the energy form (same xi at every node) -- PROPOSED, not in codebase
def energy_crn(x, theta, t, potentials, m_t, H_t, d, n_mc, generator, half_window=5):
    a, ta, s = alpha_of_t(t), tau_of_t(t), s_of_t(t)
    theta = theta.double()
    logZ = H_t + (theta * m_t.double()).sum(1).numpy()
    xi = torch.randn((n_mc,) + tuple(x.shape), generator=generator, dtype=x.dtype)
    U = np.zeros((len(t), x.shape[0]))
    for i in range(len(t)):
        sy = np.sin(a[i]) * (x.unsqueeze(0) + np.sqrt(ta[i]) * xi)
        ph = _phi(sy.reshape((-1,) + tuple(x.shape[1:])), potentials).double().reshape(n_mc, x.shape[0], -1)
        U[i] = -(ph @ theta[i]).mean(0).numpy() + logZ[i] - d * np.log(np.sin(a[i]))
    return d - local_poly_derivative(s, U, half_window=half_window)

print()
for nmc in (64, 512):
    summarize(f'(b1-CRN) energy, exact logZ, SAME xi all nodes, {nmc}',
              energy_crn(X32, TH, t_reg, pots, torch.tensor(m), H_exact, d_, nmc, g))

# MC-noise check for b1: does the error shrink ~ 1/sqrt(n_mc)?
for nmc in (64, 512, 4096):
    print(f'(b1) n_mc {nmc:5d}: RMS rel err {np.sqrt((rel(en_exact[nmc]) ** 2).mean()):.3e}')
