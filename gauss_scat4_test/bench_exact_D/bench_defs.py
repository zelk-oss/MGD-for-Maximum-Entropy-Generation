"""Definitions for the d16 Gaussian test bench (exact D_k vs estimator). Recreated 2026-11-14 (scratchpad had been cleaned).

Exact model p*: stationary Gaussian with spectrum S (E|xhat_w|^2 = S(w), codes/gauss_spectrum.py), exactly in the Scat2 family with
theta_star ((theta.psi2)(w) + (theta.psi2)(M-w) = -M^2 / S(w), checked to 1e-15).
Task sign: A_G(x) = -sum_{k in G} theta_k phi_k(x) = sum_w a_G(w) |xhat_w|^2,  a_G(w) = -sum_{k in G} theta_k psi2_k(w) / M^2.
Exact D_G = log E[e^{A - E A}] = sum over independent Fourier modes of
   0 < w < M/2 (complex mode, |xhat_w|^2 = S E, E ~ Exp(1)):   b = a(w) + a(M-w):   -log(1 - b S) - b S
   w = 0, M/2 (real mode, |xhat|^2 = S chi2_1):                 -1/2 log(1 - 2 a S) - a S
   = +inf if any 1 - b S <= 0 (q_G not normalisable)."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/chiaraz/phd/MGD-for-Maximum-Entropy-Generation')
sys.path.insert(0, str(ROOT / 'kl_groups'))
from kl_groups import H_estimate                       # noqa: E402,F401

b = torch.load(ROOT / 'gauss_scat4_test/runs/d16/results/spectrum.pt', map_location='cpu', weights_only=False)
S, th, psi2, M = b['S'][0].double().numpy(), b['theta_star'].double().numpy(), b['psi2'].double().numpy(), int(b['M'])
a_k = -th[:, None] * psi2 / M ** 2                      # (5, M): per-term weight of |xhat_w|^2 in A
CONF = 'gaussiansynth_H0.5_intermittency0.0_M16_J4_Q1_sigma3.5_nt50000_n1_10000_lam0.001_dseed0_thomas_nsub1_ridge0.0001_moment_f64_seed_900_termsc2834b6c_g4_M16'
xg = torch.load(ROOT / f'gauss_scat4_test/runs/d16/saved_results/samples/{CONF}.pt', map_location='cpu', weights_only=False)
xg = (xg if torch.is_tensor(xg) else torch.as_tensor(xg)).reshape(-1, M).double().numpy()


def exact_D(a):
    out = 0.0
    for w in range(M // 2 + 1):
        if w in (0, M // 2):
            z = 2 * a[w] * S[w]
            if z >= 1:
                return np.inf
            out += -0.5 * np.log1p(-z) - a[w] * S[w]
        else:
            z = (a[w] + a[M - w]) * S[w]
            if z >= 1:
                return np.inf
            out += -np.log1p(-z) - z
    return out


REAL_MODE = np.array([w in (0, M // 2) for w in range(M // 2 + 1)])


def mode_x(G):
    """x(w) = 1 - z(w) = (sum over the terms NOT in G of c_j(w)) / (sum over all terms of c_j(w)), computed WITHOUT cancellation (added 2026-11-16).
    c_j(w) = a_j(w) + a_j(M-w) (complex modes) or a_j(w) (real modes w = 0, M/2); sum_j c_j(w) = 1/S(w) (p* is in the Scat2 family).
    The direct 1 - z loses all digits when z -> 1: at w = 0 only the low-pass term acts (the others are ~1e-36), so 1 - z ~ 1e-35 < float64
    resolution and the old exact_D returned inf for the single term 4 (the exact value is 38.7 nats)."""
    c = np.array([[a_k[j, w] if REAL_MODE[w] else a_k[j, w] + a_k[j, M - w] for w in range(M // 2 + 1)] for j in range(a_k.shape[0])])
    free = [j for j in range(a_k.shape[0]) if j not in G]
    return c[free].sum(0) / c.sum(0) if free else np.zeros(M // 2 + 1)


def exact_D_stable(G):
    """exact D_G = KL(p* || q_G), sum over modes f (-ln x - (1 - x)), f = 1 (complex) or 1/2 (real); inf if some x <= 0."""
    x = mode_x(G)
    if np.any(x <= 0):
        return np.inf
    return float(np.sum(np.where(REAL_MODE, 0.5, 1.0) * (-np.log(x) - (1 - x))))


def kl_qp_stable(G):
    """exact reverse KL(q_G || p*): mode variance ratio r = 1/x; complex r - 1 - ln r, real half of it."""
    x = mode_x(G)
    if np.any(x <= 0):
        return np.inf
    r = 1 / x
    return float(np.sum(np.where(REAL_MODE, 0.5, 1.0) * (r - 1 - np.log(r))))


def draws(n, rng):
    """exact draws of p*, as codes/gauss_spectrum.sample_gaussian_spectrum (numpy)."""
    w = rng.standard_normal((n, M))
    return np.fft.ifft(np.fft.fft(w, axis=1) * np.sqrt(S / M), axis=1).real


def A_of(x, a):
    xh = np.fft.fft(x, axis=1)
    return (np.abs(xh) ** 2) @ a
