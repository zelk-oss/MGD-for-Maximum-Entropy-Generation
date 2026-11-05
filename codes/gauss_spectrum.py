"""Stationary Gaussian data with the power spectrum of a real data set (--gauss_spectrum).

Used by turbulence/run_SDE.py and jets/run_jet_SDE.py (gauss_spectrum/ experiment, 2026-11-03):
the real x1 (after windowing, coarse-graining, normalisation) is replaced by n1 draws of ONE
stationary Gaussian whose spectrum is the data spectrum averaged over the n1 signals,

    S(w) = mean_n |FFT(x1_n)(w)|^2 .

Construction: x = IFFT( FFT(w) * sqrt(S / M) ), w ~ N(0, I_M). This is a linear map of white
noise, so x ~ N(0, C) with C circulant, eigenvalues S(w) / M, and E |FFT(x)(w)|^2 = S(w) exactly.
S is symmetric (real signals), so the IFFT is real. Not per-signal phase randomisation: that keeps
each signal's own amplitudes, i.e. a mixture of Gaussians with different energies.

True log density (exact, no constant left out):
    log p(x) = -1/2 sum_w |FFT(x)(w)|^2 / S(w) - 1/2 sum_w log(2 pi S(w) / M).
"""
import math

import torch


def average_spectrum(x):
    """Mean power spectrum over the batch: (B, C, M) -> (C, M), float64."""
    xh = torch.fft.fft(x.to(torch.float64), dim=-1)
    return (xh.real ** 2 + xh.imag ** 2).mean(0)


def sample_gaussian_spectrum(S, n, seed=0):
    """n draws (n, C, M) of the stationary Gaussian with mean spectrum S (C, M), float64.

    Own generator (seed): the data are the same whatever the run's --seed (as for the real data).
    """
    C, M = S.shape
    if (S <= 0).any():
        raise ValueError('gauss_spectrum: S(w) has zero/negative entries, covariance would be singular')
    g = torch.Generator(device='cpu').manual_seed(seed)
    w = torch.randn(n, C, M, generator=g, dtype=torch.float64)
    x = torch.fft.ifft(torch.fft.fft(w, dim=-1) * torch.sqrt(S.cpu() / M), dim=-1)
    imag = x.imag.abs().max().item()
    if imag > 1e-8 * max(1.0, x.real.abs().max().item()):
        raise RuntimeError(f'gauss_spectrum: IFFT not real (max |imag| {imag:.2e}); S not symmetric?')
    return x.real


def true_log_p(x, S):
    """Exact log density of the gauss_spectrum law at x (B, C, M) -> (B,), float64."""
    M = S.shape[-1]
    xh = torch.fft.fft(x.to(torch.float64), dim=-1)
    S = S.to(xh.device)
    quad = ((xh.real ** 2 + xh.imag ** 2) / S).sum((-2, -1))
    return -0.5 * quad - 0.5 * torch.log(2 * math.pi * S / M).sum()


def replace_with_gaussian(x1, seed=0):
    """(x1 Gaussian with the spectrum of x1 [same shape, dtype, device], S (C, M) float64 cpu)."""
    S = average_spectrum(x1).cpu()
    xg = sample_gaussian_spectrum(S, x1.shape[0], seed=seed)
    return xg.to(dtype=x1.dtype, device=x1.device), S
