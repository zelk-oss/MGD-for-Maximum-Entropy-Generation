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


def welch_spectrum(x, nperseg):
    """Welch estimate of S (C, M), float64, on the length-M FFT grid, in the convention of average_spectrum
    (S(w) = E |FFT(x)(w)|^2). For ONE long series (S&P: M = 6062, n1 = 1) the raw periodogram of average_spectrum
    scatters like chi2_2 around the true spectrum; Welch (Hann segments of length nperseg, 50 % overlap, periodograms
    averaged) smooths it, as the MGD paper does for its covariance determinants (App. D.3, Welch 1967).
    PSD (two-sided, fs = 1) is evaluated on nperseg frequencies, interpolated linearly on |f| = min(k, M-k)/M
    (symmetric, so the IFFT of sample_gaussian_spectrum stays real) and multiplied by M."""
    import numpy as np
    from scipy.signal import welch
    xn = x.detach().cpu().double().numpy()
    M = xn.shape[-1]
    f, P = welch(xn, fs=1.0, window='hann', nperseg=nperseg, noverlap=nperseg // 2, detrend=False,
                 return_onesided=True, scaling='density', axis=-1)           # (B, C, nperseg//2+1), one-sided
    P = P.copy()
    P[..., 1:(nperseg + 1) // 2] /= 2.0                                      # one-sided -> two-sided density
    P = P.mean(0)                                                            # (C, nf)
    k = np.arange(M)
    fk = np.minimum(k, M - k) / M
    S = np.stack([np.interp(fk, f, Pc) for Pc in P]) * M
    return torch.from_numpy(S)


def replace_with_gaussian(x1, seed=0, nperseg=None):
    """(x1 Gaussian with the spectrum of x1 [same shape, dtype, device], S (C, M) float64 cpu).
    nperseg=None: raw mean periodogram over the batch; else Welch-smoothed (welch_spectrum)."""
    S = average_spectrum(x1).cpu() if nperseg is None else welch_spectrum(x1, nperseg)
    xg = sample_gaussian_spectrum(S, x1.shape[0], seed=seed)
    return xg.to(dtype=x1.dtype, device=x1.device), S
