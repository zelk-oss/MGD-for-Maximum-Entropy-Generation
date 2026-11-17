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


def whitening_multiplier(S, alpha):
    """Fourier multiplier (S / M)^(-alpha/2), (C, M) float64 (--whiten_alpha, whitening_test/, 2026-11-13).

    S / M = eigenvalues of the data covariance in the ortho-FFT basis (S as in average_spectrum). MGD is run on
    y = IFFT(FFT(x) * (S/M)^(-alpha/2)) instead of x: alpha = 0 is the usual run, alpha = 1 makes the data white.
    Exact change of variables (linear, invertible, circulant):
      - potentials: every filter is multiplied by (S/M)^(+alpha/2) (whiten_filters), so phi(y; filters') = phi(x; filters)
        for any potential that applies its filters as a Fourier multiplier (Scattering_Second_Order_1d does);
      - the MGD reference noise y0 ~ N(0, I) is x0 ~ N(0, circ((S/M)^alpha)) in x;
      - entropies: H(x) = H(y) + entropy_shift(S, alpha) for every law (data, samples, p*, the bound); the
        max-ent problem is the same, so differences such as H(p*) - H(gen) or H(p*) - bound are unchanged."""
    M = S.shape[-1]
    return (S.double() / M) ** (-alpha / 2)


def entropy_shift(S, alpha):
    """H(x) - H(y) = log|det| of the map y -> x = (alpha/2) sum_w log(S(w)/M), nats (all channels)."""
    M = S.shape[-1]
    return float(alpha / 2 * torch.log(S.double() / M).sum())


def apply_multiplier(x, mult):
    """IFFT(FFT(x) * mult) along the last axis, real part, dtype/device of x. mult (C, M) or (M,), symmetric."""
    m = mult.to(x.device, torch.float64)
    y = torch.fft.ifft(torch.fft.fft(x.to(torch.float64), dim=-1) * m, dim=-1)
    imag = y.imag.abs().max().item()
    if imag > 1e-8 * max(1.0, y.real.abs().max().item()):
        raise RuntimeError(f'apply_multiplier: result not real (max |imag| {imag:.2e}); multiplier not symmetric?')
    return y.real.to(x.dtype)


def whiten_filters(filters, S, alpha):
    """filters (1, n, M) Fourier-domain bank -> filters * (S/M)^(+alpha/2) on the last axis, same dtype/device.
    One channel only (C = 1): the 1D runs."""
    if S.shape[0] != 1:
        raise NotImplementedError('whiten_filters: one-channel spectra only')
    m = (1.0 / whitening_multiplier(S, alpha))[0]                               # (M,), (S/M)^(+alpha/2)
    return (filters.to(torch.complex128) * m.to(filters.device)).to(filters.dtype) if filters.is_complex() \
        else (filters.double() * m.to(filters.device)).to(filters.dtype)


def replace_with_gaussian(x1, seed=0, nperseg=None, n=None):
    """(Gaussian draws with the spectrum of x1 [dtype, device of x1], S (C, M) float64 cpu).
    nperseg=None: raw mean periodogram over the batch; else Welch-smoothed (welch_spectrum).
    n: number of draws (default x1.shape[0]); may exceed the number of data signals used for S."""
    S = average_spectrum(x1).cpu() if nperseg is None else welch_spectrum(x1, nperseg)
    xg = sample_gaussian_spectrum(S, x1.shape[0] if n is None else n, seed=seed)
    return xg.to(dtype=x1.dtype, device=x1.device), S
