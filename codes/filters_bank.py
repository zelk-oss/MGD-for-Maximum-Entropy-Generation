"""
MGD utilities - filter banks.
 
Build Morlet wavelet filter banks (1D and 2D) in Fourier space.
"""



import numpy as np
import torch

from filters.filters_1d import *
from filters.filters_2d import *




    ## ----------------------------------------------------- Filters related functions -----------------------------------------------------



def _as_filter(f_hat, device):
    """Fourier-domain filter -> float32 tensor, or complex64 if it has a phase (Daubechies).
    Real-valued filters (Morlet, Shannon, Meyer's zero imaginary part) stay float32 as before;
    the potentials' gradients apply conj(filter), which is a no-op on them."""
    f_hat = np.asarray(f_hat)
    if np.iscomplexobj(f_hat) and np.any(f_hat.imag != 0):
        return torch.tensor(f_hat).to(device).to(torch.complex64)
    return torch.tensor(f_hat.real if np.iscomplexobj(f_hat) else f_hat).to(device).to(torch.float32)


def return_Filters(M,J,Q=1,L=None,high_freq= 0.49,device='cpu',include_phi=False,wav_type='morlet'):
    """Build a Morlet wavelet filter bank in Fourier space.
 
    With ``L is None`` (1D): a band-pass set ``psi`` (``J`` scales, ``Q`` per octave)
    and a low-pass ``phi``, concatenated along the filter axis; if ``include_phi``,
    also a per-scale stack of low-passes. With ``L`` given (2D): an ``L*J`` oriented
    band-pass set plus a low-pass, from :class:`FiltersSet`.
 
    NOTE: in the 2D branch ``filters_phi`` is never defined, so ``include_phi=True``
    together with a non-None ``L`` raises ``NameError``.
 
    Parameters
    ----------
    M : int
        Signal length (1D) or side length (2D).
    J : int
        Number of scales.
    Q : int, optional
        Wavelets per octave (1D), by default 1.
    L : int or None, optional
        Number of orientations; None selects the 1D path.
    high_freq : float, optional
        Highest band-pass center frequency, by default 0.49.
    device : str, optional
        Torch device, by default ``'cpu'``.
    include_phi : bool, optional
        Also return the per-scale low-pass stack (1D only), by default False.
 
    Returns
    -------
    filters : torch.Tensor
        Filter bank in Fourier space; if ``include_phi``, returns
        ``(filters, filters_phi)``.
    """

    wav_norm = 'l1'
    high_freq = 0.49
    # 2D path (L is not None) still uses Morlet via FiltersSet; wav_type overrides
    # only the 1D path, which is the one used by run_SDE.py / run_jet_SDE.py.

    if L==None:
        psi = _as_filter(init_band_pass(wav_type, M, J, Q, high_freq, wav_norm), device)[None]
        phi = _as_filter(init_low_pass(wav_type, M, J, Q, high_freq), device)[None,None]
        if psi.is_complex() or phi.is_complex():
            psi, phi = psi.to(torch.complex64), phi.to(torch.complex64)
        filters =  torch.cat([psi,phi],dim=1)
        if include_phi is True:
            filters_phi = torch.stack([_as_filter(init_low_pass(wav_type, M, j, Q, high_freq), device) for j in range(1,J+1)])[None,:]
    else:
        filter_set = FiltersSet(M, M, J, L).generate_morlet(precision='single')
        filter_set_psi_real = torch.fft.fft2(torch.fft.ifft2(filter_set['psi']).real)
        filters = torch.cat((filter_set['psi'].reshape(1,L*J,M,M), torch.fft.fft2(filter_set['phi']).reshape(1,1,M,M)), 1).to(device)

    if include_phi is True:
        return filters,filters_phi
    return filters




def deduplicate_filters(filters, tol=1e-12):
    """
    Drop channels that are exact copies (up to a constant) of an earlier channel.

    filters: (1, C, M) Fourier-domain bank. Channel i is dropped when, in float64,
    1 - |<f_i, f_j>| / (|f_i| |f_j|) <= tol for some kept j < i, i.e. f_i = c f_j:
    its filtered signal is a constant multiple of channel j's, so any statistic
    built on it duplicates channel j's (exact collinearity, no information lost).
    This happens when a bank asks for more sub-octave wavelets than the frequency
    grid can resolve -- e.g. M=256, J=8, Q=3: band-pass filters 21, 22, 23 are the
    same single frequency bin 1 (deficit exactly 0). Filter 20 is kept: it also sits
    on bin 1 but has a 1.1e-4 component at bin 2 (deficit 6e-9), i.e. a little
    information of its own. tol=1e-12 keeps the removal strictly lossless; the
    comparison must be float64 (in float32, 1 - 1e-9 rounds to 1). tol=1e-8 also drops
    filter 21 (within 6e-9 of the kept filter 20): their statistics are near-duplicates
    (normalized Scalar_psi gradient matrix: cond 9.6e4 with both, 12 without; 2026-09-28).

    Returns (deduplicated filters, list of kept channel indices).
    """
    F = filters.reshape(-1, filters.shape[-1]).to(torch.complex128)
    norms = F.abs().pow(2).sum(-1).sqrt()
    keep = []
    for i in range(F.shape[0]):
        dup = norms[i] == 0        # zero-norm filters carry no information and
                                   # make Scalar_* potentials QR-blowup on fit;
                                   # happens e.g. with shannon at scales whose
                                   # xi +- sigma falls below 1/N (bands below the
                                   # Fourier resolution of the signal grid)
        for j in keep:
            if norms[i] > 0 and norms[j] > 0:
                deficit = 1 - float((F[i].conj() * F[j]).sum().abs() / (norms[i] * norms[j]))
                if deficit <= tol:
                    dup = True
                    break
        if not dup:
            keep.append(i)
    return filters[..., keep, :], keep
