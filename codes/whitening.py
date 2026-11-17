"""Whitened coordinates for MGD on real data (--whiten_alpha, whitening_test/, 2026-11-13).

Change of variables y = C^(-alpha/2) x, C = circulant covariance of the data with eigenvalues c(w) = S(w)/M
(S = mean_n |FFT(x_n)(w)|^2, codes/gauss_spectrum.average_spectrum). The model is NOT changed: every potential is
evaluated at x = C^(alpha/2) y,
    phi~_k(y) = phi_k(C^(alpha/2) y),
so the constraints, Theta, the max-ent law (as a law of x) and the statistics are the same; entropies shift by
H(x) = H(y) + (alpha/2) sum_w log c(w). Only the SDE coordinates change: Langevin in y is Langevin in x
preconditioned by C^alpha.

Implementation of the composition: a filter psi that acts DIRECTLY on x becomes psi_hat * c^(alpha/2)
(psi * x = psi~ * y). A filter acting on anything else (the envelope |psi_a * x|^2 in the fourth-order
scattering) is NOT touched. whiten_potentials applies this per class and refuses classes it was not checked for.
"""
import torch

from codes.gauss_spectrum import whitening_multiplier, apply_multiplier, entropy_shift  # noqa: F401

# class name -> attributes holding filters applied directly to x (Fourier multipliers, last axis = frequency)
_X_FILTERS = {
    'Scattering_Second_Order_1d': ('filters',),
    'L2p_norm': ('filters',),
    'L2p1_norm': ('filters',),
    'Scalar_GGD_KRegion': ('filters',),
    # two layers: filters_Q acts on x (first wavelet transform); filters acts on the envelope |filters_Q * x|^2
    'Scattering_Fourth_Order_Mod2_Real_1d': ('filters_Q',),
    'Scattering_Fourth_Order_Mod2_Imag_1d': ('filters_Q',),
}


def x_filter_multiplier(S, alpha):
    """c^(+alpha/2), (M,) float64: multiplies filters acting on x, and maps y back to x (x = C^(alpha/2) y)."""
    if S.shape[0] != 1:
        raise NotImplementedError('whitening: one-channel spectra only')
    return (1.0 / whitening_multiplier(S, alpha))[0]


def _times(f, m):
    m = m.to(f.device)
    return (f.to(torch.complex128) * m).to(f.dtype) if f.is_complex() else (f.double() * m).to(f.dtype)


def whiten_potentials(potentials, mult, log=print):
    """In place: multiply the x-acting filters of every potential by mult = c^(alpha/2) (x_filter_multiplier).
    Must run before the potentials are fitted or loaded (the loaded state takes the potential's own filters)."""
    for name, pot in potentials.items():
        cls = type(pot).__name__
        if cls not in _X_FILTERS:
            raise NotImplementedError(f'whitening: potential {name} ({cls}) not checked; add its x-acting filters '
                                      f'to codes/whitening._X_FILTERS after reading its forward/grad')
        if cls == 'Scalar_GGD_KRegion' and getattr(pot, 'dwt', None) is not None:
            raise NotImplementedError(f'whitening: {name} uses the decimated DWT (not a Fourier multiplier)')
        for attr in _X_FILTERS[cls]:
            setattr(pot, attr, _times(getattr(pot, attr), mult))
        if hasattr(pot, '_filters_Kx'):
            pot._filters_Kx = None                     # cached copy of the filters for the adjoint: rebuild
        log(f'[whitening] {name} ({cls}): x-acting filters {_X_FILTERS[cls]} multiplied by c^(alpha/2)')
    return potentials


def to_x(y, mult):
    """Samples back to x = C^(alpha/2) y (mult = x_filter_multiplier)."""
    return apply_multiplier(y, mult)
