import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from scipy import stats
import time 
import torch.nn.functional as F
import math
from scipy.integrate import quad
from scipy.optimize import minimize


from scipy.special import betaln
from scipy.optimize import minimize

import torch.nn as nn

from potentials.utils_potentials import *



    ## ----------------------------------------------------- Mother classes -----------------------------------------------------    



class Potential_Prepare(nn.Module):
    def __init__(self,potential):
        super().__init__()
        self.potential= potential
    def forward(self,x,v=None,argument = 'forward'):
        if argument == 'forward':
            return self.potential(x)
        elif argument == 'grad':
            return self.potential.grad(x,v)
        elif argument == 'fit':
            self.potential.fit(x)
        else:
            pass


class Potential_Parallel(nn.Module):
    def __init__(self,potential):
        super().__init__()
        self.potential = nn.DataParallel(Potential_Prepare(potential))
        #self.grad = potential.grad
    def forward(self,x):
        return self.potential(x,argument='forward')
    def grad(self,x,v=None):
        n_gpu = torch.cuda.device_count()
        if v is not None and n_gpu != 0:
            v = v.repeat((n_gpu,))
        return self.potential(x,v,argument='grad')

class Potential(nn.Module):
  
    def __init__(self):
        self.num_potentials = None
        super().__init__()
            
    def forward(self,x):
        pass
        
    def grad(self,x):
        pass
        
    def fit(self,x):
        pass
        
    def fit_micro(self,x):
        pass



    ## ----------------------------------------------------- Fast exact Scat3 / Scat4 (2026-11-23) -----------------------------------------------------
# Same quantities as the original code in exact arithmetic, fewer FFTs and passes (notes/scat4_fast_report_1122.md,
# notes/fast_families_report_1122.md; long-run equivalence M=64/128, 2026-11-23). x is real, so a real part of an
# inverse FFT is the irfft of the Hermitian half spectrum, and mean_t Re/Im[...] of two filtered signals is a sum over
# frequencies (Parseval). Constants are cached per (device, M, dtype) and rebuilt if filters / indices are replaced.

_fft, _ifft, _rfft, _irfft = torch.fft.fft, torch.fft.ifft, torch.fft.rfft, torch.fft.irfft


def _cdtype(dt):
    return torch.complex128 if dt == torch.float64 else torch.complex64


def _as_dtype(f, dt):
    return f.to(_cdtype(dt)) if f.is_complex() else f.to(dt)


def _cached(p, x, refs, build):
    key = (x.device, x.shape[-1], x.dtype)
    c = p.__dict__.get('_fast_cache')
    if c is not None and c['key'] == key and all(r is s for r, s in zip(c['refs'], refs)):
        return c
    c = build(x.device, x.shape[-1], x.dtype)
    c['key'], c['refs'] = key, refs
    p.__dict__['_fast_cache'] = c
    return c


def _hermitian_half(Z, neg_h, h):
    """First M/2+1 bins of (Z(k) + conj Z(-k)) / 2: its irfft is Re ifft(Z)."""
    return 0.5 * (Z[..., :h] + Z[..., neg_h].conj())


def _scat3_cache(p, x, imag, indices):
    def build(dev, M, dt):
        f = _as_dtype(p.filters.to(dev), dt)
        psi2 = (f * f.conj()).real if f.is_complex() else f * f                       # |psi_k|^2, (1, nf, M)
        psi2n = psi2[..., (-torch.arange(M, device=dev)) % M]                           # |psi_k(-w)|^2
        h = M // 2 + 1
        idx = indices.long().to(dev)
        return dict(M=M, f=f, fc=f.conj(), psi2=psi2, psi2n=psi2n, j=idx[0], k=idx[1],
                    w=((-0.5j * (psi2 - psi2n))[..., :h].to(_cdtype(dt)) if imag else (0.5 * (psi2 + psi2n))[..., :h]),
                    w_parseval=psi2 / M ** 2)
    return _cached(p, x, (p.filters,), build)


def _scat3_forward(p, x, imag, indices):
    """mean_t Re/Im[conj(W_l x) (psi_l * |W_i x|)] = (1/M^2) sum_k |psi_l|^2 conj(X) A^_i (Parseval)."""
    c = _scat3_cache(p, x, imag, indices)
    X = _fft(x)
    Ah = _fft(abs_eps(_ifft(c['f'] * X)))                                              # (B, nf, M)
    S = torch.bmm(Ah, (c['w_parseval'] * X.conj()).transpose(1, 2))                    # [i, l]
    out = S[:, c['j'], c['k']]
    return out.imag if imag else out.real


def _scat3_weight(c, x, imag):
    """Re (Real) or -Im (Imag) of |psi_k|^2 * x, (B, nf, M), by irfft."""
    W = _irfft(_rfft(x) * c['w'], n=c['M'])
    return -W if imag else W


def _scat3_jac(p, x, imag, indices):
    """Phase and modulus terms added in Fourier space, one inverse FFT on the K rows.
    Imag: Re ifft(Zp) - Im ifft(Zm) = Re ifft(Zp + i Zm)."""
    c = _scat3_cache(p, x, imag, indices)
    j, k = c['j'], c['k']
    Wx = _ifft(c['f'] * _fft(x))
    A = abs_eps(Wx)
    W = _scat3_weight(c, x, imag)
    Zp = c['fc'][:, j] * _fft((Wx / A)[:, j] * W[:, k])
    Zm = (c['psi2n'] if imag else c['psi2'])[:, k] * _fft(A)[:, j]
    return _ifft(Zp + 1j * Zm if imag else Zp + Zm).real / c['M']


def _scat3_grad_v(p, x, v, imag, indices):
    """sum_i v_i grad phi_i, all channel sums in Fourier space, one inverse FFT."""
    c = _scat3_cache(p, x, imag, indices)
    nf = c['f'].shape[1]
    m = torch.zeros((nf - 1, nf), device=x.device, dtype=x.dtype)
    m[c['j'], c['k']] = v.to(x.dtype)
    Wx = _ifft(c['f'] * _fft(x))
    A = abs_eps(Wx)
    W = _scat3_weight(c, x, imag)
    rcp = (Wx[:, :-1] / A[:, :-1]) * torch.einsum('ij, aib -> ajb', m.T, W)
    Zp = (c['fc'][:, :-1] * _fft(rcp)).sum(1, keepdim=True)
    aw = torch.einsum('ij, aib -> ajb', m, A[:, :-1])
    Zm = ((c['psi2n'] if imag else c['psi2']) * _fft(aw)).sum(1, keepdim=True)
    return _ifft(Zp + 1j * Zm if imag else Zp + Zm).real / c['M']


def _scat4_cache(p, x, imag):
    def build(dev, M, dt):
        f = _as_dtype(p.filters.to(dev), dt)
        fQ = _as_dtype(p.filters_Q.to(dev), dt)
        a, b, cc = p.indices.long().to(dev)
        K, h = a.shape[0], M // 2 + 1
        aa, bb, cc2 = torch.cat([a, b]), torch.cat([b, a]), torch.cat([cc, cc])
        psi2 = (f * f.conj()).real if f.is_complex() else f * f
        neg = (-torch.arange(M, device=dev)) % M
        psi2n = psi2[..., neg]
        ni = p.norm_indices.to(dev).to(dt)
        # weight of the second layer on the half spectrum: Re P (Real) or Im P (Imag), P = ifft(|psi_c|^2 fft(E))
        w = (-0.5j * (psi2 - psi2n))[..., :h].to(_cdtype(dt)) if imag else (0.5 * (psi2 + psi2n))[..., :h]
        return dict(M=M, K=K, h=h, f=f, fQ=fQ, a=a, b=b, c=cc, aa=aa, bb=bb, cc2=cc2, w=w, neg_h=neg[:h],
                    w_parseval=psi2 / M ** 2, fQc_aa=fQ[:, aa].conj(), ni=ni, scale2=2.0 / (M * ni))
    return _cached(p, x, (p.filters, p.filters_Q, p.indices, p.norm_indices), build)


def _scat4_first_layer(c, x):
    Wx = _ifft(c['fQ'] * _fft(x))
    return Wx, abs_eps(Wx) ** 2


def _scat4_weight(c, E):
    return _irfft(_rfft(E)[:, :, None] * c['w'], n=c['M'])                             # (B, nQ, nf, M) real


def _scat4_forward(p, x, imag):
    """mean_t Re/Im[(psi_c*E_i1) conj(psi_c*E_i0)] = (1/M^2) sum_k |psi_c(k)|^2 E^_i1 conj(E^_i0) (Parseval)."""
    c = _scat4_cache(p, x, imag)
    Eh = _fft(abs_eps(_ifft(c['fQ'] * _fft(x)) / p.norm) ** 2)                         # (B, nQ, M)
    B, nQ, M = Eh.shape
    nf = c['w_parseval'].shape[1]
    X = (c['w_parseval'][:, :, None, :] * Eh[:, None]).reshape(B, nf * nQ, M)
    S = torch.bmm(X, Eh.conj().transpose(1, 2)).reshape(B, nf, nQ, nQ)
    i0, i1, i2 = p.indices.long().to(x.device)
    out = S[:, i2, i1, i0]
    return out.imag if imag else out.real


def _scat4_jac(p, x, imag):
    """Real: partner rows (a,b,c) + (b,a,c) summed in Fourier space, one ifft on K rows.
    Imag (real filter bank): Im[W*(Wx e) + conj W*(Wx conj e)] = 2 Re W*(Wx Im e) and P~ = conj P, so the same
    one-branch algorithm as Real with weight Im P and partner sign -1."""
    c = _scat4_cache(p, x, imag)
    K = c['K']
    Wx, E = _scat4_first_layer(c, x)
    W = _scat4_weight(c, E)
    T = Wx[:, c['aa']] * W[:, c['bb'], c['cc2']]                                       # (B, 2K, M); factor 2 in scale2
    Z = c['fQc_aa'] * _fft(T)
    if imag:
        t = _ifft(Z).real
        return (t[:, :K] - t[:, K:]) * c['scale2'][:, None]
    return _ifft((Z[:, :K] + Z[:, K:]) * c['scale2'][:, None]).real


def _scat4_grad_v(p, x, v, imag):
    """sum_k v_k grad phi_k (real filter bank): h_i = sum_jk m~[i,j,k] W_jk, m~ = m + m^T (Real) or m - m^T (Imag),
    one product with W x, filter, sum over i in Fourier space, one irfft. Shape (B, 1, M)."""
    c = _scat4_cache(p, x, imag)
    nQ, nf = c['fQ'].shape[1], c['f'].shape[1]
    m = torch.zeros((nQ, nQ, nf), device=x.device, dtype=x.dtype)
    m[c['a'], c['b'], c['c']] = v.to(x.dtype) / c['ni']
    m = m - m.transpose(0, 1) if imag else m + m.transpose(0, 1)
    Wx, E = _scat4_first_layer(c, x)
    hW = torch.einsum('ijk,ajkb->aib', m, _scat4_weight(c, E))                         # (B, nQ, M)
    Y = (c['fQ'].conj() * _fft(Wx * hW)).sum(1, keepdim=True) * (2.0 / c['M'])
    return _irfft(_hermitian_half(Y, c['neg_h'], c['h']), n=c['M'])


    ## ----------------------------------------------------- Potentials definitions -----------------------------------------------------


    # ----- Scattering potentials -----


class Scattering_First_Order_1d(Potential):
    def __init__(self,filters):
        super().__init__()
        self.filters = filters
        self.num_coefficients = filters.shape[1]
        # EXPERIMENTAL (chat 2026-08-31): was unnormalized (self.norm not
        # defined, fit_micro a no-op). Testing paper's S1(x)[λ]/σ[λ]
        # normalization (Bruna & Mallat scattering-spectra eq. 16), with
        # σ[λ] fit below. Revert by deleting this line and fit_micro/the
        # /self.norm divisions in forward()/grad() if it doesn't pan out.
        self.norm = 1

    def forward(self,x):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft(filters*torch.fft.fft(x))
        x_filtered = x_filtered/self.norm  # EXPERIMENTAL normalization, see __init__ note
        return abs_eps(x_filtered).mean(-1)

    def grad(self, x, v=None, means=None):
        filters = self.filters.to(x.device)
        x_fourier = torch.fft.fft(x)
        x_filtered = torch.fft.ifft(filters*x_fourier)
        x_filtered_abs = abs_eps(x_filtered)
        x_filtered_over_abs = x_filtered/x_filtered_abs

        output = torch.real(torch.fft.ifft(torch.fft.fft(x_filtered_over_abs)*filters.conj()))
        # EXPERIMENTAL: self.norm doesn't depend on x, so dividing the final
        # (raw) gradient by it here is equivalent to normalizing the
        # coefficient before forward()'s abs() -- see __init__ note.
        output = output/self.norm

        if v==None:
            return output/x.shape[-1]
        else:
            return (output*v[None,:,None]).sum(1)[:,None]/x.shape[-1]

    def fit_micro(self,x):
        """
        EXPERIMENTAL (chat 2026-08-31, being tried out -- not yet settled):
        fits self.norm = σ[λ] = sqrt(spectrum) = sqrt(E[|Wx_s|^2]) per scale,
        averaged over batch AND time -- i.e. σ²[λ] = Ave_i S2(x_i)[λ] in the
        paper's notation (eq. 16/17). forward()/grad() divide by this. If
        this normalization is reverted, delete this method along with the
        self.norm bits in __init__/forward/grad above.
        """
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft(filters*torch.fft.fft(x))
        self.norm = (x_filtered.abs()**2).mean((0,2))[:,None]**0.5

class Scattering_Second_Order_1d(Potential):
    def __init__(self,filters):
        super().__init__()
        self.filters = filters
        self.num_coefficients = filters.shape[1]
        # EXPERIMENTAL (chat 2026-08-31): same status as
        # Scattering_First_Order_1d.__init__'s note -- was unnormalized.
        self.norm = 1

    def forward(self, x):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft(filters*torch.fft.fft(x))
        x_filtered = x_filtered/self.norm  # EXPERIMENTAL normalization, see __init__ note
        return (x_filtered*x_filtered.conj()).real.mean(-1)

    def grad(self, x, v=None,  means=None):

        filters = self.filters.to(x.device)
        if v is not None:
            # 2026-11-17: sum_j v_j 2 Re ifft(fft(x) |psi_j|^2) / norm_j^2 = 2 Re ifft(fft(x) sum_j (v_j / norm_j^2) |psi_j|^2):
            # one FFT pair instead of one per channel; same quantity up to round-off
            w = v[:, None] / self.norm**2                                       # (n, 1)
            H = ((filters * filters.conj()) * w[None]).sum(1, keepdim=True)     # (1, 1, T)
            return 2 * torch.fft.ifft(torch.fft.fft(x) * H).real / x.shape[-1]
        x_filtered_2 = torch.fft.ifft(torch.fft.fft(x)*(filters*filters.conj()))   # |psi_hat|^2: adjoint of a complex (Daubechies) filter is its conj

        output = x_filtered_2.real.reshape(x_filtered_2.shape[:1]+(-1,x.shape[-1]))
        # EXPERIMENTAL: raw statistic is quadratic in the coefficient, so the
        # normalization enters squared -- matches forward()'s
        # x_filtered/self.norm applied before the product. See __init__ note.
        output = output/self.norm**2

        if v==None:
            return 2*output/x.shape[-1]
        else:
            return (2*output*v[None,:,None]).sum(1)[:,None]/x.shape[-1]

    def fit_micro(self,x):
        # EXPERIMENTAL (chat 2026-08-31): same σ[λ] = sqrt(spectrum)
        # convention as Scattering_First_Order_1d.fit_micro (see its
        # docstring) -- fit independently here since First/Second order are
        # separate Potential instances, each against its own filters bank.
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft(filters*torch.fft.fft(x))
        self.norm = (x_filtered.abs()**2).mean((0,2))[:,None]**0.5

# experiment for lagrangian turbulence 
class Scattering_Second_Order_Bulk_1d(Potential):
    def __init__(self, filters, bulk_quantile=0.90, trans_quantiles=(0.85, 0.95)):
        super().__init__()
        self.filters = filters
        self.num_coefficients = filters.shape[1]
        self.bulk_quantile = bulk_quantile
        self.trans_quantiles = trans_quantiles

        # will be set in fit()
        self.c = None
        self.s = None

    def fit(self, x):
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real  # (B, J, T)
        az = torch.sqrt(z**2 + 1e-8)

        c_list = []
        s_list = []

        for j in range(az.shape[1]):
            ah = az[:, j, :].reshape(-1)

            c = torch.quantile(ah, self.bulk_quantile)

            q_lo, q_hi = self.trans_quantiles
            band = torch.quantile(ah, q_hi) - torch.quantile(ah, q_lo)

            s = band / 4
            s = torch.clamp(s, min=1e-6)

            c_list.append(c)
            s_list.append(s)

        self.c = torch.stack(c_list).to(x.device)  # (J,)
        self.s = torch.stack(s_list).to(x.device)  # (J,)

    def _bulk_window(self, z):
        az = torch.sqrt(z**2 + 1e-8)
        c = self.c[None, :, None]
        s = self.s[None, :, None]

        w_tail = torch.sigmoid((az - c) / s)
        return 1 - w_tail

    def forward(self, x):
        filters = self.filters.to(x.device)

        x_filtered = torch.fft.ifft(filters * torch.fft.fft(x))  # (B,J,T)

        w_bulk = self._bulk_window(x_filtered.real)

        energy = (x_filtered * x_filtered.conj()).real

        return (w_bulk * energy).mean(-1)

    def grad(self, x, v=None, means=None):
        filters = self.filters.to(x.device)

        x_filtered = torch.fft.ifft(filters * torch.fft.fft(x))
        z = x_filtered.real

        w_bulk = self._bulk_window(z)

        # derivative of energy = 2 * x_filtered
        weighted = w_bulk * x_filtered

        output = torch.real(
            torch.fft.ifft(torch.fft.fft(2 * weighted) * filters.conj())
        )

        if v is None:
            return output / x.shape[-1]
        else:
            return (output * v[None, :, None]).sum(1)[:, None] / x.shape[-1]
        
def _check_scat3_bank(J, filters, cls):
    """Scat3 indexing (indices_third_order(J, 1)) assumes a Q = 1 bank: J band-passes + the low-pass last.
    A Q > 1 bank (J*Q + 1 filters) used to be accepted silently and gave the J*(J+3)/2 pairs of its finest
    filters, without the low-pass (bug found 2026-11-16): refuse it."""
    if filters.shape[-2] != J + 1:
        raise ValueError(f'{cls}: needs a Q = 1 bank of J + 1 = {J + 1} filters (low-pass last), got '
                         f'{filters.shape[-2]}; use the Scattering_Third_Order_*_Q1 terms (Q = 1 bank `filters`)')


class Scattering_Third_Order_Real_1d(Potential):
    def __init__(self, J, filters):
        super().__init__()
        _check_scat3_bank(J, filters, 'Scattering_Third_Order_Real_1d')
        self.J = J
        self.filters = filters
        self.num_coefficients = int(J*((1+J)/2+1))

    def _indices(self):
        return indices_third_order(self.J, 1)

    def forward(self, x):
        return _scat3_forward(self, x, False, self._indices())

    def grad(self, x,  v=None,  means=None):
        # phi_jk = mean_t |W_j x| Re(h_k * x), h_k^ = |psi_k^|^2; grad phi_jk = W_j^*((W_j x/|W_j x|) Re(h_k * x)) / M
        # + h_k * |W_j x| / M (phase and modulus terms), evaluated on the K kept pairs only
        if v is not None:
            return _scat3_grad_v(self, x, v, False, self._indices())
        return _scat3_jac(self, x, False, self._indices())


class Scattering_Third_Order_Imag_1d(Potential):
    """Imaginary counterpart of Scattering_Third_Order_Real_1d: Im[conj(W_j x) (psi_j * |W_k x|)].

    Same indices and layout as the Real class. Like the Real one it is odd under x -> -x, so its
    expectation vanishes for sign-symmetric data. Gradient: with g_j = ifft(fft(x) |psi_j^|^2),
    Phi_kj = -mean(|W_k x| Im g_j); the Real class has +Re g_j, and the modulus term uses
    the frequency-reversed |psi_j^|^2 because the Imag part is not symmetric in omega.

    include_lowpass=False (2026-11-05, S&P): drop the J columns whose phase filter j is the low-pass (j = J). Their
    Imag part is identically zero (real, symmetric low-pass output), so their gradient is rounding noise and their
    Theta is set by the ridge alone (~1e9 at ridge 1e-7); the MGD paper (App. D.1) discards such potentials.
    Default True: unchanged (bit-identical).
    """
    def __init__(self, J, filters, include_lowpass=True):
        super().__init__()
        _check_scat3_bank(J, filters, 'Scattering_Third_Order_Imag_1d')
        self.J = J
        self.filters = filters
        self.include_lowpass = include_lowpass
        self.num_coefficients = int(J*((1+J)/2+1)) - (0 if include_lowpass else J)

    def _indices(self):
        indices = indices_third_order(self.J, 1)
        if not self.include_lowpass:
            indices = indices[:, indices[1] < self.J]
        return indices

    def forward(self, x):
        return _scat3_forward(self, x, True, self._indices())

    def grad(self, x,  v=None,  means=None):
        # as the Real class with -Im(h_k * x) in the phase term and |psi_k^|^2 at -omega in the modulus term
        if v is not None:
            return _scat3_grad_v(self, x, v, True, self._indices())
        return _scat3_jac(self, x, True, self._indices())


class Scattering_Fourth_Order_Real_1d(Potential):
    def __init__(self, J,Q,filters,filters_Q,include_diag = False,lite=True):
        super().__init__()
        self.J = J
        self.Q = Q
        self.filters = filters       # coarse, "second-layer" filter bank: J(+1) channels (one per octave, plus low-pass)
        self.filters_Q = filters_Q   # fine, "first-layer" filter bank: J*Q channels (Q wavelets per octave)
        if include_diag is True:
            offset = 0
        else:
            offset = 1
        # Which (s1, s2, j) coefficient triples this potential keeps: s1,s2 are
        # fine scales in [0,J*Q), j is the coarse scale in [0,J] the s1/s2
        # correlation is measured at. Only s1<=s2 is stored -- Re[z] is
        # symmetric under swapping s1<->s2, so the s1>s2 half is redundant.
        # See indices_fourth_order_Q in utils_potentials.py.
        self.indices = indices_fourth_order_Q(self.J, self.Q,offset,lite)
        self.norm = 1                 # per-fine-scale rescaling, fit once by fit_micro() below
        self.norm_indices = torch.ones((len(self.indices[0]),))  # self.norm, reshaped to match self.indices' flat coefficient layout (used by grad())

        self.num_coefficients = len(self.indices[0])

    def forward(self, x):
        filters = self.filters.to(x.device)
        filters_Q = self.filters_Q.to(x.device)
        # First wavelet transform: one complex coefficient Wx_s(t) per fine
        # scale s and time step.
        x_filtered = torch.fft.ifft(filters_Q*torch.fft.fft(x)) #(B,JQ,T)

        #Normalize micro -- divide by the per-scale factor fit_micro() computed
        # from real data, so scales with very different raw power (typical of
        # a power-law spectrum) don't dominate the statistic below just
        # because they're louder. See fit_micro()'s docstring for the "why".
        x_filtered = x_filtered/self.norm

        # Envelope used by this potential: the MODULUS |Wx_s(t)| (order 1,
        # NOT squared -- that's what distinguishes this "plain" class from
        # its Mod2 sibling, which uses |Wx_s(t)|**2 instead).
        x_filtered_abs = abs_eps(x_filtered) #(B,JQ,T)

        # Second wavelet transform of that envelope: how the envelope at fine
        # scale s fluctuates in time, as seen at coarser scale j.
        W_Wx = torch.fft.ifft(filters[:, :, None] * torch.fft.fft(x_filtered_abs)[:, None]) #(B,JQ,J+1,T)
        # The 4th-order statistic itself: correlate two fine scales s1, s2 at
        # the SAME coarse scale j, average over time, keep the real part.
        output = torch.real(W_Wx[:, :, :, None] * W_Wx[:, :, None].conj()) #(B,JQ,J+1,J+1,T)
        output = output.mean(-1) #(B,JQ,J+1,J+1)
        output = output.permute(0, 3, 2, 1) #(B,J+1,J+1,JQ)  -> (batch, s2, s1, j), matching self.indices' column order

        indices = self.indices.long()

        output = output[:, indices[0], indices[1], indices[2]]

        output = output.reshape(x.shape[0], indices.shape[1])
        return output

    # Adjoint of forward() above (used by the SDE solver's score term during
    # generation) -- same normalization convention (self.norm/self.norm_indices),
    # not walked through line-by-line here.
    def grad(self, x,  v=None, means=None):
        filters = self.filters.to(x.device)
        filters_Q = self.filters_Q.to(x.device)
        number_filters = filters.shape[1]
        number_filters_Q = filters_Q.shape[1]

        
        x_filtered_no_LF = torch.fft.ifft(filters_Q*torch.fft.fft(x)) # (B,J*Q,M,N)
        x_filtered_abs_no_LF = abs_eps(x_filtered_no_LF) # (B,J*L,M,N)
        x_filtered_over_abs_no_LF = (x_filtered_no_LF/x_filtered_abs_no_LF)

        

        x_filtered_abs_no_LF_filtered_2 = torch.fft.ifft((torch.fft.fft(x_filtered_abs_no_LF)[:,:,None]*(filters*filters.conj()))) # (B,J*Q,J+1,M) #no 2*, x_filtered_abs_no_LF

        indices = self.indices

        if v != None:
            m = torch.zeros((number_filters_Q, number_filters_Q, number_filters)).to(x.device)
            m[indices[0], indices[1], indices[2]] = v / self.norm_indices.to(v.device).to(v.dtype)
            m = m+torch.transpose(m,0,1)

            intermediate_output = x_filtered_over_abs_no_LF*torch.einsum('ijk, ajkb -> aib', m, torch.real(x_filtered_abs_no_LF_filtered_2))
            return torch.fft.ifft(filters_Q.conj()*torch.fft.fft(intermediate_output)).real.sum(1)[:,None]/x.shape[-1]

        intermediate_output = x_filtered_over_abs_no_LF[:,:,None,None]*torch.real(x_filtered_abs_no_LF_filtered_2)[:,None] #x_filtered_over_abs_no_LF
        output = torch.fft.ifft(filters_Q[:,:,None,None].conj()*torch.fft.fft(intermediate_output)).real

        output = output + torch.transpose(output, 1,2)      
        output = output[:, indices[0], indices[1], indices[2]].reshape(x.shape[0], indices.shape[1], x.shape[-1])/x.shape[-1]

        
        return output / self.norm_indices[:,None].to(x.device).to(x.dtype)
        
    def fit_micro(self,x):
        """
        Called ONCE on real data, before any fitting/generation (never on
        synthetic samples) -- computes self.norm, the per-fine-scale factor
        forward() divides the raw wavelet coefficient by (see the
        "Normalize micro" line there).

        WHY: this codebase's data (fBm/MRW, turbulence) typically has a
        power-law spectrum, so E[|Wx_s|^2] varies hugely across fine scales
        s -- some scales are just naturally louder than others. Without
        correcting for that, the 4th-order statistic in forward() would be
        dominated by whichever scale carries the most raw power, and fitting
        theta at different frequencies becomes ill-conditioned (see the
        scattering-spectra paper's renormalization section: "renormalizing
        to 1 the variance of wavelet coefficients at all scales avoids
        numerical instabilities"). Dividing Wx_s by its own scale, once,
        here, fixes that for every statistic built from it downstream.
        """
        filters_Q = self.filters_Q.to(x.device)
        x_filtered = torch.fft.ifft(filters_Q*torch.fft.fft(x))
        x_filtered = x_filtered.abs()**2   # -> E[|Wx_s|^2] per scale once .mean()'d below
        #Normalize_micro
        self.norm = x_filtered.mean((0,2))[:,None]**0.5
        # self.norm = x_filtered.mean((0,2))[:,None]**(-1/4)   # previous convention, kept for comparison (chat 2026-08-31): doesn't cancel the scale-dependence derived above -- see fit_micro docstring on Mod2_Real_1d for the worked-out reason.

        # Same normalization, reshaped onto self.indices' flat (s1,s2,j)
        # coefficient layout -- this is what grad() actually uses, since it
        # works on the flat coefficient vector, not the (J*Q, J*Q) scale grid
        # self.norm itself lives on.
        indices = self.indices
        norm_indices = self.norm[:,None]*self.norm[None,:]
        norm_indices = norm_indices.repeat((1,1,self.J+1))
        self.norm_indices =  norm_indices[indices[0], indices[1], indices[2]] #(n_pot)

class Scattering_Fourth_Order_Imag_1d(Potential):
    def __init__(self, J,Q,filters,filters_Q):
        super().__init__()
        self.J = J
        self.Q = Q
        self.filters = filters
        self.filters_Q = filters_Q
        self.indices = indices_fourth_order_Q(self.J, Q,offset = 1,lite=True,include_lowpass = False)

        self.norm_indices = torch.ones((len(self.indices[0]),))
        self.norm = 1

        self.num_coefficients = len(self.indices[0])

        

    def forward(self, x):
        filters = self.filters.to(x.device)
        filters_Q = self.filters_Q.to(x.device)
        
        x_filtered = torch.fft.ifft(filters_Q*torch.fft.fft(x)) #(B,JQ,T)
        
        #Normalize micro
        x_filtered = x_filtered/self.norm
        
        x_filtered_abs = abs_eps(x_filtered) #(B,JQ,T)
        
        W_Wx = torch.fft.ifft(filters[:, :, None] * torch.fft.fft(x_filtered_abs)[:, None]) #(B,JQ,J+1,T)
        output = torch.imag(W_Wx[:, :, :, None] * W_Wx[:, :, None].conj()) #(B,JQ,J+1,J+1,T)
        output = output.mean(-1) #(B,JQ,J+1,J+1)
        output = output.permute(0, 3, 2, 1) #(B,J+1,J+1,JQ)
        
        indices = self.indices.long()
        
        output = output[:, indices[0], indices[1], indices[2]]
        
        output = output.reshape(x.shape[0], indices.shape[1])
        return output

    def grad(self, x,  v=None, means=None):
        filters = self.filters.to(x.device)
        filters_tilde = torch.fft.fft(torch.fft.ifft(filters).conj()).conj()
        filters_Q = self.filters_Q.to(x.device)
        filters_Q_tilde = torch.fft.fft(torch.fft.ifft(filters_Q).conj()).conj()
        # psi_hat(-k): the conjugate-coefficient branch of the adjoint. For a real filter bank
        # (Morlet) filters_Q_tilde is real up to FFT round-off and the original .conj() is kept,
        # so Morlet runs stay bit-identical; a complex (Daubechies) bank needs it without conj.
        filters_Q_tilde_adj = filters_Q_tilde if filters_Q.is_complex() else filters_Q_tilde.conj()
        number_filters = filters.shape[1]
        number_filters_Q = filters_Q.shape[1]

        
        x_filtered_no_LF = torch.fft.ifft(filters_Q*torch.fft.fft(x)) # (B,J*Q,M,N)
        x_filtered_abs_no_LF = abs_eps(x_filtered_no_LF) # (B,J*L,M,N)
        x_filtered_over_abs_no_LF = (x_filtered_no_LF/x_filtered_abs_no_LF)

        x_filtered_abs_no_LF_filtered_2 = torch.fft.ifft((torch.fft.fft(x_filtered_abs_no_LF)[:,:,None]*filters.abs()**2)) # (B,J*Q,J+1,M,N) #no 2*, x_filtered_abs_no_LF
        x_filtered_abs_no_LF_filtered_2_tilde = torch.fft.ifft((torch.fft.fft(x_filtered_abs_no_LF)[:,:,None]*filters_tilde.abs()**2)) # (B,J*Q,J+1,M,N) #no 2*, x_filtered_abs_no_LF
        
        indices = self.indices

        if v != None:
            m = torch.zeros((number_filters_Q, number_filters_Q, number_filters)).to(x.device)+0*1j
            m[indices[0], indices[1], indices[2]] = v / self.norm_indices.to(v.device).to(v.dtype) +0*1j
            m_transpose = m.transpose(0,1)
           
            intermediate_output = 0.5*x_filtered_over_abs_no_LF*torch.einsum('ijk, ajkb -> aib', m, x_filtered_abs_no_LF_filtered_2)
            intermediate_output_conj = 0.5*x_filtered_over_abs_no_LF.conj()*torch.einsum('ijk, ajkb -> aib', m, x_filtered_abs_no_LF_filtered_2)

            intermediate_output = torch.fft.ifft(filters_Q.conj()*torch.fft.fft(intermediate_output)).sum(1)[:,None]/x.shape[-1]
            intermediate_output_conj = torch.fft.ifft(filters_Q_tilde*torch.fft.fft(intermediate_output_conj)).sum(1)[:,None]/x.shape[-1]

            output =intermediate_output + intermediate_output_conj

            intermediate_output = 0.5*x_filtered_over_abs_no_LF*torch.einsum('ijk, ajkb -> aib', m_transpose, x_filtered_abs_no_LF_filtered_2_tilde)
            intermediate_output_conj = 0.5*x_filtered_over_abs_no_LF.conj()*torch.einsum('ijk, ajkb -> aib', m_transpose, x_filtered_abs_no_LF_filtered_2_tilde)

            intermediate_output = torch.fft.ifft(filters_Q.conj()*torch.fft.fft(intermediate_output)).sum(1)[:,None]/x.shape[-1]
            intermediate_output_conj = torch.fft.ifft(filters_Q_tilde*torch.fft.fft(intermediate_output_conj)).sum(1)[:,None]/x.shape[-1]

            output_transpose =intermediate_output + intermediate_output_conj

            output = output + output_transpose
            return output.imag
            
        intermediate_output = 0.5*x_filtered_over_abs_no_LF[:,:,None,None]*x_filtered_abs_no_LF_filtered_2[:,None] #x_filtered_over_abs_no_LF
        intermediate_output_conj = 0.5*x_filtered_over_abs_no_LF[:,:,None,None].conj()*x_filtered_abs_no_LF_filtered_2[:,None] #x_filtered_over_abs_no_LF

        intermediate_output = torch.fft.ifft(filters_Q[:,:,None,None].conj()*torch.fft.fft(intermediate_output))
        intermediate_output_conj = torch.fft.ifft(filters_Q_tilde_adj[:,:,None,None]*torch.fft.fft(intermediate_output_conj))
        
        output =intermediate_output + intermediate_output_conj

        intermediate_output = 0.5*x_filtered_over_abs_no_LF[:,:,None,None]*x_filtered_abs_no_LF_filtered_2_tilde[:,None] #x_filtered_over_abs_no_LF
        intermediate_output_conj = 0.5*x_filtered_over_abs_no_LF[:,:,None,None].conj()*x_filtered_abs_no_LF_filtered_2_tilde[:,None] #x_filtered_over_abs_no_LF

        intermediate_output = torch.fft.ifft(filters_Q[:,:,None,None].conj()*torch.fft.fft(intermediate_output))
        intermediate_output_conj = torch.fft.ifft(filters_Q_tilde_adj[:,:,None,None]*torch.fft.fft(intermediate_output_conj))

        output_transpose =intermediate_output + intermediate_output_conj

        output = output + torch.transpose(output_transpose, 1,2)
        
        output = output[:, indices[0], indices[1], indices[2]].reshape(x.shape[0], indices.shape[1], x.shape[-1])/x.shape[-1]

        return output.imag / self.norm_indices[:,None].to(x.device).to(x.dtype)
        
    def fit_micro(self,x):
        filters_Q = self.filters_Q.to(x.device)
        x_filtered = torch.fft.ifft(filters_Q*torch.fft.fft(x))
        x_filtered = x_filtered.abs()**2
        #Normalize_micro
        self.norm = x_filtered.mean((0,2))[:,None]**0.5
        
        indices = self.indices
        norm_indices = self.norm[:,None]*self.norm[None,:] 
        norm_indices = norm_indices.repeat((1,1,self.J+1))
        self.norm_indices =  norm_indices[indices[0], indices[1], indices[2]] #(n_pot)




        # ----- Scalar potentials -----


# FOURTH ORDER WITH SQUARE 
class Scattering_Fourth_Order_Mod2_Real_1d(Potential):
    def __init__(self, J,Q,filters,filters_Q,include_diag = False,lite=True):
        super().__init__()
        self.J = J
        self.Q = Q
        self.filters = filters
        self.filters_Q = filters_Q
        if include_diag is True:
            offset = 0
        else:
            offset = 1
        self.indices = indices_fourth_order_Q(self.J, self.Q,offset,lite)
        self.norm = 1
        self.norm_indices = torch.ones((len(self.indices[0]),))
        
        self.num_coefficients = len(self.indices[0])
        
    def forward(self, x):
        # statistic (i0, i1, i2) = mean_t Re[ (psi_i2 * E_i1) conj(psi_i2 * E_i0) ], E = |W x / norm|^2
        return _scat4_forward(self, x, False)

    def grad(self, x,  v=None, means=None):
        # grad phi_abc = T[a,b,c] + T[b,a,c], T[a,b,c] = W_a^*( 2 W_a x . Re(|psi_c|^2 * |W_b x|^2) ) / M / norm_indices
        if v is None:
            return _scat4_jac(self, x, False)
        if not self.filters_Q.is_complex():
            return _scat4_grad_v(self, x, v, False)
        # complex filter bank: original grad(x, v) (the fast one is derived for real banks)
        filters = self.filters.to(x.device)
        filters_Q = self.filters_Q.to(x.device)
        number_filters = filters.shape[1]
        number_filters_Q = filters_Q.shape[1]
        x_filtered_no_LF = torch.fft.ifft(filters_Q*torch.fft.fft(x))
        x_filtered_abs2_no_LF = abs_eps(x_filtered_no_LF)**2
        x_filtered_grad = 2 * x_filtered_no_LF
        x_filtered_abs2_no_LF_filtered_2 = torch.fft.ifft(
            (torch.fft.fft(x_filtered_abs2_no_LF)[:,:,None] * (filters*filters.conj()))
        )
        indices = self.indices
        m = torch.zeros((number_filters_Q, number_filters_Q, number_filters)).to(x.device)
        m[indices[0], indices[1], indices[2]] = v / self.norm_indices.to(v.device).to(v.dtype)
        m = m+torch.transpose(m,0,1)
        intermediate_output = x_filtered_grad*torch.einsum('ijk, ajkb -> aib', m, torch.real(x_filtered_abs2_no_LF_filtered_2))
        return torch.fft.ifft(filters_Q.conj()*torch.fft.fft(intermediate_output)).real.sum(1)[:,None]/x.shape[-1]
        
    def fit_micro(self,x):
        # Same normalization role as Scattering_Fourth_Order_Real_1d.fit_micro
        # (see its docstring for the full explanation) -- the only difference
        # is this class's envelope is |Wx_s|**2 (modulus SQUARED) instead of
        # |Wx_s|, since that's what forward() feeds into the second wavelet
        # transform above. Squaring the envelope doesn't change what exponent
        # cancels the scale-dependence, though: the extra power from squaring
        # the envelope and the extra power from squaring self.norm's own
        # effect cancel each other out, so the same self.norm**0.5 is correct
        # for both this class and the plain (non-Mod2) one.
        filters_Q = self.filters_Q.to(x.device)
        x_filtered = torch.fft.ifft(filters_Q*torch.fft.fft(x))
        x_filtered = x_filtered.abs()**2
        #Normalize_micro
        self.norm = x_filtered.mean((0,2))[:,None]**0.5
        # self.norm = x_filtered.mean((0,2))[:,None]**(-1/4)   # previous convention, kept for comparison (chat 2026-08-31)

        # FIX (chat 2026-08-31): grad() derives its analytic gradient from the
        # RAW (unnormalized) coefficient and injects the normalization only via
        # this final division by norm_indices -- unlike forward(), which divides
        # the coefficient by self.norm before squaring the envelope. Because this
        # class's envelope is |Wx/norm|**2 (quadratic in 1/norm, vs. the plain
        # Real/Imag classes' |Wx/norm| which is linear), the correct per-scale
        # factor is norm**2, not norm -- squaring here is what forward()'s
        # pre-division already gets "for free" by construction, but grad() must
        # be told explicitly. Previously norm_indices used self.norm (unsquared,
        # copy-pasted from the linear-envelope classes), which under-corrected
        # grad() by a factor of norm[s1]*norm[s2] whenever self.norm != 1.
        indices = self.indices
        norm_indices = (self.norm**2)[:,None]*(self.norm**2)[None,:]
        norm_indices = norm_indices.repeat((1,1,self.J+1))
        self.norm_indices =  norm_indices[indices[0], indices[1], indices[2]] #(n_pot)

class Scattering_Fourth_Order_Mod2_Imag_1d(Potential):
    def __init__(self, J,Q,filters,filters_Q):
        super().__init__()
        self.J = J
        self.Q = Q
        self.filters = filters
        self.filters_Q = filters_Q
        self.indices = indices_fourth_order_Q(self.J, Q,offset = 1,lite=True,include_lowpass = False)

        self.norm_indices = torch.ones((len(self.indices[0]),))
        self.norm = 1

        self.num_coefficients = len(self.indices[0])

    def forward(self, x):
        # statistic (i0, i1, i2) = mean_t Im[ (psi_i2 * E_i1) conj(psi_i2 * E_i0) ], E = |W x / norm|^2
        return _scat4_forward(self, x, True)

    def grad(self, x,  v=None, means=None):
        if not self.filters_Q.is_complex():
            return _scat4_jac(self, x, True) if v is None else _scat4_grad_v(self, x, v, True)
        # complex (Daubechies) filter bank: original code below (the fast one is derived for real banks)
        filters = self.filters.to(x.device)
        filters_tilde = torch.fft.fft(torch.fft.ifft(filters).conj()).conj()
        filters_Q = self.filters_Q.to(x.device)
        filters_Q_tilde = torch.fft.fft(torch.fft.ifft(filters_Q).conj()).conj()
        # psi_hat(-k): the conjugate-coefficient branch of the adjoint. For a real filter bank
        # (Morlet) filters_Q_tilde is real up to FFT round-off and the original .conj() is kept,
        # so Morlet runs stay bit-identical; a complex (Daubechies) bank needs it without conj.
        filters_Q_tilde_adj = filters_Q_tilde if filters_Q.is_complex() else filters_Q_tilde.conj()
        number_filters = filters.shape[1]
        number_filters_Q = filters_Q.shape[1]

        x_filtered_no_LF = torch.fft.ifft(filters_Q*torch.fft.fft(x)) # (B,J*Q,M,N)
        x_filtered_abs2_no_LF = abs_eps(x_filtered_no_LF)**2 # (B,J*L,M,N)  <-- mod2
        x_filtered_grad = 2 * x_filtered_no_LF                            # <-- chain-rule term for |y|^2

        x_filtered_abs2_no_LF_filtered_2 = torch.fft.ifft(
            (torch.fft.fft(x_filtered_abs2_no_LF)[:,:,None]*filters.abs()**2))
        x_filtered_abs2_no_LF_filtered_2_tilde = torch.fft.ifft(
            (torch.fft.fft(x_filtered_abs2_no_LF)[:,:,None]*filters_tilde.abs()**2))

        indices = self.indices

        if v != None:
            m = torch.zeros((number_filters_Q, number_filters_Q, number_filters)).to(x.device)+0*1j
            m[indices[0], indices[1], indices[2]] = v / self.norm_indices.to(v.device).to(v.dtype) +0*1j
            m_transpose = m.transpose(0,1)

            intermediate_output = 0.5*x_filtered_grad*torch.einsum('ijk, ajkb -> aib', m, x_filtered_abs2_no_LF_filtered_2)
            intermediate_output_conj = 0.5*x_filtered_grad.conj()*torch.einsum('ijk, ajkb -> aib', m, x_filtered_abs2_no_LF_filtered_2)

            intermediate_output = torch.fft.ifft(filters_Q.conj()*torch.fft.fft(intermediate_output)).sum(1)[:,None]/x.shape[-1]
            intermediate_output_conj = torch.fft.ifft(filters_Q_tilde*torch.fft.fft(intermediate_output_conj)).sum(1)[:,None]/x.shape[-1]

            output = intermediate_output + intermediate_output_conj

            intermediate_output = 0.5*x_filtered_grad*torch.einsum('ijk, ajkb -> aib', m_transpose, x_filtered_abs2_no_LF_filtered_2_tilde)
            intermediate_output_conj = 0.5*x_filtered_grad.conj()*torch.einsum('ijk, ajkb -> aib', m_transpose, x_filtered_abs2_no_LF_filtered_2_tilde)

            intermediate_output = torch.fft.ifft(filters_Q.conj()*torch.fft.fft(intermediate_output)).sum(1)[:,None]/x.shape[-1]
            intermediate_output_conj = torch.fft.ifft(filters_Q_tilde*torch.fft.fft(intermediate_output_conj)).sum(1)[:,None]/x.shape[-1]

            output_transpose = intermediate_output + intermediate_output_conj

            output = output + output_transpose
            return output.imag

        # Full Jacobian (2026-11-17: gather). grad phi_abc = Im( P[a,b,c] + P~[b,a,c] ), P from the envelope filtered
        # by |psi_c|^2, P~ by |psi~_c|^2; each entry depends on its own (a,b,c) only, so only the K kept triples of P
        # and their partners (b,a,c) of P~ are computed (was: both full n_Q x n_Q x n_f tensors, then indexed).
        a, b, c = indices.long()
        g = torch.cat([x_filtered_grad[:, a], x_filtered_grad[:, b]], 1)            # (B, 2K, M): W_a x | W_b x
        e = torch.cat([x_filtered_abs2_no_LF_filtered_2[:, b, c],                    # P[a,b,c]
                       x_filtered_abs2_no_LF_filtered_2_tilde[:, a, c]], 1)          # P~[b,a,c]
        aa = torch.cat([a, b])
        P = torch.fft.ifft(filters_Q[:, aa].conj() * torch.fft.fft(0.5 * g * e)) \
            + torch.fft.ifft(filters_Q_tilde_adj[:, aa] * torch.fft.fft(0.5 * g.conj() * e))
        K = a.shape[0]
        output = (P[:, :K] + P[:, K:]) / x.shape[-1]

        return output.imag / self.norm_indices[:,None].to(x.device).to(x.dtype)

    def fit_micro(self,x):
        filters_Q = self.filters_Q.to(x.device)
        x_filtered = torch.fft.ifft(filters_Q*torch.fft.fft(x))
        x_filtered = x_filtered.abs()**2
        #Normalize_micro
        self.norm = x_filtered.mean((0,2))[:,None]**0.5   # matches Imag_1d convention (not the -1/4 used by the Real classes)

        # FIX (chat 2026-08-31): same norm**2 correction as
        # Scattering_Fourth_Order_Mod2_Real_1d.fit_micro -- see its comment.
        # This class's |Wx/norm|**2 envelope makes grad()'s raw-statistic
        # scaling quadratic in 1/norm per scale, so norm_indices needs
        # self.norm**2, not self.norm.
        indices = self.indices
        norm_indices = (self.norm**2)[:,None]*(self.norm**2)[None,:]
        norm_indices = norm_indices.repeat((1,1,self.J+1))
        self.norm_indices =  norm_indices[indices[0], indices[1], indices[2]] #(n_pot)


class Scalar(Potential):
    def __init__(self,filters, scalar_param =None,quantiles = True,confine=True):
        """ Build num_potentials windows in [-domain, domain] whose
        stride is stride_sigmas sigmas. """
        "Will reconstruct x at the finer scale, without high frequency, and compute the scalar potential of mid freqs conditionaly to low freqs"
        super().__init__()

        self.filters = filters
        

        if scalar_param is None:
            scalar_param = {}
            scalar_param['stride_sigmas'] = 0.75 
            scalar_param['num_centers'] = 20
            scalar_param['margin'] = -0.25
           
        self.stride_sigmas = scalar_param['stride_sigmas']
        self.num_potentials_scalar = scalar_param['num_centers'] 
        self.num_potentials = self.num_potentials_scalar * self.filters.shape[1] 
        

        self.quantiles = quantiles
        if self.quantiles is True:
            try: 
                self.quantile_min = scalar_param['quantile_min']
                self.quantile_max = scalar_param['quantile_max']

            except :
                self.quantile_min = 0
                self.quantile_max = 1

        else:
            self.margin = scalar_param['margin']

        self.confine = confine

        self.num_coefficients =   self.num_potentials
        

    def fit(self,x):
        """adapt for several channels"""
        self.device = x.device
        filters = self.filters.to(x.device)

        x_filtered_abs = torch.fft.ifft(filters*torch.fft.fft(x)).abs() #(B,J*Q,T)
        
        

        if self.quantiles is False:
            window_min, window_max = x_filtered_abs.min(0)[0].min(1)[0] , x_filtered_abs.max(0)[0].max(1)[0] #(J*Q)
            domain = - window_min + window_max
            self.window_min,self.window_max = window_min, window_max+domain*self.margin #(J*Q)

            sigma = ((-self.window_min+self.window_max) / (self.num_potentials_scalar-1 )) * self.stride_sigmas #(J*Q)
    
            self.sigma = sigma #(n_pots,J*Q)
    
            self.centers = torch.stack([torch.linspace(self.window_min[i], self.window_max[i], self.num_potentials_scalar, device=self.device) for i in range(self.filters.shape[1])],dim=1)  #(n_pots,J*Q)

        else:
            centers = []
            sigma = []
            for j in range(x_filtered_abs.shape[1]): 
                ct = torch.quantile(x_filtered_abs[:,j].flatten(),torch.linspace(self.quantile_min,self.quantile_max,self.num_potentials_scalar).to(self.device))
                centers.append(ct)
                sigma.append(torch.cat([(ct[1:-1]-ct[:-2])*self.stride_sigmas,self.stride_sigmas*(ct[-2]-ct[-3])[None],self.stride_sigmas*(ct[-1]-ct[-2])[None]]))

            self.centers = torch.stack(centers,dim=1) #(n_pot_scalar,J*Q)
            self.sigma = torch.stack(sigma,dim=1) #(n_pot_scalar,J*Q)

   
        
    def forward(self, x, *args):
        #(B,1,T)
        filters = self.filters.to(x.device)
        centers = self.centers.to(x.device)
        sigma = self.sigma.to(x.device)
        
        x_filtered_abs = torch.fft.ifft(filters*torch.fft.fft(x)).abs() #(B,J*Q,T)

        if self.confine is False:
            x =  torch.sigmoid(-(x_filtered_abs[:, None] - centers[:,:,None])/ sigma[...,None]) #(B,n_pots,J*Q,T)
        else:
            if self.quantiles is True:
                x = torch.sigmoid((x_filtered_abs[:, None] - centers[:-1,:,None])/ sigma[:-1,:,None]) #(B,n_pots-1,J*Q,T)
                x_ = torch.sigmoid((x_filtered_abs - centers[-1,:,None])/ sigma[-1,:,None])*(1+((x_filtered_abs - centers[-1,:,None])/ sigma[-1,:,None])**8) #(B,J*Q,T) 
                x = torch.cat([x,x_[:,None]],dim=1) #(B,n_pots,J*Q,T)
            else:
                print('to be coded')

        
        x = x.mean(3) #(B,n_pots,J*Q)
        x = x / self.centers
        
        x = x.reshape((x.shape[0],-1))
      
        return x #(B,n_pots*J*Q)
        
    def grad_autograd(self, x):
        return torch.func.vmap(torch.func.jacrev(self.forward))(x[:,None]).reshape((x.shape[0],-1, x.shape[-1]))

    def grad(self, x, v=None, means=None):
        
        filters = self.filters.to(x.device)
        centers = self.centers.to(x.device)
        sigma = self.sigma.to(x.device)
        
        x_fourier = torch.fft.fft(x)
        x_filtered = torch.fft.ifft(filters*x_fourier)
        x_filtered_abs = abs_eps(x_filtered)
        x_filtered_over_abs = x_filtered/x_filtered_abs

        x_L1 = torch.real(torch.fft.ifft(torch.fft.fft(x_filtered_over_abs)*filters.conj())) #(B,J*Q,T)

       
        
        if self.confine is False:
            x_scalar =  torch.sigmoid(-(x_filtered_abs[:, None] - centers[:,:,None])/ sigma[...,None]) #(B,n_pots,J*Q,T)
            x_scalar = -x_scalar*(1-x_scalar)/ sigma[...,None]  #(B,n_pots,J*Q,T)
        else:
             if self.quantiles is True:
                 x_scalar =  torch.sigmoid((x_filtered_abs[:, None] - centers[:,:,None])/ sigma[...,None]) #(B,n_pots,J*Q,T)
                 x_sig = x_scalar[:,-1]
                 x_scalar = x_scalar*(1-x_scalar)/ sigma[...,None]  #(B,n_pots,J*Q,T)

                 x_sig = x_scalar[:,-1]*((x_filtered_abs - centers[-1,:,None])/ sigma[-1,:,None])**8 + x_sig *8* ((x_filtered_abs - centers[-1,:,None])/ sigma[-1,:,None])**7 / sigma[-1,:,None]  #(B,n_pots,J*Q,T)

                 x_scalar = torch.cat([x_scalar[:,:-1],x_sig[:,None]],dim =1)
                 
             else:
                 print('to be coded')

            
        output = x_L1[:,None]*x_scalar#torch.fft.ifft(torch.fft.fft(x_L1)[:,None]*torch.fft.fft(x_scalar)).real #(B,n_pots,J*Q,T) 

        output = output / self.centers [...,None]

        
        output = output.reshape((output.shape[0],-1,output.shape[-1]))  #(B,n_pots*J*Q,T)
        
        if v==None:
            return output/x.shape[-1]
        else:
            return (output*v[None,:,None]).sum(1)[:,None]/x.shape[-1]



        
    # ----- Other potentials -----


class L2p_norm(Potential):
    def __init__(self, p,filters):
        super().__init__()
        self.p = p
        self.filters=filters
        self.norm = 1
        self.num_coefficients = filters.shape[1]
 
    def forward(self, x):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft(filters*torch.fft.fft(x))
        return (torch.abs(x_filtered)**(2*self.p)).mean(-1)/self.norm

    def grad_autograd(self, x):
        filters = self.filters.to(x.device)
        return torch.func.vmap(torch.func.jacrev(self.forward))(x[:,None]).reshape((x.shape[0],filters.shape[1], x.shape[-1])) /self.norm

    def grad(self, x,  v=None, means=None):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft(filters*torch.fft.fft(x))
            
        output = (x_filtered *(2* self.p) * (torch.abs(x_filtered)**(2*self.p-2)))
        if v is not None:
            # 2026-11-17: sum_j v_j Re ifft(conj(psi_j) fft(D_j)) / norm_j = Re ifft(sum_j (v_j / norm_j) conj(psi_j) fft(D_j)):
            # one inverse FFT instead of one per channel; same quantity up to round-off
            w = v[:, None] / self.norm                                          # (n, 1)
            S = (filters.conj() * torch.fft.fft(output) * w[None]).sum(1, keepdim=True)
            return torch.fft.ifft(S).real / x.shape[-1]
        output = torch.fft.ifft(filters.conj()*torch.fft.fft(output)).real

        output = output / self.norm
        
        if v==None:
            return output/x.shape[-1]
        else:
            return (output*v[None,:,None]).sum(1)[:,None]/x.shape[-1]
            
    def fit_micro(self,x):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft(filters*torch.fft.fft(x))
        self.norm =  (((torch.abs(x_filtered)**(2*self.p)).mean((0,2))) **(-1/2))[:,None]
        #self.norm = (torch.abs(x_filtered)**(2*self.p)).mean((0,2))[:,None]

class L2p1_norm(Potential):
    def __init__(self, p,filters):
        super().__init__()
        self.p = p
        self.filters=filters
        self.norm = 1
        self.num_coefficients = filters.shape[1]
 
    def forward(self, x):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft(filters*torch.fft.fft(x))
        return (torch.abs(x_filtered)**(2*self.p+1)).mean(-1) /self.norm

    def grad_autograd(self, x):
        filters = self.filters.to(x.device)
        return torch.func.vmap(torch.func.jacrev(self.forward))(x[:,None]).reshape((x.shape[0],filters.shape[1], x.shape[-1]))

    def grad(self, x,  v=None, means=None):
        filters = self.filters.to(x.device)
         
        x_filtered = torch.fft.ifft(self.filters*torch.fft.fft(x))
            
        output =  x_filtered * (2*self.p+1) * (torch.abs(x_filtered)**(2*self.p-1))
        output = torch.fft.ifft(filters.conj()*torch.fft.fft(output)).real

        output = output / self.norm
        
        if v==None:
            return output/x.shape[-1]
        else:
            return (output*v[None,:,None]).sum(1)[:,None]/x.shape[-1]
    
    def fit_micro(self,x):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft(filters*torch.fft.fft(x))
        self.norm = (torch.abs(x_filtered)**(2*self.p+1)).mean((0,2))


def compute_z(x, filters):
    filters = filters.to(x.device)
    return torch.fft.ifft(filters * torch.fft.fft(x)).real

# z0 = compute_z(x, filters) --> in the notebook 


#from potentials.potentials_classes.coshGt import Scalar_coshgt
#from potentials.potentials_classes.generalized_gauss_gamma import Scalar_GGD_GenGamma
#from potentials.potentials_classes.generalized_gaussian_pow import Scalar_GGD_GGD_Pow
#from potentials.potentials_classes.generalized_gaussianx3 import Scalar_GGD_GGD_GGD

import numpy as np
import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.special import gammaln, gammainc
from scipy.optimize import minimize

import sys
from pathlib import Path
# Get the absolute path of the directory 3 levels up from this file
# (from potentials_classes -> potentials -> codes)
project_root = Path(__file__).resolve().parents[1]
# Target the filters directory absolutely
filters_path = project_root / 'filters'
if str(filters_path) not in sys.path:
    sys.path.insert(0, str(filters_path))
from filters_1d import init_band_pass
from filters_bank import return_Filters


class Scalar_GGD_KRegion():
    def __init__(self, filters,
                 num_regions=4,
                 trans_frac=0.10,
                 alpha_bounds=(0.2, 8.0),
                 min_region_samples=30,
                 eps_abs=1e-6,
                 eps_quantile=1e-3,
                 near_empty_tol=1e-6,
                 boundary_method="auto",
                 model_criterion="bic",
                 boundary_search_subsample=20000,
                 boundary_search_iters=3,
                 boundary_search_grid=12,
                 pi_active_min=1e-3,
                 auto_prune=True,
                 cond_tol=1e-6,
                 prune_max_cols=20000,
                 kurt_thresholds=None,
                 tail_quantile=None,
                 tail_alpha_bounds=None,
                 tail_mass=False,
                 all_masses=False,
                 min_window=None,
                 min_window_bic=None,
                 dwt=None,
                 verbose=True):
        self.filters = filters
        # dwt: a PeriodicDWT (codes/ortho_wavelet/periodic_dwt.py). The channels are then the
        # critically decimated coefficients (channel j has dwt.sizes[j] of them, the last one
        # is the approximation) instead of the undecimated bank `filters` (pass None).
        self.dwt = dwt
        # tail_mass: add one statistic per channel, <w_K(|z|)>, the probability mass of the
        # outermost region (exponent 0 on the outer window). With a single statistic per
        # region the multiplier theta_K has to set both the mass of that region and its
        # decay; the MaxEnt then reproduces the fitted shape only by accident, and the
        # outer alpha that gives the right quantiles is not the MLE one (the scale floor was
        # compensating for that). With the mass statistic the two decouple, so the outer
        # alpha is fitted by plain truncated-GGD MLE, without scale floor, in [0.05, 8].
        # Offline held-out test, 12 block splits, 5 channels (turbulence/tail_form_holdout.py,
        # 2026-10-21): q.9999 model/held-out 0.95-1.04 vs 0.40-0.66 for the production fit.
        self.tail_mass = bool(tail_mass)
        if self.tail_mass:
            assert int(num_regions) >= 2, "tail_mass needs at least two regions"
        # all_masses (2026-10-27): on top of tail_mass, the mass <w_k(|z|)> of every inner window
        # too, except the first active window of each channel (the masses of all windows sum to
        # 1, so one of them is fixed by the others and would make the Gram singular). Bounded
        # statistics: the lowest-variance way to add constraints on the marginal, on the same
        # windows. Slots: K power slots, then the outer mass (slot K, as with tail_mass alone),
        # then the masses of windows 1..K-2 (slots K+1..).
        self.all_masses = bool(all_masses)
        if self.all_masses:
            assert self.tail_mass, "all_masses extends tail_mass: pass tail_mass=True too"
        # tail_quantile: if set, the outermost cut is forced to quantile(|z_j|,
        # tail_quantile) on the fit data; inner cuts found by the boundary search above it
        # are discarded. The likelihood search is bulk-dominated and places the last cut
        # from a few tail points, yet the model tail beyond it is set by that region's
        # single (theta, alpha) pair.
        # tail_alpha_bounds: alpha bounds for the outermost region only, so its
        # statistic |z|^alpha is neither blind to the tail (alpha -> 0.2) nor dominated
        # by a handful of extremes (alpha -> 8).
        # A list of quantiles (2026-10-28, e.g. [0.99, 0.999]) forces one cut per quantile; all
        # windows above the first one get the tail settings (tail bounds, tail_mass floor).
        self.tail_quantile = tail_quantile
        self.tail_alpha_bounds = tail_alpha_bounds
        # min_window: no inner window narrower than min_window * n_j, n_j = std of channel j
        # for unit white noise in x, so min_window = c * sqrt(2 h_end) * sigma makes every
        # window >= c times the noise the last SDE step adds to z_j. Narrower windows are
        # merged with the one above (with the one below if that is the forced tail cut).
        # MGD cannot hold walkers in a window narrower than one step's noise: on the finest
        # channels those constraints stayed 10-36% off at t=1, all others < 1%
        # (turbulence/window_vs_step_noise.py, mass_L6_2b, 2026-10-22).
        self.min_window = min_window
        # min_window_bic (2026-10-27): same units as min_window (times n_j, the channel's std for
        # unit white noise in x), but imposed INSIDE the likelihood search instead of merging
        # afterwards: candidate cuts that leave a window narrower than min_window_bic * n_j are
        # never considered, and BIC compares only admissible partitions (merging deletes cuts of
        # an unconstrained fit and leaves a partition that is not likelihood-optimal; on
        # mass_L6_2b it thinned the finest Q=3 tail, q.9999 0.71 -> 0.48). Why a bound at all:
        # windows narrower than one SDE step's noise are not held by the corrector (psi[6] of
        # mass_L6_2b: [0, 8.1e-5), noise > width even at the last step, 37% off at t=1).
        self.min_window_bic = min_window_bic
        self.K = int(num_regions)
        assert self.K >= 1
        self.J = filters.shape[1] if dwt is None else dwt.n_channels
        self.trans_frac = trans_frac
        self.alpha_bounds = alpha_bounds
        self.min_region_samples = min_region_samples
        # |z| is smoothed as sqrt(z^2 + eps) so that the core-region gradient
        # alpha * z * |z|^(alpha - 2) stays finite at z = 0 when alpha < 1.
        # eps_quantile (default): per-channel eps_j = quantile(|z_j|, eps_quantile)^2 on
        # the data at fit time, i.e. the floor moves at most that fraction of the
        # points, at every scale. The old absolute eps_abs = 1e-6 floored |z| at 1e-3,
        # above ALL the region cuts of the finest channels (|z| ~ 1e-4 there): their
        # regions were invisible to forward/grad and got pruned as collinear.
        # eps_quantile=None keeps the scalar eps_abs (fits saved before 2026-09-28).
        self.eps_abs = eps_abs
        self.eps_quantile = eps_quantile
        self.eps_ch = None
        # Read by the SDE (_live_potentials): at a step where a statistic's Gram
        # diagonal on the walkers is below near_empty_tol x its value on the fit data
        # (= 1, stat_scale), its region is essentially empty on the walkers; its
        # coefficient is set to 0 for that step instead of being solved for. With the
        # per-channel floor, regions that are empty early in the transport (e.g. the
        # outer regions of the coarsest channel on noise-like walkers, G_ii ~ 1e-16..
        # 1e-8) are no longer exactly dead; solving for them gave rare drifts ~100x the
        # usual size and a walker blow-up at step 163 (job 268295, 2026-09-28).
        self.near_empty_tol = near_empty_tol
        self.boundary_method = boundary_method
        self.model_criterion = model_criterion
        self.boundary_search_subsample = boundary_search_subsample
        self.boundary_search_iters = boundary_search_iters
        self.boundary_search_grid = boundary_search_grid
        self.pi_active_min = pi_active_min
        self.auto_prune = auto_prune
        self.cond_tol = cond_tol
        self.prune_max_cols = prune_max_cols
        self.verbose = verbose
        if kurt_thresholds is None:
            kurt_thresholds = np.logspace(0, 2, max(self.K - 1, 1)).tolist()
        self.kurt_thresholds = list(kurt_thresholds)

        self.alpha = self.scale = None
        self.cuts = self.sw = self.pi = None
        self.Keff = None
        self.active = None
        self.active_flat = None
        self.stat_scale = None
        self.num_coefficients = self.n_slots * self.J
        self._filters_Kx = None
        # (channel, region) statistics removed after the fit (--drop_stats), e.g. a region
        # statistic that is a near-copy of one in another family (2026-10-01:
        # Scalar_morlet ch 8 region 3 ~ Scalar_psi ch 21 region 3, 1 - |corr| = 6.7e-6)
        self.drop_slots = ()
        # whole channels removed after the fit (--drop_stats NAME:LO-HI:all), as (lo, hi) ranges with hi None =
        # up to the last channel (low-pass included); resolved into drop_slots once the active set is known
        self.drop_channels = ()

    def __call__(self, x, *args):
        return self.forward(x, *args)

    def to(self, device):
        if self.filters is not None:
            self.filters = self.filters.to(device)
        if self.dwt is not None:
            self.dwt.to(device)
        if self._filters_Kx is not None:
            self._filters_Kx = self._filters_Kx.to(device)
        if self.is_fitted:
            self.alpha = self.alpha.to(device)
            self.scale = self.scale.to(device)
            self.cuts = self.cuts.to(device)
            self.sw = self.sw.to(device)
            self.pi = self.pi.to(device)
            self.Keff = self.Keff.to(device)
            self.active = self.active.to(device)
            self.active_flat = self.active_flat.to(device)
            self.stat_scale = self.stat_scale.to(device)
            if self.eps_ch is not None:
                self.eps_ch = self.eps_ch.to(device)
        return self

    # ---- where the coefficients come from: undecimated bank (FFT) or decimated DWT ----
    def _half_filters(self, x):
        """psi_s = (psi(k) + conj psi(-k)) / 2 on the first T/2+1 bins (2026-11-23): x and the pulled-back D are real, so
        Re ifft(psi fft(x)) = irfft(rfft(x) psi_s) and Re ifft(fft(D) conj psi) = irfft(rfft(D) conj psi_s), exactly."""
        key = (x.device, x.dtype)
        c = self.__dict__.get('_half_cache')
        if c is not None and c[0] == key and c[1] is self.filters:
            return c[2]
        f = _as_dtype(self.filters.to(x.device), x.dtype)
        T = f.shape[-1]
        fs = 0.5 * (f + f[..., (-torch.arange(T, device=x.device)) % T].conj())[..., :T // 2 + 1]
        self.__dict__['_half_cache'] = (key, self.filters, fs)
        return fs

    def _coeffs(self, x):
        """Coefficients (B, J, T). DWT: channel j zero-padded beyond dwt.sizes[j]."""
        if self.dwt is not None:
            return self.dwt.analysis(x)
        return torch.fft.irfft(torch.fft.rfft(x) * self._half_filters(x), n=x.shape[-1])

    def _channel_values(self, z, j):
        """Coefficients of channel j (B, n_j), without the DWT padding."""
        return z[:, j, :] if self.dwt is None else z[:, j, :self.dwt.sizes[j]]

    def _position_mean(self, v):
        """Mean over the positions of each channel: (B, J, T) -> (B, J)."""
        if self.dwt is None:
            return v.mean(-1)
        mask = self.dwt.mask.to(device=v.device)
        return (v * mask).sum(-1) / self.dwt.counts.to(device=v.device, dtype=v.dtype)

    def _adjoint(self, D_all, x):
        """Pull d phi / d z (B, n_slots*J, T) back to x: the adjoint of _coeffs, with the
        1/T (1/n_j) of the position mean. For the orthonormal DWT the adjoint is the inverse."""
        if self.dwt is None:
            T = x.shape[-1]
            fs = self._half_filters(x)
            ch = torch.arange(D_all.shape[1], device=x.device) % self.J                # channel of each slot
            return torch.fft.irfft(torch.fft.rfft(D_all) * fs[0, ch].conj(), n=T) / T
        B, J = D_all.shape[0], self.J
        D = D_all.reshape(B, self.n_slots, J, -1)
        out = D_all.new_zeros(B, self.n_slots, J, self.dwt.d)
        for j, n in enumerate(self.dwt.sizes):
            cj = D[:, :, j, :n].reshape(-1, n) / n
            out[:, :, j] = self.dwt.synthesis_channel(cj, j).reshape(B, self.n_slots, -1)
        return out.reshape(B, self.n_slots * J, -1)

    @property
    def mass_windows(self):
        """Window of each mass slot, in slot order (slot K + i is the mass of mass_windows[i])."""
        if not self.tail_mass:
            return []
        return [self.K - 1] + (list(range(1, self.K - 1)) if self.all_masses else [])

    @property
    def n_slots(self):
        """Statistic slots per channel: the K regions, plus the window masses."""
        return self.K + len(self.mass_windows)

    def slot_window(self, k):
        """(window, is_mass) of slot k: slots < K are <w_k |z|^alpha_k>, the others masses."""
        return (k, False) if k < self.K else (self.mass_windows[k - self.K], True)

    @property
    def is_fitted(self):
        return self.alpha is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("must call fit_reference first.")

    @staticmethod
    def _ggd_cdf_abs(t, alpha, scale):
        if t <= 0:
            return 0.0
        if not np.isfinite(t):
            return 1.0
        return float(gammainc(1.0 / alpha, (t / scale) ** alpha))

    @staticmethod
    def _scale_floor(ah, floor_frac=0.02, min_floor=1e-10):
        med = float(np.median(ah)) if ah.size else 0.0
        return max(floor_frac * med, min_floor)

    @classmethod
    def _fit_ggd_truncated(cls, h, lo, hi, alpha_bounds, maxiter=8000,
                           xatol=1e-5, fatol=1e-5, floor_frac=0.02):
        h = np.asarray(h, dtype=float); ah = np.abs(h); n = ah.size
        if n < 5:
            return 1.0, float(np.std(h) + 1e-8)
        scale_floor = cls._scale_floor(ah, floor_frac=floor_frac)

        def neg_ll(lt):
            alpha = float(np.clip(np.exp(lt[0]), *alpha_bounds))
            scale = max(float(np.exp(lt[1])), scale_floor)
            logpdf = (np.log(alpha) - np.log(2.0) - np.log(scale)
                      - gammaln(1.0 / alpha) - (ah / scale) ** alpha)
            mass = cls._ggd_cdf_abs(hi, alpha, scale) - cls._ggd_cdf_abs(lo, alpha, scale)
            if mass < 1e-300:
                return 1e12
            return -np.sum(logpdf) + n * np.log(mass)

        a0 = 1.5
        s0 = max((a0 * np.mean(np.maximum(ah, 1e-300) ** a0)) ** (1.0 / a0), scale_floor)
        res = minimize(neg_ll, x0=[np.log(a0), np.log(s0)], method="Nelder-Mead",
                       options=dict(xatol=xatol, fatol=fatol, maxiter=maxiter))
        a = float(np.clip(np.exp(res.x[0]), *alpha_bounds))
        s = max(float(np.exp(res.x[1])), scale_floor)
        return a, s

    @classmethod
    def _fit_ggd_truncated_fast(cls, h, lo, hi, alpha_bounds, maxiter=250):
        return cls._fit_ggd_truncated(h, lo, hi, alpha_bounds,
                                      maxiter=maxiter, xatol=1e-3, fatol=1e-3)

    @classmethod
    def _region_nll(cls, seg, lo, hi, alpha_bounds, fast):
        fit_fn = cls._fit_ggd_truncated_fast if fast else cls._fit_ggd_truncated
        a, s = fit_fn(seg, lo, hi, alpha_bounds)
        ah = np.abs(seg)
        logpdf = (np.log(a) - np.log(2.0) - np.log(s) - gammaln(1.0 / a) - (ah / s) ** a)
        mass = max(cls._ggd_cdf_abs(hi, a, s) - cls._ggd_cdf_abs(lo, a, s), 1e-300)
        return -np.sum(logpdf) + seg.size * np.log(mass), (a, s)

    @classmethod
    def _composite_nll(cls, h, boundaries, alpha_bounds, fast=True, min_seg=5, min_width=0.0):
        ah = np.abs(h); N = h.size
        edges = [0.0] + list(boundaries) + [np.inf]
        total = 0.0; params = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if np.isfinite(hi) and hi - lo < min_width:
                return np.inf, None                       # inadmissible partition
            seg = h[(ah >= lo) & (ah < hi)]
            if seg.size < min_seg:
                return np.inf, None
            nll, (a, s) = cls._region_nll(seg, lo, hi, alpha_bounds, fast)
            total += nll - seg.size * np.log(seg.size / N)
            params.append((a, s))
        return total, params

    @classmethod
    def _fit_boundaries_k(cls, h, K, alpha_bounds, n_grid=12, n_iters=3,
                          subsample=20000, seed=0, min_width=0.0):
        if K == 1:
            return []
        ah = np.abs(h); rng = np.random.default_rng(seed)
        if ah.size > subsample:
            idx = rng.choice(ah.size, subsample, replace=False)
            h_s, ah_s = h[idx], ah[idx]
        else:
            h_s, ah_s = h, ah
        init_q = np.linspace(0.5, 0.99, K - 1) if K > 2 else np.array([0.9])
        cuts = list(np.quantile(ah_s, init_q))
        for m in range(len(cuts)):                 # admissible start: every window >= min_width
            cuts[m] = max(cuts[m], (cuts[m - 1] if m > 0 else 0.0) + min_width)
        q_los = np.linspace(0.05, 0.80, K - 1); q_his = np.linspace(0.80, 0.995, K - 1)

        def scored(cs):
            nll, _ = cls._composite_nll(h_s, sorted(cs), alpha_bounds, fast=True, min_width=min_width)
            return nll

        for _ in range(n_iters):
            for m in range(K - 1):
                grid = np.quantile(ah_s, np.linspace(q_los[m], q_his[m], n_grid))
                lo_nb = cuts[m - 1] if m > 0 else 0.0
                hi_nb = cuts[m + 1] if m < K - 2 else np.inf
                best_nll, best = np.inf, cuts[m]
                for cand in grid:
                    cand = float(cand)
                    if cand <= max(lo_nb * 1.05, 1e-10):
                        continue
                    if np.isfinite(hi_nb) and cand >= hi_nb * 0.95:
                        continue
                    if cand - lo_nb < min_width or (np.isfinite(hi_nb) and hi_nb - cand < min_width):
                        continue
                    trial = list(cuts); trial[m] = cand
                    nll = scored(trial)
                    if nll < best_nll:
                        best_nll, best = nll, cand
                cuts[m] = best
        return sorted(cuts)

    @staticmethod
    def _channel_kurtosis(h):
        h = np.asarray(h, dtype=float)
        h = h[np.isfinite(h)]
        if h.size < 8:
            return 0.0
        m = h.mean()
        v = ((h - m) ** 2).mean()
        if v <= 1e-300:
            return 0.0
        m4 = ((h - m) ** 4).mean()
        return float(m4 / (v ** 2) - 3.0)

    def _kurtosis_Kmax(self, kurt):
        Kmax = 1
        for t in self.kurt_thresholds:
            if kurt >= t:
                Kmax += 1
        return int(min(Kmax, self.K))

    def _select_model_order(self, h, seed=0, K_max=None, min_width=0.0):
        if K_max is None:
            K_max = self.K
        K_max = max(1, min(int(K_max), self.K))
        N = h.size
        pen = (np.log(N) if self.model_criterion == "bic" else 2.0)
        results = []
        for K in range(1, K_max + 1):
            cuts = self._fit_boundaries_k(
                h, K, self.alpha_bounds, n_grid=self.boundary_search_grid,
                n_iters=self.boundary_search_iters,
                subsample=self.boundary_search_subsample, seed=seed, min_width=min_width)
            nll, params = self._composite_nll(h, cuts, self.alpha_bounds,
                                              fast=False, min_seg=self.min_region_samples,
                                              min_width=min_width)
            if params is None:
                continue
            crit = 2.0 * nll + pen * (4 * K - 2)
            results.append((crit, K, cuts))
        if not results:
            return 1, []
        crit, K, cuts = min(results, key=lambda r: r[0])
        if self.verbose:
            table = "  ".join(f"K{k}={c:.0f}" for c, k, _ in sorted(results, key=lambda r: r[1]))
            print(f"    [model-select] chose K={K}  (Kmax={K_max})  "
                  f"({self.model_criterion.upper()}: {table})")
        return K, cuts

    @staticmethod
    def _merge_narrow(cuts, min_width, keep_last, protect=()):
        """Remove cuts until every inner window [c_{i-1}, c_i] (c_{-1} = 0) is >= min_width.
        A narrow window is merged with the one above; if its upper cut is locked (the last one
        with keep_last, or one of `protect`, the forced tail cuts), with the one below. A narrow
        window between two locked cuts, or a lone window [0, c], is kept: nothing can be removed."""
        cuts = list(cuts)
        protect = set(protect)
        i = 0
        while i < len(cuts):
            lo = cuts[i - 1] if i > 0 else 0.0
            if cuts[i] - lo >= min_width:
                i += 1
            elif (i == len(cuts) - 1 and keep_last) or cuts[i] in protect:
                if i == 0 or cuts[i - 1] in protect:
                    i += 1
                    continue
                del cuts[i - 1]
                i -= 1
            else:
                del cuts[i]
        return cuts

    def _tail_quantiles(self):
        """tail_quantile as a sorted list ([] if None): one forced cut per quantile."""
        if self.tail_quantile is None:
            return []
        q = self.tail_quantile
        return sorted(float(v) for v in (q if isinstance(q, (list, tuple)) else [q]))

    def _embed_slots(self, cuts, ah):
        eps = max(float(np.quantile(ah, 1e-3)), 1e-12) * 1e-2
        n_missing = (self.K - 1) - len(cuts)
        pad = [eps * (i + 1) for i in range(n_missing)]
        slots = pad + list(cuts)
        for i in range(1, len(slots)):
            if slots[i] <= slots[i - 1]:
                slots[i] = slots[i - 1] * 1.5 + eps
        return np.asarray(slots, dtype=float)

    def fit_reference(self, x, boundary_method=None):
        if boundary_method is None:
            boundary_method = self.boundary_method
        self._filters_Kx = None
        z = self._coeffs(x).detach().cpu().numpy()

        J, K = z.shape[1], self.K
        A = np.ones((J, K)); S = np.ones((J, K))
        CUT = np.zeros((J, K - 1)); SW = np.full((J, K - 1), 1e-6)
        PI = np.zeros((J, K)); KEFF = np.ones(J, dtype=int)
        ACT = np.zeros((J, K), dtype=bool)
        FLOOR = np.zeros(J)
        need_noise = self.min_window is not None or self.min_window_bic is not None
        if need_noise and self.dwt is not None:
            unit_noise = np.ones(J)             # orthonormal transform: unit variance for unit white noise
        elif need_noise:
            # z_j = Re(g_j * xi) with g_j = ifft(psi_hat_j) and xi real: Var = sum_s Re g_j(s)^2
            filters = self.filters.to(x.device)
            g = torch.fft.ifft(filters.reshape(-1, filters.shape[-2], filters.shape[-1])[0]).real
            unit_noise = torch.sqrt((g.double() ** 2).sum(-1)).cpu().numpy()

        for j in range(J):
            h = self._channel_values(z, j).reshape(-1); h = h[np.isfinite(h)]; ah = np.abs(h); N = h.size
            w_min = self.min_window_bic * float(unit_noise[j]) if self.min_window_bic is not None else 0.0

            if boundary_method == "auto":
                Keff, cuts = self._select_model_order(h, seed=0, min_width=w_min)
            elif boundary_method == "likelihood":
                Keff = K
                cuts = self._fit_boundaries_k(
                    h, K, self.alpha_bounds, n_grid=self.boundary_search_grid,
                    n_iters=self.boundary_search_iters,
                    subsample=self.boundary_search_subsample, min_width=w_min)
            else:
                qs = np.linspace(0.5, 0.97, K - 1); Keff = K
                cuts = list(np.quantile(ah, qs))

            forced = []
            if self._tail_quantiles() and K > 1:
                # slots are later forced to grow by >= 1.5x: a forced cut closer than that to the
                # previous one would be moved, so it is skipped (2026-10-28: second cut at q.999,
                # skipped on near-Gaussian channels where q.999 / q.99 < 1.5)
                for q in self._tail_quantiles():
                    v = float(np.quantile(ah, q))
                    if not forced or v >= 1.5 * forced[-1]:
                        forced.append(v)
                forced = forced[:K - 1]
                inner = [c for c in sorted(cuts) if 1.5 * c <= forced[0]][:K - 1 - len(forced)]
                cuts = inner + forced
                Keff = len(cuts) + 1

            if self.min_window is not None and len(cuts) > 0:
                cuts = self._merge_narrow(sorted(cuts), self.min_window * float(unit_noise[j]),
                                          keep_last=self.tail_quantile is not None, protect=forced)
                Keff = len(cuts) + 1
            if w_min > 0 and len(cuts) > 0:
                # the likelihood search respected w_min; the forced tail cuts did not: drop the
                # inner cut next to them if it leaves a window narrower than w_min
                cuts = self._merge_narrow(sorted(cuts), w_min, keep_last=self.tail_quantile is not None,
                                          protect=forced)
                Keff = len(cuts) + 1

            KEFF[j] = Keff
            if self.eps_quantile is not None:
                FLOOR[j] = max(float(np.quantile(ah, self.eps_quantile)), 1e-12 * float(ah.max()), 1e-30)
            slots = np.maximum(self._embed_slots(cuts, ah), 1e-8)
            for i in range(1, K - 1):
                slots[i] = max(slots[i], slots[i - 1] * 1.5)
            CUT[j] = slots

            edges = np.concatenate([[0.0], slots])
            for m in range(K - 1):
                width = slots[m] - edges[m]
                SW[j, m] = max(self.trans_frac * max(width, slots[m]), 1e-6)

            # tail windows = the windows above the first forced cut (the outer window alone when
            # there is at most one forced cut: unchanged behaviour)
            k_tail = K - len(forced) if len(forced) > 1 else K - 1
            full_edges = [0.0] + list(slots) + [np.inf]
            counts = []
            for k, (lo, hi) in enumerate(zip(full_edges[:-1], full_edges[1:])):
                seg = h[(ah >= lo) & (ah < hi)]; counts.append(seg.size)
                if seg.size < 5:
                    A[j, k], S[j, k] = 1.0, max(self._scale_floor(ah), 1e-8)
                    continue
                bounds = (self.tail_alpha_bounds
                          if (k >= k_tail and self.tail_alpha_bounds is not None)
                          else self.alpha_bounds)
                floor_frac = 0.02
                if k >= k_tail and self.tail_mass:
                    floor_frac = 1e-12
                    if self.tail_alpha_bounds is None:
                        bounds = (0.05, 8.0)
                A[j, k], S[j, k] = self._fit_ggd_truncated(seg, lo, hi, bounds,
                                                           floor_frac=floor_frac)
            counts = np.asarray(counts, float)
            PI[j] = counts / max(counts.sum(), 1)
            ACT[j] = (PI[j] >= self.pi_active_min) & (counts >= self.min_region_samples)
            if len(forced) > 1:
                # the window above q.999 holds ~1e-3 of the data by construction, at the edge of
                # pi_active_min: forced tail windows are active whenever they have enough points
                ACT[j, k_tail:] = counts[k_tail:] >= self.min_region_samples
            if not ACT[j].any():
                ACT[j, int(np.argmax(counts))] = True

            if self.verbose:
                cut_str = " ".join(f"{c:.4f}" for c in slots)
                pi_str = " ".join(f"{p:.1%}" for p in PI[j])
                a_str = " ".join(f"{a:.2f}" for a in A[j])
                floor = (f"  |z| floor={FLOOR[j]:.2e}" if self.eps_quantile is not None
                         else f"  |z| floor={np.sqrt(self.eps_abs):.2e} (scalar eps_abs)")
                mw = (f"  min window={self.min_window * float(unit_noise[j]):.2e}"
                      if self.min_window is not None else "")
                mw += f"  BIC min window={w_min:.2e}" if w_min > 0 else ""
                print(f"[GGD^{K}][ch {j}] Keff={Keff}  cuts=[{cut_str}]  "
                      f"pi=[{pi_str}]  alpha=[{a_str}]  active={ACT[j].sum()}{floor}{mw}")

        dtype = x.dtype if x.is_floating_point() else torch.float32
        dev = x.device
        mk = lambda M: torch.tensor(M, dtype=dtype, device=dev)
        self.alpha = mk(A); self.scale = mk(S); self.cuts = mk(CUT)
        self.sw = mk(SW); self.pi = mk(PI)
        self.Keff = torch.tensor(KEFF, device=dev)
        if self.tail_mass:                      # mass slot is live iff its region is
            mass_act = [ACT[:, m].copy() for m in self.mass_windows]
            if self.all_masses:                 # ... and is not the first active window
                first = ACT.argmax(1)
                for i, m in enumerate(self.mass_windows):
                    mass_act[i] &= first != m
            ACT = np.concatenate([ACT] + [a[:, None] for a in mass_act], axis=1)
        self.active = torch.tensor(ACT, device=dev)
        self.eps_ch = mk(FLOOR ** 2) if self.eps_quantile is not None else None

        flat = [k * J + j for k in range(self.n_slots) for j in range(J) if ACT[j, k]]
        self.active_flat = torch.tensor(sorted(flat), dtype=torch.long, device=dev)
        self.num_coefficients = int(self.active_flat.numel())

        n_dead = J * self.n_slots - int(ACT.sum())
        if self.verbose:
            print(f"[active] {self.num_coefficients} active statistics "
                  f"({n_dead} dead slots dropped before they can singularize the Gram)")

        self.stat_scale = torch.ones(self.num_coefficients, dtype=dtype, device=dev)
        if self.auto_prune and self.num_coefficients > 1:
            self.prune_collinear(x, cond_tol=self.cond_tol,
                                 max_cols=self.prune_max_cols, verbose=self.verbose)
        self._apply_drop_slots()
        self._compute_stat_scale(x)
        self.plot_fit(x)
        return self

    def _apply_drop_slots(self):
        """Remove the (channel, region) statistics in self.drop_slots from the active set.
        Raises if one is not active (pruned or empty), so a wrong slot never passes silently."""
        J = self.J
        flat = [int(f) for f in self.active_flat.cpu()]
        if getattr(self, 'drop_channels', ()):
            chans = set()
            for lo, hi in self.drop_channels:
                hi = J - 1 if hi is None else hi
                if not 0 <= lo <= hi < J:
                    raise ValueError(f"drop_channels ({lo}, {hi}): this potential has channels 0..{J - 1}")
                chans |= set(range(lo, hi + 1))
            self.drop_slots = tuple(self.drop_slots) + tuple(
                (f % J, f // J) for f in flat if f % J in chans and (f % J, f // J) not in set(self.drop_slots))
            self.drop_channels = ()
            print(f"[drop_channels] channels {sorted(chans)}: all their active statistics are dropped")
        if not self.drop_slots:
            return
        drop = {int(k) * J + int(j) for j, k in self.drop_slots}
        missing = sorted((f % J, f // J) for f in drop - set(flat))
        if missing:
            raise ValueError(f"drop_slots {missing} (channel, region) are not active statistics "
                             f"of this fit; active: {[(f % J, f // J) for f in flat]}")
        dev = self.active_flat.device
        self.active_flat = torch.tensor([f for f in flat if f not in drop], dtype=torch.long, device=dev)
        act = self.active.clone()
        for j, k in self.drop_slots:
            act[int(j), int(k)] = False
        self.active = act
        self.num_coefficients = int(self.active_flat.numel())
        self.stat_scale = torch.ones(self.num_coefficients, dtype=self.stat_scale.dtype, device=dev)
        print(f"[drop_slots] removed (channel, region) {sorted(tuple(map(int, s)) for s in self.drop_slots)}; "
              f"{self.num_coefficients} statistics remain")

    def drop_columns(self, local_idx):
        """Remove active statistics by their POSITION in the active set (column i of forward/grad
        is slot active_flat[i]), after the fit. Used for the cross-family near-copies found by
        codes/gram_preflight.py; same bookkeeping as _apply_drop_slots, but the remaining
        stat_scale entries are kept (a column's scale does not depend on the others)."""
        drop = sorted({int(i) for i in local_idx})
        if not drop:
            return
        n = int(self.active_flat.numel())
        if drop[0] < 0 or drop[-1] >= n:
            raise ValueError(f"drop_columns {drop}: only {n} active statistics")
        J = self.J
        flat = self.active_flat.cpu()
        keep = torch.tensor([i for i in range(n) if i not in set(drop)], dtype=torch.long)
        gone = [int(f) for f in flat[drop]]
        act = self.active.clone()
        for f in gone:
            act[f % J, f // J] = False
        dev = self.active_flat.device
        self.active = act
        self.active_flat = flat[keep].to(dev)
        self.stat_scale = self.stat_scale[keep.to(self.stat_scale.device)]
        self.num_coefficients = int(self.active_flat.numel())
        self.drop_slots = tuple(self.drop_slots) + tuple((f % J, f // J) for f in gone)
        print(f"[drop_columns] removed (channel, region) {[(f % J, f // J) for f in gone]}; "
              f"{self.num_coefficients} statistics remain")

    def _compute_stat_scale(self, x, max_cols=None):
        self.stat_scale = torch.ones(self.num_coefficients, dtype=x.dtype
                                     if x.is_floating_point() else torch.float32,
                                     device=x.device)
        G = self.gram_matrix(x, active_only=True, max_cols=max_cols, _use_scale=False)
        d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
        self.stat_scale = torch.tensor(d, dtype=self.stat_scale.dtype, device=x.device)
        return self

    def fit(self, x, **kw):
        return self.fit_reference(x, **kw)

    def _windows_from_sigmoids(self, g):
        K = self.K
        if K == 1:
            return [torch.ones_like(g[..., 0])]
        ws = [g[..., 0]]
        for k in range(1, K - 1):
            ws.append(g[..., k] - g[..., k - 1])
        ws.append(1.0 - g[..., K - 2])
        return ws

    def _magnitude(self, z):
        """Smoothed |z| = sqrt(z^2 + eps), eps per channel (eps_ch) or the scalar eps_abs."""
        if self.eps_ch is None:
            return torch.sqrt(z ** 2 + self.eps_abs)
        return torch.sqrt(z ** 2 + self.eps_ch.to(device=z.device, dtype=z.dtype)[None, :, None])

    def _all_slot_forward(self, x):
        z = self._coeffs(x)
        az = self._magnitude(z)
        c = self.cuts.to(x.device); s = self.sw.to(x.device)
        g = torch.sigmoid(-(az.unsqueeze(-1) - c[None, :, None, :]) / s[None, :, None, :])
        ws = self._windows_from_sigmoids(g)
        alpha = self.alpha.to(x.device)
        phis = [self._position_mean(ws[k] * az ** alpha[:, k][None, :, None]) for k in range(self.K)]
        phis += [self._position_mean(ws[m]) for m in self.mass_windows]
        return torch.cat(phis, dim=1)

    def forward(self, x, *args):
        self._check_fitted()
        with torch.no_grad():
            phi = self._all_slot_forward(x).index_select(1, self.active_flat.to(x.device))
            if self.stat_scale is not None:
                phi = phi / self.stat_scale.to(x.device)[None, :]
        return phi

    def _all_slot_grad(self, x):
        """Gradient of every candidate slot (B, n_slots*J, T), active or not (fit-time tools)."""
        return self._adjoint(self._all_slot_D(x), x)

    def _active_slot_grad(self, x):
        """Gradient of the active slots only (B, n_active, T), = _all_slot_grad(x)[:, active_flat] (2026-11-17).
        The pointwise derivative d phi / d z is formed for every slot (cheap), but the pull-back to x (one FFT
        pair per slot) is done for the active slots only: the inactive ones were discarded afterwards anyway."""
        idx = self.active_flat.to(x.device)
        if self.dwt is not None:
            return self._all_slot_grad(x).index_select(1, idx)
        D = self._all_slot_D(x).index_select(1, idx)
        T = x.shape[-1]
        return torch.fft.irfft(torch.fft.rfft(D) * self._half_filters(x)[0, idx % self.J].conj(), n=T) / T

    def _grad_projected(self, x, v):
        """sum_s v_s grad phi_s(x), (B, 1, T) (2026-11-17). grad phi_s = L_j(D_s) / stat_scale_s with
        L_j(f) = Re ifft(fft(f) conj(psi_j)) / T linear (real coefficients), j the channel of slot s; so
        sum_s v_s L_j(D_s) / scale_s = sum_j L_j(sum_{s in j} (v_s / scale_s) D_s), and the sum over j is taken in
        Fourier space before one inverse FFT: J forward FFTs + 1 inverse instead of one pair per active slot. Same
        quantity as (grad(x) * v).sum(1), up to round-off. The per-channel sum is a matrix product (deterministic),
        not index_add_ (atomic adds on GPU, run-to-run differences)."""
        with torch.no_grad():
            idx = self.active_flat.to(x.device)
            D = self._all_slot_D(x).index_select(1, idx)                       # (B, n_act, T)
            w = v.to(D.dtype)
            if self.stat_scale is not None:
                w = w / self.stat_scale.to(device=x.device, dtype=D.dtype)
            Wm = torch.zeros(self.J, idx.numel(), dtype=D.dtype, device=x.device)
            Wm[idx % self.J, torch.arange(idx.numel(), device=x.device)] = w   # (J, n_act): channel of each slot
            Dc = torch.matmul(Wm, D)                                            # (B, J, T)
            T = x.shape[-1]
            S = (torch.fft.rfft(Dc) * self._half_filters(x).conj()).sum(1, keepdim=True)   # (B, 1, T/2+1)
            return torch.fft.irfft(S, n=T) / T

    def _all_slot_D(self, x):
        """d phi_slot / d z (B, n_slots*J, T) for every slot, before the pull-back to x (_adjoint)."""
        device = x.device
        z = self._coeffs(x)
        az = self._magnitude(z)
        sz = z / az
        c = self.cuts.to(device); s = self.sw.to(device); alpha = self.alpha.to(device)
        g = torch.sigmoid(-(az.unsqueeze(-1) - c[None, :, None, :]) / s[None, :, None, :])
        dg = -g * (1.0 - g) * sz.unsqueeze(-1) / s[None, :, None, :]
        ws = self._windows_from_sigmoids(g)
        K = self.K
        if K == 1:
            dws = [torch.zeros_like(az)]
        else:
            dws = [dg[..., 0]]
            for k in range(1, K - 1):
                dws.append(dg[..., k] - dg[..., k - 1])
            dws.append(-dg[..., K - 2])
        D = []
        for k in range(K):
            a = alpha[:, k][None, :, None]
            D.append(dws[k] * az ** a + ws[k] * a * z * az ** (a - 2.0))
        D += [dws[m] for m in self.mass_windows]
        return torch.cat(D, dim=1)

    def grad(self, x, v=None, means=None):
        self._check_fitted()
        if v is not None and self.dwt is None:
            return self._grad_projected(x, v)
        with torch.no_grad():
            gc = self._active_slot_grad(x)
            if self.stat_scale is not None:
                gc = gc / self.stat_scale.to(x.device)[None, :, None]
        if v is None:
            return gc
        return (gc * v[None, :, None]).sum(1, keepdim=True)

    def gram_matrix(self, x, active_only=True, max_cols=None, normalize=False,
                    _use_scale=True):
        self._check_fitted()
        device = x.device
        B, _, T = x.shape
        if max_cols is None:
            max_cols = self.prune_max_cols
        idx = self.active_flat.to(device) if active_only else None
        scale = (self.stat_scale.to(device) if (active_only and _use_scale
                 and self.stat_scale is not None) else None)
        rng = np.random.default_rng(0)
        G = None; ncols = 0
        for b in range(B):
            gc = self._all_slot_grad(x[b:b + 1])
            if idx is not None:
                gc = gc.index_select(1, idx)
            g = gc[0]
            if scale is not None:
                g = g / scale[:, None]
            if g.shape[1] > max_cols:
                sel = torch.tensor(rng.choice(g.shape[1], max_cols, replace=False),
                                   device=device)
                g = g.index_select(1, sel)
            Gb = (g @ g.T).double().cpu().numpy()
            G = Gb if G is None else G + Gb
            ncols += g.shape[1]
        G = G / max(B, 1)
        if normalize:
            d = np.sqrt(np.clip(np.diag(G), 1e-300, None))
            G = G / np.outer(d, d)
        return G

    def _active_gradient_matrix(self, x, max_rows):
        device = x.device
        idx = self.active_flat.to(device)
        mats = []
        for b in range(x.shape[0]):
            gc = self._all_slot_grad(x[b:b + 1]).index_select(1, idx)[0]
            mats.append(gc.T)
        M = torch.cat(mats, 0)
        if M.shape[0] > max_rows:
            g = torch.Generator(device=device).manual_seed(0)
            sel = torch.randperm(M.shape[0], generator=g, device=device)[:max_rows]
            M = M.index_select(0, sel)
        # float64 BEFORE the column norm (2026-10-27): a gradient column of ~1e-23 (narrow core
        # window, alpha ~6) has a float32 norm that underflows to 0, so the division by
        # (0 + 1e-30) inflated it ~1e30 and the pivoted QR dropped every other column
        # (check_fit_offline, Scalar_psi Q=3 at n1=3000: 85 -> 1 statistic).
        M = M.double()
        M = M / (M.norm(dim=0, keepdim=True) + 1e-30)
        return M.cpu().numpy()

    
    def prune_collinear(self, x, cond_tol=1e-6, max_cols=20000, verbose=True,
                        block_by_channel=True, diagnose=True):
        
        from scipy.linalg import qr
        self._check_fitted()
        old_flat = self.active_flat.cpu().numpy()
        J = self.J

        if block_by_channel:
            # Two-stage pruning: first remove near-duplicate *regions within
            # the same channel* (legitimate — two GGD regions that ended up
            # fitting almost the same shape). Cross-channel collinearity is
            # handled separately with a much looser tolerance, since in
            # turbulence it's largely an artifact of intermittent extreme
            # events co-occurring across scales, not genuine redundancy of
            # the underlying potentials, and shouldn't be allowed to wipe
            # out distinct channels.
            M_full = self._active_gradient_matrix(x, max_rows=max_cols)
            keep_mask = np.zeros(len(old_flat), dtype=bool)

            for j in range(J):
                col_idx = np.where(old_flat % J == j)[0]
                if col_idx.size == 0:
                    continue
                if col_idx.size == 1:
                    keep_mask[col_idx] = True
                    continue
                Mj = M_full[:, col_idx]
                _, Rj, Pj = qr(Mj, mode="economic", pivoting=True)
                adj = np.abs(np.diag(Rj))
                thrj = np.sqrt(cond_tol) * (adj[0] if adj.size else 0.0)
                kept_local = Pj[adj > thrj]
                keep_mask[col_idx[kept_local]] = True

            surviving_idx = np.where(keep_mask)[0]
            M = M_full[:, surviving_idx]

            # Loose second pass across channels: only drop columns that are
            # essentially exact duplicates, not merely correlated because of
            # shared extreme events.
            cross_cond_tol = min(cond_tol, 1e-10)
            _, R, P = qr(M, mode="economic", pivoting=True)
            absdiag = np.abs(np.diag(R))
            if diagnose and absdiag.size:
                ratios = absdiag / absdiag[0]
                print(f"[prune-diagnostic] R-diagonal ratios (sorted desc), "
                    f"first 10 / last 10:\n  head={np.round(ratios[:10], 4)}\n"
                    f"  tail={np.round(ratios[-10:], 8)}")
            thr = np.sqrt(cross_cond_tol) * (absdiag[0] if absdiag.size else 0.0)
            keep = np.sort(surviving_idx[P[absdiag > thr]])
        else:
            M = self._active_gradient_matrix(x, max_rows=max_cols)
            _, R, P = qr(M, mode="economic", pivoting=True)
            absdiag = np.abs(np.diag(R))
            if diagnose and absdiag.size:
                ratios = absdiag / absdiag[0]
                print(f"[prune-diagnostic] R-diagonal ratios (sorted desc), "
                    f"first 10 / last 10:\n  head={np.round(ratios[:10], 4)}\n"
                    f"  tail={np.round(ratios[-10:], 8)}")
            thr = np.sqrt(cond_tol) * (absdiag[0] if absdiag.size else 0.0)
            keep = np.sort(P[absdiag > thr])

        old_flat_arr = self.active_flat.cpu().numpy()
        new_flat = old_flat_arr[keep]
        dropped = len(old_flat_arr) - len(new_flat)

        Jn, K = self.J, self.n_slots
        act = np.zeros((Jn, K), dtype=bool)
        for f in new_flat:
            act[int(f % Jn), int(f // Jn)] = True
        dev = self.active_flat.device
        self.active = torch.tensor(act, device=dev)
        self.active_flat = torch.tensor(np.sort(new_flat), dtype=torch.long, device=dev)
        self.num_coefficients = int(self.active_flat.numel())
        self.stat_scale = torch.ones(self.num_coefficients, dtype=torch.float32, device=dev)
        self._compute_stat_scale(x, max_cols=max_cols)
        if verbose:
            print(f"[prune] dropped {dropped} near-collinear statistic(s); "
                f"{self.num_coefficients} remain")
        self._last_prune_absdiag = absdiag
        return self


    def report_conditioning(self, x, max_cols=20000):
        self._check_fitted()
        def stats(G):
            w = np.linalg.eigvalsh(G)
            w = np.clip(w, 0, None)
            wmax = w.max() if w.size else 0.0
            rank = int((w > 1e-10 * max(wmax, 1e-300)).sum())
            cond = (wmax / w[w > 1e-10 * max(wmax, 1e-300)].min()
                    if rank else np.inf)
            return G.shape[0], rank, w.min(), cond
        Gp = self.gram_matrix(x, active_only=False, max_cols=max_cols)
        Ga = self.gram_matrix(x, active_only=True, max_cols=max_cols)
        for name, G in [("padded (all K slots)", Gp), ("active (exposed)", Ga)]:
            n, r, lmin, cond = stats(G)
            flag = "  <-- SINGULAR" if r < n else "  ok"
            print(f"  {name:24s}: dim={n:3d} rank={r:3d} "
                  f"lambda_min={lmin:.2e} cond={cond:.2e}{flag}")

    def save_fixed_parameters(self, filename):
        self._check_fitted()
        torch.save(dict(
            alpha=self.alpha.cpu(), scale=self.scale.cpu(), cuts=self.cuts.cpu(),
            sw=self.sw.cpu(), pi=self.pi.cpu(), Keff=self.Keff.cpu(),
            active=self.active.cpu(), active_flat=self.active_flat.cpu(), stat_scale=self.stat_scale.cpu(),
            num_coefficients=self.num_coefficients, K=self.K, J=self.J,
            tail_mass=self.tail_mass, all_masses=self.all_masses, min_window=self.min_window,
            min_window_bic=self.min_window_bic,
            trans_frac=self.trans_frac, eps_abs=self.eps_abs,
            eps_quantile=self.eps_quantile,
            eps_ch=None if self.eps_ch is None else self.eps_ch.cpu(),
            near_empty_tol=self.near_empty_tol,
            drop_slots=[tuple(map(int, s)) for s in self.drop_slots],
            alpha_bounds=self.alpha_bounds, min_region_samples=self.min_region_samples,
            boundary_method=self.boundary_method, model_criterion=self.model_criterion,
            pi_active_min=self.pi_active_min, cond_tol=self.cond_tol,
            filters_shape=None if self.filters is None else tuple(self.filters.shape),
            dwt=None if self.dwt is None else (self.dwt.order, self.dwt.levels, self.dwt.d)), filename)

    @classmethod
    def load_fixed_parameters(cls, filename, filters, map_location=None, verbose=False):
        d = torch.load(filename, map_location=map_location)
        dwt = None
        if d.get("dwt") is not None:            # decimated-DWT fit: rebuilt from its saved spec, no filter bank
            from ortho_wavelet.periodic_dwt import PeriodicDWT
            dwt, filters = PeriodicDWT(*d["dwt"]), None
        elif tuple(filters.shape) != tuple(d["filters_shape"]):
            raise ValueError(
                f"filters shape {tuple(filters.shape)} does not match the shape "
                f"the potential was fit with {tuple(d['filters_shape'])}; "
                f"this state cannot be reused with these filters.")
        obj = cls(filters, num_regions=d["K"], trans_frac=d["trans_frac"],
                   alpha_bounds=d.get("alpha_bounds", (0.2, 8.0)),
                   min_region_samples=d.get("min_region_samples", 30),
                   eps_abs=d["eps_abs"],
                   eps_quantile=d.get("eps_quantile"),     # absent in pre-2026-09-28 fits
                   near_empty_tol=d.get("near_empty_tol", 0.0),   # old fits: exact-zero test only
                   boundary_method=d.get("boundary_method", "auto"),
                   model_criterion=d.get("model_criterion", "bic"),
                   pi_active_min=d.get("pi_active_min", 1e-3),
                   cond_tol=d.get("cond_tol", 1e-6),
                   tail_mass=d.get("tail_mass", False),
                   all_masses=d.get("all_masses", False),
                   min_window=d.get("min_window"),
                   min_window_bic=d.get("min_window_bic"),
                   dwt=dwt,
                   verbose=verbose)
        obj.alpha = d["alpha"]; obj.scale = d["scale"]; obj.cuts = d["cuts"]
        obj.sw = d["sw"]; obj.pi = d["pi"]; obj.Keff = d["Keff"]
        obj.active = d["active"]; obj.active_flat = d["active_flat"]
        obj.stat_scale = d["stat_scale"]; obj.num_coefficients = d["num_coefficients"]
        obj.eps_ch = d.get("eps_ch")        # None for old fits -> scalar eps_abs, as they ran
        obj.drop_slots = tuple(tuple(s) for s in d.get("drop_slots", ()))   # provenance only:
        obj.J = d["J"]                      # active_flat already excludes them
        return obj

    @classmethod
    def _logpdf_trunc(cls, xv, alpha, scale, lo, hi, pi):
        mass = max(cls._ggd_cdf_abs(hi, alpha, scale) - cls._ggd_cdf_abs(lo, alpha, scale), 1e-300)
        lp = (np.log(alpha) - np.log(2.0) - np.log(scale) - gammaln(1.0 / alpha)
              - (np.abs(xv) / scale) ** alpha - np.log(mass) + np.log(max(pi, 1e-300)))
        return np.clip(lp, -500.0, 500.0)
    
    def _draw_fitted_windows(self, ax, j, xmax, n_grid=800):
        """Draw the per-region fitted density curves ('windows') for channel
        j onto an existing Axes, over the range [-xmax, xmax]. Shared by
        plot_fit and compare_channel so both draw identical region curves
        instead of each keeping their own copy of this logic."""
        A = self.alpha.cpu().numpy(); S = self.scale.cpu().numpy()
        C = self.cuts.cpu().numpy(); P = self.pi.cpu().numpy()
        K = A.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 0.9, K))
        edges = [0.0] + list(C[j]) + [xmax]
        for k in range(K):
            if P[j, k] < 1e-4:      # collapsed sliver, nothing to draw
                continue
            lo, hi = edges[k], edges[k + 1]
            xp = np.linspace(max(lo, 1e-6), hi, n_grid)
            lp = self._logpdf_trunc(xp, A[j, k], S[j, k], lo,
                                    (np.inf if k == K - 1 else hi), P[j, k])
            xx = np.concatenate([-xp[::-1], xp])
            yy = np.exp(np.concatenate([lp[::-1], lp]))
            ax.plot(xx, yy, lw=2, color=colors[k],
                    label=f"r{k} a={A[j,k]:.2f} sc={S[j,k]:.3f} pi={P[j,k]:.1%}")
        for c in C[j]:
            ax.axvline(c, color="k", ls=":", lw=0.8, alpha=0.4)
            ax.axvline(-c, color="k", ls=":", lw=0.8, alpha=0.4)

    def plot_fit(self, x, n_grid=800, log_scale=True, fit_if_needed=True, j=None, ax=None):
        """Plot the fitted density regions ('windows') over the real
        histogram, per channel.

        With no `j`, behaves exactly as before: every channel, each in its
        own new figure, shown automatically. Pass `j` to restrict to one
        channel; combine with `ax` to draw into an existing Axes (e.g. one
        cell of a caller-built subplot grid) instead of creating a new
        figure — `ax` requires `j` since one Axes can't hold multiple
        channels.
        """
        if ax is not None and j is None:
            raise ValueError("ax can only be used together with a specific j")
        if fit_if_needed and not self.is_fitted:
            self.fit_reference(x)
        self._check_fitted()
        wt = self._coeffs(x)
        J = self.alpha.shape[0]
        channels = range(J) if j is None else [j]
        for jj in channels:
            h = self._channel_values(wt, jj).detach().cpu().flatten().numpy()
            h = h[np.isfinite(h)]
            if h.size == 0:
                continue
            xmax = float(np.abs(h).max()) * 1.02
            fig = None
            this_ax = ax
            if this_ax is None:
                fig, this_ax = plt.subplots(figsize=(9, 4))
            this_ax.hist(h, bins=100, density=True, log=log_scale, alpha=0.5,
                    color="steelblue", label="data")
            self._draw_fitted_windows(this_ax, jj, xmax, n_grid)
            this_ax.set_xlabel("Coefficient value")
            this_ax.set_ylabel("Log density" if log_scale else "Density")
            this_ax.set_title(f"channel {jj}  (Keff={int(self.Keff[jj])})")
            this_ax.legend(fontsize=7, loc="upper right")
            if fig is not None:
                plt.show()
    # ===================== analytical shape extraction ================
    def analytical_regions(self, j, pi_floor=1e-4):
        """Active density regions defining p_j(z). Each: k, alpha, scale, lo, hi, pi
        (pi renormalised over kept regions)."""
        self._check_fitted()
        A = self.alpha[j].detach().cpu().numpy(); S = self.scale[j].detach().cpu().numpy()
        C = self.cuts[j].detach().cpu().numpy();  P = self.pi[j].detach().cpu().numpy()
        edges = np.concatenate([[0.0], C, [np.inf]])
        regs = [dict(k=int(k), alpha=float(A[k]), scale=float(S[k]),
                     lo=float(edges[k]), hi=float(edges[k + 1]), pi=float(P[k]))
                for k in range(self.K) if P[k] >= pi_floor]
        tot = sum(r["pi"] for r in regs) or 1.0
        for r in regs:
            r["pi"] /= tot
        return regs
    def analytical_pdf(self, xv, j, pi_floor=1e-4):
        """Evaluate the fitted composite density p_j(z) at signed points xv."""
        xv = np.asarray(xv, dtype=float); axv = np.abs(xv); out = np.zeros_like(xv)
        for r in self.analytical_regions(j, pi_floor):
            m = (axv >= r["lo"]) & (axv < r["hi"])
            if m.any():
                out[m] = np.exp(self._logpdf_trunc(
                    xv[m], r["alpha"], r["scale"], r["lo"], r["hi"], r["pi"]))
        return out
    def analytical_shape(self, x=None, j=None, n_grid=1000, pi_floor=1e-4):
        """Per-channel {regions, grid, pdf}. j=None -> dict over all channels.
        If x given, grid spans that channel's observed |z| range."""
        self._check_fitted()
        wt = None
        if x is not None:
            wt = self._coeffs(x).detach().cpu().numpy()
        def zmax_of(jj):
            if wt is not None:
                h = self._channel_values(wt, jj).reshape(-1); h = h[np.isfinite(h)]
                return float(np.abs(h).max()) * 1.02 if h.size else 1.0
            regs = self.analytical_regions(jj, pi_floor)
            fin = [r["hi"] for r in regs if np.isfinite(r["hi"])]
            return max(fin + [8.0 * max((r["scale"] for r in regs), default=1.0)])
        def one(jj):
            zmax = zmax_of(jj); grid = np.linspace(-zmax, zmax, n_grid)
            return dict(regions=self.analytical_regions(jj, pi_floor),
                        grid=grid, pdf=self.analytical_pdf(grid, jj, pi_floor))
        return one(j) if j is not None else {jj: one(jj) for jj in range(self.J)}
    
    # ===================== sampling from the fitted density ===========
    def sample_channel(self, n, j, pi_floor=1e-4, seed=None):
        """n i.i.d. samples from p_j(z): region ~ Multinomial(pi), |z| via
        inverse-CDF of the truncated GGD, random +/- sign."""
        from scipy.special import gammaincinv
        regs = self.analytical_regions(j, pi_floor)
        if not regs:
            return np.zeros(0)
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(n, [r["pi"] for r in regs]); chunks = []
        for r, c in zip(regs, counts):
            if c == 0:
                continue
            a, s, lo, hi = r["alpha"], r["scale"], r["lo"], r["hi"]
            Flo = self._ggd_cdf_abs(lo, a, s); Fhi = self._ggd_cdf_abs(hi, a, s)  # 1.0 if inf
            F = np.clip(Flo + rng.uniform(size=c) * (Fhi - Flo), 0.0, 1.0 - 1e-15)
            mag = s * gammaincinv(1.0 / a, F) ** (1.0 / a)
            chunks.append(rng.choice((-1.0, 1.0), size=c) * mag)
        out = np.concatenate(chunks); rng.shuffle(out)
        return out
    def sample_all_channels(self, n_per_channel, pi_floor=1e-4, seed=0):
        self._check_fitted()
        return {j: self.sample_channel(n_per_channel, j, pi_floor, seed + j)
                for j in range(self.J)}
    def compare_channel(self, x, j, n_samples=None, bins=200, log_scale=True,
                        pi_floor=1e-4, seed=0, ax=None, n_grid=1000):
        """Overlay real-coeff histogram, generated-coeff histogram (data
        sampled from the fit, in a different color from the real data), and
        the single composite analytical-shape curve (the fitted density,
        summed across regions — not the per-region 'windows'; see plot_fit
        for those).

        Pass `ax` to draw into an existing Axes (e.g. one cell of a
        caller-built subplot grid) instead of creating a new figure.
        """
        self._check_fitted()
        wt = self._coeffs(x)
        h = self._channel_values(wt, j).detach().cpu().reshape(-1).numpy(); h = h[np.isfinite(h)]
        n_samples = h.size if n_samples is None else n_samples
        g = self.sample_channel(n_samples, j, pi_floor, seed)
        xmax = float(np.abs(h).max()) * 1.02
        edges = np.linspace(-xmax, xmax, bins + 1); grid = np.linspace(-xmax, xmax, n_grid)
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(h, bins=edges, density=True, log=log_scale, alpha=0.5,
                color="steelblue", label="data (real)")
        ax.hist(g, bins=edges, density=True, log=log_scale, histtype="step",
                lw=1.6, color="crimson", label="generated")
        ax.plot(grid, np.clip(self.analytical_pdf(grid, j, pi_floor), 1e-12, None),
                "k-", lw=1.2, label="analytic pdf")
        for r in self.analytical_regions(j, pi_floor):
            if np.isfinite(r["hi"]):
                ax.axvline(r["hi"], color="k", ls=":", lw=0.7, alpha=0.4)
                ax.axvline(-r["hi"], color="k", ls=":", lw=0.7, alpha=0.4)
        ax.set_xlabel("coefficient value"); ax.set_ylabel("density")
        ax.set_title(f"channel {j}: data vs generated vs analytic shape")
        ax.legend(fontsize=7, loc="upper right")
        if fig is not None:
            plt.show()
        return fig, ax

    @classmethod
    def fit_and_compare(cls, x, M, J, Q, device, ncols=4):
        """Build filters for (M, J, Q), fit a fresh instance against x, and
        show every channel's real-vs-generated-vs-analytic-shape comparison
        in one grid. Returns the fitted model.
        """
        filters = return_Filters(M, J, Q, device=device, include_phi=False)
        model = cls(filters)
        model.fit_reference(x)
        n_ch = filters.shape[1]
        nrows = math.ceil(n_ch / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()
        for i in range(n_ch):
            model.compare_channel(x, j=i, ax=axes[i])
        for j in range(n_ch, len(axes)):
            axes[j].axis("off")
        plt.suptitle(f"Q={Q}: fit vs generated vs analytic shape", fontsize=20)
        plt.tight_layout()
        plt.show()
        return model

    # =========================== reporting ============================

    def summary(self):
        self._check_fitted()
        A = self.alpha.cpu().numpy(); S = self.scale.cpu().numpy()
        C = self.cuts.cpu().numpy(); P = self.pi.cpu().numpy()
        Ke = self.Keff.cpu().numpy(); AC = self.active.cpu().numpy()
        J, K = A.shape
        print(f"{'Ch':>3} {'Keff':>4} {'act':>3} | per-region  alpha (scale) [*=active]")
        print("-" * 70)
        for j in range(J):
            cells = " ".join(
                f"{'*' if AC[j,k] else ' '}{A[j,k]:.2f}({P[j,k]:.0%})" for k in range(K))
            print(f"{j:>3d} {Ke[j]:>4d} {AC[j].sum():>3d} | {cells}")

from potentials.potentials_classes.hermite_norm import Hermite_norm 

