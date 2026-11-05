import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from scipy.ndimage import gaussian_filter

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
        self.num_coefficients = None
        super().__init__()

    def forward(self,x):
        pass

    def grad(self,x):
        pass



    ## ----------------------------------------------------- Potentials definitions -----------------------------------------------------    


    # ----- Scattering potentials -----


class Scattering_First_Order_2d(Potential):
    def __init__(self, filters):
        super().__init__()
        self.filters = filters
        self.num_coefficients = filters.shape[1]

    def forward(self,x):        
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft2(filters*torch.fft.fft2(x))

        return abs_eps(x_filtered).mean(-1).mean(-1)

    def grad(self, x, v=None, precomputed=None, means=None):        
        filters = self.filters.to(x.device)
        
        if precomputed==None:
            x_fourier = torch.fft.fft2(x)
            x_filtered = torch.fft.ifft2(filters*x_fourier)
            x_filtered_abs = abs_eps(x_filtered)
            x_filtered_over_abs = x_filtered/x_filtered_abs
        else:
            x_filtered_over_abs = precomputed[0]['x_filtered_over_abs'][precomputed[1]:precomputed[2]]

        output = torch.real(torch.fft.ifft2(torch.fft.fft2(x_filtered_over_abs)*filters))


        if v==None:
            return output/(x.shape[-2]*x.shape[-1])
        else:
            return (output*v[None,:,None,None]).sum(1)[:,None]/(x.shape[-2]*x.shape[-1])

class Scattering_Second_Order_2d(Potential):
    def __init__(self, filters):
        super().__init__()
        #self.filters = filters

        self.num_coefficients = filters.shape[1] + 1
        
        # Define HF filter
        size = filters.shape[-1]  # e.g., 100x100 tensor

        radius = int(size*85/128)  # radius of the disk
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        center_y, center_x = size // 2, size // 2
        distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        filter_HF = torch.fft.fftshift(torch.from_numpy(gaussian_filter((distance >= radius).float(), sigma=1))[None, None])

        self.filters = torch.cat([filter_HF.to(filters.device), filters], dim=1)

    def forward(self, x):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft2(filters*torch.fft.fft2(x))
        return (x_filtered*x_filtered.conj()).real.mean(-1).mean(-1)

    def grad(self, x, v=None, precomputed=None, means=None):
        filters = self.filters.to(x.device)
        
        if precomputed==None:
            x_filtered_2 = torch.fft.ifft2(torch.fft.fft2(x)*filters**2)
        else:
            x_filtered_2 = precomputed[0]['x_filtered_2'][precomputed[1]:precomputed[2]]

        output = x_filtered_2.real.reshape(x_filtered_2.shape[:1]+(-1,x.shape[-2],x.shape[-1]))

        if means != None:
            mean_correction = torch.fft.ifft2(filters[0,-1]*torch.fft.fft2(torch.ones((x.shape[-2], x.shape[-1]), device=x.device))).real
            mean_correction = means['x_filtered_low_freq_mean']*mean_correction[None]

        if v==None:
            return 2*output/(x.shape[-2]*x.shape[-1])
        else:
            return (2*output*v[None,:,None,None]).sum(1)[:,None]/(x.shape[-2]*x.shape[-1])

class Scattering_Third_Order_Real_2d(Potential):
    def __init__(self, J, L, filters):
        super().__init__()
        self.J = J
        self.L = L
        self.filters = filters
        self.num_coefficients = int(J*L*(L*(1+J)/2+1))

    def forward(self, x):
        filters = self.filters.to(x.device)
        
        x_filtered = torch.fft.ifft2(filters*torch.fft.fft2(x))
        x_filtered_abs = abs_eps(x_filtered)

        output = torch.real(x_filtered[:, None].conj() * torch.fft.ifft2(filters[:, None] * torch.fft.fft2(x_filtered_abs)[:, :, None]))

        output = output.mean(-1).mean(-1)

        #print(output.shape)
        
        indices = indices_third_order(self.J,self.L).long()
        
        output = output[:, indices[0], indices[1]]

        return output

    def grad(self, x, v=None, precomputed=None, means=None):

        filters = self.filters.to(x.device)
        
        number_filters = self.filters.shape[1]
        if precomputed==None:
            x_fourier = torch.fft.fft2(x)
            x_filtered_2 = torch.fft.ifft2(x_fourier*filters**2)
            x_filtered = torch.fft.ifft2(filters*x_fourier)
            x_filtered_abs = abs_eps(x_filtered)
            x_filtered_over_abs = x_filtered/x_filtered_abs
        else:
            x_filtered = precomputed[0]['x_filtered'][precomputed[1]:precomputed[2]]
            x_filtered_2 = precomputed[0]['x_filtered_2'][precomputed[1]:precomputed[2]]
            x_filtered_abs = precomputed[0]['x_filtered_abs'][precomputed[1]:precomputed[2]]
            x_filtered_over_abs = precomputed[0]['x_filtered_over_abs'][precomputed[1]:precomputed[2]]


        indices = indices_third_order(self.J,self.L)
        


        if v!=None:
            m = torch.zeros((number_filters-1, number_filters)).to(x.device)
            m[indices[0], indices[1]] = v
            
            result_conv_phase = x_filtered_over_abs[:,:-1]*torch.einsum('ij, aibc -> ajbc', m.T, torch.real(x_filtered_2))
            result_conv_phase = torch.fft.ifft2(filters[:,:-1]*torch.fft.fft2(result_conv_phase))
            
            x_filtered_abs_weighted = torch.einsum('ij, aibc -> ajbc', m, x_filtered_abs[:,:-1])
            result_conv_modulus = torch.fft.ifft2(filters**2*torch.fft.fft2(x_filtered_abs_weighted))
            
            return (result_conv_phase.sum(1) + result_conv_modulus.sum(1)).real[:,None]/(x.shape[-2]*x.shape[-1])
        
        result_conv_phase = x_filtered_over_abs[:,:-1,None]*torch.real(x_filtered_2)[:,None]
        result_conv_phase = torch.fft.ifft2(filters[:,:-1,None]*torch.fft.fft2(result_conv_phase))

        result_conv_modulus = torch.fft.ifft2(filters[:,None]**2*torch.fft.fft2(x_filtered_abs)[:,:-1,None])

        output = (result_conv_phase + result_conv_modulus).real

        output = output[:, indices[0], indices[1]]/(x.shape[-2]*x.shape[-1])

        return output

class Scattering_Fourth_Order_Real_2d(Potential):
    def __init__(self, J, L, filters):
        super().__init__()
        self.J = J
        self.L = L
        self.filters = filters

        self.num_coefficients = int(L*J/12*((2*J**2+1)*L**2+(3*J*(L+3)+3)*L+6))

    def forward(self, x):

        filters = self.filters.to(x.device)
        
        x_filtered = torch.fft.ifft2(filters*torch.fft.fft2(x))
        x_filtered_abs = abs_eps(x_filtered)
        
        W_Wx = torch.fft.ifft2(filters[:, :, None] * torch.fft.fft2(x_filtered_abs)[:, None])
        output = torch.real(W_Wx[:, :, :, None] * W_Wx[:, :, None].conj())
        output = output.mean(-1).mean(-1)
        output = output.permute(0, 3, 2, 1)
        
        indices = indices_fourth_order(self.J,self.L).long()
        
        output = output[:, indices[0], indices[1], indices[2]]
        
        output = output.reshape(x.shape[0], indices.shape[1])
        return output

    def grad(self, x, v=None, precomputed=None, means=None):

        filters = self.filters.to(x.device)
        
        number_filters = self.filters.shape[1]

        if precomputed==None:
            x_filtered_no_LF = torch.fft.ifft2(filters*torch.fft.fft2(x))[:,:-1] # (B,J*L,M,N)
            x_filtered_abs_no_LF = abs_eps(x_filtered_no_LF) # (B,J*L,M,N)
            x_filtered_over_abs_no_LF = (x_filtered_no_LF/x_filtered_abs_no_LF)

        else:
            x_filtered_no_LF = precomputed[0]['x_filtered'][precomputed[1]:precomputed[2]][:,:-1] # (B,J*L,M,N)
            x_filtered_abs_no_LF = precomputed[0]['x_filtered_abs'][precomputed[1]:precomputed[2]][:,:-1] # (B,J*L,M,N)
            x_filtered_over_abs_no_LF = precomputed[0]['x_filtered_over_abs'][precomputed[1]:precomputed[2]][:,:-1] # (B,J*L,M,N)

        x_filtered_abs_no_LF_filtered_2 = torch.fft.ifft2((torch.fft.fft2(x_filtered_abs_no_LF)[:,:,None]*filters**2)) # (B,J*L,J*L+1,M,N) #no 2*, x_filtered_abs_no_LF
        
        indices = indices_fourth_order(self.J,self.L)



        if v != None:
            m = torch.zeros((number_filters-1, number_filters-1, number_filters)).to(x.device)
            m[indices[0], indices[1], indices[2]] = v
            m = m+torch.transpose(m,0,1)
            
            intermediate_output = x_filtered_over_abs_no_LF*torch.einsum('ijk, ajkbc -> aibc', m, torch.real(x_filtered_abs_no_LF_filtered_2))
            return torch.fft.ifft2(filters[:,:-1]*torch.fft.fft2(intermediate_output)).real.sum(1)[:,None]/(x.shape[-2]*x.shape[-1])
            
        intermediate_output = x_filtered_over_abs_no_LF[:,:,None,None]*torch.real(x_filtered_abs_no_LF_filtered_2)[:,None] #x_filtered_over_abs_no_LF
        output = torch.fft.ifft2(filters[:,:-1,None,None]*torch.fft.fft2(intermediate_output)).real

        output = output + torch.transpose(output, 1,2)
        output = output[:, indices[0], indices[1], indices[2]].reshape(x.shape[0], indices.shape[1], x.shape[-2], x.shape[-1])/(x.shape[-2]*x.shape[-1])

        return output

# ----------------------------------------------------------------------------------------------------------------------
# 2026-11-05 (cosmic_web): confinement and imaginary parts, ported from the 1D classes
# (L2p_norm, Scattering_Third_Order_Imag_1d, Scattering_Fourth_Order_Imag_1d). Same conventions as the Real 2D classes
# above: the 2D Morlet bank is real in Fourier space (so conj(filters) = filters and |filters|^2 = filters**2), the
# last channel is the low-pass. Gradients are analytic and checked against autograd (cosmic_web/test_potentials_2d.py).
# ----------------------------------------------------------------------------------------------------------------------

class L2p_norm_2d(Potential):
    """Confinement mean_u |W_c x|^(2p) per filter channel c (low-pass included); p = 3 is the 'L_6' of the 1D runs."""
    def __init__(self, p, filters):
        super().__init__()
        self.p = p
        self.filters = filters
        self.num_coefficients = filters.shape[1]

    def forward(self, x):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft2(filters*torch.fft.fft2(x))
        return (torch.abs(x_filtered)**(2*self.p)).mean(-1).mean(-1)

    def grad(self, x, v=None, precomputed=None, means=None):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft2(filters*torch.fft.fft2(x))
        output = x_filtered*(2*self.p)*(torch.abs(x_filtered)**(2*self.p-2))
        output = torch.fft.ifft2(filters.conj()*torch.fft.fft2(output)).real
        if v is None:
            return output/(x.shape[-2]*x.shape[-1])
        return (output*v[None, :, None, None]).sum(1)[:, None]/(x.shape[-2]*x.shape[-1])


class Scattering_Third_Order_Imag_2d(Potential):
    """Im[conj(W_j x) (psi_j * |W_k x|)], the imaginary counterpart of Scattering_Third_Order_Real_2d (same index
    layout). include_lowpass=False (default here) drops the J*L columns whose phase filter j is the low-pass: for a
    real, symmetric low-pass the Imag part is identically zero (see Scattering_Third_Order_Imag_1d)."""
    def __init__(self, J, L, filters, include_lowpass=False):
        super().__init__()
        self.J = J
        self.L = L
        self.filters = filters
        self.include_lowpass = include_lowpass
        self.num_coefficients = self._indices().shape[1]

    def _indices(self):
        indices = indices_third_order(self.J, self.L)
        if not self.include_lowpass:
            indices = indices[:, indices[1] < self.J*self.L]
        return indices

    def forward(self, x):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft2(filters*torch.fft.fft2(x))
        x_filtered_abs = abs_eps(x_filtered)
        output = torch.imag(x_filtered[:, None].conj() * torch.fft.ifft2(filters[:, None] * torch.fft.fft2(x_filtered_abs)[:, :, None]))
        output = output.mean(-1).mean(-1)
        indices = self._indices().long()
        return output[:, indices[0], indices[1]]

    def grad(self, x, v=None, precomputed=None, means=None):
        filters = self.filters.to(x.device)
        number_filters = filters.shape[1]
        x_fourier = torch.fft.fft2(x)
        filters_abs2 = filters*filters.conj()
        # |psi^|^2 at -omega: the Imag part is not symmetric in omega
        filters_abs2_flip = torch.roll(torch.flip(filters_abs2, [-2, -1]), (1, 1), (-2, -1))
        x_filtered_2 = torch.fft.ifft2(x_fourier*filters_abs2)
        x_filtered = torch.fft.ifft2(filters*x_fourier)
        x_filtered_abs = abs_eps(x_filtered)
        x_filtered_over_abs = x_filtered/x_filtered_abs
        indices = self._indices()
        norm = x.shape[-2]*x.shape[-1]

        if v is not None:
            m = torch.zeros((number_filters-1, number_filters)).to(x.device)
            m[indices[0], indices[1]] = v
            result_conv_phase = x_filtered_over_abs[:, :-1]*torch.einsum('ij, aibc -> ajbc', m.T, -torch.imag(x_filtered_2))
            result_conv_phase = torch.fft.ifft2(filters[:, :-1].conj()*torch.fft.fft2(result_conv_phase))
            x_filtered_abs_weighted = torch.einsum('ij, aibc -> ajbc', m, x_filtered_abs[:, :-1])
            result_conv_modulus = torch.fft.ifft2(filters_abs2_flip*torch.fft.fft2(x_filtered_abs_weighted))
            return (result_conv_phase.sum(1).real - result_conv_modulus.sum(1).imag)[:, None]/norm

        result_conv_phase = x_filtered_over_abs[:, :-1, None]*(-torch.imag(x_filtered_2))[:, None]
        result_conv_phase = torch.fft.ifft2(filters[:, :-1, None].conj()*torch.fft.fft2(result_conv_phase))
        result_conv_modulus = torch.fft.ifft2(filters_abs2_flip[:, None]*torch.fft.fft2(x_filtered_abs)[:, :-1, None])
        output = result_conv_phase.real - result_conv_modulus.imag
        return output[:, indices[0], indices[1]]/norm


def indices_fourth_order_imag(J, L):
    """(s, B, c) triples of the 2D fourth-order Imag statistic: the Real triples of indices_fourth_order restricted to
    s < B (the diagonal is identically zero, Im[U conj U] = 0) and to coarse filters c that are not the low-pass
    (real output, Imag = 0)."""
    indices = indices_fourth_order(J, L)
    keep = (indices[0] < indices[1]) & (indices[2] < J*L)
    return indices[:, keep]


class Scattering_Fourth_Order_Imag_2d(Potential):
    """Im[ (psi_c * |W_B x|) conj(psi_c * |W_s x|) ] averaged over pixels, s < B: the imaginary counterpart of
    Scattering_Fourth_Order_Real_2d (modulus envelope, same coarse-filter range). With A_i = |W_i x| and
    S_c = Im(ifft2(|psi_c|^2 fft2 .)) (antisymmetric), the statistic is <A_s, S_c A_B>/(MN), so the field gradient
    is S_c A_B on A_s and -S_c A_s on A_B."""
    def __init__(self, J, L, filters):
        super().__init__()
        self.J = J
        self.L = L
        self.filters = filters
        self.indices = indices_fourth_order_imag(J, L)
        self.num_coefficients = self.indices.shape[1]

    def forward(self, x):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft2(filters*torch.fft.fft2(x))
        x_filtered_abs = abs_eps(x_filtered)
        W_Wx = torch.fft.ifft2(filters[:, :, None] * torch.fft.fft2(x_filtered_abs)[:, None])
        output = torch.imag(W_Wx[:, :, :, None] * W_Wx[:, :, None].conj())
        output = output.mean(-1).mean(-1)
        output = output.permute(0, 3, 2, 1)           # (B, s, B', c), as in the Real class
        indices = self.indices.long()
        return output[:, indices[0], indices[1], indices[2]].reshape(x.shape[0], indices.shape[1])

    def grad(self, x, v=None, precomputed=None, means=None):
        filters = self.filters.to(x.device)
        number_filters = filters.shape[1]
        x_filtered_no_LF = torch.fft.ifft2(filters*torch.fft.fft2(x))[:, :-1]
        x_filtered_abs_no_LF = abs_eps(x_filtered_no_LF)
        x_filtered_over_abs_no_LF = x_filtered_no_LF/x_filtered_abs_no_LF
        # S_c A_j = Im(ifft2(|psi_c|^2 fft2 A_j)), shape (B, JL, F, M, N)
        S_A = torch.fft.ifft2(torch.fft.fft2(x_filtered_abs_no_LF)[:, :, None]*filters**2).imag
        indices = self.indices
        norm = x.shape[-2]*x.shape[-1]

        if v is not None:
            m = torch.zeros((number_filters-1, number_filters-1, number_filters)).to(x.device)
            m[indices[0], indices[1], indices[2]] = v
            m = m - torch.transpose(m, 0, 1)
            intermediate_output = x_filtered_over_abs_no_LF*torch.einsum('ijk, ajkbc -> aibc', m, S_A)
            return torch.fft.ifft2(filters[:, :-1]*torch.fft.fft2(intermediate_output)).real.sum(1)[:, None]/norm

        intermediate_output = x_filtered_over_abs_no_LF[:, :, None, None]*S_A[:, None]
        output = torch.fft.ifft2(filters[:, :-1, None, None]*torch.fft.fft2(intermediate_output)).real
        output = output - torch.transpose(output, 1, 2)
        return output[:, indices[0], indices[1], indices[2]].reshape(x.shape[0], indices.shape[1], x.shape[-2], x.shape[-1])/norm
