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



    ## ----------------------------------------------------- Potentials definitions -----------------------------------------------------    


    # ----- Scattering potentials -----


class Scattering_First_Order_1d(Potential):
    def __init__(self,filters):
        super().__init__()
        self.filters = filters
        self.num_coefficients = filters.shape[1]

    def forward(self,x):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft(filters*torch.fft.fft(x))
        return abs_eps(x_filtered).mean(-1)

    def grad(self, x, v=None, means=None):
        
        filters = self.filters.to(x.device)
        x_fourier = torch.fft.fft(x)
        x_filtered = torch.fft.ifft(filters*x_fourier)
        x_filtered_abs = abs_eps(x_filtered)
        x_filtered_over_abs = x_filtered/x_filtered_abs

        output = torch.real(torch.fft.ifft(torch.fft.fft(x_filtered_over_abs)*filters))
        
        
        if v==None:
            return output/x.shape[-1]
        else:
            return (output*v[None,:,None]).sum(1)[:,None]/x.shape[-1]

class Scattering_Second_Order_1d(Potential):
    def __init__(self,filters):
        super().__init__()
        self.filters = filters
        self.num_coefficients = filters.shape[1]

    def forward(self, x):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft(filters*torch.fft.fft(x))
        return (x_filtered*x_filtered.conj()).real.mean(-1)

    def grad(self, x, v=None,  means=None):

        filters = self.filters.to(x.device)
        x_filtered_2 = torch.fft.ifft(torch.fft.fft(x)*filters**2)
       
        output = x_filtered_2.real.reshape(x_filtered_2.shape[:1]+(-1,x.shape[-1]))


        if v==None:
            return 2*output/x.shape[-1]
        else:
            return (2*output*v[None,:,None]).sum(1)[:,None]/x.shape[-1]

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
            torch.fft.ifft(torch.fft.fft(2 * weighted) * filters)
        )

        if v is None:
            return output / x.shape[-1]
        else:
            return (output * v[None, :, None]).sum(1)[:, None] / x.shape[-1]
        
class Scattering_Third_Order_Real_1d(Potential):
    def __init__(self, J, filters):
        super().__init__()
        self.J = J
        self.filters = filters
        self.num_coefficients = int(J*((1+J)/2+1))

    def forward(self, x):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft(filters*torch.fft.fft(x))
        x_filtered_abs = abs_eps(x_filtered)

        output = torch.real(x_filtered[:, None].conj() * torch.fft.ifft(filters[:, None] * torch.fft.fft(x_filtered_abs)[:, :, None]))

        output = output.mean(-1)
        
        indices = indices_third_order(self.J, 1).long()
        
        output = output[:, indices[0], indices[1]]

        return output

    def grad(self, x,  v=None,  means=None):
        filters = self.filters.to(x.device)
        number_filters = filters.shape[1]
        
        x_fourier = torch.fft.fft(x)
        x_filtered_2 = torch.fft.ifft(x_fourier*filters**2)
        x_filtered = torch.fft.ifft(filters*x_fourier)
        x_filtered_abs = abs_eps(x_filtered)
        x_filtered_over_abs = x_filtered/x_filtered_abs


        indices = indices_third_order(self.J, 1)


        if v!=None:
            m = torch.zeros((number_filters-1, number_filters)).to(x.device)
            m[indices[0], indices[1]] = v
            
            result_conv_phase = x_filtered_over_abs[:,:-1]*torch.einsum('ij, aib -> ajb', m.T, torch.real(x_filtered_2))
            result_conv_phase = torch.fft.ifft(filters[:,:-1]*torch.fft.fft(result_conv_phase))
            
            x_filtered_abs_weighted = torch.einsum('ij, aib -> ajb', m, x_filtered_abs[:,:-1])
            result_conv_modulus = torch.fft.ifft(filters**2*torch.fft.fft(x_filtered_abs_weighted))
            
            return (result_conv_phase.sum(1) + result_conv_modulus.sum(1)).real[:,None]/x.shape[-1]
        
        result_conv_phase = x_filtered_over_abs[:,:-1,None]*torch.real(x_filtered_2)[:,None]
        result_conv_phase = torch.fft.ifft(filters[:,:-1,None]*torch.fft.fft(result_conv_phase))

        result_conv_modulus = torch.fft.ifft(filters[:,None]**2*torch.fft.fft(x_filtered_abs)[:,:-1,None])

        output = (result_conv_phase + result_conv_modulus).real

        output = output[:, indices[0], indices[1]]/x.shape[-1]

        return output
    

class Scattering_Fourth_Order_Real_1d(Potential):
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
        filters = self.filters.to(x.device)
        filters_Q = self.filters_Q.to(x.device)
        x_filtered = torch.fft.ifft(filters_Q*torch.fft.fft(x)) #(B,JQ,T)
        
        #Normalize micro
        x_filtered = x_filtered/self.norm
        
        x_filtered_abs = abs_eps(x_filtered) #(B,JQ,T)
        
        W_Wx = torch.fft.ifft(filters[:, :, None] * torch.fft.fft(x_filtered_abs)[:, None]) #(B,JQ,J+1,T)
        output = torch.real(W_Wx[:, :, :, None] * W_Wx[:, :, None].conj()) #(B,JQ,J+1,J+1,T)
        output = output.mean(-1) #(B,JQ,J+1,J+1)
        output = output.permute(0, 3, 2, 1) #(B,J+1,J+1,JQ)
        
        indices = self.indices.long()
        
        output = output[:, indices[0], indices[1], indices[2]]
        
        output = output.reshape(x.shape[0], indices.shape[1])
        return output

    def grad(self, x,  v=None, means=None):
        filters = self.filters.to(x.device)
        filters_Q = self.filters_Q.to(x.device)
        number_filters = filters.shape[1]
        number_filters_Q = filters_Q.shape[1]

        
        x_filtered_no_LF = torch.fft.ifft(filters_Q*torch.fft.fft(x)) # (B,J*Q,M,N)
        x_filtered_abs_no_LF = abs_eps(x_filtered_no_LF) # (B,J*L,M,N)
        x_filtered_over_abs_no_LF = (x_filtered_no_LF/x_filtered_abs_no_LF)

        

        x_filtered_abs_no_LF_filtered_2 = torch.fft.ifft((torch.fft.fft(x_filtered_abs_no_LF)[:,:,None]*filters**2)) # (B,J*Q,J+1,M) #no 2*, x_filtered_abs_no_LF
        
        indices = self.indices

        if v != None:
            m = torch.zeros((number_filters_Q, number_filters_Q, number_filters)).to(x.device)
            m[indices[0], indices[1], indices[2]] = v / self.norm_indices.to(v.device).to(v.dtype)
            m = m+torch.transpose(m,0,1) 
    
            intermediate_output = x_filtered_over_abs_no_LF*torch.einsum('ijk, ajkb -> aib', m, torch.real(x_filtered_abs_no_LF_filtered_2))
            return torch.fft.ifft(filters_Q*torch.fft.fft(intermediate_output)).real.sum(1)[:,None]/x.shape[-1]
            
        intermediate_output = x_filtered_over_abs_no_LF[:,:,None,None]*torch.real(x_filtered_abs_no_LF_filtered_2)[:,None] #x_filtered_over_abs_no_LF
        output = torch.fft.ifft(filters_Q[:,:,None,None]*torch.fft.fft(intermediate_output)).real

        output = output + torch.transpose(output, 1,2)      
        output = output[:, indices[0], indices[1], indices[2]].reshape(x.shape[0], indices.shape[1], x.shape[-1])/x.shape[-1]

        
        return output / self.norm_indices[:,None].to(x.device).to(x.dtype)
        
    def fit_micro(self,x):
        filters_Q = self.filters_Q.to(x.device)
        x_filtered = torch.fft.ifft(filters_Q*torch.fft.fft(x))
        x_filtered = x_filtered.abs()**2
        #Normalize_micro
        self.norm = x_filtered.mean((0,2))[:,None]**(-1/4)
        #self.norm = x_filtered.mean((0,2))[:,None]**0.5
        
        indices = self.indices
        norm_indices = self.norm[:,None]*self.norm[None,:] 
        norm_indices = norm_indices.repeat((1,1,self.J+1))
        self.norm_indices =  norm_indices[indices[0], indices[1], indices[2]] #(n_pot)


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
        filters = self.filters.to(x.device)
        filters_Q = self.filters_Q.to(x.device)
        x_filtered = torch.fft.ifft(filters_Q*torch.fft.fft(x)) #(B,JQ,T) first wavelet transform 
        
        #Normalize micro
        x_filtered = x_filtered/self.norm
        
        x_filtered_abs2 = abs_eps(x_filtered)**2 #(B,JQ,T)
        
        W_Wx = torch.fft.ifft(filters[:, :, None] * torch.fft.fft(x_filtered_abs2)[:, None]) #(B,JQ,J+1,T) second wavelet transform
        output = torch.real(W_Wx[:, :, :, None] * W_Wx[:, :, None].conj()) #(B,JQ,J+1,J+1,T)
        output = output.mean(-1) #(B,JQ,J+1,J+1)
        output = output.permute(0, 3, 2, 1) #(B,J+1,J+1,JQ)
        
        indices = self.indices.long()
        
        output = output[:, indices[0], indices[1], indices[2]]
        
        output = output.reshape(x.shape[0], indices.shape[1])
        return output

    def grad(self, x,  v=None, means=None):
        filters = self.filters.to(x.device)
        filters_Q = self.filters_Q.to(x.device)
        number_filters = filters.shape[1]
        number_filters_Q = filters_Q.shape[1]

        
        x_filtered_no_LF = torch.fft.ifft(filters_Q*torch.fft.fft(x)) # (B,J*Q,M,N)
        x_filtered_abs2_no_LF = abs_eps(x_filtered_no_LF)**2 # (B,J*L,M,N)
        x_filtered_grad = 2 * x_filtered_no_LF

        

        x_filtered_abs2_no_LF_filtered_2 = torch.fft.ifft(
            (torch.fft.fft(x_filtered_abs2_no_LF)[:,:,None] * filters**2)
        )        
        indices = self.indices

        if v != None:
            m = torch.zeros((number_filters_Q, number_filters_Q, number_filters)).to(x.device)
            m[indices[0], indices[1], indices[2]] = v / self.norm_indices.to(v.device).to(v.dtype)
            m = m+torch.transpose(m,0,1) 
    
            intermediate_output = x_filtered_grad*torch.einsum('ijk, ajkb -> aib', m, torch.real(x_filtered_abs2_no_LF_filtered_2))
            return torch.fft.ifft(filters_Q*torch.fft.fft(intermediate_output)).real.sum(1)[:,None]/x.shape[-1]
            
        intermediate_output = x_filtered_grad[:,:,None,None]*torch.real(x_filtered_abs2_no_LF_filtered_2)[:,None] #x_filtered_over_abs_no_LF
        output = torch.fft.ifft(filters_Q[:,:,None,None]*torch.fft.fft(intermediate_output)).real

        output = output + torch.transpose(output, 1,2)      
        output = output[:, indices[0], indices[1], indices[2]].reshape(x.shape[0], indices.shape[1], x.shape[-1])/x.shape[-1]

        
        return output / self.norm_indices[:,None].to(x.device).to(x.dtype)
        
    def fit_micro(self,x):
        filters_Q = self.filters_Q.to(x.device)
        x_filtered = torch.fft.ifft(filters_Q*torch.fft.fft(x))
        x_filtered = x_filtered.abs()**2
        #Normalize_micro
        self.norm = x_filtered.mean((0,2))[:,None]**(-1/4)
        #self.norm = x_filtered.mean((0,2))[:,None]**0.5
        
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

            intermediate_output = torch.fft.ifft(filters_Q*torch.fft.fft(intermediate_output)).sum(1)[:,None]/x.shape[-1]
            intermediate_output_conj = torch.fft.ifft(filters_Q_tilde*torch.fft.fft(intermediate_output_conj)).sum(1)[:,None]/x.shape[-1]

            output =intermediate_output + intermediate_output_conj

            intermediate_output = 0.5*x_filtered_over_abs_no_LF*torch.einsum('ijk, ajkb -> aib', m_transpose, x_filtered_abs_no_LF_filtered_2_tilde)
            intermediate_output_conj = 0.5*x_filtered_over_abs_no_LF.conj()*torch.einsum('ijk, ajkb -> aib', m_transpose, x_filtered_abs_no_LF_filtered_2_tilde)

            intermediate_output = torch.fft.ifft(filters_Q*torch.fft.fft(intermediate_output)).sum(1)[:,None]/x.shape[-1]
            intermediate_output_conj = torch.fft.ifft(filters_Q_tilde*torch.fft.fft(intermediate_output_conj)).sum(1)[:,None]/x.shape[-1]

            output_transpose =intermediate_output + intermediate_output_conj

            output = output + output_transpose
            return output.imag
            
        intermediate_output = 0.5*x_filtered_over_abs_no_LF[:,:,None,None]*x_filtered_abs_no_LF_filtered_2[:,None] #x_filtered_over_abs_no_LF
        intermediate_output_conj = 0.5*x_filtered_over_abs_no_LF[:,:,None,None].conj()*x_filtered_abs_no_LF_filtered_2[:,None] #x_filtered_over_abs_no_LF

        intermediate_output = torch.fft.ifft(filters_Q[:,:,None,None]*torch.fft.fft(intermediate_output))
        intermediate_output_conj = torch.fft.ifft(filters_Q_tilde[:,:,None,None].conj()*torch.fft.fft(intermediate_output_conj))
        
        output =intermediate_output + intermediate_output_conj

        intermediate_output = 0.5*x_filtered_over_abs_no_LF[:,:,None,None]*x_filtered_abs_no_LF_filtered_2_tilde[:,None] #x_filtered_over_abs_no_LF
        intermediate_output_conj = 0.5*x_filtered_over_abs_no_LF[:,:,None,None].conj()*x_filtered_abs_no_LF_filtered_2_tilde[:,None] #x_filtered_over_abs_no_LF

        intermediate_output = torch.fft.ifft(filters_Q[:,:,None,None]*torch.fft.fft(intermediate_output))
        intermediate_output_conj = torch.fft.ifft(filters_Q_tilde[:,:,None,None].conj()*torch.fft.fft(intermediate_output_conj))

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

        x_L1 = torch.real(torch.fft.ifft(torch.fft.fft(x_filtered_over_abs)*filters)) #(B,J*Q,T)

       
        
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
        output = torch.fft.ifft(filters*torch.fft.fft(output)).real

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
        output = torch.fft.ifft(filters*torch.fft.fft(output)).real

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


class Scalar_coshgt_old:
    """
    Per-channel cosh-tempered Generalized-t (coshGT) potential for wavelet coefficients.

    Density (symmetric, zero-mean):
        p(x) ∝ (1 + (g_a(x)/x0)^b)^{-t/b}
        g_a(x) = |x| cosh(a x)

    Shape:
      - Body  (x→0):          p(x) ~ |x|^b              (cusp / flatness)
      - Tail  (x→∞, a=0):     p(x) ~ |x|^{-t}           (pure power law, GT)
      - Tail  (x→∞, a>0):     p(x) ~ |x|^{-t} e^{-a t |x|}  (exponentially tempered)

    Parameters stored per channel (tensors of shape (J,)):
        b     : body cusp index      (> 0)
        t     : tail power index     (> 0)
        a     : tempering rate       (≥ 0; 0 = pure GT).  kappa = a*t is NOT stored.
        x0    : scale                (> 0)

    Key change from the (a,b,c,x0) and (b,t,kappa,x0) variants:
      - We optimize over a directly (not kappa=a*t), because a is the quantity
        that appears in the density, is naturally bounded by a_max, and does not
        co-vary with t or x0. This prevents the kappa/t blow-up seen when
        optimizing over kappa.
    """

    def __init__(self, filters, eps_abs=1e-6, eps_scale=1e-6,
                 a_max=5.0, b_bounds=(0.05, 10.0), t_bounds=(0.5, 50.0)):
        self.filters           = filters
        self.num_coefficients  = filters.shape[1]
        self.eps_abs           = eps_abs
        self.eps_scale         = eps_scale
        self.a_max             = a_max
        self.b_bounds          = b_bounds
        self.t_bounds          = t_bounds
        self.b = self.t = self.a = self.x0 = None   # set by fit_reference

    @property
    def is_fitted(self):
        return self.a is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Scalar_coshgt must be fit_reference'd first.")

    # ------------------------------------------------------------------
    # numpy helpers  (fitting only — all in (b, t, a, x0) coords)
    # ------------------------------------------------------------------
    @staticmethod
    def _logcosh_np(z):
        """Numerically stable log cosh(z)."""
        z = np.abs(z)
        return z + np.log1p(np.exp(-2.0 * z)) - np.log(2.0)

    @classmethod
    def _log_u_np(cls, x, b, a, x0):
        """log_u = b * log(g_a(x)/x0),   g_a(x) = |x| cosh(ax)."""
        lx = np.log(np.maximum(np.abs(x), 1e-300))
        return b * (lx + cls._logcosh_np(a * x) - np.log(x0))

    @classmethod
    def _logZ_np(cls, b, t, a, x0):
        """
        log normalisation constant.  Returns np.inf on numerical failure
        so the optimizer sees a large-penalty signal rather than nan/crash.
        """
        try:
            f = lambda s: np.exp(-(t / b) * np.logaddexp(0.0,
                                   cls._log_u_np(s, b, a, x0)))
            I, _ = quad(f, 0.0, np.inf, limit=200)
            if not np.isfinite(I) or I <= 0.0:
                return np.inf
            return np.log(2.0 * I)
        except Exception:
            return np.inf

    @classmethod
    def _logpdf_np(cls, x, b, t, a, x0):
        lZ = cls._logZ_np(b, t, a, x0)
        if not np.isfinite(lZ):
            return np.full_like(x, -np.inf, dtype=float)
        return -(t / b) * np.logaddexp(0.0, cls._log_u_np(x, b, a, x0)) - lZ

    # ------------------------------------------------------------------
    # Per-channel MAP fit in (b, t, a, x0)
    #
    # Key design choices:
    #   1. Optimize over log(a) so a stays positive; clip to [0, a_max].
    #   2. Channel data is normalised to unit MAD before fitting and x0 is
    #      rescaled back afterwards — this breaks the a/x0 co-linearity.
    #   3. MAP penalty: lam * a  (half-normal on a, pulls toward pure GT).
    #   4. Model selection: LR test GT (a=0) vs tempered (a free).
    #   5. logZ failures return 1e12 to keep Nelder-Mead alive.
    # ------------------------------------------------------------------
    @classmethod
    def _fit_channel(cls, h_raw,
                     b0=1.0, t0=4.0, a0=0.1,
                     lam=1.0, lr_thresh=2.0,
                     b_bounds=(0.05, 10.0), t_bounds=(0.5, 50.0),
                     a_max=5.0, eps_scale=1e-6):
        """
        Returns (b, t, a, x0) with a in [0, a_max] and x0 in original units.
        """
        # --- normalise to unit MAD (fitting in normalised space) ---
        h_scale = float(1.4826 * np.median(np.abs(h_raw - np.median(h_raw))))
        h_scale = h_scale or float(np.std(h_raw)) or 1.0
        h = h_raw / h_scale          # fitting domain: ~unit scale
        x0_unit = 1.0                # pin x0=1 in normalised space

        a_floor = 1e-6               # numerical zero for GT limit

        # ---- unpack: log-space vector → (b, t, a) --------------------
        # We always pin x0=1 in normalised space (fit_scale=False).
        # The optimiser therefore has 3 free params: [log b, log t, log a].
        def unpack(th, free_a):
            b_ = float(np.clip(np.exp(th[0]), *b_bounds))
            t_ = float(np.clip(np.exp(th[1]), *t_bounds))
            if free_a:
                a_ = float(np.clip(np.exp(th[2]), a_floor, a_max))
            else:
                a_ = a_floor
            return b_, t_, a_

        # ---- full model (a free) --------------------------------------
        th0_full = np.array([np.log(b0), np.log(t0), np.log(a0)])

        def nll_full(th):
            b_, t_, a_ = unpack(th, free_a=True)
            lZ = cls._logZ_np(b_, t_, a_, x0_unit)
            if not np.isfinite(lZ):
                return 1e12
            ll  = cls._logpdf_np(h, b_, t_, a_, x0_unit).sum()
            pen = lam * a_           # half-normal prior on a → pulls to GT
            return -ll + pen

        res_full = minimize(nll_full, th0_full, method="Nelder-Mead",
                            options=dict(xatol=1e-5, fatol=1e-5, maxiter=15000))
        b_f, t_f, a_f = unpack(res_full.x, free_a=True)
        ll_full = cls._logpdf_np(h, b_f, t_f, a_f, x0_unit).sum()

        # ---- GT limit (a pinned at floor) -----------------------------
        th0_gt = np.array([np.log(b0), np.log(t0)])

        def nll_gt(th):
            b_, t_, a_ = unpack(np.append(th, np.log(a_floor)), free_a=False)
            lZ = cls._logZ_np(b_, t_, a_, x0_unit)
            if not np.isfinite(lZ):
                return 1e12
            return -cls._logpdf_np(h, b_, t_, a_, x0_unit).sum()

        res_gt = minimize(nll_gt, th0_gt, method="Nelder-Mead",
                          options=dict(xatol=1e-5, fatol=1e-5, maxiter=15000))
        b_g, t_g, a_g = unpack(np.append(res_gt.x, np.log(a_floor)), free_a=False)
        ll_gt = cls._logpdf_np(h, b_g, t_g, a_g, x0_unit).sum()

        # ---- likelihood-ratio model selection -------------------------
        tempered = 2.0 * (ll_full - ll_gt) > lr_thresh
        if tempered:
            b_out, t_out, a_out = b_f, t_f, a_f
        else:
            b_out, t_out, a_out = b_g, t_g, 0.0

        # --- rescale x0 back to original units -------------------------
        # In normalised space x0=1 and |x|_typical=1, so in original units
        # x0 = h_scale * x0_unit = h_scale.
        x0_out = h_scale * x0_unit

        return b_out, t_out, a_out, x0_out

    # ------------------------------------------------------------------
    # fit_reference: fit all channels from a batch of signals x
    # ------------------------------------------------------------------
    def fit_reference(self, x, lam=1.0, lr_thresh=2.0):
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real.detach().cpu().numpy()

        bb, tt, aa, xx = [], [], [], []
        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            if h.size < 10:
                b_, t_, a_, x0_ = 1.0, 4.0, 0.0, 1.0
            else:
                try:
                    b_, t_, a_, x0_ = self._fit_channel(
                        h, lam=lam, lr_thresh=lr_thresh,
                        b_bounds=self.b_bounds, t_bounds=self.t_bounds,
                        a_max=self.a_max, eps_scale=self.eps_scale)
                except Exception as e:
                    print(f"[coshGT][ch {j}] fit failed ({e}) → fallback")
                    b_, t_, a_, x0_ = 1.0, 4.0, 0.0, float(np.std(h) + self.eps_scale)

            b_  = float(np.clip(b_,  *self.b_bounds))
            t_  = float(np.clip(t_,  *self.t_bounds))
            a_  = float(np.clip(a_,  0.0, self.a_max))
            x0_ = float(max(x0_, self.eps_scale))

            tag = "tempered" if a_ > 1e-6 else "GT"
            print(f"[coshGT][ch {j}] [{tag:>8s}]  b={b_:.3f}  t={t_:.3f}  "
                  f"a={a_:.4f}  x0={x0_:.5f}  (kappa=a*t={a_*t_:.4f})")
            bb.append(b_); tt.append(t_); aa.append(a_); xx.append(x0_)

        dtype = x.dtype if x.is_floating_point() else torch.float32
        self.b  = torch.tensor(bb, dtype=dtype, device=x.device)
        self.t  = torch.tensor(tt, dtype=dtype, device=x.device)
        self.a  = torch.tensor(aa, dtype=dtype, device=x.device)
        self.x0 = torch.tensor(xx, dtype=dtype, device=x.device)

    def fit(self, x, **kwargs):
        return self.fit_reference(x, **kwargs)

    # ------------------------------------------------------------------
    # Stable torch building blocks
    # ------------------------------------------------------------------
    @staticmethod
    def _logcosh(z):
        """Numerically stable log cosh(z) in torch."""
        az = z.abs()
        return az + F.softplus(-2.0 * az) - math.log(2.0)

    def _params(self, device):
        """Broadcast to (1, J, 1) for (B, J, T) coefficient tensors."""
        return (self.a .to(device)[None, :, None],
                self.b .to(device)[None, :, None],
                self.t .to(device)[None, :, None],
                self.x0.to(device)[None, :, None])

    def _log_u(self, z, a, b, x0):
        """log_u = b * log(g_a(z)/x0),   g_a(z) = |z| cosh(az)."""
        az = torch.sqrt(z ** 2 + self.eps_abs)
        return b * (torch.log(az) + self._logcosh(a * z) - torch.log(x0))

    # ------------------------------------------------------------------
    # φ(x) = (t/b) * softplus(log_u)                    → shape (B, J)
    # ------------------------------------------------------------------
    def forward(self, x, *args):
        """Per-channel potential, averaged over time. Returns (B, J)."""
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real   # (B, J, T)
        a, b, t, x0 = self._params(x.device)
        log_u = self._log_u(z, a, b, x0)
        return ((t / b) * F.softplus(log_u)).mean(-1)          # (B, J)

    # ------------------------------------------------------------------
    # φ'(z) = t * σ(log_u) * ( z/(z²+ε) + a·tanh(az) )
    # ------------------------------------------------------------------
    def grad(self, x, v=None, means=None):
        """
        Gradient of the potential w.r.t. x, back-projected through the filters.
        If v (shape J) is given, returns the weighted sum over channels → (B,1,T).
        """
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real    # (B, J, T)
        a, b, t, x0 = self._params(x.device)

        log_u  = self._log_u(z, a, b, x0)
        dlog_g = z / (z ** 2 + self.eps_abs) + a * torch.tanh(a * z)
        dphi_dz = t * torch.sigmoid(log_u) * dlog_g            # (B, J, T)

        grad_coeff = torch.fft.ifft(
            torch.fft.fft(dphi_dz) * filters
        ).real / x.shape[-1]                                    # (B, J, T)

        if v is None:
            return grad_coeff
        return (grad_coeff * v[None, :, None]).sum(1)[:, None]  # (B, 1, T)

    # ------------------------------------------------------------------
    def summary(self):
        self._check_fitted()
        b  = self.b.cpu().numpy()
        t  = self.t.cpu().numpy()
        a  = self.a.cpu().numpy()
        x0 = self.x0.cpu().numpy()
        print(f"{'Ch':>4}  {'b':>7}  {'t':>7}  {'a':>7}  {'x0':>10}  "
              f"{'kappa=a*t':>10}  {'mode':>10}")
        print("-" * 68)
        for j in range(len(b)):
            tag = "tempered" if a[j] > 1e-6 else "GT"
            print(f"{j:>4d}  {b[j]:>7.3f}  {t[j]:>7.3f}  {a[j]:>7.4f}  "
                  f"{x0[j]:>10.5f}  {a[j]*t[j]:>10.4f}  {tag:>10}")


# coshGt but fitting on two separate windows 
class Scalar_coshgt:
    """
    Per-channel cosh-tempered Generalized-t (coshGT) potential for wavelet
    coefficients, fitted SEPARATELY on the bulk and on the tails of each
    channel's coefficient histogram.

    Why two fits
    ------------
    A single coshGT must compromise between two regimes that obey different
    laws:
        Body  (|z| small):  p(z) ~ |z|^b                    -> set by  b, x0
        Tail  (|z| large):  p(z) ~ |z|^{-t} e^{-a t |z|}     -> set by  t, a
    Fitting the whole histogram at once lets the dense body dominate the
    likelihood and washes out the (sparse but decisive) tail shape.  We
    therefore run two *weighted* maximum-likelihood fits and expose two
    *windowed* energy terms per channel:

        bulk fit :  weights w_bulk(z) ~ 1 on the body, ~0 on the tail
        tail fit :  weights w_tail(z) ~ 1 on the tail, ~0 on the body

        Phi_bulk(z) = w_bulk(z) * phi_bulk(z)    (active mainly on the body)
        Phi_tail(z) = w_tail(z) * phi_tail(z)    (active mainly on the tail)

    forward() returns (B, 2J) -> [bulk_0..J-1, tail_0..J-1] and
    num_coefficients = 2J.  grad()/v follow the same ordering.

    Non-colinearity (guaranteed by construction)
    --------------------------------------------
    The windows form a smooth partition of unity on |z|:
        w_tail(z) = sigmoid((|z| - c)/s),   w_bulk(z) = 1 - w_tail(z).
    Because w_bulk and w_tail have essentially disjoint support, there is no
    constant lambda with  w_bulk*phi_bulk == lambda * w_tail*phi_tail  for all
    z: where one feature is non-zero the other is ~0, and (1 - w_tail) is not
    proportional to a non-constant sigmoid w_tail.  Hence the two energy terms
    are linearly independent / not colinear.  fit_reference() additionally
    measures the cosine between the two sampled potentials per channel
    (self.cos_bt) and warns if it ever approaches 1.

    Choice of the bulk/tail border c
    --------------------------------
    c is set per channel from a high quantile of |z| (default 0.90): the body
    is the dense central mass, the tail the sparse remainder.  The transition
    width s is set from the spread of |z| between two quantiles
    (default 0.85-0.97) so the switch is gentle and data-adaptive.

    Stored per channel (tensors of shape (J,)), with _bulk / _tail suffixes:
        b, t, a, x0  for each regime, plus the border c, the width s and the
        diagnostic cosine cos_bt.
    """

    def __init__(self, filters, eps_abs=1e-6, eps_scale=1e-6,
                 a_max=5.0, b_bounds=(0.05, 10.0), t_bounds=(0.5, 50.0),
                 bulk_quantile=0.99, trans_quantiles=(0.989, 0.991),
                 min_eff_samples=50.0):
        self.filters           = filters
        self.num_coefficients  = 2 * filters.shape[1]      # bulk + tail per channel
        self.eps_abs           = eps_abs
        self.eps_scale         = eps_scale
        self.a_max             = a_max
        self.b_bounds          = b_bounds
        self.t_bounds          = t_bounds
        self.bulk_quantile     = bulk_quantile
        self.trans_quantiles   = trans_quantiles
        self.min_eff_samples   = min_eff_samples
        # bulk params (J,)
        self.b_bulk = self.t_bulk = self.a_bulk = self.x0_bulk = None
        # tail params (J,)
        self.b_tail = self.t_tail = self.a_tail = self.x0_tail = None
        # window params (J,) and diagnostic
        self.c = self.s = None
        self.cos_bt = None

    @property
    def is_fitted(self):
        return self.a_bulk is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Scalar_coshgt must be fit_reference'd first.")

    # ------------------------------------------------------------------
    # numpy helpers  (fitting only — all in (b, t, a, x0) coords)
    # ------------------------------------------------------------------
    @staticmethod
    def _logcosh_np(z):
        """Numerically stable log cosh(z)."""
        z = np.abs(z)
        return z + np.log1p(np.exp(-2.0 * z)) - np.log(2.0)

    @classmethod
    def _log_u_np(cls, x, b, a, x0):
        """log_u = b * log(g_a(x)/x0),   g_a(x) = |x| cosh(ax)."""
        lx = np.log(np.maximum(np.abs(x), 1e-300))
        return b * (lx + cls._logcosh_np(a * x) - np.log(x0))

    @classmethod
    def _logZ_np(cls, b, t, a, x0):
        """
        log normalisation constant.  Returns np.inf on numerical failure
        so the optimizer sees a large-penalty signal rather than nan/crash.
        """
        try:
            f = lambda s: np.exp(-(t / b) * np.logaddexp(0.0,
                                   cls._log_u_np(s, b, a, x0)))
            I, _ = quad(f, 0.0, np.inf, limit=200)
            if not np.isfinite(I) or I <= 0.0:
                return np.inf
            return np.log(2.0 * I)
        except Exception:
            return np.inf

    @classmethod
    def _logpdf_np(cls, x, b, t, a, x0):
        lZ = cls._logZ_np(b, t, a, x0)
        if not np.isfinite(lZ):
            return np.full_like(x, -np.inf, dtype=float)
        return -(t / b) * np.logaddexp(0.0, cls._log_u_np(x, b, a, x0)) - lZ

    @staticmethod
    def _weighted_median(values, weights):
        """Weighted median of `values` (used for a region-aware robust scale)."""
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if values.size == 0:
            return 0.0
        order = np.argsort(values)
        v, w = values[order], weights[order]
        cw = np.cumsum(w)
        if cw[-1] <= 0:
            return float(np.median(values))
        idx = int(np.searchsorted(cw, 0.5 * cw[-1]))
        idx = min(idx, len(v) - 1)
        return float(v[idx])

    # ------------------------------------------------------------------
    # Per-channel WEIGHTED MAP fit in (b, t, a, x0).
    #
    #   * `weights` (>=0, same shape as h_raw) reweight the log-likelihood so
    #     the fit concentrates on the bulk or on the tail.
    #   * Data is normalised to a region-aware unit scale before fitting and
    #     x0 is rescaled back afterwards — this breaks the a/x0 co-linearity.
    #   * MAP penalty lam*a (half-normal on a, pulls toward pure GT).
    #   * Model selection: weighted LR test GT (a=0) vs tempered (a free).
    # ------------------------------------------------------------------
    @classmethod
    def _fit_channel(cls, h_raw, weights=None,
                     b0=1.0, t0=4.0, a0=0.1,
                     lam=1.0, lr_thresh=2.0,
                     b_bounds=(0.05, 10.0), t_bounds=(0.5, 50.0),
                     a_max=5.0, eps_scale=1e-6):
        """Returns (b, t, a, x0) with a in [0, a_max] and x0 in original units."""
        h_raw = np.asarray(h_raw, dtype=float)
        if weights is None:
            weights = np.ones_like(h_raw)
        w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
        if w.sum() <= 0:
            w = np.ones_like(h_raw)

        # --- region-aware robust scale (weighted MAD about 0; coeffs are
        #     zero-mean & symmetric) → normalise the region to ~unit scale ---
        h_scale = float(1.4826 * cls._weighted_median(np.abs(h_raw), w))
        if not (h_scale > 0):
            h_scale = float(np.sqrt((w * h_raw ** 2).sum() / w.sum()))
        h_scale = h_scale or 1.0
        h = h_raw / h_scale
        x0_unit = 1.0
        a_floor = 1e-6

        def unpack(th, free_a):
            b_ = float(np.clip(np.exp(th[0]), *b_bounds))
            t_ = float(np.clip(np.exp(th[1]), *t_bounds))
            a_ = float(np.clip(np.exp(th[2]), a_floor, a_max)) if free_a else a_floor
            return b_, t_, a_

        # ---- full model (a free) ----
        th0_full = np.array([np.log(b0), np.log(t0), np.log(a0)])

        def nll_full(th):
            b_, t_, a_ = unpack(th, free_a=True)
            if not np.isfinite(cls._logZ_np(b_, t_, a_, x0_unit)):
                return 1e12
            ll  = float((w * cls._logpdf_np(h, b_, t_, a_, x0_unit)).sum())
            return -ll + lam * a_

        res_full = minimize(nll_full, th0_full, method="Nelder-Mead",
                            options=dict(xatol=1e-5, fatol=1e-5, maxiter=15000))
        b_f, t_f, a_f = unpack(res_full.x, free_a=True)
        ll_full = float((w * cls._logpdf_np(h, b_f, t_f, a_f, x0_unit)).sum())

        # ---- GT limit (a pinned at floor) ----
        th0_gt = np.array([np.log(b0), np.log(t0)])

        def nll_gt(th):
            b_, t_, a_ = unpack(np.append(th, np.log(a_floor)), free_a=False)
            if not np.isfinite(cls._logZ_np(b_, t_, a_, x0_unit)):
                return 1e12
            return -float((w * cls._logpdf_np(h, b_, t_, a_, x0_unit)).sum())

        res_gt = minimize(nll_gt, th0_gt, method="Nelder-Mead",
                          options=dict(xatol=1e-5, fatol=1e-5, maxiter=15000))
        b_g, t_g, a_g = unpack(np.append(res_gt.x, np.log(a_floor)), free_a=False)
        ll_gt = float((w * cls._logpdf_np(h, b_g, t_g, a_g, x0_unit)).sum())

        # ---- weighted likelihood-ratio model selection ----
        if 2.0 * (ll_full - ll_gt) > lr_thresh:
            b_out, t_out, a_out = b_f, t_f, a_f
        else:
            b_out, t_out, a_out = b_g, t_g, 0.0

        x0_out = h_scale * x0_unit   # |x|_typical=1 in normalised space
        return b_out, t_out, a_out, x0_out

    # ------------------------------------------------------------------
    # Diagnostic: cosine between the two windowed, mean-removed potentials
    # sampled on a |z| grid.  ~0 => well separated, ->1 => colinear.
    # ------------------------------------------------------------------
    @classmethod
    def _potential_cosine(cls, c, s, par_bulk, par_tail, n=1024):
        b_b, t_b, a_b, x0_b = par_bulk
        b_t, t_t, a_t, x0_t = par_tail
        span = c + 10.0 * s
        zg = np.linspace(-span, span, n)
        az = np.abs(zg)
        wt = 1.0 / (1.0 + np.exp(-(az - c) / s))
        wb = 1.0 - wt
        def phi(zz, b, t, a, x0):
            return (t / b) * np.logaddexp(0.0, cls._log_u_np(zz, b, a, x0))
        fb = wb * phi(zg, b_b, t_b, a_b, x0_b)
        ft = wt * phi(zg, b_t, t_t, a_t, x0_t)
        fb = fb - fb.mean(); ft = ft - ft.mean()
        denom = np.linalg.norm(fb) * np.linalg.norm(ft)
        return float(abs(fb @ ft) / denom) if denom > 0 else 0.0

    # ------------------------------------------------------------------
    # fit_reference: fit bulk + tail for all channels from a batch x
    # ------------------------------------------------------------------
    def fit_reference(self, x, lam=1.0, lr_thresh=2.0,
                      bulk_quantile=None, trans_quantiles=None):
        if bulk_quantile is None:
            bulk_quantile = self.bulk_quantile
        if trans_quantiles is None:
            trans_quantiles = self.trans_quantiles

        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real.detach().cpu().numpy()

        bb_b, tt_b, aa_b, xx_b = [], [], [], []
        bb_t, tt_t, aa_t, xx_t = [], [], [], []
        cc, ss, cosines = [], [], []

        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            ah = np.abs(h)

            if h.size < 10:
                b_, t_, x0_ = 1.0, 4.0, 1.0
                c_ = float(np.median(ah)) if ah.size else 1.0
                s_ = max(0.1 * c_, self.eps_scale)
                bb_b += [b_]; tt_b += [t_]; aa_b += [0.0]; xx_b += [x0_]
                bb_t += [b_]; tt_t += [max(t_, 1.0)]; aa_t += [0.0]; xx_t += [x0_]
                cc += [c_]; ss += [s_]; cosines += [0.0]
                continue

            # --- bulk/tail border c and transition width s from |z| quantiles ---
            c_ = float(np.quantile(ah, bulk_quantile))
            q_lo, q_hi = trans_quantiles
            band = float(np.quantile(ah, q_hi) - np.quantile(ah, q_lo))
            s_ = band / 4.0                      # ~ +/-2 logistic scales over the band
            s_ = max(s_, 0.05 * (c_ + self.eps_scale), self.eps_scale)

            # --- smooth partition-of-unity windows on |z| ---
            w_tail = 1.0 / (1.0 + np.exp(-(ah - c_) / s_))
            w_bulk = 1.0 - w_tail

            eff_tail = (w_tail.sum() ** 2) / max((w_tail ** 2).sum(), 1e-12)
            if eff_tail < self.min_eff_samples:
                print(f"[coshGT2][ch {j}] tail eff. N={eff_tail:.1f} < "
                      f"{self.min_eff_samples:.0f}: tail fit may be weak "
                      f"(consider lowering bulk_quantile).")

            # --- weighted fits: bulk (body shape) and tail (tail shape) ---
            try:
                b_bk, t_bk, a_bk, x0_bk = self._fit_channel(
                    h, weights=w_bulk, a0=0.05, lam=lam, lr_thresh=lr_thresh,
                    b_bounds=self.b_bounds, t_bounds=self.t_bounds,
                    a_max=self.a_max, eps_scale=self.eps_scale)
            except Exception as e:
                print(f"[coshGT2][ch {j}] bulk fit failed ({e}) -> fallback")
                b_bk, t_bk, a_bk, x0_bk = 1.0, 6.0, 0.0, float(np.std(h) + self.eps_scale)

            try:
                b_tl, t_tl, a_tl, x0_tl = self._fit_channel(
                    h, weights=w_tail, a0=0.10, lam=lam, lr_thresh=lr_thresh,
                    b_bounds=self.b_bounds, t_bounds=self.t_bounds,
                    a_max=self.a_max, eps_scale=self.eps_scale)
            except Exception as e:
                print(f"[coshGT2][ch {j}] tail fit failed ({e}) -> fallback")
                b_tl, t_tl, a_tl, x0_tl = 1.0, 3.0, 0.0, float(np.std(h) + self.eps_scale)

            # --- clip to bounds ---
            b_bk = float(np.clip(b_bk, *self.b_bounds)); t_bk = float(np.clip(t_bk, *self.t_bounds))
            a_bk = float(np.clip(a_bk, 0.0, self.a_max)); x0_bk = float(max(x0_bk, self.eps_scale))
            b_tl = float(np.clip(b_tl, *self.b_bounds)); t_tl = float(np.clip(t_tl, *self.t_bounds))
            a_tl = float(np.clip(a_tl, 0.0, self.a_max)); x0_tl = float(max(x0_tl, self.eps_scale))

            # --- non-colinearity diagnostic ---
            cos_bt = self._potential_cosine(
                c_, s_, (b_bk, t_bk, a_bk, x0_bk), (b_tl, t_tl, a_tl, x0_tl))
            if cos_bt > 0.98:
                print(f"[coshGT2][ch {j}] WARNING: potentials nearly colinear "
                      f"(cos={cos_bt:.3f}); move the bulk/tail border (bulk_quantile) "
                      f"or widen the gap between regimes.")

            print(f"[coshGT2][ch {j}] c={c_:.4f} s={s_:.4f} | "
                  f"bulk(b={b_bk:.3f},t={t_bk:.3f},a={a_bk:.4f},x0={x0_bk:.4f}) | "
                  f"tail(b={b_tl:.3f},t={t_tl:.3f},a={a_tl:.4f},x0={x0_tl:.4f}) | "
                  f"cos={cos_bt:.3f}")

            bb_b += [b_bk]; tt_b += [t_bk]; aa_b += [a_bk]; xx_b += [x0_bk]
            bb_t += [b_tl]; tt_t += [t_tl]; aa_t += [a_tl]; xx_t += [x0_tl]
            cc += [c_]; ss += [s_]; cosines += [cos_bt]

        dtype = x.dtype if x.is_floating_point() else torch.float32
        dev = x.device
        mk = lambda L: torch.tensor(L, dtype=dtype, device=dev)
        self.b_bulk, self.t_bulk, self.a_bulk, self.x0_bulk = mk(bb_b), mk(tt_b), mk(aa_b), mk(xx_b)
        self.b_tail, self.t_tail, self.a_tail, self.x0_tail = mk(bb_t), mk(tt_t), mk(aa_t), mk(xx_t)
        self.c, self.s = mk(cc), mk(ss)
        self.cos_bt = mk(cosines)

    def fit(self, x, **kwargs):
        return self.fit_reference(x, **kwargs)

    # ------------------------------------------------------------------
    # Stable torch building blocks
    # ------------------------------------------------------------------
    @staticmethod
    def _logcosh(z):
        """Numerically stable log cosh(z) in torch."""
        az = z.abs()
        return az + F.softplus(-2.0 * az) - math.log(2.0)

    def _params_bulk(self, device):
        return (self.a_bulk.to(device)[None, :, None], self.b_bulk.to(device)[None, :, None],
                self.t_bulk.to(device)[None, :, None], self.x0_bulk.to(device)[None, :, None])

    def _params_tail(self, device):
        return (self.a_tail.to(device)[None, :, None], self.b_tail.to(device)[None, :, None],
                self.t_tail.to(device)[None, :, None], self.x0_tail.to(device)[None, :, None])

    def _log_u(self, z, a, b, x0):
        """log_u = b * log(g_a(z)/x0),   g_a(z) = |z| cosh(az)."""
        az = torch.sqrt(z ** 2 + self.eps_abs)
        return b * (torch.log(az) + self._logcosh(a * z) - torch.log(x0))

    def _phi(self, z, a, b, t, x0):
        """phi(z) = (t/b) * softplus(log_u)  =  -log p(z) + const."""
        return (t / b) * F.softplus(self._log_u(z, a, b, x0))

    def _dphi(self, z, a, b, t, x0):
        """phi'(z) = t * sigmoid(log_u) * ( z/(z^2+eps) + a*tanh(a z) )."""
        log_u = self._log_u(z, a, b, x0)
        dlog_g = z / (z ** 2 + self.eps_abs) + a * torch.tanh(a * z)
        return t * torch.sigmoid(log_u) * dlog_g

    def _windows(self, z):
        """Smooth partition of unity on |z|:  (w_tail, w_bulk)."""
        c = self.c.to(z.device)[None, :, None]
        s = self.s.to(z.device)[None, :, None]
        az = torch.sqrt(z ** 2 + self.eps_abs)
        w_tail = torch.sigmoid((az - c) / s)
        return w_tail, 1.0 - w_tail

    # ------------------------------------------------------------------
    # forward:  [w_bulk*phi_bulk ; w_tail*phi_tail] averaged over time
    #           -> (B, 2J)   (first J = bulk, last J = tail)
    # ------------------------------------------------------------------
    def forward(self, x, *args):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real        # (B, J, T)
        w_tail, w_bulk = self._windows(z)
        a_b, b_b, t_b, x0_b = self._params_bulk(x.device)
        a_t, b_t, t_t, x0_t = self._params_tail(x.device)
        phi_bulk = (w_bulk * self._phi(z, a_b, b_b, t_b, x0_b)).mean(-1)   # (B, J)
        phi_tail = (w_tail * self._phi(z, a_t, b_t, t_t, x0_t)).mean(-1)   # (B, J)
        return torch.cat([phi_bulk, phi_tail], dim=1)                      # (B, 2J)

    # ------------------------------------------------------------------
    # grad:  d/dx of each windowed potential, back-projected through filters.
    #        Product rule:  d/dz [w(z) phi(z)] = w'(z) phi(z) + w(z) phi'(z),
    #        with  w_tail'(z) = w_tail(1-w_tail) * (z/|z|) / s,  w_bulk' = -w_tail'.
    #        Returns (B, 2J, T), or (B, 1, T) if v (length 2J) is given.
    # ------------------------------------------------------------------
    def grad(self, x, v=None, means=None):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real        # (B, J, T)
        a_b, b_b, t_b, x0_b = self._params_bulk(x.device)
        a_t, b_t, t_t, x0_t = self._params_tail(x.device)

        c = self.c.to(x.device)[None, :, None]
        s = self.s.to(x.device)[None, :, None]
        az = torch.sqrt(z ** 2 + self.eps_abs)
        w_tail = torch.sigmoid((az - c) / s)
        w_bulk = 1.0 - w_tail
        dw_tail = w_tail * (1.0 - w_tail) * (z / az) / s           # d w_tail / dz
        dw_bulk = -dw_tail

        phi_b = self._phi(z, a_b, b_b, t_b, x0_b); dphi_b = self._dphi(z, a_b, b_b, t_b, x0_b)
        phi_t = self._phi(z, a_t, b_t, t_t, x0_t); dphi_t = self._dphi(z, a_t, b_t, t_t, x0_t)
        D_bulk = dw_bulk * phi_b + w_bulk * dphi_b                 # (B, J, T)
        D_tail = dw_tail * phi_t + w_tail * dphi_t                 # (B, J, T)

        def backproj(D):
            return torch.fft.ifft(torch.fft.fft(D) * filters).real / x.shape[-1]

        grad_coeff = torch.cat([backproj(D_bulk), backproj(D_tail)], dim=1)  # (B, 2J, T)

        if v is None:
            return grad_coeff
        return (grad_coeff * v[None, :, None]).sum(1)[:, None]     # (B, 1, T)

    # ------------------------------------------------------------------
    def summary(self):
        self._check_fitted()
        bb = self.b_bulk.cpu().numpy(); tb = self.t_bulk.cpu().numpy()
        ab = self.a_bulk.cpu().numpy(); xb = self.x0_bulk.cpu().numpy()
        bt = self.b_tail.cpu().numpy(); tt = self.t_tail.cpu().numpy()
        at = self.a_tail.cpu().numpy(); xt = self.x0_tail.cpu().numpy()
        c = self.c.cpu().numpy(); s = self.s.cpu().numpy()
        cos = self.cos_bt.cpu().numpy() if self.cos_bt is not None else np.zeros_like(c)
        print(f"{'Ch':>3} {'c':>9} {'s':>8} | "
              f"{'b_bk':>6} {'t_bk':>6} {'a_bk':>6} {'x0_bk':>8} | "
              f"{'b_tl':>6} {'t_tl':>6} {'a_tl':>6} {'x0_tl':>8} | {'cos':>5}")
        print("-" * 98)
        for j in range(len(bb)):
            print(f"{j:>3d} {c[j]:>9.4f} {s[j]:>8.4f} | "
                  f"{bb[j]:>6.3f} {tb[j]:>6.3f} {ab[j]:>6.3f} {xb[j]:>8.4f} | "
                  f"{bt[j]:>6.3f} {tt[j]:>6.3f} {at[j]:>6.3f} {xt[j]:>8.4f} | "
                  f"{cos[j]:>5.3f}")


# coshGt but fitting on two separate windows 
class Scalar_coshgt_imag:
    """
    Per-channel cosh-tempered Generalized-t (coshGT) potential for wavelet
    coefficients, fitted SEPARATELY on the bulk and on the tails of each
    channel's coefficient histogram.

    Why two fits
    ------------
    A single coshGT must compromise between two regimes that obey different
    laws:
        Body  (|z| small):  p(z) ~ |z|^b                    -> set by  b, x0
        Tail  (|z| large):  p(z) ~ |z|^{-t} e^{-a t |z|}     -> set by  t, a
    Fitting the whole histogram at once lets the dense body dominate the
    likelihood and washes out the (sparse but decisive) tail shape.  We
    therefore run two *weighted* maximum-likelihood fits and expose two
    *windowed* energy terms per channel:

        bulk fit :  weights w_bulk(z) ~ 1 on the body, ~0 on the tail
        tail fit :  weights w_tail(z) ~ 1 on the tail, ~0 on the body

        Phi_bulk(z) = w_bulk(z) * phi_bulk(z)    (active mainly on the body)
        Phi_tail(z) = w_tail(z) * phi_tail(z)    (active mainly on the tail)

    forward() returns (B, 2J) -> [bulk_0..J-1, tail_0..J-1] and
    num_coefficients = 2J.  grad()/v follow the same ordering.

    Non-colinearity (guaranteed by construction)
    --------------------------------------------
    The windows form a smooth partition of unity on |z|:
        w_tail(z) = sigmoid((|z| - c)/s),   w_bulk(z) = 1 - w_tail(z).
    Because w_bulk and w_tail have essentially disjoint support, there is no
    constant lambda with  w_bulk*phi_bulk == lambda * w_tail*phi_tail  for all
    z: where one feature is non-zero the other is ~0, and (1 - w_tail) is not
    proportional to a non-constant sigmoid w_tail.  Hence the two energy terms
    are linearly independent / not colinear.  fit_reference() additionally
    measures the cosine between the two sampled potentials per channel
    (self.cos_bt) and warns if it ever approaches 1.

    Choice of the bulk/tail border c
    --------------------------------
    c is set per channel from a high quantile of |z| (default 0.90): the body
    is the dense central mass, the tail the sparse remainder.  The transition
    width s is set from the spread of |z| between two quantiles
    (default 0.85-0.97) so the switch is gentle and data-adaptive.

    Stored per channel (tensors of shape (J,)), with _bulk / _tail suffixes:
        b, t, a, x0  for each regime, plus the border c, the width s and the
        diagnostic cosine cos_bt.
    """

    def __init__(self, filters, eps_abs=1e-6, eps_scale=1e-6,
                 a_max=5.0, b_bounds=(0.05, 10.0), t_bounds=(0.5, 50.0),
                 bulk_quantile=0.99, trans_quantiles=(0.985, 0.995),
                 min_eff_samples=50.0):
        self.filters           = filters
        self.num_coefficients  = 2 * filters.shape[1]      # bulk + tail per channel
        self.eps_abs           = eps_abs
        self.eps_scale         = eps_scale
        self.a_max             = a_max
        self.b_bounds          = b_bounds
        self.t_bounds          = t_bounds
        self.bulk_quantile     = bulk_quantile
        self.trans_quantiles   = trans_quantiles
        self.min_eff_samples   = min_eff_samples
        # bulk params (J,)
        self.b_bulk = self.t_bulk = self.a_bulk = self.x0_bulk = None
        # tail params (J,)
        self.b_tail = self.t_tail = self.a_tail = self.x0_tail = None
        # window params (J,) and diagnostic
        self.c = self.s = None
        self.cos_bt = None

    @property
    def is_fitted(self):
        return self.a_bulk is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Scalar_coshgt must be fit_reference'd first.")

    # ------------------------------------------------------------------
    # numpy helpers  (fitting only — all in (b, t, a, x0) coords)
    # ------------------------------------------------------------------
    @staticmethod
    def _logcosh_np(z):
        """Numerically stable log cosh(z)."""
        z = np.abs(z)
        return z + np.log1p(np.exp(-2.0 * z)) - np.log(2.0)

    @classmethod
    def _log_u_np(cls, x, b, a, x0):
        """log_u = b * log(g_a(x)/x0),   g_a(x) = |x| cosh(ax)."""
        lx = np.log(np.maximum(np.abs(x), 1e-300))
        return b * (lx + cls._logcosh_np(a * x) - np.log(x0))

    @classmethod
    def _logZ_np(cls, b, t, a, x0):
        """
        log normalisation constant.  Returns np.inf on numerical failure
        so the optimizer sees a large-penalty signal rather than nan/crash.
        """
        try:
            f = lambda s: np.exp(-(t / b) * np.logaddexp(0.0,
                                   cls._log_u_np(s, b, a, x0)))
            I, _ = quad(f, 0.0, np.inf, limit=200)
            if not np.isfinite(I) or I <= 0.0:
                return np.inf
            return np.log(2.0 * I)
        except Exception:
            return np.inf

    @classmethod
    def _logpdf_np(cls, x, b, t, a, x0):
        lZ = cls._logZ_np(b, t, a, x0)
        if not np.isfinite(lZ):
            return np.full_like(x, -np.inf, dtype=float)
        return -(t / b) * np.logaddexp(0.0, cls._log_u_np(x, b, a, x0)) - lZ

    @staticmethod
    def _weighted_median(values, weights):
        """Weighted median of `values` (used for a region-aware robust scale)."""
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if values.size == 0:
            return 0.0
        order = np.argsort(values)
        v, w = values[order], weights[order]
        cw = np.cumsum(w)
        if cw[-1] <= 0:
            return float(np.median(values))
        idx = int(np.searchsorted(cw, 0.5 * cw[-1]))
        idx = min(idx, len(v) - 1)
        return float(v[idx])

    # ------------------------------------------------------------------
    # Per-channel WEIGHTED MAP fit in (b, t, a, x0).
    #
    #   * `weights` (>=0, same shape as h_raw) reweight the log-likelihood so
    #     the fit concentrates on the bulk or on the tail.
    #   * Data is normalised to a region-aware unit scale before fitting and
    #     x0 is rescaled back afterwards — this breaks the a/x0 co-linearity.
    #   * MAP penalty lam*a (half-normal on a, pulls toward pure GT).
    #   * Model selection: weighted LR test GT (a=0) vs tempered (a free).
    # ------------------------------------------------------------------
    @classmethod
    def _fit_channel(cls, h_raw, weights=None,
                     b0=1.0, t0=4.0, a0=0.1,
                     lam=1.0, lr_thresh=2.0,
                     b_bounds=(0.05, 10.0), t_bounds=(0.5, 50.0),
                     a_max=5.0, eps_scale=1e-6):
        """Returns (b, t, a, x0) with a in [0, a_max] and x0 in original units."""
        h_raw = np.asarray(h_raw, dtype=float)
        if weights is None:
            weights = np.ones_like(h_raw)
        w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
        if w.sum() <= 0:
            w = np.ones_like(h_raw)

        # --- region-aware robust scale (weighted MAD about 0; coeffs are
        #     zero-mean & symmetric) → normalise the region to ~unit scale ---
        h_scale = float(1.4826 * cls._weighted_median(np.abs(h_raw), w))
        if not (h_scale > 0):
            h_scale = float(np.sqrt((w * h_raw ** 2).sum() / w.sum()))
        h_scale = h_scale or 1.0
        h = h_raw / h_scale
        x0_unit = 1.0
        a_floor = 1e-6

        def unpack(th, free_a):
            b_ = float(np.clip(np.exp(th[0]), *b_bounds))
            t_ = float(np.clip(np.exp(th[1]), *t_bounds))
            a_ = float(np.clip(np.exp(th[2]), a_floor, a_max)) if free_a else a_floor
            return b_, t_, a_

        # ---- full model (a free) ----
        th0_full = np.array([np.log(b0), np.log(t0), np.log(a0)])

        def nll_full(th):
            b_, t_, a_ = unpack(th, free_a=True)
            if not np.isfinite(cls._logZ_np(b_, t_, a_, x0_unit)):
                return 1e12
            ll  = float((w * cls._logpdf_np(h, b_, t_, a_, x0_unit)).sum())
            return -ll + lam * a_

        res_full = minimize(nll_full, th0_full, method="Nelder-Mead",
                            options=dict(xatol=1e-5, fatol=1e-5, maxiter=15000))
        b_f, t_f, a_f = unpack(res_full.x, free_a=True)
        ll_full = float((w * cls._logpdf_np(h, b_f, t_f, a_f, x0_unit)).sum())

        # ---- GT limit (a pinned at floor) ----
        th0_gt = np.array([np.log(b0), np.log(t0)])

        def nll_gt(th):
            b_, t_, a_ = unpack(np.append(th, np.log(a_floor)), free_a=False)
            if not np.isfinite(cls._logZ_np(b_, t_, a_, x0_unit)):
                return 1e12
            return -float((w * cls._logpdf_np(h, b_, t_, a_, x0_unit)).sum())

        res_gt = minimize(nll_gt, th0_gt, method="Nelder-Mead",
                          options=dict(xatol=1e-5, fatol=1e-5, maxiter=15000))
        b_g, t_g, a_g = unpack(np.append(res_gt.x, np.log(a_floor)), free_a=False)
        ll_gt = float((w * cls._logpdf_np(h, b_g, t_g, a_g, x0_unit)).sum())

        # ---- weighted likelihood-ratio model selection ----
        if 2.0 * (ll_full - ll_gt) > lr_thresh:
            b_out, t_out, a_out = b_f, t_f, a_f
        else:
            b_out, t_out, a_out = b_g, t_g, 0.0

        x0_out = h_scale * x0_unit   # |x|_typical=1 in normalised space
        return b_out, t_out, a_out, x0_out

    # ------------------------------------------------------------------
    # Diagnostic: cosine between the two windowed, mean-removed potentials
    # sampled on a |z| grid.  ~0 => well separated, ->1 => colinear.
    # ------------------------------------------------------------------
    @classmethod
    def _potential_cosine(cls, c, s, par_bulk, par_tail, n=1024):
        b_b, t_b, a_b, x0_b = par_bulk
        b_t, t_t, a_t, x0_t = par_tail
        span = c + 10.0 * s
        zg = np.linspace(-span, span, n)
        az = np.abs(zg)
        wt = 1.0 / (1.0 + np.exp(-(az - c) / s))
        wb = 1.0 - wt
        def phi(zz, b, t, a, x0):
            return (t / b) * np.logaddexp(0.0, cls._log_u_np(zz, b, a, x0))
        fb = wb * phi(zg, b_b, t_b, a_b, x0_b)
        ft = wt * phi(zg, b_t, t_t, a_t, x0_t)
        fb = fb - fb.mean(); ft = ft - ft.mean()
        denom = np.linalg.norm(fb) * np.linalg.norm(ft)
        return float(abs(fb @ ft) / denom) if denom > 0 else 0.0

    # ------------------------------------------------------------------
    # fit_reference: fit bulk + tail for all channels from a batch x
    # ------------------------------------------------------------------
    def fit_reference(self, x, lam=1.0, lr_thresh=2.0,
                      bulk_quantile=None, trans_quantiles=None):
        if bulk_quantile is None:
            bulk_quantile = self.bulk_quantile
        if trans_quantiles is None:
            trans_quantiles = self.trans_quantiles

        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).imag.detach().cpu().numpy()

        bb_b, tt_b, aa_b, xx_b = [], [], [], []
        bb_t, tt_t, aa_t, xx_t = [], [], [], []
        cc, ss, cosines = [], [], []

        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            ah = np.abs(h)

            if h.size < 10:
                b_, t_, x0_ = 1.0, 4.0, 1.0
                c_ = float(np.median(ah)) if ah.size else 1.0
                s_ = max(0.1 * c_, self.eps_scale)
                bb_b += [b_]; tt_b += [t_]; aa_b += [0.0]; xx_b += [x0_]
                bb_t += [b_]; tt_t += [max(t_, 1.0)]; aa_t += [0.0]; xx_t += [x0_]
                cc += [c_]; ss += [s_]; cosines += [0.0]
                continue

            # --- bulk/tail border c and transition width s from |z| quantiles ---
            c_ = float(np.quantile(ah, bulk_quantile))
            q_lo, q_hi = trans_quantiles
            band = float(np.quantile(ah, q_hi) - np.quantile(ah, q_lo))
            s_ = band / 4.0                      # ~ +/-2 logistic scales over the band
            s_ = max(s_, 0.05 * (c_ + self.eps_scale), self.eps_scale)

            # --- smooth partition-of-unity windows on |z| ---
            w_tail = 1.0 / (1.0 + np.exp(-(ah - c_) / s_))
            w_bulk = 1.0 - w_tail

            eff_tail = (w_tail.sum() ** 2) / max((w_tail ** 2).sum(), 1e-12)
            if eff_tail < self.min_eff_samples:
                print(f"[coshGT2][ch {j}] tail eff. N={eff_tail:.1f} < "
                      f"{self.min_eff_samples:.0f}: tail fit may be weak "
                      f"(consider lowering bulk_quantile).")

            # --- weighted fits: bulk (body shape) and tail (tail shape) ---
            try:
                b_bk, t_bk, a_bk, x0_bk = self._fit_channel(
                    h, weights=w_bulk, a0=0.05, lam=lam, lr_thresh=lr_thresh,
                    b_bounds=self.b_bounds, t_bounds=self.t_bounds,
                    a_max=self.a_max, eps_scale=self.eps_scale)
            except Exception as e:
                print(f"[coshGT2][ch {j}] bulk fit failed ({e}) -> fallback")
                b_bk, t_bk, a_bk, x0_bk = 1.0, 6.0, 0.0, float(np.std(h) + self.eps_scale)

            try:
                b_tl, t_tl, a_tl, x0_tl = self._fit_channel(
                    h, weights=w_tail, a0=0.10, lam=lam, lr_thresh=lr_thresh,
                    b_bounds=self.b_bounds, t_bounds=self.t_bounds,
                    a_max=self.a_max, eps_scale=self.eps_scale)
            except Exception as e:
                print(f"[coshGT2][ch {j}] tail fit failed ({e}) -> fallback")
                b_tl, t_tl, a_tl, x0_tl = 1.0, 3.0, 0.0, float(np.std(h) + self.eps_scale)

            # --- clip to bounds ---
            b_bk = float(np.clip(b_bk, *self.b_bounds)); t_bk = float(np.clip(t_bk, *self.t_bounds))
            a_bk = float(np.clip(a_bk, 0.0, self.a_max)); x0_bk = float(max(x0_bk, self.eps_scale))
            b_tl = float(np.clip(b_tl, *self.b_bounds)); t_tl = float(np.clip(t_tl, *self.t_bounds))
            a_tl = float(np.clip(a_tl, 0.0, self.a_max)); x0_tl = float(max(x0_tl, self.eps_scale))

            # --- non-colinearity diagnostic ---
            cos_bt = self._potential_cosine(
                c_, s_, (b_bk, t_bk, a_bk, x0_bk), (b_tl, t_tl, a_tl, x0_tl))
            if cos_bt > 0.98:
                print(f"[coshGT2][ch {j}] WARNING: potentials nearly colinear "
                      f"(cos={cos_bt:.3f}); move the bulk/tail border (bulk_quantile) "
                      f"or widen the gap between regimes.")

            print(f"[coshGT2][ch {j}] c={c_:.4f} s={s_:.4f} | "
                  f"bulk(b={b_bk:.3f},t={t_bk:.3f},a={a_bk:.4f},x0={x0_bk:.4f}) | "
                  f"tail(b={b_tl:.3f},t={t_tl:.3f},a={a_tl:.4f},x0={x0_tl:.4f}) | "
                  f"cos={cos_bt:.3f}")

            bb_b += [b_bk]; tt_b += [t_bk]; aa_b += [a_bk]; xx_b += [x0_bk]
            bb_t += [b_tl]; tt_t += [t_tl]; aa_t += [a_tl]; xx_t += [x0_tl]
            cc += [c_]; ss += [s_]; cosines += [cos_bt]

        dtype = x.dtype if x.is_floating_point() else torch.float32
        dev = x.device
        mk = lambda L: torch.tensor(L, dtype=dtype, device=dev)
        self.b_bulk, self.t_bulk, self.a_bulk, self.x0_bulk = mk(bb_b), mk(tt_b), mk(aa_b), mk(xx_b)
        self.b_tail, self.t_tail, self.a_tail, self.x0_tail = mk(bb_t), mk(tt_t), mk(aa_t), mk(xx_t)
        self.c, self.s = mk(cc), mk(ss)
        self.cos_bt = mk(cosines)

    def fit(self, x, **kwargs):
        return self.fit_reference(x, **kwargs)

    # ------------------------------------------------------------------
    # Stable torch building blocks
    # ------------------------------------------------------------------
    @staticmethod
    def _logcosh(z):
        """Numerically stable log cosh(z) in torch."""
        az = z.abs()
        return az + F.softplus(-2.0 * az) - math.log(2.0)

    def _params_bulk(self, device):
        return (self.a_bulk.to(device)[None, :, None], self.b_bulk.to(device)[None, :, None],
                self.t_bulk.to(device)[None, :, None], self.x0_bulk.to(device)[None, :, None])

    def _params_tail(self, device):
        return (self.a_tail.to(device)[None, :, None], self.b_tail.to(device)[None, :, None],
                self.t_tail.to(device)[None, :, None], self.x0_tail.to(device)[None, :, None])

    def _log_u(self, z, a, b, x0):
        """log_u = b * log(g_a(z)/x0),   g_a(z) = |z| cosh(az)."""
        az = torch.sqrt(z ** 2 + self.eps_abs)
        return b * (torch.log(az) + self._logcosh(a * z) - torch.log(x0))

    def _phi(self, z, a, b, t, x0):
        """phi(z) = (t/b) * softplus(log_u)  =  -log p(z) + const."""
        return (t / b) * F.softplus(self._log_u(z, a, b, x0))

    def _dphi(self, z, a, b, t, x0):
        """phi'(z) = t * sigmoid(log_u) * ( z/(z^2+eps) + a*tanh(a z) )."""
        log_u = self._log_u(z, a, b, x0)
        dlog_g = z / (z ** 2 + self.eps_abs) + a * torch.tanh(a * z)
        return t * torch.sigmoid(log_u) * dlog_g

    def _windows(self, z):
        """Smooth partition of unity on |z|:  (w_tail, w_bulk)."""
        c = self.c.to(z.device)[None, :, None]
        s = self.s.to(z.device)[None, :, None]
        az = torch.sqrt(z ** 2 + self.eps_abs)
        w_tail = torch.sigmoid((az - c) / s)
        return w_tail, 1.0 - w_tail

    # ------------------------------------------------------------------
    # forward:  [w_bulk*phi_bulk ; w_tail*phi_tail] averaged over time
    #           -> (B, 2J)   (first J = bulk, last J = tail)
    # ------------------------------------------------------------------
    def forward(self, x, *args):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).imag        # (B, J, T)
        w_tail, w_bulk = self._windows(z)
        a_b, b_b, t_b, x0_b = self._params_bulk(x.device)
        a_t, b_t, t_t, x0_t = self._params_tail(x.device)
        phi_bulk = (w_bulk * self._phi(z, a_b, b_b, t_b, x0_b)).mean(-1)   # (B, J)
        phi_tail = (w_tail * self._phi(z, a_t, b_t, t_t, x0_t)).mean(-1)   # (B, J)
        return torch.cat([phi_bulk, phi_tail], dim=1)                      # (B, 2J)

    # ------------------------------------------------------------------
    # grad:  d/dx of each windowed potential, back-projected through filters.
    #        Product rule:  d/dz [w(z) phi(z)] = w'(z) phi(z) + w(z) phi'(z),
    #        with  w_tail'(z) = w_tail(1-w_tail) * (z/|z|) / s,  w_bulk' = -w_tail'.
    #        Returns (B, 2J, T), or (B, 1, T) if v (length 2J) is given.
    # ------------------------------------------------------------------
    def grad(self, x, v=None, means=None):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).imag        # (B, J, T)
        a_b, b_b, t_b, x0_b = self._params_bulk(x.device)
        a_t, b_t, t_t, x0_t = self._params_tail(x.device)

        c = self.c.to(x.device)[None, :, None]
        s = self.s.to(x.device)[None, :, None]
        az = torch.sqrt(z ** 2 + self.eps_abs)
        w_tail = torch.sigmoid((az - c) / s)
        w_bulk = 1.0 - w_tail
        dw_tail = w_tail * (1.0 - w_tail) * (z / az) / s           # d w_tail / dz
        dw_bulk = -dw_tail

        phi_b = self._phi(z, a_b, b_b, t_b, x0_b); dphi_b = self._dphi(z, a_b, b_b, t_b, x0_b)
        phi_t = self._phi(z, a_t, b_t, t_t, x0_t); dphi_t = self._dphi(z, a_t, b_t, t_t, x0_t)
        D_bulk = dw_bulk * phi_b + w_bulk * dphi_b                 # (B, J, T)
        D_tail = dw_tail * phi_t + w_tail * dphi_t                 # (B, J, T)

        def backproj(D):
            return torch.fft.ifft(torch.fft.fft(D) * filters).imag / x.shape[-1]

        grad_coeff = torch.cat([backproj(D_bulk), backproj(D_tail)], dim=1)  # (B, 2J, T)

        if v is None:
            return grad_coeff
        return (grad_coeff * v[None, :, None]).sum(1)[:, None]     # (B, 1, T)

    # ------------------------------------------------------------------
    def summary(self):
        self._check_fitted()
        bb = self.b_bulk.cpu().numpy(); tb = self.t_bulk.cpu().numpy()
        ab = self.a_bulk.cpu().numpy(); xb = self.x0_bulk.cpu().numpy()
        bt = self.b_tail.cpu().numpy(); tt = self.t_tail.cpu().numpy()
        at = self.a_tail.cpu().numpy(); xt = self.x0_tail.cpu().numpy()
        c = self.c.cpu().numpy(); s = self.s.cpu().numpy()
        cos = self.cos_bt.cpu().numpy() if self.cos_bt is not None else np.zeros_like(c)
        print(f"{'Ch':>3} {'c':>9} {'s':>8} | "
              f"{'b_bk':>6} {'t_bk':>6} {'a_bk':>6} {'x0_bk':>8} | "
              f"{'b_tl':>6} {'t_tl':>6} {'a_tl':>6} {'x0_tl':>8} | {'cos':>5}")
        print("-" * 98)
        for j in range(len(bb)):
            print(f"{j:>3d} {c[j]:>9.4f} {s[j]:>8.4f} | "
                  f"{bb[j]:>6.3f} {tb[j]:>6.3f} {ab[j]:>6.3f} {xb[j]:>8.4f} | "
                  f"{bt[j]:>6.3f} {tt[j]:>6.3f} {at[j]:>6.3f} {xt[j]:>8.4f} | "
                  f"{cos[j]:>5.3f}")


from scipy.optimize import minimize_scalar
from scipy.integrate import IntegrationWarning
import warnings


class Scalar_coshgt_multiregion(Scalar_coshgt):
    """
    Generalization of Scalar_coshgt (bulk/tail, 2 windows) to an arbitrary
    number `n_regions` >= 2 of windowed coshGT regimes ("multi-region coshGT").

    Motivation
    ----------
    Scalar_coshgt fits a coshGT separately on the bulk and tail of |z|. In
    practice two regimes are sometimes not enough to track a histogram across
    its full range (e.g. core / shoulder / tail, or core / near-tail /
    far-tail / extreme-tail). This class partitions |z| into `n_regions`
    contiguous, smoothly-blended windows and fits ONE coshGT per window.
    Everything that doesn't depend on the number of regions (the (b,t,a,x0)
    parametrization, the weighted-MLE channel fit with GT-vs-tempered model
    selection, the cosine non-colinearity diagnostic, the stable torch
    building blocks) is inherited UNCHANGED from Scalar_coshgt.

    Toggle
    ------
    `n_regions=2` recovers exactly Scalar_coshgt's bulk/tail windowing.
    `n_regions=3` or `n_regions=4` are the requested 3-/4-area splits: pick
    one via this single constructor argument. `Scalar_coshgt_3region` and
    `Scalar_coshgt_4region` below are thin convenience subclasses that pin
    this argument, for call sites that want a named class instead of a
    kwarg toggle.

    Smooth partition of unity (telescoping sigmoids)
    --------------------------------------------------
    Let K = n_regions - 1 interior boundaries c_1 < c_2 < ... < c_K on |z|,
    each with its own transition width s_k. Define the "right of boundary k"
    gate
        r_k(z) = sigmoid((|z| - c_k) / s_k),     r_0 := 1,   r_{K+1} := 0 .
    Region i (i = 0 .. n_regions-1, innermost/bulk to outermost/tail) gets
    window
        w_i(z) = r_i(z) - r_{i+1}(z) .
    These telescope to sum_i w_i(z) = r_0 - r_{K+1} = 1 for every z (an exact
    partition of unity), and w_i >= 0 everywhere as long as the boundaries
    are increasing (each gate dominates the next, as in Scalar_coshgt's own
    c). For n_regions=2 this is exactly Scalar_coshgt's
    (w_bulk, w_tail) = (1 - w_tail, w_tail).

    Optimizing the boundaries
    --------------------------
    A boundary's *position* only enters the windows of its two neighbouring
    regions (telescoping cancels it out of every other window). So boundary
    k is optimized by a bounded 1-D line search (`scipy.optimize.
    minimize_scalar`, Brent on a bounded interval) that, at each trial
    position, refits ONLY the two adjacent regions (via the inherited,
    unmodified `Scalar_coshgt._fit_channel`) and scores the trial by their
    total weighted log-likelihood. Boundaries are swept left-to-right for
    `boundary_opt_rounds` coordinate-descent passes, each boundary optimized
    holding all others fixed -- i.e. block coordinate ascent on the joint
    weighted log-likelihood across all regions. Set
    `optimize_boundaries=False` to skip this and just keep the fixed
    `boundary_quantiles` (cheap fallback, in the spirit of Scalar_coshgt's
    fixed `bulk_quantile`).

    Stored per channel (tensors of shape (n_regions, J)):
        b, t, a, x0   -- one coshGT per region, stacked along dim 0.
    Stored per boundary (tensors of shape (n_regions-1, J)):
        c, s          -- boundary position and transition width.
    Diagnostic (n_regions-1, J): cos_adjacent -- cosine between each pair of
    *adjacent* regions' windowed potentials (same role/threshold as
    Scalar_coshgt.cos_bt, computed with the inherited `_potential_cosine`).
    """

    # default boundary-quantile seeds, skewed toward the tail like
    # Scalar_coshgt's own bulk_quantile=0.90 default; only used to *seed*
    # the search when optimize_boundaries=True, and used as-is otherwise.
    _DEFAULT_BOUNDARY_QUANTILES = {
        2: [0.990],
        3: [0.80, 0.95],
        4: [0.55, 0.80, 0.95],
    }

    def __init__(self, filters, n_regions=3, eps_abs=1e-6, eps_scale=1e-6,
                 a_max=5.0, b_bounds=(0.05, 10.0), t_bounds=(0.5, 50.0),
                 boundary_quantiles=None, trans_pad=0.04,
                 optimize_boundaries=False, boundary_opt_rounds=2,
                 boundary_search_bounds=(0.02, 0.98), min_boundary_gap=0.04,
                 min_eff_samples=50.0):
        if n_regions < 2:
            raise ValueError("n_regions must be >= 2 (Scalar_coshgt already "
                              "covers the 2-region case).")
        # NOTE: intentionally does NOT call Scalar_coshgt.__init__ -- that
        # allocates bulk/tail-specific attributes (b_bulk, t_bulk, ...) we
        # replace with stacked (n_regions, J) tensors below. Every method we
        # inherit (the _fit_channel/_logpdf_np/_phi/_dphi/... family) only
        # ever touches self.eps_abs/eps_scale/a_max/b_bounds/t_bounds, which
        # are set here identically to Scalar_coshgt.
        self.filters            = filters
        self.n_regions           = n_regions
        self.num_coefficients    = n_regions * filters.shape[1]
        self.eps_abs             = eps_abs
        self.eps_scale           = eps_scale
        self.a_max               = a_max
        self.b_bounds            = b_bounds
        self.t_bounds            = t_bounds
        self.min_eff_samples     = min_eff_samples
        self.trans_pad           = trans_pad
        self.optimize_boundaries = optimize_boundaries
        self.boundary_opt_rounds = boundary_opt_rounds
        self.boundary_search_bounds = boundary_search_bounds
        self.min_boundary_gap    = min_boundary_gap

        K = n_regions - 1
        if boundary_quantiles is None:
            boundary_quantiles = self._DEFAULT_BOUNDARY_QUANTILES.get(
                n_regions,
                list(np.linspace(0.0, 1.0, n_regions + 1)[1:-1]))
        if len(boundary_quantiles) != K:
            raise ValueError(f"boundary_quantiles must have length "
                              f"n_regions-1={K}, got {len(boundary_quantiles)}")
        self.boundary_quantiles_init = list(boundary_quantiles)

        # region params (n_regions, J) once fitted
        self.b = self.t = self.a = self.x0 = None
        # boundary params (n_regions-1, J) once fitted
        self.c = self.s = None
        self.cos_adjacent = None
        self.boundary_quantiles = None   # final per-channel (K, J), for inspection

    @property
    def is_fitted(self):
        return self.a is not None

    # ------------------------------------------------------------------
    # numpy helpers specific to the multi-region windowing.
    # (_logcosh_np, _log_u_np, _logZ_np, _logpdf_np, _weighted_median,
    #  _fit_channel, _potential_cosine are inherited verbatim from
    #  Scalar_coshgt.)
    # ------------------------------------------------------------------
    def _boundary_c_s(self, ah, q):
        """Boundary center c and transition width s from a quantile q of
        |z|, via the same robust-quantile-band heuristic Scalar_coshgt uses
        for its single bulk/tail border (band of +/-trans_pad quantiles,
        width = band/4)."""
        pad = self.trans_pad
        q_lo = max(q - pad, 1e-4)
        q_hi = min(q + pad, 1.0 - 1e-4)
        c = float(np.quantile(ah, q))
        band = float(np.quantile(ah, q_hi) - np.quantile(ah, q_lo))
        s = band / 4.0
        s = max(s, 0.05 * (c + self.eps_scale), self.eps_scale)
        return c, s

    @staticmethod
    def _telescoped_weights_np(ah, c_list, s_list):
        """Partition of unity on |z| into len(c_list)+1 regions via
        telescoping sigmoid gates (see class docstring)."""
        K = len(c_list)
        r = [np.ones_like(ah)]
        for k in range(K):
            r.append(1.0 / (1.0 + np.exp(-(ah - c_list[k]) / s_list[k])))
        r.append(np.zeros_like(ah))
        return [r[i] - r[i + 1] for i in range(K + 1)]

    def _boundary_neg_ll(self, q_i, i, h, ah, c_list, s_list, lam, lr_thresh):
        """Negative total weighted log-likelihood of the two regions
        adjacent to boundary i (regions i and i+1), as a function of
        boundary i's quantile position -- every other window is unaffected
        by q_i (telescoping), so only these two regions need refitting."""
        c_i, s_i = self._boundary_c_s(ah, q_i)
        c_trial = list(c_list); c_trial[i] = c_i
        s_trial = list(s_list); s_trial[i] = s_i
        weights = self._telescoped_weights_np(ah, c_trial, s_trial)
        ll = 0.0
        with warnings.catch_warnings():
            # the optimizer routinely probes (b,t,a,x0) with a non-convergent
            # tail integral while searching; _logZ_np already turns that into
            # a finite-penalty signal (np.inf -> guarded below), so the
            # IntegrationWarning itself is just noise here.
            warnings.simplefilter('ignore', category=IntegrationWarning)
            for r_idx in (i, i + 1):
                w = weights[r_idx]
                eff = (w.sum() ** 2) / max((w ** 2).sum(), 1e-12)
                if eff < 5.0:           # near-empty window at this trial position
                    return 1e8
                try:
                    b_, t_, a_, x0_ = self._fit_channel(
                        h, weights=w, lam=lam, lr_thresh=lr_thresh,
                        b_bounds=self.b_bounds, t_bounds=self.t_bounds,
                        a_max=self.a_max, eps_scale=self.eps_scale)
                    # guard against a degenerate fitted optimum whose own
                    # log-normalizer is non-finite -- without this, logpdf is
                    # an all -inf array and `0 * -inf = nan` poisons the sum
                    # at every index where w==0 (this was the actual bug:
                    # the nan then propagates through minimize_scalar).
                    if not np.isfinite(self._logZ_np(b_, t_, a_, x0_)):
                        return 1e8
                    ll += float((w * self._logpdf_np(h, b_, t_, a_, x0_)).sum())
                except Exception:
                    return 1e8
        if not np.isfinite(ll):
            return 1e8
        return -ll

    def _optimize_boundaries_channel(self, h, ah, lam, lr_thresh):
        """Block coordinate ascent on the boundary positions: sweep each
        boundary left-to-right with a bounded scalar line search, holding
        the others fixed, for `boundary_opt_rounds` passes."""
        K = self.n_regions - 1
        q = list(self.boundary_quantiles_init)
        c_list = [None] * K; s_list = [None] * K
        for k in range(K):
            c_list[k], s_list[k] = self._boundary_c_s(ah, q[k])

        if K == 0 or not self.optimize_boundaries:
            return q, c_list, s_list

        lo_bound, hi_bound = self.boundary_search_bounds
        gap = self.min_boundary_gap
        for _round in range(self.boundary_opt_rounds):
            for i in range(K):
                q_left  = q[i - 1] if i > 0 else lo_bound
                q_right = q[i + 1] if i < K - 1 else hi_bound
                b_lo = max(q_left + gap, lo_bound)
                b_hi = min(q_right - gap, hi_bound)
                if b_lo >= b_hi:
                    continue        # neighbours too close: keep current position
                res = minimize_scalar(
                    lambda qq: self._boundary_neg_ll(
                        qq, i, h, ah, c_list, s_list, lam, lr_thresh),
                    bounds=(b_lo, b_hi), method='bounded',
                    options=dict(xatol=1e-3))
                if np.isfinite(res.x):     # defensive: never move to a nan/inf position
                    q[i] = float(res.x)
                c_list[i], s_list[i] = self._boundary_c_s(ah, q[i])
        return q, c_list, s_list

    # ------------------------------------------------------------------
    # fit_reference: optimize boundaries, then fit all n_regions regimes
    # for every channel.
    # ------------------------------------------------------------------
    def fit_reference(self, x, lam=1.0, lr_thresh=2.0):
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real.detach().cpu().numpy()

        N, K = self.n_regions, self.n_regions - 1
        all_b  = [[] for _ in range(N)]
        all_t  = [[] for _ in range(N)]
        all_a  = [[] for _ in range(N)]
        all_x0 = [[] for _ in range(N)]
        all_c, all_s, all_q, all_cos = ([[] for _ in range(K)] for _ in range(4))

        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            ah = np.abs(h)

            if h.size < 10:
                # degenerate channel: mirror Scalar_coshgt's tiny-sample branch
                c_med = float(np.median(ah)) if ah.size else 1.0
                for i in range(N):
                    all_b[i] += [1.0]; all_t[i] += [max(4.0 - i, 1.0)]
                    all_a[i] += [0.0]; all_x0[i] += [c_med if c_med > 0 else 1.0]
                for k in range(K):
                    q_fb = self.boundary_quantiles_init[k]
                    all_c[k] += [c_med * (k + 1) / (K + 1)]
                    all_s[k] += [max(0.1 * c_med, self.eps_scale)]
                    all_q[k] += [q_fb]; all_cos[k] += [0.0]
                continue

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=IntegrationWarning)
                q, c_list, s_list = self._optimize_boundaries_channel(h, ah, lam, lr_thresh)

                # final fit of ALL regions at the converged boundaries
                weights = self._telescoped_weights_np(ah, c_list, s_list)
                region_params = []
                for i in range(N):
                    w = weights[i]
                    eff = (w.sum() ** 2) / max((w ** 2).sum(), 1e-12)
                    if eff < self.min_eff_samples:
                        print(f"[coshGT{N}][ch {j}] region {i} eff. N={eff:.1f} < "
                              f"{self.min_eff_samples:.0f}: fit may be weak "
                              f"(consider fewer regions, or widen boundary_search_bounds).")
                    try:
                        b_, t_, a_, x0_ = self._fit_channel(
                            h, weights=w, lam=lam, lr_thresh=lr_thresh,
                            b_bounds=self.b_bounds, t_bounds=self.t_bounds,
                            a_max=self.a_max, eps_scale=self.eps_scale)
                    except Exception as e:
                        print(f"[coshGT{N}][ch {j}] region {i} fit failed ({e}) -> fallback")
                        b_, t_, a_, x0_ = 1.0, max(6.0 - 2 * i, 1.0), 0.0, float(np.std(h) + self.eps_scale)
                    b_  = float(np.clip(b_, *self.b_bounds))
                    t_  = float(np.clip(t_, *self.t_bounds))
                    a_  = float(np.clip(a_, 0.0, self.a_max))
                    x0_ = float(max(x0_, self.eps_scale))
                    region_params.append((b_, t_, a_, x0_))
                    all_b[i] += [b_]; all_t[i] += [t_]; all_a[i] += [a_]; all_x0[i] += [x0_]

            for k in range(K):
                cos_k = self._potential_cosine(
                    c_list[k], s_list[k], region_params[k], region_params[k + 1])
                if cos_k > 0.98:
                    print(f"[coshGT{N}][ch {j}] WARNING: regions {k} & {k + 1} nearly "
                          f"colinear (cos={cos_k:.3f}) across boundary {k}; widen "
                          f"min_boundary_gap or boundary_search_bounds.")
                all_cos[k] += [cos_k]
                all_c[k]   += [c_list[k]]
                all_s[k]   += [s_list[k]]
                all_q[k]   += [q[k]]

            q_str = " ".join(f"{v:.3f}" for v in q)
            reg_str = " | ".join(
                f"r{i}(b={p[0]:.3f},t={p[1]:.3f},a={p[2]:.4f},x0={p[3]:.4f})"
                for i, p in enumerate(region_params))
            print(f"[coshGT{N}][ch {j}] q=[{q_str}] | {reg_str}")

        dtype = x.dtype if x.is_floating_point() else torch.float32
        dev = x.device
        mk = lambda L: torch.tensor(L, dtype=dtype, device=dev)
        self.b  = torch.stack([mk(all_b[i])  for i in range(N)], dim=0)
        self.t  = torch.stack([mk(all_t[i])  for i in range(N)], dim=0)
        self.a  = torch.stack([mk(all_a[i])  for i in range(N)], dim=0)
        self.x0 = torch.stack([mk(all_x0[i]) for i in range(N)], dim=0)
        if K > 0:
            self.c = torch.stack([mk(all_c[k]) for k in range(K)], dim=0)
            self.s = torch.stack([mk(all_s[k]) for k in range(K)], dim=0)
            self.cos_adjacent = torch.stack([mk(all_cos[k]) for k in range(K)], dim=0)
            self.boundary_quantiles = np.stack(
                [np.array(all_q[k]) for k in range(K)], axis=0)
        else:
            self.c = self.s = self.cos_adjacent = None
            self.boundary_quantiles = None

    def fit(self, x, **kwargs):
        return self.fit_reference(x, **kwargs)

    # ------------------------------------------------------------------
    # torch building blocks
    # (_logcosh, _phi, _dphi are inherited verbatim from Scalar_coshgt.)
    # ------------------------------------------------------------------
    def _params_region(self, i, device):
        return (self.a[i].to(device)[None, :, None], self.b[i].to(device)[None, :, None],
                self.t[i].to(device)[None, :, None], self.x0[i].to(device)[None, :, None])

    def _windows(self, z):
        """List of n_regions windows via telescoping sigmoid gates on |z|."""
        az = torch.sqrt(z ** 2 + self.eps_abs)
        K = self.n_regions - 1
        r = [torch.ones_like(az)]
        if K > 0:
            c = self.c.to(z.device); s = self.s.to(z.device)
            for k in range(K):
                ck = c[k][None, :, None]; sk = s[k][None, :, None]
                r.append(torch.sigmoid((az - ck) / sk))
        r.append(torch.zeros_like(az))
        return [r[i] - r[i + 1] for i in range(self.n_regions)]

    # ------------------------------------------------------------------
    # forward: [w_0*phi_0 ; ... ; w_{N-1}*phi_{N-1}] averaged over time
    #          -> (B, N*J)   (region 0 = innermost/bulk .. N-1 = outermost/tail)
    # ------------------------------------------------------------------
    def forward(self, x, *args):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real        # (B, J, T)
        windows = self._windows(z)
        phis = []
        for i in range(self.n_regions):
            a_i, b_i, t_i, x0_i = self._params_region(i, x.device)
            phis.append((windows[i] * self._phi(z, a_i, b_i, t_i, x0_i)).mean(-1))
        return torch.cat(phis, dim=1)                                # (B, N*J)

    # ------------------------------------------------------------------
    # grad: d/dx of each windowed potential, back-projected through filters.
    #       Product rule:  d/dz [w_i(z) phi_i(z)] = w_i'(z) phi_i(z) + w_i(z) phi_i'(z),
    #       with w_i = r_i - r_{i+1} and dr_k/dz = r_k(1-r_k)*(z/|z|)/s_k.
    #       Returns (B, N*J, T), or (B, 1, T) if v (length N*J) is given.
    # ------------------------------------------------------------------
    def grad(self, x, v=None, means=None):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real        # (B, J, T)
        az = torch.sqrt(z ** 2 + self.eps_abs)
        K = self.n_regions - 1

        r = [torch.ones_like(az)]; dr = [torch.zeros_like(az)]
        if K > 0:
            c = self.c.to(x.device); s = self.s.to(x.device)
            for k in range(K):
                ck = c[k][None, :, None]; sk = s[k][None, :, None]
                rk = torch.sigmoid((az - ck) / sk)
                drk = rk * (1.0 - rk) * (z / az) / sk
                r.append(rk); dr.append(drk)
        r.append(torch.zeros_like(az)); dr.append(torch.zeros_like(az))

        windows  = [r[i] - r[i + 1] for i in range(self.n_regions)]
        dwindows = [dr[i] - dr[i + 1] for i in range(self.n_regions)]

        Ds = []
        for i in range(self.n_regions):
            a_i, b_i, t_i, x0_i = self._params_region(i, x.device)
            phi_i  = self._phi(z, a_i, b_i, t_i, x0_i)
            dphi_i = self._dphi(z, a_i, b_i, t_i, x0_i)
            Ds.append(dwindows[i] * phi_i + windows[i] * dphi_i)        # (B, J, T)

        def backproj(D):
            return torch.fft.ifft(torch.fft.fft(D) * filters).real / x.shape[-1]

        grad_coeff = torch.cat([backproj(D) for D in Ds], dim=1)       # (B, N*J, T)

        if v is None:
            return grad_coeff
        return (grad_coeff * v[None, :, None]).sum(1)[:, None]         # (B, 1, T)

    # ------------------------------------------------------------------
    def summary(self):
        self._check_fitted()
        N = self.n_regions
        b = self.b.cpu().numpy(); t = self.t.cpu().numpy()
        a = self.a.cpu().numpy(); x0 = self.x0.cpu().numpy()
        c = self.c.cpu().numpy() if self.c is not None else None
        s = self.s.cpu().numpy() if self.s is not None else None
        cos = self.cos_adjacent.cpu().numpy() if self.cos_adjacent is not None else None
        for j in range(b.shape[1]):
            bnd = (" ".join(f"c{k}={c[k, j]:>8.4f}(s={s[k, j]:>7.4f})" for k in range(N - 1))
                   if c is not None else "")
            reg = " | ".join(
                f"r{i}(b={b[i, j]:>6.3f},t={t[i, j]:>6.3f},a={a[i, j]:>6.3f},x0={x0[i, j]:>8.4f})"
                for i in range(N))
            cs = " ".join(f"cos{k}={cos[k, j]:>5.3f}" for k in range(N - 1)) if cos is not None else ""
            print(f"[ch {j:>3d}] {bnd} | {reg} | {cs}")


class Scalar_coshgt_3region(Scalar_coshgt_multiregion):
    """Scalar_coshgt_multiregion pinned to n_regions=3 (core / shoulder / tail).
    Convenience subclass for call sites that prefer a named class over the
    n_regions kwarg toggle; identical behaviour to
    Scalar_coshgt_multiregion(filters, n_regions=3, ...)."""
    def __init__(self, filters, **kwargs):
        kwargs.pop('n_regions', None)
        super().__init__(filters, n_regions=3, **kwargs)


class Scalar_coshgt_4region(Scalar_coshgt_multiregion):
    """Scalar_coshgt_multiregion pinned to n_regions=4
    (core / near-tail / far-tail / extreme-tail). Convenience subclass;
    identical behaviour to
    Scalar_coshgt_multiregion(filters, n_regions=4, ...)."""
    def __init__(self, filters, **kwargs):
        kwargs.pop('n_regions', None)
        super().__init__(filters, n_regions=4, **kwargs)


import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import gammaln, gammainc
from scipy.optimize import minimize, minimize_scalar
from scipy.integrate import quad


class Scalar_GGD_GenGamma:
    """
    Two-region potential per channel:
        bulk  : weight g(x),     suff. stat |x|^alpha          -> generalized Gaussian
        outer : weight 1-g(x),   suff. stats {|x|^beta, log|x|} -> generalized Gamma
                p_outer(x) ~ |x|^{-theta3} * exp(-theta2 * |x|^beta)

    g(x) = sigmoid(-(|x|-c)/s)  (smooth partition of unity, used in forward()/grad())

    Fitting uses HARD cutoff |x|<c / |x|>=c subsets (well-conditioned,
    avoids the near-boundary-mass bias smooth weighting caused before).
    Both regions are fit with TRUNCATED normalization (correctly
    accounting for the cutoff), so there is no truncation bias and no
    anchoring hack needed for plotting.

    forward()/grad() return/operate on 3 channels per scale:
        [bulk_energy, outer_beta_energy, outer_log_energy]  -> (B, 3J)
    matching the three sufficient statistics above.
    """

    def __init__(self, filters, bulk_quantile=0.99, trans_frac=0.15,
                 alpha_bounds=(0.2, 8.0), beta_bounds=(0.3, 3.0),
                 theta3_bounds=(-1.0, 1.0), min_region_samples=30,
                 eps_abs=1e-6):
        self.filters = filters
        self.num_coefficients = 3 * filters.shape[1]
        self.bulk_quantile = bulk_quantile
        self.trans_frac = trans_frac
        self.alpha_bounds = alpha_bounds
        self.beta_bounds = beta_bounds
        self.theta3_bounds = theta3_bounds
        self.min_region_samples = min_region_samples
        self.eps_abs = eps_abs

        self.alpha = self.scale_bulk = None   # bulk:  (J,)
        self.beta = self.theta2 = self.theta3 = None  # outer: (J,)
        self.c = self.s = None                # boundary, transition width (J,)
        self.pi_bulk = self.pi_outer = None

    @property
    def is_fitted(self):
        return self.alpha is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("must call fit_reference first.")

    # ------------------------------------------------------------------
    # Bulk: truncated generalized-Gaussian MLE on |x| < c
    # (1D profile over alpha; scale closed-form via regularized lower
    # incomplete gamma normalizer -- same stable construction as before)
    # ------------------------------------------------------------------
    @staticmethod
    def _ggd_logmass_leq(c, alpha, scale):
        val = gammainc(1.0 / alpha, (c / scale) ** alpha)  # P(|Z|<=c)
        return np.log(max(val, 1e-300))

    @classmethod
    def _fit_bulk_ggd(cls, h, c, alpha_bounds=(0.2, 8.0)):
        h = np.asarray(h, dtype=float)
        n = h.size
        if n < 5:
            return 1.0, float(np.std(h) + 1e-8)
        ah = np.abs(h)

        def neg_ll(theta):
            alpha = np.clip(theta[0], *alpha_bounds)
            scale = max(theta[1], 1e-8)
            logpdf = (np.log(alpha) - np.log(2.0) - np.log(scale)
                      - gammaln(1.0 / alpha) - (ah / scale) ** alpha)
            log_mass = cls._ggd_logmass_leq(c, alpha, scale)
            return -np.sum(logpdf) + n * log_mass

        alpha0 = 1.5
        scale0 = (alpha0 * np.mean(ah ** alpha0)) ** (1.0 / alpha0)
        res = minimize(neg_ll, x0=[alpha0, scale0], method="Nelder-Mead",
                        options=dict(xatol=1e-5, fatol=1e-5, maxiter=5000))
        alpha_hat = float(np.clip(res.x[0], *alpha_bounds))
        scale_hat = float(max(res.x[1], 1e-8))
        return alpha_hat, scale_hat

    # ------------------------------------------------------------------
    # Outer: generalized-Gamma MLE on |x| >= c
    #   for fixed beta: NLL(theta2,theta3) is convex (exponential family)
    #   -> stable 2D solve.  beta itself: 1D bounded profile scan.
    # ------------------------------------------------------------------
    @staticmethod
    def _outer_logZ(theta2, theta3, beta, c):
        # Map integration from [c, inf] to [0, 1] via v = c/u
        # This completely avoids quad failing on long heavy tails
        lam = theta2 * (c ** beta)
        
        def f(v):
            if v < 1e-10:  # Guard against overflow in v**-beta
                return 0.0
            return (v ** (theta3 - 2.0)) * np.exp(-lam * (v ** -beta))
        
        I_v, _ = quad(f, 0, 1, limit=200)
        if not np.isfinite(I_v) or I_v <= 0:
            return np.inf
            
        # log(2 * I) = log(2) + (1 - theta3)*log(c) + log(I_v)
        return np.log(2.0) + (1.0 - theta3) * np.log(c) + np.log(I_v)

    @classmethod
    def _fit_outer_given_beta(cls, ah_outer, c, beta, theta3_bounds):
        n = ah_outer.size
        mean_log = np.mean(np.log(ah_outer))

        def neg_ll(params):
            log_theta2, theta3 = params
            theta2 = np.exp(log_theta2)
            theta3 = np.clip(theta3, *theta3_bounds)
            logZ = cls._outer_logZ(theta2, theta3, beta, c)
            if not np.isfinite(logZ):
                return 1e12
            ll = -theta3 * mean_log - theta2 * np.mean(ah_outer ** beta) - logZ
            return -n * ll

        x0 = [np.log(1.0 / max(c, 1e-3) ** beta), 0.0]
        res = minimize(neg_ll, x0=x0, method="Nelder-Mead",
                        options=dict(xatol=1e-5, fatol=1e-5, maxiter=3000))
        theta2 = float(np.exp(res.x[0]))
        theta3 = float(np.clip(res.x[1], *theta3_bounds))
        return theta2, theta3, float(res.fun)

    @classmethod
    def _fit_outer_gengamma(cls, h, c, beta_bounds=(0.3, 3.0),
                             theta3_bounds=(-2.0, 15.0)):
        ah_outer = np.abs(np.asarray(h, dtype=float))
        ah_outer = ah_outer[ah_outer >= c]
        if ah_outer.size < 5:
            return 1.0, 1.0 / max(c, 1e-3), 0.0

        def profile_nll(beta):
            _, _, nll = cls._fit_outer_given_beta(ah_outer, c, beta, theta3_bounds)
            return nll

        res = minimize_scalar(profile_nll, bounds=beta_bounds, method="bounded",
                               options=dict(xatol=1e-3))
        beta_hat = float(res.x)
        theta2_hat, theta3_hat, _ = cls._fit_outer_given_beta(
            ah_outer, c, beta_hat, theta3_bounds)
        return beta_hat, theta2_hat, theta3_hat

    # ------------------------------------------------------------------
    # fit_reference
    # ------------------------------------------------------------------
    def fit_reference(self, x, bulk_quantile=None):
        if bulk_quantile is None:
            bulk_quantile = self.bulk_quantile

        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real.detach().cpu().numpy()

        alphas, scales_b = [], []
        betas, theta2s, theta3s = [], [], []
        cc, ss = [], []
        pi_bulks, pi_outers = [], []

        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            ah = np.abs(h)

            c_ = float(np.quantile(ah, bulk_quantile))
            c_ = max(c_, 1e-8)
            s_ = max(self.trans_frac * c_, 1e-6)

            bulk_h = h[ah < c_]
            outer_h = h[ah >= c_]
            n_b, n_o = bulk_h.size, outer_h.size

            pi_bulk_j = n_b / max(n_b + n_o, 1)
            pi_outer_j = n_o / max(n_b + n_o, 1)
            
            print(f"[GGD/GenGamma][ch {j}] c={c_:.4f} s={s_:.4f} | "
                  f"N_bulk={n_b} N_outer={n_o}")
            if min(n_b, n_o) < self.min_region_samples:
                print(f"[GGD/GenGamma][ch {j}] WARNING: a region has "
                      f"< {self.min_region_samples} samples; adjust bulk_quantile.")

            alpha_j, scale_j = self._fit_bulk_ggd(bulk_h, c_, self.alpha_bounds)
            beta_j, theta2_j, theta3_j = self._fit_outer_gengamma(
                outer_h, c_, self.beta_bounds, self.theta3_bounds)

            print(f"[GGD/GenGamma][ch {j}] bulk(alpha={alpha_j:.3f},"
                  f"scale={scale_j:.4f}) | outer(beta={beta_j:.3f},"
                  f"theta2={theta2_j:.4f},theta3={theta3_j:.3f})")

            alphas += [alpha_j]; scales_b += [scale_j]
            betas += [beta_j]; theta2s += [theta2_j]; theta3s += [theta3_j]
            cc += [c_]; ss += [s_]
            pi_bulks += [pi_bulk_j]; pi_outers += [pi_outer_j]
            

        dtype = x.dtype if x.is_floating_point() else torch.float32
        dev = x.device
        mk = lambda L: torch.tensor(L, dtype=dtype, device=dev)
        self.alpha, self.scale_bulk = mk(alphas), mk(scales_b)
        self.beta, self.theta2, self.theta3 = mk(betas), mk(theta2s), mk(theta3s)
        self.c, self.s = mk(cc), mk(ss)
        self.pi_bulk = mk(pi_bulks)
        self.pi_outer = mk(pi_outers)

    def fit(self, x, **kwargs):
        return self.fit_reference(x, **kwargs)

    # ------------------------------------------------------------------
    # Differentiable forward/grad (smooth window g(x), for MGD machinery)
    # ------------------------------------------------------------------
    def _g(self, z, device):
        c = self.c.to(device)[None, :, None]
        s = self.s.to(device)[None, :, None]
        az = torch.sqrt(z ** 2 + self.eps_abs)
        return torch.sigmoid(-(az - c) / s)   # ~1 bulk, ~0 outer

    def forward(self, x, *args):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real
        device = x.device
        g = self._g(z, device)
        az = torch.sqrt(z ** 2 + self.eps_abs)

        alpha = self.alpha.to(device)[None, :, None]
        beta = self.beta.to(device)[None, :, None]

        phi_bulk = (g * az ** alpha).mean(-1)
        phi_outer_beta = ((1 - g) * az ** beta).mean(-1)
        phi_outer_log = ((1 - g) * torch.log(az)).mean(-1)
        return torch.cat([phi_bulk, phi_outer_beta, phi_outer_log], dim=1)  # (B,3J)

    def grad(self, x, v=None, means=None):
        self._check_fitted()
        filters = self.filters.to(x.device)
        device = x.device
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real
        z = z.detach().requires_grad_(True)

        g = self._g(z, device)
        az = torch.sqrt(z ** 2 + self.eps_abs)
        alpha = self.alpha.to(device)[None, :, None]
        beta = self.beta.to(device)[None, :, None]

        phi_bulk = g * az ** alpha
        phi_outer_beta = (1 - g) * az ** beta
        phi_outer_log = (1 - g) * torch.log(az)

        outs = []
        for phi in (phi_bulk, phi_outer_beta, phi_outer_log):
            gr = torch.autograd.grad(phi.sum(), z, create_graph=True, retain_graph=True)[0]
            outs.append(gr)
        D_all = torch.cat(outs, dim=1)

        def backproj(D):
            return torch.fft.ifft(torch.fft.fft(D) * filters.repeat(1, 3, 1)).real / x.shape[-1]
        grad_coeff = backproj(D_all)

        if v is None:
            return grad_coeff
        return (grad_coeff * v[None, :, None]).sum(1)[:, None]

    def plot_fit(self, x, label="Wavelet", n_grid=1000, fit_if_needed=True):
        if fit_if_needed and not self.is_fitted:
            self.fit_reference(x)
        self._check_fitted()

        filters = self.filters.to(x.device)
        wt = torch.fft.ifft(filters * torch.fft.fft(x)).real
        n_wavelets = filters.shape[1]

        alpha  = self.alpha.cpu().numpy()
        scale_b = self.scale_bulk.cpu().numpy()
        beta   = self.beta.cpu().numpy()
        theta2 = self.theta2.cpu().numpy()
        theta3 = self.theta3.cpu().numpy()
        c_all  = self.c.cpu().numpy()

        def safe_exp(lp):
            return np.exp(np.clip(lp, -500.0, 500.0))

        for j in range(n_wavelets):
            h = wt[:, j, :].detach().cpu().flatten().numpy()
            h = h[np.isfinite(h)]
            if h.size == 0:
                continue
            ah = np.abs(h)
            c_j         = float(c_all[j])
            a_j, sb_j   = alpha[j], scale_b[j]
            b_j, t2_j, t3_j = beta[j], theta2[j], theta3[j]

            # FIX: actual data maximum, not a quantile
            xmax = float(ah.max()) * 1.02

            pi_outer_j  = float((ah >= c_j).mean())
            pi_bulk_j   = 1.0 - pi_outer_j
            log_pi_bulk  = np.log(max(pi_bulk_j,  1e-300))
            log_pi_outer = np.log(max(pi_outer_j, 1e-300))

            x_bulk = np.linspace(-c_j, c_j, n_grid)
            log_mass_bulk = Scalar_GGD_GenGamma._ggd_logmass_leq(c_j, a_j, sb_j)
            logpdf_bulk = (np.log(a_j) - np.log(2.0) - np.log(sb_j)
                        - gammaln(1.0 / a_j)
                        - (np.abs(x_bulk) / sb_j) ** a_j
                        - log_mass_bulk + log_pi_bulk)

            # outer grid now runs all the way to ah.max()
            x_outer_pos = np.linspace(c_j, xmax, n_grid)
            logZ_outer  = Scalar_GGD_GenGamma._outer_logZ(t2_j, t3_j, b_j, c_j)
            logpdf_outer_pos = (-t3_j * np.log(x_outer_pos)
                                - t2_j * x_outer_pos ** b_j
                                - logZ_outer + log_pi_outer)
            x_outer  = np.concatenate([-x_outer_pos[::-1], x_outer_pos])
            logpdf_outer = np.concatenate([logpdf_outer_pos[::-1], logpdf_outer_pos])

            # data-driven y limits so clipped curves never rescale the axis
            hist_vals, _ = np.histogram(h, bins=150, density=True)
            hist_pos = hist_vals[hist_vals > 0]
            y_min = hist_pos.min() * 0.3
            y_max = hist_pos.max() * 5.0

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(h, bins=150, density=True, log=True,
                    alpha=0.4, color="steelblue", label="data")
            ax.plot(x_bulk,  safe_exp(logpdf_bulk),  lw=2, color="tab:orange",
                    label=f"bulk (GGD)     α={a_j:.2f} sc={sb_j:.3f} π={pi_bulk_j:.3f}")
            ax.plot(x_outer, safe_exp(logpdf_outer), lw=2, color="tab:red",
                    label=f"outer (genΓ)  β={b_j:.2f} θ2={t2_j:.3f} θ3={t3_j:.2f} π={pi_outer_j:.3f}")
            ax.axvline( c_j, color="black", ls="--", lw=1, alpha=0.4)
            ax.axvline(-c_j, color="black", ls="--", lw=1, alpha=0.4)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("Coefficient value")
            ax.set_ylabel("Log density")
            ax.set_title(f"{label} — channel {j}  (c={c_j:.3f})")
            ax.legend(frameon=False, fontsize=8)
            plt.tight_layout()
            plt.show()


import numpy as np
import torch
from scipy.special import gammaln, gammainc
from scipy.optimize import minimize, minimize_scalar
from scipy.integrate import quad
from scipy.stats import norm

class Scalar_GGD_GGD_Pow:
    """
    Three-region potential per wavelet channel.

    Mathematical model
    ------------------
    Two smooth sigmoid windows g1, g2 define a partition of unity on |x|:

        g1(x) = sigmoid(-(|x| - c1) / s1)   ~1 for |x|<<c1, ~0 for |x|>>c1
        g2(x) = sigmoid(-(|x| - c2) / s2)   ~1 for |x|<<c2, ~0 for |x|>>c2

        w_bulk(x) = g1(x)                     active near zero
        w_mid(x)  = g2(x) - g1(x)            active in shoulder region
        w_tail(x) = 1 - g2(x)                active in far tail

    Three sufficient statistics (per channel):
        phi_bulk(x) = w_bulk(x) * |x|^alpha1   -> maxent: GGD exp(-(|x|/s)^alpha1)
        phi_mid(x)  = w_mid(x)  * |x|^alpha2   -> maxent: GGD exp(-(|x|/s)^alpha2)
        phi_tail(x) = w_tail(x) * log|x|       -> maxent: power law |x|^{-beta}

    Boundaries
    ----------
        c1 = E[|x|]  (empirical mean absolute value)
             Natural bulk/mid split: sits at the "body" of the distribution
             regardless of tail heaviness, unlike sigma which is inflated
             by outliers. For a Gaussian: E[|x|] = sigma*sqrt(2/pi).
             For heavy-tailed: E[|x|] << sigma.

        c2 = high quantile of |x| (default: 99th percentile)
             Where the power-law tail clearly dominates, verified by the
             Hill estimator producing a stable estimate.

    Fitting: hard-cutoff disjoint subsets, truncation-corrected MLE
    ---------------------------------------------------------------
    Each region is fit on its own hard subset with a truncation-corrected
    likelihood — the normalizer integrates only over the region's own
    support, not the full line. This avoids the truncation bias
    (scale/sigma underestimation) that caused overshoot in earlier versions.

        Bulk and mid: 2D Nelder-Mead over (alpha, scale) with truncated
            GGD normalizer via regularized incomplete gamma.
        Tail: Hill estimator — exact closed-form MLE for a Pareto tail
            index given threshold c2. Pure power law p(x) ~ x^{-beta},
            NO exponential component, preserving fat tails in samples.

    forward() -> (B, 3J): [phi_bulk_mean, phi_mid_mean, phi_tail_mean]
    grad()    -> (B, 3J, T) or (B, 1, T) if v given — fully analytic,
                 no autograd graph retained, O(BJT) memory.
    """

    def __init__(self, filters,
                 tail_quantile=0.99,
                 trans_frac=0.10,
                 alpha_bounds=(0.2, 8.0),
                 beta_bounds=(1.1, 30.0),
                 min_region_samples=30,
                 eps_abs=1e-6):
        self.filters = filters
        self.num_coefficients = 3 * filters.shape[1]
        self.tail_quantile = tail_quantile
        self.trans_frac = trans_frac
        self.alpha_bounds = alpha_bounds
        self.beta_bounds = beta_bounds
        self.min_region_samples = min_region_samples
        self.eps_abs = eps_abs

        # fitted parameters (J,) each
        self.alpha1 = self.scale1 = None   # bulk GGD
        self.alpha2 = self.scale2 = None   # mid  GGD
        self.beta   = None                 # tail power law
        self.c1 = self.c2 = None           # boundaries
        self.s1 = self.s2 = None           # transition widths
        self.pi_bulk = self.pi_mid = self.pi_tail = None  # mixture weights

        self._filters_3x = None            # cached repeated filter bank

    @property
    def is_fitted(self):
        return self.alpha1 is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("must call fit_reference first.")

    # ------------------------------------------------------------------
    # Truncation-corrected GGD MLE
    # log P(a <= |Z| < b) for GGD uses regularized incomplete gamma:
    #   P(|Z| <= t) = gammainc(1/alpha, (t/scale)^alpha)
    # The truncated NLL adds n * log[P(a <= |Z| < b)] as a correction,
    # which is what was missing in the non-truncated version and caused
    # the scale to be underestimated (curves too narrow / too tall).
    # ------------------------------------------------------------------
    @staticmethod
    def _ggd_cdf_abs(t, alpha, scale):
        """P(|Z| <= t) for a GGD with given alpha, scale."""
        return float(gammainc(1.0 / alpha, (max(t, 0.0) / scale) ** alpha))

    @classmethod
    def _fit_ggd_truncated(cls, h, lo, hi, alpha_bounds):
        """
        Truncation-corrected GGD MLE on the subset lo <= |x| < hi.
        Normalizer: P(lo <= |Z| < hi) = CDF(hi) - CDF(lo).
        Both alpha and scale are optimized (2D Nelder-Mead), well-
        conditioned because we have a good closed-form starting point.
        """
        h = np.asarray(h, dtype=float)
        ah = np.abs(h)
        n = ah.size
        if n < 5:
            return 1.0, float(np.std(h) + 1e-8)

        def neg_ll(theta):
            alpha = float(theta[0])
            scale = float(theta[1])
            logpdf = (np.log(alpha) - np.log(2.0) - np.log(scale)
                      - gammaln(1.0 / alpha)
                      - (ah / scale) ** alpha)
            cdf_hi = cls._ggd_cdf_abs(hi, alpha, scale)
            cdf_lo = cls._ggd_cdf_abs(lo, alpha, scale) if lo > 0 else 0.0
            mass = cdf_hi - cdf_lo
            if mass <= 1e-12:
                return np.inf
                
            return -np.sum(logpdf) + n * np.log(mass)

        # closed-form untruncated init as starting point
        alpha0 = 1.5
        scale0 = max((alpha0 * np.mean(ah ** alpha0)) ** (1.0 / alpha0), 1e-8)
        res = minimize(
            neg_ll,
            x0=[alpha0, scale0],
            method="L-BFGS-B",
            bounds=[
                alpha_bounds,
                (1e-8, None)
            ]
        )
        print(res.success)
        print(res.message)
        print(res.x)
        print(res.fun)

        alpha_hat = float(np.clip(res.x[0], *alpha_bounds))
        scale_hat = float(max(res.x[1], 1e-8))
        return alpha_hat, scale_hat

    # ------------------------------------------------------------------
    # Hill estimator: pure power-law tail, no exponential cutoff.
    # This is the exact closed-form MLE for p(x) ~ x^{-beta} on [c2, inf).
    # beta = 1 + 1/mean(log(x/c2)) for x > c2.
    # ------------------------------------------------------------------
    @staticmethod
    def _fit_hill_beta(h, c2, beta_bounds):
        ah = np.abs(np.asarray(h, dtype=float))
        excess = ah[ah > c2]
        if excess.size < 5:
            return float(beta_bounds[0] + 1.0)
        xi = float(np.mean(np.log(excess / c2)))
        if xi <= 1e-8:
            return float(beta_bounds[1])
        return float(np.clip(1.0 + 1.0 / xi, *beta_bounds))

    # ------------------------------------------------------------------
    # fit_reference
    # ------------------------------------------------------------------
    def fit_reference(self, x, tail_quantile=None):
        if tail_quantile is None:
            tail_quantile = self.tail_quantile

        self._filters_3x = None  # invalidate filter cache
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real.detach().cpu().numpy()

        a1s, s1s, a2s, s2s, betas = [], [], [], [], []
        c1s, c2s, sw1s, sw2s = [], [], [], []
        pi_bs, pi_ms, pi_ts = [], [], []

        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            ah = np.abs(h)

            # --- c1: E[|x|], the professor's natural bulk/mid boundary ---
            c1_ = float(np.mean(ah))
            c1_ = max(c1_, 1e-8)

            # --- c2: high quantile where power law clearly dominates ---
            c2_ = float(np.quantile(ah, tail_quantile))
            c2_ = max(c2_, c1_ * 1.5)  # enforce c2 > c1

            # transition widths: fraction of each region's width
            sw1_ = max(self.trans_frac * c1_, 1e-6)
            sw2_ = max(self.trans_frac * (c2_ - c1_), 1e-6)

            bulk_h = h[ah < c1_]
            mid_h  = h[(ah >= c1_) & (ah < c2_)]
            tail_h = h[ah >= c2_]
            n_b, n_m, n_t = bulk_h.size, mid_h.size, tail_h.size

            pi_b = n_b / max(n_b + n_m + n_t, 1)
            pi_m = n_m / max(n_b + n_m + n_t, 1)
            pi_t = n_t / max(n_b + n_m + n_t, 1)

            print(f"[GGD/GGD/Pow][ch {j}]  "
                  f"c1={c1_:.4f}(E[|x|])  c2={c2_:.4f}({tail_quantile*100:.0f}pct) | "
                  f"N_bulk={n_b}({pi_b:.2%})  N_mid={n_m}({pi_m:.2%})  "
                  f"N_tail={n_t}({pi_t:.2%})")

            if min(n_b, n_m, n_t) < self.min_region_samples:
                print(f"[GGD/GGD/Pow][ch {j}] WARNING: a region has "
                      f"< {self.min_region_samples} samples.")

            a1_, s1_ = self._fit_ggd_truncated(bulk_h, 0.0, c1_, self.alpha_bounds)
            a2_, s2_ = self._fit_ggd_truncated(mid_h,  c1_, c2_, self.alpha_bounds)
            b_       = self._fit_hill_beta(h, c2_, self.beta_bounds)

            print(f"[GGD/GGD/Pow][ch {j}]  "
                  f"bulk(alpha={a1_:.3f}, scale={s1_:.4f}) | "
                  f"mid(alpha={a2_:.3f}, scale={s2_:.4f}) | "
                  f"tail(beta={b_:.3f})")

            a1s += [a1_];  s1s += [s1_]
            a2s += [a2_];  s2s += [s2_]
            betas += [b_]
            c1s += [c1_];  c2s += [c2_]
            sw1s += [sw1_]; sw2s += [sw2_]
            pi_bs += [pi_b]; pi_ms += [pi_m]; pi_ts += [pi_t]

        dtype = x.dtype if x.is_floating_point() else torch.float32
        dev   = x.device
        mk    = lambda L: torch.tensor(L, dtype=dtype, device=dev)

        self.alpha1, self.scale1 = mk(a1s), mk(s1s)
        self.alpha2, self.scale2 = mk(a2s), mk(s2s)
        self.beta = mk(betas)
        self.c1, self.c2 = mk(c1s), mk(c2s)
        self.s1, self.s2 = mk(sw1s), mk(sw2s)
        self.pi_bulk, self.pi_mid, self.pi_tail = mk(pi_bs), mk(pi_ms), mk(pi_ts)

    def fit(self, x, **kwargs):
        return self.fit_reference(x, **kwargs)

    # ------------------------------------------------------------------
    # Smooth windows — partition of unity on |x| into three regions.
    # g1 ~ 1 near zero (bulk), g2 ~ 1 below c2 (bulk+mid).
    # w_bulk = g1,  w_mid = g2-g1,  w_tail = 1-g2.
    # ------------------------------------------------------------------
    def _windows(self, z, device):
        c1 = self.c1.to(device)[None, :, None]
        c2 = self.c2.to(device)[None, :, None]
        s1 = self.s1.to(device)[None, :, None]
        s2 = self.s2.to(device)[None, :, None]
        az = torch.sqrt(z ** 2 + self.eps_abs)
        g1 = torch.sigmoid(-(az - c1) / s1)
        g2 = torch.sigmoid(-(az - c2) / s2)
        return g1, g2 - g1, 1.0 - g2   # w_bulk, w_mid, w_tail

    # ------------------------------------------------------------------
    # forward — O(BJT), no graph, safe inside torch.no_grad()
    # ------------------------------------------------------------------
    def forward(self, x, *args):
        self._check_fitted()
        filters = self.filters.to(x.device)
        with torch.no_grad():
            z  = torch.fft.ifft(filters * torch.fft.fft(x)).real
            az = torch.sqrt(z ** 2 + self.eps_abs)
            w_b, w_m, w_t = self._windows(z, x.device)
            alpha1 = self.alpha1.to(x.device)[None, :, None]
            alpha2 = self.alpha2.to(x.device)[None, :, None]
            phi_bulk = (w_b * az ** alpha1).mean(-1)
            phi_mid  = (w_m * az ** alpha2).mean(-1)
            phi_tail = (w_t * torch.log(az)).mean(-1)
        return torch.cat([phi_bulk, phi_mid, phi_tail], dim=1)   # (B, 3J)

    # ------------------------------------------------------------------
    # grad — fully analytic, no autograd graph retained.
    #
    # Product rule for each windowed potential phi_k = w_k(z) * f_k(z):
    #   d phi_k / dz = (dw_k/dz) * f_k(z) + w_k(z) * (df_k/dz)
    #
    # Window derivatives (chain rule through sigmoid and az=sqrt(z²+eps)):
    #   daz/dz = z/az                              (smooth |z|')
    #   dg1/dz = -g1*(1-g1) * (z/az) / s1
    #   dg2/dz = -g2*(1-g2) * (z/az) / s2
    #   dw_bulk/dz =  dg1/dz
    #   dw_mid/dz  =  dg2/dz - dg1/dz
    #   dw_tail/dz = -dg2/dz
    #
    # Potential derivatives:
    #   d/dz [az^alpha]  = alpha * z * az^(alpha-2)
    #   d/dz [log(az)]   = z / az^2
    # ------------------------------------------------------------------
    def _get_filters_3x(self, device):
        if (self._filters_3x is None
                or self._filters_3x.device != device):
            self._filters_3x = self.filters.repeat(1, 3, 1).to(device)
        return self._filters_3x

    def grad(self, x, v=None, means=None):
        self._check_fitted()
        device  = x.device
        filters = self.filters.to(device)

        with torch.no_grad():
            z  = torch.fft.ifft(filters * torch.fft.fft(x)).real
            az = torch.sqrt(z ** 2 + self.eps_abs)
            sign_z = z / az                       # smooth sign(z), in (-1,1)

            c1 = self.c1.to(device)[None, :, None]
            c2 = self.c2.to(device)[None, :, None]
            s1 = self.s1.to(device)[None, :, None]
            s2 = self.s2.to(device)[None, :, None]
            alpha1 = self.alpha1.to(device)[None, :, None]
            alpha2 = self.alpha2.to(device)[None, :, None]

            g1 = torch.sigmoid(-(az - c1) / s1)
            g2 = torch.sigmoid(-(az - c2) / s2)
            w_b =  g1
            w_m  =  g2 - g1
            w_t  =  1.0 - g2

            dg1 = -g1 * (1.0 - g1) * sign_z / s1
            dg2 = -g2 * (1.0 - g2) * sign_z / s2
            dw_b =  dg1
            dw_m  =  dg2 - dg1
            dw_t  = -dg2

            # d/dz [az^alpha] = alpha * z * az^(alpha-2)
            daz_a1 = alpha1 * z * az ** (alpha1 - 2.0)
            daz_a2 = alpha2 * z * az ** (alpha2 - 2.0)
            # d/dz [log az]  = z / az^2
            dlog_az = z / az ** 2

            D_bulk = dw_b  * az ** alpha1 + w_b  * daz_a1
            D_mid  = dw_m  * az ** alpha2 + w_m  * daz_a2
            D_tail = dw_t  * torch.log(az) + w_t  * dlog_az

            D_all = torch.cat([D_bulk, D_mid, D_tail], dim=1)   # (B, 3J, T)

            f3 = self._get_filters_3x(device)
            grad_coeff = torch.fft.ifft(
                torch.fft.fft(D_all) * f3
            ).real / x.shape[-1]                                 # (B, 3J, T)

        if v is None:
            return grad_coeff
        return (grad_coeff * v[None, :, None]).sum(1, keepdim=True)  # (B,1,T)

    def plot_fit(self, x, label="Wavelet", n_grid=1000, fit_if_needed=True):
        """
        Plots the histogram of wavelet coefficients against the fitted 
        GGD (bulk), GGD (mid), and Power Law (tail) distributions.
        """
        import matplotlib.pyplot as plt
        from scipy.special import gammaln

        if fit_if_needed and not self.is_fitted:
            self.fit_reference(x)
        self._check_fitted()

        filters = self.filters.to(x.device)
        # Extract coefficients exactly as the model does
        wt = torch.fft.ifft(filters * torch.fft.fft(x)).real
        n_wavelets = filters.shape[1]

        # Extract fitted parameters using 'self' instead of 'model'
        a1  = self.alpha1.cpu().numpy()
        s1  = self.scale1.cpu().numpy()
        a2  = self.alpha2.cpu().numpy()
        s2  = self.scale2.cpu().numpy()
        b   = self.beta.cpu().numpy()
        c1a = self.c1.cpu().numpy()
        c2a = self.c2.cpu().numpy()
        pb  = self.pi_bulk.cpu().numpy()
        pm  = self.pi_mid.cpu().numpy()
        pt  = self.pi_tail.cpu().numpy()

        def safe_exp(lp):
            """Clip before exp so neither overflow nor exact-zero underflow reaches matplotlib."""
            return np.exp(np.clip(lp, -500.0, 500.0))

        for j in range(n_wavelets):
            h = wt[:, j, :].detach().cpu().flatten().numpy()
            h = h[np.isfinite(h)]
            if h.size == 0:
                continue
            
            ah = np.abs(h)
            c1_j, c2_j = float(c1a[j]), float(c2a[j])

            # Use actual data maximum, not a quantile, to ensure the tail curve reaches the last bin
            xmax = float(ah.max()) * 1.02

            # --- Bulk: truncated GGD on [-c1, c1], scaled by pi_bulk ---
            x_bulk = np.linspace(-c1_j, c1_j, n_grid)
            cdf_hi = self._ggd_cdf_abs(c1_j, a1[j], s1[j])
            mass_b = max(cdf_hi, 1e-300)
            
            logp_b = (np.log(a1[j]) - np.log(2.0) - np.log(s1[j])
                      - gammaln(1.0 / a1[j])
                      - (np.abs(x_bulk) / s1[j])**a1[j]
                      - np.log(mass_b) + np.log(max(pb[j], 1e-300)))

            # --- Mid: truncated GGD on [c1, c2], scaled by pi_mid ---
            x_mid_pos = np.linspace(c1_j, c2_j, n_grid // 2)
            cdf_hi2   = self._ggd_cdf_abs(c2_j, a2[j], s2[j])
            cdf_lo2   = self._ggd_cdf_abs(c1_j, a2[j], s2[j])
            mass_m    = max(cdf_hi2 - cdf_lo2, 1e-300)
            
            logp_m_pos = (np.log(a2[j]) - np.log(2.0) - np.log(s2[j])
                          - gammaln(1.0 / a2[j])
                          - (x_mid_pos / s2[j])**a2[j]
                          - np.log(mass_m) + np.log(max(pm[j], 1e-300)))
            
            x_mid  = np.concatenate([-x_mid_pos[::-1], x_mid_pos])
            logp_m = np.concatenate([logp_m_pos[::-1], logp_m_pos])

            # --- Tail: power law on [c2, xmax], scaled by pi_tail ---
            x_tail_pos = np.linspace(c2_j, xmax, n_grid // 2)
            logp_t_pos = (np.log(b[j] - 1.0) - np.log(2.0 * c2_j)
                          - b[j] * np.log(x_tail_pos / c2_j)
                          + np.log(max(pt[j], 1e-300)))
            
            x_tail  = np.concatenate([-x_tail_pos[::-1], x_tail_pos])
            logp_t  = np.concatenate([logp_t_pos[::-1], logp_t_pos])

            hist_vals, _ = np.histogram(h, bins=150, density=True)
            hist_pos = hist_vals[hist_vals > 0]
            if hist_pos.size == 0:
                continue
                
            y_min = hist_pos.min() * 0.3
            y_max = hist_pos.max() * 5.0

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(h, bins=150, density=True, log=True,
                    alpha=0.4, color="steelblue", label="data")
            
            ax.plot(x_bulk, safe_exp(logp_b), lw=2, color="tab:orange",
                    label=f"bulk GGD  α={a1[j]:.2f} sc={s1[j]:.3f} π={pb[j]:.2%}")
            ax.plot(x_mid,  safe_exp(logp_m), lw=2, color="tab:green",
                    label=f"mid  GGD  α={a2[j]:.2f} sc={s2[j]:.3f} π={pm[j]:.2%}")
            ax.plot(x_tail, safe_exp(logp_t), lw=2, color="tab:red",
                    label=f"tail PL   β={b[j]:.2f} π={pt[j]:.2%}")
            
            ax.axvline( c1_j, color="black", ls="--", lw=1, alpha=0.4,
                        label=f"c1=E[|x|]={c1_j:.3f}")
            ax.axvline(-c1_j, color="black", ls="--", lw=1, alpha=0.4)
            ax.axvline( c2_j, color="black", ls=":",  lw=1, alpha=0.4,
                        label=f"c2(q99)={c2_j:.3f}")
            ax.axvline(-c2_j, color="black", ls=":",  lw=1, alpha=0.4)
            
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("Coefficient value")
            ax.set_ylabel("Log density")
            ax.set_title(f"{label} — channel {j}")
            ax.legend(frameon=False, fontsize=7.5)
            
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------
    def summary(self):
        self._check_fitted()
        a1 = self.alpha1.cpu().numpy(); s1 = self.scale1.cpu().numpy()
        a2 = self.alpha2.cpu().numpy(); s2 = self.scale2.cpu().numpy()
        b  = self.beta.cpu().numpy()
        c1 = self.c1.cpu().numpy();     c2 = self.c2.cpu().numpy()
        pb = self.pi_bulk.cpu().numpy(); pm = self.pi_mid.cpu().numpy()
        pt = self.pi_tail.cpu().numpy()
        print(f"{'Ch':>3} {'c1(E|x|)':>10} {'c2(q99)':>9} | "
              f"{'a1':>5} {'sc1':>7} | {'a2':>5} {'sc2':>7} | "
              f"{'beta':>5} | {'pi_b':>5} {'pi_m':>5} {'pi_t':>5}")
        print("-" * 88)
        for j in range(len(a1)):
            print(f"{j:>3d} {c1[j]:>10.4f} {c2[j]:>9.4f} | "
                  f"{a1[j]:>5.2f} {s1[j]:>7.4f} | "
                  f"{a2[j]:>5.2f} {s2[j]:>7.4f} | "
                  f"{b[j]:>5.2f} | "
                  f"{pb[j]:>5.2%} {pm[j]:>5.2%} {pt[j]:>5.2%}")




import numpy as np
import torch
from scipy.special import gammaln, gammainc
from scipy.optimize import minimize


class Scalar_GGD_GGD_GGD:
    """
    Three-region potential per wavelet channel: bulk / mid / tail,
    each modelled as a truncated Generalized Gaussian Distribution (GGD).

    Mathematical model
    ------------------
    p_k(x) ~ exp(-(|x|/scale_k)^{alpha_k})   on its truncated support

    Two boundaries split the real line into three regions:
        bulk : |x| < c1        fitted with GGD_1(alpha1, scale1)
        mid  : c1 <= |x| < c2  fitted with GGD_2(alpha2, scale2)
        tail : |x| >= c2       fitted with GGD_3(alpha3, scale3)

    Why GGD for the tail (not power law)?
    --------------------------------------
    Fine-scale wavelet channels (high kurtosis) have tails that decay
    FASTER than any power law — they are Laplace-like or stretched-
    exponential, not Pareto. Fitting a power law there gives beta that
    describes the shoulder curvature, not the true asymptotic tail, and
    produces samples with far too many extreme values. GGD with alpha<=1
    captures sub-Laplace heavy tails; with alpha>=1 it captures lighter
    tails; the MLE picks the right exponent from the data.

    Boundaries
    ----------
        c1 = E[|x|]   (mean absolute value, professor's suggestion)
             Scale-adaptive: sits at the "body" regardless of kurtosis.
             For Gaussian: c1 = sigma * sqrt(2/pi).
             For heavy-tailed: c1 << sigma (unlike sigma which is
             inflated by rare large events).

        c2 = high quantile of |x| (default: 99th percentile)
             Separates the shoulder from the far tail.

    Fitting: log-space optimizer + truncation-corrected MLE
    -------------------------------------------------------
    All three regions use the same _fit_ggd_truncated routine:
        - Parameters (alpha, scale) optimized in LOG-SPACE, so the
          optimizer never explores negative scale values (was the cause
          of density=10^286 blowups in the previous version).
        - Truncation correction: NLL += n * log P(lo <= |Z| < hi),
          computed via regularized incomplete gamma. Without this,
          scale is underestimated because the truncated subsample
          looks artificially narrow.

    forward() -> (B, 3J): [E[w_b*|z|^a1], E[w_m*|z|^a2], E[w_t*|z|^a3]]
    grad()    -> fully analytic product-rule derivatives, no autograd.
    """

    def __init__(self, filters,
                 tail_quantile=0.97,
                 trans_frac=0.10,
                 alpha_bounds=(0.2, 8.0),
                 min_region_samples=30,
                 eps_abs=1e-6):
        self.filters = filters
        self.num_coefficients = 3 * filters.shape[1]
        self.tail_quantile = tail_quantile
        self.trans_frac = trans_frac
        self.alpha_bounds = alpha_bounds
        self.min_region_samples = min_region_samples
        self.eps_abs = eps_abs

        self.alpha1 = self.scale1 = None  # bulk  (J,)
        self.alpha2 = self.scale2 = None  # mid   (J,)
        self.alpha3 = self.scale3 = None  # tail  (J,)
        self.c1 = self.c2 = None
        self.s1 = self.s2 = None
        self.pi_bulk = self.pi_mid = self.pi_tail = None
        self._filters_3x = None

    @property
    def is_fitted(self):
        return self.alpha1 is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("must call fit_reference first.")

    # ------------------------------------------------------------------
    # Core fitting primitive: truncation-corrected GGD MLE in log-space
    # ------------------------------------------------------------------
    @staticmethod
    def _ggd_cdf_abs(t, alpha, scale):
        """P(|Z| <= t) for GGD via regularized lower incomplete gamma."""
        if t <= 0:
            return 0.0
        return float(gammainc(1.0 / alpha, (t / scale) ** alpha))

    @classmethod
    def _fit_ggd_truncated(cls, h, lo, hi, alpha_bounds):
        """
        Truncation-corrected GGD MLE on lo <= |x| < hi.

        Optimizes log(alpha) and log(scale) — guaranteed positive scale
        regardless of where the optimizer steps, fixing the -0.001 scale
        / density=10^286 blowup from L-BFGS-B on the unconstrained
        parameterization.

        NLL = -sum log p(x_i; alpha, scale)
              + n * log P(lo <= |Z| < hi; alpha, scale)

        The second term corrects for the fact that we only see samples
        in [lo, hi), not the full support.
        """
        h = np.asarray(h, dtype=float)
        ah = np.abs(h)
        n = ah.size
        if n < 5:
            return 1.0, float(np.std(h) + 1e-8)

        def neg_ll(log_theta):
            alpha = float(np.clip(np.exp(log_theta[0]), *alpha_bounds))
            scale = float(np.exp(log_theta[1]))
            logpdf = (np.log(alpha) - np.log(2.0) - np.log(scale)
                      - gammaln(1.0 / alpha)
                      - (ah / scale) ** alpha)
            cdf_hi = cls._ggd_cdf_abs(hi, alpha, scale)
            cdf_lo = cls._ggd_cdf_abs(lo, alpha, scale)
            mass = cdf_hi - cdf_lo
            if mass < 1e-300:
                return 1e12
            return -np.sum(logpdf) + n * np.log(mass)

        # Init: untruncated closed-form MLE as warm start
        alpha0 = 1.5
        # clamp ah to avoid 0^alpha0 issues
        safe_ah = np.maximum(ah, 1e-300)
        scale0 = max((alpha0 * np.mean(safe_ah ** alpha0)) ** (1.0 / alpha0), 1e-8)
        x0 = [np.log(alpha0), np.log(scale0)]

        res = minimize(neg_ll, x0=x0, method="Nelder-Mead",
                       options=dict(xatol=1e-5, fatol=1e-5, maxiter=8000))
        alpha_hat = float(np.clip(np.exp(res.x[0]), *alpha_bounds))
        scale_hat = float(np.exp(res.x[1]))
        return alpha_hat, scale_hat

    # ------------------------------------------------------------------
    # fit_reference
    # ------------------------------------------------------------------
    def fit_reference(self, x, tail_quantile=None):
        if tail_quantile is None:
            tail_quantile = self.tail_quantile

        self._filters_3x = None
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real.detach().cpu().numpy()

        a1s, s1s, a2s, s2s, a3s, s3s = [], [], [], [], [], []
        c1s, c2s, sw1s, sw2s = [], [], [], []
        pi_bs, pi_ms, pi_ts = [], [], []

        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            ah = np.abs(h)

            c1_ = float(np.mean(ah))          # E[|x|]: professor's boundary
            c1_ = max(c1_, 1e-8)
            c2_ = float(np.quantile(ah, tail_quantile))
            c2_ = max(c2_, c1_ * 1.5)

            sw1_ = max(self.trans_frac * c1_, 1e-6)
            sw2_ = max(self.trans_frac * (c2_ - c1_), 1e-6)

            bulk_h = h[ah < c1_]
            mid_h  = h[(ah >= c1_) & (ah < c2_)]
            tail_h = h[ah >= c2_]
            n_b, n_m, n_t = bulk_h.size, mid_h.size, tail_h.size
            total = max(n_b + n_m + n_t, 1)
            pi_b, pi_m, pi_t = n_b/total, n_m/total, n_t/total

            print(f"[GGD³][ch {j}]  "
                  f"c1={c1_:.4f}(E[|x|])  c2={c2_:.4f}({tail_quantile*100:.0f}pct) | "
                  f"N_bulk={n_b}({pi_b:.1%})  N_mid={n_m}({pi_m:.1%})  "
                  f"N_tail={n_t}({pi_t:.1%})")
            if min(n_b, n_m, n_t) < self.min_region_samples:
                print(f"[GGD³][ch {j}] WARNING: region has < "
                      f"{self.min_region_samples} samples.")

            # hi=inf for bulk: pass large number; CDF->1 is handled gracefully
            a1_, s1_ = self._fit_ggd_truncated(bulk_h, 0.0,  c1_,       self.alpha_bounds)
            a2_, s2_ = self._fit_ggd_truncated(mid_h,  c1_,  c2_,       self.alpha_bounds)
            a3_, s3_ = self._fit_ggd_truncated(tail_h, c2_,  np.inf,    self.alpha_bounds)

            print(f"[GGD³][ch {j}]  "
                  f"bulk(α={a1_:.3f}, sc={s1_:.4f}) | "
                  f"mid (α={a2_:.3f}, sc={s2_:.4f}) | "
                  f"tail(α={a3_:.3f}, sc={s3_:.4f})")

            a1s += [a1_]; s1s += [s1_]
            a2s += [a2_]; s2s += [s2_]
            a3s += [a3_]; s3s += [s3_]
            c1s += [c1_]; c2s += [c2_]
            sw1s += [sw1_]; sw2s += [sw2_]
            pi_bs += [pi_b]; pi_ms += [pi_m]; pi_ts += [pi_t]

        dtype = x.dtype if x.is_floating_point() else torch.float32
        dev = x.device
        mk = lambda L: torch.tensor(L, dtype=dtype, device=dev)
        self.alpha1, self.scale1 = mk(a1s), mk(s1s)
        self.alpha2, self.scale2 = mk(a2s), mk(s2s)
        self.alpha3, self.scale3 = mk(a3s), mk(s3s)
        self.c1, self.c2 = mk(c1s), mk(c2s)
        self.s1, self.s2 = mk(sw1s), mk(sw2s)
        self.pi_bulk = mk(pi_bs)
        self.pi_mid  = mk(pi_ms)
        self.pi_tail = mk(pi_ts)

    def fit(self, x, **kwargs):
        return self.fit_reference(x, **kwargs)

    # ------------------------------------------------------------------
    # Smooth windows
    # ------------------------------------------------------------------
    def _windows(self, z, device):
        c1 = self.c1.to(device)[None, :, None]
        c2 = self.c2.to(device)[None, :, None]
        s1 = self.s1.to(device)[None, :, None]
        s2 = self.s2.to(device)[None, :, None]
        az = torch.sqrt(z ** 2 + self.eps_abs)
        g1 = torch.sigmoid(-(az - c1) / s1)
        g2 = torch.sigmoid(-(az - c2) / s2)
        return g1, g2 - g1, 1.0 - g2

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x, *args):
        self._check_fitted()
        filters = self.filters.to(x.device)
        with torch.no_grad():
            z  = torch.fft.ifft(filters * torch.fft.fft(x)).real
            az = torch.sqrt(z ** 2 + self.eps_abs)
            w_b, w_m, w_t = self._windows(z, x.device)
            a1 = self.alpha1.to(x.device)[None, :, None]
            a2 = self.alpha2.to(x.device)[None, :, None]
            a3 = self.alpha3.to(x.device)[None, :, None]
            phi_b = (w_b * az ** a1).mean(-1)
            phi_m = (w_m * az ** a2).mean(-1)
            phi_t = (w_t * az ** a3).mean(-1)
        return torch.cat([phi_b, phi_m, phi_t], dim=1)

    # ------------------------------------------------------------------
    # grad — fully analytic
    #
    # d/dz [w_k * az^ak] = (dw_k/dz)*az^ak + w_k*ak*z*az^(ak-2)
    #
    # dg1/dz = -g1*(1-g1)*(z/az)/s1
    # dg2/dz = -g2*(1-g2)*(z/az)/s2
    # dw_b/dz =  dg1
    # dw_m/dz =  dg2 - dg1
    # dw_t/dz = -dg2
    # ------------------------------------------------------------------
    def _get_filters_3x(self, device):
        if self._filters_3x is None or self._filters_3x.device != device:
            self._filters_3x = self.filters.repeat(1, 3, 1).to(device)
        return self._filters_3x

    def grad(self, x, v=None, means=None):
        self._check_fitted()
        device  = x.device
        filters = self.filters.to(device)
        with torch.no_grad():
            z  = torch.fft.ifft(filters * torch.fft.fft(x)).real
            az = torch.sqrt(z ** 2 + self.eps_abs)
            sz = z / az   # smooth sign(z)

            c1 = self.c1.to(device)[None, :, None]
            c2 = self.c2.to(device)[None, :, None]
            s1 = self.s1.to(device)[None, :, None]
            s2 = self.s2.to(device)[None, :, None]
            a1 = self.alpha1.to(device)[None, :, None]
            a2 = self.alpha2.to(device)[None, :, None]
            a3 = self.alpha3.to(device)[None, :, None]

            g1 = torch.sigmoid(-(az - c1) / s1)
            g2 = torch.sigmoid(-(az - c2) / s2)
            w_b = g1;   w_m = g2 - g1;   w_t = 1.0 - g2

            dg1 = -g1 * (1.0 - g1) * sz / s1
            dg2 = -g2 * (1.0 - g2) * sz / s2
            dw_b =  dg1
            dw_m =  dg2 - dg1
            dw_t = -dg2

            D_b = dw_b * az**a1 + w_b * a1 * z * az**(a1 - 2.0)
            D_m = dw_m * az**a2 + w_m * a2 * z * az**(a2 - 2.0)
            D_t = dw_t * az**a3 + w_t * a3 * z * az**(a3 - 2.0)

            D_all = torch.cat([D_b, D_m, D_t], dim=1)
            f3 = self._get_filters_3x(device)
            grad_coeff = torch.fft.ifft(
                torch.fft.fft(D_all) * f3
            ).real / x.shape[-1]

        if v is None:
            return grad_coeff
        return (grad_coeff * v[None, :, None]).sum(1, keepdim=True)

    def plot_fit(self, x, label="Wavelet", n_grid=1000, fit_if_needed=True):
        import matplotlib.pyplot as plt
        from scipy.special import gammaln

        if fit_if_needed and not self.is_fitted:
            self.fit_reference(x)
        self._check_fitted()

        filters = self.filters.to(x.device)
        # Extract coefficients exactly as the model does
        wt = torch.fft.ifft(filters * torch.fft.fft(x)).real
        n_wavelets = filters.shape[1]

        filters = self.filters.to(x.device)
        wt = torch.fft.ifft(torch.fft.fft(x) * filters).real

        a1=self.alpha1.cpu().numpy(); s1=self.scale1.cpu().numpy()
        a2=self.alpha2.cpu().numpy(); s2=self.scale2.cpu().numpy()
        a3=self.alpha3.cpu().numpy(); s3=self.scale3.cpu().numpy()
        c1a=self.c1.cpu().numpy();   c2a=self.c2.cpu().numpy()
        pb=self.pi_bulk.cpu().numpy(); pm=self.pi_mid.cpu().numpy()
        pt=self.pi_tail.cpu().numpy()

        def logpdf_ggd_trunc(xv, alpha, scale, lo, hi, pi):
            """
            log[ pi * p_trunc(x) ] for a GGD truncated to [lo, hi).
            Clipped to [-500, 500] before return so exp() never overflows —
            the clip only affects the plotted curve's floor/ceiling, not the fit.
            """
            cdf_hi = Scalar_GGD_GGD_GGD._ggd_cdf_abs(hi, alpha, scale)
            cdf_lo = Scalar_GGD_GGD_GGD._ggd_cdf_abs(lo, alpha, scale)
            mass = max(cdf_hi - cdf_lo, 1e-300)
            lp = (np.log(alpha) - np.log(2.0) - np.log(scale)
                - gammaln(1.0 / alpha)
                - (np.abs(xv) / scale) ** alpha
                - np.log(mass)
                + np.log(max(pi, 1e-300)))
            return np.clip(lp, -500.0, 500.0)   # guard against float64 overflow

        for j in range(self.filters.shape[1]):
            h = wt[:, j, :].detach().cpu().flatten().numpy()
            h = h[np.isfinite(h)]
            if h.size == 0:
                continue
            ah = np.abs(h)
            c1_j, c2_j = float(c1a[j]), float(c2a[j])

            # Use actual data maximum so tail curve always reaches the last bin
            xmax = float(ah.max()) * 1.02

            x_b  = np.linspace(-c1_j, c1_j, n_grid)
            x_mp = np.linspace(c1_j,  c2_j, n_grid // 2)
            x_tp = np.linspace(c2_j,  xmax, n_grid // 2)

            lp_b  = logpdf_ggd_trunc(x_b,  a1[j], s1[j], 0.0,  c1_j,   pb[j])
            lp_mp = logpdf_ggd_trunc(x_mp, a2[j], s2[j], c1_j, c2_j,   pm[j])
            lp_tp = logpdf_ggd_trunc(x_tp, a3[j], s3[j], c2_j, np.inf, pt[j])

            x_m  = np.concatenate([-x_mp[::-1], x_mp])
            lp_m = np.concatenate([lp_mp[::-1], lp_mp])
            x_t  = np.concatenate([-x_tp[::-1], x_tp])
            lp_t = np.concatenate([lp_tp[::-1], lp_tp])

            # Empirical y-range: set axis limits from data, not from curves,
            # so an overflow-clipped curve can't force the y-axis to [1e-300, inf]
            hist_vals, _ = np.histogram(h, bins=150, density=True)
            hist_vals = hist_vals[hist_vals > 0]
            y_min = max(hist_vals.min() * 0.5, 1e-6)
            y_max = hist_vals.max() * 3.0

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(h, bins=150, density=True, log=True,
                    alpha=0.4, color="steelblue", label="data")
            ax.plot(x_b, np.exp(lp_b),  lw=2, color="tab:orange",
                    label=f"bulk  GGD α={a1[j]:.2f} sc={s1[j]:.3f} π={pb[j]:.1%}")
            ax.plot(x_m, np.exp(lp_m),  lw=2, color="tab:green",
                    label=f"mid   GGD α={a2[j]:.2f} sc={s2[j]:.3f} π={pm[j]:.1%}")
            ax.plot(x_t, np.exp(lp_t),  lw=2, color="tab:red",
                    label=f"tail  GGD α={a3[j]:.2f} sc={s3[j]:.3f} π={pt[j]:.1%}")
            ax.axvline( c1_j, color="black", ls="--", lw=1, alpha=0.4,
                        label=f"c1=E[|x|]={c1_j:.3f}")
            ax.axvline(-c1_j, color="black", ls="--", lw=1, alpha=0.4)
            ax.axvline( c2_j, color="black", ls=":",  lw=1, alpha=0.4,
                        label=f"c2(q{self.tail_quantile*100:.0f})={c2_j:.3f}")
            ax.axvline(-c2_j, color="black", ls=":",  lw=1, alpha=0.4)

            ax.set_ylim(y_min, y_max)   # data-driven limits, immune to curve overflow
            ax.set_xlabel("Coefficient value"); ax.set_ylabel("Log density")
            ax.set_title(f"{label} — channel {j}")
            ax.legend(frameon=False, fontsize=7.5)
            plt.tight_layout(); plt.show()


    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------
    def summary(self):
        self._check_fitted()
        for arr, name in [(self.alpha1,'a1'),(self.scale1,'sc1'),
                          (self.alpha2,'a2'),(self.scale2,'sc2'),
                          (self.alpha3,'a3'),(self.scale3,'sc3'),
                          (self.c1,'c1'),(self.c2,'c2'),
                          (self.pi_bulk,'pi_b'),(self.pi_mid,'pi_m'),
                          (self.pi_tail,'pi_t')]:
            pass  # just for reference
        a1=self.alpha1.cpu().numpy(); s1=self.scale1.cpu().numpy()
        a2=self.alpha2.cpu().numpy(); s2=self.scale2.cpu().numpy()
        a3=self.alpha3.cpu().numpy(); s3=self.scale3.cpu().numpy()
        c1=self.c1.cpu().numpy(); c2=self.c2.cpu().numpy()
        pb=self.pi_bulk.cpu().numpy(); pm=self.pi_mid.cpu().numpy()
        pt=self.pi_tail.cpu().numpy()
        print(f"{'Ch':>3} {'c1':>8} {'c2':>8} | "
              f"{'a1':>5} {'sc1':>7} | {'a2':>5} {'sc2':>7} | "
              f"{'a3':>5} {'sc3':>7} | {'pi_b':>6} {'pi_m':>6} {'pi_t':>6}")
        print("-"*95)
        for j in range(len(a1)):
            print(f"{j:>3d} {c1[j]:>8.4f} {c2[j]:>8.4f} | "
                  f"{a1[j]:>5.2f} {s1[j]:>7.4f} | "
                  f"{a2[j]:>5.2f} {s2[j]:>7.4f} | "
                  f"{a3[j]:>5.2f} {s3[j]:>7.4f} | "
                  f"{pb[j]:>6.2%} {pm[j]:>6.2%} {pt[j]:>6.2%}")






class Hermite_norm(Potential):
    def __init__(self, p,filters):
        super().__init__()
        self.p = p
        self.L2 = L2p_norm(1,filters)
        self.L4 = L2p_norm(2,filters)
        self.L6 = L2p_norm(3,filters)
        self.L8 = L2p_norm(4,filters)
        self.filters = filters
 
    def forward(self, x):
        if self.p == 1:
            x = self.L2(x)/2
        if self.p == 2:
            x = self.L4(x)/4-self.L2(x)*3/2
        if self.p == 3:
            x = self.L6(x)/6-self.L4(x)*10/4+self.L2(x)*15/2
        if self.p == 4:
            x = self.L8(x)/8- self.L6(x)*21/6+ self.L4(x)*105/4- self.L2(x)*105/2
            
        return x

    def grad(self, x, v=None, means=None):
        
        if self.p == 1:
            x = self.L2.grad(x, v,  means)/2
        if self.p == 2:
            x = self.L4.grad(x,v, means)/4-self.L2.grad(x, v, means)*3/2
        if self.p == 3:
            x = self.L6.grad(x, v, means)/6-self.L4.grad(x, v, means)*10/4+self.L2.grad(x,v, means)*15/2
        if self.p == 4:
            x = self.L8.grad(x, v, means)/8- self.L6.grad(x, v, means)*21/6+ self.L4.grad(x, v, means)*105/4- self.L2.grad(x, v, means)*105/2
        return x 
        
    def fit_micro(self,x):
        self.L2.fit_micro(x)
        self.L4.fit_micro(x)
        self.L6.fit_micro(x)
        self.L8.fit_micro(x)