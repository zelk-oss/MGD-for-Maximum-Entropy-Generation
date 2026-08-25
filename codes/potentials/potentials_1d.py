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
        filters = self.filters.to(x.device)
        filters_Q = self.filters_Q.to(x.device)

        x_filtered = torch.fft.ifft(filters_Q*torch.fft.fft(x)) #(B,JQ,T)

        #Normalize micro
        x_filtered = x_filtered/self.norm

        x_filtered_abs2 = abs_eps(x_filtered)**2 #(B,JQ,T)  <-- mod2

        W_Wx = torch.fft.ifft(filters[:, :, None] * torch.fft.fft(x_filtered_abs2)[:, None]) #(B,JQ,J+1,T)
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

            intermediate_output = torch.fft.ifft(filters_Q*torch.fft.fft(intermediate_output)).sum(1)[:,None]/x.shape[-1]
            intermediate_output_conj = torch.fft.ifft(filters_Q_tilde*torch.fft.fft(intermediate_output_conj)).sum(1)[:,None]/x.shape[-1]

            output = intermediate_output + intermediate_output_conj

            intermediate_output = 0.5*x_filtered_grad*torch.einsum('ijk, ajkb -> aib', m_transpose, x_filtered_abs2_no_LF_filtered_2_tilde)
            intermediate_output_conj = 0.5*x_filtered_grad.conj()*torch.einsum('ijk, ajkb -> aib', m_transpose, x_filtered_abs2_no_LF_filtered_2_tilde)

            intermediate_output = torch.fft.ifft(filters_Q*torch.fft.fft(intermediate_output)).sum(1)[:,None]/x.shape[-1]
            intermediate_output_conj = torch.fft.ifft(filters_Q_tilde*torch.fft.fft(intermediate_output_conj)).sum(1)[:,None]/x.shape[-1]

            output_transpose = intermediate_output + intermediate_output_conj

            output = output + output_transpose
            return output.imag

        intermediate_output = 0.5*x_filtered_grad[:,:,None,None]*x_filtered_abs2_no_LF_filtered_2[:,None]
        intermediate_output_conj = 0.5*x_filtered_grad[:,:,None,None].conj()*x_filtered_abs2_no_LF_filtered_2[:,None]

        intermediate_output = torch.fft.ifft(filters_Q[:,:,None,None]*torch.fft.fft(intermediate_output))
        intermediate_output_conj = torch.fft.ifft(filters_Q_tilde[:,:,None,None].conj()*torch.fft.fft(intermediate_output_conj))

        output = intermediate_output + intermediate_output_conj

        intermediate_output = 0.5*x_filtered_grad[:,:,None,None]*x_filtered_abs2_no_LF_filtered_2_tilde[:,None]
        intermediate_output_conj = 0.5*x_filtered_grad[:,:,None,None].conj()*x_filtered_abs2_no_LF_filtered_2_tilde[:,None]

        intermediate_output = torch.fft.ifft(filters_Q[:,:,None,None]*torch.fft.fft(intermediate_output))
        intermediate_output_conj = torch.fft.ifft(filters_Q_tilde[:,:,None,None].conj()*torch.fft.fft(intermediate_output_conj))

        output_transpose = intermediate_output + intermediate_output_conj

        output = output + torch.transpose(output_transpose, 1,2)

        output = output[:, indices[0], indices[1], indices[2]].reshape(x.shape[0], indices.shape[1], x.shape[-1])/x.shape[-1]

        return output.imag / self.norm_indices[:,None].to(x.device).to(x.dtype)

    def fit_micro(self,x):
        filters_Q = self.filters_Q.to(x.device)
        x_filtered = torch.fft.ifft(filters_Q*torch.fft.fft(x))
        x_filtered = x_filtered.abs()**2
        #Normalize_micro
        self.norm = x_filtered.mean((0,2))[:,None]**0.5   # matches Imag_1d convention (not the -1/4 used by the Real classes)

        indices = self.indices
        norm_indices = self.norm[:,None]*self.norm[None,:]
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



import numpy as np
import torch
from scipy.special import gammaln, gammainc
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Scalar_GGD_KRegion():
    def __init__(self, filters,
                 num_regions=4,
                 trans_frac=0.10,
                 alpha_bounds=(0.2, 8.0),
                 min_region_samples=30,
                 eps_abs=1e-6,
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
                 verbose=True):
        self.filters = filters
        self.K = int(num_regions)
        assert self.K >= 1
        self.J = filters.shape[1]
        self.trans_frac = trans_frac
        self.alpha_bounds = alpha_bounds
        self.min_region_samples = min_region_samples
        self.eps_abs = eps_abs
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
        self.num_coefficients = self.K * self.J
        self._filters_Kx = None

    def __call__(self, x, *args):
        return self.forward(x, *args)

    def to(self, device):
        self.filters = self.filters.to(device)
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
        return self

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
                           xatol=1e-5, fatol=1e-5):
        h = np.asarray(h, dtype=float); ah = np.abs(h); n = ah.size
        if n < 5:
            return 1.0, float(np.std(h) + 1e-8)
        scale_floor = cls._scale_floor(ah)

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
    def _composite_nll(cls, h, boundaries, alpha_bounds, fast=True, min_seg=5):
        ah = np.abs(h); N = h.size
        edges = [0.0] + list(boundaries) + [np.inf]
        total = 0.0; params = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            seg = h[(ah >= lo) & (ah < hi)]
            if seg.size < min_seg:
                return np.inf, None
            nll, (a, s) = cls._region_nll(seg, lo, hi, alpha_bounds, fast)
            total += nll - seg.size * np.log(seg.size / N)
            params.append((a, s))
        return total, params

    @classmethod
    def _fit_boundaries_k(cls, h, K, alpha_bounds, n_grid=12, n_iters=3,
                          subsample=20000, seed=0):
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
        q_los = np.linspace(0.05, 0.80, K - 1); q_his = np.linspace(0.80, 0.995, K - 1)

        def scored(cs):
            nll, _ = cls._composite_nll(h_s, sorted(cs), alpha_bounds, fast=True)
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

    def _select_model_order(self, h, seed=0, K_max=None):
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
                subsample=self.boundary_search_subsample, seed=seed)
            nll, params = self._composite_nll(h, cuts, self.alpha_bounds,
                                              fast=False, min_seg=self.min_region_samples)
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
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real.detach().cpu().numpy()

        J, K = z.shape[1], self.K
        A = np.ones((J, K)); S = np.ones((J, K))
        CUT = np.zeros((J, K - 1)); SW = np.full((J, K - 1), 1e-6)
        PI = np.zeros((J, K)); KEFF = np.ones(J, dtype=int)
        ACT = np.zeros((J, K), dtype=bool)

        for j in range(J):
            h = z[:, j, :].reshape(-1); h = h[np.isfinite(h)]; ah = np.abs(h); N = h.size

            if boundary_method == "auto":
                Keff, cuts = self._select_model_order(h, seed=0)
            elif boundary_method == "likelihood":
                Keff = K
                cuts = self._fit_boundaries_k(
                    h, K, self.alpha_bounds, n_grid=self.boundary_search_grid,
                    n_iters=self.boundary_search_iters,
                    subsample=self.boundary_search_subsample)
            else:
                qs = np.linspace(0.5, 0.97, K - 1); Keff = K
                cuts = list(np.quantile(ah, qs))

            KEFF[j] = Keff
            slots = np.maximum(self._embed_slots(cuts, ah), 1e-8)
            for i in range(1, K - 1):
                slots[i] = max(slots[i], slots[i - 1] * 1.5)
            CUT[j] = slots

            edges = np.concatenate([[0.0], slots])
            for m in range(K - 1):
                width = slots[m] - edges[m]
                SW[j, m] = max(self.trans_frac * max(width, slots[m]), 1e-6)

            full_edges = [0.0] + list(slots) + [np.inf]
            counts = []
            for k, (lo, hi) in enumerate(zip(full_edges[:-1], full_edges[1:])):
                seg = h[(ah >= lo) & (ah < hi)]; counts.append(seg.size)
                if seg.size < 5:
                    A[j, k], S[j, k] = 1.0, max(self._scale_floor(ah), 1e-8)
                    continue
                A[j, k], S[j, k] = self._fit_ggd_truncated(seg, lo, hi, self.alpha_bounds)
            counts = np.asarray(counts, float)
            PI[j] = counts / max(counts.sum(), 1)
            ACT[j] = (PI[j] >= self.pi_active_min) & (counts >= self.min_region_samples)
            if not ACT[j].any():
                ACT[j, int(np.argmax(counts))] = True

            if self.verbose:
                cut_str = " ".join(f"{c:.4f}" for c in slots)
                pi_str = " ".join(f"{p:.1%}" for p in PI[j])
                a_str = " ".join(f"{a:.2f}" for a in A[j])
                print(f"[GGD^{K}][ch {j}] Keff={Keff}  cuts=[{cut_str}]  "
                      f"pi=[{pi_str}]  alpha=[{a_str}]  active={ACT[j].sum()}")

        dtype = x.dtype if x.is_floating_point() else torch.float32
        dev = x.device
        mk = lambda M: torch.tensor(M, dtype=dtype, device=dev)
        self.alpha = mk(A); self.scale = mk(S); self.cuts = mk(CUT)
        self.sw = mk(SW); self.pi = mk(PI)
        self.Keff = torch.tensor(KEFF, device=dev)
        self.active = torch.tensor(ACT, device=dev)

        flat = [k * J + j for k in range(K) for j in range(J) if ACT[j, k]]
        self.active_flat = torch.tensor(sorted(flat), dtype=torch.long, device=dev)
        self.num_coefficients = int(self.active_flat.numel())

        n_dead = J * K - int(ACT.sum())
        if self.verbose:
            print(f"[active] {self.num_coefficients} active statistics "
                  f"({n_dead} dead slots dropped before they can singularize the Gram)")

        self.stat_scale = torch.ones(self.num_coefficients, dtype=dtype, device=dev)
        if self.auto_prune and self.num_coefficients > 1:
            self.prune_collinear(x, cond_tol=self.cond_tol,
                                 max_cols=self.prune_max_cols, verbose=self.verbose)
        self._compute_stat_scale(x)
        self.plot_fit(x)
        return self

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

    def _all_slot_forward(self, x):
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real
        az = torch.sqrt(z ** 2 + self.eps_abs)
        c = self.cuts.to(x.device); s = self.sw.to(x.device)
        g = torch.sigmoid(-(az.unsqueeze(-1) - c[None, :, None, :]) / s[None, :, None, :])
        ws = self._windows_from_sigmoids(g)
        alpha = self.alpha.to(x.device)
        phis = [(ws[k] * az ** alpha[:, k][None, :, None]).mean(-1) for k in range(self.K)]
        return torch.cat(phis, dim=1)

    def forward(self, x, *args):
        self._check_fitted()
        with torch.no_grad():
            phi = self._all_slot_forward(x).index_select(1, self.active_flat.to(x.device))
            if self.stat_scale is not None:
                phi = phi / self.stat_scale.to(x.device)[None, :]
        return phi

    def _get_filters_Kx(self, device):
        if self._filters_Kx is None or self._filters_Kx.device != device:
            self._filters_Kx = self.filters.repeat(1, self.K, 1).to(device)
        return self._filters_Kx

    def _all_slot_grad(self, x):
        device = x.device
        filters = self.filters.to(device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real
        az = torch.sqrt(z ** 2 + self.eps_abs)
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
        D_all = torch.cat(D, dim=1)
        fKx = self._get_filters_Kx(device)
        return torch.fft.ifft(torch.fft.fft(D_all) * fKx).real / x.shape[-1]

    def grad(self, x, v=None, means=None):
        self._check_fitted()
        with torch.no_grad():
            gc = self._all_slot_grad(x).index_select(1, self.active_flat.to(x.device))
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
        M = M / (M.norm(dim=0, keepdim=True) + 1e-30)
        return M.double().cpu().numpy()

    
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

        Jn, K = self.J, self.K
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
            trans_frac=self.trans_frac, eps_abs=self.eps_abs,
            alpha_bounds=self.alpha_bounds, min_region_samples=self.min_region_samples,
            boundary_method=self.boundary_method, model_criterion=self.model_criterion,
            pi_active_min=self.pi_active_min, cond_tol=self.cond_tol,
            filters_shape=tuple(self.filters.shape)), filename)

    @classmethod
    def load_fixed_parameters(cls, filename, filters, map_location=None, verbose=False):
        d = torch.load(filename, map_location=map_location)
        if tuple(filters.shape) != tuple(d["filters_shape"]):
            raise ValueError(
                f"filters shape {tuple(filters.shape)} does not match the shape "
                f"the potential was fit with {tuple(d['filters_shape'])}; "
                f"this state cannot be reused with these filters.")
        obj = cls(filters, num_regions=d["K"], trans_frac=d["trans_frac"],
                   alpha_bounds=d.get("alpha_bounds", (0.2, 8.0)),
                   min_region_samples=d.get("min_region_samples", 30),
                   eps_abs=d["eps_abs"],
                   boundary_method=d.get("boundary_method", "auto"),
                   model_criterion=d.get("model_criterion", "bic"),
                   pi_active_min=d.get("pi_active_min", 1e-3),
                   cond_tol=d.get("cond_tol", 1e-6),
                   verbose=verbose)
        obj.alpha = d["alpha"]; obj.scale = d["scale"]; obj.cuts = d["cuts"]
        obj.sw = d["sw"]; obj.pi = d["pi"]; obj.Keff = d["Keff"]
        obj.active = d["active"]; obj.active_flat = d["active_flat"]
        obj.stat_scale = d["stat_scale"]; obj.num_coefficients = d["num_coefficients"]
        obj.J = d["J"]
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
        filters = self.filters.to(x.device)
        wt = torch.fft.ifft(torch.fft.fft(x) * filters).real
        J = self.alpha.shape[0]
        channels = range(J) if j is None else [j]
        for jj in channels:
            h = wt[:, jj, :].detach().cpu().flatten().numpy()
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
            wt = torch.fft.ifft(self.filters.to(x.device) * torch.fft.fft(x)).real
            wt = wt.detach().cpu().numpy()
        def zmax_of(jj):
            if wt is not None:
                h = wt[:, jj, :].reshape(-1); h = h[np.isfinite(h)]
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
        wt = torch.fft.ifft(self.filters.to(x.device) * torch.fft.fft(x)).real
        h = wt[:, j, :].detach().cpu().reshape(-1).numpy(); h = h[np.isfinite(h)]
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

