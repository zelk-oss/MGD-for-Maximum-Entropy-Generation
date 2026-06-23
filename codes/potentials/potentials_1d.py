import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from scipy import stats
import time 

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


class Scalar_generalized_gaussian(Potential):
    def __init__(
        self,
        filters,
        eps_abs=1e-6,
        eps_scale=1e-6,
        beta_min=0.3,
        beta_max=8.0,
        use_scipy_fit=True,
    ):
        super().__init__()
        self.filters = filters
        self.num_coefficients = filters.shape[1]
        self.eps_abs = eps_abs
        self.eps_scale = eps_scale
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.use_scipy_fit = use_scipy_fit

        self.beta = None
        self.alpha = None

    @property
    def is_fitted(self):
        return self.beta is not None and self.alpha is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Scalar_generalized_gaussian must be fit_reference before forward or grad")

    def fit_reference(self, x):
        filters = self.filters.to(x.device)
        coeff = torch.fft.ifft(filters * torch.fft.fft(x))
        z = abs_eps(coeff, self.eps_abs)
        z_cpu = z.detach().cpu().numpy()

        betas = []
        alphas = []
        for j in range(z_cpu.shape[1]):
            h = z_cpu[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            if h.size == 0:
                beta_hat = 1.0
                alpha_hat = 1.0
            else:
                # 1. Directly calculate alpha from standard deviation
                alpha_hat = float(np.std(h) + self.eps_scale)
                
                # 2. Only fit beta using the fixed alpha (scale)
                if self.use_scipy_fit:
                    try:
                        beta_hat, _, _ = stats.gennorm.fit(h, floc=0, fscale=alpha_hat)
                    except Exception:
                        beta_hat = 1.0
                else:
                    beta_hat = 1.0

            alpha_hat = max(alpha_hat, self.eps_scale)
            beta_hat = float(np.clip(beta_hat, self.beta_min, self.beta_max))
            betas.append(beta_hat)
            alphas.append(alpha_hat)

        dtype = x.dtype if x.is_floating_point() else torch.float32
        device = x.device
        self.beta = torch.tensor(betas, dtype=dtype, device=device)
        self.alpha = torch.tensor(alphas, dtype=dtype, device=device)

    def fit(self, x):
        return self.fit_reference(x)

    def forward(self, x, *args):
        self._check_fitted()
        filters = self.filters.to(x.device)
        coeff = torch.fft.ifft(filters * torch.fft.fft(x))
        z = abs_eps(coeff, self.eps_abs)

        beta = self.beta.to(x.device)[None, :, None]
        alpha = self.alpha.to(x.device)[None, :, None]

        r = torch.sqrt((z) ** 2 + self.eps_abs**2)
        phi = (r / alpha) ** beta
        return phi.mean(-1)

    def grad(self, x, v=None, means=None):
        self._check_fitted()
        filters = self.filters.to(x.device)
        coeff = torch.fft.ifft(filters * torch.fft.fft(x))
        z = abs_eps(coeff, self.eps_abs)

        beta = self.beta.to(x.device)[None, :, None]
        alpha = self.alpha.to(x.device)[None, :, None]

        r = torch.sqrt((z) ** 2 + self.eps_abs**2)
        dphi_dz = (beta / alpha) * (r / alpha) ** (beta - 1.0) * (z) / r
        coeff_over_abs = coeff / abs_eps(coeff, self.eps_abs)
        grad_coeff = dphi_dz * coeff_over_abs
        output = torch.fft.ifft(torch.fft.fft(grad_coeff) * filters).real / x.shape[-1]

        if v is None:
            return output
        return (output * v[None, :, None]).sum(1)[:, None]


# ---------------------------------------------------------------------------
# Student-t scalar potential
# ---------------------------------------------------------------------------

class Scalar_student_t:
    """
    Per-channel Student-t potential for wavelet coefficients.

    The marginal density is

        p(x) ∝ (1 + x² / (ν α²))^{-(ν+1)/2}

    and the corresponding potential (negative log up to constants) is

        φ(x) = (ν+1)/2 · log(1 + x² / (ν α²))

    Tail behaviour: p(x) ~ |x|^{-(ν+1)}, i.e. algebraic — much heavier
    than the GGD stretched-exponential, which is exactly what the residuals
    call for.

    Parameters
    ----------
    filters : torch.Tensor, shape (1, J, M)
        Wavelet filters in Fourier space (same convention as the rest of
        the codebase).
    eps_abs : float
        Regularisation inside abs_eps.
    eps_scale : float
        Floor on the fitted scale α to avoid division by zero.
    nu_min, nu_max : float
        Bounds on the fitted degrees-of-freedom ν.  ν→∞ recovers Gaussian;
        ν=1 is Cauchy.  For turbulence ν ∈ [1, 10] is typical.
    """

    def __init__(
        self,
        filters: torch.Tensor,
        eps_abs: float = 1e-6,
        eps_scale: float = 1e-6,
        nu_min: float = 0.5,
        nu_max: float = 50.0,
    ):
        self.filters = filters
        self.num_coefficients = filters.shape[1]
        self.eps_abs = eps_abs
        self.eps_scale = eps_scale
        self.nu_min = nu_min
        self.nu_max = nu_max

        # set by fit_reference()
        self.nu: torch.Tensor | None = None    # shape (J,)
        self.alpha: torch.Tensor | None = None  # shape (J,)  — the scale σ

    # ------------------------------------------------------------------
    # Fitted state
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self.nu is not None and self.alpha is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                "Scalar_student_t must be fit_reference'd before forward or grad"
            )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_reference(self, x: torch.Tensor):
        """
        Fit ν and α independently for each wavelet channel by MLE.

        scipy.stats.t parameterises as t(df, loc, scale); we fix loc=0
        (the marginals are visually symmetric) and fit df and scale jointly.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, T)
            Reference signal batch (same format as the rest of the pipeline).
        """
        t0_total = time.time()
        print("[StudentT] Starting fit_reference")
        print(f"[StudentT] Input shape: {x.shape}, device: {x.device}")

        t0_fft = time.time() 
        filters = self.filters.to(x.device)
        coeff = torch.fft.ifft(filters * torch.fft.fft(x))
        # Use real part of the (real-valued) wavelet response
        z_cpu = coeff.real.detach().cpu().numpy()   # (B, J, T)
        print(f"[StudentT] FFT + transfer to CPU took {time.time() - t0_fft:.3f}s")

        nus, alphas = [], []
        print(f"[StudentT] Fitting {z_cpu.shape[1]} channels...")

        for j in range(z_cpu.shape[1]):
            t0_ch = time.time()

            h = z_cpu[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]

            print(f"[StudentT][ch {j}] samples: {h.size}", end=" | ")

            if h.size < 10:
                nu_hat, alpha_hat = 3.0, 1.0
                print("too few samples → fallback")
            else:
                try:
                    t0_mle = time.time()
                    df_hat, _, scale_hat = stats.t.fit(h, floc=0)
                    print(f"MLE took {time.time() - t0_mle:.3f}s", end=" | ")

                    nu_hat = float(np.clip(df_hat, self.nu_min, self.nu_max))
                    alpha_hat = float(max(scale_hat, self.eps_scale))
                    print(f"nu={nu_hat:.3f}, alpha={alpha_hat:.3f}")

                except Exception as e:
                    print(f"MLE failed ({e}) → fallback")
                    nu_hat, alpha_hat = 3.0, float(np.std(h) + self.eps_scale)

            print(f"[StudentT][ch {j}] total time {time.time() - t0_ch:.3f}s")

            nus.append(nu_hat)
            alphas.append(alpha_hat)

        dtype = x.dtype if x.is_floating_point() else torch.float32
        device = x.device
        self.nu    = torch.tensor(nus,    dtype=dtype, device=device)
        self.alpha = torch.tensor(alphas, dtype=dtype, device=device)

        print(f"[StudentT] Total fit time: {time.time() - t0_total:.3f}s")

    # Alias used by the rest of the codebase
    def fit(self, x: torch.Tensor):
        return self.fit_reference(x)

    # ------------------------------------------------------------------
    # Potential  φ(x) = (ν+1)/2 · log(1 + x²/(ν α²))
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """
        Returns φ averaged over time, shape (B, J).

        φ_j(x) = mean_t  (ν_j+1)/2 · log(1 + z_{j,t}² / (ν_j α_j²))
        """
        self._check_fitted()
        filters = self.filters.to(x.device)
        coeff = torch.fft.ifft(filters * torch.fft.fft(x))
        z = coeff.real                                      # (B, J, T)

        nu    = self.nu.to(x.device)[None, :, None]        # (1, J, 1)
        alpha = self.alpha.to(x.device)[None, :, None]

        ratio = z ** 2 / (nu * alpha ** 2 + self.eps_abs)
        phi   = 0.5 * (nu + 1.0) * torch.log1p(ratio)     # numerically stable

        return phi.mean(-1)                                 # (B, J)

    # ------------------------------------------------------------------
    # Gradient  ∂φ/∂x  (needed for score / MALA)
    # ------------------------------------------------------------------

    def grad(
        self,
        x: torch.Tensor,
        v: torch.Tensor | None = None,
        means=None,
    ) -> torch.Tensor:
        """
        Gradient of the mean-time potential w.r.t. x.

        ∂φ_j/∂z_{j,t} = (ν_j+1) · z_{j,t} / (ν_j α_j² + z_{j,t}²)

        The chain rule through the FFT then gives the gradient w.r.t. x.

        If v is provided, returns the vector-Jacobian product summed over
        channels, matching the calling convention of the other potentials.
        """
        self._check_fitted()
        filters = self.filters.to(x.device)
        coeff   = torch.fft.ifft(filters * torch.fft.fft(x))
        z       = coeff.real                                # (B, J, T)

        nu    = self.nu.to(x.device)[None, :, None]
        alpha = self.alpha.to(x.device)[None, :, None]

        # ∂φ/∂z  (Cauchy / Lorentzian score)
        denom    = nu * alpha ** 2 + z ** 2 + self.eps_abs
        dphi_dz  = (nu + 1.0) * z / denom                  # (B, J, T)

        # Chain rule back through the convolution:
        # coeff = ifft(filter * fft(x))  =>  d/dx = ifft(filter * fft(·)).real
        # So  ∂L/∂x = real[ ifft( conj(filter) * fft(∂L/∂z) ) ] / T
        # (factor 1/T from the mean over time in forward)
        grad_coeff = torch.fft.ifft(
            torch.fft.fft(dphi_dz) * filters
        ).real / x.shape[-1]                               # (B, J, T)

        if v is None:
            return grad_coeff                               # (B, J, T)

        # v has shape (J,): project onto channels and sum
        return (grad_coeff * v[None, :, None]).sum(1)[:, None]   # (B, 1, T)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print fitted ν and α for each channel."""
        self._check_fitted()
        nu    = self.nu.cpu().numpy()
        alpha = self.alpha.cpu().numpy()
        print(f"{'Channel':>8}  {'nu (df)':>10}  {'alpha (scale)':>14}")
        print("-" * 38)
        for j, (n, a) in enumerate(zip(nu, alpha)):
            print(f"{j:>8d}  {n:>10.4f}  {a:>14.6f}")


class Scalar_generalized_t:
    """
    Per-channel generalized-t potential for wavelet coefficients.

        p(x) ∝ (1 + |x|^p / (q α^p))^{-(q + 1/p)}
        φ(x) = (q + 1/p) · log(1 + |x|^p / (q α^p))

    Two shape knobs: p (body/cusp exponent), q (tail weight).
    Limits: q→∞ ⇒ GGD(shape p);  p=2 ⇒ Student-t with ν = 2q.
    Tail: p(x) ~ |x|^{-(pq+1)}.
    """

    def __init__(self, filters, eps_abs=1e-6, eps_scale=1e-6,
                 p_min=0.2, p_max=4.0, q_min=0.3, q_max=1e4):
        self.filters = filters
        self.num_coefficients = filters.shape[1]
        self.eps_abs, self.eps_scale = eps_abs, eps_scale
        self.p_min, self.p_max = p_min, p_max
        self.q_min, self.q_max = q_min, q_max
        self.p = self.q = self.alpha = None          # shape (J,), set by fit_reference

    @property
    def is_fitted(self):
        return self.p is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Scalar_generalized_t must be fit_reference'd first")

    # ------------------------------------------------------------------
    # Per-channel MLE
    # ------------------------------------------------------------------
    @staticmethod
    def _fit_channel(h):
        def nll(theta):
            p, q, s = np.exp(theta)
            t = np.power(np.abs(h) / s, p) / q
            lp = (np.log(p) - np.log(2.0) - np.log(s)
                  - np.log(q) / p - betaln(1.0 / p, q)
                  - (q + 1.0 / p) * np.log1p(t))
            return -lp.sum()
        theta0 = np.log([1.0, 5.0, np.std(h) or 1.0])
        res = minimize(nll, theta0, method="L-BFGS-B")
        return np.exp(res.x)                          # (p, q, s)

    def fit_reference(self, x):
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real.detach().cpu().numpy()
        ps, qs, al = [], [], []
        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            if h.size < 10:
                p_hat, q_hat, s_hat = 1.0, 5.0, 1.0
            else:
                try:
                    p_hat, q_hat, s_hat = self._fit_channel(h)
                except Exception as e:
                    print(f"[GenT][ch {j}] MLE failed ({e}) → fallback")
                    p_hat, q_hat, s_hat = 1.0, 5.0, float(np.std(h) + self.eps_scale)
            p_hat = float(np.clip(p_hat, self.p_min, self.p_max))
            q_hat = float(np.clip(q_hat, self.q_min, self.q_max))
            s_hat = float(max(s_hat, self.eps_scale))
            print(f"[GenT][ch {j}] p={p_hat:.3f}, q={q_hat:.3f}, alpha={s_hat:.4f}")
            ps.append(p_hat); qs.append(q_hat); al.append(s_hat)

        dtype = x.dtype if x.is_floating_point() else torch.float32
        self.p     = torch.tensor(ps, dtype=dtype, device=x.device)
        self.q     = torch.tensor(qs, dtype=dtype, device=x.device)
        self.alpha = torch.tensor(al, dtype=dtype, device=x.device)

    def fit(self, x):
        return self.fit_reference(x)

    # ------------------------------------------------------------------
    # φ(x) = (q + 1/p) · log(1 + |x|^p / (q α^p))
    # ------------------------------------------------------------------
    def _params(self, device):
        return (self.p.to(device)[None, :, None],
                self.q.to(device)[None, :, None],
                self.alpha.to(device)[None, :, None])

    def forward(self, x, *args):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real          # (B, J, T)
        p, q, a = self._params(x.device)
        az = torch.sqrt(z ** 2 + self.eps_abs)                       # stabilized |z|
        u = az.pow(p) / (q * a.pow(p))
        return ((q + 1.0 / p) * torch.log1p(u)).mean(-1)             # (B, J)

    # ------------------------------------------------------------------
    # ∂φ/∂z = (q + 1/p) · p · |z|^{p-1} sign(z) / (q α^p + |z|^p)
    # ------------------------------------------------------------------
    def grad(self, x, v=None, means=None):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real          # (B, J, T)
        p, q, a = self._params(x.device)
        az = torch.sqrt(z ** 2 + self.eps_abs)
        azp = az.pow(p)
        dphi_dz = (q + 1.0 / p) * p * az.pow(p - 2.0) * z / (q * a.pow(p) + azp)

        grad_coeff = torch.fft.ifft(
            torch.fft.fft(dphi_dz) * filters
        ).real / x.shape[-1]                                          # (B, J, T)

        if v is None:
            return grad_coeff
        return (grad_coeff * v[None, :, None]).sum(1)[:, None]        # (B, 1, T)

    # ------------------------------------------------------------------
    def summary(self):
        self._check_fitted()
        p, q, a = self.p.cpu().numpy(), self.q.cpu().numpy(), self.alpha.cpu().numpy()
        print(f"{'Channel':>8}  {'p':>8}  {'q':>8}  {'alpha':>12}  {'tail pq':>9}")
        print("-" * 52)
        for j, (pp, qq, aa) in enumerate(zip(p, q, a)):
            print(f"{j:>8d}  {pp:>8.4f}  {qq:>8.4f}  {aa:>12.6f}  {pp*qq:>9.3f}")


import numpy as np
import torch
import torch.nn.functional as F
import math
from scipy.integrate import quad
from scipy.optimize import minimize


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
                 bulk_quantile=0.90, trans_quantiles=(0.85, 0.97),
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
        2: [0.90],
        3: [0.70, 0.92],
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




def _robust_scale(h):
    s = 1.4826 * np.median(np.abs(h - np.median(h)))
    return float(s) or float(np.std(h)) or 1.0

def maxent_fit(h, p0_bounds=(0.2, 1.5), th3_min=None, fixed_p0=None, fixed_s0=None):
    scale_ref = _robust_scale(h)
    if th3_min is None:
        th3_min = 1e-3 / scale_ref               # constant exp-tail floor (≠ fitted s0)
    free_p0 = fixed_p0 is None
    p0_lo, p0_hi = p0_bounds

    def unpack(u):
        t1 = _softplus(u[0]); t2 = _softplus(u[1]); t3 = th3_min + _softplus(u[2])
        k = 3
        if free_p0:
            p0 = p0_lo + (p0_hi - p0_lo) / (1.0 + np.exp(-u[k])); k += 1
        else:
            p0 = float(fixed_p0)
        if fixed_s0 is not None:
            s0 = float(fixed_s0)
        else:
            s0 = np.exp(u[k])                         # s0 > 0, fitted
        return np.array([t1, t2, t3]), p0, s0

    def nll(u):                                  # NLL/N = θ·Ê[φ] + log Z(θ,p0,s0)
        theta, p0, s0 = unpack(u)
        E1 = np.mean(np.abs(h) ** p0)
        E2 = np.mean(np.log1p((h / s0) ** 2))
        E3 = np.mean(np.abs(h))
        return theta[0] * E1 + theta[1] * E2 + theta[2] * E3 + maxent_logZ(theta, p0, s0)

    n = 3 + (1 if free_p0 else 0) + (0 if fixed_s0 is not None else 1)
    u0 = np.zeros(n)
    if fixed_s0 is None:
        u0[-1] = np.log(scale_ref)  # s0 init = robust scale
    res = minimize(nll, u0, method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-4, maxiter=15000))
    theta, p0, s0 = unpack(res.x)
    return theta, float(p0), float(s0)

def maxent_logZ(theta, p0, s0):
    t1, t2, t3 = theta
    f = lambda x: np.exp(-(t1 * x ** p0 + t2 * np.log1p((x / s0) ** 2) + t3 * x))
    I, _ = quad(f, 0.0, np.inf, limit=200)
    return np.log(2.0 * I)                       # symmetric → 2 × half-integral

def maxent_logpdf(x, theta, p0, s0):
    t1, t2, t3 = theta
    psi = t1 * np.abs(x) ** p0 + t2 * np.log1p((x / s0) ** 2) + t3 * np.abs(x)
    return -psi - maxent_logZ(theta, p0, s0)

def maxent_pdf(x, theta, p0, s0):
    return np.exp(maxent_logpdf(x, theta, p0, s0))

def _softplus(z):
    return np.logaddexp(0.0, z)

# --- stage 2: convex θ-only fit at FROZEN (p0, s0) = moment matching --
def maxent_fit_theta(h, p0, s0, th3_min=None):
    if th3_min is None:
        th3_min = 1e-3 / s0
    E1 = np.mean(np.abs(h) ** p0)
    E2 = np.mean(np.log1p((h / s0) ** 2))
    E3 = np.mean(np.abs(h))
    def unpack(u):
        return np.array([_softplus(u[0]), _softplus(u[1]), th3_min + _softplus(u[2])])
    def obj(u):                                  # convex in θ
        th = unpack(u)
        return th[0] * E1 + th[1] * E2 + th[2] * E3 + maxent_logZ(th, p0, s0)
    res = minimize(obj, np.zeros(3), method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-4, maxiter=8000))
    return unpack(res.x)

class Scalar_maxent:
    """
    Per-channel max-entropy potential, linear in θ over fixed features:

        φ(x)     = ( |x|^{p0},  log(1 + (x/s0)^2),  |x| )
        φ_pot(x) = θ · φ(x)        (scalar potential; const log Z dropped)

    fit_reference jointly fits (θ, p0, s0) per channel, then FREEZES p0 and s0
    → φ is fixed and θ is a convex moment-matching parameter.
    refit_theta re-solves θ alone (frozen φ), e.g. on new data.

    θ1≥0 cusp | θ2≥0 power shoulder (tail idx 2θ2) | θ3>0 exp tempering.
    """

    def __init__(self, filters, eps_abs=1e-6, eps_scale=1e-6,
                 fixed_p0=None, p0_bounds=(0.2, 1.5)):
        self.filters = filters
        self.num_coefficients = filters.shape[1]*3
        self.eps_abs, self.eps_scale = eps_abs, eps_scale
        self.fixed_p0, self.p0_bounds = fixed_p0, p0_bounds
        self.theta = self.p0 = self.s0 = None     # (J,3), (J,), (J,)

    @property
    def is_fitted(self):
        return self.theta is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Scalar_maxent must be fit_reference'd first")

    def _coeffs_np(self, x):
        filters = self.filters.to(x.device)
        return torch.fft.ifft(filters * torch.fft.fft(x)).real.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Stage 1: fit (θ, p0, s0) per channel; keep p0 and s0  → φ fixed
    # ------------------------------------------------------------------
    def fit_reference(self, x):
        z = self._coeffs_np(x)
        TH, P0, S0 = [], [], []
        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            if h.size < 10:
                s0 = _robust_scale(h) if h.size else 1.0
                theta, p0 = np.array([1.0, 1.0, 1.0 / s0]), 0.5
            else:
                try:
                    theta, p0, s0 = maxent_fit(h, p0_bounds=self.p0_bounds,
                                               fixed_p0=self.fixed_p0)
                except Exception as e:
                    print(f"[MaxEnt][ch {j}] fit failed ({e}) → fallback")
                    s0 = _robust_scale(h)
                    theta, p0 = np.array([1.0, 1.0, 1.0 / s0]), 0.5
            print(f"[MaxEnt][ch {j}] p0={p0:.3f}, s0={s0:.4f}, "
                  f"theta=({theta[0]:.3f},{theta[1]:.3f},{theta[2]:.3f})")
            TH.append(theta); P0.append(p0); S0.append(s0)

        dtype = x.dtype if x.is_floating_point() else torch.float32
        self.theta = torch.tensor(np.array(TH), dtype=dtype, device=x.device)
        self.p0    = torch.tensor(P0, dtype=dtype, device=x.device)
        self.s0    = torch.tensor(S0, dtype=dtype, device=x.device)

    def fit(self, x):
        return self.fit_reference(x)

    # ------------------------------------------------------------------
    # Re-solve θ only, (p0, s0) frozen — convex moment matching
    # ------------------------------------------------------------------
    def refit_theta(self, x):
        self._check_fitted()
        z = self._coeffs_np(x)
        TH = []
        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            TH.append(maxent_fit_theta(h, float(self.p0[j]), float(self.s0[j])))
        self.theta = torch.tensor(np.array(TH), dtype=self.theta.dtype,
                                  device=self.theta.device)

    # ------------------------------------------------------------------
    def _bcast(self, x):
        th = self.theta.to(x.device)
        return (th[:, 0][None, :, None], th[:, 1][None, :, None], th[:, 2][None, :, None],
                self.p0.to(x.device)[None, :, None], self.s0.to(x.device)[None, :, None])

    # φ_pot(x) = θ1|x|^p0 + θ2 log(1+(x/s0)^2) + θ3|x|
    def forward(self, x, *args):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real          # (B, J, T)
        t1, t2, t3, p0, s0 = self._bcast(x)
        az = torch.sqrt(z ** 2 + self.eps_abs)
        # full potential 
        psi = torch.stack([az.pow(p0), torch.log1p((z / s0) ** 2), az], dim=1) # (B, 3, J, T) 
        psi = psi.mean(dim=-1)    # (B, 3, J)  
        return psi.reshape((psi.shape[0], -1))  # (B, 3J)

    # ∂φ_pot/∂z = θ1·p0·|z|^{p0-2}z + θ2·2z/(s0²+z²) + θ3·z/|z|
    def grad(self, x, v=None, means=None):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real          # (B, J, T)
        t1, t2, t3, p0, s0 = self._bcast(x)
        az = torch.sqrt(z ** 2 + self.eps_abs)
        dpsi = torch.stack([p0 * az.pow(p0 - 2.0) * z, (2.0 * z / (s0 ** 2 + z ** 2 + self.eps_abs)), (z / az)], dim=1)  # (B, 3, J, T)

        grad_coeff = torch.fft.ifft(
            torch.fft.fft(dpsi) * filters
        ).real / x.shape[-1]                                          # (B, 3, J, T)
        grad_coeff = grad_coeff.reshape((grad_coeff.shape[0],-1, grad_coeff.shape[-1])) # (B, 3J, T) 

        if v is None:
            return grad_coeff
        return (grad_coeff * v[None, :, None]).sum(1)[:, None]        # (B, 1, T)

    # ------------------------------------------------------------------
    def summary(self):
        self._check_fitted()
        th = self.theta.cpu().numpy(); p0 = self.p0.cpu().numpy(); s0 = self.s0.cpu().numpy()
        print(f"{'Channel':>8}  {'p0':>6}  {'s0':>10}  {'θ1':>8}  {'θ2':>8}  "
              f"{'θ3':>8}  {'tail 2θ2':>9}")
        print("-" * 64)
        for j in range(len(p0)):
            print(f"{j:>8d}  {p0[j]:>6.3f}  {s0[j]:>10.5f}  {th[j,0]:>8.4f}  "
                  f"{th[j,1]:>8.4f}  {th[j,2]:>8.4f}  {2*th[j,1]:>9.3f}")
            


class Scalar_maxent_log:
    """
    Per-channel max-entropy potential, linear in θ over fixed features:

        φ(x)     = (log(1 + (x/s0)^2))
        φ_pot(x) = θ · φ(x)        (scalar potential; const log Z dropped)

    fit_reference jointly fits (θ, p0, s0) per channel, then FREEZES p0 and s0
    → φ is fixed and θ is a convex moment-matching parameter.
    refit_theta re-solves θ alone (frozen φ), e.g. on new data.

    θ1≥0 cusp | θ2≥0 power shoulder (tail idx 2θ2) | θ3>0 exp tempering.
    """

    def __init__(self, filters, eps_abs=1e-6, eps_scale=1e-6,
                 fixed_p0=None, p0_bounds=(0.2, 1.5)):
        self.filters = filters
        self.num_coefficients = filters.shape[1]
        self.eps_abs, self.eps_scale = eps_abs, eps_scale
        self.fixed_p0, self.p0_bounds = fixed_p0, p0_bounds
        self.theta = self.p0 = self.s0 = None     # (J,3), (J,), (J,)

    @property
    def is_fitted(self):
        return self.theta is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Scalar_maxent must be fit_reference'd first")

    def _coeffs_np(self, x):
        filters = self.filters.to(x.device)
        return torch.fft.ifft(filters * torch.fft.fft(x)).real.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Stage 1: fit (θ, p0, s0) per channel; keep p0 and s0  → φ fixed
    # ------------------------------------------------------------------
    def fit_reference(self, x=None, z=None, s0=None):
        if z is not None:
            z_np = z.detach().cpu().numpy()
        else:
            z_np = self._coeffs_np(x)
        fixed_s0 = None
        if s0 is not None:
            if isinstance(s0, torch.Tensor):
                fixed_s0 = s0.detach().cpu().numpy()
            else:
                fixed_s0 = np.asarray(s0)
        TH, P0, S0 = [], [], []
        for j in range(z_np.shape[1]):
            h = z_np[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            fixed_s0_j = None if fixed_s0 is None else float(fixed_s0[j])
            if h.size < 10:
                s0 = fixed_s0_j if fixed_s0_j is not None else (_robust_scale(h) if h.size else 1.0)
                theta, p0 = np.array([1.0, 1.0, 1.0 / s0]), 0.5
            else:
                try:
                    theta, p0, s0 = maxent_fit(h, p0_bounds=self.p0_bounds,
                                               fixed_p0=self.fixed_p0,
                                               fixed_s0=fixed_s0_j)
                except Exception as e:
                    print(f"[MaxEnt][ch {j}] fit failed ({e}) → fallback")
                    s0 = _robust_scale(h)
                    theta, p0 = np.array([1.0, 1.0, 1.0 / s0]), 0.5
            print(f"[MaxEnt][ch {j}] p0={p0:.3f}, s0={s0:.4f}, "
                  f"theta=({theta[0]:.3f},{theta[1]:.3f},{theta[2]:.3f})")
            TH.append(theta); P0.append(p0); S0.append(s0)

        dtype = x.dtype if x.is_floating_point() else torch.float32
        self.theta = torch.tensor(np.array(TH), dtype=dtype, device=x.device)
        self.p0    = torch.tensor(P0, dtype=dtype, device=x.device)
        self.s0    = torch.tensor(S0, dtype=dtype, device=x.device)

    def fit(self, x=None, z=None, s0=None):
        return self.fit_reference(x=x, z=z, s0=s0)

    # ------------------------------------------------------------------
    # Re-solve θ only, (p0, s0) frozen — convex moment matching
    # ------------------------------------------------------------------
    def refit_theta(self, x):
        self._check_fitted()
        z = self._coeffs_np(x)
        TH = []
        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            TH.append(maxent_fit_theta(h, float(self.p0[j]), float(self.s0[j])))
        self.theta = torch.tensor(np.array(TH), dtype=self.theta.dtype,
                                  device=self.theta.device)

    # ------------------------------------------------------------------
    def _bcast(self, x):
        th = self.theta.to(x.device)
        return (th[:, 0][None, :, None], th[:, 1][None, :, None], th[:, 2][None, :, None],
                self.p0.to(x.device)[None, :, None], self.s0.to(x.device)[None, :, None])

    # φ_pot(x) = θ1|x|^p0 + θ2 log(1+(x/s0)^2) + θ3|x|
    def forward(self, x, *args):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real          # (B, J, T)
        _, t2, _, _, s0 = self._bcast(x)
        # full potential 
        # psi = t1 * az.pow(p0) + t2 * torch.log1p((z / s0) ** 2) + t3 * az
        # only the log 
        psi = t2 * torch.log1p(1 + (z / s0) ** 2) 
        return psi.mean(-1)                                          # (B, J)

    # ∂φ_pot/∂z = θ1·p0·|z|^{p0-2}z + θ2·2z/(s0²+z²) + θ3·z/|z|
    def grad(self, x, v=None, means=None):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real          # (B, J, T)
        _, t2, _, _, s0 = self._bcast(x)
        dpsi = (t2 * (2.0 * z / (s0 ** 2 + z ** 2 + self.eps_abs)))  # (B, J, T)

        grad_coeff = torch.fft.ifft(
            torch.fft.fft(dpsi) * filters
        ).real / x.shape[-1]                                          # (B, J, T)

        if v is None:
            return grad_coeff
        return (grad_coeff * v[None, :, None]).sum(1)[:, None]        # (B, 1, T)

    # ------------------------------------------------------------------
    def summary(self):
        self._check_fitted()
        th = self.theta.cpu().numpy(); p0 = self.p0.cpu().numpy(); s0 = self.s0.cpu().numpy()
        print(f"{'Channel':>8}  {'p0':>6}  {'s0':>10}  {'θ1':>8}  {'θ2':>8}  "
              f"{'θ3':>8}  {'tail 2θ2':>9}")
        print("-" * 64)
        for j in range(len(p0)):
            print(f"{j:>8d}  {p0[j]:>6.3f}  {s0[j]:>10.5f}  {th[j,0]:>8.4f}  "
                  f"{th[j,1]:>8.4f}  {th[j,2]:>8.4f}  {2*th[j,1]:>9.3f}")
            





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