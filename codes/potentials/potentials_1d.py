import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

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

