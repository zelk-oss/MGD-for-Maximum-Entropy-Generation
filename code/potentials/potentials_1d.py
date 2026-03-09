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
    def __init__(self, J, Q, filters, filters_Q):
        super().__init__()
        self.J = J
        self.Q = Q
        self.filters = filters
        self.filters_Q = filters_Q
        self.indices = indices_fourth_order_Q(self.J, self.Q)
        self.num_coefficients = len(self.indices[0])

    def forward(self, x):
        filters = self.filters.to(x.device)
        filters_Q = self.filters_Q.to(x.device)
        x_filtered = torch.fft.ifft(filters_Q*torch.fft.fft(x)) #(B,JQ,T)
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

        

        x_filtered_abs_no_LF_filtered_2 = torch.fft.ifft((torch.fft.fft(x_filtered_abs_no_LF)[:,:,None]*filters**2)) # (B,J*Q,J+1,M,N) #no 2*, x_filtered_abs_no_LF
        
        indices = self.indices

        if v != None:
            m = torch.zeros((number_filters_Q, number_filters_Q, number_filters)).to(x.device)
            m[indices[0], indices[1], indices[2]] = v
            m = m+torch.transpose(m,0,1)
            
            intermediate_output = x_filtered_over_abs_no_LF*torch.einsum('ijk, ajkb -> aib', m, torch.real(x_filtered_abs_no_LF_filtered_2))
            return torch.fft.ifft(filters_Q*torch.fft.fft(intermediate_output)).real.sum(1)[:,None]/x.shape[-1]
            
        intermediate_output = x_filtered_over_abs_no_LF[:,:,None,None]*torch.real(x_filtered_abs_no_LF_filtered_2)[:,None] #x_filtered_over_abs_no_LF
        output = torch.fft.ifft(filters_Q[:,:,None,None]*torch.fft.fft(intermediate_output)).real

        output = output + torch.transpose(output, 1,2)
        output = output[:, indices[0], indices[1], indices[2]].reshape(x.shape[0], indices.shape[1], x.shape[-1])/x.shape[-1]

        return output



        
    # ----- Other potentials -----


class L2p_norm(Potential):
    def __init__(self, p,filters):
        super().__init__()
        self.p = p
        self.filters=filters
 
    def forward(self, x):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft(filters*torch.fft.fft(x))
        return (torch.abs(x_filtered)**(2*self.p)).mean(-1)

    def grad_autograd(self, x):
        filters = self.filters.to(x.device)
        return torch.func.vmap(torch.func.jacrev(self.forward))(x[:,None]).reshape((x.shape[0],filters.shape[1], x.shape[-1]))

    def grad(self, x,  v=None, means=None):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft(filters*torch.fft.fft(x))
            
        output = (x_filtered *(2* self.p) * (torch.abs(x_filtered)**(2*self.p-2)))
        output = torch.fft.ifft(filters*torch.fft.fft(output)).real
        
        if v==None:
            return output/x.shape[-1]
        else:
            return (output*v[None,:,None]).sum(1)[:,None]/x.shape[-1]

class L2p1_norm(Potential):
    def __init__(self, p,filters):
        super().__init__()
        self.p = p
        self.filters=filters
 
    def forward(self, x):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft(filters*torch.fft.fft(x))
        return (torch.abs(x_filtered)**(2*self.p+1)).mean(-1)

    def grad_autograd(self, x):
        filters = self.filters.to(x.device)
        return torch.func.vmap(torch.func.jacrev(self.forward))(x[:,None]).reshape((x.shape[0],filters.shape[1], x.shape[-1]))

    def grad(self, x,  v=None, means=None):
        filters = self.filters.to(x.device)
         
        x_filtered = torch.fft.ifft(self.filters*torch.fft.fft(x))
            
        output =  x_filtered * (2*self.p+1) * (torch.abs(x_filtered)**(2*self.p-1))
        output = torch.fft.ifft(filters*torch.fft.fft(output)).real
        
        if v==None:
            return output/x.shape[-1]
        else:
            return (output*v[None,:,None]).sum(1)[:,None]/x.shape[-1]

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


class Log_norm(Potential):
    def __init__(self,filters,alpha,epsilon= 1e-5):
        super().__init__()
        self.filters = filters
        self.alpha = alpha
        self.epsilon = epsilon 

    def forward(self,x):
        filters = self.filters.to(x.device)
        x_filtered = torch.fft.ifft(filters*torch.fft.fft(x))
        x_filtered_abs = abs_eps(x_filtered)
        return (torch.log(x_filtered_abs+self.epsilon)**self.alpha).mean(-1)
        
    def grad_autograd(self, x):
        return torch.func.vmap(torch.func.jacrev(self.forward))(x[:,None]).reshape((x.shape[0],-1, x.shape[-1]))

    def grad(self, x, v=None, means=None):
        filters = self.filters.to(x.device)
        
        x_fourier = torch.fft.fft(x)
        x_filtered = torch.fft.ifft(filters*x_fourier)
        x_filtered_abs = abs_eps(x_filtered)
        x_filtered_over_abs = x_filtered/x_filtered_abs

        #x_L1 = torch.real(torch.fft.ifft(torch.fft.fft(x_filtered_over_abs)*self.filters)) #(B,J*Q,T)

        x_log =  self.alpha*torch.log(x_filtered_abs+self.epsilon)**(self.alpha-1) / (x_filtered_abs+self.epsilon)
        x_log = x_log * x_filtered_over_abs
        x_log = torch.real(torch.fft.ifft(torch.fft.fft(x_log)*self.filters)) #(B,J*Q,T)
        
        output = x_log#torch.fft.ifft(torch.fft.fft(x_L1)[:,None]*torch.fft.fft(x_scalar)).real #(B,J*Q,T) 
        output = output.reshape((output.shape[0],-1,output.shape[-1]))  #(B,n_pots*J*Q,T)
        
        if v==None:
            return output/x.shape[-1]
        else:
            return (output*v[None,:,None]).sum(1)[:,None]/x.shape[-1]