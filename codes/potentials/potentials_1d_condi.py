import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

import torch.nn as nn

from potentials.utils_potentials import *
from .potentials_1d import *



    ## ----------------------------------------------------- Mother classes -----------------------------------------------------    


class Potential_Prepare_condi(nn.Module):
    def __init__(self,potential):
        super().__init__()
        self.potential= potential
    def forward(self,x,x_condi,v=None,argument = 'forward'):
        if argument == 'forward':
            #print(x.device,x_condi.device,'here')
            return self.potential(x,x_condi)
        elif argument == 'grad':
            return self.potential.grad(x,x_condi,v)
        elif argument == 'fit':
            print(x.device,x_condi.device,'fit')
            self.potential.fit(x,x_condi)
        else:
            pass

class Potential_Parallel_condi(nn.Module):
    def __init__(self,potential):
        super().__init__()
        self._wrapped = potential
        self.potential = nn.DataParallel(Potential_Prepare(potential))
     
        #self.grad = potential.grad

    @property
    def is_fitted(self):
        return getattr(self._wrapped, 'is_fitted', True)

    @property
    def requires_reference_fit(self):
        return getattr(self._wrapped, 'requires_reference_fit', False)

    def forward(self,x):
        return self.potential(x,argument='forward')

    def grad(self,x,v=None):
        n_gpu = torch.cuda.device_count()
        if v is not None and n_gpu != 0:
            v = v.repeat((n_gpu,))
        return self.potential(x,v,argument='grad')
    @property
    def num_coefficients(self):
        return getattr(self._wrapped, 'num_coefficients', None)
    def fit(self,x):
        print('fit_Parallel')
        self.potential(x,argument='fit')

class Potential_Condi(nn.Module):
  
    def __init__(self,potential,W,parallel =False):
        super().__init__()
        self._wrapped = potential
        if parallel is False:
            self.potential = potential
        else: 
            self.potential =  Potential_Parallel_condi(nn.DataParallel(Potential_Prepare(potential)))
        self.W = W
        self.num_potentials = None
        self.num_coefficients = self.potential.num_coefficients
        
    @property
    def is_fitted(self):
        return getattr(self._wrapped, 'is_fitted', True)

    @property
    def requires_reference_fit(self):
        return getattr(self._wrapped, 'requires_reference_fit', False)

    def forward(self,x,x_condi):
        return self.potential(self.W(x,x_condi))
    def grad(self,x,x_condi,v=None):
        x = self.W(x,x_condi)
        gradient = self.potential.grad(x,v) #(B,n_potentials,T)
        gradient,gradient_condi = self.W.decompose(gradient) #deco on last dimension
        return gradient
         
    def fit(self,x,x_condi):
        self.potential.fit(self.W(x,x_condi))
        
    def fit_micro(self,x,x_condi):
        self.potential.fit_micro(self.W(x,x_condi))



    ## ----------------------------------------------------- Potentials definitions -----------------------------------------------------    


    # ----- Scattering potentials -----

class Scattering_Fourth_Order_Real_1d_Condi(Scattering_Fourth_Order_Real_1d):
    def __init__(self, J,Q,filters,filters_Q,include_diag = True,lite=False):
        super().__init__(J,Q,filters,filters_Q,include_diag,lite)

        #Overwrite
        if include_diag is True:
            offset = 0
        else:
            offset = 1
        self.indices = indices_fourth_order_Q_Condi(self.J, self.Q,offset,lite)
        self.norm_indices = torch.ones((len(self.indices[0]),))
        self.num_coefficients = len(self.indices[0])