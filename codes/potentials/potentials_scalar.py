import torch
import torch.nn as nn

import numpy as np

    ## ----------------------------------------------------- Mother classes -----------------------------------------------------    

class Potential(nn.Module):
  
    def __init__(self):
        super().__init__()
            
    def forward(self,x):
        pass
        
    def grad(self,x):
        pass

    ## ----------------------------------------------------- Potentials definitions -----------------------------------------------------    

class Monomial(Potential):
    def __init__(self, degree):
        super().__init__()
        self.num_coefficients = 1
        self.degree = degree

    def forward(self,x):
        return x**self.degree

    def grad(self, x, v=None):
        output = self.degree*x**(self.degree-1)
        if v==None:
            return output
        else:
            return output*v

class Identity(Potential):
    def __init__(self):
        super().__init__()
        self.num_coefficients = 1

    def forward(self,x):
        return x

    def grad(self, x, v=None):
        output = torch.ones_like(x)
        if v==None:
            return output
        else:
            return output*v

class Abs(Potential):
    def __init__(self, del1=0.001):
        super().__init__()
        self.num_coefficients = 1
        self.del1 = del1

    def forward(self,x):
        return torch.sqrt(self.del1 + x**2)

    def grad(self, x, v=None):
        return x/torch.sqrt(self.del1 + x**2)

class Bimodal(Potential):
    def __init__(self, beta=.8):
        super().__init__()
        self.beta = beta
        self.num_coefficients = 1

    def forward(self,x):
        return self.beta*(x**4 - 5 * x**2 - .5* x)

    def grad(self, x, v=None):
        return self.beta*(4*x**3 - 10*x - .5)