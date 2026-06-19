import torch
import torch.nn as nn

import numpy as np

## Potentials

class Potential(nn.Module):
  
    def __init__(self):
        super().__init__()
            
    def forward(self,x):
        pass
        
    def grad(self,x):
        pass

class Identity(Potential):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x

    def grad(self, x):
        return torch.ones_like(x)

class Abs(Potential):
    def __init__(self, del1=0.001):
        super().__init__()
        self.del1 = del1

    def forward(self,x):
        return torch.sqrt(self.del1 + x**2)

    def grad(self, x):
        return x/torch.sqrt(self.del1 + x**2)

class Squared(Potential):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return .5*x**2

    def grad(self, x):
        return x

class Third(Potential):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x**3/3

    def grad(self, x):
        return x**2

class Third_modulus(Potential):
    def __init__(self, del1=0.001):
        super().__init__()
        self.del1 = del1

    def forward(self,x):
        return torch.sqrt(self.del1 + x**2)**3/3

    def grad(self, x):
        return x**3/torch.sqrt(self.del1 + x**2)

class Quartic(Potential):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return .25*x**4

    def grad(self, x):
        return x**3

class Fifth(Potential):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x**5/5

    def grad(self, x):
        return x**4

class Sixth(Potential):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x**6/6

    def grad(self, x):
        return x**5

"""class Bimodal(Potential):
    def __init__(self, beta=15, del1=0.001):
        super().__init__()
        self.beta = beta
        self.del1 = del1

    def forward(self,x):
        return self.beta*(.5*x**2-torch.sqrt(self.del1 + x**2)+x/8)

    def grad(self, x):
        return self.beta*(x-x/torch.sqrt(self.del1 + x**2)+1/8)"""

class Bimodal(Potential):
    def __init__(self, beta=.8):
        super().__init__()
        self.beta = beta

    def forward(self,x):
        return self.beta*(x**4 - 5 * x**2 - .5* x)

    def grad(self, x):
        return self.beta*(4*x**3 - 10*x - .5)



    

class Gaussian_mixture(Potential):
    def __init__(self, mu1=torch.Tensor([-10]), sigma1=torch.Tensor([1.0]), mu2=torch.Tensor([10]), sigma2=torch.Tensor([1.0]), w1=torch.Tensor([0.2]), device='cpu'):
        super().__init__()
        
        self.device = device
        
        # Register parameters
        self.mu1 = mu1.to(device)
        self.sigma1 = sigma1.to(device)
        self.mu2 = mu2.to(device)
        self.sigma2 = sigma2.to(device)
        
        # Weight (using logit parameterization for unconstrained optimization)
        self.w1_logit = torch.log(w1 / (1 - w1)).to(device)

    
    def forward(self, x):
        return -self.log_pdf(x)

    def grad(self, x):

        # Compute log PDFs
        log_sqrt_2pi = 0.5 * torch.log(torch.tensor(2 * np.pi, device=self.device))
        
        log_pdf1 = -torch.log(self.sigma1) - log_sqrt_2pi - 0.5 * ((x - self.mu1) / self.sigma1)**2
        log_pdf2 = -torch.log(self.sigma2) - log_sqrt_2pi - 0.5 * ((x - self.mu2) / self.sigma2)**2
        
        # Log of total density
        log_w1 = torch.log(self.w1)
        log_w2 = torch.log(self.w2)
        log_p = torch.logsumexp(
            torch.stack([log_w1 + log_pdf1, log_w2 + log_pdf2], dim=0),
            dim=0
        )
        
        # Compute weights for each component in the gradient
        weight1 = torch.exp(log_w1 + log_pdf1 - log_p)
        weight2 = torch.exp(log_w2 + log_pdf2 - log_p)
        
        # Gradient components
        grad_component1 = -(x - self.mu1) / (self.sigma1**2)
        grad_component2 = -(x - self.mu2) / (self.sigma2**2)
        
        # Total gradient
        grad = -(weight1 * grad_component1 + weight2 * grad_component2)
        
        return grad
    
    @property
    def w1(self):
        """Weight of first component (sigmoid of logit)."""
        return torch.sigmoid(self.w1_logit)
    
    @property
    def w2(self):
        """Weight of second component."""
        return 1 - self.w1

    def gaussian_pdf(self, x, mu, sigma):
        sqrt_2pi = torch.sqrt(torch.tensor(2 * np.pi, device=self.device))
        return (1 / (sigma * sqrt_2pi)) * torch.exp(-0.5 * ((x - mu) / sigma)**2)
    
    def pdf(self, x):
        pdf1 = self.gaussian_pdf(x, self.mu1, self.sigma1)
        pdf2 = self.gaussian_pdf(x, self.mu2, self.sigma2)
        
        return self.w1 * pdf1 + self.w2 * pdf2
    
    def log_pdf(self, x):

        # Log of Gaussian PDF
        log_pdf1 = -torch.log(self.sigma1) - 0.5 * torch.log(
            torch.tensor(2 * np.pi, device=self.device)
        ) - 0.5 * ((x - self.mu1) / self.sigma1)**2
        
        log_pdf2 = -torch.log(self.sigma2) - 0.5 * torch.log(
            torch.tensor(2 * np.pi, device=self.device)
        ) - 0.5 * ((x - self.mu2) / self.sigma2)**2
        
        # Log-sum-exp trick for numerical stability
        log_w1 = torch.log(self.w1)
        log_w2 = torch.log(self.w2)
        
        return torch.logsumexp(
            torch.stack([log_w1 + log_pdf1, log_w2 + log_pdf2], dim=0),
            dim=0
        )

## Phi functions
        
def barphi(x, potentials):
    """Compute the feature vector [mean(abs_d(x)), 0.5*mean(x^2)]"""
    output = torch.zeros(len(potentials), device=x.device)
    
    for i, potential in enumerate(potentials):
        output[i] = torch.mean(potential(x))

    return output

def gradphi(x, potentials):
    output = torch.zeros((x.shape[0], len(potentials)), device=x.device)
    
    for i, potential in enumerate(potentials):
        output[:,i] = potential.grad(x)
    
    return output

def gradmat(x, potentials):
    """Compute the gradient matrix for the smoothed feature vector"""

    gradphi_ = gradphi(x, potentials)

    grad_matrix = gradphi_.T@gradphi_

    return grad_matrix/x.shape[0]