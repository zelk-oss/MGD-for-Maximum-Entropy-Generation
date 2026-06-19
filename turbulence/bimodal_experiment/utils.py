import torch
import torch.nn as nn

import numpy as np

from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import erfcx, erf, erfinv

from Mala import Mala_Sampler

# 1D data generation

class BimodalGaussianMixture(nn.Module):
    """Bimodal Gaussian mixture in PyTorch with energy function."""
    
    def __init__(self, mu1, sigma1, mu2, sigma2, w1=0.5, device='cpu'):
        super().__init__()
        
        self.device = device
        
        # Register parameters
        self.mu1 = nn.Parameter(torch.tensor(mu1, dtype=torch.float32, device=device))
        self.sigma1 = nn.Parameter(torch.tensor(sigma1, dtype=torch.float32, device=device))
        self.mu2 = nn.Parameter(torch.tensor(mu2, dtype=torch.float32, device=device))
        self.sigma2 = nn.Parameter(torch.tensor(sigma2, dtype=torch.float32, device=device))
        
        # Weight (using logit parameterization for unconstrained optimization)
        self.w1_logit = nn.Parameter(torch.tensor(
            np.log(w1 / (1 - w1)), dtype=torch.float32, device=device
        ))
    
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
    
    def energy(self, x):

        return -self.log_pdf(x)

    def energy_gradient(self, x):

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
    
    def sample(self, n_samples=1):
        # Sample component indicators
        components = torch.rand(n_samples, device=self.device) < self.w1
        
        # Sample from each component
        samples = torch.zeros(n_samples, device=self.device)
        
        # Samples from component 1
        n1 = components.sum().item()
        if n1 > 0:
            samples[components] = torch.randn(n1, device=self.device) * self.sigma1 + self.mu1
        
        # Samples from component 2
        n2 = n_samples - n1
        if n2 > 0:
            samples[~components] = torch.randn(n2, device=self.device) * self.sigma2 + self.mu2
        
        return samples

def generate_signed_exponential(n, scale=1.0):
    """Generate exponential random numbers with random signs"""
    # Generate exponential samples
    exp_samples = torch.from_numpy(-np.log(np.random.rand(n, 1))).float() * scale
    # Generate random signs
    signs = torch.sign(torch.randn(n, 1))
    return exp_samples * signs
    
def l1l2_sampler(n1, d, a, b):
    """
    Generate samples from L1-L2 regularized distribution
    """
    # Generate random signs and uniform random variables
    signs = torch.sign(torch.randn(n1, d))
    u = torch.rand(n1, d)
    
    # Compute the inverse transform sampling
    erf_a = erf(a / np.sqrt(2 * b))
    term = u * (1 + erf_a) - erf_a
    
    # Convert to numpy for erfinv, then back to torch
    term_np = term.numpy()
    erfinv_term = erfinv(term_np)
    erfinv_term = torch.from_numpy(erfinv_term).float()
    
    samples = signs * (np.sqrt(2 * b) * erfinv_term + a) / b
    return samples


def integrand(x, beta=6):
    return np.exp(-beta * x**4 + 5 * beta * x**2 +  0.5*beta * x)


def bimodal(n1, beta=2):
    Z, _ = quad(integrand, -np.inf, np.inf, args=(beta,))
    print(f"Constante de normalisation Z ≈ {Z:.4f}")
        
    # Grille de points pour calculer la CDF
    x_grid = np.linspace(-5, 5, 10000)
    cdf_grid = np.zeros_like(x_grid)
    
    # Calcul de la CDF en chaque point
    for i, x in enumerate(x_grid):
        cdf_grid[i], _ = quad(integrand, -np.inf, x, args=(beta,))
    cdf_grid /= Z  # Normalisation
    
    # Interpolation pour la transformation inverse
    cdf_inv = interp1d(cdf_grid, x_grid, kind='linear', fill_value="extrapolate")
    
    # Échantillonnage
    u = np.random.uniform(0, 1, size=n1)
    return cdf_inv(u)

def score_unbalanced(x, a, b, c):
    #(B,) to (B,)
    return 2*a*x+b*torch.sign(x)+c

def energy_unbalanced(x, a, b, c):
    #(B,) to (B,)
    return a*x**2+b*x.abs()+c*x

def gen_unbalanced(n_samples, n_steps = 10000, alpha=1, device='cpu'):
    step_size = 1/alpha
    
    x = torch.randn((n_samples,)).to(device)*10
    
    a = alpha /2
    b = -2 *alpha /2
    c = 0.25 *alpha /2

    energy_unbalanced_ = lambda x_: energy_unbalanced(x_, a, b, c)
    score_unbalanced_ = lambda x_: score_unbalanced(x_, a, b, c)

    return Mala_Sampler(score_unbalanced_, energy_unbalanced_, x, n_steps, step_size)



# Entropy related funtions

def kl_divergence(p, q, n_bins, bins = None, epsilon=1e-5):
   
    #p is reference
    
    if bins is not None:
        p = np.histogram(p, bins, range=None, density=True, weights=None)[0]+epsilon
        q = np.histogram(q, bins, range=None, density=True, weights=None)[0]+epsilon
        d_bins = bins[1:]-bins[:-1]
    else:
        #minus = min(np.min(p),np.min(q))
        #maxus = max(np.max(p),np.max(q))
        #bins = np.linspace(minus,maxus,n_bins)
        #d_bins = (maxus-minus)/n_bins

        bins = histedges_equalN((p+q)/2, n_bins)
        d_bins = bins[1:]-bins[:-1]
        
        p = np.histogram(p, bins, range=None, density=True, weights=None)[0]+epsilon
        q = np.histogram(q, bins, range=None, density=True, weights=None)[0]+epsilon
        
    return np.sum(np.where(p != 0, p * np.log(p / q), 0)*d_bins)

def entropy(p, n_bins, bins = None, epsilon=1e-5):
   
    #p is reference
    
    if bins is not None:
        p = np.histogram(p, bins, range=None, density=True, weights=None)[0]+epsilon
    else:
        bins =  histedges_equalN(p, n_bins)
        p = np.histogram(p, bins, range=None, density=True, weights=None)[0]+epsilon

    d_bins = bins[1:]-bins[:-1]
    
    return np.sum(np.where(p != 0,  -np.log(p)*p, 0)*d_bins)

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))