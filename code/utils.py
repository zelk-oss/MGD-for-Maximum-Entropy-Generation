import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
import scipy.fftpack as sfft

from filters.filters_1d import *
from filters.filters_2d import *

from potentials.potentials_scalar import *
from potentials.potentials_1d import *
from potentials.potentials_2d import *



    ## ----------------------------------------------------- Filters related functions -----------------------------------------------------



def return_Filters(M,J,Q=1,L=None,high_freq= 0.49,device='cpu'):
    wav_norm = 'l1'
    wav_type='morlet'
    high_freq = 0.49

    if L==None:
        psi = torch.tensor(init_band_pass(wav_type, M, J, Q, high_freq, wav_norm))[None].to(device).to(torch.float32)
        phi = torch.tensor(init_low_pass(wav_type, M, J, Q, high_freq))[None,None].to(device).to(torch.float32)
            
        filters =  torch.cat([psi,phi],dim=1)
    else:
        filter_set = FiltersSet(M, M, J, L).generate_morlet(precision='single')
        filter_set_psi_real = torch.fft.fft2(torch.fft.ifft2(filter_set['psi']).real)
        filters = torch.cat((filter_set['psi'].reshape(1,L*J,M,M), torch.fft.fft2(filter_set['phi']).reshape(1,1,M,M)), 1).to(device)
    
    return filters



    ## ----------------------------------------------------- Potentials related functions -----------------------------------------------------



def get_scalar_potentials(terms):
    potentials = {}

    for i in range(1,10):
        if 'x'+str(i) in terms:
            potentials['x'+str(i)] = Monomial(i)

    if 'x_abs' in terms:
        potentials['x_abs'] = Abs()

    if 'bimodal' in terms:
        potentials['bimodal'] = Bimodal()
    
    return potentials

def get_1d_potentials(terms, J, filters, Q=1, filters_Q=None, filters_Phi=None, parallel=False):

    if filters_Q is None:
        filters_Q = filters
        Q = 1 
    
    potentials = {}

    if 'Scattering_First_Order' in terms:
        potentials['Scattering_First_Order'] = Scattering_First_Order_1d(filters_Q[:,:-1])
    
    if 'Scattering_Second_Order' in terms:
        potentials['Scattering_Second_Order'] = Scattering_Second_Order_1d(filters_Q)

    if 'Scattering_Third_Order_Real' in terms:
        potentials['Scattering_Third_Order_Real'] = Scattering_Third_Order_Real_1d(J, filters_Q)
    if 'Scattering_Third_Order_Imag' in terms:
        pass
        #potentials['Scattering_Third_Order_Imag'] = Scattering_Third_Order_Imag_1d(J, filters_Q)

    if 'Scattering_Fourth_Order_Real' in terms:
        potentials['Scattering_Fourth_Order_Real'] = Scattering_Fourth_Order_Real_1d(J, Q, filters, filters_Q[:,:-1])
    if 'Scattering_Fourth_Order_Imag' in terms:
        pass
        #potentials['Scattering_Fourth_Order_Imag'] = Scattering_Fourth_Order_Imag_1d(J, Q, filters, filters_Q[:,:-1])

    if parallel:
        for i in range(len(order_terms_potentials)):
            order_terms_potentials[i] = Potential_Parallel(order_terms_potentials[i])#nn.DataParallel(order_terms_potentials[i])

    return potentials

def get_2d_potentials(terms, J, L, filters, parallel=False):
    
    potentials = {}

    if 'Scattering_First_Order' in terms:
        potentials['Scattering_First_Order'] = Scattering_First_Order_2d(filters)
    
    if 'Scattering_Second_Order' in terms:
        potentials['Scattering_Second_Order'] = Scattering_Second_Order_2d(filters)

    if 'Scattering_Third_Order_Real' in terms:
        potentials['Scattering_Third_Order_Real'] = Scattering_Third_Order_Real_2d(J, L, filters)

    if 'Scattering_Fourth_Order_Real' in terms:
        potentials['Scattering_Fourth_Order_Real'] = Scattering_Fourth_Order_Real_2d(J, L, filters)

    if parallel:
        for i in range(len(order_terms_potentials)):
            potentials[i] = Potential_Parallel(potentials[i])

    return potentials

    

    ## ----------------------------------------------------- Entropy related funtions -----------------------------------------------------    



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



    ## ----------------------------------------------------- Display funtions -----------------------------------------------------    



def plot_SD_results(x0, x1, xt, barphi_e, barphi_p, t, sigma, nt, terms):
    print("SDE interpolation complete!")

    # Plotting
    plt.figure(figsize=(10, 5))
    
    # Plot 1: Final comparison (matches figure(1) in MATLAB)
    plt.subplot(1, 2, 1)
    It_final = (1 - t[-2]) * x0 + t[-2] * x1  # Using t[i] from last iteration
    plt.hist(It_final.cpu().numpy(), bins=100, density=True, alpha=0.7, label='Exact (It)', color='blue')
    plt.hist(xt.cpu().numpy(), bins=100, density=True, alpha=0.7, label='SDE Interpolant', color='orange')
    plt.legend()
    plt.title('Final Distributions (SDE)')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Feature evolution (matches figure(3) in MATLAB)

    label_phi = []
    label_SDE = []

    for i in range(len(terms)):
        label_phi.append('Exact $\phi_' + str(i+1) + '$')
        label_SDE.append('SDE $\phi_' + str(i+1) + '$')
    
    plt.subplot(1, 2, 2)
    plt.plot(t.numpy(), barphi_e.numpy(), linewidth=1, label=label_phi)
    plt.plot(t.numpy(), barphi_p.numpy(), linewidth=1, label=label_SDE)
    plt.legend()
    plt.title('Feature Evolution (SDE)')
    plt.xlabel('Time t')
    plt.ylabel('Feature Values')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Additional analysis
    print(f"\nFinal Results:")
    print(f"Final feature error: {torch.norm(barphi_e[-1] - barphi_p[-1]):.6f}")
    print(f"Max feature error during interpolation: {torch.max(torch.norm(barphi_e - barphi_p, dim=1)):.6f}")

    # Show statistics of final distributions
    print(f"\nDistribution Statistics:")
    print(f"Target (x1) - Mean: {torch.mean(x1):.4f}, Std: {torch.std(x1):.4f}")
    print(f"Initial (x0) - Mean: {torch.mean(x0):.4f}, Std: {torch.std(x0):.4f}")
    print(f"Final SDE interpolant - Mean: {torch.mean(xt):.4f}, Std: {torch.std(xt):.4f}")
    
    return torch.norm(barphi_e[-1] - barphi_p[-1])



    ## ----------------------------------------------------- Divers -----------------------------------------------------    



class TensorDataset(torch.utils.data.Dataset):
    """ We have to create our own class because PyTorch's TensorDataset returns lists... """

    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item]

def add_noise(x, t):
    e_minus_t = np.exp(-t)
    std = np.sqrt(1 - e_minus_t ** 2)

    #torch.manual_seed(13)
    z = torch.randn_like(x)
    x_t = e_minus_t * x + std * z

    return x_t, z


def symmetrize_functional(x):
    """
    Functional version (no class needed).
    
    Args:
        x: Tensor of shape (B, C, M, N)
    Returns:
        Symmetrized tensor of shape (B, C, M*2, N*2)
    """
    # Create the 4 quadrants
    top = torch.cat([x, torch.flip(x, dims=[3])], dim=3)
    bottom = torch.cat([torch.flip(x, dims=[2]), torch.flip(x, dims=[2, 3])], dim=3)
    return torch.cat([top, bottom], dim=2)

def save_results(xt, theta_t, dH_t_bound, t, root, config):
    torch.save(xt.cpu(), root / 'saved_results/samples' / config)
    torch.save(theta_t.cpu(), root / 'saved_results/lagrange_multipliers' / config)
    torch.save(dH_t_bound.cpu(), root / 'saved_results/entropy_bounds' / config)
    torch.save(t.cpu(), root / 'saved_results/sampling_times' / config)
    
    return