"""
Moment-Guided Diffusion (MGD) - utilities.
"""


import numpy as np
import torch
import matplotlib.pyplot as plt


class TensorDataset(torch.utils.data.Dataset):
    """ We have to create our own class because PyTorch's TensorDataset returns lists... """

    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item]

def add_noise(x, t):
    """Apply Ornstein-Uhlenbeck noising at time ``t``.
 
    Returns ``x_t = e^{-t} x + sqrt(1 - e^{-2t}) z`` with ``z`` standard Gaussian.
 
    Parameters
    ----------
    x : torch.Tensor
        Clean input.
    t : float
        Noise time.
 
    Returns
    -------
    x_t : torch.Tensor
        Noised input.
    z : torch.Tensor
        The Gaussian noise used.
    """

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

def load_results(root, config):
    """Load run outputs saved by :func:`save_results`.
 
    Parameters
    ----------
    root : pathlib.Path
        Results root directory.
    config : str
        Filename for the run.
 
    Returns
    -------
    tuple
        ``(x_t, theta_t, dH_t_bound, t)``.
    """

    x_t = torch.load(root / 'saved_results/samples' / config)
    theta_t = torch.load(root / 'saved_results/lagrange_multipliers' / config)
    dH_t_bound = torch.load(root / 'saved_results/entropy_bounds' / config)
    t = torch.load( root / 'saved_results/sampling_times' / config)
    
    return (x_t,theta_t,dH_t_bound,t)

def normalize(Data):
    """Standardize ``Data`` to zero mean and unit std, cast to float32.
 
    Parameters
    ----------
    Data : torch.Tensor
        Input tensor.
 
    Returns
    -------
    torch.Tensor
        Normalized float32 tensor.
    """

    Data = (Data-Data.mean())/Data.std()
    Data = Data.to(torch.float32)
    return Data

