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

def save_results(
    xt,
    theta_t,
    dH_t_bound,
    t,
    root,
    config,
    Theta_reg,
):
    base = root / 'saved_results'

    torch.save(xt.cpu(), base / 'samples' / config)
    torch.save(theta_t.cpu(), base / 'lagrange_multipliers' / config)
    torch.save(dH_t_bound.cpu(), base / 'entropy_bounds' / config)
    torch.save(t.cpu(), base / 'sampling_times' / config)

    torch.save(
        Theta_reg.cpu(),
        base / 'lagrange_multipliers_regularised' / config
    )

def load_results(root, config):
    base = root / 'saved_results'

    x_t = torch.load(base / 'samples' / config)
    theta_t = torch.load(base / 'lagrange_multipliers' / config)
    dH_t_bound = torch.load(base / 'entropy_bounds' / config)
    t = torch.load(base / 'sampling_times' / config)

    path_theta_reg = base / 'lagrange_multipliers_regularised' / config

    if path_theta_reg.exists():
        Theta_reg = torch.load(path_theta_reg)

    return (x_t, theta_t, dH_t_bound, t, Theta_reg)

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

def split_periodize_reshape(Data, n1):
    """
    Splits Data into non-overlapping subseries of length n1 along the last axis,
    discards the end if not divisible, periodizes each subseries, and reshapes
    the output to [batch * num_subseries, channels, n1].
    """
    num_subseries = Data.size(-1) // n1
    Data_sub = Data[:, :, :num_subseries * n1]
    Data_sub = Data_sub.unfold(-1, n1, n1)

    first = Data_sub[..., 0:1]
    last = Data_sub[..., -1:]
    a = (last - first) / (n1 - 1)
    b = first
    x = torch.arange(n1, device=Data.device, dtype=Data.dtype)
    linear = a * x + b 
    Data_periodized = Data_sub - linear 
    Data_periodized = Data_periodized + first 

    batch, channels, _, _ = Data_periodized.shape
    return Data_periodized.reshape(batch * num_subseries, channels, n1)