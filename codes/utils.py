"""
Moment-Guided Diffusion (MGD) - utilities.
"""

from pathlib import Path
import numpy as np
import torch
from typing import Any, Tuple
import sys

_UTILS_DIR = Path(__file__).resolve().parent
_UTILS_PROJECT_ROOT = _UTILS_DIR.parent

def _extend_syspath(root: Path, project_root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    codes_path = project_root / 'codes'
    if codes_path.is_dir() and str(codes_path) not in sys.path:
        sys.path.insert(0, str(codes_path))
    data_path = project_root / 'data'
    if data_path.is_dir() and str(data_path) not in sys.path:
        sys.path.insert(0, str(data_path))

_extend_syspath(_UTILS_DIR, _UTILS_PROJECT_ROOT)


from codes.check_moments import *     # noqa: E402
from codes.ortho_wavelet.ReadyToUseWavelets import *


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
):
    base = root / 'saved_results'

    torch.save(xt.cpu(), base / 'samples' / f'{config}.pt')
    torch.save(theta_t.cpu(), base / 'lagrange_multipliers' / f'{config}.pt')
    torch.save(dH_t_bound.cpu(), base / 'entropy_bounds' / f'{config}.pt')
    torch.save(t.cpu(), base / 'sampling_times' / f'{config}.pt')

def save_results_theta_reg(
    xt,
    theta_t,
    dH_t_bound,
    t,
    root,
    config,
    Theta_reg=None,  # Set to None by default
):
    base = root / 'saved_results'

    # 1. Save the core variables first to secure them on disk
    torch.save(xt.cpu(), base / 'samples' / f'{config}.pt')
    torch.save(theta_t.cpu(), base / 'lagrange_multipliers' / f'{config}.pt')
    torch.save(dH_t_bound.cpu(), base / 'entropy_bounds' / f'{config}.pt')
    torch.save(t.cpu(), base / 'sampling_times' / f'{config}.pt')

    # 2. Safely process Theta_reg only if it was passed in
    if Theta_reg is not None:
        try:
            torch.save(
                Theta_reg.cpu(),
                base / 'lagrange_multipliers_regularised' / f'{config}.pt'
            )
        except RuntimeError as e:
            # Catch the OOM (or any other PyTorch error) so the run doesn't crash entirely
            print(f"Warning: Failed to save Theta_reg. Error: {e}")

def _load_tensor(path_no_ext: Path):
    """Load a tensor saved under `path_no_ext`'s name, preferring the current
    '<name>.pt' file but falling back to the legacy extensionless file so
    results saved before the .pt convention still load."""
    pt_path = path_no_ext.with_name(path_no_ext.name + '.pt')
    return torch.load(pt_path if pt_path.exists() else path_no_ext)

def load_results(root: Path, exact_config: str) -> Tuple[Any, Any, Any, Any, Any]:
    base = root / 'saved_results'

    x_t = _load_tensor(base / 'samples' / exact_config)
    theta_t = _load_tensor(base / 'lagrange_multipliers' / exact_config)
    dH_t_bound = _load_tensor(base / 'entropy_bounds' / exact_config)
    t = _load_tensor(base / 'sampling_times' / exact_config)

    path_theta_reg = base / 'lagrange_multipliers_regularised' / exact_config
    path_theta_reg_pt = path_theta_reg.with_name(path_theta_reg.name + '.pt')

    if path_theta_reg_pt.exists():
        Theta_reg = torch.load(path_theta_reg_pt)
    elif path_theta_reg.exists():
        Theta_reg = torch.load(path_theta_reg)
    else:
        Theta_reg = None # Always return 5 items to prevent unpacking crashes

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

