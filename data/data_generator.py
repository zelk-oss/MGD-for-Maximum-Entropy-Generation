import torch
import numpy as np

from torchvision.transforms import v2
from scipy.special import erfcx, erf, erfinv
from scipy.ndimage import gaussian_filter
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy import io

# -------- scalar signals --------

def scalar_generator(n1, energy):

    def integrand(x):
        return np.exp(-energy(x))
        
    Z, _ = quad(integrand, -np.inf, np.inf)
    print(f"Normalizing constant Z ≈ {Z:.4f}")
        
    # Grille de points pour calculer la CDF
    x_grid = np.linspace(-5, 5, 10000)
    cdf_grid = np.zeros_like(x_grid)
    
    # Calcul de la CDF en chaque point
    for i, x in enumerate(x_grid):
        cdf_grid[i], _ = quad(integrand, -np.inf, x)
    cdf_grid /= Z  # Normalisation
    
    # Interpolation pour la transformation inverse
    cdf_inv = interp1d(cdf_grid, x_grid, kind='linear', fill_value="extrapolate")
    
    # Échantillonnage
    u = np.random.uniform(0, 1, size=n1)
    return torch.from_numpy(cdf_inv(u)[:, None]).float()


# -------- 1d and 2d signals --------


def gaussian_spectrum(patch_size, c=0.05, alpha=2):
    """ Returns a (*L) real covariance spectrum for a stationary circular Gaussian process of spectrum 1/(c + ||omega||^alpha).
    The spectrum is normalized so that pixel marginals have unit variance.
    :param patch_size: desired spatial shape (can be a 1- or 2-tuple)
    :param c: constant in the spectrum definition
    :param alpha: exponent in the spectrum definition
    """
    device = torch.device("cpu")
    omega = torch.stack(torch.meshgrid(*(
        N * torch.fft.fftfreq(N, device=device) for N in patch_size
    ), indexing="ij"), dim=-1)  # (H, W, d)
    omega_norm = torch.sqrt(torch.sum(omega ** 2, dim=-1))  # (H, W)
    spectrum = 1 / (c + omega_norm ** alpha)
    spectrum /= spectrum.mean()
    return spectrum

def generate_gaussian(patch_size, num_patches, c=0.05, alpha=2):
    """ Returns a (N, 1, *L) set of samples from a stationary circular Gaussian distribution with a power-law spectrum.
    :param patch_size: desired spatial shape (can be a 1- or 2-tuple for 1d or 2d process)
    :param num_patches: number N of samples to generate
    :param c: constant in the spectrum definition
    :param alpha: exponent in the spectrum definition
    """
    device = torch.device("cpu")

    # Compute normalized spectrum.
    spectrum = gaussian_spectrum(patch_size, c, alpha)

    # Sample from a stationary Gaussian in Fourier domain.
    shape = (num_patches, 1) + patch_size + (2,)
    noise_fft = torch.view_as_complex(torch.randn(shape, device=device))  # (*, 1, H, W) complex
    # E[|noise_fft|²] = 2 for now, so we rescale to have the right spectrum.
    # Note: we keep this factor of sqrt(2) because we discard the imaginary part after the IFFT.
    noise_fft *= torch.sqrt(spectrum)
    # Go to space domain, discarding the imaginary part.
    noise = torch.real(torch.fft.ifftn(noise_fft, norm="ortho", dim=tuple(range(-len(patch_size), 0))))  # (*, 1, H, W) real

    return noise