    ## ----------------------------------------------------- Entropy related funtions -----------------------------------------------------    
import numpy as np
import torch
import matplotlib.pyplot as plt



def kl_divergence(p, q, n_bins, bins = None, epsilon=1e-5):
   
    """Histogram estimate of the KL divergence ``KL(p || q)`` (``p`` reference).
 
    Both samples are binned into densities (with ``epsilon`` floor) and the discrete
    KL is integrated against the bin widths. If ``bins`` is None, equal-count edges
    on ``(p + q) / 2`` are used (:func:`histedges_equalN`).
 
    Parameters
    ----------
    p, q : array_like
        Samples; ``p`` is the reference distribution.
    n_bins : int
        Number of bins when ``bins`` is None.
    bins : array_like, optional
        Explicit bin edges.
    epsilon : float, optional
        Density floor, by default 1e-5.
 
    Returns
    -------
    float
        Estimated KL divergence.
    """

    
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
   
    """Histogram estimate of the differential entropy of ``p``.
 
    Bins ``p`` into a density (with ``epsilon`` floor) and integrates ``-p log p``
    against the bin widths. Equal-count edges are used when ``bins`` is None.
 
    Parameters
    ----------
    p : array_like
        Samples.
    n_bins : int
        Number of bins when ``bins`` is None.
    bins : array_like, optional
        Explicit bin edges.
    epsilon : float, optional
        Density floor, by default 1e-5.
 
    Returns
    -------
    float
        Estimated differential entropy.
    """

    
    if bins is not None:
        p = np.histogram(p, bins, range=None, density=True, weights=None)[0]+epsilon
    else:
        bins =  histedges_equalN(p, n_bins)
        p = np.histogram(p, bins, range=None, density=True, weights=None)[0]+epsilon

    d_bins = bins[1:]-bins[:-1]
    
    return np.sum(np.where(p != 0,  -np.log(p)*p, 0)*d_bins)

def histedges_equalN(x, nbin):
    """Bin edges placing (approximately) equal sample counts per bin.
 
    Quantile binning: interpolates the sorted samples at evenly spaced ranks.
 
    Parameters
    ----------
    x : array_like
        Samples.
    nbin : int
        Number of bins.
 
    Returns
    -------
    numpy.ndarray
        ``nbin + 1`` bin edges.
    """

    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

def compute_gaussian_entropy(x1, interpolant, t):
    
    """Closed-form entropy of the Gaussian interpolant marginal along ``t``.
 
    Assumes a single channel. The data spectrum is the value variance (1D case,
    ``len(x1.shape) == 2``) or the diagonal of the Fourier covariance (2D). With
    base entropy ``H_p_0 = (log(2 pi) + 1) d / 2`` (``d`` the dimension), returns
    ``H_p_0 + 0.5 * sum_k log(var_t,k)`` where the per-mode variance follows the
    chosen interpolant schedule (Linear / VarPreserv / Sqrt / Cos).
 
    Parameters
    ----------
    x1 : torch.Tensor
        Data samples, shape (B, T) or (B, M, N).
    interpolant : str
        Interpolant schedule.
    t : torch.Tensor
        Times, shape (n_t, 1).
 
    Returns
    -------
    torch.Tensor
        Gaussian entropy at each time, shape (n_t,).
    """

    # assume the number of channels to be 1


    if len(x1.shape)==2:
        spectrum_x1 = torch.var(x1.flatten()).cpu()
        
        d = x1.shape[-1]
        
    else:
        x1_fourier = torch.fft.fft2(x1)
        cov_x1 = (x1_fourier.reshape(x1.shape[0], x1.shape[-2]*x1.shape[-1]).T@x1_fourier.reshape(x1.shape[0], x1.shape[-2]*x1.shape[-1])).cpu()
        spectrum_x1 = torch.diag(cov_x1)/(x1.shape[-2]*x1.shape[-1]*np.sqrt(x1.shape[0]))
    
        d = x1.shape[-2]*x1.shape[-1]
    
    H_p_0 = (np.log(2*np.pi)+1)*d/2

    match interpolant:
        case 'Linear':
            return H_p_0 + torch.log(spectrum_x1.abs()[None]*t[:,None]**2+(1-t[:,None])**2).sum(1)/2
        case 'VarPreserv':
            return H_p_0 + torch.log(spectrum_x1.abs()[None]*t[:,None]+(1-t[:,None])).sum(1)/2
        case 'Sqrt':
            return H_p_0 + torch.log(spectrum_x1.abs()[None]*t[:,None]+(1-np.sqrt(t[:,None]))**2).sum(1)/2
        case 'Cos':
            return H_p_0 + torch.log(spectrum_x1.abs()[None]*np.sin(np.pi * t[:,None] / 2)**2+np.cos(np.pi * t[:,None] / 2)**2).sum(1)/2