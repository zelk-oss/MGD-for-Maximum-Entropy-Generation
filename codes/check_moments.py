"""
Moment-Guided Diffusion (MGD) - plotting and diagnostics utilities.
 
Visual comparisons between the original data and MGD-sampled output. Throughout,
"Original" / ``Data`` / ``M_original`` is the reference data and "Synthesis" /
"Synth" / ``synth`` / ``M_sampled`` is the generated sample.
 
Two families of diagnostics:
  - Wavelet / scattering moments: per-scale and per-orientation moments of the
    wavelet coefficients ``W_j x`` (e.g. L2, L1, second-order covariance terms),
    plotted as original-vs-sampled error bars.
  - Turbulence statistics: structure functions ``S_p(tau)``, cross structure
    functions ``S_{p,q}(tau1, tau2)``, increment PDFs, and Fourier spectra.
 
Common symbols
--------------
J     : number of wavelet scales.
L     : number of orientations / angles per scale.
tau   : increment lag.
M_*, Std_* : mean and std of a moment, taken over samples; sliced per scale as
        ``[j*L:(j+1)*L]``, with the trailing entry the low-frequency term.
 
Most ``*_plot`` functions return None and display matplotlib figures.
"""


import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
import scipy.fftpack as sfft
from filters import init_band_pass
from scipy.ndimage import gaussian_filter1d


class StructureFunctionCache:
    def __init__(self):
        self.cache = {}

    def get_or_compute(self, data, p, taus):
        """
        Retrieves structure functions from cache if computed, 
        otherwise computes and stores them.
        """
        p_arr = np.array(p)
        taus_arr = np.array(taus, dtype=int)
        
        # Create a unique key based on the data memory id, powers, and taus
        key = (id(data), p_arr.tobytes(), taus_arr.tobytes())
        
        if key not in self.cache:
            print(f"Computing structure functions for {len(taus_arr)} lags...")
            sf = np.zeros(shape=(len(p_arr), len(taus_arr)))
            
            # Handle PyTorch tensors gracefully
            data_np = data.cpu().numpy() if hasattr(data, 'cpu') else data
            
            for i, tau in enumerate(taus_arr):
                d_data = data_np[..., tau:] - data_np[..., :-tau]
                for j, power in enumerate(p_arr):
                    sf[j, i] = np.abs(np.mean(np.power(d_data.reshape(-1), power)))
            self.cache[key] = sf
        else:
            print("Loaded structure functions from cache!")
            
        return self.cache[key]

# Instantiate the cache globally so it persists across function calls
sf_cache = StructureFunctionCache()


def spec_plot(Data,synth, save=None):
  """Plot the power spectrum and value histogram, original vs synthesis.
 
  First panel: mean power spectrum (over batch/channel) on log-log axes, keeping
  the first half of frequencies. Second panel: histogram of all sample values on a
  log y-axis.
 
  Parameters
  ----------
  Data, synth : torch.Tensor
      Original and synthetic signals, shape (..., T).
  """


  plt.plot((torch.fft.ifft(Data.cpu()).abs()**2).mean((0,1))[:Data.shape[-1]//2])
  plt.plot((torch.fft.ifft(synth.cpu()).abs()**2).mean((0,1))[:Data.shape[-1]//2])
  plt.yscale('log')
  plt.xscale('log')
  plt.show()

  plt.hist(Data.reshape((-1,)).cpu(),density=True,bins=50,label='Orig')
  plt.hist(synth.reshape((-1,)).cpu(),density=True,bins=50,alpha=0.7,label='Synth')
  plt.legend()
  plt.yscale('log')

  plt.show()

def hist_plot(Data, synth, psi=None, save=None):
    """
    Plot per-band histograms of wavelet coefficients.
    If save is None:
        show one tall figure (one histogram below another).
    If save is provided:
        save the complete figure.
    """
    M = Data.shape[-1]
    J = int(np.log2(M)) - 2
    Q = 3
    if psi is None:
        psi = torch.tensor(
            init_band_pass(
                'morlet',
                M,
                J=J,
                Q=Q,
                high_freq=0.49,
                wav_norm='l1'
            )
        )
    x = torch.fft.ifft(torch.fft.fft(Data.cpu()) * psi)
    x_synth = torch.fft.ifft(torch.fft.fft(synth.cpu()) * psi)

    # One subplot for every (j,q)
    nplots = J * Q
    fig, axes = plt.subplots(
        nplots, 1,
        figsize=(7, 2.8 * nplots),
        squeeze=False
    )
    axes = axes.ravel()
    for j in range(J):
        for q in range(Q):
            ax = axes[j * Q + q]
            x_j = x[:, j * Q + q].flatten().real
            x_j_synth = x_synth[:, j * Q + q].flatten().real
            ax.hist(x_j, bins=100, density=True, alpha = 0.5, color="steelblue", label='Orig')
            ax.hist(x_j_synth, bins=100, density=True,
                    alpha=0.5, color="orange", label='Synth')
            ax.set_yscale('log')
            ax.set_title(f'Wavelet coefficients (j={j}, q={q})')
            if j == 0 and q == 0:
                ax.legend()

    if save is not None:
        fig.suptitle(save["title"])
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(save["filename"], dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        fig.tight_layout()
        plt.show()


def cross_structure_function(data, pq=[(1,1)], max_tau=10):
    """Cross structure function ``S_{p,q}(tau_i, tau_j) = <|du_i|^p |du_j|^q>``.
 
    For each pair of lags ``(tau_i, tau_j)`` forms the increments
    ``du = data[..., tau:] - data[..., :-tau]``, truncates both to the common
    length, and averages ``|du_i|^p |du_j|^q`` over all samples and positions.
 
    Parameters
    ----------
    data : numpy.ndarray
        Signals, shape (..., T).
    pq : list of (int, int)
        Exponent pairs ``(p, q)``.
    max_tau : int
        Lags run over ``1 .. max_tau - 1``.
 
    Returns
    -------
    second_order : numpy.ndarray
        Shape (len(pq), len(taus), len(taus)).
    """

    taus = np.arange(1, max_tau, 1)
    second_order = np.zeros(shape=(len(pq), len(taus),len(taus)))
    for i in range(len(taus)):
        for j in range(len(taus)):
            tau_i = taus[i]
            tau_j = taus[j]
            d_data_i = data[..., tau_i:] - data[..., :-tau_i]
            d_data_j = data[..., tau_j:] - data[..., :-tau_j]
            for k, power in enumerate(pq):
                lenght = min(d_data_i.shape[-1],d_data_j.shape[-1])
                second_order[k, i,j] = (np.abs(d_data_i[...,:lenght])**power[0]*np.abs(d_data_j[...,:lenght])**power[1]).mean()
    return second_order


def cross_plot(Data, synth,
               pq=[(2,1), (2,2), (3,1), (3,2), (3,3)],
               epsilon=1e-8,
               save=None):
    """
    Plot log cross-structure functions and their relative error.
    If save is None:
        show one figure per (p,q) pair (original behaviour).
    If save is provided:
        save one large figure containing all (p,q) panels.
    """

    max_tau = Data.shape[-1] // 2
    second_order = cross_structure_function(
        Data.cpu().numpy(), pq=pq, max_tau=max_tau
    )
    second_order_gen = cross_structure_function(
        synth.cpu().numpy(), pq=pq, max_tau=max_tau
    )
    log_second_order = np.log(second_order) + epsilon
    log_second_order_gen = np.log(second_order_gen) + epsilon
    error = np.abs(second_order - second_order_gen) / (
        second_order + second_order
    )
    vmin = min(log_second_order.min(), log_second_order_gen.min())
    vmax = max(log_second_order.max(), log_second_order_gen.max())

    if save is None:

        # Original behaviour
        for i, (p, q) in enumerate(pq):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            ax1, ax2, ax3 = axes
            ax1.imshow(log_second_order[i], vmin=vmin, vmax=vmax)
            ax1.set_title(f'Data ({p},{q})')
            ax2.imshow(log_second_order_gen[i], vmin=vmin, vmax=vmax)
            ax2.set_title(f'Synth ({p},{q})')
            ax3.imshow(error[i], vmin=0, vmax=1, cmap='Greys')
            ax3.set_title('Relative error')
            for ax in axes:
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(1e-1, max_tau)
                ax.set_ylim(1e-1, max_tau)

            plt.tight_layout()
            plt.show()

    else:

        # One portable figure
        fig, axes = plt.subplots(
            len(pq), 3,
            figsize=(15, 4 * len(pq)),
            squeeze=False
        )

        for i, (p, q) in enumerate(pq):
            ax1, ax2, ax3 = axes[i]
            ax1.imshow(log_second_order[i], vmin=vmin, vmax=vmax)
            ax1.set_title(f'Data (p={p}, q={q})')
            im = ax2.imshow(log_second_order_gen[i], vmin=vmin, vmax=vmax)
            ax2.set_title(f'Synth (p={p}, q={q})')
            ax3.imshow(error[i], vmin=0, vmax=1, cmap='Greys')
            ax3.set_title('Relative error')
            for ax in (ax1, ax2, ax3):
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(1e-1, max_tau)
                ax.set_ylim(1e-1, max_tau)

        fig.colorbar(im, ax=axes[:, :2], shrink=0.8, label='log S')
        fig.suptitle(save["title"])
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(save["filename"], dpi=200, bbox_inches="tight")
        plt.close(fig)


def second_order_structure_function(data, p=np.array([2, 4, 6, 8]), max_tau=10):
    """Structure function ``S_p(tau) = |mean (du)^p|`` for several exponents.
 
    Forms increments ``du = data[..., tau:] - data[..., :-tau]`` at each lag and
    returns the absolute value of the mean of ``du**p`` over the flattened array.
 
    Parameters
    ----------
    data : numpy.ndarray
        Signals, shape (..., T).
    p : numpy.ndarray
        Exponents.
    max_tau : int
        Lags run over ``1 .. max_tau - 1``.
 
    Returns
    -------
    second_order : numpy.ndarray
        Shape (len(p), len(taus)).
    """

    taus = np.arange(1, max_tau, 1)
    return sf_cache.get_or_compute(data, p, taus)

def structure_plot(Data, synth, save=None):
    """
    Plot structure functions S_p(tau) for p in {2,4,6,8}.

    If save is None:
        show the two figures as before.

    If save is provided:
        save both plots into a single figure.
    """

    max_tau = Data.shape[-1] // 2

    second_order = second_order_structure_function(
        Data, p=np.array([2, 4, 6, 8]), max_tau=max_tau
    )
    second_order_gen = second_order_structure_function(
        synth, p=np.array([2, 4, 6, 8]), max_tau=max_tau
    )

    if save is None:

        # ---------- Figure 1 : normalized structure functions ----------
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot()

        ax.plot(second_order[0], 'b--', label='original_2', ms=3)
        ax.plot(second_order[1] / second_order[0] ** 2, 'r--', label='original_4', ms=3)
        ax.plot(second_order[2] / second_order[0] ** 3, 'g--', label='original_6', ms=3)

        ax.plot(second_order_gen[0], 'bo', label='gen_2')
        ax.plot(second_order_gen[1] / second_order_gen[0] ** 2, 'ro', label='gen_4')
        ax.plot(second_order_gen[2] / second_order_gen[0] ** 3, 'go', label='gen_6')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('tau')
        ax.set_ylabel(r'$S_p/S_2^{p/2}$')
        ax.set_title('Normalized structure functions')
        ax.legend()

        plt.tight_layout()
        plt.show()

        # ---------- Figure 2 : raw structure functions ----------
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot()

        ax.plot(second_order[0], 'b--', label='original_2', ms=3)
        ax.plot(second_order[1], 'r--', label='original_4', ms=3)
        ax.plot(second_order[2], 'g--', label='original_6', ms=3)

        ax.plot(second_order_gen[0], 'bo', label='gen_2')
        ax.plot(second_order_gen[1], 'ro', label='gen_4')
        ax.plot(second_order_gen[2], 'go', label='gen_6')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('tau')
        ax.set_ylabel(r'$S_p(\tau)$')
        ax.set_title('Structure functions')
        ax.legend()

        plt.tight_layout()
        plt.show()

    else:

        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(12, 4)
        )

        # ---------- Top panel ----------
        ax1.plot(second_order[0], 'b--', label='original_2', ms=3)
        ax1.plot(second_order[1] / second_order[0] ** 2, 'r--', label='original_4', ms=3)
        ax1.plot(second_order[2] / second_order[0] ** 3, 'g--', label='original_6', ms=3)

        ax1.plot(second_order_gen[0], 'bo', label='gen_2')
        ax1.plot(second_order_gen[1] / second_order_gen[0] ** 2, 'ro', label='gen_4')
        ax1.plot(second_order_gen[2] / second_order_gen[0] ** 3, 'go', label='gen_6')

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('tau')
        ax1.set_ylabel(r'$S_p/S_2^{p/2}$')
        ax1.set_title('Normalized structure functions')
        ax1.legend()

        # ---------- Bottom panel ----------
        ax2.plot(second_order[0], 'b--', label='original_2', ms=3)
        ax2.plot(second_order[1], 'r--', label='original_4', ms=3)
        ax2.plot(second_order[2], 'g--', label='original_6', ms=3)

        ax2.plot(second_order_gen[0], 'bo', label='gen_2')
        ax2.plot(second_order_gen[1], 'ro', label='gen_4')
        ax2.plot(second_order_gen[2], 'go', label='gen_6')

        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('tau')
        ax2.set_ylabel(r'$S_p(\tau)$')
        ax2.set_title('Structure functions')
        ax2.legend()

        fig.suptitle(save["title"])
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(save["filename"], dpi=200, bbox_inches="tight")
        plt.close(fig)


def log_spaced_taus(max_tau, num_points=30, min_tau=1):
    """Unique integer lags, log-spaced over ``[min_tau, max_tau)``.

    Standard practice for structure-function / flatness diagnostics: linear
    tau spacing over-samples large tau (dense near max_tau) and under-samples
    small tau, where most of the scaling behaviour lives. geomspace fixes that.
    """
    return np.unique(np.geomspace(min_tau, max_tau - 1, num_points, dtype=int))


def flatness(data, p_hi=4, p_lo=2, taus=None, num_points=30):
    """Flatness factor F(tau) = S_hi(tau) / S_lo(tau)**(p_hi/p_lo).

    For Gaussian increments and p_hi=4, p_lo=2, F=3 (kurtosis); departure from
    a flat F(tau) reflects intermittency. Uses sf_cache, so calling this after
    structure_plot on the same Data/synth tensor is free (cache hit).

    Parameters
    ----------
    data : torch.Tensor or numpy.ndarray
    taus : array-like of int, optional
        Defaults to log_spaced_taus(data.shape[-1]//2, num_points).

    Returns
    -------
    taus, F : numpy.ndarray, numpy.ndarray
    """
    if taus is None:
        taus = log_spaced_taus(data.shape[-1] // 2, num_points)
    S = sf_cache.get_or_compute(data, p=[p_lo, p_hi], taus=taus)
    F = S[1] / S[0] ** (p_hi / p_lo)
    return taus, F


def scaling_exponent_ratio(data, taus=None, num_points=30):
    """Local scaling-exponent ratio d(log S4)/d(log S2) on log-spaced tau.

    = [d log S4/d tau] / [d log S2/d tau] via the chain rule, i.e. ESS-style
    ratio zeta_4(tau)/zeta_2(tau). For NS solutions this is expected to sit
    near 2 in the inertial range, dip below 2 from intermittency, and rise
    again approaching the dissipation scale (S2's log-derivative -> 0).

    Returns
    -------
    taus, ratio : numpy.ndarray
    """
    if taus is None:
        taus = log_spaced_taus(data.shape[-1] // 2, num_points)
    S = sf_cache.get_or_compute(data, p=[2, 4], taus=taus)
    ratio = logder(S[1], taus) / logder(S[0], taus)
    return taus, ratio


def scaling_exponent_ratio_compare(Data, synth, num_points=30, kill_points=0):
    """Compare d(log S4)/d(log S2) of Data vs synth on shared log-spaced lags.

    Reproduces Buzzicotti et al. 2016 (NJP 18 113047) figure 4: log-lin plot of
    the local slope zeta_4/zeta_2, with a dashed reference at the
    non-intermittent dimensional value 2.
    """
    max_tau = Data.shape[-1] // 2
    taus = log_spaced_taus(max_tau, num_points)

    _, ratio_data = scaling_exponent_ratio(Data, taus=taus)
    _, ratio_synth = scaling_exponent_ratio(synth, taus=taus)

    # Cleanly drop the messy edge points if kill_points > 0
    if kill_points > 0:
        taus = taus[:-kill_points]
        ratio_data = ratio_data[:-kill_points]
        ratio_synth = ratio_synth[:-kill_points]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(taus, ratio_data, 'ko-', ms=4, label='Data')
    ax.plot(taus, ratio_synth, 'ro-', ms=4, label='Synth')
    ax.axhline(2.0, color='grey', ls='--', lw=1, label=r'$\zeta_4/\zeta_2=2$ (dimensional)')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$\zeta_4^L/\zeta_2^L = d\log S_4 / d\log S_2$')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()

    return taus, ratio_data, ratio_synth


# Non periodic structure function. Ne = number of particles = x1.shape[0]
# Nx = domain size = x1.shape[-1]
def SF(x,ells,n,per=False):
    Ne,B,Nx=x.shape
    nl=len(ells)
    S=np.zeros((len(n),nl))
    for i in range(nl):
        ii=int(ells[i])
        if per==False:
            temp1=np.roll(x,-ii,axis=1)[:,:,:-ii]-x[:,:,:-ii]
        else:
            temp1=np.roll(x,-ii,axis=1)-x

        for j in range(len(n)):
            S[j,i]=np.mean(np.abs(temp1)**n[j])

    return S

def logder(y,x):
    return np.gradient(np.log(y),np.log(x),edge_order=2)


def signals_plot(synth):
    """Plot up to 50 individual synthesized signal traces (channel 0), one per figure.
 
    Parameters
    ----------
    synth : torch.Tensor
        Synthetic signals, shape (B, C, T).
    """

    for i in range(min(50,len(synth))):
      plt.plot(synth.cpu()[i,0])
      plt.show()

def Compare_Spectrum(DATA, X_fake, log = False):
    """Plot azimuthally-averaged 2D Fourier power spectra, true vs synthesis.
 
    Computes ``|FFT2(.)|**2`` averaged over the batch, azimuthally averages it via
    :func:`azimuthalAverage`, and overlays the two radial profiles.
 
    Parameters
    ----------
    DATA, X_fake : array_like
        True and synthetic 2D fields, shape (B, M, N).
    log : bool
        If True, use log-log axes.
    """
    plt.plot(azimuthalAverage(sfft.fftshift((np.abs(np.fft.fft2(DATA))**2).mean(axis=0))), label='True')
    plt.plot(azimuthalAverage(sfft.fftshift((np.abs(np.fft.fft2(X_fake))**2).mean(axis=0))), label='Synthesis')
    if log == True:
        plt.xscale('log')
        plt.yscale('log')
    plt.legend()
    plt.xlabel('k')
    plt.title('Fourier Spectrum')
    plt.show()

def azimuthalAverage(image, center=None, Fourier=True):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fractional pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)[-2:]
    '''added modification a and b'''
    a, b = image.shape[-2:]

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])

    r = np.hypot(x - center[0], (y - center[1]) * a / b)
    if Fourier == False:
        r = np.hypot(x - center[0], (y - center[1]))

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

def increment_pdf_plot(Data, synth, taus=(1, 2, 4, 8, 16, 32),
                       n_bins=201, standardize=True, log_y=True,
                       xlim=None, ylim=None, decade_shift=True,
                       smooth=False, smooth_sigma=2.0):
    """
    Plot PDFs of velocity increments delta u(t, tau) = u(t+tau) - u(t)
    for several lags tau, superimposed on one figure.

    Each PDF is first peak-normalized (max = 1 at the center), then
    multiplied by 10**(-k) for the k-th scale, so the peaks are stacked
    one decade apart on the y-axis.

    Parameters
    ----------
    Data, synth : torch.Tensor
        Original and synthetic data, shape (..., T).
    taus : iterable of int
        Lags. Order sets the vertical stacking.
    n_bins : int
        Number of bins used to estimate each PDF.
    standardize : bool
        Divide each increment series by its own std before binning.
    log_y : bool
        Log scale on the y-axis.
    xlim, ylim : tuple or None
        Fixed axis ranges.
    decade_shift : bool
        If True, apply the 10**(-k) vertical shift between scales.
    smooth : bool
        If True, smooth each PDF with a Gaussian filter before plotting.
    smooth_sigma : float
        Gaussian smoothing width, in *bins*. Typical values 1-4.
        Larger -> smoother but more bias on sharp features.
    """
    data_np  = Data.cpu().numpy()
    synth_np = synth.cpu().numpy()

    taus = list(taus)
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0.1, 0.9, len(taus)))

    fig, ax = plt.subplots(figsize=(7, 6.5))

    for k, (tau, color) in enumerate(zip(taus, colors)):
        d_data  = (data_np[...,  tau:] - data_np[...,  :-tau]).reshape(-1)
        d_synth = (synth_np[..., tau:] - synth_np[..., :-tau]).reshape(-1)

        if standardize:
            d_data  = d_data  / (d_data.std()  + 1e-12)
            d_synth = d_synth / (d_synth.std() + 1e-12)

        if xlim is not None:
            lo, hi = xlim
        else:
            lo = min(d_data.min(), d_synth.min())
            hi = max(d_data.max(), d_synth.max())

        bins = np.linspace(lo, hi, n_bins)
        centers = 0.5 * (bins[1:] + bins[:-1])

        pdf_data,  _ = np.histogram(d_data,  bins=bins, density=True)
        pdf_synth, _ = np.histogram(d_synth, bins=bins, density=True)

        # Optional smoothing (done BEFORE peak-normalization so the peak
        # used for normalization is the smoothed one — cleaner alignment)
        if smooth:
            pdf_data  = gaussian_filter1d(pdf_data,  sigma=smooth_sigma)
            pdf_synth = gaussian_filter1d(pdf_synth, sigma=smooth_sigma)

        # Peak-normalize so each curve has max = 1
        pdf_data  = pdf_data  / (pdf_data.max()  + 1e-12)
        pdf_synth = pdf_synth / (pdf_synth.max() + 1e-12)

        # Decade stacking
        shift = 10.0 ** (-k) if decade_shift else 1.0
        pdf_data  = pdf_data  * shift
        pdf_synth = pdf_synth * shift

        # Mask zeros for log axis
        pdf_data  = np.where(pdf_data  > 0, pdf_data,  np.nan)
        pdf_synth = np.where(pdf_synth > 0, pdf_synth, np.nan)

        ax.plot(centers, pdf_data,  '.', color=color, lw=2,
                label=rf'$\tau={tau}$')
        ax.plot(centers, pdf_synth,      color=color, lw=2)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    key_data  = plt.Line2D([0], [0], color='k', ls='-',  lw=2, label='Data')
    key_synth = plt.Line2D([0], [0], color='k', ls='--', lw=2, label='Synth')
    ax.legend(handles + [key_data, key_synth],
              labels  + ['Data', 'Synth'],
              loc='best', frameon=False, ncol=2)

    xlabel = r'$\delta u / \sigma_{\delta u}$' if standardize else r'$\delta u$'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'PDF (peak-normalized, shifted)')
    if log_y:
        ax.set_yscale('log')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    elif decade_shift and log_y:
        ax.set_ylim(10.0 ** (-(len(taus) + 1)), 5.0)

    ax.grid(True, ls=':', alpha=0.5)
    ax.set_title('PDF of increments across scales (decade-shifted)')

    plt.tight_layout()
    plt.show()


def cross_plot_1d_normalized(Data, synth, fixed_tau=1, fix_axis=0, epsilon=1e-8):
    """
    Plot 1D slices of the (2,2) cross structure function, normalized by the
    4th-order structure function at the same scale (the varying tau).

    For each curve:
        ratio(tau) = S_{2,2}(tau1, tau2) / S_4(tau_varying)

    where tau_varying is whichever of (tau1, tau2) is NOT fixed.

    Parameters
    ----------
    Data : torch.Tensor
        Original data.
    synth : torch.Tensor
        Synthetic / generated data (same shape as Data).
    fixed_tau : int
        Index of the lag to hold fixed along the axis chosen by fix_axis.
    fix_axis : {0, 1}
        0 -> fix tau1 = fixed_tau, plot vs tau2 (normalize by S_4(tau2)).
        1 -> fix tau2 = fixed_tau, plot vs tau1 (normalize by S_4(tau1)).
    epsilon : float
        Small constant to avoid division by zero.
    """
    pq = [(2, 2)]
    max_tau = Data.shape[-1] // 2

    # 2D cross structure function S_{2,2}(tau1, tau2)
    s22_data  = cross_structure_function(Data.cpu().numpy(),  pq=pq, max_tau=max_tau)[0]
    s22_synth = cross_structure_function(synth.cpu().numpy(), pq=pq, max_tau=max_tau)[0]

    # 1D 4th-order structure function S_4(tau)
    s4_data  = second_order_structure_function(Data.cpu().numpy(),  p=np.array([4]), max_tau=max_tau)[0]
    s4_synth = second_order_structure_function(synth.cpu().numpy(), p=np.array([4]), max_tau=max_tau)[0]

    n_tau_2d = s22_data.shape[-1]
    n_tau_1d = s4_data.shape[-1]
    n_tau = min(n_tau_2d, n_tau_1d)  # align lengths in case the two helpers differ by 1
    taus = np.arange(1, n_tau + 1)

    if not (0 <= fixed_tau < n_tau):
        raise ValueError(f"fixed_tau={fixed_tau} out of range [0, {n_tau-1}]")

    if fix_axis == 0:
        slice_data  = s22_data[fixed_tau, :n_tau]
        slice_synth = s22_synth[fixed_tau, :n_tau]
        xlabel = r'$\tau_2$'
        fixed_label = rf'$\tau_1 = {fixed_tau}$'
    elif fix_axis == 1:
        slice_data  = s22_data[:n_tau, fixed_tau]
        slice_synth = s22_synth[:n_tau, fixed_tau]
        xlabel = r'$\tau_1$'
        fixed_label = rf'$\tau_2 = {fixed_tau}$'
    else:
        raise ValueError("fix_axis must be 0 or 1")

    # Normalize by S_4 at the *varying* scale
    ratio_data  = slice_data  / (s4_data[:n_tau]  + epsilon)
    ratio_synth = slice_synth / (s4_synth[:n_tau] + epsilon)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(taus, ratio_data,  'o-',  label='Data',  linewidth=2)
    ax.plot(taus, ratio_synth, 's--', label='Synth', linewidth=2)

    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$S_{2,2}(\tau_1,\tau_2) \,/\, S_{4}(\tau)$')
    ax.set_title(f'(p, q) = (2, 2),  {fixed_label}')
    ax.legend()
    ax.grid(True, ls=':', alpha=0.5)

    plt.tight_layout()
    plt.show()



def apply_exp_kernel(x, beta, ksize):
    """Causal exponential smoothing: x_tilde(t) = sum_{k=0}^{ksize-1} beta**k x(t-k),
    weights normalized to sum to 1. beta=None or ksize<=1 -> no-op (returns x)."""
    if beta is None or ksize is None or ksize <= 1:
        return x
    w = beta ** np.arange(ksize)
    w = w / w.sum()
    pad = [(0, 0)] * (x.ndim - 1) + [(ksize - 1, 0)]
    x_pad = np.pad(x, pad)
    x_tilde = np.zeros_like(x, dtype=float)
    for k in range(ksize):
        x_tilde += w[k] * x_pad[..., ksize - 1 - k: ksize - 1 - k + x.shape[-1]]
    return x_tilde

def leverage_correlation(data, p=2.0, scale=1, max_tau=50, beta=None, ksize=1):
    """
    Parity-even leverage correlation:
        L(tau) = < |X(t-tau-2^j) - X(t-tau)| * |X(t-2^j) - X(t)|^p >_t,  2^j = scale
    tau in [-(max_tau-1), max_tau-1].

    Built entirely from |.|, so unlike the finance leverage
    L(tau) = <dX(t-tau)|dX(t)|^p> (odd under x -> -x, hence identically zero
    for parity-symmetric turbulence), this survives parity and instead probes
    time-reversal symmetry directly: L(tau) = L(-tau) iff the (scale-2^j)
    activity process is time-reversible.

    Computed independently per element of the leading (batch/channel) dims,
    then averaged over those dims with error bars.

    Parameters
    ----------
    data : torch.Tensor or numpy.ndarray, shape (..., T), e.g. (B, C, T)
    p : float
        Power on the "current" side, e(t)^p. Use p != 1 (p=2 is the finance
        convention) so L(tau) != L(-tau) is a nontrivial test.
    scale : int
        The scale 2^j (in samples) defining e(s) = |X(s) - X(s-scale)|.
    max_tau : int
        Lags run over -(max_tau-1) .. max_tau-1, capped at T - scale.
    beta : float or None
        Exponential smoothing rate applied to e(t)^p before correlating
        (denoising only; None -> no smoothing).
    ksize : int
        Smoothing kernel length in samples.

    Returns
    -------
    taus : (2*max_tau_eff - 1,) ndarray
    L_mean : ndarray, same shape as taus
        Mean of L(tau) over all leading (batch/channel/...) elements.
    L_err : ndarray, same shape as taus
        Standard error of the mean over those elements.
    """
    x = data.cpu().numpy() if hasattr(data, 'cpu') else np.asarray(data)
    T = x.shape[-1]
    if scale >= T:
        raise ValueError(f"scale={scale} must be < T={T}")

    lead_shape = x.shape[:-1]
    x2 = x.reshape(-1, T)               # (N, T), N = product of leading dims
    N = x2.shape[0]

    # activity signal e(t) = |X(t) - X(t-scale)|, defined for t = scale..T-1;
    # store as e[:, k] with k = t - scale, so k runs 0..T-scale-1
    e = np.abs(x2[:, scale:] - x2[:, :-scale])
    e_p = e ** p
    if beta is not None and ksize is not None and ksize > 1:
        e_p = apply_exp_kernel(e_p, beta, ksize)

    Te = e.shape[-1]
    max_tau = int(min(max_tau, Te))

    taus = np.arange(-(max_tau - 1), max_tau)
    L_samples = np.empty((N, len(taus)))

    for i, tau in enumerate(taus):
        if tau >= 0:
            a = e[:, :Te - tau] if tau > 0 else e          # e(t-tau)
            b = e_p[:, tau:]    if tau > 0 else e_p         # e(t)^p
        else:
            m = -tau
            a = e[:, m:]                                    # e(t-tau) = e(t+m)
            b = e_p[:, :Te - m]                              # e(t)^p
        L_samples[:, i] = (a * b).mean(axis=-1)

    L_samples = L_samples.reshape(*lead_shape, len(taus)).reshape(-1, len(taus))
    L_mean = L_samples.mean(axis=0)
    L_err = L_samples.std(axis=0, ddof=1) / np.sqrt(L_samples.shape[0])

    return taus, L_mean, L_err


def leverage_plot(Data, synth, p=2.0, scale=1, max_tau=50, beta=None, ksize=1,
                   save=None):
    """Plot parity-even leverage correlation L(tau) +/- SEM, Data vs Synth.
    See leverage_correlation for the definition."""
    tau, L_data,  err_data  = leverage_correlation(Data,  p=p, scale=scale, max_tau=max_tau, beta=beta, ksize=ksize)
    _,   L_synth, err_synth = leverage_correlation(synth, p=p, scale=scale, max_tau=max_tau, beta=beta, ksize=ksize)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    ax.plot(tau, L_data, 'k-', label='Data')
    ax.fill_between(tau, L_data - err_data, L_data + err_data, color='k', alpha=0.2)

    ax.plot(tau, L_synth, 'r--', label='Synth')
    ax.fill_between(tau, L_synth - err_synth, L_synth + err_synth, color='r', alpha=0.2)

    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(rf'$\langle |X(t{{-}}\tau{{-}}2^j){{-}}X(t{{-}}\tau)|\,|X(t{{-}}2^j){{-}}X(t)|^{{{p:g}}}\rangle$')
    ax.set_title(rf'Parity-even leverage, scale $2^j={scale}$ (mean $\pm$ SEM over batch)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save is not None:
        fig.suptitle(save["title"])
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(save["filename"], dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

    return tau, L_data, err_data, L_synth, err_synth

def C_p_q(data, p=2.0, tau_star=1, max_tau=50, beta=None, ksize=1,
          normalize=True, epsilon=1e-12):
    """
    Structure-function-normalized, parity-even leverage-type correlation:

        L(tau) = < |X(t-tau-tau*) - X(t-tau)| * |X(t-tau*) - X(t)|^p >_t
                 -------------------------------------------------------
                              S_1(tau*) * S_p(tau*)

    tau in [-(max_tau-1), max_tau-1]. S_n(tau*) is the n-th order structure
    function at fixed scale tau*, computed via second_order_structure_function
    (reuses the module-level sf_cache).

    e(s) = |X(s) - X(s - tau*)| is a parity-even activity signal at fixed
    scale tau* (unchanged under x -> -x, unlike a raw signed increment).
    The numerator is <e(t-tau) e(t)^p>_t; tau>0 correlates current activity
    e(t)^p with PAST activity e(t-tau), tau<0 with FUTURE activity. A
    turbulence signal with no time-reversal asymmetry has L(tau) = L(-tau);
    departures from that are the signature you're after (the analog of the
    finance-leverage asymmetry, but immune to x -> -x parity).

    NOTE on the denominator: second_order_structure_function returns
    S_n(tau) = |mean(du^n)| (abs of the SIGNED mean), which equals
    mean(|du|^n) only for even n. For odd p this is a different, and
    possibly much smaller/noisier, quantity than <e^p> = mean(|du|^p). If
    that matters for your choice of p, either set normalize=False, or
    comment out the block marked below and supply your own denominator
    (e.g. built from mean(|du|^p) directly, as in the SF() helper).

    Computed per element of the leading (batch/channel) dims for the
    numerator, averaged with error bars; the denominator (structure
    functions) is a single population-level value, not resampled per
    batch element.

    Parameters
    ----------
    data : torch.Tensor or numpy.ndarray, shape (..., T), e.g. (B, C, T)
    p : float
        Power on the e(t)^p term. Free choice.
    tau_star : int
        Fixed scale tau* (samples) defining e(s) = |X(s) - X(s-tau*)|.
        Free choice.
    max_tau : int
        Lags run over -(max_tau-1) .. max_tau-1.
    beta, ksize : optional
        Exponential smoothing on e(t)^p before correlating (denoising only).
        None/1 -> no-op.
    normalize : bool
        Toggle for the structure-function normalization (see block below).
    epsilon : float

    Returns
    -------
    taus : (2*max_tau_eff - 1,) ndarray
    L_mean, L_err : ndarray, same shape as taus
        Mean and SEM over batch/channel elements.
    """
    x = data.cpu().numpy() if hasattr(data, 'cpu') else np.asarray(data)
    T = x.shape[-1]
    if tau_star >= T:
        raise ValueError(f"tau_star={tau_star} must be < T={T}")

    lead_shape = x.shape[:-1]
    x2 = x.reshape(-1, T)                                   # (N, T)

    # activity signal e(s) = |X(s) - X(s - tau_star)|, s = tau_star .. T-1
    e = np.abs(x2[:, tau_star:] - x2[:, :-tau_star])         # (N, T - tau_star)
    e_p = e ** p
    if beta is not None and ksize is not None and ksize > 1:
        e_p = apply_exp_kernel(e_p, beta, ksize)

    Te = e.shape[-1]
    max_tau_eff = int(min(max_tau, Te))
    taus = np.arange(-(max_tau_eff - 1), max_tau_eff)

    num_samples = np.empty((x2.shape[0], len(taus)))
    for i, tau in enumerate(taus):
        if tau >= 0:
            a = e[:, :Te - tau] if tau > 0 else e            # e(t-tau)
            b = e_p[:, tau:]    if tau > 0 else e_p           # e(t)^p
        else:
            m = -tau
            a = e[:, m:]                                      # e(t-tau) = e(t+m)
            b = e_p[:, :Te - m]
        num_samples[:, i] = (a * b).mean(axis=-1)

    # ---------------- normalization block: comment out / set normalize=False to disable ----------------
    if normalize:
        S = second_order_structure_function(x, p=np.array([1.0, p]), max_tau=tau_star + 1)
        denom = np.abs(S[0, -1]) * np.abs(S[1, -1]) + epsilon   # S_1(tau*) * S_p(tau*)
        num_samples = num_samples / denom
    # -------------------------------------------------------------------------------------------------

    num_samples = num_samples.reshape(*lead_shape, len(taus)).reshape(-1, len(taus))
    L_mean = num_samples.mean(axis=0)
    L_err = num_samples.std(axis=0, ddof=1) / np.sqrt(num_samples.shape[0])

    return taus, L_mean, L_err


def C_pq_plot(Data, synth, p=2.0, tau_star=1, max_tau=50, beta=None, ksize=1,
              normalize=True, save=None):
    """Plot L(tau) +/- SEM, Data vs Synth. See C_p_q for the definition."""
    tau, L_data,  err_data  = C_p_q(Data,  p=p, tau_star=tau_star, max_tau=max_tau,
                                     beta=beta, ksize=ksize, normalize=normalize)
    _,   L_synth, err_synth = C_p_q(synth, p=p, tau_star=tau_star, max_tau=max_tau,
                                     beta=beta, ksize=ksize, normalize=normalize)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    ax.plot(tau, L_data, 'k-', label='Data')
    ax.fill_between(tau, L_data - err_data, L_data + err_data, color='k', alpha=0.2)

    ax.plot(tau, L_synth, 'r--', label='Synth')
    ax.fill_between(tau, L_synth - err_synth, L_synth + err_synth, color='r', alpha=0.2)

    ax.set_xlabel(r'$\tau$')
    ylabel = (rf'$L(\tau)$, normalized by $S_1(\tau^\star) S_{{{p:g}}}(\tau^\star)$'
              if normalize else rf'$L(\tau)$ (unnormalized)')
    ax.set_ylabel(ylabel)
    ax.set_title(rf'$\tau^\star={tau_star}$, $p={p:g}$ (mean $\pm$ SEM over batch)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save is not None:
        fig.suptitle(save["title"])
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(save["filename"], dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

    return tau, L_data, err_data, L_synth, err_synth



# time irreversibility measures 
"""
Energy-increment statistics: PDFs, tail-folding, skewness(tau).
Assumptions (flagged):
- v: torch tensor, shape (B, C, T). Energy = 0.5*sum_C v^2 (per-particle KE).
- W(tau) = E(t+tau) - E(t), tau > 0 only for PDFs (folding needs a well-defined
  right tail).
- mode='power' (default) -> delta_v * v_present, summed over C. Matches your
  original snippet; use this unless you specifically want the exact KE increment.
  mode='exact' -> true energy increment 0.5*(v_future^2 - v_present^2).sum(C).
- PDFs normalized by std(W(tau)).
"""
import torch
import numpy as np
import matplotlib.pyplot as plt


def _increment(v, tau, mode="power"):
    # v: (B, C, T) -> returns W(tau): (B, T-|tau|). tau can be + or -.
    if tau == 0:
        raise ValueError("tau must be nonzero")
    if tau > 0:
        v_future, v_present = v[:, :, tau:], v[:, :, :-tau]
    else:
        v_future, v_present = v[:, :, :tau], v[:, :, -tau:]

    if mode == "exact":
        E_future = 0.5 * (v_future ** 2).sum(dim=1)
        E_present = 0.5 * (v_present ** 2).sum(dim=1)
        return E_future - E_present
    elif mode == "power":
        delta_v = v_future - v_present
        return (delta_v * v_present).sum(dim=1)
    else:
        raise ValueError("mode must be 'exact' or 'power'")


def logspaced_taus(T, start_tau=1, num_points=45, two_sided=False):
    """Same construction as your snippet: log10-spaced floats -> unique ints.
    end_tau adapts to the signal length: min(T-1, 1024)."""
    end_tau = min(T - 1, 1024)
    taus_float = torch.logspace(
        start=torch.log10(torch.tensor(start_tau, dtype=torch.float32)),
        end=torch.log10(torch.tensor(end_tau, dtype=torch.float32)),
        steps=num_points,
        base=10.0,
    )
    pos_taus = torch.unique(taus_float.long()).tolist()
    if two_sided:
        return [-t for t in reversed(pos_taus)] + pos_taus
    return pos_taus


def pdf_energy_increments(v, taus, mode="power", nbins=100, bin_range=(-20, 20)):
    """taus should be POSITIVE only. Returns {tau: (bin_centers, pdf)},
    W(tau) normalized by its own std."""
    assert all(t > 0 for t in taus), "pass positive taus only"
    out = {}
    for tau in taus:
        W = _increment(v, tau, mode=mode).flatten().cpu().numpy()
        Wn = W / W.std()
        counts, edges = np.histogram(Wn, bins=nbins, range=bin_range, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        out[tau] = (centers, counts)
    return out


def plot_pdf_panel(pdf_dict, ax=None, title=None, cmap="tab10"):
    """One semilog PDF curve per tau. Keep num taus small (<=6) or this gets messy."""
    ax = ax or plt.gca()
    taus = sorted(pdf_dict.keys())
    colors = plt.colormaps.get_cmap(cmap)
    for i, tau in enumerate(taus):
        c, p = pdf_dict[tau]
        mask = p > 0
        ax.semilogy(c[mask], p[mask], color=colors(i % 10), label=f"τ={tau}")
    ax.set_xlabel(r"$W(\tau)/\sigma_{W(\tau)}$")
    ax.set_ylabel("PDF")
    ax.legend(fontsize=7)
    if title:
        ax.set_title(title)
    return ax


def _right_and_folded_left(c, p):
    right, left = c >= 0, c <= 0
    mr, ml = p[right] > 0, p[left] > 0
    x_r, y_r = c[right][mr], p[right][mr]
    x_l = (-c[left][::-1])[ml[::-1]]
    y_l = (p[left][::-1])[ml[::-1]]
    return x_r, y_r, x_l, y_l


def plot_pdf_folded_grid(pdf_data, pdf_synth, ncols=3, figsize_per=(3.0, 2.6),
                          title=None, color_data="#1f77b4", color_synth="#e0592a"):
    """One subplot PER tau, data and synth superimposed: solid = right tail P(x),
    dashed = folded left tail P(-x). One color per dataset, one shared legend."""
    taus = sorted(set(pdf_data.keys()) & set(pdf_synth.keys()))
    n = len(taus)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols,
                                                      figsize_per[1] * nrows),
                              squeeze=False)
    for i, tau in enumerate(taus):
        ax = axes[i // ncols, i % ncols]
        xr, yr, xl, yl = _right_and_folded_left(*pdf_data[tau])
        ax.semilogy(xr, yr, color=color_data, lw=1.6)
        ax.semilogy(xl, yl, color=color_data, lw=1.3, linestyle="--")
        xr, yr, xl, yl = _right_and_folded_left(*pdf_synth[tau])
        ax.semilogy(xr, yr, color=color_synth, lw=1.6)
        ax.semilogy(xl, yl, color=color_synth, lw=1.3, linestyle="--")
        ax.set_title(f"τ={tau}", fontsize=9)
        ax.grid(alpha=0.25)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=color_data, lw=1.6, label="data  P(x)"),
        Line2D([0], [0], color=color_data, lw=1.3, ls="--", label="data  P(-x)"),
        Line2D([0], [0], color=color_synth, lw=1.6, label="synth P(x)"),
        Line2D([0], [0], color=color_synth, lw=1.3, ls="--", label="synth P(-x)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    if title:
        fig.suptitle(title, y=1.08)
    fig.tight_layout()
    return fig, axes


def skewness_vs_tau(v, taus, mode="power"):
    skews = np.empty(len(taus))
    for i, tau in enumerate(taus):
        W = _increment(v, tau, mode=mode).flatten()
        Wc = W - W.mean()
        skews[i] = (Wc ** 3).mean() / (Wc ** 2).mean() ** 1.5
    return np.array(taus), skews


def plot_skewness_comparison(v_data, v_synth, taus=None, mode="power", ax=None,
                              color_data="#1f77b4", color_synth="#e0592a"):
    """taus defaults to positive-only log-spaced sampling (start_tau=1,
    end_tau=min(T-1,1024), num_points=45), plotted on a log-x axis."""
    ax = ax or plt.gca()
    if taus is None:
        T = v_data.shape[-1]
        taus = logspaced_taus(T, num_points=45, two_sided=False)
    t_d, s_d = skewness_vs_tau(v_data, taus, mode=mode)
    t_s, s_s = skewness_vs_tau(v_synth, taus, mode=mode)
    ax.plot(t_d[:-1], s_d[:-1], label="data", color=color_data, marker="o", ms=3)
    ax.plot(t_s[:-1], s_s[:-1], label="synth", color=color_synth, marker="x", ms=3, linestyle="--")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("Skewness")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Example usage:
#
# T = x1.shape[-1]
# taus = logspaced_taus(T, num_points=6)              # positive only, for PDFs
# pdf_data  = pdf_energy_increments(x1, taus, mode="power")
# pdf_synth = pdf_energy_increments(result["xt"], taus, mode="power")
#
# fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# plot_pdf_panel(pdf_data, ax=axes[0], title="data")
# plot_pdf_panel(pdf_synth, ax=axes[1], title="synth")
#
# plot_pdf_folded_grid(pdf_data, pdf_synth, title="right P(x) vs folded-left P(-x)")
#
# fig2, ax2 = plt.subplots()
# plot_skewness_comparison(x1, result["xt"], ax=ax2)   # positive-only log-x taus by default
# ---------------------------------------------------------------------------



# C_pq structure 
def _abs_moment(x, tau, n):
    """S_n(tau) = mean(|X(t+tau)-X(t)|^n), single-lag marginal moment."""
    d = np.abs(x[..., tau:] - x[..., :-tau])
    return np.mean(np.power(d.reshape(-1), n))


def C_pq_structure(x, p, q, tau_star, max_tau=None, epsilon=1e-8):
    """C_{p,q}(tau, tau*) = E[|du_tau*|^p |du_tau|^q] / (S_p(tau*) S_q(tau)).

    C_{p,p} = C_2p in the paper's notation (e.g. p=q=2 -> "C_4").
    x : np.ndarray, shape (..., T)
    """
    T = x.shape[-1]
    if max_tau is None:
        max_tau = T // 2
    taus = np.arange(1, max_tau)
    Sp_star = _abs_moment(x, tau_star, p)

    ratio = np.zeros(len(taus))
    for i, t in enumerate(taus):
        L = min(x.shape[-1] - tau_star, x.shape[-1] - t)
        du_star = np.abs(x[..., tau_star:tau_star + L] - x[..., :L])
        du_t    = np.abs(x[..., t:t + L]                - x[..., :L])
        num = np.mean((du_star ** p * du_t ** q).reshape(-1))
        Sq_t = _abs_moment(x, t, q)
        ratio[i] = num / (Sp_star * Sq_t + epsilon)
    return taus, ratio


def C_pq_structure_plot(Data, synth, p, q, tau_star, max_tau=None, save=None):
    """Data vs synth comparison of C_pq_structure."""
    data_np  = Data.cpu().numpy()
    synth_np = synth.cpu().numpy()
    taus, r_o = C_pq_structure(data_np,  p, q, tau_star, max_tau)
    _,    r_s = C_pq_structure(synth_np, p, q, tau_star, max_tau)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(taus, r_o, 'o-', color=C_ORIG,  lw=3, ms=5, label='Original')
    ax.plot(taus, r_s, 's-', color=C_SYNTH, lw=3, ms=5, label='Synthesis')
    ax.axvline(tau_star, color='grey', ls=':', lw=1.5)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(rf'$C_{{{p},{q}}}(\tau,\tau^\star={tau_star})$')
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(frameon=False)

    if save is not None:
        fig.suptitle(save["title"])
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(save["filename"], dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()
    return taus, r_o, r_s


# convenience wrappers, following C_{p,p} = C_2p naming
def C4_plot(Data, synth, tau_star, max_tau=None, save=None):
    return C_pq_structure_plot(Data, synth, 2, 2, tau_star, max_tau, save)

def C6_plot(Data, synth, tau_star, max_tau=None, save=None):
    return C_pq_structure_plot(Data, synth, 3, 3, tau_star, max_tau, save)

def C42_plot(Data, synth, tau_star=20, max_tau=None, save=None):
    return C_pq_structure_plot(Data, synth, 4, 2, tau_star, max_tau, save)










import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter1d


C_ORIG  = 'tab:blue'   # original
C_SYNTH = 'tab:red'    # synthesis


def _shade(color, frac):
    """Lighten `color` slightly toward white for small frac; never dark.
    frac in [0,1]: 0 -> a little lighter, 1 -> base color."""
    r, g, b = to_rgb(color)
    t = 0.30 * (1.0 - frac)          # gentle: max 0.30 toward white
    return (r + (1 - r) * t,
            g + (1 - g) * t,
            b + (1 - b) * t)


def visual_comparison(Data, synth,
                      pdf_taus=(1, 2, 4, 8, 16, 32),
                      sf_orders=(4, 6, 8),
                      c4_fixed_tau=10,
                      n_bins=201, smooth=True, smooth_sigma=1.5,
                      pdf_xlim=(-8, 8), a_ylim=None, epsilon=1e-8):
    """
    Statistical comparison: original (blue) vs synthesis (red).
    Layout: 2x2 block on the left (a, b, d, e), tall panel (c) on the right.
    """
    data_np  = Data.cpu().numpy()
    synth_np = synth.cpu().numpy()
    max_tau  = Data.shape[-1] // 2

    plt.rcParams.update({
        'font.size': 20,
        'font.weight': 'normal',
        'axes.labelweight': 'normal',
        'axes.titleweight': 'normal',
        'axes.linewidth': 1.4,
    })
    LW    = 5.0
    LW_A  = 4.0
    LABEL_FS = 22

    taus_arr = np.arange(1, max_tau)

    def abs_increments_pow_mean(x, tau, p):
        d = np.abs(x[..., tau:] - x[..., :-tau])
        return np.mean(np.power(d.reshape(-1), p))

    fig = plt.figure(figsize=(20, 12))
    gs  = fig.add_gridspec(2, 3)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axC = fig.add_subplot(gs[:, 2])

    # ----- (a) marginal distribution -----
    s_orig  = data_np.reshape(-1)
    s_synth = synth_np.reshape(-1)
    lo = min(s_orig.min(), s_synth.min())
    hi = max(s_orig.max(), s_synth.max())
    bins = np.linspace(lo, hi, n_bins)
    ctr  = 0.5 * (bins[1:] + bins[:-1])
    p_o, _ = np.histogram(s_orig,  bins=bins, density=True)
    p_s, _ = np.histogram(s_synth, bins=bins, density=True)
    if smooth:
        p_o = gaussian_filter1d(p_o, smooth_sigma)
        p_s = gaussian_filter1d(p_s, smooth_sigma)
    p_o = np.where(p_o > 0, p_o, np.nan)
    p_s = np.where(p_s > 0, p_s, np.nan)
    axA.plot(ctr, p_o, color=C_ORIG,  lw=LW_A, label='Original')
    axA.plot(ctr, p_s, color=C_SYNTH, lw=LW_A, label='Synthesis')
    axA.set_yscale('log')
    axA.set_ylabel('PDF')
    if a_ylim is not None:
        axA.set_ylim(a_ylim)
    axA.legend(frameon=False, fontsize=16)
    axA.grid(True, ls=':', alpha=0.5)

    # ----- (b) power spectrum -----
    def psd(x):
        x2 = x.reshape(-1, x.shape[-1])
        f  = np.fft.rfft(x2, axis=-1)
        return np.mean(np.abs(f) ** 2, axis=0)
    P_o = psd(data_np)
    P_s = psd(synth_np)
    omega = np.fft.rfftfreq(data_np.shape[-1])
    axB.plot(omega[1:], P_o[1:], color=C_ORIG,  lw=LW)
    axB.plot(omega[1:], P_s[1:], color=C_SYNTH, lw=LW)
    axB.set_xscale('log')
    axB.set_yscale('log')
    #axB.set_xlabel(r'$\omega$')
    axB.set_ylabel('PSD')
    axB.grid(True, which='both', ls=':', alpha=0.5)

    # ----- (d) structure functions SF_k -----
    sf_colors = {4: 'tab:red', 6: 'tab:green', 8: 'tab:blue'}
    for p in sf_orders:
        SF_o = np.array([abs_increments_pow_mean(data_np,  t, p) for t in taus_arr])
        SF_s = np.array([abs_increments_pow_mean(synth_np, t, p) for t in taus_arr])
        col = sf_colors.get(p, 'k')
        axD.plot(taus_arr, SF_o + epsilon, '--', color=col, lw=LW)
        axD.plot(taus_arr, SF_s + epsilon, '-',  color=col, lw=LW,
                 label=rf'$\mathrm{{SF}}_{{{p}}}$')
    axD.set_xscale('log')
    axD.set_yscale('log')
    axD.set_xlabel(r'$\tau$')
    axD.set_ylabel(r'$\mathrm{SF}_k(\tau)$')
    axD.legend(frameon=False, fontsize=16)
    axD.grid(True, which='both', ls=':', alpha=0.5)

    # ----- (e) C_4 coefficient -----
    if not (1 <= c4_fixed_tau < max_tau):
        raise ValueError(f"c4_fixed_tau={c4_fixed_tau} must be in [1, {max_tau-1}]")

    _, r_o = C_pq_structure(data_np,  2, 2, c4_fixed_tau, max_tau=max_tau, epsilon=epsilon)
    _, r_s = C_pq_structure(synth_np, 2, 2, c4_fixed_tau, max_tau=max_tau, epsilon=epsilon)
    axE.plot(taus_arr, r_o, 'o-', color=C_ORIG,  lw=LW, ms=5)
    axE.plot(taus_arr, r_s, 's-', color=C_SYNTH, lw=LW, ms=5)
    axE.set_xscale('log')
    axE.set_xlabel(r'$\tau$')
    axE.set_ylabel(r'$C_4(\tau,\tau^\star)$')
    axE.grid(True, ls=':', alpha=0.5)

    # ----- (c) increment PDFs (waterfall), gentle gradient -----
    taus = list(pdf_taus)
    n = len(taus)
    for k, tau in enumerate(taus):
        frac = k / max(n - 1, 1)
        col_o = _shade(C_ORIG,  frac)
        col_s = _shade(C_SYNTH, frac)

        d_o = (data_np[...,  tau:] - data_np[...,  :-tau]).reshape(-1)
        d_s = (synth_np[..., tau:] - synth_np[..., :-tau]).reshape(-1)
        d_o = d_o / (d_o.std() + 1e-12)
        d_s = d_s / (d_s.std() + 1e-12)
        b   = np.linspace(pdf_xlim[0], pdf_xlim[1], n_bins)
        c   = 0.5 * (b[1:] + b[:-1])
        h_o, _ = np.histogram(d_o, bins=b, density=True)
        h_s, _ = np.histogram(d_s, bins=b, density=True)
        if smooth:
            h_o = gaussian_filter1d(h_o, smooth_sigma)
            h_s = gaussian_filter1d(h_s, smooth_sigma)
        h_o = h_o / (h_o.max() + 1e-12) * 10.0 ** (-k)
        h_s = h_s / (h_s.max() + 1e-12) * 10.0 ** (-k)
        peak = np.nanmax(h_o)
        h_o = np.where(h_o > 0, h_o, np.nan)
        h_s = np.where(h_s > 0, h_s, np.nan)
        axC.plot(c, h_o, color=col_o, lw=LW)
        axC.plot(c, h_s, color=col_s, lw=LW)
        axC.text(0.0, peak * 1.9, rf'$\tau={tau}$',
                 color='black', fontsize=16,
                 ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='white', edgecolor='black', linewidth=1.5))
    axC.set_yscale('log')
    axC.set_xlim(pdf_xlim)
    axC.set_ylim(10.0 ** (-(n + 1)), 8.0)
    axC.set_ylabel('PDF (shifted)')
    axC.grid(True, ls=':', alpha=0.5)

    # ----- finalize layout FIRST, then place panel letters -----
    plt.tight_layout()
    fig.canvas.draw()

    def letter(ax, text, dy):
        pos = ax.get_position()
        x_fig = 0.5 * (pos.x0 + pos.x1)
        fig.text(x_fig, pos.y0 - dy, text, ha='center', va='top',
                 fontsize=LABEL_FS)

    for ax, txt, dy in [(axA, '(a)', 0.03), (axB, '(b)', 0.03),
                        (axD, '(d)', 0.075), (axE, '(e)', 0.075),
                        (axC, '(c)', 0.075)]:
        letter(ax, txt, dy)

    #plt.savefig('visual_comparison.pdf', bbox_inches='tight')

    plt.show()
    ## ----------------------------------------------------- Display funtions -----------------------------------------------------    

def plot_SD_results(x0, x1, xt, barphi_e, barphi_p, t, sigma, nt, terms):
    """Summarize an SDE run: final distributions and moment evolution.
 
    Left panel overlays the histogram of the exact interpolant at the last time and
    the SDE sample ``xt``; right panel plots the interpolant and SDE moment paths
    ``barphi_e`` / ``barphi_p`` vs time. Prints final and worst-case moment errors
    and basic distribution statistics.
 
    Parameters
    ----------
    x0, x1, xt : torch.Tensor
        Noise endpoint, data, and final SDE sample.
    barphi_e, barphi_p : torch.Tensor
        Interpolant and SDE moment paths, shape (n_t, r).
    t : torch.Tensor
        Time grid.
    sigma, nt : float, int
        Noise amplitude and step count (display only).
    terms : iterable
        Moment names (used to label the legend).
 
    Returns
    -------
    torch.Tensor
        Final moment error ``||barphi_e[-1] - barphi_p[-1]||``.
    """

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


def plot_entropy_bound_evolution(dH_t_bound, H_t_bound, H_t_gaussian, t):
    plt.figure(figsize=(12,5))

    plt.subplot(121)
    plt.plot(t[1:-1].cpu(), dH_t_bound[1:-1].cpu(), marker='.')
    plt.xlabel('t')
    plt.title(''r'$\theta_t^T \frac{d}{dt} m_t$')
    
    plt.subplot(122)
    plt.plot(t[1:-1].cpu(), H_t_bound[1:-1], label=''r'$H(p_0) + \int \theta_t^T \frac{d}{dt} m_t dt$')
    if H_t_gaussian is not None:
        plt.plot(t[1:-1], H_t_gaussian[1:-1], label=''r'$H(p_t^\text{gaussian})$')
    plt.xlabel('t')
    plt.legend(loc='best')
    
    plt.show()

def plot_moment_matching(barphi_e, barphi_p, t, threshold, save=None):
    """
    Plot the relative moment-matching error between interpolant and walkers.

    If save is None:
        show the two figures as before.

    If save is provided:
        save one figure with two side-by-side panels.
    """

    # Move everything to CPU once
    barphi_e = barphi_e.cpu()
    barphi_p = barphi_p.cpu()
    t = t.cpu()

    keep_mask = barphi_e[-1] > threshold

    if not keep_mask.any():
        print(f"Warning: No moments exceeded threshold {threshold}.")

        error_last = (
            2 * (barphi_e - barphi_p).abs()
            / (barphi_e.abs() + barphi_p.abs())
        )[-1]

        if save is None:
            plt.figure(figsize=(6, 4))
            plt.hist(error_last, bins=100)
            plt.yscale("log")
            plt.title("Distribution of moment matching error (all moments)")
            plt.tight_layout()
            plt.show()
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(error_last, bins=100)
            ax.set_yscale("log")
            ax.set_title("Distribution of moment matching error (all moments)")
            fig.suptitle(["title"])
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(save["filename"], dpi=200, bbox_inches="tight")
            plt.close(fig)

        return

    # Keep only significant moments
    barphi_e = barphi_e[:, keep_mask]
    barphi_p = barphi_p[:, keep_mask]

    rel_error = (
        2 * (barphi_e - barphi_p).abs()
        / (barphi_e.abs() + barphi_p.abs())
    )

    min_len = min(t.shape[0], rel_error.shape[0])
    t_sliced = t[:min_len][2:-1]
    error_mean = rel_error.mean(dim=1)[:min_len][2:-1]
    error_last = rel_error[-1]

    if save is None:

        # -------- Time evolution --------
        plt.figure(figsize=(6, 4))
        plt.plot(t_sliced, error_mean, marker='.')
        plt.xlabel("t")
        plt.ylabel("Mean relative error")
        plt.yscale("log")
        plt.title("Relative moment matching error")
        plt.tight_layout()
        plt.show()

        # -------- Final histogram --------
        plt.figure(figsize=(6, 4))
        plt.hist(error_last, bins=100)
        plt.xlabel("Relative error")
        plt.ylabel("Count")
        plt.yscale("log")
        plt.title("Distribution of moment matching error")
        plt.tight_layout()
        plt.show()

    else:

        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(12, 4)
        )

        # Left panel
        ax1.plot(t_sliced, error_mean, marker='.')
        ax1.set_xlabel("t")
        ax1.set_ylabel("Mean relative error")
        ax1.set_yscale("log")
        ax1.set_title("Time evolution")

        # Right panel
        ax2.hist(error_last, bins=100)
        ax2.set_xlabel("Relative error")
        ax2.set_ylabel("Count")
        ax2.set_yscale("log")
        ax2.set_title("Final distribution")

        fig.suptitle(save["title"])
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(save["filename"], dpi=200, bbox_inches="tight")
        plt.close(fig)



def plot_image_row(Data, N):
    """Show the first ``N`` images (channel 0) in a single row.
 
    Parameters
    ----------
    Data : torch.Tensor
        Images, shape (B, C, M, N_pix).
    N : int
        Number of images.
    """

    random_indexes = np.arange(N)

    fig, axs = plt.subplots(1, N, figsize=(20, 8))
    for i in range(N):
        idx = random_indexes[i]
        axs[i].get_xaxis().set_visible(False)
        axs[i].imshow(Data[idx,0].cpu())
    plt.show()

def plot_time_series_row(Data, N):
    """Show the first ``N`` time series (channel 0) in a single row.
 
    Parameters
    ----------
    Data : torch.Tensor
        Signals, shape (B, C, T).
    N : int
        Number of series.
    """

    random_indexes = np.arange(N)

    fig, axs = plt.subplots(1, N, figsize=(4*N, 4))
    for i in range(N):
        idx = random_indexes[i]
        axs[i].get_xaxis().set_visible(False)
        axs[i].plot(Data[idx,0].cpu())
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def Compare_time_series_row(Data, Synth, N, save=None):
    """Compare the first N time series: data (top) vs synthesis (bottom)."""

    if Data.shape[0] == 0 or Synth.shape[0] == 0:
        # A caller sliced past the end of the batch (e.g. a fixed-size loop
        # over more groups than there are samples) — nothing to plot.
        print(f"Compare_time_series_row: empty batch (Data={tuple(Data.shape)}, "
              f"Synth={tuple(Synth.shape)}) — skipping.")
        return

    # Increased figure height slightly to accommodate larger labels without overlapping
    fig, axs = plt.subplots(
        2,
        N,
        figsize=(3.5 * N, 5.5),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )

    # Handle N=1 edge case where axs is 1D array instead of 2D
    if N == 1:
        axs = axs[:, np.newaxis]

    x = np.arange(Data.shape[-1])

    for row, (dataset, color, ylabel) in enumerate(
        zip((Data, Synth), ("tab:blue", "navy"), ("Data", "MGD synthesis"))
    ):

        # Row / Axis Title Font Size (16pt)
        axs[row, 0].set_ylabel(
            ylabel, fontsize=16, fontweight="bold", labelpad=8
        )

        for col, ax in enumerate(axs[row]):
            y = dataset[col, 0].cpu()
            ax.plot(
                x,
                y,
                color=color,
                lw=2.2,  # Thicker line for paper clarity
                marker="o",
                ms=3,  # Slightly larger marker
                mfc="white",
                mec=color,
                mew=1.0,
            )

            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)

            if col:
                ax.set_yticklabels([])

            # Axis Tick Label Font Size (14pt)
            ax.tick_params(axis="both", which="major", labelsize=14)

    if save:
        if "title" in save:
            # Figure Title Font Size (18pt)
            fig.suptitle(save["title"], fontsize=18, fontweight="bold")
        fig.savefig(save["filename"], dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()