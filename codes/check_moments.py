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



def spec_plot(Data,synth):
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

def hist_plot(Data,synth,psi = None):
  """Plot per-band histograms of wavelet-coefficient magnitudes.
 
  Filters both signals with a Morlet band-pass bank ``psi`` (built by default from
  the signal length), then for each scale ``j`` and quality index ``q`` overlays the
  histograms of ``|W x|`` for original vs synthesis on a log y-axis.
 
  Parameters
  ----------
  Data, synth : torch.Tensor
      Original and synthetic signals, shape (..., T).
  psi : torch.Tensor, optional
      Precomputed filter bank in Fourier space. Defaults to a Morlet bank with
      ``J = log2(M) - 2`` scales and ``Q = 3``.
  """

  M =Data.shape[-1]
  if psi is None:
      psi = torch.tensor(init_band_pass('morlet', M, J=int(np.log2(M))-2, Q=3, high_freq=0.49, wav_norm='l1'))
  x = torch.fft.ifft(torch.fft.fft(Data.cpu())*psi)
  x_synth = torch.fft.ifft(torch.fft.fft(synth.cpu())*psi)

  for j in range(int(np.log2(M))-2):
    for q in range(3):

      x_j = x[:,j*3+q].flatten().abs()
      x_j_synth = x_synth[:,j*3+q].flatten().abs()

      plt.hist(x_j,bins=50,density=True,label='Orig')
      plt.hist(x_j_synth,bins=50,density=True,alpha=0.7,label='Synth')
      plt.legend()

      plt.title('j,q='+str(j)+','+str(q))
      plt.yscale('log')
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


def cross_plot(Data,synth,pq=[(2,1),(2,2),(3,1),(3,2),(3,3)],epsilon = 1e-8):
  """Plot log cross structure functions and their relative error per ``(p, q)``.
 
  For each exponent pair, shows three 2D images on log-log axes: original
  ``log S_{p,q}``, synthesis ``log S_{p,q}``, and a relative-error map (greyscale).
 
  NOTE: the error denominator is ``second_order + second_order`` (i.e. twice the
  original), not ``second_order + second_order_gen`` -- confirm this is intended.
  Also ``epsilon`` is added after the log rather than inside it.
 
  Parameters
  ----------
  Data, synth : torch.Tensor
      Original and synthetic signals, shape (..., T).
  pq : list of (int, int)
      Exponent pairs.
  epsilon : float
      Stabilizer for the log / error map.
  """

  max_tau = Data.shape[-1]//2
  second_order = cross_structure_function(Data.cpu().numpy(), pq=pq, max_tau=max_tau) #
  second_order_gen = cross_structure_function(synth.cpu().numpy(), pq=pq,max_tau=max_tau)
  log_second_order = np.log(second_order)+epsilon
  log_second_order_gen = np.log(second_order_gen)+epsilon
  error = np.abs((second_order-second_order_gen))/(second_order+second_order)
  vmin,vmax = min(np.min(log_second_order),np.min(log_second_order_gen)), max(np.max(log_second_order),np.max(log_second_order_gen)) 
  #fig = plt.figure(figsize=(5,5))
  #ax = fig.add_subplot()
  #ax.imshow(np.log(second_order[0]+1e-8))
  for i in range(len(second_order)):
      print('(p,q)=',pq[i])
      fig, [ax1,ax2,ax3] = plt.subplots(nrows=1, ncols=3,figsize=(15,5))
        
      ax1.imshow(log_second_order[i],vmin=vmin, vmax=vmax)
      ax1.set_xscale('log')
      ax1.set_yscale('log')
      ax1.set_xlim(1e-1,max_tau)
      ax1.set_ylim(1e-1,max_tau)

      im = ax2.imshow(log_second_order_gen[i],vmin=vmin, vmax=vmax)
      ax2.set_xscale('log')
      ax2.set_yscale('log')
      ax2.set_xlim(1e-1,max_tau)
      ax2.set_ylim(1e-1,max_tau)
      #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
      #fig.colorbar(im, cax=cbar_ax)

      ax3.imshow(error[i],vmin=0,vmax=1,cmap = 'Greys')
      ax3.set_xscale('log')
      ax3.set_yscale('log')
      ax3.set_xlim(1e-1,max_tau)
      ax3.set_ylim(1e-1,max_tau)
     
    
      plt.show()


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
    second_order = np.zeros(shape=(len(p), len(taus)))
    for i in range(len(taus)):
        d_data = data[..., taus[i]:] - data[..., :-taus[i]]
        for j, power in enumerate(p):
            second_order[j, i] = np.abs(np.mean(np.power(d_data.reshape(-1), power)))
    return second_order

def structure_plot(Data,synth):
  """Plot structure functions ``S_p(tau)`` for ``p in {2,4,6,8}``, original vs gen.
 
  First figure: self-similarity ratios ``S_p(tau) / S_2(tau)**(p/2)`` (flat curves
  indicate scaling). Second figure: the raw ``S_p(tau)``. Both on log-log axes;
  dashed = original, markers = generated.
 
  Parameters
  ----------
  Data, synth : torch.Tensor
      Original and synthetic signals, shape (..., T).
  """

  max_tau = Data.shape[-1]//2
  second_order = second_order_structure_function(Data.cpu().numpy(), p=np.array([2, 4, 6, 8]), max_tau=max_tau)
  second_order_gen = second_order_structure_function(synth.cpu().numpy(), p=np.array([2, 4, 6, 8]), max_tau=max_tau)
  fig = plt.figure()
  ax = fig.add_subplot()
  #ax.plot(second_order[0], 'b--', label='original_2', ms=3)
  ax.plot(second_order[1]/second_order[0]**(4/2), 'r--', label='original_4', ms=3)
  ax.plot(second_order[2]/second_order[0]**(6/2), 'g--', label='original_6', ms=3)
  ax.plot(second_order[3]/second_order[0]**(8/2), 'b--', label='original_8', ms=3)

  #ax.plot(second_order_gen[0], 'bo', label='gen_2')
  ax.plot(second_order_gen[1]/second_order_gen[0]**(4/2), 'ro', label='gen_4')
  ax.plot(second_order_gen[2]/second_order_gen[0]**(6/2), 'go', label='gen_6')
  ax.plot(second_order_gen[3]/second_order_gen[0]**(8/2), 'bo', label='gen_8')
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlabel('tau')
  ax.set_ylabel('F_tau_p')
  ax.legend()
  plt.show()

  fig = plt.figure()
  ax = fig.add_subplot()
  #ax.plot(second_order[0], 'b--', label='original_2', ms=3)
  ax.plot(second_order[1], 'r--', label='original_4', ms=3)
  ax.plot(second_order[2], 'g--', label='original_6', ms=3)
  ax.plot(second_order[3], 'b--', label='original_8', ms=3)

  #ax.plot(second_order_gen[0], 'bo', label='gen_2')
  ax.plot(second_order_gen[1], 'ro', label='gen_4')
  ax.plot(second_order_gen[2], 'go', label='gen_6')
  ax.plot(second_order_gen[3], 'bo', label='gen_8')
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlabel('tau')
  ax.set_ylabel('S_tau_p')
  ax.legend()
  plt.show()


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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset




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

    def s22_over_s4(x, t_star):
        ratio = np.zeros(len(taus_arr))
        for i, t in enumerate(taus_arr):
            L = min(x.shape[-1] - t_star, x.shape[-1] - t)
            du_star = np.abs(x[..., t_star:t_star + L] - x[..., :L])
            du_t    = np.abs(x[..., t:t + L]           - x[..., :L])
            s22 = np.mean((du_star ** 2 * du_t ** 2).reshape(-1))
            s4  = np.mean((du_t ** 4).reshape(-1))
            ratio[i] = s22 / (s4 + epsilon)
        return ratio
    r_o = s22_over_s4(data_np,  c4_fixed_tau)
    r_s = s22_over_s4(synth_np, c4_fixed_tau)
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

def plot_moment_matching(barphi_e, barphi_p, t, threshold):
    """Plot the relative moment-matching error between interpolant and walkers.
 
    Restricts to moments whose final interpolant value exceeds ``threshold``, then
    plots the mean symmetric relative error
    ``2 |barphi_e - barphi_p| / (|barphi_e| + |barphi_p|)`` over time and its
    final-time distribution. Falls back to just the histogram on error.
 
    Parameters
    ----------
    barphi_e, barphi_p : torch.Tensor
        Interpolant and walker moment paths, shape (n_t, r).
    t : torch.Tensor
        Time grid.
    threshold : float
        Minimum final moment magnitude to keep.
    """
    # Move everything to CPU once to avoid repetitive .cpu() calls
    barphi_e = barphi_e.cpu()
    barphi_p = barphi_p.cpu()
    t = t.cpu()
    
    # 1. Use PyTorch native boolean masking instead of np.where
    keep_mask = barphi_e[-1] > threshold
    
    # Safety check: If nothing survives the threshold, we can't plot the time series
    if not keep_mask.any():
        print(f"Warning: No moments exceeded the threshold of {threshold}. Plotting fallback histogram.")
        # Calculate error for all moments just to show the fallback histogram
        error_last = (2 * (barphi_e - barphi_p).abs() / (barphi_e.abs() + barphi_p.abs()))[-1]
        plt.hist(error_last, bins=100)
        plt.title('Distribution of moment matching error (All Moments)')
        plt.yscale('log')
        plt.show()
        return

    # Filter tensors
    barphi_e = barphi_e[:, keep_mask]
    barphi_p = barphi_p[:, keep_mask]

    # Calculate the symmetric relative error matrix
    rel_error = 2 * (barphi_e - barphi_p).abs() / (barphi_e.abs() + barphi_p.abs())
    
    try:
        # --- FIX: Align dimensions along the time axis ---
        # Find the minimum available length between time grid and error steps
        min_len = min(t.shape[0], rel_error.shape[0])
        
        # Force both to share the same length, then apply your [2:-1] slice
        t_sliced = t[:min_len][2:-1]
        error_mean_sliced = rel_error.mean(dim=1)[:min_len][2:-1]
        
        plt.plot(t_sliced, error_mean_sliced, marker='.')
        plt.xlabel('t')
        plt.yscale('log')
        plt.title('Relative moment matching error')
        plt.show()
        
    except Exception as e:
        print(f"Time-plot failed due to: {e}. Falling back to histogram.")
    
    # This will now run regardless of whether the first plot succeeded
    plt.hist(rel_error[-1], bins=100)
    plt.title('Distribution of moment matching error')
    plt.yscale('log')
    plt.show()



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

def Compare_time_series_row( Data,Synth ,N):
    """Compare the first ``N`` time series, data on top row, synthesis on bottom.
 
    Parameters
    ----------
    Data, Synth : torch.Tensor
        Original and synthetic signals, shape (B, C, T).
    N : int
        Number of series per row.
    """

    random_indexes = np.arange(N)

    fig, axs = plt.subplots(2, N, figsize=(20, 8))
    for i in range(N):
        idx = random_indexes[i]
        axs[0,i].get_xaxis().set_visible(False)
        axs[0,i].plot(Data[idx,0].cpu())
        axs[1,i].get_xaxis().set_visible(False)
        axs[1,i].plot(Synth[idx,0].cpu())
    axs[0,0].set_ylabel('Data')
    axs[1,0].set_ylabel('Synth')
    plt.show()