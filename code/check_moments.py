import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
import scipy.fftpack as sfft

from .Filters_1d import init_band_pass

def plot_term(term, J, L, M_original, M_sampled, Std_original, Std_sampled):

    if term == 'WxWx':
        plot_L2(J, L, M_original, M_sampled, Std_original, Std_sampled)
    if term == 'W|Wx|W|Wx|_re':
        plot_Second_Order(J, L, M_original, M_sampled, Std_original, Std_sampled)
    if term == 'L1':
        plot_L1(J, L, M_original, M_sampled, Std_original, Std_sampled)


def plot_L2(J, L, M_original, M_sampled, Std_original, Std_sampled):

    for j in range(J):
        fig , ax1 = plt.subplots(nrows=1, ncols=1)


        X = np.arange(len(M_original[j*L:(j+1)*L]))
        ax1.errorbar(X, M_original[j*L:(j+1)*L], yerr=Std_original[j*L:(j+1)*L], label = 'Original',fmt = '--go',elinewidth = 2, capsize = 3, capthick = 3)
        ax1.errorbar(X, M_sampled[j*L:(j+1)*L], yerr=Std_sampled[j*L:(j+1)*L], label = 'Sampled',fmt = '--ro',elinewidth = 2, capsize = 3, capthick = 3)
        ax1.set_title(r'$||W_{j}x-E(W_{j}x)||_2$')
        ax1.legend()

        ax1.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

        plt.show()

    # Low freq

    fig , ax1 = plt.subplots(nrows=1, ncols=1)

    X = np.arange(1)
    ax1.errorbar(X, M_original[-1], yerr=Std_original[-1], label = 'Original',fmt = '--go',elinewidth = 2, capsize = 3, capthick = 3)
    ax1.errorbar(X, M_sampled[-1], yerr=Std_sampled[-1], label = 'Sampled',fmt = '--ro',elinewidth = 2, capsize = 3, capthick = 3)
    ax1.set_title(r'$||W_{LF}x-E(W_{LF}x)||_2$')
    ax1.legend()

    ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

    plt.show()
    
    return

def plot_L1(J, L, M_original, M_sampled, Std_original, Std_sampled):

    for j in range(J):
        fig , ax1 = plt.subplots(nrows=1, ncols=1)


        X = np.arange(len(M_original[j*L:(j+1)*L]))
        ax1.errorbar(X, M_original[j*L:(j+1)*L], yerr=Std_original[j*L:(j+1)*L], label = 'Original',fmt = '--go',elinewidth = 2, capsize = 3, capthick = 3)
        ax1.errorbar(X, M_sampled[j*L:(j+1)*L], yerr=Std_sampled[j*L:(j+1)*L], label = 'Sampled',fmt = '--ro',elinewidth = 2, capsize = 3, capthick = 3)
        ax1.set_title(r'$||W_{j}x||_1$')
        ax1.legend()

        ax1.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

        plt.show()

    fig , ax1 = plt.subplots(nrows=1, ncols=1)

    X = np.arange(1)
    ax1.errorbar(X, M_original[-1], yerr=Std_original[-1], label = 'Original',fmt = '--go',elinewidth = 2, capsize = 3, capthick = 3)
    ax1.errorbar(X, M_sampled[-1], yerr=Std_sampled[-1], label = 'Sampled',fmt = '--ro',elinewidth = 2, capsize = 3, capthick = 3)
    ax1.set_title(r'$||W_{j}x||_1$')
    ax1.legend()

    ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

    plt.show()
    
    return


def plot_Second_Order(J, L, M_original, M_sampled, Std_original, Std_sampled):

    for j in range(J):
        for l in range(L):
            fig , ax1 = plt.subplots(nrows=1, ncols=1)

            X =np.arange(len(PW_r))
            ax1.errorbar(X,PW_r[:,0],  yerr=PW_r[:,2],label = 'Originals',fmt = '--go',elinewidth = 2, capsize = 3, capthick = 3)
            ax1.errorbar(X,PW_r[:,1],  yerr=PW_r[:,3],label = 'Synthesis',fmt = '--ro',elinewidth = 2, capsize = 3, capthick = 3)
            ax1.set_title(r'Re$[<W_{jl}|W|x,W_{jl}|Wx|>]$')
            ax1.legend()
    
    return

def plot_moments(P_r, PW_r, j,l):

    fig , (ax1,ax2) = plt.subplots(nrows=2, ncols=1)

    ############# Wx W|Wx| ##########
    #Real
    X =np.arange(len(P_r))
    ax1.errorbar(X,P_r[:,0],  yerr=P_r[:,2],label = 'Originals',fmt = '--go',elinewidth = 2, capsize = 3, capthick = 3)
    ax1.errorbar(X,P_r[:,1],  yerr=P_r[:,3],label = 'Synthesis',fmt = '--ro',elinewidth = 2, capsize = 3, capthick = 3)
    ax1.set_title(r'Re$[<W_{jl}x,W_{jl}|Wx|>]$')
    ax1.legend()

    ########## W|W|x W|Wx| ##########
    #Real
    X =np.arange(len(PW_r))
    ax2.errorbar(X,PW_r[:,0],  yerr=PW_r[:,2],label = 'Originals',fmt = '--go',elinewidth = 2, capsize = 3, capthick = 3)
    ax2.errorbar(X,PW_r[:,1],  yerr=PW_r[:,3],label = 'Synthesis',fmt = '--ro',elinewidth = 2, capsize = 3, capthick = 3)
    ax2.set_title(r'Re$[<W_{jl}|W|x,W_{jl}|Wx|>]$')
    ax2.legend()

    ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off


    ax2.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off


    fig.suptitle(r'$j =$'+str(j)+', '+ r'$l =$'+str(l), fontsize=14)

    plt.show()

def spec_plot(Data,synth):

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
    taus = np.arange(1, max_tau, 1)
    second_order = np.zeros(shape=(len(p), len(taus)))
    for i in range(len(taus)):
        d_data = data[..., taus[i]:] - data[..., :-taus[i]]
        for j, power in enumerate(p):
            second_order[j, i] = np.abs(np.mean(np.power(d_data.reshape(-1), power)))
    return second_order

def structure_plot(Data,synth):
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
    for i in range(min(50,len(synth))):
      plt.plot(synth.cpu()[i,0])
      plt.show()

def Compare_Spectrum(DATA, X_fake, log = False):
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