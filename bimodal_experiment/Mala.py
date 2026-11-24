

import numpy as np
import scipy
import torch
import torch.fft
import torch.nn as nn
import matplotlib.pyplot as plt


def Mala_Sampler(score,energy,x_0,n_steps,step_size,epsilon=0,window_min=None,window_max=None) :
    """Conditionnal Windowed MALA Dynamic

    Parameters:
    score  : returns the score with forward
    energy : returns the erngy with forward
    x_0 (tensor) : (B,C,T) Seed from which we start MALA dynamic
    n_steps (int) : number of steps
    step_size (float) : step size
    epsilon: x^4 penalisation of energy, if needed
    window_min: boundary for windowed langevin
    window_max: boundary for windowed langevin


    Returns:
        x (tensor) :Result of MALA Dynamic (B,C,T)

    """

    mean = []

    x=torch.clone(x_0)
    n_batch = len(x)
    for _ in range(n_steps):
      #COMPUTE GRADIENT IN X
      gradient = score(x) #(B,C,T)
      noise = np.sqrt(2*step_size)*torch.randn_like(x) # (B,C,T)
      x_new = x - step_size * gradient + noise # (B,C,T)
      x_new = x_new.detach()

      #METROPOLIS
      gradient_new = score(x_new) # (B,C,T)

      log_qx,log_qx_new  = log_Q_forward(x_new.reshape((n_batch,-1)), x.reshape((n_batch,-1)),gradient.reshape((n_batch,-1)),gradient_new.reshape((n_batch,-1)), step_size) #(B))
      log_pix = - energy(x) #(B,)
      log_pix_new = - energy(x_new) #(B,)

      log_ratio = log_pix_new-log_pix+log_qx - log_qx_new

      #X^4
      x_new += -step_size*epsilon*4*x**3

      #ACCEPTANCE RULE
      RANDOM = torch.rand(log_ratio.shape,device = log_ratio.device )
      ind_mala = torch.where((RANDOM-torch.exp(log_ratio))>0)[0]
      if _%5000 ==0:
        print('Acceptance_rate = '+str(1-len(ind_mala)/n_batch))

      #Windowed Langevin, do not update outside of the window
      if window_max is not None:
        ind_max =  torch.where(torch.max(x_new.reshape((x_new.shape[0],-1)),1)[0]>window_max)[0]
        x_new[ind_max] = x[ind_max]
      else:
        pass
      if window_min is not None:
        ind_min =  torch.where(torch.max(-x_new.reshape((x_new.shape[0],-1)),1)[0]>-window_min)[0]
        x_new[ind_min] = x[ind_min]
      else:
        pass


      #UPDATE
      x_new[ind_mala] = x[ind_mala]
      x = x_new


    return(x)

def log_Q_forward(x_prime, x,grad_x,grad_x_prime, step_size):
    """MALA transition proba

    Parameters:
    x_prime (tensor): x_n+1}
    x (tensor): x_{n}
    grad_x (tensor): \nabla{log p}x_{n}
    grad_x_prime (tensor): \nabla{log p}x_{n+1}
    step_size (float)



    Returns:
        log_qx (tensor) :log q(x{x_n}) with q = MALA transition proba
        log_qx_prime (tensor) :log q(x{x_{n+1}}) with q = MALA transition proba

    """
    log_qx_prime = -(torch.norm(x_prime - x + step_size * grad_x, p=2, dim=1) ** 2) / (4 * step_size)
    log_qx = -(torch.norm(x - x_prime + step_size * grad_x_prime, p=2, dim=1) ** 2) / (4 * step_size)
    return log_qx,log_qx_prime