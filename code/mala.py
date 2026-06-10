import numpy as np
import scipy
import torch
import torch.fft
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm

class Energy:
    """
    Wraps a dict of potentials into energy/gradient callables for mala.

    Each potential must implement:
      .forward(x)           -> (B, num_coefficients)
      .grad(x, theta_i)     -> (B, C, T)
    """

    def __init__(self, potentials: dict, theta: torch.Tensor):
        self.potentials = potentials
        self.theta = theta
        self._slices = {}
        cursor = 0
        for name, p in self.potentials.items():
            self._slices[name] = slice(cursor, cursor + p.num_coefficients)
            cursor += p.num_coefficients
        assert cursor == theta.shape[0], (
            f"theta size {theta.shape[0]} != total num_coefficients {cursor}"
        )

    def energy(self, x):
        out = None
        for name, p in self.potentials.items():
            theta_i = self.theta[self._slices[name]]
            phi_i   = p.forward(x)
            contrib = phi_i @ theta_i
            out = contrib if out is None else out + contrib
        return out

    def gradient(self, x):
        B, C, T = x.shape
        out = None
        for name, p in self.potentials.items():
            theta_i = self.theta[self._slices[name]]
            grad_i  = p.grad(x, theta_i)
            out = grad_i if out is None else out + grad_i
        return out.reshape(B, C * T)

    def fit(self,x):
        for p in self.potentials.values():
            try:
                p.fit(x)
            except:
                pass

class CondiEnergy:
    """
    Wraps a dict of potentials into energy/gradient callables for mala_conditional.

    Each potential must implement:
      .forward(x, x_condi)          -> (B, num_coefficients)
      .grad(x, x_condi, theta_i)    -> (B, C, T)   where theta_i ~ (num_coefficients,)

    theta: (n_pots,) where n_pots = sum of p.num_coefficients for p in potentials.values()
    """

    def __init__(self, potentials: dict, theta: torch.Tensor,x_condi):
        self.potentials = potentials
        self.theta = theta
        self.x_condi = x_condi
        # precompute slice boundaries once
        self._slices = {}
        cursor = 0
        for name, p in self.potentials.items():
            self._slices[name] = slice(cursor, cursor + p.num_coefficients)
            cursor += p.num_coefficients
        assert cursor == theta.shape[0], (
            f"theta size {theta.shape[0]} != total num_coefficients {cursor}"
        )

    def energy(self, x):
        """Returns scalar energy per batch element: (B,)"""
        x_condi = self.x_condi
        out = None
        for name, p in self.potentials.items():
            theta_i = self.theta[self._slices[name]]          # (num_coefficients_i,)
            phi_i   = p.forward(x, x_condi)                  # (B, num_coefficients_i)
            contrib = phi_i @ theta_i # (B,)
            out = contrib if out is None else out + contrib
        return out

    def gradient(self, x):
        """Returns gradient of energy w.r.t. x, flattened: (B, C*T)"""
        x_condi = self.x_condi
        B, C, T = x.shape
        out = None
        for name, p in self.potentials.items():
            theta_i = self.theta[self._slices[name]]          # (num_coefficients_i,)
            grad_i  = p.grad(x, x_condi, theta_i)            # (B, C, T)
            out = grad_i if out is None else out + grad_i
        return out.reshape(B, C * T)
        
    def fit(self,x):
        x_condi = self.x_condi
        for p in self.potentials.values():
            try:
                p.fit(x,x_condi)
            except:
                pass

def Mala_Sampler(x_0,potentials,theta, n_steps, step_size, epsilon=0, window_min=None, window_max=None, device='cpu', freq=1000) :
    """Conditionnal Windowed MALA Dynamic

    Parameters:
    score  : returns the score with forward
    energy : returns the energy with forward
    x_0 (tensor) : (B,C,T) Seed from which we start MALA dynamic
    n_steps (int) : number of steps
    step_size (float) : step size
    epsilon: x^4 penalisation of energy, if needed
    window_min: boundary for windowed langevin
    window_max: boundary for windowed langevin


    Returns:
        x (tensor) :Result of MALA Dynamic (B,C,T)

    """
      
    shape_original = x_0.shape
    shape_flatten = (shape_original[0], shape_original[1], shape_original[-2]*shape_original[-1])

    #energy and gradient
    wrapped = Energy(potentials, theta= theta)
    wrapped.fit(x_0)
    energy   = wrapped.energy
    score = wrapped.gradient
    mean = []
    x_memory = []

    x=torch.clone(x_0).reshape(shape_flatten)
    n_batch = len(x)
    for _ in tqdm(range(n_steps)):
      if _%freq==0:
          x_memory.append(x.detach().cpu().reshape(shape_original))
      #COMPUTE GRADIENT IN X
      gradient = score(x)[:,None] #(B,C,T)
      #print(x, gradient)
      noise = np.sqrt(2*step_size)*torch.randn_like(x) # (B,C,T)
      x_new = x - step_size * gradient + noise # (B,C,T)
      x_new = x_new.detach()

      #METROPOLIS
      gradient_new = score(x_new)[:,None] # (B,C,T)

      log_qx,log_qx_new  = log_Q_forward(x_new.reshape((n_batch,-1)), x.reshape((n_batch,-1)),gradient.reshape((n_batch,-1)),gradient_new.reshape((n_batch,-1)), step_size) #(B))
      log_pix = - energy(x) #(B,)
      log_pix_new = - energy(x_new) #(B,)

      #print(-log_pix, -log_pix_new)

      log_ratio = log_pix_new-log_pix+log_qx - log_qx_new

      #X^4
      x_new += -step_size*epsilon*4*x**3

      #ACCEPTANCE RULE
      RANDOM = torch.rand(log_ratio.shape,device = log_ratio.device )
      ind_mala = torch.where((RANDOM-torch.exp(log_ratio))>0)[0]
      if _%500 == 0:
        print('Acceptance_rate = '+str(1-len(ind_mala)/n_batch))

      #Windowed Langevin, do not update outside of the window
      if window_max is not None:
        ind_max = torch.where(torch.max(x_new.reshape((x_new.shape[0],-1)),1)[0]>window_max)[0]
        x_new[ind_max] = x[ind_max]
      else:
        pass
      if window_min is not None:
        ind_min = torch.where(torch.max(-x_new.reshape((x_new.shape[0],-1)),1)[0]>-window_min)[0]
        x_new[ind_min] = x[ind_min]
      else:
        pass


      #UPDATE
      x_new[ind_mala] = x[ind_mala]
      x = x_new

    return(x.reshape(shape_original)), torch.stack(x_memory, dim=0)

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

def Mala_Sampler_condi(x_0,x_condi,potentials,theta, n_steps, step_size, epsilon=0, window_min=None, window_max=None, device='cpu', freq=1000) :
    """Conditionnal Windowed MALA Dynamic

    Parameters:
    score  : returns the score with forward
    energy : returns the energy with forward
    x_0 (tensor) : (B,C,T) Seed from which we start MALA dynamic
    n_steps (int) : number of steps
    step_size (float) : step size
    epsilon: x^4 penalisation of energy, if needed
    window_min: boundary for windowed langevin
    window_max: boundary for windowed langevin


    Returns:
        x (tensor) :Result of MALA Dynamic (B,C,T)

    """
      
    shape_original = x_0.shape
    shape_flatten = (shape_original[0], shape_original[1], shape_original[-2]*shape_original[-1])

    #energy and gradient
    wrapped = CondiEnergy(potentials, theta= theta,x_condi = x_condi)
    wrapped.fit(x_0)
    energy   = wrapped.energy
    score = wrapped.gradient
    mean = []
    x_memory = []

    x=torch.clone(x_0).reshape(shape_flatten)
    n_batch = len(x)
    for _ in tqdm(range(n_steps)):
      if _%freq==0:
          x_memory.append(x.detach().cpu().reshape(shape_original))
      #COMPUTE GRADIENT IN X
      gradient = score(x)[:,None] #(B,C,T)
      #print(x, gradient)
      noise = np.sqrt(2*step_size)*torch.randn_like(x) # (B,C,T)
      x_new = x - step_size * gradient + noise # (B,C,T)
      x_new = x_new.detach()

      #METROPOLIS
      gradient_new = score(x_new)[:,None] # (B,C,T)

      log_qx,log_qx_new  = log_Q(x_new.reshape((n_batch,-1)), x.reshape((n_batch,-1)),gradient.reshape((n_batch,-1)),gradient_new.reshape((n_batch,-1)), step_size) #(B))
      log_pix = - energy(x) #(B,)
      log_pix_new = - energy(x_new) #(B,)

      #print(-log_pix, -log_pix_new)

      log_ratio = log_pix_new-log_pix+log_qx - log_qx_new

      #X^4
      x_new += -step_size*epsilon*4*x**3

      #ACCEPTANCE RULE
      RANDOM = torch.rand(log_ratio.shape,device = log_ratio.device )
      ind_mala = torch.where((RANDOM-torch.exp(log_ratio))>0)[0]
      if _%500 == 0:
        print('Acceptance_rate = '+str(1-len(ind_mala)/n_batch))

      #Windowed Langevin, do not update outside of the window
      if window_max is not None:
        ind_max = torch.where(torch.max(x_new.reshape((x_new.shape[0],-1)),1)[0]>window_max)[0]
        x_new[ind_max] = x[ind_max]
      else:
        pass
      if window_min is not None:
        ind_min = torch.where(torch.max(-x_new.reshape((x_new.shape[0],-1)),1)[0]>-window_min)[0]
        x_new[ind_min] = x[ind_min]
      else:
        pass


      #UPDATE
      x_new[ind_mala] = x[ind_mala]
      x = x_new

    return(x.reshape(shape_original)), torch.stack(x_memory, dim=0)

def log_Q(x_prime, x,grad_x,grad_x_prime, step_size):
    """MALA transition proba
    @Parameters:
    -x_prime (tensor): x_{n+1}
    -x (tensor): x_{n}
    -grad_x (tensor): \nabla{log p}x_{n}
    -grad_x_prime (tensor): \nabla{log p}x_{n+1}
    -step_size (float)
    @Returns:
    -log_qx (tensor) :log q(x{x_n}) with q = MALA transition proba
    -log_qx_prime (tensor) :log q(x{x_{n+1}}) with q = MALA transition proba
    """
    log_qx_prime = -(torch.norm(x_prime - x + step_size * grad_x, p=2, dim=1) ** 2) / (4 * step_size)
    log_qx = -(torch.norm(x - x_prime + step_size * grad_x_prime, p=2, dim=1) ** 2) / (4 * step_size)
    return log_qx,log_qx_prime

