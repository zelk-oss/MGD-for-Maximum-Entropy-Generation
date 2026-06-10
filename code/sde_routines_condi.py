"""
Moment-Guided Diffusion (MGD) - conditional sampler (SDE_condi).
 
Conditional variant of Algorithm 1. The operator ``W`` splits a signal into a
``direct`` part (evolved by the SDE) and a ``condi`` part (conditioning, held fixed
to the data conditioning ``x_1_condi``); ``W(direct, condi)`` recomposes them. Every
moment / gradient / Gram-matrix call carries both parts, but only ``direct`` moves.
 
Notation
--------
r           : number of scalar potentials, ``self.num_potentials``.
phi          : potentials, evaluated on ``(direct, condi)`` pairs.
phi_bar(.)   : empirical moments, ``mean_n phi(.)`` (length r).
G            : Gram matrix ``mean_n grad phi grad phi^T`` (r x r), built on ``direct``.
I_t          : interpolant on the ``direct`` part between ``x_0_direct`` and ``x_1_direct``.
eta_k        : solves ``G eta = d/dt phi_bar(I_t)``.
theta_k      : solves ``G theta = phi_bar(I_{k+1}) - phi_bar(y_k)`` (corrector).
 
Signal layouts (``signal_dim``, from ``x_1_direct.shape``): 1 -> (B,C,T), 2 -> (B,C,M,N).
 
Each ``potential`` exposes: ``forward(d, c)``, ``grad(d, c)``, ``grad(d, c, v=vec)``,
``num_coefficients``, optional ``fit(d, c)``.
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.special import erfcx, erf, erfinv
from scipy.integrate import trapezoid
from scipy import stats
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from potentials.utils_potentials import *
from potentials_builder import *
from filters_bank import * 
from utils import *
from utils_entropy import*


class SDE_condi(torch.nn.Module):
    """
    Conditional predictor-corrector interpolant sampler (Algorithm 1 of MGD).
 
    ``W.decompose(x_1)`` splits the data into ``(x_1_direct, x_1_condi)``. The SDE
    evolves only ``direct``; ``condi`` stays fixed at ``x_1_condi``. At each step the
    predictor follows the moment velocity ``d/dt phi_bar(I_t)`` and the corrector
    moves the walkers onto the moment target ``phi_bar(I_{k+1})``. ``forward`` returns
    the recomposed samples ``W(x_k_direct, x_k_condi)``.
 
    Parameters
    ----------
    W : object
        Decomposition operator with ``decompose(x) -> (direct, condi)`` and
        ``__call__(direct, condi) -> signal``.
    x_1 : torch.Tensor
        Data samples (decomposed internally into direct/condi endpoints).
    n_rep : int
        Number of walkers.
    nb_interpolants : int
        Number of interpolant samples used to estimate the moment path.
    t : array_like
        Time grid in [0, 1]; integration runs over ``t[:-1]``.
    sigma : float
        Predictor noise amplitude (0 -> deterministic predictor).
    potentials : dict[str, object]
        Named conditional potentials (interface in the module docstring).
    batch_size : int
        Mini-batch size for the per-sample reductions.
    device : str, optional
        Torch device, by default ``'cpu'``.
    regularization : tuple, optional
        Value added to the diagonal of the rescaled Gram matrix, by default (0, 0, 0).
    interpolant : str, optional
        ``'Linear'`` | ``'VarPreserv'`` | ``'Sqrt'`` | ``'Cos'`` (default).
    x_0_direct, x_k_direct, x_k_condi : torch.Tensor, optional
        Interpolant noise endpoint, initial walkers (direct), and walker conditioning;
        defaulted from noise / data conditioning if None.
    """


    def __init__(
        self,
        W,
        x_1,
        n_rep,
        nb_interpolants,
        t,
        sigma,
        potentials,
        batch_size,
        device='cpu',
        regularization=(0, 0, 0),
        interpolant='Cos',
        x_0_direct=None,
        x_k_direct=None,
        x_k_condi = None,
        
    ):
        super().__init__()

        self.W = W

        self.x_1_direct = x_1
        self.x_1_direct,self.x_1_condi = self.W.decompose(x_1)
        self.original_signal_shape = self.x_1_direct.shape

        match len(self.x_1_direct.shape):
            case 3:
                self.signal_dim = 1
                print(f'Signal detected as 1D: (B, C, T) = ({batch_size}, {self.original_signal_shape[1]}, {self.original_signal_shape[2]}).')
            case 4:
                self.signal_dim = 2
                print(f'Signal detected as 2D: (B, C, M, N) = ({batch_size}, {self.original_signal_shape[1]}, {self.original_signal_shape[2]}, {self.original_signal_shape[3]}).')

        self.n_rep           = n_rep
        self.nb_interpolants = nb_interpolants
        self.t               = t
        self.sigma           = sigma
        self.potentials      = potentials
        self.batch_size      = batch_size
        self.device          = device
        self.regularization  = regularization
        self.interpolant     = interpolant
        self.x_0_direct      = x_0_direct
            
         
        self.x_k_direct = x_k_direct
        self.x_k_condi = x_k_condi
        
            

        self.init_interpolants_and_workers()

        list_potential_num_coefficients = [p.num_coefficients for p in self.potentials.values()]
        self.num_potentials    = sum(list_potential_num_coefficients)
        self.indices_potentials = np.cumsum([0] + list_potential_num_coefficients)
        print(f'The model has {self.num_potentials} potentials.')


    # ------------------------------------------------------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------------------------------------------------------

    def init_interpolants_and_workers(self,):
        """Set the direct interpolant endpoints (noise std-scaled to the data) and the
        direct walkers; conditioning (``x_0_condi``, ``x_k_condi``) is tied to
        ``x_1_condi``. Supplied tensors are tiled to the required counts."""

        std = self.x_1_direct.std()

        match len(self.original_signal_shape):
            case 3:
                if self.x_0_direct is None:
                    self.x_0_direct = torch.randn(self.nb_interpolants, self.original_signal_shape[1], self.original_signal_shape[2]).to(self.device) * std
                else:
                    self.x_0_direct = self.x_0_direct.repeat((self.nb_interpolants // self.x_0_direct.shape[0] + 1, 1, 1))[:self.nb_interpolants]
                    
                self.x_1_direct = self.x_1_direct.repeat((self.nb_interpolants // self.original_signal_shape[0] + 1, 1, 1))[:self.nb_interpolants]
                self.x_1_condi = self.x_1_condi.repeat((self.nb_interpolants // self.original_signal_shape[0] + 1, 1, 1))[:self.nb_interpolants]

                self.x_0_condi = self.x_1_condi

                if self.x_k_direct is None:
                    self.x_k_direct = torch.randn(self.n_rep, self.original_signal_shape[1], self.original_signal_shape[2]).to(self.device) * std
                    
                if self.x_k_condi is None:
                    try:
                        self.x_k_condi = self.x_1_condi.repeat((self.n_rep//self.x_1_condi.shape[0], 1, 1))
                    except:
                        self.x_k_condi = self.x_1_condi.repeat[ : self.n_rep]


    # ------------------------------------------------------------------------------------------------------------------
    # Main function
    # ------------------------------------------------------------------------------------------------------------------

    def forward(self, param_storage_frequency=1):
        """Integrate the SDE over ``t[:-1]`` and recompose the result.
 
        Returns ``(x_k, barphi_e, barphi_p, eta_k_list, theta_k_list, dH_k_list)``:
        the recomposed samples ``W(x_k_direct, x_k_condi)``, the interpolant and
        walker moment paths, the predictor/corrector coefficients, and the entropy
        increments. ``param_storage_frequency`` sets the recording cadence."""


        #Fiting
        self.fit(self.x_k_direct,self.x_k_condi)

        barphi_e = [self.compute_moments(self.x_0_direct,self.x_0_condi).mean(0)]
        barphi_p = [self.compute_moments(self.x_k_direct,self.x_k_condi).mean(0)]

        eta_k_list   = []
        theta_k_list = []
        dH_k_list    = []

        for k, t_k in tqdm(enumerate(self.t[:-1])):

            #Fiting
            self.fit(self.x_k_direct,self.x_k_condi)
            
            
            self.x_k_direct, I_k_direct, eta_k, theta_k, dH_k = self.iteration_step_projection(self.x_k_direct,self.x_k_condi, k)

            if (k + 1) % param_storage_frequency == 0:
                eta_k_list.append(eta_k)
                theta_k_list.append(theta_k)
                dH_k_list.append(dH_k)
                barphi_e.append(self.compute_moments(I_k_direct,self.x_0_condi).mean(0))
                barphi_p.append(self.compute_moments(self.x_k_direct,self.x_k_condi).mean(0))

        # Store final parameters and statistics
        eta_k_list.append(eta_k)
        theta_k_list.append(theta_k)
        dH_k_list.append(dH_k)
        barphi_e.append(self.compute_moments(I_k_direct,self.x_0_condi).mean(0))
        barphi_p.append(self.compute_moments(self.x_k_direct,self.x_k_condi).mean(0))

        self.x_k = self.W(self.x_k_direct,self.x_k_condi)
        
        return (
            self.x_k,
            torch.stack(barphi_e),
            torch.stack(barphi_p),
            torch.stack(eta_k_list),
            torch.stack(theta_k_list),
            torch.cat(dH_k_list),
        )


        
    def iteration_step_projection(self,x_k_direct,x_k_condi, k):
        """One predictor-corrector step on ``direct`` (``condi`` fixed).
 
        Predictor: ``y = x + h (grad phi)^T eta_k + sqrt(2h) sigma noise``.
        Corrector: ``x' = y + (grad phi)^T theta_k`` toward ``phi_bar(I_{k+1})``.
        ``theta_k`` is rescaled by ``h sigma**2`` and the entropy increment is
        ``dH_k = -theta_k^T d/dt phi_bar(I_t)``. Returns
        ``(x', I_{k+1}, eta_k, theta_k, dH_k)``."""

        h = self.t[k + 1] - self.t[k]

        # Predictor
        eta_k       = self.compute_eta(x_k_direct,x_k_condi, k)
        drift       = self.compute_grad_phi_projected(x_k_direct,x_k_condi, eta_k)
        noise_scale = (2 * h) ** 0.5 * self.sigma
        noise       = noise_scale * torch.randn_like(x_k_direct).to(self.device)
        y_k_direct  = x_k_direct + h * drift + noise

        # Corrector
        theta_k               = self.compute_theta(y_k_direct,x_k_condi, k)
        corrector             = self.compute_grad_phi_projected(y_k_direct,x_k_condi, theta_k)
        x_k_direct_plus_one   = y_k_direct + corrector

        # Normalise theta
        if self.sigma > 0:
            theta_k = theta_k / (h * self.sigma ** 2)
        else:
            theta_k = torch.zeros_like(theta_k)

        # Entropy estimate
        I_k_direct   = self.compute_interpolant(k + 1)
        dt_phi_I_k   = self.compute_rhs_dt_phi_I_t(I_k_direct, k)
        dH_k         = -theta_k @ dt_phi_I_k

        return x_k_direct_plus_one, I_k_direct, eta_k, theta_k, dH_k


    # ------------------------------------------------------------------------------------------------------------------
    # Intermediate steps (in call order)
    # ------------------------------------------------------------------------------------------------------------------

    def compute_eta(self, x_k_direct,x_k_condi, k):
        """Predictor coefficients: solve ``G eta = d/dt phi_bar(I_t)`` at step ``k``.
        ``G`` is rescaled by its diagonal ``D=diag(G)``, symmetrized and regularized
        before the solve, then the solution is rescaled back by ``D**-0.5``.
        Returns shape (r,)."""

 
        I_k_direct       = self.compute_interpolant(k)
        rhs_dt_phi_I_k   = self.compute_rhs_dt_phi_I_t(I_k_direct, k)
        G_k              = self.compute_G(x_k_direct,x_k_condi)
        
        ################################Regularization###################################
        D_k12 = torch.diag(G_k)**0.5 
        G_k = G_k /(D_k12[:,None]*D_k12[None,:])
        G_k = (G_k+G_k.T)/2
        G_k+= self.regularization*torch.eye(G_k.shape[-1],).to(G_k.dtype).to(G_k.device)
        
        eta_k = torch.linalg.solve(G_k, rhs_dt_phi_I_k/D_k12[:,None])[:, 0]
        eta_k = eta_k/D_k12

        return eta_k


    def compute_grad_phi_projected(self, x_direct,x_condi, vector):
        """Field ``sum_i vector_i grad phi_i(direct, condi)``, mini-batched over
        samples. Same per-sample shape as ``x_direct``."""


        batch_size  = self.batch_size
        num_samples = x_direct.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        if self.signal_dim == 0:
            grad_phi_eta = torch.zeros((0, self.original_signal_shape[1]), device=self.device)
        elif self.signal_dim == 1:
            grad_phi_eta = torch.zeros((0, self.original_signal_shape[1], self.original_signal_shape[2]), device=self.device)
        elif self.signal_dim == 2:
            grad_phi_eta = torch.zeros((0, self.original_signal_shape[1], self.original_signal_shape[2], self.original_signal_shape[3]), device=self.device)

        for idx_batch in range(num_batches):
            batch_direct        = x_direct[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            batch_condi         = x_condi[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            
            grad_phi_eta_batch   = self.compute_grad_potentials(batch_direct,batch_condi, vector)
            grad_phi_eta         = torch.cat([grad_phi_eta, grad_phi_eta_batch], dim=0)

        return grad_phi_eta


    def compute_theta(self, y_k,x_k_condi, k):
        """Corrector coefficients: solve ``G theta = phi_bar(I_{k+1}) - phi_bar(y_k)``
        with the same diagonal rescaling / regularization as ``compute_eta``.
        Returns shape (r,)."""


        I_k_direct                = self.compute_interpolant(k + 1)
        rhs_constraint_correction = self.compute_rhs_constraint_correction(y_k,x_k_condi, I_k_direct)
        G_k                       = self.compute_G(y_k,x_k_condi)
        #return torch.linalg.solve(G_k, rhs_constraint_correction)
        
        ################################Regularization###################################
        D_k12 = torch.diag(G_k)**0.5 
        G_k = G_k /(D_k12[:,None]*D_k12[None,:])
        G_k = (G_k+G_k.T)/2
        G_k+= self.regularization*torch.eye(G_k.shape[-1],).to(G_k.dtype).to(G_k.device)
        
        theta_k = torch.linalg.solve(G_k, rhs_constraint_correction/D_k12)
        theta_k = theta_k/D_k12
        
        return theta_k

        


    def compute_rhs_dt_phi_I_t(self, I_k_direct, k):
        """Moment velocity ``d/dt phi_bar(I_t) = mean_n <grad phi(I, x_1_condi), I_dot>``,
        with ``I_dot`` from ``gradient_interpolant(k)``. Mini-batched; returns (r, 1)."""


        batch_size  = self.batch_size
        num_samples = I_k_direct.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        rhs      = torch.zeros((self.num_potentials, 1)).to(self.device)
        I_k_direct_dot  = self.gradient_interpolant(k)

        for idx_batch in range(num_batches):
            batch_direct         = I_k_direct[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            batch_condi          = self.x_1_condi[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            
            I_k_direct_dot_batch = I_k_direct_dot[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            grad_potential       = self.compute_grad_potentials(batch_direct,batch_condi)

            if self.signal_dim == 0:
                rhs += torch.matmul(grad_potential, I_k_direct_dot_batch.reshape(batch_direct.shape[0], 1, 1)).sum(0)
            elif self.signal_dim == 1:
                rhs += torch.matmul(grad_potential, I_k_direct_dot_batch.reshape(batch_direct.shape[0], batch_direct.shape[-1], 1)).sum(0)
            elif self.signal_dim == 2:
                rhs += torch.matmul(
                    grad_potential.reshape(batch_direct.shape[0], self.num_potentials, batch_direct.shape[-2] * batch_direct.shape[-1]),
                    I_k_direct_dot_batch.reshape(batch_direct.shape[0], batch_direct.shape[-2] * batch_direct.shape[-1], 1),
                ).sum(0)

        rhs /= num_samples

        return rhs


    def compute_G(self, x_direct,x_condi):
        """Gram matrix ``G_ij = mean_n <grad phi_i, grad phi_j>`` on ``(direct, condi)``,
        accumulated in mini-batches. Returns symmetric (r, r)."""

        batch_size  = self.batch_size
        num_samples = x_direct.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        G = torch.zeros((self.num_potentials, self.num_potentials)).to(self.device)

        for idx_batch in range(num_batches):
            batch_direct          = x_direct[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            batch_condi          = x_condi[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            grad_potential = self.compute_grad_potentials(batch_direct,batch_condi)

            if self.signal_dim == 2:
                grad_potential = grad_potential.reshape(batch_direct.shape[0], self.num_potentials, batch_direct.shape[-2] * batch_direct.shape[-1])

            G += torch.bmm(grad_potential, grad_potential.transpose(1, 2)).sum(0)

        G /= num_samples

        return G


    def compute_grad_potentials(self, x_direct,x_condi, vector=None):
        """If ``vector is None``: stacked gradients ``grad phi(direct, condi)``, shape
        (B, r, *signal). Else: contracted field ``sum_i grad phi_i @ vector[block_i]``,
        shaped like ``x_direct``."""


        if vector is None:
            grad_potential = torch.tensor([], device=x_direct.device)
            for potential in self.potentials.values():
                grad_potential = torch.cat((grad_potential, potential.grad(x_direct,x_condi)), dim=1).detach()

            if self.signal_dim == 0:
                grad_potential = grad_potential.reshape(x_direct.shape[0], self.num_potentials, 1)
            elif self.signal_dim == 1:
                grad_potential = grad_potential.reshape(x_direct.shape[0], self.num_potentials, x_direct.shape[-1])
            elif self.signal_dim == 2:
                grad_potential = grad_potential.reshape(x_direct.shape[0], self.num_potentials, x_direct.shape[-2], x_direct.shape[-1])

            return grad_potential

        else:
            grad_phi_eta = torch.zeros_like(x_direct)
            for i, potential in enumerate(self.potentials.values()):
                grad_phi_eta += potential.grad(x_direct,x_condi, v=vector[self.indices_potentials[i]:self.indices_potentials[i + 1]])

            return grad_phi_eta


    def compute_rhs_constraint_correction(self, x_k_direct,x_k_condi,I_k_direct):
        """Moment mismatch driving the corrector:
        ``phi_bar(I_k, x_1_condi) - phi_bar(x_k_direct, x_k_condi)``. Length r."""


        bar_phi_I_k      = self.compute_moments(I_k_direct,self.x_1_condi).mean(0)
        bar_phi_x_current = self.compute_moments(x_k_direct,self.x_k_condi).mean(0)

        return bar_phi_I_k - bar_phi_x_current


    def compute_moments(self, x_direct,x_condi):
        """Per-sample potentials ``phi(direct, condi)`` over all potentials,
        mini-batched. Returns (N, r); average over samples gives ``phi_bar``."""


        batch_size  = self.batch_size
        num_samples = x_direct.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        moments = torch.zeros((0, self.num_potentials)).to(self.device)

        for idx_batch in range(num_batches):
            batch_direct         = x_direct[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            batch_condi          = x_condi[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            moments_batch = torch.tensor([], device=x_direct.device)
            
            for potential in self.potentials.values():
                moments_batch = torch.cat((moments_batch, potential.forward(batch_direct,batch_condi)), dim=1)
            moments = torch.cat([moments, moments_batch], dim=0)

        return moments


    # ------------------------------------------------------------------------------------------------------------------
    # Interpolant
    # ------------------------------------------------------------------------------------------------------------------

    def compute_interpolant(self, k):
        """Interpolant ``I(t_k)`` on the direct part between ``x_0_direct`` and
        ``x_1_direct`` (Linear / VarPreserv / Sqrt / Cos)."""


        t, x_0_direct, x_1_direct = self.t, self.x_0_direct, self.x_1_direct

        match self.interpolant:
            case 'Linear':
                return (1 - t[k]) * x_0_direct + t[k] * x_1_direct
            case 'VarPreserv':
                return np.sqrt(1 - t[k]) * x_0_direct + np.sqrt(t[k]) * x_1_direct
            case 'Sqrt':
                return (1 - np.sqrt(t[k])) * x_0_direct + np.sqrt(t[k]) * x_1_direct
            case 'Cos':
                return np.cos(np.pi * t[k] / 2) * x_0_direct + np.sin(np.pi * t[k] / 2) * x_1_direct


    def gradient_interpolant(self, k):
        """Time derivative ``I_dot(t_k)`` of the direct interpolant (matches the
        schedule in ``compute_interpolant``)."""


        t, x_0_direct, x_1_direct = self.t, self.x_0_direct, self.x_1_direct

        match self.interpolant:
            case 'Linear':
                return x_1_direct - x_0_direct
            case 'VarPreserv':
                return x_1_direct / (2 * np.sqrt(t[k])) - x_0_direct / (2 * np.sqrt(1 - t[k]))
            case 'Sqrt':
                return (x_1_direct - x_0_direct) / (2 * np.sqrt(t[k]))
            case 'Cos':
                return (np.pi / 2) * (-np.sin(np.pi * t[k] / 2) * x_0_direct + np.cos(np.pi * t[k] / 2) * x_1_direct)
    # ------------------------------------------------------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------------------------------------------------------
    
    def fit(self,x_k_direct,x_k_condi):
        """Refit potentials implementing ``fit(direct, condi)``; skip the rest."""
        for pot in self.potentials.values():
                try:
                    pot.fit(x_k_direct,x_k_condi)
                    
                except:
                    pass
