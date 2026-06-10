"""
Moment-Guided Diffusion (MGD) — stochastic interpolant sampler.
 
Implements Algorithm 1 of the Moment-Guided Diffusion paper together with two
Langevin baselines.
 
Idea
----
Given data samples ``x_1`` and a noise reference ``x_0``, an interpolant ``I_t`` is
built between them (``I_0 = x_0``, ``I_1 = x_1``). A walker ensemble ``x_k`` is
evolved by an SDE so that, at every step, the empirical moments of the walkers
follow the moments of the interpolant ``phi_bar(I_t)``.
 
Notation
--------
r           : number of scalar potentials, ``self.num_potentials``.
N           : number of samples in the ensemble being processed.
phi         : potentials ``phi = (phi_1, ..., phi_r)``.
phi_bar(x)  : empirical moments, ``phi_bar(x) = mean_n phi(x_n)``  (length r).
grad phi(x) : gradients of the potentials w.r.t. the signal.
G(x)        : Gram matrix ``G_ij = mean_n <grad phi_i(x_n), grad phi_j(x_n)>`` (r x r).
I_t         : interpolant at time t; I_dot_t its time derivative.
eta_k       : solves ``G eta = d/dt phi_bar(I_t)``.
theta_k     : solves ``G theta = phi_bar(I_{k+1}) - phi_bar(x_k)``.
 
Signal layouts
--------------
``signal_dim`` is inferred from ``x_1.shape``:
    0 -> scalar  : (B, C)
    1 -> 1D      : (B, C, T)
    2 -> 2D      : (B, C, M, N)
 
Each ``potential`` object exposes:
    forward(x)        -> moments,        shape (B, num_coefficients)
    grad(x)           -> gradient fields, flattened over the potential axis
    grad(x, v=vec)    -> sum_i v_i grad phi_i(x)
    num_coefficients  : int
    fit(x)            : optional refit (no-op if absent)
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





class SDE(torch.nn.Module):
    """
    Predictor-corrector stochastic interpolant sampler (Algorithm 1 of MGD).
 
    Integrates an SDE forward in interpolant time ``t``. At each step a predictor
    moves the walkers along a drift reproducing the moment velocity
    ``d/dt phi_bar(I_t)``, and a corrector moves them back onto the moment target
    ``phi_bar(I_{k+1})``. Both solve a linear system against the Gram matrix ``G``.
 
    Structure
    ---------
        - Initialization
        - forward                     : main loop integrating the SDE
        - iteration_step_projection   : a single predictor-corrector step
        - intermediate helpers         (in call order)
        - interpolant utilities
 
    Parameters
    ----------
    x_1 : torch.Tensor
        Data samples (interpolant endpoint at t = 1). Its shape sets the signal
        layout: (B, C), (B, C, T) or (B, C, M, N).
    n_rep : int
        Number of walkers in the ensemble ``x_k``.
    nb_interpolants : int
        Number of interpolant samples used to estimate ``phi_bar(I_t)`` and its time
        derivative.
    t : array_like
        Monotone time grid in [0, 1]. Integration runs over ``t[:-1]``.
    sigma : float
        Noise amplitude of the predictor. ``sigma = 0`` gives a deterministic
        predictor.
    potentials : dict[str, object]
        Named potential objects (see module docstring for the expected interface).
    batch_size : int
        Mini-batch size used inside the per-sample reductions (moments, gradients,
        Gram matrix) to bound memory.
    device : str, optional
        Torch device, by default ``'cpu'``.
    regularization : float, optional
        Value added to the diagonal of the (rescaled) Gram matrix before each solve,
        by default 0.
    interpolant : str, optional
        Interpolant schedule: ``'Linear'``, ``'VarPreserv'``, ``'Sqrt'`` or
        ``'Cos'`` (default).
    x_0 : torch.Tensor, optional
        Noise endpoint of the interpolant (t = 0). Drawn as Gaussian noise if None.
    x_k : torch.Tensor, optional
        Initial walker ensemble. Drawn as Gaussian noise if None.
 
    Attributes
    ----------
    original_signal_shape : torch.Size
        Shape of the supplied ``x_1``.
    signal_dim : int
        0, 1 or 2 (scalar / 1D / 2D), inferred from ``x_1``.
    num_potentials : int
        Total number of scalar potentials ``r= sum_i num_coefficients_i``.
    indices_potentials : numpy.ndarray
        Cumulative offsets delimiting each potential's coefficient block, used to
        slice ``eta`` / ``theta`` per potential.
    """


    def __init__(
        self,
        x_1,
        n_rep,
        nb_interpolants,
        t,
        sigma,
        potentials,
        batch_size,
        device='cpu',
        regularization=0,
        interpolant='Cos',
        x_0=None,
        x_k=None,
    ):
        super().__init__()

        self.x_1 = x_1
        self.original_signal_shape = self.x_1.shape

        match len(self.x_1.shape):
            case 2:
                self.signal_dim = 0
                print(f'Signal detected as scalar: (B, C) = ({batch_size}, {self.original_signal_shape[1]}).')
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
        self.x_0             = x_0
        self.x_k             = x_k

        self.init_interpolants_and_workers()

        list_potential_num_coefficients = [p.num_coefficients for p in self.potentials.values()]
        self.num_potentials    = sum(list_potential_num_coefficients)
        self.indices_potentials = np.cumsum([0] + list_potential_num_coefficients)
        print(f'The model has {self.num_potentials} potentials.')


    # ------------------------------------------------------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------------------------------------------------------

    def init_interpolants_and_workers(self,):
        """
        Initialize the interpolant endpoints and the walker ensemble.
 
        Sets ``x_0`` (noise endpoint), ``x_1`` (data endpoint, tiled to length
        ``nb_interpolants``) and ``x_k`` (walkers, length ``n_rep``) according to the
        detected ``signal_dim``. Any endpoint supplied to the constructor is reused
        and tiled to the required count; otherwise it is drawn as Gaussian noise.
        Dispatched per signal layout (scalar / 1D / 2D).
 
        Notes
        -----
        Mutates ``self.x_0``, ``self.x_1`` and ``self.x_k`` in place.
        """


        match len(self.original_signal_shape):

            case 2:
                if self.x_0 is None:
                    self.x_0 = torch.randn(self.nb_interpolants, 1).to(self.device)
                else:
                    self.x_0 = self.x_0.repeat((self.nb_interpolants // self.x_0.shape[0] + 1, 1))[:self.nb_interpolants]
                self.x_1 = self.x_1.repeat((self.nb_interpolants // self.x_1.shape[0] + 1, 1))[:self.nb_interpolants]

                if self.x_k is None:
                    self.x_k = torch.randn(self.n_rep, 1).to(self.device)

            case 3:
                if self.x_0 is None:
                    self.x_0 = torch.randn(self.nb_interpolants, self.original_signal_shape[1], self.original_signal_shape[2]).to(self.device)
                else:
                    self.x_0 = self.x_0.repeat((self.nb_interpolants // self.x_0.shape[0] + 1, 1, 1))[:self.nb_interpolants]
                self.x_1 = self.x_1.repeat((self.nb_interpolants // self.original_signal_shape[0] + 1, 1, 1))[:self.nb_interpolants]

                if self.x_k is None:
                    self.x_k = torch.randn(self.n_rep, self.original_signal_shape[1], self.original_signal_shape[2]).to(self.device)

            case 4:
                if self.x_0 is None:
                    self.x_0 = torch.randn(self.nb_interpolants, self.original_signal_shape[1], self.original_signal_shape[2], self.original_signal_shape[3]).to(self.device)
                else:
                    self.x_0 = self.x_0.repeat((self.nb_interpolants // self.x_0.shape[0] + 1, 1, 1, 1))[:self.nb_interpolants]
                self.x_1 = self.x_1.repeat((self.nb_interpolants // self.original_signal_shape[0] + 1, 1, 1, 1))[:self.nb_interpolants]

                if self.x_k is None:
                    self.x_k = torch.randn(self.n_rep, self.original_signal_shape[1], self.original_signal_shape[2], self.original_signal_shape[3]).to(self.device)


    # ------------------------------------------------------------------------------------------------------------------
    # Main function
    # ------------------------------------------------------------------------------------------------------------------

    def forward(self, param_storage_frequency=1):
        """
        Integrate the SDE over the time grid and collect the trajectories.
 
        Fits the potentials on the current walkers, then loops over ``t[:-1]``,
        calling :meth:`iteration_step_projection` at each step. At a configurable
        cadence it records the predictor / corrector coefficients, the entropy
        increment, and the moments of both the interpolant and the walkers.
 
        Parameters
        ----------
        param_storage_frequency : int, optional
            Store diagnostics every ``param_storage_frequency`` steps, by default 1.
 
        Returns
        -------
        x_k : torch.Tensor
            Final walker ensemble (the generated samples).
        barphi_e : torch.Tensor
            Interpolant moments ``phi_bar(I_t)`` stacked over stored steps,
            shape (num_stored, r).
        barphi_p : torch.Tensor
            Walker moments ``phi_bar(x_k)`` stacked over stored steps,
            shape (num_stored, r).
        eta_k_list : torch.Tensor
            Predictor coefficients at the stored steps, shape (num_stored, r).
        theta_k_list : torch.Tensor
            Corrector coefficients (normalized) at the stored steps,
            shape (num_stored, r).
        dH_k_list : torch.Tensor
            Entropy increments, concatenated over stored steps.
 
        Notes
        -----
        ``barphi_e`` / ``barphi_p`` are seeded before the loop with the moments of
        ``x_0`` and ``x_k``, and the final state is appended after the loop so the
        trajectories include both endpoints.
        """


        #Fiting
        self.fit(self.x_k)

        barphi_e = [self.compute_moments(self.x_0).mean(0)]
        barphi_p = [self.compute_moments(self.x_k).mean(0)]

        eta_k_list   = []
        theta_k_list = []
        dH_k_list    = []

        for k, t_k in tqdm(enumerate(self.t[:-1])):

            #Fiting
            self.fit(self.x_k)
            
            
            self.x_k, I_k, eta_k, theta_k, dH_k = self.iteration_step_projection(self.x_k, k)

            if (k + 1) % param_storage_frequency == 0:
                eta_k_list.append(eta_k)
                theta_k_list.append(theta_k)
                dH_k_list.append(dH_k)
                barphi_e.append(self.compute_moments(I_k).mean(0))
                barphi_p.append(self.compute_moments(self.x_k).mean(0))

        # Store final parameters and statistics
        eta_k_list.append(eta_k)
        theta_k_list.append(theta_k)
        dH_k_list.append(dH_k)
        barphi_e.append(self.compute_moments(I_k).mean(0))
        barphi_p.append(self.compute_moments(self.x_k).mean(0))

        return (
            self.x_k,
            torch.stack(barphi_e),
            torch.stack(barphi_p),
            torch.stack(eta_k_list),
            torch.stack(theta_k_list),
            torch.cat(dH_k_list),
        )


    def iteration_step_projection(self, x_k, k):
        """
        Perform a single predictor-corrector SDE integration step.
 
        Predictor::
 
            eta_k = solve  G(x_k) eta = d/dt phi_bar(I_t)
            y_k   = x_k + h * (grad phi)^T eta_k + sqrt(2 h) * sigma * noise
 
        Corrector (toward the moments at t_{k+1})::
 
            theta_k = solve  G(y_k) theta = phi_bar(I_{k+1}) - phi_bar(y_k)
            x_{k+1} = y_k + (grad phi)^T theta_k
 
        ``theta_k`` is then rescaled by ``h * sigma**2`` (zeroed if ``sigma == 0``),
        and the entropy increment is ``dH_k = - theta_k^T d/dt phi_bar(I_t)``.
 
        Parameters
        ----------
        x_k : torch.Tensor
            Current walker ensemble.
        k : int
            Index of the current time step (``h = t[k+1] - t[k]``).
 
        Returns
        -------
        x_k_plus_one : torch.Tensor
            Updated walker ensemble after predictor and corrector.
        I_k : torch.Tensor
            Interpolant evaluated at ``t[k+1]`` (the corrector target).
        eta_k : torch.Tensor
            Predictor coefficients, shape (m,).
        theta_k : torch.Tensor
            Corrector coefficients, normalized by ``h * sigma**2`` (zeroed if
            ``sigma == 0``), shape (m,).
        dH_k : torch.Tensor
            Entropy increment for this step.
        """


        h = self.t[k + 1] - self.t[k]

        # Predictor
        eta_k       = self.compute_eta(x_k, k)
        drift       = self.compute_grad_phi_projected(x_k, eta_k)
        noise_scale = (2 * h) ** 0.5 * self.sigma
        noise       = noise_scale * torch.randn_like(x_k).to(self.device)
        y_k         = x_k + h * drift + noise

        # Corrector
        theta_k        = self.compute_theta(y_k, k)
        corrector      = self.compute_grad_phi_projected(y_k, theta_k)
        x_k_plus_one   = y_k + corrector

        # Normalise theta
        if self.sigma > 0:
            theta_k = theta_k / (h * self.sigma ** 2)
        else:
            theta_k = torch.zeros_like(theta_k)

        # Entropy estimate
        I_k          = self.compute_interpolant(k + 1)
        dt_phi_I_k   = self.compute_rhs_dt_phi_I_t(I_k, k)
        dH_k         = -theta_k @ dt_phi_I_k

        return x_k_plus_one, I_k, eta_k, theta_k, dH_k


    # ------------------------------------------------------------------------------------------------------------------
    # Intermediate steps (in call order)
    # ------------------------------------------------------------------------------------------------------------------

    def compute_eta(self, x_k, k):
        """
        Solve for the predictor coefficients ``eta_k``.
 
        Solves ``G(x_k) eta = d/dt phi_bar(I_t)``, where the right-hand side is the
        moment velocity of the interpolant at step ``k``. Before the solve the Gram
        matrix is rescaled by its diagonal ``D = diag(G)`` to ``D^{-1/2} G D^{-1/2}``,
        symmetrized, and has ``regularization`` added to its diagonal; the solution
        is rescaled back by ``D^{-1/2}``.
 
        Parameters
        ----------
        x_k : torch.Tensor
            Current walker ensemble (defines ``G``).
        k : int
            Current time-step index.
 
        Returns
        -------
        eta_k : torch.Tensor
            Predictor coefficients, one per potential, shape (m,).
        """
 


        I_k              = self.compute_interpolant(k)
        rhs_dt_phi_I_k   = self.compute_rhs_dt_phi_I_t(I_k, k)
        G_k              = self.compute_G(x_k)
        
        ################################Regularization###################################
        D_k12 = torch.diag(G_k)**0.5 
        G_k = G_k /(D_k12[:,None]*D_k12[None,:])
        G_k = (G_k+G_k.T)/2
        G_k+= self.regularization*torch.eye(G_k.shape[-1],).to(G_k.dtype).to(G_k.device)
        
        eta_k = torch.linalg.solve(G_k, rhs_dt_phi_I_k/D_k12[:,None])[:, 0]
        eta_k = eta_k/D_k12

        return eta_k


    def compute_grad_phi_projected(self, x, vector):
        """
        Evaluate the field ``sum_i vector_i * grad phi_i(x)``.
 
        The signal-space drift induced by a coefficient vector. Mini-batched over the
        sample axis to bound memory, with results concatenated into a tensor matching
        ``x``'s signal layout.
 
        Parameters
        ----------
        x : torch.Tensor
            Samples at which to evaluate.
        vector : torch.Tensor
            Coefficients (length r), e.g. ``eta_k`` or ``theta_k``.
 
        Returns
        -------
        grad_phi_eta : torch.Tensor
            Field with the same per-sample shape as ``x``.
        """
 


        batch_size  = self.batch_size
        num_samples = x.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        if self.signal_dim == 0:
            grad_phi_eta = torch.zeros((0, self.original_signal_shape[1]), device=self.device)
        elif self.signal_dim == 1:
            grad_phi_eta = torch.zeros((0, self.original_signal_shape[1], self.original_signal_shape[2]), device=self.device)
        elif self.signal_dim == 2:
            grad_phi_eta = torch.zeros((0, self.original_signal_shape[1], self.original_signal_shape[2], self.original_signal_shape[3]), device=self.device)

        for idx_batch in range(num_batches):
            batch                = x[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            grad_phi_eta_batch   = self.compute_grad_potentials(batch, vector)
            grad_phi_eta         = torch.cat([grad_phi_eta, grad_phi_eta_batch], dim=0)

        return grad_phi_eta


    def compute_theta(self, y_k, k):
        """
        Solve for the corrector coefficients ``theta_k``.
 
        Solves ``G(y_k) theta = phi_bar(I_{k+1}) - phi_bar(y_k)``, the moment
        mismatch between the interpolant target at ``t[k+1]`` and the predicted
        walkers. Uses the same diagonal rescaling, symmetrization and regularization
        as :meth:`compute_eta`.
 
        Parameters
        ----------
        y_k : torch.Tensor
            Predicted walker ensemble (post-predictor).
        k : int
            Current time-step index (target is at ``k + 1``).
 
        Returns
        -------
        theta_k : torch.Tensor
            Corrector coefficients, one per potential, shape (m,).
        """


        I_k                      = self.compute_interpolant(k + 1)
        rhs_constraint_correction = self.compute_rhs_constraint_correction(y_k, I_k)
        G_k                      = self.compute_G(y_k)
        #return torch.linalg.solve(G_k, rhs_constraint_correction)
        
        ################################Regularization###################################
        D_k12 = torch.diag(G_k)**0.5 
        G_k = G_k /(D_k12[:,None]*D_k12[None,:])
        G_k = (G_k+G_k.T)/2
        G_k+= self.regularization*torch.eye(G_k.shape[-1],).to(G_k.dtype).to(G_k.device)
        
        theta_k = torch.linalg.solve(G_k, rhs_constraint_correction/D_k12)
        theta_k = theta_k/D_k12
        
        return theta_k

        


    def compute_rhs_dt_phi_I_t(self, I_k, k):
        """
        Compute the moment velocity of the interpolant, ``d/dt phi_bar(I_t)``.
 
        Evaluates ``mean_n <grad phi(I_t,n), I_dot_t,n>`` over the interpolant
        samples, with ``I_dot_t`` from :meth:`gradient_interpolant`. Mini-batched over
        the sample axis; the spatial contraction is handled per ``signal_dim``.
 
        Parameters
        ----------
        I_k : torch.Tensor
            Interpolant samples at the current step.
        k : int
            Current time-step index (selects ``I_dot_t``).
 
        Returns
        -------
        rhs : torch.Tensor
            Moment velocity, shape (m, 1). Right-hand side of the ``eta`` system and
            the contraction term in the entropy increment.
        """


        batch_size  = self.batch_size
        num_samples = I_k.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        rhs      = torch.zeros((self.num_potentials, 1)).to(self.device)
        I_k_dot  = self.gradient_interpolant(k)

        for idx_batch in range(num_batches):
            batch        = I_k[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            I_k_dot_batch = I_k_dot[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            grad_potential = self.compute_grad_potentials(batch)

            if self.signal_dim == 0:
                rhs += torch.matmul(grad_potential, I_k_dot_batch.reshape(batch.shape[0], 1, 1)).sum(0)
            elif self.signal_dim == 1:
                rhs += torch.matmul(grad_potential, I_k_dot_batch.reshape(batch.shape[0], batch.shape[-1], 1)).sum(0)
            elif self.signal_dim == 2:
                rhs += torch.matmul(
                    grad_potential.reshape(batch.shape[0], self.num_potentials, batch.shape[-2] * batch.shape[-1]),
                    I_k_dot_batch.reshape(batch.shape[0], batch.shape[-2] * batch.shape[-1], 1),
                ).sum(0)

        rhs /= num_samples

        return rhs


    def compute_G(self, x):
        """
        Compute the Gram matrix of the potential gradients.
 
        ``G_ij = mean_n <grad phi_i(x_n), grad phi_j(x_n)>``, accumulated in
        mini-batches via ``grad_potential @ grad_potential^T`` (spatial dimensions
        flattened for 2D signals) and averaged over the samples. ``G`` is the matrix
        solved against in the predictor and corrector.
 
        Parameters
        ----------
        x : torch.Tensor
            Samples defining ``G``.
 
        Returns
        -------
        G : torch.Tensor
            Symmetric matrix, shape (r, r).
        """


        batch_size  = self.batch_size
        num_samples = x.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        G = torch.zeros((self.num_potentials, self.num_potentials)).to(self.device)

        for idx_batch in range(num_batches):
            batch          = x[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            grad_potential = self.compute_grad_potentials(batch)

            if self.signal_dim == 2:
                grad_potential = grad_potential.reshape(batch.shape[0], self.num_potentials, batch.shape[-2] * batch.shape[-1])

            G += torch.bmm(grad_potential, grad_potential.transpose(1, 2)).sum(0)

        G /= num_samples

        return G


    def compute_grad_potentials(self, x, vector=None):
        """
        Evaluate potential gradients, optionally contracted with a coefficient vector.
 
        Two modes:
 
        - ``vector is None``: concatenates ``grad phi_i(x)`` over all potentials and
          reshapes to (B, m, *signal_shape). Used to assemble ``G`` and the moment
          velocity.
        - ``vector`` given: returns ``sum_i grad phi_i(x) @ vector[block_i]`` as a
          single field shaped like ``x``, slicing ``vector`` per potential via
          ``indices_potentials``.
 
        Parameters
        ----------
        x : torch.Tensor
            Samples at which to evaluate gradients.
        vector : torch.Tensor, optional
            Coefficients (length r). If provided, returns the contracted field;
            otherwise returns the stacked gradients.
 
        Returns
        -------
        torch.Tensor
            Stacked gradients (B, m, *signal_shape) when ``vector is None``, otherwise
            the contracted field with the same per-sample shape as ``x``.
        """


        if vector is None:
            grad_potential = torch.tensor([], device=x.device)
            for potential in self.potentials.values():
                grad_potential = torch.cat((grad_potential, potential.grad(x)), dim=1).detach()

            if self.signal_dim == 0:
                grad_potential = grad_potential.reshape(x.shape[0], self.num_potentials, 1)
            elif self.signal_dim == 1:
                grad_potential = grad_potential.reshape(x.shape[0], self.num_potentials, x.shape[-1])
            elif self.signal_dim == 2:
                grad_potential = grad_potential.reshape(x.shape[0], self.num_potentials, x.shape[-2], x.shape[-1])

            return grad_potential

        else:
            grad_phi_eta = torch.zeros_like(x)
            for i, potential in enumerate(self.potentials.values()):
                grad_phi_eta += potential.grad(x, v=vector[self.indices_potentials[i]:self.indices_potentials[i + 1]])

            return grad_phi_eta


    def compute_rhs_constraint_correction(self, x_k, I_k):
        """
        Compute the moment mismatch driving the corrector.
 
        Returns ``phi_bar(I_k) - phi_bar(x_k)``, the difference between the target
        moments and the current walker moments. Right-hand side of the ``theta``
        system in :meth:`compute_theta`.
 
        Parameters
        ----------
        x_k : torch.Tensor
            Current (predicted) walkers.
        I_k : torch.Tensor
            Interpolant target samples.
 
        Returns
        -------
        torch.Tensor
            Moment mismatch vector, length m.
        """
 

        bar_phi_I_k      = self.compute_moments(I_k).mean(0)
        bar_phi_x_current = self.compute_moments(x_k).mean(0)

        return bar_phi_I_k - bar_phi_x_current


    def compute_moments(self, x):
        """
        Evaluate the per-sample potentials (moments).
 
        Concatenates ``potential.forward(x)`` over all potentials, mini-batched over
        the sample axis. Averaging over samples yields ``phi_bar(x)``.
 
        Parameters
        ----------
        x : torch.Tensor
            Samples to evaluate.
 
        Returns
        -------
        moments : torch.Tensor
            Per-sample moments, shape (N, r).
        """


        batch_size  = self.batch_size
        num_samples = x.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        moments = torch.zeros((0, self.num_potentials)).to(self.device)

        for idx_batch in range(num_batches):
            batch         = x[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            moments_batch = torch.tensor([], device=x.device)
            for potential in self.potentials.values():
                moments_batch = torch.cat((moments_batch, potential.forward(batch)), dim=1)
            moments = torch.cat([moments, moments_batch], dim=0)

        return moments


    # ------------------------------------------------------------------------------------------------------------------
    # Interpolant
    # ------------------------------------------------------------------------------------------------------------------

    def compute_interpolant(self, k):
        """
        Evaluate the interpolant ``I(t_k)`` between noise ``x_0`` and data ``x_1``.
 
        Schedules:
 
        - ``'Linear'``      : (1 - t) x_0 + t x_1
        - ``'VarPreserv'``  : sqrt(1 - t) x_0 + sqrt(t) x_1
        - ``'Sqrt'``        : (1 - sqrt(t)) x_0 + sqrt(t) x_1
        - ``'Cos'``         : cos(pi t / 2) x_0 + sin(pi t / 2) x_1
 
        Parameters
        ----------
        k : int
            Time-step index into ``self.t``.
 
        Returns
        -------
        torch.Tensor
            Interpolant samples at ``t[k]``.
        """
 


        t, x_0, x_1 = self.t, self.x_0, self.x_1

        match self.interpolant:
            case 'Linear':
                return (1 - t[k]) * x_0 + t[k] * x_1
            case 'VarPreserv':
                return np.sqrt(1 - t[k]) * x_0 + np.sqrt(t[k]) * x_1
            case 'Sqrt':
                return (1 - np.sqrt(t[k])) * x_0 + np.sqrt(t[k]) * x_1
            case 'Cos':
                return np.cos(np.pi * t[k] / 2) * x_0 + np.sin(np.pi * t[k] / 2) * x_1


    def gradient_interpolant(self, k):
        """
        Evaluate the time derivative of the interpolant, ``I_dot(t_k)``.
 
        Returns the derivative matching the schedule of :meth:`compute_interpolant`.
        Used as the contraction vector in :meth:`compute_rhs_dt_phi_I_t`.
 
        Parameters
        ----------
        k : int
            Time-step index into ``self.t``.
 
        Returns
        -------
        torch.Tensor
            Interpolant time derivative at ``t[k]``.
        """
 


        t, x_0, x_1 = self.t, self.x_0, self.x_1

        match self.interpolant:
            case 'Linear':
                return x_1 - x_0
            case 'VarPreserv':
                return x_1 / (2 * np.sqrt(t[k])) - x_0 / (2 * np.sqrt(1 - t[k]))
            case 'Sqrt':
                return (x_1 - x_0) / (2 * np.sqrt(t[k]))
            case 'Cos':
                return (np.pi / 2) * (-np.sin(np.pi * t[k] / 2) * x_0 + np.cos(np.pi * t[k] / 2) * x_1)
    # ------------------------------------------------------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------------------------------------------------------
    
    def fit(self,x_k):
        """
        Refit any potentials that implement ``fit`` on the current samples.
 
        Calls ``potential.fit(x_k)`` on every potential, silently skipping those that
        do not implement (or raise inside) ``fit``.
 
        Parameters
        ----------
        x_k : torch.Tensor
            Current samples to refit on.
        """

        for pot in self.potentials.values():
                try:
                    pot.fit(x_k)
                except:
                    pass

class Adaptive_Langevin(SDE):
    """
    Adaptive Langevin sampler with moment correction.
 
    Drops the interpolant path: each step takes a pure noise predictor step and then
    corrects the walkers toward the fixed moments ``phi_bar(x_1)`` using the same
    projected correction as the parent class. The corrector coefficients ``theta_k``
    are recorded along the run.
 
    Parameters
    ----------
    x_1 : torch.Tensor
        Data samples; also used to initialize the walkers ``x_k``.
    potentials : dict[str, object]
        Named potential objects.
    nt : int
        Number of Langevin steps.
    delta_t : float
        Step size ``h``.
    batch_size : int
        Mini-batch size for the per-sample reductions.
    device : str, optional
        Torch device, by default ``'cpu'``.
    regularization : float, optional
        Value added to the diagonal of the (rescaled) Gram matrix, by default 0.
 
    Notes
    -----
    ``sigma`` is fixed to 1. The constructor calls ``torch.nn.Module.__init__``
    directly and reuses the parent's helpers for moments, gradients and ``G``.
    """


    def __init__(
        self,
        x_1,
        potentials,
        nt,
        delta_t,
        batch_size,
        device='cpu',
        regularization=0,

    
    ):
        torch.nn.Module.__init__(self)#super().__init__()

        self.x_1 = x_1
        self.original_signal_shape = self.x_1.shape

        self.nt             = nt
        self.delta_t         = delta_t
        self.sigma           = 1
        self.potentials      = potentials
        self.batch_size      = batch_size
        self.device          = device
        self.regularization  = regularization
        
        #Initialize
        self.x_k             = x_1

        match len(self.x_1.shape):
            case 2:
                self.signal_dim = 0
                print(f'Signal detected as scalar: (B, C) = ({batch_size}, {self.original_signal_shape[1]}).')
            case 3:
                self.signal_dim = 1
                print(f'Signal detected as 1D: (B, C, T) = ({batch_size}, {self.original_signal_shape[1]}, {self.original_signal_shape[2]}).')
            case 4:
                self.signal_dim = 2
                print(f'Signal detected as 2D: (B, C, M, N) = ({batch_size}, {self.original_signal_shape[1]}, {self.original_signal_shape[2]}, {self.original_signal_shape[3]}).')
        


        list_potential_num_coefficients = [p.num_coefficients for p in self.potentials.values()]
        self.num_potentials    = sum(list_potential_num_coefficients)
        self.indices_potentials = np.cumsum([0] + list_potential_num_coefficients)
        print(f'The model has {self.num_potentials} potentials.')


    # ------------------------------------------------------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------------------------------
    # Main function
    # ------------------------------------------------------------------------------------------------------------------

    def forward(self, param_storage_frequency=1):
        """
        Run the adaptive Langevin loop for ``nt`` steps.
 
        Fits the potentials and steps the walkers with
        :meth:`iteration_step_projection`, recording the corrector coefficients and
        the walker moments at the chosen cadence.
 
        Parameters
        ----------
        param_storage_frequency : int, optional
            Store diagnostics every ``param_storage_frequency`` steps, by default 1.
 
        Returns
        -------
        x_k : torch.Tensor
            Final walker ensemble.
        barphi_e : torch.Tensor
            Target moments ``phi_bar(x_1)`` (length r), constant over the run.
        barphi_p : torch.Tensor
            Walker moments stacked over stored steps, shape (num_stored, r).
        theta_k_list : torch.Tensor
            Corrector coefficients at the stored steps, shape (num_stored, r).
        """


        #Fiting
        self.fit(self.x_k)

        barphi_e = self.compute_moments(self.x_1).mean(0)
        barphi_p = [self.compute_moments(self.x_k).mean(0)]

        theta_k_list = []

        for k in tqdm(range(self.nt)):

            #Fiting
            self.fit(self.x_k)
            
            self.x_k, theta_k = self.iteration_step_projection(self.x_k, k)

            if (k + 1) % param_storage_frequency == 0:
                theta_k_list.append(theta_k)
                
                barphi_p.append(self.compute_moments(self.x_k).mean(0))

        # Store final parameters and statistics
        theta_k_list.append(theta_k)
        barphi_p.append(self.compute_moments(self.x_k).mean(0))

        return (
            self.x_k,
            barphi_e,
            torch.stack(barphi_p),
            torch.stack(theta_k_list),
        )


    def iteration_step_projection(self, x_k, k):
        """
        Perform one adaptive Langevin step (noise predictor + moment corrector).
 
        Predictor::
 
            y_k = x_k + sqrt(2 h) * noise
 
        Corrector (toward the fixed moments ``phi_bar(x_1)``)::
 
            theta_k = solve  G(y_k) theta = phi_bar(x_1) - phi_bar(y_k)
            x_{k+1} = y_k + (grad phi)^T theta_k
 
        The returned ``theta_k`` is rescaled by ``h``.
 
        Parameters
        ----------
        x_k : torch.Tensor
            Current walker ensemble.
        k : int
            Step index (``h = delta_t`` is constant).
 
        Returns
        -------
        x_k_plus_one : torch.Tensor
            Updated walker ensemble.
        theta_k : torch.Tensor
            Corrector coefficients, normalized by ``h``, shape (m,).
        """


        h = self.delta_t

        # Predictor
        noise_scale = (2 * h) ** 0.5
        noise       = noise_scale * torch.randn_like(x_k).to(self.device)
        y_k         = x_k +  noise

        # Corrector
        theta_k        = self.compute_theta(y_k, k)
        corrector      = self.compute_grad_phi_projected(y_k, theta_k)
        x_k_plus_one   = y_k + corrector

        # Normalise theta
        theta_k = theta_k / h 
   

        return x_k_plus_one,  theta_k 

    def compute_theta(self, y_k, k):
        
        rhs_constraint_correction = self.compute_rhs_constraint_correction(y_k, self.x_1)
        G_k                      = self.compute_G(y_k)
        #return torch.linalg.solve(G_k, rhs_constraint_correction)
        
        ################################Regularization###################################
        D_k12 = torch.diag(G_k)**0.5 
        G_k = G_k /(D_k12[:,None]*D_k12[None,:])
        G_k = (G_k+G_k.T)/2
        G_k+= self.regularization*torch.eye(G_k.shape[-1],).to(G_k.dtype).to(G_k.device)
        
        theta_k = torch.linalg.solve(G_k, rhs_constraint_correction/D_k12)
        theta_k = theta_k/D_k12
        
        return theta_k


