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
        #self.fit(self.x_k)
        self.fit(self.x_1)

        barphi_e = [self.compute_moments(self.x_0).mean(0)]
        barphi_p = [self.compute_moments(self.x_k).mean(0)]

        eta_k_list   = []
        theta_k_list = []
        dH_k_list    = []

        for k, t_k in tqdm(enumerate(self.t[:-1])):

            #Fiting
            #self.fit(self.x_k)
            
            
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

    def forward_regularised(self, lam=1.0, n_subsample=1, param_storage_frequency=1):
        """
        As forward, but stores the regularised-problem quantities ON THE WALKERS X_t = x_k
        and solves A Theta = f (section 4) on a coarse grid after the loop.

        Everything is tied to the walker. phi and grad phi are evaluated at X_t, and the
        data endpoint is reconstructed from the interpolant relation
            X = (X_t - cos(a_t) Z) / sin(a_t),     Z = x_0,
        so x_1 never enters. The space target simplifies to Z/cos. t = 0 is skipped
        (sin = 0). The SDE evolution is unchanged. Needs n_rep == nb_interpolants to pair
        X_t with Z.

        Coarse grid keeps the points t[n_subsample*j]; metrics and second members are
        averaged over the n_subsample neighbouring (valid) fine steps.
        """
        assert self.interpolant == 'Cos', "this routine assumes the Cos schedule"
        assert self.x_k.shape[0] == self.x_0.shape[0], "need n_rep == nb_interpolants to pair X_t with Z"
        self.fit(self.x_1)

        x0  = self.x_0
        B   = x0.shape[0]
        d   = int(np.prod(x0.shape[1:]))
        z2  = x0.reshape(B, -1).pow(2).sum(1)              # ||Z||^2  (fixed, Z = x_0)
        adot, eps = np.pi / 2.0, 1e-8
        eye = torch.eye(self.num_potentials).to(self.device)

        barphi_e = [self.compute_moments(self.x_0).mean(0)]
        barphi_p = [self.compute_moments(self.x_k).mean(0)]
        eta_k_list, theta_k_list, dH_k_list = [], [], []
        M, Gf, bb, cc, t_used = [], [], [], [], []
        accM = accG = accb = accc = None
        cnt = 0

        for k, t_k in tqdm(enumerate(self.t[:-1])):
            self.x_k, y_k, I_k, eta_k, theta_k, dH_k = self.iteration_step_projection(self.x_k, k)
            t_node = self.t[k + 1]
            ak = np.pi * t_node / 2.0
            cos_k, sin_k, tan_k = np.cos(ak), np.sin(ak), np.tan(ak)

            if sin_k > eps and cos_k > eps:                # skip t = 0 (sin = 0), t = 1 (cos = 0)
                Xt    = y_k                                           # predicted walkers at target time
                mom   = self.compute_moments(Xt)                       # phi(y_k)              (B, r)
                Mk    = self.compute_G(Xt) + self.regularization * eye # grad Gram at y_k
                Gk    = mom.T @ mom / B                                # moment Gram at y_k
                bk    = self.compute_rhs_constraint_correction(y_k, I_k)  # phi_bar(I_{k+1}) - phi_bar(y_k)
                X_eff = (Xt - cos_k * x0) / sin_k                      # X = (X_t - cos a Z)/sin a
                zx    = (x0 * X_eff).reshape(B, -1).sum(1)             # Z . X                 (B,)
                tau   = -adot * (tan_k * (d - z2) + zx)                # tau_k^i               (B,)
                ck    = torch.einsum('br,b->r', mom, tau) / B          # E[phi(y_k) tau]

                if cnt == 0:
                    t_used.append(t_node)                             # coarse node at target time
                    accM, accG, accb, accc = Mk.clone(), Gk.clone(), bk.clone(), ck.clone()
                else:
                    accM += Mk; accG += Gk; accb += bk; accc += ck
                cnt += 1
                if cnt == n_subsample:
                    M.append(accM / cnt); Gf.append(accG / cnt)
                    bb.append(accb / cnt); cc.append(accc / cnt)
                    cnt = 0

            if (k + 1) % param_storage_frequency == 0:
                eta_k_list.append(eta_k); theta_k_list.append(theta_k); dH_k_list.append(dH_k)
                barphi_e.append(self.compute_moments(I_k).mean(0))
                barphi_p.append(self.compute_moments(self.x_k).mean(0))

        if cnt > 0:                                                    # final partial block
            M.append(accM / cnt); Gf.append(accG / cnt)
            bb.append(accb / cnt); cc.append(accc / cnt)

        Theta_reg = self._solve_regularised(t_used, M, Gf, bb, cc, lam)

        return (
            self.x_k,
            torch.stack(barphi_e), torch.stack(barphi_p),
            torch.stack(eta_k_list), torch.stack(theta_k_list), torch.cat(dH_k_list),
            Theta_reg,
        )

    def _solve_regularised(self, t, M, Gf, bb, cc, lam):
        """
        Assemble and solve the block-tridiagonal system (section 4).

        t is the COARSE grid (the kept points t[n_subsample*j]), so dt = diff(t) is the
        coarse spacing -- the only Delta t entering the time coupling. The matrix is now
        (n_coarse * r)^2.
        """
        t = np.asarray(t, dtype=float)
        n, r, dev = len(t), self.num_potentials, self.device
        dt = np.diff(t)                                   # coarse spacing
        A = torch.zeros((n, r, n, r)).to(dev)
        f = torch.zeros((n, r)).to(dev)
        for k in range(n):
            A[k, :, k, :] += M[k]
            f[k]          += bb[k]
        for k in range(n - 1):
            w = lam / dt[k] ** 2
            A[k,     :, k,     :] += w * Gf[k]
            A[k + 1, :, k + 1, :] += w * Gf[k]
            A[k,     :, k + 1, :] -= w * Gf[k]
            A[k + 1, :, k,     :] -= w * Gf[k]
            f[k]     -= (lam / dt[k]) * cc[k]
            f[k + 1] += (lam / dt[k]) * cc[k]
        return torch.linalg.solve(A.reshape(n * r, n * r), f.reshape(n * r)).reshape(n, r)


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
        y_k : torch.Tensor
            Predicted walker ensemble after the predictor step.
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

        return x_k_plus_one, y_k, I_k, eta_k, theta_k, dH_k


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

    
    # methods for regularised theta 
    def _grad_contract(self, S, V):
        """E_n[ grad phi(S_n) . V_n ] -> (r,). Same contraction as
        compute_rhs_dt_phi_I_t, but with an arbitrary per-sample field V."""
        bs, num = self.batch_size, S.shape[0]
        nb = (num + bs - 1) // bs
        out = torch.zeros((self.num_potentials, 1)).to(self.device)
        for ib in range(nb):
            sb = S[ib * bs:(ib + 1) * bs]
            vb = V[ib * bs:(ib + 1) * bs]
            gp = self.compute_grad_potentials(sb)               # (B, r, *signal)
            if self.signal_dim == 0:
                out += torch.matmul(gp, vb.reshape(sb.shape[0], 1, 1)).sum(0)
            elif self.signal_dim == 1:
                out += torch.matmul(gp, vb.reshape(sb.shape[0], sb.shape[-1], 1)).sum(0)
            elif self.signal_dim == 2:
                out += torch.matmul(
                    gp.reshape(sb.shape[0], self.num_potentials, sb.shape[-2] * sb.shape[-1]),
                    vb.reshape(sb.shape[0], sb.shape[-2] * sb.shape[-1], 1)).sum(0)
        return (out / num).squeeze(1)                           # (r,)


    def regularised_theta(self, lam=1.0, t=None):
        assert self.interpolant == 'Cos', "this routine assumes the Cos schedule"
        t = self.t if t is None else t
        t = np.asarray(t, dtype=float)
        if np.isclose(t[-1], 1.0):
            t = t[:-1]                                          # drop data endpoint (cos = 0)
        n, r, dev = len(t), self.num_potentials, self.device

        x0, x1 = self.x_0, self.x_1
        B    = x0.shape[0]
        d    = int(np.prod(x0.shape[1:]))                       # ambient dimension
        z2   = x0.reshape(B, -1).pow(2).sum(1)                  # ||Z||^2   (B,)
        zx   = (x0 * x1).reshape(B, -1).sum(1)                  # Z . X     (B,)
        adot = np.pi / 2.0                                      # d alpha / dt (Cos)
        eye  = torch.eye(r).to(dev)

        M, Gf, bb, cc = [], [], [], []
        for tk in t:
            ak = np.pi * tk / 2.0
            cos_k, tan_k = np.cos(ak), np.tan(ak)
            I_k = np.cos(ak) * x0 + np.sin(ak) * x1             # interpolant at t_k
            moments = self.compute_moments(I_k)                 # (B, r)
            M.append(self.compute_G(I_k) + self.regularization * eye)   # grad Gram (doc M_k)
            Gf.append(moments.T @ moments / B)                  # moment Gram (doc G_k)
            bb.append(self._grad_contract(I_k, x0) / cos_k)     # b_k = E[grad phi . Z]/cos
            tau = -adot * (tan_k * (d - z2) + zx)               # (B,)
            cc.append(torch.einsum('br,b->r', moments, tau) / B)  # c_k = E[phi tau]

        dt = np.diff(t)
        A  = torch.zeros((n, r, n, r)).to(dev)
        f  = torch.zeros((n, r)).to(dev)
        for k in range(n):
            A[k, :, k, :] += M[k]
            f[k]          += bb[k]
        for k in range(n - 1):
            w = lam / dt[k] ** 2
            A[k,     :, k,     :] += w * Gf[k]
            A[k + 1, :, k + 1, :] += w * Gf[k]
            A[k,     :, k + 1, :] -= w * Gf[k]
            A[k + 1, :, k,     :] -= w * Gf[k]
            f[k]     -= (lam / dt[k]) * cc[k]
            f[k + 1] += (lam / dt[k]) * cc[k]

        Theta = torch.linalg.solve(A.reshape(n * r, n * r), f.reshape(n * r))
        return Theta.reshape(n, r)


    # windowing approximation 
    def max_window_for_budget(r, budget_gb=32, dtype_bytes=8):
        budget = budget_gb * 1e9
        return int((budget / (dtype_bytes * r**2)) ** 0.5)
    
    def regularised_theta_chunked(self, lam=1.0, t=None, chunk_size=60, halo=20,
                                eps_reg_theta=1e-6, use_float64=True):
        """
        Windowed version of regularised_theta. Solves the same block-tridiagonal
        smoothing system, but the dense reshape-and-solve is done on small,
        overlapping windows of the FULL-resolution grid rather than on all n
        points at once.

        Every M_k, Gf_k, b_k, c_k is still built from the exact entries of t
        (no subsampling) -- chunking only bounds how many adjacent steps are
        coupled in one dense solve. Each window is (chunk_size + 2*halo) points;
        only the central chunk_size "core" is kept (the halo gives that core
        correct boundary information from its true neighbours -- without it,
        every chunk edge would behave like an artificial free boundary and you'd
        see a kink at every seam).

        Memory per window: O((chunk_size+2*halo)^2 * r^2), independent of n.
        This is an approximation relative to the single global solve: coupling
        beyond the halo is dropped. halo=0 recovers fully decoupled, seam-prone
        chunks; growing halo trades cost for fidelity to the true global
        smoother.

        Parameters mirror regularised_theta; chunk_size/halo control the window,
        eps_reg_theta is scaled by each window's own diagonal magnitude rather
        than used as a fixed absolute constant (the lam/dt**2 coupling weight
        can vary by orders of magnitude with resolution, so an absolute eps
        isn't portable across grids -- see earlier debugging in this thread).
        """
        assert self.interpolant == 'Cos', "this routine assumes the Cos schedule"
        t_full = self.t if t is None else t
        t_full = np.asarray(t_full, dtype=float)
        if np.isclose(t_full[-1], 1.0):
            t_full = t_full[:-1]
        n, r, dev = len(t_full), self.num_potentials, self.device
        solve_dtype = torch.float64 if use_float64 else self.x_1.dtype

        W_max = chunk_size + 2 * halo
        est_bytes = (W_max ** 2) * (r ** 2) * (8 if use_float64 else 4)
        print(f'window size W={W_max}, r={r} -> dense block ~{est_bytes/1e9:.1f} GB per window')

        x0, x1 = self.x_0, self.x_1
        B    = x0.shape[0]
        d    = int(np.prod(x0.shape[1:]))
        z2   = x0.reshape(B, -1).pow(2).sum(1)
        zx   = (x0 * x1).reshape(B, -1).sum(1)
        adot = np.pi / 2.0

        # Per-step matrices, computed once over the full grid -- O(n) memory of
        # r x r matrices (cheap: r~500 -> ~2MB each, a few thousand of these is
        # fine). It's the COUPLED solve that's expensive, not storing these.
        M_all, Gf_all, bb_all, cc_all = [], [], [], []
        for tk in t_full:
            ak = np.pi * tk / 2.0
            cos_k, tan_k = np.cos(ak), np.tan(ak)
            I_k = np.cos(ak) * x0 + np.sin(ak) * x1
            moments = self.compute_moments(I_k)
            Mk  = (self.compute_G(I_k) + self.regularization * torch.eye(r, device=dev)).to(solve_dtype)
            Gfk = (moments.T @ moments / B).to(solve_dtype)
            bk  = (self._grad_contract(I_k, x0) / cos_k).to(solve_dtype)
            tau = -adot * (tan_k * (d - z2) + zx)
            ck  = (torch.einsum('br,b->r', moments, tau) / B).to(solve_dtype)
            M_all.append(Mk); Gf_all.append(Gfk); bb_all.append(bk); cc_all.append(ck)

        Theta_full = torch.zeros((n, r), dtype=solve_dtype, device=dev)

        core_start = 0
        while core_start < n:
            core_end = min(core_start + chunk_size, n)
            win_start = max(0, core_start - halo)
            win_end   = min(n, core_end + halo)

            idx    = range(win_start, win_end)
            Wn     = win_end - win_start
            t_win  = t_full[win_start:win_end]
            dt_win = np.diff(t_win)

            M  = [M_all[i]  for i in idx]
            Gf = [Gf_all[i] for i in idx]
            bb = [bb_all[i] for i in idx]
            cc = [cc_all[i] for i in idx]

            A = torch.zeros((Wn, r, Wn, r), dtype=solve_dtype, device=dev)
            f = torch.zeros((Wn, r), dtype=solve_dtype, device=dev)
            for k in range(Wn):
                A[k, :, k, :] += M[k]
                f[k]          += bb[k]
            for k in range(Wn - 1):
                w_k = lam / dt_win[k] ** 2
                A[k,     :, k,     :] += w_k * Gf[k]
                A[k + 1, :, k + 1, :] += w_k * Gf[k]
                A[k,     :, k + 1, :] -= w_k * Gf[k]
                A[k + 1, :, k,     :] -= w_k * Gf[k]
                f[k]     -= (lam / dt_win[k]) * cc[k]
                f[k + 1] += (lam / dt_win[k]) * cc[k]

            A_flat = A.reshape(Wn * r, Wn * r)
            scale  = A_flat.diagonal().abs().mean()              # scale-aware reg,
            A_flat = A_flat + eps_reg_theta * scale * torch.eye(  # not a fixed eps
                Wn * r, dtype=solve_dtype, device=dev)

            Theta_win = torch.linalg.solve(A_flat, f.reshape(Wn * r)).reshape(Wn, r)

            keep_lo = core_start - win_start
            keep_hi = keep_lo + (core_end - core_start)
            Theta_full[core_start:core_end] = Theta_win[keep_lo:keep_hi]

            core_start = core_end

        return Theta_full.to(self.x_1.dtype)


    # block thomas approximation 
    def regularised_theta_thomas(self, lam=1.0, eps_reg_theta = 1e-6, t=None):
        """Same system as regularised_theta, but solved via block-tridiagonal
        (block Thomas) elimination instead of a dense (n*r, n*r) solve.
        Memory: O(n * r**2) instead of O(n**2 * r**2) -> works at full
        resolution (n = len(self.t)) without needing to subsample t."""
        assert self.interpolant == 'Cos', "this routine assumes the Cos schedule"
        t = self.t if t is None else t
        t = np.asarray(t, dtype=float)
        if np.isclose(t[-1], 1.0):
            t = t[:-1]
        n, r, dev = len(t), self.num_potentials, self.device

        x0, x1 = self.x_0, self.x_1
        B    = x0.shape[0]
        d    = int(np.prod(x0.shape[1:]))
        z2   = x0.reshape(B, -1).pow(2).sum(1)
        zx   = (x0 * x1).reshape(B, -1).sum(1)
        adot = np.pi / 2.0
        eye  = torch.eye(r).to(dev)

        M, Gf, bb, cc = [], [], [], []
        for tk in t:
            ak = np.pi * tk / 2.0
            cos_k, tan_k = np.cos(ak), np.tan(ak)
            I_k = np.cos(ak) * x0 + np.sin(ak) * x1
            moments = self.compute_moments(I_k)
            M.append(self.compute_G(I_k) + self.regularization * eye)
            Gf.append(moments.T @ moments / B)
            bb.append(self._grad_contract(I_k, x0) / cos_k)
            tau = -adot * (tan_k * (d - z2) + zx)
            cc.append(torch.einsum('br,b->r', moments, tau) / B)

        dt = np.diff(t)
        w  = [lam / dk ** 2 for dk in dt]

        D = [M[k].clone() for k in range(n)]
        for k in range(n - 1):
            D[k]     = D[k]     + w[k] * Gf[k]
            D[k + 1] = D[k + 1] + w[k] * Gf[k]
        U = [-w[k] * Gf[k] for k in range(n - 1)]
        L = [Uk.transpose(0, 1) for Uk in U]

        f = [bb[k].clone() for k in range(n)]
        for k in range(n - 1):
            f[k]     = f[k]     - (lam / dt[k]) * cc[k]
            f[k + 1] = f[k + 1] + (lam / dt[k]) * cc[k]

        c_prime = [None] * max(n - 1, 0)
        d_prime = [None] * n

        if n > 1:
            sol0 = torch.linalg.solve(D[0], torch.cat([U[0], f[0][:, None]], dim=1))
            c_prime[0], d_prime[0] = sol0[:, :-1], sol0[:, -1]
        else:
            d_prime[0] = torch.linalg.solve(D[0], f[0])

        for k in range(1, n):
            denom = D[k] - L[k - 1] @ c_prime[k - 1]
            rhs   = f[k] - L[k - 1] @ d_prime[k - 1]

            eps = eps_reg_theta  # Adjust this if you still get errors (e.g., 1e-5 or 1e-4)
            denom = denom + eps * torch.eye(denom.shape[0], device=denom.device)            
            if k < n - 1:
                sol = torch.linalg.solve(denom, torch.cat([U[k], rhs[:, None]], dim=1))
                c_prime[k], d_prime[k] = sol[:, :-1], sol[:, -1]
            else:
                d_prime[k] = torch.linalg.solve(denom, rhs)

        Theta = [None] * n
        Theta[-1] = d_prime[-1]
        for k in range(n - 2, -1, -1):
            Theta[k] = d_prime[k] - c_prime[k] @ Theta[k + 1]

        return torch.stack(Theta)