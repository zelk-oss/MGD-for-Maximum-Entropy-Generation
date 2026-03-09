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
from utils import *


class SDE(torch.nn.Module):
    """
    Implements Algorithm 1 of the paper.

    Structure:
        - Initialization
        - forward: main loop integrating the SDE
        - iteration_step_projection: single SDE integration step
        - Intermediate functions called within one integration step (in call order)
        - Interpolant utilities
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
        regularization=(0, 0, 0),
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

        self.init_interpolants_and_workers(x_0, x_1, x_k)

        list_potential_num_coefficients = [p.num_coefficients for p in self.potentials.values()]
        self.num_potentials    = sum(list_potential_num_coefficients)
        self.indices_potentials = np.cumsum([0] + list_potential_num_coefficients)
        print(f'The model has {self.num_potentials} potentials.')


    # ------------------------------------------------------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------------------------------------------------------

    def init_interpolants_and_workers(self, x_0, x_1, x_k):
        """Initialize workers and interpolant to random noise or to constructor-supplied values."""

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
        """Integrate the SDE by looping over time and calling iteration_step_projection at each step."""

        barphi_e = [self.compute_moments(self.x_0).mean(0)]
        barphi_p = [self.compute_moments(self.x_k).mean(0)]

        eta_k_list   = []
        theta_k_list = []
        dH_k_list    = []

        for k, t_k in tqdm(enumerate(self.t[:-1])):

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
        """Perform a single SDE integration step."""

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

        I_k              = self.compute_interpolant(k)
        rhs_dt_phi_I_k   = self.compute_rhs_dt_phi_I_t(I_k, k)
        G_k              = self.compute_G(x_k)
        eta_k            = torch.linalg.solve(G_k, rhs_dt_phi_I_k)[:, 0]

        return eta_k


    def compute_grad_phi_projected(self, x, vector):

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

        I_k                      = self.compute_interpolant(k + 1)
        rhs_constraint_correction = self.compute_rhs_constraint_correction(y_k, I_k)
        G_k                      = self.compute_G(y_k)

        return torch.linalg.solve(G_k, rhs_constraint_correction)


    def compute_rhs_dt_phi_I_t(self, I_k, k):

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

        D       = torch.diag(G)
        ind     = torch.where(D < self.regularization[1])[0]
        D[ind] += self.regularization[2]
        D       = self.regularization[0] * torch.diag(D)

        return G + D


    def compute_grad_potentials(self, x, vector=None):

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

        bar_phi_I_k      = self.compute_moments(I_k).mean(0)
        bar_phi_x_current = self.compute_moments(x_k).mean(0)

        return bar_phi_I_k - bar_phi_x_current


    def compute_moments(self, x):

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
        """Evaluate the interpolant I(t_k)."""

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
        """Evaluate the time derivative of the interpolant at t_k."""

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