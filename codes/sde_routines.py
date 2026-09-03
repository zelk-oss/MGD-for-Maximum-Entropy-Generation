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

from pathlib import Path
current = Path.cwd()
MGD_project_folder = current.parent




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
    # 1) Add a small helper that (re)computes the potential-dimension bookkeeping
    # added because of generalized gaussian class pruning the number of potentials 
    #    from whatever num_coefficients the potentials currently report.
    def _sync_potential_dims(self):
        sizes = [p.num_coefficients for p in self.potentials.values()]
        self.num_potentials = sum(sizes)
        self.indices_potentials = np.cumsum([0] + sizes)

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
        regularization=1e-4,
        interpolant='Cos',
        x_0=None,
        x_k=None,
        use_coshgt_s0=True,
        potentials_save_dir=None,  
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
        self.use_coshgt_s0   = use_coshgt_s0

        self.init_interpolants_and_workers()
        
        # 1. Resolve where fitted-potential state gets saved for THIS run.
        if potentials_save_dir is not None:
            saved_results_dir = Path(potentials_save_dir)
        else:
            saved_results_dir = MGD_project_folder / "turbulence/saved_results"
            print(f"[SDE] WARNING: no potentials_save_dir given — falling back to the "
                f"shared global path {saved_results_dir}. This WILL be overwritten by "
                f"other/concurrent runs. Pass potentials_save_dir=<exp_dir>/'fitted_potentials'.")

        saved_results_dir.mkdir(parents=True, exist_ok=True)

        for name, pot in self.potentials.items():
            if not hasattr(pot, "fit"):
                continue
            state_path = saved_results_dir / f"{name}.pt"

            # load a previously fitted/pruned state instead of refitting, if present
            if hasattr(pot, "is_fitted") and hasattr(pot, "load_fixed_parameters") and state_path.exists():
                own_filters = pot.filters
                loaded = type(pot).load_fixed_parameters(state_path, own_filters, map_location=device)
                loaded.to(device)
                self.potentials[name] = loaded
                print(f"[SDE] loaded fitted state for '{name}' from {state_path} "
                    f"({loaded.num_coefficients} active statistics)")
                continue

            pot.fit(self.x_1)
            if hasattr(pot, "is_fitted"):
                pot.save_fixed_parameters(state_path)
                print(f"Fixed parameters for '{name}' have been saved in: {state_path}")

        self._sync_potential_dims()     # <-- reads post-prune sizes
        print(f'The model has {self.num_potentials} potentials.')

        list_potential_num_coefficients = [p.num_coefficients for p in self.potentials.values()]
        self.num_potentials    = sum(list_potential_num_coefficients)
        self.indices_potentials = np.cumsum([0] + list_potential_num_coefficients)
        print(f'The model has {self.num_potentials} potentials.')


    # ------------------------------------------------------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------------------------------------------------------

    def _print_memory(self, msg):
        import gc
        import psutil
        import os

        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss / 1024**3

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved  = torch.cuda.memory_reserved() / 1024**3
            print(f"{msg}: CPU={rss:.2f} GB | GPU alloc={allocated:.2f} GB | GPU reserved={reserved:.2f} GB")
        else:
            print(f"{msg}: CPU={rss:.2f} GB")

        gc.collect()


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

    def _check_time_budget(self, loop_t0, n_done, time_limit_min,
                            min_iters=30, margin=0.9):
        """
        Abort the SDE loop early if the average iteration time seen so far,
        extrapolated to the full loop, would blow through the SLURM time
        budget. No-op if `time_limit_min` is None (the default — opt-in per
        run) or before `min_iters` iterations have completed, since early
        iterations are slower (CUDA warm-up) and noisy.

        `margin` reserves headroom under `time_limit_min` for the work that
        happens after this loop (regularised solve, save_results I/O), which
        isn't accounted for by iteration count alone.
        """
        if time_limit_min is None or n_done < min_iters:
            return
        elapsed = time.time() - loop_t0
        avg_iter_s = elapsed / n_done
        total_iters = len(self.t) - 1
        projected_s = avg_iter_s * total_iters
        budget_s = time_limit_min * 60 * margin
        if projected_s > budget_s:
            raise RuntimeError(
                f"Aborting SDE loop: projected to take {projected_s / 3600:.2f}h "
                f"({avg_iter_s:.2f}s/it x {total_iters} it, measured over "
                f"{n_done} iterations) which exceeds {margin:.0%} of the "
                f"{time_limit_min / 60:.1f}h time budget (--time_limit_min "
                f"{time_limit_min}). Aborting now instead of letting SLURM kill "
                f"the job at the wall-clock limit with nothing saved."
            )

    def forward(self, param_storage_frequency=1, time_limit_min=None):
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
        #self.fit(self.x_1)
        #self._sync_potential_dims()  # update after possible potential pruning 

        barphi_e = [self.compute_moments(self.x_0).mean(0)]
        barphi_p = [self.compute_moments(self.x_k).mean(0)]

        eta_k_list   = []
        theta_k_list = []
        dH_k_list    = []

        loop_t0 = time.time()
        for k, t_k in tqdm(enumerate(self.t[:-1])):

            #Fiting
            #self.fit(self.x_k)


            self.x_k, y_k, I_k, eta_k, theta_k, dH_k, bk= self.iteration_step_projection(self.x_k, k)

            if (k + 1) % param_storage_frequency == 0:
                eta_k_list.append(eta_k)
                theta_k_list.append(theta_k)
                dH_k_list.append(dH_k)
                barphi_e.append(self.compute_moments(I_k).mean(0))
                barphi_p.append(self.compute_moments(self.x_k).mean(0))

            self._check_time_budget(loop_t0, k + 1, time_limit_min)

        # Store final parameters and statistics
        eta_k_list.append(eta_k)
        theta_k_list.append(theta_k)
        dH_k_list.append(dH_k)
        barphi_e.append(self.compute_moments(I_k).mean(0))
        barphi_p.append(self.compute_moments(self.x_k).mean(0))

        print("Finished time integration")
        self._print_memory("After loop")

        print("Stacking barphi_e")
        barphi_e = torch.stack(barphi_e)
        self._print_memory("After stacking barphi_e")

        print("Stacking barphi_p")
        barphi_p = torch.stack(barphi_p)
        self._print_memory("After stacking barphi_p")

        print("Stacking eta")
        eta_k_list = torch.stack(eta_k_list)
        self._print_memory("After stacking eta")

        print("Stacking theta")
        theta_k_list = torch.stack(theta_k_list)
        self._print_memory("After stacking theta")

        print("Concatenating dH")
        dH_k_list = torch.cat(dH_k_list)
        self._print_memory("After concatenating dH")

        print("Returning outputs")

        return (
            self.x_k,
            barphi_e,
            barphi_p,
            eta_k_list,
            theta_k_list,
            dH_k_list,
        )

    def _cut_close_time_nodes(self, t, M, Gf, bb, cc, min_dt=None):
        """
        Greedily keep only nodes at least min_dt apart (checked against the last
        KEPT node, not the previous raw node — a plain np.diff(t) > tol mask
        fails on runs of many close points because it doesn't re-check spacing
        after dropping). Points that violate the tolerance are simply cut.
        """
        t = np.asarray(t, dtype=np.float64)  # keep full precision, no float32 cast
        if min_dt is None:
            span = t[-1] - t[0] if len(t) > 1 else 1.0
            min_dt = max(span * 1e-5, 1e-6)

        keep = [0]
        for i in range(1, len(t)):
            if t[i] - t[keep[-1]] >= min_dt:
                keep.append(i)

        keep = np.asarray(keep)
        return (t[keep], [M[i] for i in keep], [Gf[i] for i in keep],
                [bb[i] for i in keep], [cc[i] for i in keep])

    def forward_regularised(self, lam=1.0, n_subsample=1, param_storage_frequency=1,
                             time_limit_min=None):
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
        #self.fit(self.x_1)
        #self._sync_potential_dims()  

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

  

        loop_t0 = time.time()
        for k, t_k in tqdm(enumerate(self.t[:-1])):
            h = self.t[k + 1] - self.t[k]
            self.x_k, y_k, I_k, eta_k, theta_k, dH_k, bk = self.iteration_step_projection(self.x_k, k)
            t_node = self.t[k + 1]
            ak = np.pi * t_node / 2.0
            cos_k, sin_k, tan_k = np.cos(ak), np.sin(ak), np.tan(ak)

            if sin_k > eps and cos_k > eps:                # skip t = 0 (sin = 0), t = 1 (cos = 0)
                Xt    = y_k                                           # predicted walkers at target time
                mom   = self.compute_moments(Xt)                       # phi(y_k)              (B, r)
                Mk    = self.compute_G(Xt)                             # raw Gram at y_k
                #if self.regularization != 0:
                #    Mk = Mk + self.regularization * torch.diag(torch.diag(Mk))

                Gk    = mom.T @ mom / B                                # moment Gram at y_k
                bk    = bk / (h * self.sigma**2)  # phi_bar(I_{k+1}) - phi_bar(y_k)
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

            self._check_time_budget(loop_t0, k + 1, time_limit_min)

        if cnt > 0:                                                    # final partial block
            M.append(accM / cnt); Gf.append(accG / cnt)
            bb.append(accb / cnt); cc.append(accc / cnt)

        #Theta_reg_thomas = self._solve_regularised_thomas(t_used[1:], M[1:], Gf[1:], bb[1:], cc[1:], lam)

        print("Loop finished")
        self._print_memory("After loop")

        print("Preparing regularised solve")
        self._print_memory("Before _solve_regularised")

        t_reg, M_reg, Gf_reg, bb_reg, cc_reg = self._cut_close_time_nodes(
            t_used[1:], M[1:], Gf[1:], bb[1:], cc[1:]
        )

        print("Dropped close-in-time nodes:", len(t_used) - 1 - len(t_reg))
        print("Last times:", t_reg[-5:])
        print("Last dt:", np.diff(t_reg[-5:]))

        Theta_reg = self._solve_regularised(t_reg, M_reg, Gf_reg, bb_reg, cc_reg, lam)

        self._print_memory("After _solve_regularised")

        print("Stacking outputs")

        # barphi_e/barphi_p start with one extra entry appended BEFORE the loop
        # (the t=0 seed, `compute_moments(x_0)`/`compute_moments(x_k)` above), so
        # `[1:]` drops that redundant seed and leaves one entry per loop iteration.
        # eta_k_list/theta_k_list/dH_k_list have NO such pre-loop seed -- they
        # start as empty lists and are only appended inside the loop -- so slicing
        # them with `[1:]` as well silently discarded their genuine first step
        # (k=0) and left them 2 shorter than `t` instead of 1 shorter. Fixed: no
        # slice for these three, so all five end up the same length, one entry
        # per loop iteration, aligned with `t[1:]`.
        barphi_e = torch.stack(barphi_e)[1:]
        barphi_p = torch.stack(barphi_p)[1:]
        eta_k_list = torch.stack(eta_k_list)
        theta_k_list = torch.stack(theta_k_list)
        dH_k_list = torch.cat(dH_k_list)

        self._print_memory("Everything stacked")

        print("Returning")

        return (
            self.x_k,
            barphi_e,
            barphi_p,
            eta_k_list,
            theta_k_list,
            dH_k_list,
            Theta_reg[1:],
        )

    def _solve_regularised(self, t, M, Gf, bb, cc, lam):
        t = np.asarray(t, dtype=float)
        n, r, dev = len(t), self.num_potentials, self.device
        dt = np.diff(t)
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

        # --- diagonal (Jacobi) preconditioning, mirrors compute_eta / compute_theta ---
        A_flat = A.reshape(n * r, n * r)
        f_flat = f.reshape(n * r)
        S = torch.diagonal(A_flat).clamp_min(1e-30).sqrt()        # per-(k,potential) scale
        A_flat = A_flat / (S[:, None] * S[None, :])
        A_flat = (A_flat + A_flat.T) / 2
        f_flat = f_flat / S
        if self.regularization:
            A_flat = A_flat + self.regularization * torch.eye(n * r, device=dev, dtype=A_flat.dtype)
        
        print("Calling torch.linalg.solve")
        self._print_memory("Before solve")

        z = torch.linalg.solve(A_flat, f_flat)

        print("Solve finished")
        self._print_memory("After solve")

        return (z / S).reshape(n, r) 
    
    def _solve_regularised_thomas(self, t, M, Gf, bb, cc, lam, eps_reg_theta=1e-6):
        """
        Same block-tridiagonal system as _solve_regularised, solved via block-Thomas
        elimination with block-Jacobi preconditioning instead of a dense (n*r, n*r)
        solve. Memory: O(n * r**2) instead of O(n**2 * r**2). At lam=0 this reduces
        exactly to compute_theta's preconditioned per-step solve.
        """
        t = np.asarray(t, dtype=float)
        n, r, dev = len(t), self.num_potentials, self.device
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

        # block-Jacobi preconditioning, mirrors compute_eta / compute_theta
        S = [torch.diagonal(Dk).clamp_min(1e-30).sqrt() for Dk in D]
        for k in range(n):
            D[k] = D[k] / (S[k][:, None] * S[k][None, :])
            D[k] = (D[k] + D[k].T) / 2
            f[k] = f[k] / S[k]
        for k in range(n - 1):
            U[k] = U[k] / (S[k][:, None] * S[k + 1][None, :])
            L[k] = U[k].transpose(0, 1)

        eye = torch.eye(r, device=dev, dtype=D[0].dtype)
        c_prime, d_prime = [None] * max(n - 1, 0), [None] * n

        denom0 = D[0] + eps_reg_theta * D[0].diagonal().abs().mean() * eye
        if n > 1:
            sol0 = torch.linalg.solve(denom0, torch.cat([U[0], f[0][:, None]], dim=1))
            c_prime[0], d_prime[0] = sol0[:, :-1], sol0[:, -1]
        else:
            d_prime[0] = torch.linalg.solve(denom0, f[0])

        for k in range(1, n):
            denom = D[k] - L[k - 1] @ c_prime[k - 1]
            rhs   = f[k] - L[k - 1] @ d_prime[k - 1]
            denom = denom + eps_reg_theta * denom.diagonal().abs().mean() * eye
            if k < n - 1:
                sol = torch.linalg.solve(denom, torch.cat([U[k], rhs[:, None]], dim=1))
                c_prime[k], d_prime[k] = sol[:, :-1], sol[:, -1]
            else:
                d_prime[k] = torch.linalg.solve(denom, rhs)

        Theta_scaled = [None] * n
        Theta_scaled[-1] = d_prime[-1]
        for k in range(n - 2, -1, -1):
            Theta_scaled[k] = d_prime[k] - c_prime[k] @ Theta_scaled[k + 1]

        return torch.stack([Theta_scaled[k] / S[k] for k in range(n)])


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
        theta_k_raw, bk       = self.compute_theta(y_k, k)
        corrector      = self.compute_grad_phi_projected(y_k, theta_k_raw)
        x_k_plus_one   = y_k + corrector

        # Normalise theta
        if self.sigma > 0:
            theta_k = theta_k_raw / (h * self.sigma ** 2)
        else:
            theta_k = torch.zeros_like(theta_k_raw)

        # Entropy estimate
        I_k          = self.compute_interpolant(k + 1)
        dt_phi_I_k   = self.compute_rhs_dt_phi_I_t(I_k, k)
        dH_k         = -theta_k @ dt_phi_I_k

        return x_k_plus_one, y_k, I_k, eta_k, theta_k, dH_k, bk


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
        D_k12 = torch.diag(G_k).sqrt()
        G_k = G_k /(D_k12[:,None]*D_k12[None,:])
        G_k = (G_k+G_k.T)/2
        G_k+= self.regularization*torch.eye(G_k.shape[-1],).to(G_k.dtype).to(G_k.device)
        
        # double precision to avoid fit colinearity 
        eta_k = torch.linalg.solve(G_k, (rhs_dt_phi_I_k/D_k12[:,None]))[:, 0]
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
        
        return theta_k, rhs_constraint_correction

        


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
    """
    def fit(self,x_k):
        #Refit any potentials that implement ``fit`` on the current samples.
        #Calls ``potential.fit(x_k)`` on every potential, silently skipping those that
        #do not implement (or raise inside) ``fit``.
        #Parameters
        #x_k : torch.Tensor
        #    Current samples to refit on.

        coshgt_x0 = None
        morlet_coshgt_x0 = None

        if self.use_coshgt_s0:
            # first fit coshGT potentials, capturing x0 to reuse as s0 in maxent_log
            for name, pot in self.potentials.items():
                if name in ('Scalar_psi_maxent_log', 'Scalar_morlet_maxent_log'):
                    continue
                try:
                    pot.fit(x_k)
                    if name == 'Scalar_psi_coshgt' and getattr(pot, 'is_fitted', False):
                        coshgt_x0 = pot.x0
                    elif name == 'Scalar_morlet_coshgt' and getattr(pot, 'is_fitted', False):
                        morlet_coshgt_x0 = pot.x0
                except:
                    pass

            for name, pot in self.potentials.items():
                if name == 'Scalar_psi_maxent_log':
                    try:
                        pot.fit(x_k, s0=coshgt_x0 * 10)
                    except:
                        pass
                elif name == 'Scalar_morlet_maxent_log':
                    try:
                        pot.fit(x_k, s0=morlet_coshgt_x0 * 10)
                    except:
                        pass
        else:
            for name, pot in self.potentials.items():
                try:
                    pot.fit(x_k)
                except:
                    pass
    """
