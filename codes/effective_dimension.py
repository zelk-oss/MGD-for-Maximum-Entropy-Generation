"""
Guth et al.'s local effective dimensionality d_eff(x, tau), reusing MGD's own
fitted theta trajectory (Theta_reg) instead of any new sampling/fitting run.

See notes/effective_dimension_MGD.tex for the full derivation linking Guth's
variance-exploding local-dimension estimator to MGD's Cos-schedule stochastic
interpolant, including this codebase's (+) sign convention (see
utils_entropy.py's "Sign convention audit" -- p_theta(x) ~ exp(+theta^T
phi(x)), the OPPOSITE of the MGD paper's exp(-theta^T phi(x))).

Everything here operates on already-saved results -- no SDE rerun:
  - Theta_reg / t_reg             regularised theta trajectory + its own
                                   coarse time grid (reconstructed offline
                                   via reconstruct_t_reg if not saved)
  - a run's fitted potentials     phi, grad phi -- analytic, not autograd
  - m_t                            moment trajectory at the t_reg nodes,
                                   loaded from aux_moments if present, else
                                   recomputed from the run's own data x1 via
                                   fresh Monte Carlo over the interpolant
                                   (data-only, no particle trajectory needed)
  - dH_t_bound / t (fine grid)     entropy-increment trajectory, integrated
                                   to give H_bound(t) at any t
"""
from pathlib import Path

import numpy as np
import torch

from utils_entropy import _phi, standard_gaussian_entropy


# ============================================================================
# Cos-schedule <-> Guth noise-scale bookkeeping
# ============================================================================

def alpha_of_t(t):
    """Cos-schedule angle alpha_t = pi*t/2 (sde_routines.compute_interpolant,
    interpolant='Cos' -- the only schedule forward_regularised supports)."""
    return np.pi * np.asarray(t, dtype=np.float64) / 2.0


def tau_of_t(t):
    """Guth's noise scale tau(t) = cot^2(alpha_t).

    From y_t := I_t / sin(alpha_t) = x_1 + cot(alpha_t) x_0, so conditional on
    x_1 = x, y_t | x ~ N(x, tau(t) I). Monotonically decreasing, tau(0) = inf,
    tau(1) = 0.
    """
    a = alpha_of_t(t)
    return (np.cos(a) / np.sin(a)) ** 2


def s_of_t(t):
    """s = (1/2) log tau(t) = log cos(alpha_t) - log sin(alpha_t).

    Monotonically DEcreasing in t (tau decreases as t -> 1), unlike t or tau
    (which is why every derivative-in-s below sorts by s before differencing
    rather than assuming t_reg's own order is monotonic in s).
    """
    a = alpha_of_t(t)
    return np.log(np.cos(a)) - np.log(np.sin(a))


# ============================================================================
# t_reg reconstruction (Theta_reg's own coarse grid, when not saved to disk)
# ============================================================================

def reconstruct_t_reg(t, n_subsample, eps=1e-8, min_dt=None):
    """Offline replay of SDE.forward_regularised's coarse-grid bookkeeping
    (block accumulation over n_subsample fine steps, SDE._cut_close_time_nodes,
    then the routine's own trailing [1:] slice) -- reproduces Theta_reg's time
    axis purely from the FINE sampling-times grid `t` and `n_subsample`, for
    runs saved before t_reg was written to disk (every d=256 turbulence run as
    of this writing). Verified byte-identical in length against an on-disk
    Theta_reg for one such run (197 nodes) during the investigation this
    module follows up on.

    Parameters
    ----------
    t : array_like, shape (len_fine+1,)
        The run's saved fine sampling-times grid.
    n_subsample : int
        The run's --n_subsample (see config.json).

    Returns
    -------
    numpy.ndarray, shape matching Theta_reg.shape[0]
    """
    t = np.asarray(t, dtype=np.float64)
    t_used, cnt = [], 0
    for k in range(len(t) - 1):
        t_node = t[k + 1]
        a = np.pi * t_node / 2.0
        if np.sin(a) > eps and np.cos(a) > eps:
            if cnt == 0:
                t_used.append(t_node)
            cnt = (cnt + 1) % n_subsample
    tt = np.asarray(t_used[1:], dtype=np.float64)
    if min_dt is None:
        span = tt[-1] - tt[0] if len(tt) > 1 else 1.0
        min_dt = max(span * 1e-5, 1e-6)
    keep = [0]
    for i in range(1, len(tt)):
        if tt[i] - tt[keep[-1]] >= min_dt:
            keep.append(i)
    return tt[np.asarray(keep)][1:]


# ============================================================================
# Standalone phi / grad-phi evaluation (no live SDE instance needed)
# ============================================================================

def potential_indices(potentials):
    """Cumulative per-potential offsets into a flat theta/phi vector, in the
    SAME dict iteration order SDE._sync_potential_dims uses."""
    sizes = [p.num_coefficients for p in potentials.values()]
    return np.cumsum([0] + sizes)


def grad_phi_projected(potentials, x, vector):
    """sum_i vector_i * grad phi_i(x), standalone -- mirrors
    SDE.compute_grad_potentials(x, vector=...) without needing a live SDE
    instance. `vector` has length r = sum of num_coefficients, sliced per
    potential via potential_indices.
    """
    idx = potential_indices(potentials)
    out = torch.zeros_like(x)
    for i, p in enumerate(potentials.values()):
        out = out + p.grad(x, v=vector[idx[i]:idx[i + 1]])
    return out


def load_fitted_potentials(potentials, potentials_dir, device='cpu'):
    """Load each potential's saved fitted/pruned state (e.g. the
    auto-pruned Scalar_*_gaussianK active-statistic masks) in place of a
    freshly-built (unfitted) potentials dict, exactly as SDE.__init__ does
    for a fresh run. Required: rebuilding potentials without this step gives
    the WRONG r (e.g. r=329 instead of the fitted r=272 for the d=256
    turbulence runs) and silently mis-indexes every theta coefficient.

    Mutates nothing in place; returns a new dict.
    """
    potentials_dir = Path(potentials_dir)
    out = dict(potentials)
    for name, pot in potentials.items():
        state_path = potentials_dir / f'{name}.pt'
        if hasattr(pot, 'is_fitted') and hasattr(pot, 'load_fixed_parameters') and state_path.exists():
            loaded = type(pot).load_fixed_parameters(state_path, pot.filters, map_location=device)
            loaded.to(device)
            out[name] = loaded
    return out


# ============================================================================
# m_t: saved aux_moments, or recomputed from data alone (no SDE rerun)
# ============================================================================

def recompute_m_t(potentials, x1, t_query, n_mc_noise=1, generator=None):
    """m_t = E_Z[phi(cos(alpha_t) Z + sin(alpha_t) X)], X ~ the run's own
    training data x1, Z fresh N(0, I) noise. Depends only on the data and the
    schedule -- NOT on any saved x_0/particle trajectory -- so this needs no
    SDE rerun, only the potentials and x1 already required elsewhere.

    n_mc_noise independently redraws Z per data point and averages; the outer
    expectation is already over X ~ x1, so n_mc_noise=1 is already unbiased
    (more just reduces variance from Z).

    Returns
    -------
    torch.Tensor, shape (len(t_query), r)
    """
    x1 = x1.detach()
    out = []
    for t in t_query:
        a = np.pi * float(t) / 2.0
        acc = None
        for _ in range(n_mc_noise):
            z = torch.randn(x1.shape, generator=generator, dtype=x1.dtype, device=x1.device)
            I_t = np.cos(a) * z + np.sin(a) * x1
            m = _phi(I_t, potentials).mean(0)
            acc = m if acc is None else acc + m
        out.append((acc / n_mc_noise).detach().cpu())
    return torch.stack(out)


def load_or_recompute_m_t(aux_moments_path, t_fine, t_query, potentials, x1,
                           n_mc_noise=1, generator=None):
    """Prefer the run's saved aux_moments (barphi_e -- the exact realized
    target moments the corrector step was driving toward, on the run's OWN
    fine grid t_fine[1:]); fall back to recompute_m_t (data + schedule only,
    same distribution up to Monte Carlo noise) when aux_moments wasn't saved
    or hasn't been pulled from the cluster yet.

    Returns (m_t_query, source) with source in {'aux_moments', 'recomputed'}.
    """
    aux_moments_path = Path(aux_moments_path)
    if aux_moments_path.exists():
        aux = torch.load(aux_moments_path, map_location='cpu')
        barphi_e = aux['barphi_e'].detach().cpu()
        t_fine = np.asarray(t_fine, dtype=np.float64)
        t_axis = t_fine[1:1 + barphi_e.shape[0]]
        idx = np.searchsorted(t_axis, t_query)
        idx = np.clip(idx, 0, len(t_axis) - 1)
        return barphi_e[idx], 'aux_moments'
    return recompute_m_t(potentials, x1, t_query, n_mc_noise=n_mc_noise, generator=generator), 'recomputed'


def h_bound_at(t_fine, dH_fine, t_query, d):
    """H_bound(t) = H(p_0) + int_0^t dH_s ds (MGD Prop. 4.3 / Eq. 30),
    evaluated at each node of t_query via linear interpolation of the
    trapezoid-cumulated integral over the FINE grid (matching
    utils_entropy.entropy_bound's own quadrature).

    dH_fine is aligned with t_fine[1:] (utils_entropy.entropy_bound's length
    convention) -- NOT t_fine itself.
    """
    t_fine = np.asarray(t_fine, dtype=np.float64)
    dH_fine = np.asarray(dH_fine, dtype=np.float64)
    t_grid = t_fine[1:1 + len(dH_fine)]                 # aligned with dH_fine, one entry per step
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dH_fine[1:] + dH_fine[:-1]) * np.diff(t_grid))])
    # cum[i] = integral of dH from t_grid[0] to t_grid[i] -> indexed by t_grid itself,
    # NOT t_grid prepended with t_fine[0] (that extra point has no matching cum entry).
    H_p0 = standard_gaussian_entropy(d)
    return H_p0 + np.interp(t_query, t_grid, cum)


# ============================================================================
# Rare / typical selection (ported verbatim from
# turbulence/turb_theta_reg_rare_typical.ipynb's extract_rare_typical)
# ============================================================================

def compute_acceleration(Data):
    """a(t) = Data(t+1) - Data(t), lag-1 velocity increment."""
    return Data[..., 1:] - Data[..., :-1]


def extract_rare_typical(Data, criterion='percentile', std_threshold=40,
                          sustained_sigma=10, min_exceedances=3, percentile=95):
    """Split Data's leading (batch) axis into typical / rare indices by the
    per-trajectory max |acceleration| (lag-1 increment), same criterion and
    default (95th percentile) as the theta_reg rare/typical analysis
    notebook -- kept identical here so d_eff's rare/typical split lines up
    with that notebook's, on the SAME Data[:n1] ordering (i.e. these indices
    index directly into x1 = normalize(Data[:n1]) too).
    """
    accel = compute_acceleration(Data)
    accel_std = accel.flatten().std(unbiased=True).item()
    B = accel.shape[0]
    reduce_dims = tuple(range(1, accel.ndim))
    per_traj_max_abs_accel = accel.abs().amax(dim=reduce_dims)

    if criterion == 'max_sigma':
        is_rare = per_traj_max_abs_accel > std_threshold * accel_std
        info = dict(accel_std=accel_std, threshold=std_threshold * accel_std)
    elif criterion == 'sustained_sigma':
        exceed_mask = accel.abs() > sustained_sigma * accel_std
        per_traj_exceed_count = exceed_mask.reshape(B, -1).sum(dim=1)
        is_rare = per_traj_exceed_count >= min_exceedances
        info = dict(accel_std=accel_std, threshold=sustained_sigma * accel_std,
                    min_exceedances=min_exceedances, exceed_counts=per_traj_exceed_count)
    elif criterion == 'percentile':
        cutoff = np.percentile(per_traj_max_abs_accel.cpu().numpy(), percentile)
        is_rare = per_traj_max_abs_accel > cutoff
        info = dict(accel_std=accel_std, percentile=percentile, cutoff=cutoff)
    else:
        raise ValueError(f"unknown criterion '{criterion}'")

    idx_rare = torch.nonzero(is_rare).flatten().tolist()
    idx_typical = torch.nonzero(~is_rare).flatten().tolist()
    return idx_typical, idx_rare, info


# ============================================================================
# d_eff -- denoiser form
# ============================================================================

def denoiser_d_eff(x, theta_reg, t_reg, potentials, n_mc=64, generator=None):
    """Guth's denoiser-form d_eff(x, tau(t)) at every node of t_reg, for a
    batch of points x (n_x, C, T), reusing this run's regularised theta
    trajectory Theta_reg (already fit on the run's own guided transport).

    Derivation (code's (+) convention; full derivation in
    notes/effective_dimension_MGD.tex):

        d_eff(x,tau) = (1/tau) E_y[ ||x - E[x|y]||^2 | x ]
        E[x|y] = y + tau * grad_y log p_tau(y)      (Tweedie)
        grad_y log p_tau(y) ~= +sin(alpha_t) theta_t^T grad_phi(sin(alpha_t) y)

    which collapses (1/tau cancels analytically) to

        d_eff(x,tau) = E_xi[ ||xi + sqrt(tau) sin(alpha_t) theta_t^T grad_phi(sin(alpha_t) y)||^2 ],
        y = x + sqrt(tau) xi,  xi ~ N(0, I)

    Note sin(alpha_t) y = sin(alpha_t) x + cos(alpha_t) xi -- literally
    compute_interpolant's own Cos schedule with x as the single-point data
    endpoint and xi as fresh noise, so this never evaluates phi/grad_phi
    outside the numerical range the fit itself already explores.

    Returns
    -------
    numpy.ndarray, shape (len(t_reg), n_x)
    """
    a = alpha_of_t(t_reg)
    tau = tau_of_t(t_reg)
    n_t, n_x = len(t_reg), x.shape[0]
    sig_shape = x.shape

    out = np.zeros((n_t, n_x))
    for i in range(n_t):
        ai, taui = a[i], tau[i]
        theta_i = theta_reg[i].to(x.dtype)          # potentials' grad() requires x's own dtype
        xi = torch.randn((n_mc,) + sig_shape, generator=generator, dtype=x.dtype, device=x.device)
        y = x.unsqueeze(0) + np.sqrt(taui) * xi                       # (n_mc, n_x, C, T)
        sy = np.sin(ai) * y
        sy_flat = sy.reshape((n_mc * n_x,) + sig_shape[1:])
        g_flat = grad_phi_projected(potentials, sy_flat, theta_i)
        score = np.sin(ai) * g_flat.reshape((n_mc, n_x) + sig_shape[1:])
        term = xi + np.sqrt(taui) * score
        out[i] = term.reshape(n_mc, n_x, -1).pow(2).sum(-1).mean(0).detach().cpu().numpy()
    return out


# ============================================================================
# d_eff -- energy form
# ============================================================================

def energy_d_eff(x, theta_reg, t_reg, potentials, m_t_reg, H_bound_t_reg, d,
                  n_mc=64, generator=None):
    """Guth's energy-form d_eff(x, tau) = d - d/ds E_y[U(y,tau) | x],
    s = (1/2) log tau, evaluated by finite-differencing E_y[U] directly on
    the t_reg grid (rather than splitting analytic/MC pieces).

    U(y,tau) = -theta_t^T phi(sin(alpha_t) y) + logZ_t - d*log(sin(alpha_t))
             (code's (+) convention -- the sign on the phi term and on
             logZ_t both flip relative to a paper-convention derivation;
             see notes/effective_dimension_MGD.tex)
    logZ_t ~= H_bound(t) + theta_t^T m_t
             (MGD Eq. 8/30, generalised from its stated t=1 endpoint to any
             t -- this is a LOWER BOUND on log Z_t, not log Z_t itself, so
             the resulting energy-form d_eff inherits that bound's bias;
             see the caveat in notes/effective_dimension_MGD.tex)

    Parameters
    ----------
    m_t_reg : torch.Tensor, shape (len(t_reg), r)
    H_bound_t_reg : numpy.ndarray, shape (len(t_reg),)
    d : int
        Ambient dimension (prod of x's per-sample shape).

    Returns
    -------
    numpy.ndarray, shape (len(t_reg), n_x)
    """
    a = alpha_of_t(t_reg)
    tau = tau_of_t(t_reg)
    s = s_of_t(t_reg)
    n_t, n_x = len(t_reg), x.shape[0]
    sig_shape = x.shape

    theta_reg = theta_reg.double()
    m_t_reg = m_t_reg.to(theta_reg.dtype)
    logZ = H_bound_t_reg + torch.einsum('tr,tr->t', theta_reg, m_t_reg).numpy()

    U = np.zeros((n_t, n_x))
    for i in range(n_t):
        ai, taui = a[i], tau[i]
        theta_i = theta_reg[i]                       # kept double: theta^T phi is a large-cancellation sum
        xi = torch.randn((n_mc,) + sig_shape, generator=generator, dtype=x.dtype, device=x.device)
        y = x.unsqueeze(0) + np.sqrt(taui) * xi
        sy = np.sin(ai) * y
        sy_flat = sy.reshape((n_mc * n_x,) + sig_shape[1:])
        phi_sy = _phi(sy_flat, potentials).double().reshape(n_mc, n_x, -1)  # potentials run in x's dtype (float32)
        mean_phi_term = (phi_sy @ theta_i).mean(0).detach().cpu().numpy()
        U[i] = -mean_phi_term + logZ[i] - d * np.log(np.sin(ai))

    order = np.argsort(s)                       # s is decreasing in t_reg; sort ascending for np.gradient
    dUds_sorted = np.gradient(U[order], s[order], axis=0)
    dUds = np.empty_like(dUds_sorted)
    dUds[order] = dUds_sorted
    return d - dUds
