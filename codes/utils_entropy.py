    ## ----------------------------------------------------- Entropy related funtions -----------------------------------------------------
import re

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid



def kl_divergence(p, q, n_bins, bins = None, epsilon=1e-5):
   
    """Histogram estimate of the KL divergence ``KL(p || q)`` (``p`` reference).
 
    Both samples are binned into densities (with ``epsilon`` floor) and the discrete
    KL is integrated against the bin widths. If ``bins`` is None, equal-count edges
    on ``(p + q) / 2`` are used (:func:`histedges_equalN`).
 
    Parameters
    ----------
    p, q : array_like
        Samples; ``p`` is the reference distribution.
    n_bins : int
        Number of bins when ``bins`` is None.
    bins : array_like, optional
        Explicit bin edges.
    epsilon : float, optional
        Density floor, by default 1e-5.
 
    Returns
    -------
    float
        Estimated KL divergence.
    """

    
    if bins is not None:
        p = np.histogram(p, bins, range=None, density=True, weights=None)[0]+epsilon
        q = np.histogram(q, bins, range=None, density=True, weights=None)[0]+epsilon
        d_bins = bins[1:]-bins[:-1]
    else:
        #minus = min(np.min(p),np.min(q))
        #maxus = max(np.max(p),np.max(q))
        #bins = np.linspace(minus,maxus,n_bins)
        #d_bins = (maxus-minus)/n_bins

        bins = histedges_equalN((p+q)/2, n_bins)
        d_bins = bins[1:]-bins[:-1]
        
        p = np.histogram(p, bins, range=None, density=True, weights=None)[0]+epsilon
        q = np.histogram(q, bins, range=None, density=True, weights=None)[0]+epsilon
        
    return np.sum(np.where(p != 0, p * np.log(p / q), 0)*d_bins)

def entropy(p, n_bins, bins = None, epsilon=1e-5):
   
    """Histogram estimate of the differential entropy of ``p``.
 
    Bins ``p`` into a density (with ``epsilon`` floor) and integrates ``-p log p``
    against the bin widths. Equal-count edges are used when ``bins`` is None.
 
    Parameters
    ----------
    p : array_like
        Samples.
    n_bins : int
        Number of bins when ``bins`` is None.
    bins : array_like, optional
        Explicit bin edges.
    epsilon : float, optional
        Density floor, by default 1e-5.
 
    Returns
    -------
    float
        Estimated differential entropy.
    """

    
    if bins is not None:
        p = np.histogram(p, bins, range=None, density=True, weights=None)[0]+epsilon
    else:
        bins =  histedges_equalN(p, n_bins)
        p = np.histogram(p, bins, range=None, density=True, weights=None)[0]+epsilon

    d_bins = bins[1:]-bins[:-1]
    
    return np.sum(np.where(p != 0,  -np.log(p)*p, 0)*d_bins)

def histedges_equalN(x, nbin):
    """Bin edges placing (approximately) equal sample counts per bin.
 
    Quantile binning: interpolates the sorted samples at evenly spaced ranks.
 
    Parameters
    ----------
    x : array_like
        Samples.
    nbin : int
        Number of bins.
 
    Returns
    -------
    numpy.ndarray
        ``nbin + 1`` bin edges.
    """

    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

def compute_gaussian_entropy(x1, interpolant, t):
    
    """Closed-form entropy of the Gaussian interpolant marginal along ``t``.
 
    Assumes a single channel. The data spectrum is the value variance (1D case,
    ``len(x1.shape) == 2``) or the diagonal of the Fourier covariance (2D). With
    base entropy ``H_p_0 = (log(2 pi) + 1) d / 2`` (``d`` the dimension), returns
    ``H_p_0 + 0.5 * sum_k log(var_t,k)`` where the per-mode variance follows the
    chosen interpolant schedule (Linear / VarPreserv / Sqrt / Cos).
 
    Parameters
    ----------
    x1 : torch.Tensor
        Data samples, shape (B, T) or (B, M, N).
    interpolant : str
        Interpolant schedule.
    t : torch.Tensor
        Times, shape (n_t, 1).
 
    Returns
    -------
    torch.Tensor
        Gaussian entropy at each time, shape (n_t,).
    """

    # assume the number of channels to be 1


    if len(x1.shape)==2:
        spectrum_x1 = torch.var(x1.flatten()).cpu()
        
        d = x1.shape[-1]
        
    else:
        x1_fourier = torch.fft.fft2(x1)
        cov_x1 = (x1_fourier.reshape(x1.shape[0], x1.shape[-2]*x1.shape[-1]).T@x1_fourier.reshape(x1.shape[0], x1.shape[-2]*x1.shape[-1])).cpu()
        spectrum_x1 = torch.diag(cov_x1)/(x1.shape[-2]*x1.shape[-1]*np.sqrt(x1.shape[0]))
    
        d = x1.shape[-2]*x1.shape[-1]
    
    H_p_0 = (np.log(2*np.pi)+1)*d/2

    match interpolant:
        case 'Linear':
            return H_p_0 + torch.log(spectrum_x1.abs()[None]*t[:,None]**2+(1-t[:,None])**2).sum(1)/2
        case 'VarPreserv':
            return H_p_0 + torch.log(spectrum_x1.abs()[None]*t[:,None]+(1-t[:,None])).sum(1)/2
        case 'Sqrt':
            return H_p_0 + torch.log(spectrum_x1.abs()[None]*t[:,None]+(1-np.sqrt(t[:,None]))**2).sum(1)/2
        case 'Cos':
            return H_p_0 + torch.log(spectrum_x1.abs()[None]*np.sin(np.pi * t[:,None] / 2)**2+np.cos(np.pi * t[:,None] / 2)**2).sum(1)/2


    ## ----------------------------------------------------- MGD lower bound on log Z -----------------------------------------------------
"""
Sign convention audit
----------------------
This codebase's ``theta_t``/``Theta_reg`` are the natural parameter of
``p_theta(x) = Z_theta^{-1} exp(+theta^T phi(x))`` -- the OPPOSITE of the
``exp(-theta^T phi(x))`` convention the MGD paper states its formulas with.
All formulas below are re-derived directly in the code's own (+) convention,
not by substituting theta_paper = -theta_code into the paper's formulas, so
nothing here needs an extra sign flip at the call site.

Checked directly against the SDE drift, ``sde_routines.py``
(``iteration_step_projection``, ~lines 692-715)::

    corrector    = self.compute_grad_phi_projected(y_k, theta_k_raw)   # = +theta_raw . grad_phi(y_k)
    x_k_plus_one = y_k + corrector                                     # PLUS
    theta_k      = theta_k_raw / (h * self.sigma ** 2)

Per-step this is ``x_{k+1} = y_k + h*sigma^2*theta_k*grad_phi(y_k)``, i.e. a
continuous-time drift ``eta_t + sigma^2 theta_t`` (PLUS). The MGD paper's
stated SDE is ``dX_t = (eta_t - sigma^2 theta_t)^T grad_phi dt + ...``
(MINUS). The two agree only if the code's ``theta_t`` is the negative of the
paper's ``theta_t``, i.e. the code's exponential family is
``exp(+theta^T phi(x))``.

Independently corroborated by ``dH_k = -theta_k @ dt_phi_I_k`` (the
already-implemented entropy-bound integrand computed in
``iteration_step_projection``, line ~713 -- see ``entropy_bound`` below) and
by ``turbulence/bimodal_experiment/bimodal_theta.ipynb`` cells 4/14/15,
which fit the MGD paper's 1D bimodal case directly and compare the *raw,
unnegated* fitted theta to ``expected_theta = [beta/2, 5*beta, 0, -beta]``
-- the coefficients of ``+phi(x)`` in ``-U(x) = log(p(x)) + const``.

Derivation of ``log_Z_bound`` in the code's own convention
------------------------------------------------------------
``S(p_theta) = -E[log p_theta] = -E[theta^T phi(x) - log Z_theta]
             = log Z_theta - theta^T m``                      (note: MINUS)

    H(p_0)  = log Z_theta0 - theta_0^T m_0                    [t=0, under p_0 = p_theta0]
    H_bound = H(p_0) + int_0^1 dH_t_bound dt                  [Prop 4.3 / Eq. 30, dH_t_bound as coded]
    H(p_1^sigma) <= S(p_theta1) = log Z_theta1 - theta_1^T m_1  [t=1, definition of p_theta1]

    => log Z_theta1 >= H_bound + theta_1^T m_1  =:  log Z^sigma

Expanded, this is ``log Z_theta0 - theta_0^T m_0 + theta_1^T m_1 +
int_0^1 dH_t_bound dt`` -- since ``dH_t_bound = -theta_t^T dm_t/dt`` exactly
as computed by the pipeline, this is the same statement as
``log Z_theta0 - theta_0^T m_0 + theta_1^T m_1 - int_0^1 theta_t^T (dm_t/dt) dt``.

``log_Z_bound`` is a LOWER bound on ``log Z_theta1``, not an estimate.
"""


def standard_gaussian_entropy(d, log_det_cov=0.0):
    """Differential entropy of ``N(0, Sigma)`` in ``d`` dimensions.

    Distinct from ``compute_gaussian_entropy`` above (entropy of the
    time-``t`` interpolant marginal along an SDE schedule) -- this is the
    plain closed-form Gaussian entropy.

    ``log_det_cov`` is ``log det(Sigma)``: 0 for ``Sigma = I_d`` (this
    codebase's actual ``x_0`` initialization -- ``sde_routines.py``'s
    ``init_interpolants_and_workers`` draws ``x_0`` via bare ``torch.randn``,
    i.e. ``N(0, I_d)``, whenever ``x_0`` isn't passed explicitly, which is
    the case for every entry point in this repo), ``d * log(s**2)`` for an
    isotropic ``s**2 * I_d``, or a general ``log(det(Sigma))`` otherwise.
    """
    return 0.5 * d * (np.log(2 * np.pi) + 1) + 0.5 * log_det_cov


def data_gaussian_entropy(x_ref):
    """H(g), g the Gaussian matching the empirical covariance of ``x_ref``
    (NOT the standard-normal ``H(p_0)`` from ``standard_gaussian_entropy``)
    -- the quantity the negentropy ``H(g) - H_*^sigma`` needs, per its
    definition. These coincide with ``standard_gaussian_entropy(d)`` only if
    the data were whitened to unit covariance before fitting; check rather
    than assume."""
    x = x_ref.detach().cpu().reshape(x_ref.shape[0], -1).double()
    d = x.shape[1]
    x = x - x.mean(0, keepdim=True)
    cov = (x.T @ x) / (x.shape[0] - 1)
    sign, logdet = torch.linalg.slogdet(cov)
    if sign.item() <= 0:
        raise ValueError("Empirical covariance is not positive definite "
                          "(singular or numerically degenerate); cannot take log det.")
    return standard_gaussian_entropy(d, log_det_cov=logdet.item())


def _phi(x, potentials):
    """Stack potential.forward(x) over potentials.values(), matching
    SDE.compute_moments's concatenation order exactly (same dict, same
    iteration order) but standalone -- no live Solver/SDE instance needed."""
    feats = []
    with torch.no_grad():
        for p in potentials.values():
            f = p.forward(x)
            if f.ndim == 1:
                f = f.unsqueeze(1)
            feats.append(f)
    return torch.cat(feats, dim=1)


def _check_p0_hypothesis(theta_0, potentials, sample_shape, device, seed=0):
    """Diagnostic for the ``p_0 = p_theta0`` hypothesis H_p0's validity
    depends on: draws fresh iid N(0, I_d) samples (p_0 is known analytically
    -- no need for saved x_0 particles) and checks whether ``theta_0^T
    phi(x)`` reproduces ``-|x|^2/2`` up to an additive constant (the
    constant is free -- it's absorbed into log Z_theta0). Reports the
    residual's std relative to the target's std: near 0 means the
    hypothesis holds; not small means phi's quadratic components don't span
    |x|^2, and H_p0 is not actually H(p_theta0)."""
    gen = torch.Generator().manual_seed(seed)
    x0_samples = torch.randn(sample_shape, generator=gen).to(device)
    phi = _phi(x0_samples, potentials)
    pred = (phi @ theta_0.to(device)).detach().cpu()
    target = -0.5 * x0_samples.detach().cpu().reshape(x0_samples.shape[0], -1).pow(2).sum(1)

    resid = pred - target
    resid = resid - resid.mean()
    target_c = target - target.mean()
    target_std = target_c.std().item()
    ratio = float('nan') if target_std == 0 else (resid.std() / target_std).item()

    return {
        'relative_residual_std': ratio,
        'n_check': sample_shape[0],
        'note': ('near 0 => phi spans |x|^2/2 and theta_0 reproduces it '
                 '(p_0 = p_theta0 supported); not small => hypothesis fails, '
                 'H_p0 is not the right constant.'),
    }


def entropy_bound(results, key, potentials, d=None, i_final=-1, device='cpu',
                   check_p0=True, n_p0_check=2000, seed=0):
    """Lower bound on H(p_1^sigma) (MGD Prop. 4.3 / Eq. 30):

        H_*^sigma = H(p_0) + int_0^1 dH_t_bound dt

    using the pipeline's own ``dH_t_bound`` (``sde_routines.py``'s
    ``dH_k = -theta_k @ dt_phi_I_k``) directly as the integrand -- already
    correctly signed for this codebase's theta convention, see this file's
    sign-convention audit above. No plotting; returns a dict.

    ``d`` (ambient dimension) defaults to ``prod(xt.shape[1:])`` -- every
    axis but the batch axis -- NOT the earlier snippet's
    ``x_ref.shape[-2]*x_ref.shape[-1]``, which only happened to be correct
    because every current run uses ``channels=1``; it would silently break
    for a genuinely multi-channel signal or a scalar ``(B, C)`` signal where
    ``shape[-2]`` is the batch axis, not a signal axis.

    ``i_final`` indexes into ``dH_t_bound``/``theta_t`` (both length
    ``len(t) - 1``, aligned with ``t[1:]``, one entry per SDE step -- see
    the length-mismatch fix in ``sde_routines.py``'s ``forward_regularised``).
    Default -1 = the last computed step (t close to 1). ``log_Z_bound``
    below must be called with the SAME ``i_final`` so the integral's
    endpoint and ``theta_1`` refer to the same point.

    Returns a dict with (at minimum): H_p0, integral (trapezoid value),
    H_bound, i_final, left_riemann, quadrature_gap (trapezoid - left_riemann),
    is_uniform_grid, refinement (drift under every-2nd/every-4th subsampling),
    d, p0_check (or None if check_p0=False).
    """
    res = results[key]
    t = res['t'].detach().cpu()
    dH = res['dH_t_bound'].detach().cpu()

    expected_len = t.shape[0] - 1
    if dH.shape[0] != expected_len:
        raise ValueError(
            f"len(dH_t_bound)={dH.shape[0]} != len(t)-1={expected_len} for "
            f"key={key!r}. This run was likely computed before the "
            f"sde_routines.py length fix (forward_regularised used to "
            f"over-slice theta_k_list/dH_k_list by one entry) -- re-run it "
            f"rather than masking the mismatch with a min()."
        )

    n = dH.shape[0]
    i_final_pos = i_final if i_final >= 0 else n + i_final
    if not (0 <= i_final_pos < n):
        raise IndexError(f"i_final={i_final} out of range for length-{n} arrays")

    dH_used = dH[:i_final_pos + 1]
    t_full_used = t[:i_final_pos + 2]           # includes t[0]
    t_grid = t_full_used[1:]                     # arrival times, aligned with dH_used
    h = t_full_used[1:] - t_full_used[:-1]

    is_uniform = bool(torch.allclose(h, h[0].expand_as(h), rtol=1e-3, atol=1e-8))

    trapz = float(trapezoid(dH_used.numpy(), x=t_grid.numpy()))
    left_riemann = float((dH_used * h).sum().item())
    quadrature_gap = trapz - left_riemann

    refinement = {}
    for stride, name in ((2, 'every_2nd'), (4, 'every_4th')):
        if len(dH_used) // stride >= 2:
            sub_dH = dH_used[::stride]
            sub_t = t_grid[::stride]
            sub_trapz = float(trapezoid(sub_dH.numpy(), x=sub_t.numpy()))
            refinement[name] = {'trapezoid': sub_trapz, 'drift_from_full': sub_trapz - trapz}

    xt_d = int(np.prod(res['xt'].shape[1:]))
    if d is None:
        d = xt_d
    elif d != xt_d:
        raise ValueError(f"d={d} does not match particle dimensionality inferred "
                          f"from xt ({xt_d})")

    H_p0 = standard_gaussian_entropy(d)
    H_bound = H_p0 + trapz

    p0_check = None
    if check_p0:
        theta_0 = res['theta_t'][0].detach().cpu()
        sample_shape = (n_p0_check,) + tuple(res['xt'].shape[1:])
        p0_check = _check_p0_hypothesis(theta_0, potentials, sample_shape, device, seed)

    return {
        'H_p0': H_p0, 'integral': trapz, 'H_bound': H_bound, 'i_final': i_final_pos,
        'left_riemann': left_riemann, 'quadrature_gap': quadrature_gap,
        'is_uniform_grid': is_uniform, 'refinement': refinement,
        'd': d, 'p0_check': p0_check,
    }


def _m1_from_particles(xt, potentials, device):
    return _phi(xt.to(device), potentials).mean(0)


def log_Z_bound(results, key, theta_key, potentials, device='cpu', m1_source='target',
                 i_final=-1, n_bootstrap=200, seed=0, **entropy_bound_kwargs):
    """Lower bound on log Z_theta1 (MGD Eq. 8, re-derived in this codebase's
    own theta convention -- see this file's sign-convention audit above):

        log_Z_bound = H_bound + theta_1^T m_1

    ``theta_key``: ``'theta_t'`` (raw per-step MGD theta) or ``'Theta_reg'``
    (time-regularised). For ``'theta_t'``, ``i_final`` selects the SAME
    index ``entropy_bound`` uses for H_bound's integral endpoint. For
    ``'Theta_reg'``: it's fit on its own coarser, separately time-subsampled
    grid (``t_reg`` inside ``forward_regularised``) that isn't saved in
    ``results`` at all, so there's no comparable index to align it to --
    its endpoint is always ``Theta_reg[-1]`` (the regularised fit's value at
    t~1), not governed by ``i_final``.

    ``m1_source``: ``'target'`` (default) uses the solver's own constraint
    moments at t=1, ``results[key]['barphi_e'][i_final]`` -- the interpolant
    moments the corrector step was actually driving toward (matches how
    Eq. 8 is stated). ``'particles'`` uses the empirical moments of the
    final walkers, ``mean_i phi(xt_i)``. Both are always computed and
    returned (``m1_target``/``m1_particles``) together with their relative
    discrepancy ``m1_discrepancy = ||m1_target - m1_particles|| /
    ||m1_target||`` -- itself a diagnostic on how well the corrector step
    converged. Falls back to ``'particles'`` (with ``m1_source`` set
    accordingly in the return value) when ``barphi_e`` wasn't saved for this
    run (e.g. ``--no_save_aux_moments``).

    Error bars: ``theta1_dot_m1_bootstrap_std`` is a PARTICLE bootstrap over
    ``xt`` only -- resample the walkers with replacement, recompute
    ``m1_particles``, hold ``theta_1`` fixed, report the std of ``theta_1 @
    m1_particles_boot`` across resamples. Since ``theta_1`` was fit on this
    same particle trajectory, this is NOT independent of the
    ``m1_target``/``m1_particles`` discrepancy above -- the two errors are
    correlated, not separate independent sources. It is also NOT a total
    uncertainty on ``log_Z_bound``: it doesn't cover the SDE-integration
    uncertainty in ``H_bound`` (see ``entropy_bound``'s ``quadrature_gap``/
    ``refinement`` for that piece) or any across-seed variability -- use
    ``summarize_log_Z_bound`` for across-run aggregation when several seeds
    at the same sigma are available; with only one run, this bootstrap std
    is reported alone, explicitly NOT as the total error.
    """
    eb = entropy_bound(results, key, potentials, i_final=i_final, device=device,
                        **entropy_bound_kwargs)
    i_final_pos = eb['i_final']
    H_bound = eb['H_bound']

    res = results[key]

    if theta_key == 'theta_t':
        theta_1 = res['theta_t'][i_final_pos].detach().cpu()
    elif theta_key == 'Theta_reg':
        theta_1 = res['Theta_reg'][-1].detach().cpu()
    else:
        raise ValueError("theta_key must be 'theta_t' or 'Theta_reg'")

    xt = res['xt']
    phi_xt = _phi(xt.to(device), potentials).detach().cpu()
    m1_particles = phi_xt.mean(0)

    m1_target = None
    barphi_e = res.get('barphi_e')
    if barphi_e is not None:
        barphi_e = barphi_e.detach().cpu()
        # barphi_e is aligned with theta_t/dH_t_bound (same length, same
        # i_final indexing) after the sde_routines.py length fix.
        m1_target = barphi_e[i_final_pos] if barphi_e.shape[0] > i_final_pos else barphi_e[-1]

    if m1_source not in ('target', 'particles'):
        raise ValueError("m1_source must be 'target' or 'particles'")
    if m1_source == 'target' and m1_target is None:
        m1_source = 'particles'   # no barphi_e saved for this run -- fall back
    m1 = m1_target if m1_source == 'target' else m1_particles

    m1_discrepancy = None
    if m1_target is not None:
        denom = m1_target.norm().item()
        m1_discrepancy = float((m1_target - m1_particles).norm() / denom) if denom > 0 else float('nan')

    theta1_dot_m1 = float(theta_1 @ m1)
    log_Z = H_bound + theta1_dot_m1

    rng = np.random.default_rng(seed)
    B = phi_xt.shape[0]
    boot_vals = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, B, size=B)
        boot_vals[b] = float(theta_1 @ phi_xt[idx].mean(0))
    theta1_dot_m1_bootstrap_std = float(boot_vals.std(ddof=1))

    return {
        'log_Z_bound': log_Z, 'H_bound': H_bound, 'theta1_dot_m1': theta1_dot_m1,
        'm1_source': m1_source, 'i_final': i_final_pos,
        'm1_target': m1_target, 'm1_particles': m1_particles, 'm1_discrepancy': m1_discrepancy,
        'theta1_dot_m1_bootstrap_std': theta1_dot_m1_bootstrap_std,
        'entropy_bound': eb,
    }


_SIGMA_RE = re.compile(r'sigma\s*=\s*([0-9.eE+-]+)')


def _sigma_from_key(key):
    m = _SIGMA_RE.search(key)
    if m is None:
        raise ValueError(f"Could not parse sigma from results key {key!r}; pass sigma_fn=...")
    return float(m.group(1))


def summarize_log_Z_bound(results, potentials, device='cpu', theta_keys=('theta_t', 'Theta_reg'),
                           sigma_fn=None, **kwargs):
    """Sweep log_Z_bound() over every run in ``results``, grouped by
    (sigma, theta_key) -- sigma parsed from each results key via
    ``sigma_fn`` (default: a ``'sigma = <value>'`` regex, matching the
    convention already used in ``jets/jets_KL_divergence.ipynb``). Prints a
    table: sigma, sigma^2, H_bound, theta1_dot_m1, log_Z_bound, and its
    uncertainty -- across-seed std when >=2 runs share a sigma (this also
    captures the theta_t trajectory's run-to-run variance, not just the
    particle-bootstrap piece), else the single-run particle-bootstrap std,
    explicitly labeled as such (not a total uncertainty -- see
    log_Z_bound's docstring). Returns the summary DataFrame; no plotting.
    """
    sigma_fn = sigma_fn or _sigma_from_key
    rows = []
    for key in results:
        sigma = sigma_fn(key)
        for tk in theta_keys:
            try:
                out = log_Z_bound(results, key, tk, potentials, device=device, **kwargs)
            except Exception as e:
                print(f"[summarize_log_Z_bound] skipping key={key!r} theta_key={tk!r}: {e}")
                continue
            rows.append({
                'key': key, 'sigma': sigma, 'sigma2': sigma ** 2, 'theta_key': tk,
                'H_bound': out['H_bound'], 'theta1_dot_m1': out['theta1_dot_m1'],
                'log_Z_bound': out['log_Z_bound'], 'bootstrap_std': out['theta1_dot_m1_bootstrap_std'],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        print("No runs summarized.")
        return df

    summary_rows = []
    for (sigma, tk), group in df.groupby(['sigma', 'theta_key']):
        n_runs = len(group)
        if n_runs >= 2:
            unc = float(group['log_Z_bound'].std(ddof=1))
            unc_source = 'across_seed_std'
        else:
            unc = float(group['bootstrap_std'].iloc[0])
            unc_source = 'particle_bootstrap_std (single run -- NOT total uncertainty)'
        summary_rows.append({
            'sigma': sigma, 'sigma2': sigma ** 2, 'theta_key': tk, 'n_runs': n_runs,
            'H_bound': float(group['H_bound'].mean()),
            'theta1_dot_m1': float(group['theta1_dot_m1'].mean()),
            'log_Z_bound': float(group['log_Z_bound'].mean()),
            'uncertainty': unc, 'uncertainty_source': unc_source,
        })
    summary = pd.DataFrame(summary_rows).sort_values(['theta_key', 'sigma']).reset_index(drop=True)
    print(summary.to_string(index=False))
    return summary