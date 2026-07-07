import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import gammaln, gammainc
from scipy.optimize import minimize, minimize_scalar
from scipy.integrate import quad


class Scalar_GGD_GenGamma:
    """
    Two-region potential per channel:
        bulk  : weight g(x),     suff. stat |x|^alpha          -> generalized Gaussian
        outer : weight 1-g(x),   suff. stats {|x|^beta, log|x|} -> generalized Gamma
                p_outer(x) ~ |x|^{-theta3} * exp(-theta2 * |x|^beta)

    g(x) = sigmoid(-(|x|-c)/s)  (smooth partition of unity, used in forward()/grad())

    Fitting uses HARD cutoff |x|<c / |x|>=c subsets (well-conditioned,
    avoids the near-boundary-mass bias smooth weighting caused before).
    Both regions are fit with TRUNCATED normalization (correctly
    accounting for the cutoff), so there is no truncation bias and no
    anchoring hack needed for plotting.

    forward()/grad() return/operate on 3 channels per scale:
        [bulk_energy, outer_beta_energy, outer_log_energy]  -> (B, 3J)
    matching the three sufficient statistics above.
    """

    def __init__(self, filters, bulk_quantile=0.95, trans_frac=0.15,
                 alpha_bounds=(0.2, 8.0), beta_bounds=(0.3, 3.0),
                 theta3_bounds=(-1.0, 1.0), min_region_samples=30,
                 eps_abs=1e-6):
        self._filters_3x = None # cached repeated filter bank

        self.filters = filters
        self.num_coefficients = 3 * filters.shape[1]
        self.bulk_quantile = bulk_quantile
        self.trans_frac = trans_frac
        self.alpha_bounds = alpha_bounds
        self.beta_bounds = beta_bounds
        self.theta3_bounds = theta3_bounds
        self.min_region_samples = min_region_samples
        self.eps_abs = eps_abs

        self.alpha = self.scale_bulk = None   # bulk:  (J,)
        self.beta = self.theta2 = self.theta3 = None  # outer: (J,)
        self.c = self.s = None                # boundary, transition width (J,)
        self.pi_bulk = self.pi_outer = None

    @property
    def is_fitted(self):
        return self.alpha is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("must call fit_reference first.")

    # ------------------------------------------------------------------
    # Bulk: truncated generalized-Gaussian MLE on |x| < c
    # (1D profile over alpha; scale closed-form via regularized lower
    # incomplete gamma normalizer -- same stable construction as before)
    # ------------------------------------------------------------------
    @staticmethod
    def _ggd_logmass_leq(c, alpha, scale):
        val = gammainc(1.0 / alpha, (c / scale) ** alpha)  # P(|Z|<=c)
        return np.log(max(val, 1e-300))

    @classmethod
    def _fit_bulk_ggd(cls, h, c, alpha_bounds=(0.2, 8.0)):
        h = np.asarray(h, dtype=float)
        n = h.size
        if n < 5:
            return 1.0, float(np.std(h) + 1e-8)
        ah = np.abs(h)

        def neg_ll(theta):
            alpha = np.clip(theta[0], *alpha_bounds)
            scale = max(theta[1], 1e-8)
            logpdf = (np.log(alpha) - np.log(2.0) - np.log(scale)
                      - gammaln(1.0 / alpha) - (ah / scale) ** alpha)
            log_mass = cls._ggd_logmass_leq(c, alpha, scale)
            return -np.sum(logpdf) + n * log_mass

        alpha0 = 1.5
        scale0 = (alpha0 * np.mean(ah ** alpha0)) ** (1.0 / alpha0)
        res = minimize(neg_ll, x0=[alpha0, scale0], method="Nelder-Mead",
                        options=dict(xatol=1e-5, fatol=1e-5, maxiter=5000))
        alpha_hat = float(np.clip(res.x[0], *alpha_bounds))
        scale_hat = float(max(res.x[1], 1e-8))
        return alpha_hat, scale_hat

    # ------------------------------------------------------------------
    # Outer: generalized-Gamma MLE on |x| >= c
    #   for fixed beta: NLL(theta2,theta3) is convex (exponential family)
    #   -> stable 2D solve.  beta itself: 1D bounded profile scan.
    # ------------------------------------------------------------------
    @staticmethod
    def _outer_logZ(theta2, theta3, beta, c):
        # Map integration from [c, inf] to [0, 1] via v = c/u
        # This completely avoids quad failing on long heavy tails
        lam = theta2 * (c ** beta)
        
        def f(v):
            if v < 1e-10:  # Guard against overflow in v**-beta
                return 0.0
            return (v ** (theta3 - 2.0)) * np.exp(-lam * (v ** -beta))
        
        I_v, _ = quad(f, 0, 1, limit=200)
        if not np.isfinite(I_v) or I_v <= 0:
            return np.inf
            
        # log(2 * I) = log(2) + (1 - theta3)*log(c) + log(I_v)
        return np.log(2.0) + (1.0 - theta3) * np.log(c) + np.log(I_v)

    @classmethod
    def _fit_outer_given_beta(cls, ah_outer, c, beta, theta3_bounds):
        n = ah_outer.size
        mean_log = np.mean(np.log(ah_outer))

        def neg_ll(params):
            log_theta2, theta3 = params
            theta2 = np.exp(log_theta2)
            theta3 = np.clip(theta3, *theta3_bounds)
            logZ = cls._outer_logZ(theta2, theta3, beta, c)
            if not np.isfinite(logZ):
                return 1e12
            ll = -theta3 * mean_log - theta2 * np.mean(ah_outer ** beta) - logZ
            return -n * ll

        x0 = [np.log(1.0 / max(c, 1e-3) ** beta), 0.0]
        res = minimize(neg_ll, x0=x0, method="Nelder-Mead",
                        options=dict(xatol=1e-5, fatol=1e-5, maxiter=3000))
        theta2 = float(np.exp(res.x[0]))
        theta3 = float(np.clip(res.x[1], *theta3_bounds))
        return theta2, theta3, float(res.fun)

    @classmethod
    def _fit_outer_gengamma(cls, h, c, beta_bounds=(0.3, 3.0),
                             theta3_bounds=(-2.0, 15.0)):
        ah_outer = np.abs(np.asarray(h, dtype=float))
        ah_outer = ah_outer[ah_outer >= c]
        if ah_outer.size < 5:
            return 1.0, 1.0 / max(c, 1e-3), 0.0

        def profile_nll(beta):
            _, _, nll = cls._fit_outer_given_beta(ah_outer, c, beta, theta3_bounds)
            return nll

        res = minimize_scalar(profile_nll, bounds=beta_bounds, method="bounded",
                               options=dict(xatol=1e-3))
        beta_hat = float(res.x)
        theta2_hat, theta3_hat, _ = cls._fit_outer_given_beta(
            ah_outer, c, beta_hat, theta3_bounds)
        return beta_hat, theta2_hat, theta3_hat

    # ------------------------------------------------------------------
    # fit_reference
    # ------------------------------------------------------------------
    def _get_filters_3x(self, device):
        if (self._filters_3x is None
                or self._filters_3x.device != device):
            self._filters_3x = self.filters.repeat(1, 3, 1).to(device)
        return self._filters_3x

    def fit_reference(self, x, bulk_quantile=None):
        if bulk_quantile is None:
            bulk_quantile = self.bulk_quantile

        self._filters_3x = None
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real.detach().cpu().numpy()

        alphas, scales_b = [], []
        betas, theta2s, theta3s = [], [], []
        cc, ss = [], []
        pi_bulks, pi_outers = [], []

        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            ah = np.abs(h)

            c_ = float(np.quantile(ah, bulk_quantile))
            c_ = max(c_, 1e-8)
            s_ = max(self.trans_frac * c_, 1e-6)

            bulk_h = h[ah < c_]
            outer_h = h[ah >= c_]
            n_b, n_o = bulk_h.size, outer_h.size

            pi_bulk_j = n_b / max(n_b + n_o, 1)
            pi_outer_j = n_o / max(n_b + n_o, 1)
            
            print(f"[GGD/GenGamma][ch {j}] c={c_:.4f} s={s_:.4f} | "
                  f"N_bulk={n_b} N_outer={n_o}")
            if min(n_b, n_o) < self.min_region_samples:
                print(f"[GGD/GenGamma][ch {j}] WARNING: a region has "
                      f"< {self.min_region_samples} samples; adjust bulk_quantile.")

            alpha_j, scale_j = self._fit_bulk_ggd(bulk_h, c_, self.alpha_bounds)
            beta_j, theta2_j, theta3_j = self._fit_outer_gengamma(
                outer_h, c_, self.beta_bounds, self.theta3_bounds)

            print(f"[GGD/GenGamma][ch {j}] bulk(alpha={alpha_j:.3f},"
                  f"scale={scale_j:.4f}) | outer(beta={beta_j:.3f},"
                  f"theta2={theta2_j:.4f},theta3={theta3_j:.3f})")

            alphas += [alpha_j]; scales_b += [scale_j]
            betas += [beta_j]; theta2s += [theta2_j]; theta3s += [theta3_j]
            cc += [c_]; ss += [s_]
            pi_bulks += [pi_bulk_j]; pi_outers += [pi_outer_j]
            

        dtype = x.dtype if x.is_floating_point() else torch.float32
        dev = x.device
        mk = lambda L: torch.tensor(L, dtype=dtype, device=dev)
        self.alpha, self.scale_bulk = mk(alphas), mk(scales_b)
        self.beta, self.theta2, self.theta3 = mk(betas), mk(theta2s), mk(theta3s)
        self.c, self.s = mk(cc), mk(ss)
        self.pi_bulk = mk(pi_bulks)
        self.pi_outer = mk(pi_outers)

    def fit(self, x, **kwargs):
        return self.fit_reference(x, **kwargs)

    # ------------------------------------------------------------------
    # Differentiable forward/grad (smooth window g(x), for MGD machinery)
    # ------------------------------------------------------------------
    def _g(self, z, device):
        c = self.c.to(device)[None, :, None]
        s = self.s.to(device)[None, :, None]
        az = torch.sqrt(z ** 2 + self.eps_abs)
        return torch.sigmoid(-(az - c) / s)   # ~1 bulk, ~0 outer

    def forward(self, x, *args):
        self._check_fitted()

        filters = self.filters.to(x.device)

        with torch.no_grad():

            z = torch.fft.ifft(filters * torch.fft.fft(x)).real

            device = x.device

            g = self._g(z, device)
            az = torch.sqrt(z ** 2 + self.eps_abs)

            alpha = self.alpha.to(device)[None, :, None]
            beta = self.beta.to(device)[None, :, None]

            phi_bulk = (g * az ** alpha).mean(-1)
            phi_outer_beta = ((1 - g) * az ** beta).mean(-1)
            phi_outer_log = ((1 - g) * torch.log(az)).mean(-1)

        return torch.cat(
            [phi_bulk, phi_outer_beta, phi_outer_log],
            dim=1
        )

    def grad(self, x, v=None, means=None):
        self._check_fitted()

        device = x.device
        filters = self.filters.to(device)

        with torch.no_grad():
            # ------------------------------------------------------------
            # Wavelet coefficients
            # ------------------------------------------------------------
            z = torch.fft.ifft(filters * torch.fft.fft(x)).real
            az = torch.sqrt(z ** 2 + self.eps_abs)
            sign_z = z / az
            # ------------------------------------------------------------
            # Parameters
            # ------------------------------------------------------------
            c = self.c.to(device)[None, :, None]
            s = self.s.to(device)[None, :, None]

            alpha = self.alpha.to(device)[None, :, None]
            beta = self.beta.to(device)[None, :, None]
            # ------------------------------------------------------------
            # Smooth window
            # ------------------------------------------------------------
            g = torch.sigmoid(-(az - c) / s)

            dg = -g * (1.0 - g) * sign_z / s
            # ------------------------------------------------------------
            # Derivatives of sufficient statistics
            # ------------------------------------------------------------
            # d/dz (|z|^alpha)
            daz_alpha = alpha * z * az ** (alpha - 2.0)

            # d/dz (|z|^beta)
            daz_beta = beta * z * az ** (beta - 2.0)

            # d/dz log|z|
            dlog = z / az ** 2
            # ------------------------------------------------------------
            # Product rule
            # ------------------------------------------------------------
            D_bulk = dg * az ** alpha + g * daz_alpha

            D_outer_beta = (
                -dg * az ** beta
                + (1.0 - g) * daz_beta
            )
            D_outer_log = (
                -dg * torch.log(az)
                + (1.0 - g) * dlog
            )
            D_all = torch.cat(
                [D_bulk, D_outer_beta, D_outer_log],
                dim=1
            )
            # ------------------------------------------------------------
            # Backprojection
            # ------------------------------------------------------------
            f3 = self._get_filters_3x(device)
            grad_coeff = torch.fft.ifft(
                torch.fft.fft(D_all) * f3
            ).real / x.shape[-1]

        if v is None:
            return grad_coeff

        return (grad_coeff * v[None, :, None]).sum(1, keepdim=True)

    def plot_fit(self, x, label="Wavelet", n_grid=1000, fit_if_needed=True):
        if fit_if_needed and not self.is_fitted:
            self.fit_reference(x)
        self._check_fitted()

        filters = self.filters.to(x.device)
        wt = torch.fft.ifft(filters * torch.fft.fft(x)).real
        n_wavelets = filters.shape[1]

        alpha  = self.alpha.cpu().numpy()
        scale_b = self.scale_bulk.cpu().numpy()
        beta   = self.beta.cpu().numpy()
        theta2 = self.theta2.cpu().numpy()
        theta3 = self.theta3.cpu().numpy()
        c_all  = self.c.cpu().numpy()

        def safe_exp(lp):
            return np.exp(np.clip(lp, -500.0, 500.0))

        for j in range(n_wavelets):
            h = wt[:, j, :].detach().cpu().flatten().numpy()
            h = h[np.isfinite(h)]
            if h.size == 0:
                continue
            ah = np.abs(h)
            c_j         = float(c_all[j])
            a_j, sb_j   = alpha[j], scale_b[j]
            b_j, t2_j, t3_j = beta[j], theta2[j], theta3[j]

            # FIX: actual data maximum, not a quantile
            xmax = float(ah.max()) * 1.02

            pi_outer_j  = float((ah >= c_j).mean())
            pi_bulk_j   = 1.0 - pi_outer_j
            log_pi_bulk  = np.log(max(pi_bulk_j,  1e-300))
            log_pi_outer = np.log(max(pi_outer_j, 1e-300))

            x_bulk = np.linspace(-c_j, c_j, n_grid)
            log_mass_bulk = Scalar_GGD_GenGamma._ggd_logmass_leq(c_j, a_j, sb_j)
            logpdf_bulk = (np.log(a_j) - np.log(2.0) - np.log(sb_j)
                        - gammaln(1.0 / a_j)
                        - (np.abs(x_bulk) / sb_j) ** a_j
                        - log_mass_bulk + log_pi_bulk)

            # outer grid now runs all the way to ah.max()
            x_outer_pos = np.linspace(c_j, xmax, n_grid)
            logZ_outer  = Scalar_GGD_GenGamma._outer_logZ(t2_j, t3_j, b_j, c_j)
            logpdf_outer_pos = (-t3_j * np.log(x_outer_pos)
                                - t2_j * x_outer_pos ** b_j
                                - logZ_outer + log_pi_outer)
            x_outer  = np.concatenate([-x_outer_pos[::-1], x_outer_pos])
            logpdf_outer = np.concatenate([logpdf_outer_pos[::-1], logpdf_outer_pos])

            # data-driven y limits so clipped curves never rescale the axis
            hist_vals, _ = np.histogram(h, bins=150, density=True)
            hist_pos = hist_vals[hist_vals > 0]
            y_min = hist_pos.min() * 0.3
            y_max = hist_pos.max() * 5.0

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(h, bins=150, density=True, log=True,
                    alpha=0.4, color="steelblue", label="data")
            ax.plot(x_bulk,  safe_exp(logpdf_bulk),  lw=2, color="tab:orange",
                    label=f"bulk (GGD)     α={a_j:.2f} sc={sb_j:.3f} π={pi_bulk_j:.3f}")
            ax.plot(x_outer, safe_exp(logpdf_outer), lw=2, color="tab:red",
                    label=f"outer (genΓ)  β={b_j:.2f} θ2={t2_j:.3f} θ3={t3_j:.2f} π={pi_outer_j:.3f}")
            ax.axvline( c_j, color="black", ls="--", lw=1, alpha=0.4)
            ax.axvline(-c_j, color="black", ls="--", lw=1, alpha=0.4)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("Coefficient value")
            ax.set_ylabel("Log density")
            ax.set_title(f"{label} — channel {j}  (c={c_j:.3f})")
            ax.legend(frameon=False, fontsize=8)
            plt.tight_layout()
            plt.show()
