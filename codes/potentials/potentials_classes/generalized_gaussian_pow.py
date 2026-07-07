import numpy as np
import torch
from scipy.special import gammaln, gammainc
from scipy.optimize import minimize, minimize_scalar
from scipy.integrate import quad
from scipy.stats import norm

class Scalar_GGD_GGD_Pow:
    """
    Three-region potential per wavelet channel.

    Mathematical self
    ------------------
    Two smooth sigmoid windows g1, g2 define a partition of unity on |x|:

        g1(x) = sigmoid(-(|x| - c1) / s1)   ~1 for |x|<<c1, ~0 for |x|>>c1
        g2(x) = sigmoid(-(|x| - c2) / s2)   ~1 for |x|<<c2, ~0 for |x|>>c2

        w_bulk(x) = g1(x)                     active near zero
        w_mid(x)  = g2(x) - g1(x)            active in shoulder region
        w_tail(x) = 1 - g2(x)                active in far tail

    Three sufficient statistics (per channel):
        phi_bulk(x) = w_bulk(x) * |x|^alpha1   -> maxent: GGD exp(-(|x|/s)^alpha1)
        phi_mid(x)  = w_mid(x)  * |x|^alpha2   -> maxent: GGD exp(-(|x|/s)^alpha2)
        phi_tail(x) = w_tail(x) * log|x|       -> maxent: power law |x|^{-beta}

    Boundaries
    ----------
        c1 = E[|x|]  (empirical mean absolute value)
             Natural bulk/mid split: sits at the "body" of the distribution
             regardless of tail heaviness, unlike sigma which is inflated
             by outliers. For a Gaussian: E[|x|] = sigma*sqrt(2/pi).
             For heavy-tailed: E[|x|] << sigma.

        c2 = high quantile of |x| (default: 99th percentile)
             Where the power-law tail clearly dominates, verified by the
             Hill estimator producing a stable estimate.

    Fitting: hard-cutoff disjoint subsets, truncation-corrected MLE
    ---------------------------------------------------------------
    Each region is fit on its own hard subset with a truncation-corrected
    likelihood — the normalizer integrates only over the region's own
    support, not the full line. This avoids the truncation bias
    (scale/sigma underestimation) that caused overshoot in earlier versions.

        Bulk and mid: 2D Nelder-Mead over (alpha, scale) with truncated
            GGD normalizer via regularized incomplete gamma.
        Tail: Hill estimator — exact closed-form MLE for a Pareto tail
            index given threshold c2. Pure power law p(x) ~ x^{-beta},
            NO exponential component, preserving fat tails in samples.

    forward() -> (B, 3J): [phi_bulk_mean, phi_mid_mean, phi_tail_mean]
    grad()    -> (B, 3J, T) or (B, 1, T) if v given — fully analytic,
                 no autograd graph retained, O(BJT) memory.
    """

    def __init__(self, filters,
                 tail_quantile=0.99,
                 trans_frac=0.10,
                 alpha_bounds=(0.2, 8.0),
                 beta_bounds=(1.1, 30.0),
                 min_region_samples=30,
                 eps_abs=1e-6):
        self.filters = filters
        self.num_coefficients = 3 * filters.shape[1]
        self.tail_quantile = tail_quantile
        self.trans_frac = trans_frac
        self.alpha_bounds = alpha_bounds
        self.beta_bounds = beta_bounds
        self.min_region_samples = min_region_samples
        self.eps_abs = eps_abs

        # fitted parameters (J,) each
        self.alpha1 = self.scale1 = None   # bulk GGD
        self.alpha2 = self.scale2 = None   # mid  GGD
        self.beta   = None                 # tail power law
        self.c1 = self.c2 = None           # boundaries
        self.s1 = self.s2 = None           # transition widths
        self.pi_bulk = self.pi_mid = self.pi_tail = None  # mixture weights

        self._filters_3x = None            # cached repeated filter bank

    @property
    def is_fitted(self):
        return self.alpha1 is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("must call fit_reference first.")

    # ------------------------------------------------------------------
    # Truncation-corrected GGD MLE
    # log P(a <= |Z| < b) for GGD uses regularized incomplete gamma:
    #   P(|Z| <= t) = gammainc(1/alpha, (t/scale)^alpha)
    # The truncated NLL adds n * log[P(a <= |Z| < b)] as a correction,
    # which is what was missing in the non-truncated version and caused
    # the scale to be underestimated (curves too narrow / too tall).
    # ------------------------------------------------------------------
    @staticmethod
    def _ggd_cdf_abs(t, alpha, scale):
        """P(|Z| <= t) for a GGD with given alpha, scale."""
        return float(gammainc(1.0 / alpha, (max(t, 0.0) / scale) ** alpha))

    @classmethod
    def _fit_ggd_truncated(cls, h, lo, hi, alpha_bounds):
        """
        Truncation-corrected GGD MLE on the subset lo <= |x| < hi.
        Normalizer: P(lo <= |Z| < hi) = CDF(hi) - CDF(lo).
        Both alpha and scale are optimized (2D Nelder-Mead), well-
        conditioned because we have a good closed-form starting point.
        """
        h = np.asarray(h, dtype=float)
        ah = np.abs(h)
        n = ah.size
        if n < 5:
            return 1.0, float(np.std(h) + 1e-8)

        def neg_ll(theta):
            alpha = float(theta[0])
            scale = float(theta[1])
            logpdf = (np.log(alpha) - np.log(2.0) - np.log(scale)
                      - gammaln(1.0 / alpha)
                      - (ah / scale) ** alpha)
            cdf_hi = cls._ggd_cdf_abs(hi, alpha, scale)
            cdf_lo = cls._ggd_cdf_abs(lo, alpha, scale) if lo > 0 else 0.0
            mass = cdf_hi - cdf_lo
            if mass <= 1e-12:
                return np.inf
                
            return -np.sum(logpdf) + n * np.log(mass)

        # closed-form untruncated init as starting point
        alpha0 = 1.5
        scale0 = max((alpha0 * np.mean(ah ** alpha0)) ** (1.0 / alpha0), 1e-8)
        res = minimize(
            neg_ll,
            x0=[alpha0, scale0],
            method="L-BFGS-B",
            bounds=[
                alpha_bounds,
                (1e-8, None)
            ]
        )
        print(res.success)
        print(res.message)
        print(res.x)
        print(res.fun)

        alpha_hat = float(np.clip(res.x[0], *alpha_bounds))
        scale_hat = float(max(res.x[1], 1e-8))
        return alpha_hat, scale_hat

    # ------------------------------------------------------------------
    # Hill estimator: pure power-law tail, no exponential cutoff.
    # This is the exact closed-form MLE for p(x) ~ x^{-beta} on [c2, inf).
    # beta = 1 + 1/mean(log(x/c2)) for x > c2.
    # ------------------------------------------------------------------
    @staticmethod
    def _fit_hill_beta(h, c2, beta_bounds):
        ah = np.abs(np.asarray(h, dtype=float))
        excess = ah[ah > c2]
        if excess.size < 5:
            return float(beta_bounds[0] + 1.0)
        xi = float(np.mean(np.log(excess / c2)))
        if xi <= 1e-8:
            return float(beta_bounds[1])
        return float(np.clip(1.0 + 1.0 / xi, *beta_bounds))

    # ------------------------------------------------------------------
    # fit_reference
    # ------------------------------------------------------------------
    def fit_reference(self, x, tail_quantile=None):
        if tail_quantile is None:
            tail_quantile = self.tail_quantile

        self._filters_3x = None  # invalidate filter cache
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real.detach().cpu().numpy()

        a1s, s1s, a2s, s2s, betas = [], [], [], [], []
        c1s, c2s, sw1s, sw2s = [], [], [], []
        pi_bs, pi_ms, pi_ts = [], [], []

        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            ah = np.abs(h)

            # --- c1: E[|x|], the professor's natural bulk/mid boundary ---
            c1_ = float(np.mean(ah))
            c1_ = max(c1_, 1e-8)

            # --- c2: high quantile where power law clearly dominates ---
            c2_ = float(np.quantile(ah, tail_quantile))
            c2_ = max(c2_, c1_ * 1.5)  # enforce c2 > c1

            # transition widths: fraction of each region's width
            sw1_ = max(self.trans_frac * c1_, 1e-6)
            sw2_ = max(self.trans_frac * (c2_ - c1_), 1e-6)

            bulk_h = h[ah < c1_]
            mid_h  = h[(ah >= c1_) & (ah < c2_)]
            tail_h = h[ah >= c2_]
            n_b, n_m, n_t = bulk_h.size, mid_h.size, tail_h.size

            pi_b = n_b / max(n_b + n_m + n_t, 1)
            pi_m = n_m / max(n_b + n_m + n_t, 1)
            pi_t = n_t / max(n_b + n_m + n_t, 1)

            print(f"[GGD/GGD/Pow][ch {j}]  "
                  f"c1={c1_:.4f}(E[|x|])  c2={c2_:.4f}({tail_quantile*100:.0f}pct) | "
                  f"N_bulk={n_b}({pi_b:.2%})  N_mid={n_m}({pi_m:.2%})  "
                  f"N_tail={n_t}({pi_t:.2%})")

            if min(n_b, n_m, n_t) < self.min_region_samples:
                print(f"[GGD/GGD/Pow][ch {j}] WARNING: a region has "
                      f"< {self.min_region_samples} samples.")

            a1_, s1_ = self._fit_ggd_truncated(bulk_h, 0.0, c1_, self.alpha_bounds)
            a2_, s2_ = self._fit_ggd_truncated(mid_h,  c1_, c2_, self.alpha_bounds)
            b_       = self._fit_hill_beta(h, c2_, self.beta_bounds)

            print(f"[GGD/GGD/Pow][ch {j}]  "
                  f"bulk(alpha={a1_:.3f}, scale={s1_:.4f}) | "
                  f"mid(alpha={a2_:.3f}, scale={s2_:.4f}) | "
                  f"tail(beta={b_:.3f})")

            a1s += [a1_];  s1s += [s1_]
            a2s += [a2_];  s2s += [s2_]
            betas += [b_]
            c1s += [c1_];  c2s += [c2_]
            sw1s += [sw1_]; sw2s += [sw2_]
            pi_bs += [pi_b]; pi_ms += [pi_m]; pi_ts += [pi_t]

        dtype = x.dtype if x.is_floating_point() else torch.float32
        dev   = x.device
        mk    = lambda L: torch.tensor(L, dtype=dtype, device=dev)

        self.alpha1, self.scale1 = mk(a1s), mk(s1s)
        self.alpha2, self.scale2 = mk(a2s), mk(s2s)
        self.beta = mk(betas)
        self.c1, self.c2 = mk(c1s), mk(c2s)
        self.s1, self.s2 = mk(sw1s), mk(sw2s)
        self.pi_bulk, self.pi_mid, self.pi_tail = mk(pi_bs), mk(pi_ms), mk(pi_ts)

    def fit(self, x, **kwargs):
        return self.fit_reference(x, **kwargs)

    # ------------------------------------------------------------------
    # Smooth windows — partition of unity on |x| into three regions.
    # g1 ~ 1 near zero (bulk), g2 ~ 1 below c2 (bulk+mid).
    # w_bulk = g1,  w_mid = g2-g1,  w_tail = 1-g2.
    # ------------------------------------------------------------------
    def _windows(self, z, device):
        c1 = self.c1.to(device)[None, :, None]
        c2 = self.c2.to(device)[None, :, None]
        s1 = self.s1.to(device)[None, :, None]
        s2 = self.s2.to(device)[None, :, None]
        az = torch.sqrt(z ** 2 + self.eps_abs)
        g1 = torch.sigmoid(-(az - c1) / s1)
        g2 = torch.sigmoid(-(az - c2) / s2)
        return g1, g2 - g1, 1.0 - g2   # w_bulk, w_mid, w_tail

    # ------------------------------------------------------------------
    # forward — O(BJT), no graph, safe inside torch.no_grad()
    # ------------------------------------------------------------------
    def forward(self, x, *args):
        self._check_fitted()
        filters = self.filters.to(x.device)
        with torch.no_grad():
            z  = torch.fft.ifft(filters * torch.fft.fft(x)).real
            az = torch.sqrt(z ** 2 + self.eps_abs)
            w_b, w_m, w_t = self._windows(z, x.device)
            alpha1 = self.alpha1.to(x.device)[None, :, None]
            alpha2 = self.alpha2.to(x.device)[None, :, None]
            phi_bulk = (w_b * az ** alpha1).mean(-1)
            phi_mid  = (w_m * az ** alpha2).mean(-1)
            phi_tail = (w_t * torch.log(az)).mean(-1)
        return torch.cat([phi_bulk, phi_mid, phi_tail], dim=1)   # (B, 3J)

    # ------------------------------------------------------------------
    # grad — fully analytic, no autograd graph retained.
    #
    # Product rule for each windowed potential phi_k = w_k(z) * f_k(z):
    #   d phi_k / dz = (dw_k/dz) * f_k(z) + w_k(z) * (df_k/dz)
    #
    # Window derivatives (chain rule through sigmoid and az=sqrt(z²+eps)):
    #   daz/dz = z/az                              (smooth |z|')
    #   dg1/dz = -g1*(1-g1) * (z/az) / s1
    #   dg2/dz = -g2*(1-g2) * (z/az) / s2
    #   dw_bulk/dz =  dg1/dz
    #   dw_mid/dz  =  dg2/dz - dg1/dz
    #   dw_tail/dz = -dg2/dz
    #
    # Potential derivatives:
    #   d/dz [az^alpha]  = alpha * z * az^(alpha-2)
    #   d/dz [log(az)]   = z / az^2
    # ------------------------------------------------------------------
    def _get_filters_3x(self, device):
        if (self._filters_3x is None
                or self._filters_3x.device != device):
            self._filters_3x = self.filters.repeat(1, 3, 1).to(device)
        return self._filters_3x

    def grad(self, x, v=None, means=None):
        self._check_fitted()
        device  = x.device
        filters = self.filters.to(device)

        with torch.no_grad():
            z  = torch.fft.ifft(filters * torch.fft.fft(x)).real
            az = torch.sqrt(z ** 2 + self.eps_abs)
            sign_z = z / az                       # smooth sign(z), in (-1,1)

            c1 = self.c1.to(device)[None, :, None]
            c2 = self.c2.to(device)[None, :, None]
            s1 = self.s1.to(device)[None, :, None]
            s2 = self.s2.to(device)[None, :, None]
            alpha1 = self.alpha1.to(device)[None, :, None]
            alpha2 = self.alpha2.to(device)[None, :, None]

            g1 = torch.sigmoid(-(az - c1) / s1)
            g2 = torch.sigmoid(-(az - c2) / s2)
            w_b =  g1
            w_m  =  g2 - g1
            w_t  =  1.0 - g2

            dg1 = -g1 * (1.0 - g1) * sign_z / s1
            dg2 = -g2 * (1.0 - g2) * sign_z / s2
            dw_b =  dg1
            dw_m  =  dg2 - dg1
            dw_t  = -dg2

            # d/dz [az^alpha] = alpha * z * az^(alpha-2)
            daz_a1 = alpha1 * z * az ** (alpha1 - 2.0)
            daz_a2 = alpha2 * z * az ** (alpha2 - 2.0)
            # d/dz [log az]  = z / az^2
            dlog_az = z / az ** 2

            D_bulk = dw_b  * az ** alpha1 + w_b  * daz_a1
            D_mid  = dw_m  * az ** alpha2 + w_m  * daz_a2
            D_tail = dw_t  * torch.log(az) + w_t  * dlog_az

            D_all = torch.cat([D_bulk, D_mid, D_tail], dim=1)   # (B, 3J, T)

            f3 = self._get_filters_3x(device)
            grad_coeff = torch.fft.ifft(
                torch.fft.fft(D_all) * f3
            ).real / x.shape[-1]                                 # (B, 3J, T)

        if v is None:
            return grad_coeff
        return (grad_coeff * v[None, :, None]).sum(1, keepdim=True)  # (B,1,T)

    def plot_fit(self, x, label="Wavelet", n_grid=1000, fit_if_needed=True):
        """
        Plots the histogram of wavelet coefficients against the fitted 
        GGD (bulk), GGD (mid), and Power Law (tail) distributions.
        """
        import matplotlib.pyplot as plt
        from scipy.special import gammaln

        if fit_if_needed and not self.is_fitted:
            self.fit_reference(x)
        self._check_fitted()

        filters = self.filters.to(x.device)
        # Extract coefficients exactly as the self does
        wt = torch.fft.ifft(filters * torch.fft.fft(x)).real
        n_wavelets = filters.shape[1]

        # Extract fitted parameters using 'self' instead of 'self'
        a1  = self.alpha1.cpu().numpy()
        s1  = self.scale1.cpu().numpy()
        a2  = self.alpha2.cpu().numpy()
        s2  = self.scale2.cpu().numpy()
        b   = self.beta.cpu().numpy()
        c1a = self.c1.cpu().numpy()
        c2a = self.c2.cpu().numpy()
        pb  = self.pi_bulk.cpu().numpy()
        pm  = self.pi_mid.cpu().numpy()
        pt  = self.pi_tail.cpu().numpy()

        def safe_exp(lp):
            """Clip before exp so neither overflow nor exact-zero underflow reaches matplotlib."""
            return np.exp(np.clip(lp, -500.0, 500.0))

        for j in range(n_wavelets):
            h = wt[:, j, :].detach().cpu().flatten().numpy()
            h = h[np.isfinite(h)]
            if h.size == 0:
                continue
            
            ah = np.abs(h)
            c1_j, c2_j = float(c1a[j]), float(c2a[j])

            # Use actual data maximum, not a quantile, to ensure the tail curve reaches the last bin
            xmax = float(ah.max()) * 1.02

            # --- Bulk: truncated GGD on [-c1, c1], scaled by pi_bulk ---
            x_bulk = np.linspace(-c1_j, c1_j, n_grid)
            cdf_hi = self._ggd_cdf_abs(c1_j, a1[j], s1[j])
            mass_b = max(cdf_hi, 1e-300)
            
            logp_b = (np.log(a1[j]) - np.log(2.0) - np.log(s1[j])
                      - gammaln(1.0 / a1[j])
                      - (np.abs(x_bulk) / s1[j])**a1[j]
                      - np.log(mass_b) + np.log(max(pb[j], 1e-300)))

            # --- Mid: truncated GGD on [c1, c2], scaled by pi_mid ---
            x_mid_pos = np.linspace(c1_j, c2_j, n_grid // 2)
            cdf_hi2   = self._ggd_cdf_abs(c2_j, a2[j], s2[j])
            cdf_lo2   = self._ggd_cdf_abs(c1_j, a2[j], s2[j])
            mass_m    = max(cdf_hi2 - cdf_lo2, 1e-300)
            
            logp_m_pos = (np.log(a2[j]) - np.log(2.0) - np.log(s2[j])
                          - gammaln(1.0 / a2[j])
                          - (x_mid_pos / s2[j])**a2[j]
                          - np.log(mass_m) + np.log(max(pm[j], 1e-300)))
            
            x_mid  = np.concatenate([-x_mid_pos[::-1], x_mid_pos])
            logp_m = np.concatenate([logp_m_pos[::-1], logp_m_pos])

            # --- Tail: power law on [c2, xmax], scaled by pi_tail ---
            x_tail_pos = np.linspace(c2_j, xmax, n_grid // 2)
            logp_t_pos = (np.log(b[j] - 1.0) - np.log(2.0 * c2_j)
                          - b[j] * np.log(x_tail_pos / c2_j)
                          + np.log(max(pt[j], 1e-300)))
            
            x_tail  = np.concatenate([-x_tail_pos[::-1], x_tail_pos])
            logp_t  = np.concatenate([logp_t_pos[::-1], logp_t_pos])

            hist_vals, _ = np.histogram(h, bins=150, density=True)
            hist_pos = hist_vals[hist_vals > 0]
            if hist_pos.size == 0:
                continue
                
            y_min = hist_pos.min() * 0.3
            y_max = hist_pos.max() * 5.0

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(h, bins=150, density=True, log=True,
                    alpha=0.4, color="steelblue", label="data")
            
            ax.plot(x_bulk, safe_exp(logp_b), lw=2, color="tab:orange",
                    label=f"bulk GGD  α={a1[j]:.2f} sc={s1[j]:.3f} π={pb[j]:.2%}")
            ax.plot(x_mid,  safe_exp(logp_m), lw=2, color="tab:green",
                    label=f"mid  GGD  α={a2[j]:.2f} sc={s2[j]:.3f} π={pm[j]:.2%}")
            ax.plot(x_tail, safe_exp(logp_t), lw=2, color="tab:red",
                    label=f"tail PL   β={b[j]:.2f} π={pt[j]:.2%}")
            
            ax.axvline( c1_j, color="black", ls="--", lw=1, alpha=0.4,
                        label=f"c1=E[|x|]={c1_j:.3f}")
            ax.axvline(-c1_j, color="black", ls="--", lw=1, alpha=0.4)
            ax.axvline( c2_j, color="black", ls=":",  lw=1, alpha=0.4,
                        label=f"c2(q99)={c2_j:.3f}")
            ax.axvline(-c2_j, color="black", ls=":",  lw=1, alpha=0.4)
            
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("Coefficient value")
            ax.set_ylabel("Log density")
            ax.set_title(f"{label} — channel {j}")
            ax.legend(frameon=False, fontsize=7.5)
            
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------
    def summary(self):
        self._check_fitted()
        a1 = self.alpha1.cpu().numpy(); s1 = self.scale1.cpu().numpy()
        a2 = self.alpha2.cpu().numpy(); s2 = self.scale2.cpu().numpy()
        b  = self.beta.cpu().numpy()
        c1 = self.c1.cpu().numpy();     c2 = self.c2.cpu().numpy()
        pb = self.pi_bulk.cpu().numpy(); pm = self.pi_mid.cpu().numpy()
        pt = self.pi_tail.cpu().numpy()
        print(f"{'Ch':>3} {'c1(E|x|)':>10} {'c2(q99)':>9} | "
              f"{'a1':>5} {'sc1':>7} | {'a2':>5} {'sc2':>7} | "
              f"{'beta':>5} | {'pi_b':>5} {'pi_m':>5} {'pi_t':>5}")
        print("-" * 88)
        for j in range(len(a1)):
            print(f"{j:>3d} {c1[j]:>10.4f} {c2[j]:>9.4f} | "
                  f"{a1[j]:>5.2f} {s1[j]:>7.4f} | "
                  f"{a2[j]:>5.2f} {s2[j]:>7.4f} | "
                  f"{b[j]:>5.2f} | "
                  f"{pb[j]:>5.2%} {pm[j]:>5.2%} {pt[j]:>5.2%}")
