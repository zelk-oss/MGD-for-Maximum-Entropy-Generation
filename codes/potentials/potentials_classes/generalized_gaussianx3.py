import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.special import gammaln, gammainc
from scipy.optimize import minimize


class Scalar_GGD_GGD_GGD:
    """
    Three-region potential per wavelet channel: bulk / mid / tail,
    each a truncated GGD. See original docstring for the model itself.

    CHANGE vs original: boundary selection.
    -----------------------------------------------------------------
    v1 ('moment'): c1 = E[|x|], c2 = quantile(|x|, q).
        Fails for sparse/intermittent channels (Q>1, rare oscillatory
        events): the near-zero spike drags E[|x|] to ~0, so c1 stops
        marking an actual regime change.

    v2 ('curvature'): boundaries from kinks in the log-log survival
        function. Discarded: second derivatives amplify noise, and
        np.convolve edge padding creates spurious curvature spikes near
        the search-window boundaries, so it reliably locked onto the
        edge artifact instead of the real transition (pi_tail pinned at
        a near-constant value across channels regardless of shape).

    v3 ('likelihood', default): choose (c1, c2) by BLOCK COORDINATE
        ASCENT directly on the total log-likelihood of the 3-region
        truncated-GGD mixture -- i.e. search over candidate boundaries,
        refit all three regions at each candidate, keep the pair that
        actually fits best. This optimizes what we care about (fit
        quality) instead of a proxy shape statistic, so it can't be
        fooled by where the probability mass happens to sit. Search
        uses a subsample + coarse (low-maxiter) fits for speed; the
        final regions are refit at full precision once boundaries are
        fixed. Same pattern as the coshgt_multiregion boundary search.

    'moment' and 'curvature' kept only for comparison.
    """

    def __init__(self, filters,
                 tail_quantile=0.97,
                 trans_frac=0.10,
                 alpha_bounds=(0.2, 8.0),
                 min_region_samples=30,
                 eps_abs=1e-6,
                 boundary_method="likelihood",
                 boundary_search_subsample=20000,
                 boundary_search_iters=3,
                 boundary_search_grid=12):
        self.filters = filters
        self.num_coefficients = 3 * filters.shape[1]
        self.tail_quantile = tail_quantile
        self.trans_frac = trans_frac
        self.alpha_bounds = alpha_bounds
        self.min_region_samples = min_region_samples
        self.eps_abs = eps_abs
        self.boundary_method = boundary_method  # 'likelihood' | 'curvature' | 'moment'
        self.boundary_search_subsample = boundary_search_subsample
        self.boundary_search_iters = boundary_search_iters
        self.boundary_search_grid = boundary_search_grid

        self.alpha1 = self.scale1 = None  # bulk  (J,)
        self.alpha2 = self.scale2 = None  # mid   (J,)
        self.alpha3 = self.scale3 = None  # tail  (J,)
        self.c1 = self.c2 = None
        self.s1 = self.s2 = None
        self.pi_bulk = self.pi_mid = self.pi_tail = None
        self._filters_3x = None

    @property
    def is_fitted(self):
        return self.alpha1 is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("must call fit_reference first.")

    # ------------------------------------------------------------------
    # NEW: curvature-based boundary detection on the log-log survival fn
    # ------------------------------------------------------------------
    @staticmethod
    def _survival_curvature_boundaries(ah, n_grid=300, smooth_window=9,
                                        lo_search=(0.01, 0.5),
                                        hi_search=(0.5, 0.999)):
        """
        Find (c1, c2) as the two dominant kinks in log P(|X|>t) vs log t.

        lo_search/hi_search are fractions of the probability grid used to
        restrict where each kink is searched for (c1 in the lower half,
        c2 in the upper half) so the two boundaries can't collide.
        """
        ah = np.sort(ah[ah > 0])
        n = ah.size
        if n < 50:
            return float(np.quantile(ah, 0.5)), float(np.quantile(ah, 0.97))

        probs = np.linspace(1e-4, 1 - 1e-4, n_grid)
        t_grid = np.quantile(ah, probs)
        surv = 1.0 - probs
        logt = np.log(np.maximum(t_grid, 1e-300))
        logS = np.log(np.maximum(surv, 1e-300))

        w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        kernel = np.ones(w) / w
        logS_s = np.convolve(logS, kernel, mode="same")

        slope = np.gradient(logS_s, logt)
        curvature = np.gradient(slope, logt)

        def pick(i0, i1, fallback_q):
            i0, i1 = max(i0, 1), min(i1, n_grid - 1)
            if i1 <= i0:
                return float(np.quantile(ah, fallback_q))
            rel = np.argmax(np.abs(curvature[i0:i1]))
            return float(np.exp(logt[i0:i1][rel]))

        c1 = pick(int(lo_search[0] * n_grid), int(lo_search[1] * n_grid), 0.5)
        c2 = pick(int(hi_search[0] * n_grid), int(hi_search[1] * n_grid), 0.97)

        if c2 <= c1:
            c2 = c1 * 1.5
        return c1, c2

    # ------------------------------------------------------------------
    # Core fitting primitive: truncation-corrected GGD MLE in log-space
    # (unchanged)
    # ------------------------------------------------------------------
    @staticmethod
    def _ggd_cdf_abs(t, alpha, scale):
        if t <= 0:
            return 0.0
        return float(gammainc(1.0 / alpha, (t / scale) ** alpha))

    @staticmethod
    def _scale_floor(ah, floor_frac=0.02, min_floor=1e-10):
        """
        Robust lower bound on `scale` for a truncated region.

        Truncated-GGD MLE with alpha<1 and unbounded upper support (the
        tail region, hi=inf) has a spurious global optimum as scale->0:
        density piles up at the truncation edge and the few far outliers
        barely penalize it (classic boundary degeneracy for left-
        truncated shape<1 fits -- not a numerical accident, the
        objective genuinely scores lower there than at any "reasonable"
        (alpha, scale)). Anchoring the floor to the region's own median
        |x| means bulk/mid regions that legitimately have tiny scale
        aren't over-constrained -- only the pathological near-zero
        solution is excluded.
        """
        med = float(np.median(ah)) if ah.size else 0.0
        return max(floor_frac * med, min_floor)

    @classmethod
    def _fit_ggd_truncated(cls, h, lo, hi, alpha_bounds):
        h = np.asarray(h, dtype=float)
        ah = np.abs(h)
        n = ah.size
        if n < 5:
            return 1.0, float(np.std(h) + 1e-8)

        scale_floor = cls._scale_floor(ah)

        def neg_ll(log_theta):
            alpha = float(np.clip(np.exp(log_theta[0]), *alpha_bounds))
            scale = max(float(np.exp(log_theta[1])), scale_floor)
            logpdf = (np.log(alpha) - np.log(2.0) - np.log(scale)
                      - gammaln(1.0 / alpha)
                      - (ah / scale) ** alpha)
            cdf_hi = cls._ggd_cdf_abs(hi, alpha, scale)
            cdf_lo = cls._ggd_cdf_abs(lo, alpha, scale)
            mass = cdf_hi - cdf_lo
            if mass < 1e-300:
                return 1e12
            return -np.sum(logpdf) + n * np.log(mass)

        alpha0 = 1.5
        safe_ah = np.maximum(ah, 1e-300)
        scale0 = max((alpha0 * np.mean(safe_ah ** alpha0)) ** (1.0 / alpha0), scale_floor)
        x0 = [np.log(alpha0), np.log(scale0)]

        res = minimize(neg_ll, x0=x0, method="Nelder-Mead",
                        options=dict(xatol=1e-5, fatol=1e-5, maxiter=8000))
        alpha_hat = float(np.clip(np.exp(res.x[0]), *alpha_bounds))
        scale_hat = max(float(np.exp(res.x[1])), scale_floor)
        return alpha_hat, scale_hat

    @classmethod
    def _fit_ggd_truncated_fast(cls, h, lo, hi, alpha_bounds, maxiter=250):
        """Same as _fit_ggd_truncated but with a low maxiter, for use
        inside the boundary grid search where we only need the NLL
        ranking to be roughly right, not each fit to be converged."""
        h = np.asarray(h, dtype=float)
        ah = np.abs(h)
        n = ah.size
        if n < 5:
            return 1.0, float(np.std(h) + 1e-8)

        scale_floor = cls._scale_floor(ah)

        def neg_ll(log_theta):
            alpha = float(np.clip(np.exp(log_theta[0]), *alpha_bounds))
            scale = max(float(np.exp(log_theta[1])), scale_floor)
            logpdf = (np.log(alpha) - np.log(2.0) - np.log(scale)
                      - gammaln(1.0 / alpha)
                      - (ah / scale) ** alpha)
            cdf_hi = cls._ggd_cdf_abs(hi, alpha, scale)
            cdf_lo = cls._ggd_cdf_abs(lo, alpha, scale)
            mass = cdf_hi - cdf_lo
            if mass < 1e-300:
                return 1e12
            return -np.sum(logpdf) + n * np.log(mass)

        alpha0 = 1.5
        safe_ah = np.maximum(ah, 1e-300)
        scale0 = max((alpha0 * np.mean(safe_ah ** alpha0)) ** (1.0 / alpha0), scale_floor)
        x0 = [np.log(alpha0), np.log(scale0)]
        res = minimize(neg_ll, x0=x0, method="Nelder-Mead",
                        options=dict(xatol=1e-3, fatol=1e-3, maxiter=maxiter))
        alpha_hat = float(np.clip(np.exp(res.x[0]), *alpha_bounds))
        scale_hat = max(float(np.exp(res.x[1])), scale_floor)
        return alpha_hat, scale_hat

    @classmethod
    def _total_nll_for_boundaries(cls, h, ah, c1, c2, alpha_bounds, fast=True):
        """Total NLL of the 3-region truncated-GGD mixture for a given
        (c1, c2). Used as the objective for boundary search."""
        bulk_h = h[ah < c1]
        mid_h  = h[(ah >= c1) & (ah < c2)]
        tail_h = h[ah >= c2]
        if min(bulk_h.size, mid_h.size, tail_h.size) < 5:
            return np.inf
        fit_fn = cls._fit_ggd_truncated_fast if fast else cls._fit_ggd_truncated
        total = 0.0
        for seg, lo, hi in [(bulk_h, 0.0, c1), (mid_h, c1, c2), (tail_h, c2, np.inf)]:
            a, s = fit_fn(seg, lo, hi, alpha_bounds)
            ah_seg = np.abs(seg)
            logpdf = (np.log(a) - np.log(2.0) - np.log(s) - gammaln(1.0 / a)
                       - (ah_seg / s) ** a)
            cdf_hi = cls._ggd_cdf_abs(hi, a, s)
            cdf_lo = cls._ggd_cdf_abs(lo, a, s)
            mass = max(cdf_hi - cdf_lo, 1e-300)
            total += -np.sum(logpdf) + seg.size * np.log(mass)
        return total

    @classmethod
    def _fit_boundaries_by_likelihood(cls, h, alpha_bounds,
                                       n_grid=12, n_iters=3,
                                       subsample=20000, seed=0,
                                       c1_q_range=(0.05, 0.80),
                                       c2_q_range=(0.80, 0.995)):
        """
        Block coordinate ascent on total mixture log-likelihood:
        alternately fix c2 and grid-search c1 (and vice versa), each
        candidate scored by refitting all three regions on a subsample
        with a coarse/fast optimizer. A few iterations are enough since
        each step is a 1-D search and the two boundaries are only
        weakly coupled.
        """
        ah = np.abs(h)
        rng = np.random.default_rng(seed)
        if ah.size > subsample:
            idx = rng.choice(ah.size, subsample, replace=False)
            h_s, ah_s = h[idx], ah[idx]
        else:
            h_s, ah_s = h, ah

        c1 = float(np.quantile(ah_s, 0.5))
        c2 = float(np.quantile(ah_s, 0.97))

        for _ in range(n_iters):
            grid1 = np.quantile(ah_s, np.linspace(*c1_q_range, n_grid))
            best_nll, best_c1 = np.inf, c1
            for cand in grid1:
                if cand <= 1e-10 or cand >= c2 * 0.95:
                    continue
                nll = cls._total_nll_for_boundaries(h_s, ah_s, cand, c2, alpha_bounds)
                if nll < best_nll:
                    best_nll, best_c1 = nll, float(cand)
            c1 = best_c1

            grid2 = np.quantile(ah_s, np.linspace(*c2_q_range, n_grid))
            best_nll, best_c2 = np.inf, c2
            for cand in grid2:
                if cand <= c1 * 1.05:
                    continue
                nll = cls._total_nll_for_boundaries(h_s, ah_s, c1, cand, alpha_bounds)
                if nll < best_nll:
                    best_nll, best_c2 = nll, float(cand)
            c2 = best_c2

        return c1, c2

    # ------------------------------------------------------------------
    # fit_reference
    # ------------------------------------------------------------------
    def fit_reference(self, x, tail_quantile=None, boundary_method=None):
        if tail_quantile is None:
            tail_quantile = self.tail_quantile
        if boundary_method is None:
            boundary_method = self.boundary_method

        self._filters_3x = None
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real.detach().cpu().numpy()

        a1s, s1s, a2s, s2s, a3s, s3s = [], [], [], [], [], []
        c1s, c2s, sw1s, sw2s = [], [], [], []
        pi_bs, pi_ms, pi_ts = [], [], []

        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            ah = np.abs(h)

            if boundary_method == "likelihood":
                c1_, c2_ = self._fit_boundaries_by_likelihood(
                    h, self.alpha_bounds,
                    n_grid=self.boundary_search_grid,
                    n_iters=self.boundary_search_iters,
                    subsample=self.boundary_search_subsample)
            elif boundary_method == "curvature":
                c1_, c2_ = self._survival_curvature_boundaries(ah)
            else:  # 'moment' -- original behaviour
                c1_ = float(np.mean(ah))
                c2_ = float(np.quantile(ah, tail_quantile))
            c1_ = max(c1_, 1e-8)
            c2_ = max(c2_, c1_ * 1.5)

            sw1_ = max(self.trans_frac * c1_, 1e-6)
            sw2_ = max(self.trans_frac * (c2_ - c1_), 1e-6)

            bulk_h = h[ah < c1_]
            mid_h  = h[(ah >= c1_) & (ah < c2_)]
            tail_h = h[ah >= c2_]
            n_b, n_m, n_t = bulk_h.size, mid_h.size, tail_h.size
            total = max(n_b + n_m + n_t, 1)
            pi_b, pi_m, pi_t = n_b/total, n_m/total, n_t/total

            print(f"[GGD³][ch {j}] method={boundary_method}  "
                  f"c1={c1_:.4f}  c2={c2_:.4f} | "
                  f"N_bulk={n_b}({pi_b:.1%})  N_mid={n_m}({pi_m:.1%})  "
                  f"N_tail={n_t}({pi_t:.1%})")
            if min(n_b, n_m, n_t) < self.min_region_samples:
                print(f"[GGD³][ch {j}] WARNING: region has < "
                      f"{self.min_region_samples} samples.")

            a1_, s1_ = self._fit_ggd_truncated(bulk_h, 0.0,  c1_,       self.alpha_bounds)
            a2_, s2_ = self._fit_ggd_truncated(mid_h,  c1_,  c2_,       self.alpha_bounds)
            a3_, s3_ = self._fit_ggd_truncated(tail_h, c2_,  np.inf,    self.alpha_bounds)

            print(f"[GGD³][ch {j}]  "
                  f"bulk(α={a1_:.3f}, sc={s1_:.4f}) | "
                  f"mid (α={a2_:.3f}, sc={s2_:.4f}) | "
                  f"tail(α={a3_:.3f}, sc={s3_:.4f})")

            a1s += [a1_]; s1s += [s1_]
            a2s += [a2_]; s2s += [s2_]
            a3s += [a3_]; s3s += [s3_]
            c1s += [c1_]; c2s += [c2_]
            sw1s += [sw1_]; sw2s += [sw2_]
            pi_bs += [pi_b]; pi_ms += [pi_m]; pi_ts += [pi_t]

        dtype = x.dtype if x.is_floating_point() else torch.float32
        dev = x.device
        mk = lambda L: torch.tensor(L, dtype=dtype, device=dev)
        self.alpha1, self.scale1 = mk(a1s), mk(s1s)
        self.alpha2, self.scale2 = mk(a2s), mk(s2s)
        self.alpha3, self.scale3 = mk(a3s), mk(s3s)
        self.c1, self.c2 = mk(c1s), mk(c2s)
        self.s1, self.s2 = mk(sw1s), mk(sw2s)
        self.pi_bulk = mk(pi_bs)
        self.pi_mid  = mk(pi_ms)
        self.pi_tail = mk(pi_ts)

    def fit(self, x, **kwargs):
        return self.fit_reference(x, **kwargs)

    # ------------------------------------------------------------------
    # Smooth windows (unchanged)
    # ------------------------------------------------------------------
    def _windows(self, z, device):
        c1 = self.c1.to(device)[None, :, None]
        c2 = self.c2.to(device)[None, :, None]
        s1 = self.s1.to(device)[None, :, None]
        s2 = self.s2.to(device)[None, :, None]
        az = torch.sqrt(z ** 2 + self.eps_abs)
        g1 = torch.sigmoid(-(az - c1) / s1)
        g2 = torch.sigmoid(-(az - c2) / s2)
        return g1, g2 - g1, 1.0 - g2

    def forward(self, x, *args):
        self._check_fitted()
        filters = self.filters.to(x.device)
        with torch.no_grad():
            z  = torch.fft.ifft(filters * torch.fft.fft(x)).real
            az = torch.sqrt(z ** 2 + self.eps_abs)
            w_b, w_m, w_t = self._windows(z, x.device)
            a1 = self.alpha1.to(x.device)[None, :, None]
            a2 = self.alpha2.to(x.device)[None, :, None]
            a3 = self.alpha3.to(x.device)[None, :, None]
            phi_b = (w_b * az ** a1).mean(-1)
            phi_m = (w_m * az ** a2).mean(-1)
            phi_t = (w_t * az ** a3).mean(-1)
        return torch.cat([phi_b, phi_m, phi_t], dim=1)

    def _get_filters_3x(self, device):
        if self._filters_3x is None or self._filters_3x.device != device:
            self._filters_3x = self.filters.repeat(1, 3, 1).to(device)
        return self._filters_3x

    def grad(self, x, v=None, means=None):
        self._check_fitted()
        device  = x.device
        filters = self.filters.to(device)
        with torch.no_grad():
            z  = torch.fft.ifft(filters * torch.fft.fft(x)).real
            az = torch.sqrt(z ** 2 + self.eps_abs)
            sz = z / az

            c1 = self.c1.to(device)[None, :, None]
            c2 = self.c2.to(device)[None, :, None]
            s1 = self.s1.to(device)[None, :, None]
            s2 = self.s2.to(device)[None, :, None]
            a1 = self.alpha1.to(device)[None, :, None]
            a2 = self.alpha2.to(device)[None, :, None]
            a3 = self.alpha3.to(device)[None, :, None]

            g1 = torch.sigmoid(-(az - c1) / s1)
            g2 = torch.sigmoid(-(az - c2) / s2)
            w_b = g1;   w_m = g2 - g1;   w_t = 1.0 - g2

            dg1 = -g1 * (1.0 - g1) * sz / s1
            dg2 = -g2 * (1.0 - g2) * sz / s2
            dw_b =  dg1
            dw_m =  dg2 - dg1
            dw_t = -dg2

            D_b = dw_b * az**a1 + w_b * a1 * z * az**(a1 - 2.0)
            D_m = dw_m * az**a2 + w_m * a2 * z * az**(a2 - 2.0)
            D_t = dw_t * az**a3 + w_t * a3 * z * az**(a3 - 2.0)

            D_all = torch.cat([D_b, D_m, D_t], dim=1)
            f3 = self._get_filters_3x(device)
            grad_coeff = torch.fft.ifft(
                torch.fft.fft(D_all) * f3
            ).real / x.shape[-1]

        if v is None:
            return grad_coeff
        return (grad_coeff * v[None, :, None]).sum(1, keepdim=True)

    def summary(self):
        self._check_fitted()
        a1=self.alpha1.cpu().numpy(); s1=self.scale1.cpu().numpy()
        a2=self.alpha2.cpu().numpy(); s2=self.scale2.cpu().numpy()
        a3=self.alpha3.cpu().numpy(); s3=self.scale3.cpu().numpy()
        c1=self.c1.cpu().numpy(); c2=self.c2.cpu().numpy()
        pb=self.pi_bulk.cpu().numpy(); pm=self.pi_mid.cpu().numpy()
        pt=self.pi_tail.cpu().numpy()
        print(f"{'Ch':>3} {'c1':>8} {'c2':>8} | "
              f"{'a1':>5} {'sc1':>7} | {'a2':>5} {'sc2':>7} | "
              f"{'a3':>5} {'sc3':>7} | {'pi_b':>6} {'pi_m':>6} {'pi_t':>6}")
        print("-"*95)
        for j in range(len(a1)):
            print(f"{j:>3d} {c1[j]:>8.4f} {c2[j]:>8.4f} | "
                  f"{a1[j]:>5.2f} {s1[j]:>7.4f} | "
                  f"{a2[j]:>5.2f} {s2[j]:>7.4f} | "
                  f"{a3[j]:>5.2f} {s3[j]:>7.4f} | "
                  f"{pb[j]:>6.2%} {pm[j]:>6.2%} {pt[j]:>6.2%}")

    def plot_fit(self, x, label="Wavelet", n_grid=1000, fit_if_needed=True):
        if fit_if_needed and not self.is_fitted:
            self.fit_reference(x)
        self._check_fitted()

        filters = self.filters.to(x.device)
        wt = torch.fft.ifft(torch.fft.fft(x) * filters).real

        a1=self.alpha1.cpu().numpy(); s1=self.scale1.cpu().numpy()
        a2=self.alpha2.cpu().numpy(); s2=self.scale2.cpu().numpy()
        a3=self.alpha3.cpu().numpy(); s3=self.scale3.cpu().numpy()
        c1a=self.c1.cpu().numpy();   c2a=self.c2.cpu().numpy()
        pb=self.pi_bulk.cpu().numpy(); pm=self.pi_mid.cpu().numpy()
        pt=self.pi_tail.cpu().numpy()

        def logpdf_ggd_trunc(xv, alpha, scale, lo, hi, pi):
            cdf_hi = Scalar_GGD_GGD_GGD._ggd_cdf_abs(hi, alpha, scale)
            cdf_lo = Scalar_GGD_GGD_GGD._ggd_cdf_abs(lo, alpha, scale)
            mass = max(cdf_hi - cdf_lo, 1e-300)
            lp = (np.log(alpha) - np.log(2.0) - np.log(scale)
                - gammaln(1.0 / alpha)
                - (np.abs(xv) / scale) ** alpha
                - np.log(mass)
                + np.log(max(pi, 1e-300)))
            return np.clip(lp, -500.0, 500.0)

        for j in range(self.filters.shape[1]):
            h = wt[:, j, :].detach().cpu().flatten().numpy()
            h = h[np.isfinite(h)]
            if h.size == 0:
                continue
            ah = np.abs(h)
            c1_j, c2_j = float(c1a[j]), float(c2a[j])

            xmax = float(ah.max()) * 1.02

            x_b  = np.linspace(-c1_j, c1_j, n_grid)
            x_mp = np.linspace(c1_j,  c2_j, n_grid // 2)
            x_tp = np.linspace(c2_j,  xmax, n_grid // 2)

            lp_b  = logpdf_ggd_trunc(x_b,  a1[j], s1[j], 0.0,  c1_j,   pb[j])
            lp_mp = logpdf_ggd_trunc(x_mp, a2[j], s2[j], c1_j, c2_j,   pm[j])
            lp_tp = logpdf_ggd_trunc(x_tp, a3[j], s3[j], c2_j, np.inf, pt[j])

            x_m  = np.concatenate([-x_mp[::-1], x_mp])
            lp_m = np.concatenate([lp_mp[::-1], lp_mp])
            x_t  = np.concatenate([-x_tp[::-1], x_tp])
            lp_t = np.concatenate([lp_tp[::-1], lp_tp])

            hist_vals, _ = np.histogram(h, bins=150, density=True)
            hist_vals = hist_vals[hist_vals > 0]
            y_min = max(hist_vals.min() * 0.5, 1e-6)
            y_max = hist_vals.max() * 3.0

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(h, bins=150, density=True, log=True,
                    alpha=0.4, color="steelblue", label="data")
            ax.plot(x_b, np.exp(lp_b),  lw=2, color="tab:orange",
                    label=f"bulk  GGD α={a1[j]:.2f} sc={s1[j]:.3f} π={pb[j]:.1%}")
            ax.plot(x_m, np.exp(lp_m),  lw=2, color="tab:green",
                    label=f"mid   GGD α={a2[j]:.2f} sc={s2[j]:.3f} π={pm[j]:.1%}")
            ax.plot(x_t, np.exp(lp_t),  lw=2, color="tab:red",
                    label=f"tail  GGD α={a3[j]:.2f} sc={s3[j]:.3f} π={pt[j]:.1%}")
            ax.axvline( c1_j, color="black", ls="--", lw=1, alpha=0.4,
                        label=f"c1={c1_j:.3f} ({self.boundary_method})")
            ax.axvline(-c1_j, color="black", ls="--", lw=1, alpha=0.4)
            ax.axvline( c2_j, color="black", ls=":",  lw=1, alpha=0.4,
                        label=f"c2={c2_j:.3f}")
            ax.axvline(-c2_j, color="black", ls=":",  lw=1, alpha=0.4)

            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("Coefficient value"); ax.set_ylabel("Log density")
            ax.set_title(f"{label} — channel {j}")
            ax.legend(frameon=False, fontsize=7.5)
            plt.tight_layout(); plt.show()