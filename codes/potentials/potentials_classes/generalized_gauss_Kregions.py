import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.special import gammaln, gammainc
from scipy.optimize import minimize

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.special import gammaln, gammainc
from scipy.optimize import minimize


class Scalar_GGD_KRegion:
    """
    K-region composite truncated-GGD potential per wavelet channel, exposing a
    set of scalar statistics phi_c(x) = mean_t w_c(|z|) * |z|^{alpha_c} whose
    gradients feed an MGD (microcanonical / max-entropy gradient descent) solver.

    WHAT CHANGED AND WHY (singular-Gram fix)
    ========================================
    The MGD step inverts the Gram matrix  G_{c c'} = < grad phi_c , grad phi_c' >.
    The previous version stored every channel padded to K fixed slots; channels
    that model-selection assigned K_eff < K carried (K - K_eff) *collapsed*
    regions -- epsilon-width slivers near 0 whose smooth window is ~0 over the
    whole data range. Such a region has phi ~ 0 AND grad phi ~ 0, i.e. it is a
    ZERO ROW of the Gram matrix. From the reported fit that was 29 dead rows out
    of 84 (psi bank) and 9/32 (morlet): the Gram is singular *by construction*,
    which is why even the fixed-parameter version was singular -- it is a
    structural property of what statistics are exposed, not a numerical accident
    in the optimizer.

    Two levels of degeneracy are removed:

      (1) DEAD statistics (exact zero gradient): a slot is exposed only if it is
          a real region -- empirical mass pi_k >= pi_active_min AND at least
          min_region_samples samples. Collapsed slivers are never exposed.

      (2) COLLINEAR statistics (near-linearly-dependent gradients): even among
          real regions, gradients can be nearly dependent -- e.g. duplicated
          coarse channels (the identical K=1 fits on ch17-20), or two adjacent
          regions with near-equal alpha. `fit_reference` therefore evaluates the
          gradients on the reference batch and runs a rank-revealing pivoted
          Cholesky on their CORRELATION matrix, keeping a maximal subset whose
          mutual conditioning stays below 1/cond_tol. This guarantees the
          exposed Gram is well-conditioned and invertible. (Set auto_prune=False
          to skip and instead regularize G yourself.)

    forward()/grad()/num_coefficients all report ONLY the surviving active
    statistics, in a fixed, stored order, so the reference means, the descent
    variable v, and the Gram all stay aligned. Per-slot parameters are still
    kept in full for summary()/plot_fit().

    The composite density model, boundary search and BIC model-order selection
    are unchanged from the fitting version; see method docstrings.
    """

    def __init__(self, filters,
                 num_regions=4,
                 trans_frac=0.10,
                 alpha_bounds=(0.2, 8.0),
                 min_region_samples=30,
                 eps_abs=1e-6,
                 boundary_method="auto",
                 model_criterion="bic",
                 boundary_search_subsample=20000,
                 boundary_search_iters=3,
                 boundary_search_grid=12,
                 pi_active_min=1e-3,
                 auto_prune=True,
                 cond_tol=1e-6,
                 prune_max_cols=20000,
                 kurt_thresholds=None,
                 verbose=True):
        self.filters = filters
        self.K = int(num_regions)
        assert self.K >= 1
        self.J = filters.shape[1]
        self.trans_frac = trans_frac
        self.alpha_bounds = alpha_bounds
        self.min_region_samples = min_region_samples
        self.eps_abs = eps_abs
        self.boundary_method = boundary_method
        self.model_criterion = model_criterion
        self.boundary_search_subsample = boundary_search_subsample
        self.boundary_search_iters = boundary_search_iters
        self.boundary_search_grid = boundary_search_grid
        self.pi_active_min = pi_active_min
        self.auto_prune = auto_prune
        self.cond_tol = cond_tol
        self.prune_max_cols = prune_max_cols
        self.verbose = verbose
        # kurtosis -> region-count search budget (see _kurtosis_Kmax). Default:
        # K-1 log-spaced thresholds from kurt=1 (near-Gaussian) to kurt=100
        # (very sparse/intermittent), e.g. for K=4: [1, ~4.6, ~21.5, 100].
        if kurt_thresholds is None:
            kurt_thresholds = np.logspace(0, 2, max(self.K - 1, 1)).tolist()
        self.kurt_thresholds = list(kurt_thresholds)

        # per-region params, each (J, K) once fitted
        self.alpha = self.scale = None
        self.cuts = self.sw = self.pi = None
        self.Keff = None
        # active-statistic bookkeeping
        self.active = None            # (J, K) bool
        self.active_flat = None       # LongTensor of kept flat indices (region-major: k*J + j)
        self.stat_scale = None        # (num_active,) per-statistic normalizer
        self.num_coefficients = self.K * self.J
        self._filters_Kx = None

    # ------------------------------------------------------------------
    @property
    def is_fitted(self):
        return self.alpha is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("must call fit_reference first.")

    # ================= core GGD primitives ============================
    @staticmethod
    def _ggd_cdf_abs(t, alpha, scale):
        if t <= 0:
            return 0.0
        if not np.isfinite(t):
            return 1.0
        return float(gammainc(1.0 / alpha, (t / scale) ** alpha))

    @staticmethod
    def _scale_floor(ah, floor_frac=0.02, min_floor=1e-10):
        med = float(np.median(ah)) if ah.size else 0.0
        return max(floor_frac * med, min_floor)

    @classmethod
    def _fit_ggd_truncated(cls, h, lo, hi, alpha_bounds, maxiter=8000,
                           xatol=1e-5, fatol=1e-5):
        h = np.asarray(h, dtype=float); ah = np.abs(h); n = ah.size
        if n < 5:
            return 1.0, float(np.std(h) + 1e-8)
        scale_floor = cls._scale_floor(ah)

        def neg_ll(lt):
            alpha = float(np.clip(np.exp(lt[0]), *alpha_bounds))
            scale = max(float(np.exp(lt[1])), scale_floor)
            logpdf = (np.log(alpha) - np.log(2.0) - np.log(scale)
                      - gammaln(1.0 / alpha) - (ah / scale) ** alpha)
            mass = cls._ggd_cdf_abs(hi, alpha, scale) - cls._ggd_cdf_abs(lo, alpha, scale)
            if mass < 1e-300:
                return 1e12
            return -np.sum(logpdf) + n * np.log(mass)

        a0 = 1.5
        s0 = max((a0 * np.mean(np.maximum(ah, 1e-300) ** a0)) ** (1.0 / a0), scale_floor)
        res = minimize(neg_ll, x0=[np.log(a0), np.log(s0)], method="Nelder-Mead",
                       options=dict(xatol=xatol, fatol=fatol, maxiter=maxiter))
        a = float(np.clip(np.exp(res.x[0]), *alpha_bounds))
        s = max(float(np.exp(res.x[1])), scale_floor)
        return a, s

    @classmethod
    def _fit_ggd_truncated_fast(cls, h, lo, hi, alpha_bounds, maxiter=250):
        return cls._fit_ggd_truncated(h, lo, hi, alpha_bounds,
                                      maxiter=maxiter, xatol=1e-3, fatol=1e-3)

    @classmethod
    def _region_nll(cls, seg, lo, hi, alpha_bounds, fast):
        fit_fn = cls._fit_ggd_truncated_fast if fast else cls._fit_ggd_truncated
        a, s = fit_fn(seg, lo, hi, alpha_bounds)
        ah = np.abs(seg)
        logpdf = (np.log(a) - np.log(2.0) - np.log(s) - gammaln(1.0 / a) - (ah / s) ** a)
        mass = max(cls._ggd_cdf_abs(hi, a, s) - cls._ggd_cdf_abs(lo, a, s), 1e-300)
        return -np.sum(logpdf) + seg.size * np.log(mass), (a, s)

    @classmethod
    def _composite_nll(cls, h, boundaries, alpha_bounds, fast=True, min_seg=5):
        ah = np.abs(h); N = h.size
        edges = [0.0] + list(boundaries) + [np.inf]
        total = 0.0; params = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            seg = h[(ah >= lo) & (ah < hi)]
            if seg.size < min_seg:
                return np.inf, None
            nll, (a, s) = cls._region_nll(seg, lo, hi, alpha_bounds, fast)
            total += nll - seg.size * np.log(seg.size / N)
            params.append((a, s))
        return total, params

    @classmethod
    def _fit_boundaries_k(cls, h, K, alpha_bounds, n_grid=12, n_iters=3,
                          subsample=20000, seed=0):
        if K == 1:
            return []
        ah = np.abs(h); rng = np.random.default_rng(seed)
        if ah.size > subsample:
            idx = rng.choice(ah.size, subsample, replace=False)
            h_s, ah_s = h[idx], ah[idx]
        else:
            h_s, ah_s = h, ah
        init_q = np.linspace(0.5, 0.99, K - 1) if K > 2 else np.array([0.9])
        cuts = list(np.quantile(ah_s, init_q))
        q_los = np.linspace(0.05, 0.80, K - 1); q_his = np.linspace(0.80, 0.995, K - 1)

        def scored(cs):
            nll, _ = cls._composite_nll(h_s, sorted(cs), alpha_bounds, fast=True)
            return nll

        for _ in range(n_iters):
            for m in range(K - 1):
                grid = np.quantile(ah_s, np.linspace(q_los[m], q_his[m], n_grid))
                lo_nb = cuts[m - 1] if m > 0 else 0.0
                hi_nb = cuts[m + 1] if m < K - 2 else np.inf
                best_nll, best = np.inf, cuts[m]
                for cand in grid:
                    cand = float(cand)
                    if cand <= max(lo_nb * 1.05, 1e-10):
                        continue
                    if np.isfinite(hi_nb) and cand >= hi_nb * 0.95:
                        continue
                    trial = list(cuts); trial[m] = cand
                    nll = scored(trial)
                    if nll < best_nll:
                        best_nll, best = nll, cand
                cuts[m] = best
        return sorted(cuts)

    # ================= kurtosis-adaptive region budget =================
    #
    # Rationale (from the real Q=1/Q=3 wavelet data): kurtosis descends
    # smoothly and monotonically from very sparse/intermittent fine scales
    # (kurt in the hundreds) to near-Gaussian coarse scales (kurt ~ 0), in
    # BOTH filter banks. There is no fixed K right for every channel: a
    # near-Gaussian channel forced into K=4 regions fits noise into an
    # arbitrary partition (exactly the "bulk fit almost absent" failure
    # mode seen earlier), while a kurt~600 channel genuinely needs several
    # regions to resolve spike -> shoulder -> tail. Rather than paying for
    # full AIC/BIC order search (K=1..K) on every channel, kurtosis gives a
    # near-free way to cap the search budget per channel BEFORE any
    # boundary optimization runs.
    @staticmethod
    def _channel_kurtosis(h):
        """Excess (Fisher) kurtosis of the raw (signed) coefficients."""
        h = np.asarray(h, dtype=float)
        h = h[np.isfinite(h)]
        if h.size < 8:
            return 0.0
        m = h.mean()
        v = ((h - m) ** 2).mean()
        if v <= 1e-300:
            return 0.0
        m4 = ((h - m) ** 4).mean()
        return float(m4 / (v ** 2) - 3.0)

    def _kurtosis_Kmax(self, kurt):
        """
        Map kurtosis to a cap on region count: Kmax = 1 + (number of
        self.kurt_thresholds exceeded). With the default log-spaced
        thresholds (kurt=1..100), a near-Gaussian channel (kurt<1) gets
        Kmax=1 (single GGD, no fake partition); a very peaked channel
        (kurt>=100) gets the full Kmax=self.K search budget. This directly
        implements "fewer regions used if fewer are needed": Kmax only
        caps the search, _select_model_order / AIC-BIC can still pick
        something smaller within that budget.
        """
        Kmax = 1
        for t in self.kurt_thresholds:
            if kurt >= t:
                Kmax += 1
        return int(min(Kmax, self.K))

    def _select_model_order(self, h, seed=0, K_max=None):
        if K_max is None:
            K_max = self.K
        K_max = max(1, min(int(K_max), self.K))
        N = h.size
        pen = (np.log(N) if self.model_criterion == "bic" else 2.0)
        results = []
        for K in range(1, K_max + 1):
            cuts = self._fit_boundaries_k(
                h, K, self.alpha_bounds, n_grid=self.boundary_search_grid,
                n_iters=self.boundary_search_iters,
                subsample=self.boundary_search_subsample, seed=seed)
            nll, params = self._composite_nll(h, cuts, self.alpha_bounds,
                                              fast=False, min_seg=self.min_region_samples)
            if params is None:
                continue
            crit = 2.0 * nll + pen * (4 * K - 2)
            results.append((crit, K, cuts))
        if not results:
            return 1, []
        crit, K, cuts = min(results, key=lambda r: r[0])
        if self.verbose:
            table = "  ".join(f"K{k}={c:.0f}" for c, k, _ in sorted(results, key=lambda r: r[1]))
            print(f"    [model-select] chose K={K}  (Kmax={K_max})  "
                  f"({self.model_criterion.upper()}: {table})")
        return K, cuts

    def _embed_slots(self, cuts, ah):
        eps = max(float(np.quantile(ah, 1e-3)), 1e-12) * 1e-2
        n_missing = (self.K - 1) - len(cuts)
        pad = [eps * (i + 1) for i in range(n_missing)]
        slots = pad + list(cuts)
        for i in range(1, len(slots)):
            if slots[i] <= slots[i - 1]:
                slots[i] = slots[i - 1] * 1.5 + eps
        return np.asarray(slots, dtype=float)

    # =========================== FIT ==================================
    def fit_reference(self, x, boundary_method=None):
        if boundary_method is None:
            boundary_method = self.boundary_method
        self._filters_Kx = None
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real.detach().cpu().numpy()

        J, K = z.shape[1], self.K
        A = np.ones((J, K)); S = np.ones((J, K))
        CUT = np.zeros((J, K - 1)); SW = np.full((J, K - 1), 1e-6)
        PI = np.zeros((J, K)); KEFF = np.ones(J, dtype=int)
        ACT = np.zeros((J, K), dtype=bool)

        for j in range(J):
            h = z[:, j, :].reshape(-1); h = h[np.isfinite(h)]; ah = np.abs(h); N = h.size

            kurt = self._channel_kurtosis(h)
            K_max = self._kurtosis_Kmax(kurt)

            if boundary_method == "auto":
                Keff, cuts = self._select_model_order(h, seed=0, K_max=K_max)
            elif boundary_method == "likelihood":
                Keff = K_max
                cuts = self._fit_boundaries_k(
                    h, K_max, self.alpha_bounds, n_grid=self.boundary_search_grid,
                    n_iters=self.boundary_search_iters,
                    subsample=self.boundary_search_subsample)
            else:
                Keff = K_max
                qs = np.linspace(0.5, 0.97, max(K_max - 1, 0))
                cuts = list(np.quantile(ah, qs)) if K_max > 1 else []

            KEFF[j] = Keff
            slots = np.maximum(self._embed_slots(cuts, ah), 1e-8)
            for i in range(1, K - 1):
                slots[i] = max(slots[i], slots[i - 1] * 1.5)
            CUT[j] = slots

            edges = np.concatenate([[0.0], slots])
            for m in range(K - 1):
                width = slots[m] - edges[m]
                SW[j, m] = max(self.trans_frac * max(width, slots[m]), 1e-6)

            full_edges = [0.0] + list(slots) + [np.inf]
            counts = []
            for k, (lo, hi) in enumerate(zip(full_edges[:-1], full_edges[1:])):
                seg = h[(ah >= lo) & (ah < hi)]; counts.append(seg.size)
                if seg.size < 5:
                    A[j, k], S[j, k] = 1.0, max(self._scale_floor(ah), 1e-8)
                    continue
                A[j, k], S[j, k] = self._fit_ggd_truncated(seg, lo, hi, self.alpha_bounds)
            counts = np.asarray(counts, float)
            PI[j] = counts / max(counts.sum(), 1)
            # a slot is ACTIVE iff it is a real region
            ACT[j] = (PI[j] >= self.pi_active_min) & (counts >= self.min_region_samples)
            if not ACT[j].any():                      # safety: keep the fullest slot
                ACT[j, int(np.argmax(counts))] = True

            if self.verbose:
                cut_str = " ".join(f"{c:.4f}" for c in slots)
                pi_str = " ".join(f"{p:.1%}" for p in PI[j])
                a_str = " ".join(f"{a:.2f}" for a in A[j])
                print(f"[GGD^{K}][ch {j}] kurt={kurt:7.2f}  Kmax={K_max}  Keff={Keff}  "
                      f"cuts=[{cut_str}]  pi=[{pi_str}]  alpha=[{a_str}]  "
                      f"active={ACT[j].sum()}")

        dtype = x.dtype if x.is_floating_point() else torch.float32
        dev = x.device
        mk = lambda M: torch.tensor(M, dtype=dtype, device=dev)
        self.alpha = mk(A); self.scale = mk(S); self.cuts = mk(CUT)
        self.sw = mk(SW); self.pi = mk(PI)
        self.Keff = torch.tensor(KEFF, device=dev)
        self.active = torch.tensor(ACT, device=dev)

        # flat active indices in region-major layout (matches forward()'s cat order)
        flat = [k * J + j for k in range(K) for j in range(J) if ACT[j, k]]
        self.active_flat = torch.tensor(sorted(flat), dtype=torch.long, device=dev)
        self.num_coefficients = int(self.active_flat.numel())

        n_dead = J * K - int(ACT.sum())
        if self.verbose:
            print(f"[active] {self.num_coefficients} active statistics "
                  f"({n_dead} dead slots dropped before they can singularize the Gram)")

        # level (2): remove residual near-collinear statistics on the reference
        self.stat_scale = torch.ones(self.num_coefficients, dtype=dtype, device=dev)
        if self.auto_prune and self.num_coefficients > 1:
            self.prune_collinear(x, cond_tol=self.cond_tol,
                                 max_cols=self.prune_max_cols, verbose=self.verbose)
        # level (3): normalize each exposed statistic by its reference gradient
        # norm so the raw Gram the MGD forms is well-conditioned (not just the
        # correlation matrix). Rescaling a statistic by a constant leaves the
        # mean-matching fixed point unchanged.
        self._compute_stat_scale(x)
        return self

    def _compute_stat_scale(self, x, max_cols=None):
        self.stat_scale = torch.ones(self.num_coefficients, dtype=x.dtype
                                     if x.is_floating_point() else torch.float32,
                                     device=x.device)
        G = self.gram_matrix(x, active_only=True, max_cols=max_cols, _use_scale=False)
        d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
        self.stat_scale = torch.tensor(d, dtype=self.stat_scale.dtype, device=x.device)
        return self

    def fit(self, x, **kw):
        return self.fit_reference(x, **kw)

    # ===================== windows / forward / grad ===================
    def _windows_from_sigmoids(self, g):
        K = self.K
        if K == 1:
            return [torch.ones_like(g[..., 0])]
        ws = [g[..., 0]]
        for k in range(1, K - 1):
            ws.append(g[..., k] - g[..., k - 1])
        ws.append(1.0 - g[..., K - 2])
        return ws

    def _all_slot_forward(self, x):
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real
        az = torch.sqrt(z ** 2 + self.eps_abs)
        c = self.cuts.to(x.device); s = self.sw.to(x.device)
        g = torch.sigmoid(-(az.unsqueeze(-1) - c[None, :, None, :]) / s[None, :, None, :])
        ws = self._windows_from_sigmoids(g)
        alpha = self.alpha.to(x.device)
        phis = [(ws[k] * az ** alpha[:, k][None, :, None]).mean(-1) for k in range(self.K)]
        return torch.cat(phis, dim=1)                 # (B, J*K), region-major

    def forward(self, x, *args):
        self._check_fitted()
        with torch.no_grad():
            phi = self._all_slot_forward(x).index_select(1, self.active_flat.to(x.device))
            if self.stat_scale is not None:
                phi = phi / self.stat_scale.to(x.device)[None, :]
        return phi

    def _get_filters_Kx(self, device):
        if self._filters_Kx is None or self._filters_Kx.device != device:
            self._filters_Kx = self.filters.repeat(1, self.K, 1).to(device)
        return self._filters_Kx

    def _all_slot_grad(self, x):
        device = x.device
        filters = self.filters.to(device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real
        az = torch.sqrt(z ** 2 + self.eps_abs)
        sz = z / az
        c = self.cuts.to(device); s = self.sw.to(device); alpha = self.alpha.to(device)
        g = torch.sigmoid(-(az.unsqueeze(-1) - c[None, :, None, :]) / s[None, :, None, :])
        dg = -g * (1.0 - g) * sz.unsqueeze(-1) / s[None, :, None, :]
        ws = self._windows_from_sigmoids(g)
        K = self.K
        if K == 1:
            dws = [torch.zeros_like(az)]
        else:
            dws = [dg[..., 0]]
            for k in range(1, K - 1):
                dws.append(dg[..., k] - dg[..., k - 1])
            dws.append(-dg[..., K - 2])
        D = []
        for k in range(K):
            a = alpha[:, k][None, :, None]
            D.append(dws[k] * az ** a + ws[k] * a * z * az ** (a - 2.0))
        D_all = torch.cat(D, dim=1)
        fKx = self._get_filters_Kx(device)
        return torch.fft.ifft(torch.fft.fft(D_all) * fKx).real / x.shape[-1]

    def grad(self, x, v=None, means=None):
        self._check_fitted()
        with torch.no_grad():
            gc = self._all_slot_grad(x).index_select(1, self.active_flat.to(x.device))
            if self.stat_scale is not None:
                gc = gc / self.stat_scale.to(x.device)[None, :, None]
        if v is None:
            return gc                                 # (B, num_active, T)
        return (gc * v[None, :, None]).sum(1, keepdim=True)

    # ===================== Gram / conditioning ========================
    def gram_matrix(self, x, active_only=True, max_cols=None, normalize=False,
                    _use_scale=True):
        """
        G_{c c'} = mean_b sum_t d phi_c/dx . d phi_c'/dx, computed batch-by-batch
        (memory-safe). This is exactly the matrix the MGD step inverts.
        active_only=False returns the full padded Gram (to expose the original
        singularity). active_only=True returns the EXPOSED Gram (normalized by
        stat_scale when _use_scale). normalize=True returns the correlation
        matrix (scale-invariant collinearity view).
        """
        self._check_fitted()
        device = x.device
        B, _, T = x.shape
        if max_cols is None:
            max_cols = self.prune_max_cols
        idx = self.active_flat.to(device) if active_only else None
        scale = (self.stat_scale.to(device) if (active_only and _use_scale
                 and self.stat_scale is not None) else None)
        rng = np.random.default_rng(0)
        G = None; ncols = 0
        for b in range(B):
            gc = self._all_slot_grad(x[b:b + 1])          # (1, J*K, T)
            if idx is not None:
                gc = gc.index_select(1, idx)
            g = gc[0]                                      # (C, T)
            if scale is not None:
                g = g / scale[:, None]
            if g.shape[1] > max_cols:
                sel = torch.tensor(rng.choice(g.shape[1], max_cols, replace=False),
                                   device=device)
                g = g.index_select(1, sel)
            Gb = (g @ g.T).double().cpu().numpy()
            G = Gb if G is None else G + Gb
            ncols += g.shape[1]
        G = G / max(B, 1)
        if normalize:
            d = np.sqrt(np.clip(np.diag(G), 1e-300, None))
            G = G / np.outer(d, d)
        return G

    def _active_gradient_matrix(self, x, max_rows):
        """(n_samples_sub, num_active) matrix of unit-normalized gradient columns."""
        device = x.device
        idx = self.active_flat.to(device)
        mats = []
        for b in range(x.shape[0]):
            gc = self._all_slot_grad(x[b:b + 1]).index_select(1, idx)[0]   # (C, T)
            mats.append(gc.T)                                              # (T, C)
        M = torch.cat(mats, 0)                                            # (B*T, C)
        if M.shape[0] > max_rows:
            sel = torch.randperm(M.shape[0], device=device)[:max_rows]
            M = M.index_select(0, sel)
        M = M / (M.norm(dim=0, keepdim=True) + 1e-30)
        return M.double().cpu().numpy()

    def prune_collinear(self, x, cond_tol=1e-6, max_cols=20000, verbose=True):
        """
        Drop active statistics whose gradients are near-linearly-dependent on the
        others, so the exposed Gram is well-conditioned / invertible. Uses a
        rank-revealing QR with column pivoting on the unit-normalized gradient
        matrix M: after pivoting, |R_ii| is the residual norm of column i once
        the earlier-selected columns are projected out. Keeping columns with
        |R_ii| > sqrt(cond_tol) * |R_00| bounds the exposed Gram's condition
        number at ~1/cond_tol. Updates active state in place.
        """
        from scipy.linalg import qr
        self._check_fitted()
        M = self._active_gradient_matrix(x, max_rows=max_cols)   # cols already unit-norm
        _, R, P = qr(M, mode="economic", pivoting=True)
        absdiag = np.abs(np.diag(R))
        thr = np.sqrt(cond_tol) * (absdiag[0] if absdiag.size else 0.0)
        keep = np.sort(P[absdiag > thr])

        old_flat = self.active_flat.cpu().numpy()
        new_flat = old_flat[keep]
        dropped = len(old_flat) - len(new_flat)

        # rebuild active mask
        J, K = self.J, self.K
        act = np.zeros((J, K), dtype=bool)
        for f in new_flat:
            act[int(f % J), int(f // J)] = True
        dev = self.active_flat.device
        self.active = torch.tensor(act, device=dev)
        self.active_flat = torch.tensor(np.sort(new_flat), dtype=torch.long, device=dev)
        self.num_coefficients = int(self.active_flat.numel())
        # keep stat_scale aligned with the new active set, then recompute it
        self.stat_scale = torch.ones(self.num_coefficients, dtype=torch.float32, device=dev)
        self._compute_stat_scale(x, max_cols=max_cols)
        if verbose:
            print(f"[prune] dropped {dropped} near-collinear statistic(s); "
                  f"{self.num_coefficients} remain")
        return self

    def report_conditioning(self, x, max_cols=20000):
        """Print rank / condition number of the padded vs active Gram."""
        self._check_fitted()
        def stats(G):
            w = np.linalg.eigvalsh(G)
            w = np.clip(w, 0, None)
            wmax = w.max() if w.size else 0.0
            rank = int((w > 1e-10 * max(wmax, 1e-300)).sum())
            cond = (wmax / w[w > 1e-10 * max(wmax, 1e-300)].min()
                    if rank else np.inf)
            return G.shape[0], rank, w.min(), cond
        Gp = self.gram_matrix(x, active_only=False, max_cols=max_cols)
        Ga = self.gram_matrix(x, active_only=True, max_cols=max_cols)
        for name, G in [("padded (all K slots)", Gp), ("active (exposed)", Ga)]:
            n, r, lmin, cond = stats(G)
            flag = "  <-- SINGULAR" if r < n else "  ok"
            print(f"  {name:24s}: dim={n:3d} rank={r:3d} "
                  f"lambda_min={lmin:.2e} cond={cond:.2e}{flag}")

    # =========================== persistence ==========================
    def save_fixed_parameters(self, filename):
        self._check_fitted()
        torch.save(dict(
            alpha=self.alpha.cpu(), scale=self.scale.cpu(), cuts=self.cuts.cpu(),
            sw=self.sw.cpu(), pi=self.pi.cpu(), Keff=self.Keff.cpu(),
            active=self.active.cpu(), active_flat=self.active_flat.cpu(), stat_scale=self.stat_scale.cpu(),
            num_coefficients=self.num_coefficients, K=self.K, J=self.J,
            trans_frac=self.trans_frac, eps_abs=self.eps_abs), filename)

    @classmethod
    def _logpdf_trunc(cls, xv, alpha, scale, lo, hi, pi):
        mass = max(cls._ggd_cdf_abs(hi, alpha, scale) - cls._ggd_cdf_abs(lo, alpha, scale), 1e-300)
        lp = (np.log(alpha) - np.log(2.0) - np.log(scale) - gammaln(1.0 / alpha)
              - (np.abs(xv) / scale) ** alpha - np.log(mass) + np.log(max(pi, 1e-300)))
        return np.clip(lp, -500.0, 500.0)
    
    def plot_fit(self, x, n_grid=800, log_scale=True, fit_if_needed=True):
        if fit_if_needed and not self.is_fitted:
            self.fit_reference(x)
        self._check_fitted()
        filters = self.filters.to(x.device)
        wt = torch.fft.ifft(torch.fft.fft(x) * filters).real
        A = self.alpha.cpu().numpy(); S = self.scale.cpu().numpy()
        C = self.cuts.cpu().numpy(); P = self.pi.cpu().numpy()
        J, K = A.shape
        colors = plt.cm.viridis(np.linspace(0, 0.9, K))
        paths = []
        for j in range(J):
            h = wt[:, j, :].detach().cpu().flatten().numpy()
            h = h[np.isfinite(h)]
            if h.size == 0:
                continue
            ah = np.abs(h); xmax = float(ah.max()) * 1.02
            edges = [0.0] + list(C[j]) + [xmax]
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(h, bins=200, density=True, log=log_scale, alpha=0.35,
                    color="steelblue", label="data")
            for k in range(K):
                if P[j, k] < 1e-4:      # collapsed sliver, nothing to draw
                    continue
                lo, hi = edges[k], edges[k + 1]
                xp = np.linspace(max(lo, 1e-6), hi, n_grid)
                lp = self._logpdf_trunc(xp, A[j, k], S[j, k], lo,
                                        (np.inf if k == K - 1 else hi), P[j, k])
                xx = np.concatenate([-xp[::-1], xp])
                yy = np.exp(np.concatenate([lp[::-1], lp]))
                ax.plot(xx, yy, lw=2, color=colors[k],
                        label=f"r{k} a={A[j,k]:.2f} sc={S[j,k]:.3f} pi={P[j,k]:.1%}")
            for c in C[j]:
                ax.axvline(c, color="k", ls=":", lw=0.8, alpha=0.4)
                ax.axvline(-c, color="k", ls=":", lw=0.8, alpha=0.4)
            ax.set_xlabel("Coefficient value")
            ax.set_ylabel("Log density" if log_scale else "Density")
            ax.set_title(f"channel {j}  (Keff={int(self.Keff[j])})")
            ax.legend(fontsize=7, loc="upper right")
            plt.show()

    # =========================== reporting ============================
    def summary(self):
        self._check_fitted()
        A = self.alpha.cpu().numpy(); S = self.scale.cpu().numpy()
        C = self.cuts.cpu().numpy(); P = self.pi.cpu().numpy()
        Ke = self.Keff.cpu().numpy(); AC = self.active.cpu().numpy()
        J, K = A.shape
        print(f"{'Ch':>3} {'Keff':>4} {'act':>3} | per-region  alpha (scale) [*=active]")
        print("-" * 70)
        for j in range(J):
            cells = " ".join(
                f"{'*' if AC[j,k] else ' '}{A[j,k]:.2f}({P[j,k]:.0%})" for k in range(K))
            print(f"{j:>3d} {Ke[j]:>4d} {AC[j].sum():>3d} | {cells}")


class Scalar_GGD_KRegion_Fixed(Scalar_GGD_KRegion):
    """
    Deterministic: parameters (including the active-statistic set) loaded from
    disk. No optimization, and -- crucially -- it inherits the active/pruned
    statistic set, so it can no longer expose the dead/collinear statistics that
    made the Gram singular.
    """
    def __init__(self, filters, filename):
        super().__init__(filters, auto_prune=False, verbose=False)
        p = torch.load(str(filename), map_location="cpu")
        self.K = p["K"]; self.J = p["J"]
        self.alpha = p["alpha"]; self.scale = p["scale"]; self.cuts = p["cuts"]
        self.sw = p["sw"]; self.pi = p["pi"]; self.Keff = p["Keff"]
        self.active = p["active"]; self.active_flat = p["active_flat"]
        self.stat_scale = p["stat_scale"]
        self.num_coefficients = p["num_coefficients"]
        self.trans_frac = p["trans_frac"]; self.eps_abs = p["eps_abs"]

    def fit_reference(self, *a, **k):
        return self
    fit = fit_reference



class Scalar_GGD_KRegion_Fixed(Scalar_GGD_KRegion):

    """
    Same forward()/grad() behaviour as Scalar_GGD_KRegion,
    but parameters are loaded from disk instead of fitted.

    This class is deterministic and cannot become unstable
    because no optimization is performed.
    """
    def __init__(self, filters, filename):

        super().__init__(filters)

        filename = str(filename)

        pars = torch.load(filename, map_location="cpu")

        self.K = pars["K"]
        self.alpha = pars["alpha"]
        self.cuts = pars["cuts"]
        self.sw = pars["sw"]
        self.scale = pars["scale"]
        self.pi = pars["pi"]
        self.Keff = pars["Keff"]
        self.trans_frac = pars["trans_frac"]
        self.eps_abs = pars["eps_abs"]

    def fit_reference(self, *args, **kwargs):
        return self

    fit = fit_reference