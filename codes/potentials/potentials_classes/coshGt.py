

class Scalar_coshgt_old:
    """
    Per-channel cosh-tempered Generalized-t (coshGT) potential for wavelet coefficients.

    Density (symmetric, zero-mean):
        p(x) ∝ (1 + (g_a(x)/x0)^b)^{-t/b}
        g_a(x) = |x| cosh(a x)

    Shape:
      - Body  (x→0):          p(x) ~ |x|^b              (cusp / flatness)
      - Tail  (x→∞, a=0):     p(x) ~ |x|^{-t}           (pure power law, GT)
      - Tail  (x→∞, a>0):     p(x) ~ |x|^{-t} e^{-a t |x|}  (exponentially tempered)

    Parameters stored per channel (tensors of shape (J,)):
        b     : body cusp index      (> 0)
        t     : tail power index     (> 0)
        a     : tempering rate       (≥ 0; 0 = pure GT).  kappa = a*t is NOT stored.
        x0    : scale                (> 0)

    Key change from the (a,b,c,x0) and (b,t,kappa,x0) variants:
      - We optimize over a directly (not kappa=a*t), because a is the quantity
        that appears in the density, is naturally bounded by a_max, and does not
        co-vary with t or x0. This prevents the kappa/t blow-up seen when
        optimizing over kappa.
    """

    def __init__(self, filters, eps_abs=1e-6, eps_scale=1e-6,
                 a_max=5.0, b_bounds=(0.05, 10.0), t_bounds=(0.5, 50.0)):
        self.filters           = filters
        self.num_coefficients  = filters.shape[1]
        self.eps_abs           = eps_abs
        self.eps_scale         = eps_scale
        self.a_max             = a_max
        self.b_bounds          = b_bounds
        self.t_bounds          = t_bounds
        self.b = self.t = self.a = self.x0 = None   # set by fit_reference

    @property
    def is_fitted(self):
        return self.a is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Scalar_coshgt must be fit_reference'd first.")

    # ------------------------------------------------------------------
    # numpy helpers  (fitting only — all in (b, t, a, x0) coords)
    # ------------------------------------------------------------------
    @staticmethod
    def _logcosh_np(z):
        """Numerically stable log cosh(z)."""
        z = np.abs(z)
        return z + np.log1p(np.exp(-2.0 * z)) - np.log(2.0)

    @classmethod
    def _log_u_np(cls, x, b, a, x0):
        """log_u = b * log(g_a(x)/x0),   g_a(x) = |x| cosh(ax)."""
        lx = np.log(np.maximum(np.abs(x), 1e-300))
        return b * (lx + cls._logcosh_np(a * x) - np.log(x0))

    @classmethod
    def _logZ_np(cls, b, t, a, x0):
        """
        log normalisation constant.  Returns np.inf on numerical failure
        so the optimizer sees a large-penalty signal rather than nan/crash.
        """
        try:
            f = lambda s: np.exp(-(t / b) * np.logaddexp(0.0,
                                   cls._log_u_np(s, b, a, x0)))
            I, _ = quad(f, 0.0, np.inf, limit=200)
            if not np.isfinite(I) or I <= 0.0:
                return np.inf
            return np.log(2.0 * I)
        except Exception:
            return np.inf

    @classmethod
    def _logpdf_np(cls, x, b, t, a, x0):
        lZ = cls._logZ_np(b, t, a, x0)
        if not np.isfinite(lZ):
            return np.full_like(x, -np.inf, dtype=float)
        return -(t / b) * np.logaddexp(0.0, cls._log_u_np(x, b, a, x0)) - lZ

    # ------------------------------------------------------------------
    # Per-channel MAP fit in (b, t, a, x0)
    #
    # Key design choices:
    #   1. Optimize over log(a) so a stays positive; clip to [0, a_max].
    #   2. Channel data is normalised to unit MAD before fitting and x0 is
    #      rescaled back afterwards — this breaks the a/x0 co-linearity.
    #   3. MAP penalty: lam * a  (half-normal on a, pulls toward pure GT).
    #   4. self selection: LR test GT (a=0) vs tempered (a free).
    #   5. logZ failures return 1e12 to keep Nelder-Mead alive.
    # ------------------------------------------------------------------
    @classmethod
    def _fit_channel(cls, h_raw,
                     b0=1.0, t0=4.0, a0=0.1,
                     lam=1.0, lr_thresh=2.0,
                     b_bounds=(0.05, 10.0), t_bounds=(0.5, 50.0),
                     a_max=5.0, eps_scale=1e-6):
        """
        Returns (b, t, a, x0) with a in [0, a_max] and x0 in original units.
        """
        # --- normalise to unit MAD (fitting in normalised space) ---
        h_scale = float(1.4826 * np.median(np.abs(h_raw - np.median(h_raw))))
        h_scale = h_scale or float(np.std(h_raw)) or 1.0
        h = h_raw / h_scale          # fitting domain: ~unit scale
        x0_unit = 1.0                # pin x0=1 in normalised space

        a_floor = 1e-6               # numerical zero for GT limit

        # ---- unpack: log-space vector → (b, t, a) --------------------
        # We always pin x0=1 in normalised space (fit_scale=False).
        # The optimiser therefore has 3 free params: [log b, log t, log a].
        def unpack(th, free_a):
            b_ = float(np.clip(np.exp(th[0]), *b_bounds))
            t_ = float(np.clip(np.exp(th[1]), *t_bounds))
            if free_a:
                a_ = float(np.clip(np.exp(th[2]), a_floor, a_max))
            else:
                a_ = a_floor
            return b_, t_, a_

        # ---- full self (a free) --------------------------------------
        th0_full = np.array([np.log(b0), np.log(t0), np.log(a0)])

        def nll_full(th):
            b_, t_, a_ = unpack(th, free_a=True)
            lZ = cls._logZ_np(b_, t_, a_, x0_unit)
            if not np.isfinite(lZ):
                return 1e12
            ll  = cls._logpdf_np(h, b_, t_, a_, x0_unit).sum()
            pen = lam * a_           # half-normal prior on a → pulls to GT
            return -ll + pen

        res_full = minimize(nll_full, th0_full, method="Nelder-Mead",
                            options=dict(xatol=1e-5, fatol=1e-5, maxiter=15000))
        b_f, t_f, a_f = unpack(res_full.x, free_a=True)
        ll_full = cls._logpdf_np(h, b_f, t_f, a_f, x0_unit).sum()

        # ---- GT limit (a pinned at floor) -----------------------------
        th0_gt = np.array([np.log(b0), np.log(t0)])

        def nll_gt(th):
            b_, t_, a_ = unpack(np.append(th, np.log(a_floor)), free_a=False)
            lZ = cls._logZ_np(b_, t_, a_, x0_unit)
            if not np.isfinite(lZ):
                return 1e12
            return -cls._logpdf_np(h, b_, t_, a_, x0_unit).sum()

        res_gt = minimize(nll_gt, th0_gt, method="Nelder-Mead",
                          options=dict(xatol=1e-5, fatol=1e-5, maxiter=15000))
        b_g, t_g, a_g = unpack(np.append(res_gt.x, np.log(a_floor)), free_a=False)
        ll_gt = cls._logpdf_np(h, b_g, t_g, a_g, x0_unit).sum()

        # ---- likelihood-ratio self selection -------------------------
        tempered = 2.0 * (ll_full - ll_gt) > lr_thresh
        if tempered:
            b_out, t_out, a_out = b_f, t_f, a_f
        else:
            b_out, t_out, a_out = b_g, t_g, 0.0

        # --- rescale x0 back to original units -------------------------
        # In normalised space x0=1 and |x|_typical=1, so in original units
        # x0 = h_scale * x0_unit = h_scale.
        x0_out = h_scale * x0_unit

        return b_out, t_out, a_out, x0_out

    # ------------------------------------------------------------------
    # fit_reference: fit all channels from a batch of signals x
    # ------------------------------------------------------------------
    def fit_reference(self, x, lam=1.0, lr_thresh=2.0):
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real.detach().cpu().numpy()

        bb, tt, aa, xx = [], [], [], []
        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            if h.size < 10:
                b_, t_, a_, x0_ = 1.0, 4.0, 0.0, 1.0
            else:
                try:
                    b_, t_, a_, x0_ = self._fit_channel(
                        h, lam=lam, lr_thresh=lr_thresh,
                        b_bounds=self.b_bounds, t_bounds=self.t_bounds,
                        a_max=self.a_max, eps_scale=self.eps_scale)
                except Exception as e:
                    print(f"[coshGT][ch {j}] fit failed ({e}) → fallback")
                    b_, t_, a_, x0_ = 1.0, 4.0, 0.0, float(np.std(h) + self.eps_scale)

            b_  = float(np.clip(b_,  *self.b_bounds))
            t_  = float(np.clip(t_,  *self.t_bounds))
            a_  = float(np.clip(a_,  0.0, self.a_max))
            x0_ = float(max(x0_, self.eps_scale))

            tag = "tempered" if a_ > 1e-6 else "GT"
            print(f"[coshGT][ch {j}] [{tag:>8s}]  b={b_:.3f}  t={t_:.3f}  "
                  f"a={a_:.4f}  x0={x0_:.5f}  (kappa=a*t={a_*t_:.4f})")
            bb.append(b_); tt.append(t_); aa.append(a_); xx.append(x0_)

        dtype = x.dtype if x.is_floating_point() else torch.float32
        self.b  = torch.tensor(bb, dtype=dtype, device=x.device)
        self.t  = torch.tensor(tt, dtype=dtype, device=x.device)
        self.a  = torch.tensor(aa, dtype=dtype, device=x.device)
        self.x0 = torch.tensor(xx, dtype=dtype, device=x.device)

    def fit(self, x, **kwargs):
        return self.fit_reference(x, **kwargs)

    # ------------------------------------------------------------------
    # Stable torch building blocks
    # ------------------------------------------------------------------
    @staticmethod
    def _logcosh(z):
        """Numerically stable log cosh(z) in torch."""
        az = z.abs()
        return az + F.softplus(-2.0 * az) - math.log(2.0)

    def _params(self, device):
        """Broadcast to (1, J, 1) for (B, J, T) coefficient tensors."""
        return (self.a .to(device)[None, :, None],
                self.b .to(device)[None, :, None],
                self.t .to(device)[None, :, None],
                self.x0.to(device)[None, :, None])

    def _log_u(self, z, a, b, x0):
        """log_u = b * log(g_a(z)/x0),   g_a(z) = |z| cosh(az)."""
        az = torch.sqrt(z ** 2 + self.eps_abs)
        return b * (torch.log(az) + self._logcosh(a * z) - torch.log(x0))

    # ------------------------------------------------------------------
    # φ(x) = (t/b) * softplus(log_u)                    → shape (B, J)
    # ------------------------------------------------------------------
    def forward(self, x, *args):
        """Per-channel potential, averaged over time. Returns (B, J)."""
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real   # (B, J, T)
        a, b, t, x0 = self._params(x.device)
        log_u = self._log_u(z, a, b, x0)
        return ((t / b) * F.softplus(log_u)).mean(-1)          # (B, J)

    # ------------------------------------------------------------------
    # φ'(z) = t * σ(log_u) * ( z/(z²+ε) + a·tanh(az) )
    # ------------------------------------------------------------------
    def grad(self, x, v=None, means=None):
        """
        Gradient of the potential w.r.t. x, back-projected through the filters.
        If v (shape J) is given, returns the weighted sum over channels → (B,1,T).
        """
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real    # (B, J, T)
        a, b, t, x0 = self._params(x.device)

        log_u  = self._log_u(z, a, b, x0)
        dlog_g = z / (z ** 2 + self.eps_abs) + a * torch.tanh(a * z)
        dphi_dz = t * torch.sigmoid(log_u) * dlog_g            # (B, J, T)

        grad_coeff = torch.fft.ifft(
            torch.fft.fft(dphi_dz) * filters
        ).real / x.shape[-1]                                    # (B, J, T)

        if v is None:
            return grad_coeff
        return (grad_coeff * v[None, :, None]).sum(1)[:, None]  # (B, 1, T)

    # ------------------------------------------------------------------
    def summary(self):
        self._check_fitted()
        b  = self.b.cpu().numpy()
        t  = self.t.cpu().numpy()
        a  = self.a.cpu().numpy()
        x0 = self.x0.cpu().numpy()
        print(f"{'Ch':>4}  {'b':>7}  {'t':>7}  {'a':>7}  {'x0':>10}  "
              f"{'kappa=a*t':>10}  {'mode':>10}")
        print("-" * 68)
        for j in range(len(b)):
            tag = "tempered" if a[j] > 1e-6 else "GT"
            print(f"{j:>4d}  {b[j]:>7.3f}  {t[j]:>7.3f}  {a[j]:>7.4f}  "
                  f"{x0[j]:>10.5f}  {a[j]*t[j]:>10.4f}  {tag:>10}")


# coshGt but fitting on two separate windows 
class Scalar_coshgt:
    """
    Per-channel cosh-tempered Generalized-t (coshGT) potential for wavelet
    coefficients, fitted SEPARATELY on the bulk and on the tails of each
    channel's coefficient histogram.

    Why two fits
    ------------
    A single coshGT must compromise between two regimes that obey different
    laws:
        Body  (|z| small):  p(z) ~ |z|^b                    -> set by  b, x0
        Tail  (|z| large):  p(z) ~ |z|^{-t} e^{-a t |z|}     -> set by  t, a
    Fitting the whole histogram at once lets the dense body dominate the
    likelihood and washes out the (sparse but decisive) tail shape.  We
    therefore run two *weighted* maximum-likelihood fits and expose two
    *windowed* energy terms per channel:

        bulk fit :  weights w_bulk(z) ~ 1 on the body, ~0 on the tail
        tail fit :  weights w_tail(z) ~ 1 on the tail, ~0 on the body

        Phi_bulk(z) = w_bulk(z) * phi_bulk(z)    (active mainly on the body)
        Phi_tail(z) = w_tail(z) * phi_tail(z)    (active mainly on the tail)

    forward() returns (B, 2J) -> [bulk_0..J-1, tail_0..J-1] and
    num_coefficients = 2J.  grad()/v follow the same ordering.

    Non-colinearity (guaranteed by construction)
    --------------------------------------------
    The windows form a smooth partition of unity on |z|:
        w_tail(z) = sigmoid((|z| - c)/s),   w_bulk(z) = 1 - w_tail(z).
    Because w_bulk and w_tail have essentially disjoint support, there is no
    constant lambda with  w_bulk*phi_bulk == lambda * w_tail*phi_tail  for all
    z: where one feature is non-zero the other is ~0, and (1 - w_tail) is not
    proportional to a non-constant sigmoid w_tail.  Hence the two energy terms
    are linearly independent / not colinear.  fit_reference() additionally
    measures the cosine between the two sampled potentials per channel
    (self.cos_bt) and warns if it ever approaches 1.

    Choice of the bulk/tail border c
    --------------------------------
    c is set per channel from a high quantile of |z| (default 0.90): the body
    is the dense central mass, the tail the sparse remainder.  The transition
    width s is set from the spread of |z| between two quantiles
    (default 0.85-0.97) so the switch is gentle and data-adaptive.

    Stored per channel (tensors of shape (J,)), with _bulk / _tail suffixes:
        b, t, a, x0  for each regime, plus the border c, the width s and the
        diagnostic cosine cos_bt.
    """

    def __init__(self, filters, eps_abs=1e-6, eps_scale=1e-6,
                 a_max=5.0, b_bounds=(0.05, 10.0), t_bounds=(0.5, 50.0),
                 bulk_quantile=0.99, trans_quantiles=(0.989, 0.991),
                 min_eff_samples=50.0):
        self.filters           = filters
        self.num_coefficients  = 2 * filters.shape[1]      # bulk + tail per channel
        self.eps_abs           = eps_abs
        self.eps_scale         = eps_scale
        self.a_max             = a_max
        self.b_bounds          = b_bounds
        self.t_bounds          = t_bounds
        self.bulk_quantile     = bulk_quantile
        self.trans_quantiles   = trans_quantiles
        self.min_eff_samples   = min_eff_samples
        # bulk params (J,)
        self.b_bulk = self.t_bulk = self.a_bulk = self.x0_bulk = None
        # tail params (J,)
        self.b_tail = self.t_tail = self.a_tail = self.x0_tail = None
        # window params (J,) and diagnostic
        self.c = self.s = None
        self.cos_bt = None

    @property
    def is_fitted(self):
        return self.a_bulk is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Scalar_coshgt must be fit_reference'd first.")

    # ------------------------------------------------------------------
    # numpy helpers  (fitting only — all in (b, t, a, x0) coords)
    # ------------------------------------------------------------------
    @staticmethod
    def _logcosh_np(z):
        """Numerically stable log cosh(z)."""
        z = np.abs(z)
        return z + np.log1p(np.exp(-2.0 * z)) - np.log(2.0)

    @classmethod
    def _log_u_np(cls, x, b, a, x0):
        """log_u = b * log(g_a(x)/x0),   g_a(x) = |x| cosh(ax)."""
        lx = np.log(np.maximum(np.abs(x), 1e-300))
        return b * (lx + cls._logcosh_np(a * x) - np.log(x0))

    @classmethod
    def _logZ_np(cls, b, t, a, x0):
        """
        log normalisation constant.  Returns np.inf on numerical failure
        so the optimizer sees a large-penalty signal rather than nan/crash.
        """
        try:
            f = lambda s: np.exp(-(t / b) * np.logaddexp(0.0,
                                   cls._log_u_np(s, b, a, x0)))
            I, _ = quad(f, 0.0, np.inf, limit=200)
            if not np.isfinite(I) or I <= 0.0:
                return np.inf
            return np.log(2.0 * I)
        except Exception:
            return np.inf

    @classmethod
    def _logpdf_np(cls, x, b, t, a, x0):
        lZ = cls._logZ_np(b, t, a, x0)
        if not np.isfinite(lZ):
            return np.full_like(x, -np.inf, dtype=float)
        return -(t / b) * np.logaddexp(0.0, cls._log_u_np(x, b, a, x0)) - lZ

    @staticmethod
    def _weighted_median(values, weights):
        """Weighted median of `values` (used for a region-aware robust scale)."""
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if values.size == 0:
            return 0.0
        order = np.argsort(values)
        v, w = values[order], weights[order]
        cw = np.cumsum(w)
        if cw[-1] <= 0:
            return float(np.median(values))
        idx = int(np.searchsorted(cw, 0.5 * cw[-1]))
        idx = min(idx, len(v) - 1)
        return float(v[idx])

    # ------------------------------------------------------------------
    # Per-channel WEIGHTED MAP fit in (b, t, a, x0).
    #
    #   * `weights` (>=0, same shape as h_raw) reweight the log-likelihood so
    #     the fit concentrates on the bulk or on the tail.
    #   * Data is normalised to a region-aware unit scale before fitting and
    #     x0 is rescaled back afterwards — this breaks the a/x0 co-linearity.
    #   * MAP penalty lam*a (half-normal on a, pulls toward pure GT).
    #   * self selection: weighted LR test GT (a=0) vs tempered (a free).
    # ------------------------------------------------------------------
    @classmethod
    def _fit_channel(cls, h_raw, weights=None,
                     b0=1.0, t0=4.0, a0=0.1,
                     lam=1.0, lr_thresh=2.0,
                     b_bounds=(0.05, 10.0), t_bounds=(0.5, 50.0),
                     a_max=5.0, eps_scale=1e-6):
        """Returns (b, t, a, x0) with a in [0, a_max] and x0 in original units."""
        h_raw = np.asarray(h_raw, dtype=float)
        if weights is None:
            weights = np.ones_like(h_raw)
        w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
        if w.sum() <= 0:
            w = np.ones_like(h_raw)

        # --- region-aware robust scale (weighted MAD about 0; coeffs are
        #     zero-mean & symmetric) → normalise the region to ~unit scale ---
        h_scale = float(1.4826 * cls._weighted_median(np.abs(h_raw), w))
        if not (h_scale > 0):
            h_scale = float(np.sqrt((w * h_raw ** 2).sum() / w.sum()))
        h_scale = h_scale or 1.0
        h = h_raw / h_scale
        x0_unit = 1.0
        a_floor = 1e-6

        def unpack(th, free_a):
            b_ = float(np.clip(np.exp(th[0]), *b_bounds))
            t_ = float(np.clip(np.exp(th[1]), *t_bounds))
            a_ = float(np.clip(np.exp(th[2]), a_floor, a_max)) if free_a else a_floor
            return b_, t_, a_

        # ---- full self (a free) ----
        th0_full = np.array([np.log(b0), np.log(t0), np.log(a0)])

        def nll_full(th):
            b_, t_, a_ = unpack(th, free_a=True)
            if not np.isfinite(cls._logZ_np(b_, t_, a_, x0_unit)):
                return 1e12
            ll  = float((w * cls._logpdf_np(h, b_, t_, a_, x0_unit)).sum())
            return -ll + lam * a_

        res_full = minimize(nll_full, th0_full, method="Nelder-Mead",
                            options=dict(xatol=1e-5, fatol=1e-5, maxiter=15000))
        b_f, t_f, a_f = unpack(res_full.x, free_a=True)
        ll_full = float((w * cls._logpdf_np(h, b_f, t_f, a_f, x0_unit)).sum())

        # ---- GT limit (a pinned at floor) ----
        th0_gt = np.array([np.log(b0), np.log(t0)])

        def nll_gt(th):
            b_, t_, a_ = unpack(np.append(th, np.log(a_floor)), free_a=False)
            if not np.isfinite(cls._logZ_np(b_, t_, a_, x0_unit)):
                return 1e12
            return -float((w * cls._logpdf_np(h, b_, t_, a_, x0_unit)).sum())

        res_gt = minimize(nll_gt, th0_gt, method="Nelder-Mead",
                          options=dict(xatol=1e-5, fatol=1e-5, maxiter=15000))
        b_g, t_g, a_g = unpack(np.append(res_gt.x, np.log(a_floor)), free_a=False)
        ll_gt = float((w * cls._logpdf_np(h, b_g, t_g, a_g, x0_unit)).sum())

        # ---- weighted likelihood-ratio self selection ----
        if 2.0 * (ll_full - ll_gt) > lr_thresh:
            b_out, t_out, a_out = b_f, t_f, a_f
        else:
            b_out, t_out, a_out = b_g, t_g, 0.0

        x0_out = h_scale * x0_unit   # |x|_typical=1 in normalised space
        return b_out, t_out, a_out, x0_out

    # ------------------------------------------------------------------
    # Diagnostic: cosine between the two windowed, mean-removed potentials
    # sampled on a |z| grid.  ~0 => well separated, ->1 => colinear.
    # ------------------------------------------------------------------
    @classmethod
    def _potential_cosine(cls, c, s, par_bulk, par_tail, n=1024):
        b_b, t_b, a_b, x0_b = par_bulk
        b_t, t_t, a_t, x0_t = par_tail
        span = c + 10.0 * s
        zg = np.linspace(-span, span, n)
        az = np.abs(zg)
        wt = 1.0 / (1.0 + np.exp(-(az - c) / s))
        wb = 1.0 - wt
        def phi(zz, b, t, a, x0):
            return (t / b) * np.logaddexp(0.0, cls._log_u_np(zz, b, a, x0))
        fb = wb * phi(zg, b_b, t_b, a_b, x0_b)
        ft = wt * phi(zg, b_t, t_t, a_t, x0_t)
        fb = fb - fb.mean(); ft = ft - ft.mean()
        denom = np.linalg.norm(fb) * np.linalg.norm(ft)
        return float(abs(fb @ ft) / denom) if denom > 0 else 0.0

    # ------------------------------------------------------------------
    # fit_reference: fit bulk + tail for all channels from a batch x
    # ------------------------------------------------------------------
    def fit_reference(self, x, lam=1.0, lr_thresh=2.0,
                      bulk_quantile=None, trans_quantiles=None):
        if bulk_quantile is None:
            bulk_quantile = self.bulk_quantile
        if trans_quantiles is None:
            trans_quantiles = self.trans_quantiles

        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real.detach().cpu().numpy()

        bb_b, tt_b, aa_b, xx_b = [], [], [], []
        bb_t, tt_t, aa_t, xx_t = [], [], [], []
        cc, ss, cosines = [], [], []

        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            ah = np.abs(h)

            if h.size < 10:
                b_, t_, x0_ = 1.0, 4.0, 1.0
                c_ = float(np.median(ah)) if ah.size else 1.0
                s_ = max(0.1 * c_, self.eps_scale)
                bb_b += [b_]; tt_b += [t_]; aa_b += [0.0]; xx_b += [x0_]
                bb_t += [b_]; tt_t += [max(t_, 1.0)]; aa_t += [0.0]; xx_t += [x0_]
                cc += [c_]; ss += [s_]; cosines += [0.0]
                continue

            # --- bulk/tail border c and transition width s from |z| quantiles ---
            c_ = float(np.quantile(ah, bulk_quantile))
            q_lo, q_hi = trans_quantiles
            band = float(np.quantile(ah, q_hi) - np.quantile(ah, q_lo))
            s_ = band / 4.0                      # ~ +/-2 logistic scales over the band
            s_ = max(s_, 0.05 * (c_ + self.eps_scale), self.eps_scale)

            # --- smooth partition-of-unity windows on |z| ---
            w_tail = 1.0 / (1.0 + np.exp(-(ah - c_) / s_))
            w_bulk = 1.0 - w_tail

            eff_tail = (w_tail.sum() ** 2) / max((w_tail ** 2).sum(), 1e-12)
            if eff_tail < self.min_eff_samples:
                print(f"[coshGT2][ch {j}] tail eff. N={eff_tail:.1f} < "
                      f"{self.min_eff_samples:.0f}: tail fit may be weak "
                      f"(consider lowering bulk_quantile).")

            # --- weighted fits: bulk (body shape) and tail (tail shape) ---
            try:
                b_bk, t_bk, a_bk, x0_bk = self._fit_channel(
                    h, weights=w_bulk, a0=0.05, lam=lam, lr_thresh=lr_thresh,
                    b_bounds=self.b_bounds, t_bounds=self.t_bounds,
                    a_max=self.a_max, eps_scale=self.eps_scale)
            except Exception as e:
                print(f"[coshGT2][ch {j}] bulk fit failed ({e}) -> fallback")
                b_bk, t_bk, a_bk, x0_bk = 1.0, 6.0, 0.0, float(np.std(h) + self.eps_scale)

            try:
                b_tl, t_tl, a_tl, x0_tl = self._fit_channel(
                    h, weights=w_tail, a0=0.10, lam=lam, lr_thresh=lr_thresh,
                    b_bounds=self.b_bounds, t_bounds=self.t_bounds,
                    a_max=self.a_max, eps_scale=self.eps_scale)
            except Exception as e:
                print(f"[coshGT2][ch {j}] tail fit failed ({e}) -> fallback")
                b_tl, t_tl, a_tl, x0_tl = 1.0, 3.0, 0.0, float(np.std(h) + self.eps_scale)

            # --- clip to bounds ---
            b_bk = float(np.clip(b_bk, *self.b_bounds)); t_bk = float(np.clip(t_bk, *self.t_bounds))
            a_bk = float(np.clip(a_bk, 0.0, self.a_max)); x0_bk = float(max(x0_bk, self.eps_scale))
            b_tl = float(np.clip(b_tl, *self.b_bounds)); t_tl = float(np.clip(t_tl, *self.t_bounds))
            a_tl = float(np.clip(a_tl, 0.0, self.a_max)); x0_tl = float(max(x0_tl, self.eps_scale))

            # --- non-colinearity diagnostic ---
            cos_bt = self._potential_cosine(
                c_, s_, (b_bk, t_bk, a_bk, x0_bk), (b_tl, t_tl, a_tl, x0_tl))
            if cos_bt > 0.98:
                print(f"[coshGT2][ch {j}] WARNING: potentials nearly colinear "
                      f"(cos={cos_bt:.3f}); move the bulk/tail border (bulk_quantile) "
                      f"or widen the gap between regimes.")

            print(f"[coshGT2][ch {j}] c={c_:.4f} s={s_:.4f} | "
                  f"bulk(b={b_bk:.3f},t={t_bk:.3f},a={a_bk:.4f},x0={x0_bk:.4f}) | "
                  f"tail(b={b_tl:.3f},t={t_tl:.3f},a={a_tl:.4f},x0={x0_tl:.4f}) | "
                  f"cos={cos_bt:.3f}")

            bb_b += [b_bk]; tt_b += [t_bk]; aa_b += [a_bk]; xx_b += [x0_bk]
            bb_t += [b_tl]; tt_t += [t_tl]; aa_t += [a_tl]; xx_t += [x0_tl]
            cc += [c_]; ss += [s_]; cosines += [cos_bt]

        dtype = x.dtype if x.is_floating_point() else torch.float32
        dev = x.device
        mk = lambda L: torch.tensor(L, dtype=dtype, device=dev)
        self.b_bulk, self.t_bulk, self.a_bulk, self.x0_bulk = mk(bb_b), mk(tt_b), mk(aa_b), mk(xx_b)
        self.b_tail, self.t_tail, self.a_tail, self.x0_tail = mk(bb_t), mk(tt_t), mk(aa_t), mk(xx_t)
        self.c, self.s = mk(cc), mk(ss)
        self.cos_bt = mk(cosines)

    def fit(self, x, **kwargs):
        return self.fit_reference(x, **kwargs)

    # ------------------------------------------------------------------
    # Stable torch building blocks
    # ------------------------------------------------------------------
    @staticmethod
    def _logcosh(z):
        """Numerically stable log cosh(z) in torch."""
        az = z.abs()
        return az + F.softplus(-2.0 * az) - math.log(2.0)

    def _params_bulk(self, device):
        return (self.a_bulk.to(device)[None, :, None], self.b_bulk.to(device)[None, :, None],
                self.t_bulk.to(device)[None, :, None], self.x0_bulk.to(device)[None, :, None])

    def _params_tail(self, device):
        return (self.a_tail.to(device)[None, :, None], self.b_tail.to(device)[None, :, None],
                self.t_tail.to(device)[None, :, None], self.x0_tail.to(device)[None, :, None])

    def _log_u(self, z, a, b, x0):
        """log_u = b * log(g_a(z)/x0),   g_a(z) = |z| cosh(az)."""
        az = torch.sqrt(z ** 2 + self.eps_abs)
        return b * (torch.log(az) + self._logcosh(a * z) - torch.log(x0))

    def _phi(self, z, a, b, t, x0):
        """phi(z) = (t/b) * softplus(log_u)  =  -log p(z) + const."""
        return (t / b) * F.softplus(self._log_u(z, a, b, x0))

    def _dphi(self, z, a, b, t, x0):
        """phi'(z) = t * sigmoid(log_u) * ( z/(z^2+eps) + a*tanh(a z) )."""
        log_u = self._log_u(z, a, b, x0)
        dlog_g = z / (z ** 2 + self.eps_abs) + a * torch.tanh(a * z)
        return t * torch.sigmoid(log_u) * dlog_g

    def _windows(self, z):
        """Smooth partition of unity on |z|:  (w_tail, w_bulk)."""
        c = self.c.to(z.device)[None, :, None]
        s = self.s.to(z.device)[None, :, None]
        az = torch.sqrt(z ** 2 + self.eps_abs)
        w_tail = torch.sigmoid((az - c) / s)
        return w_tail, 1.0 - w_tail

    # ------------------------------------------------------------------
    # forward:  [w_bulk*phi_bulk ; w_tail*phi_tail] averaged over time
    #           -> (B, 2J)   (first J = bulk, last J = tail)
    # ------------------------------------------------------------------
    def forward(self, x, *args):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real        # (B, J, T)
        w_tail, w_bulk = self._windows(z)
        a_b, b_b, t_b, x0_b = self._params_bulk(x.device)
        a_t, b_t, t_t, x0_t = self._params_tail(x.device)
        phi_bulk = (w_bulk * self._phi(z, a_b, b_b, t_b, x0_b)).mean(-1)   # (B, J)
        phi_tail = (w_tail * self._phi(z, a_t, b_t, t_t, x0_t)).mean(-1)   # (B, J)
        return torch.cat([phi_bulk, phi_tail], dim=1)                      # (B, 2J)

    # ------------------------------------------------------------------
    # grad:  d/dx of each windowed potential, back-projected through filters.
    #        Product rule:  d/dz [w(z) phi(z)] = w'(z) phi(z) + w(z) phi'(z),
    #        with  w_tail'(z) = w_tail(1-w_tail) * (z/|z|) / s,  w_bulk' = -w_tail'.
    #        Returns (B, 2J, T), or (B, 1, T) if v (length 2J) is given.
    # ------------------------------------------------------------------
    def grad(self, x, v=None, means=None):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real        # (B, J, T)
        a_b, b_b, t_b, x0_b = self._params_bulk(x.device)
        a_t, b_t, t_t, x0_t = self._params_tail(x.device)

        c = self.c.to(x.device)[None, :, None]
        s = self.s.to(x.device)[None, :, None]
        az = torch.sqrt(z ** 2 + self.eps_abs)
        w_tail = torch.sigmoid((az - c) / s)
        w_bulk = 1.0 - w_tail
        dw_tail = w_tail * (1.0 - w_tail) * (z / az) / s           # d w_tail / dz
        dw_bulk = -dw_tail

        phi_b = self._phi(z, a_b, b_b, t_b, x0_b); dphi_b = self._dphi(z, a_b, b_b, t_b, x0_b)
        phi_t = self._phi(z, a_t, b_t, t_t, x0_t); dphi_t = self._dphi(z, a_t, b_t, t_t, x0_t)
        D_bulk = dw_bulk * phi_b + w_bulk * dphi_b                 # (B, J, T)
        D_tail = dw_tail * phi_t + w_tail * dphi_t                 # (B, J, T)

        def backproj(D):
            return torch.fft.ifft(torch.fft.fft(D) * filters).real / x.shape[-1]

        grad_coeff = torch.cat([backproj(D_bulk), backproj(D_tail)], dim=1)  # (B, 2J, T)

        if v is None:
            return grad_coeff
        return (grad_coeff * v[None, :, None]).sum(1)[:, None]     # (B, 1, T)

    # ------------------------------------------------------------------
    def summary(self):
        self._check_fitted()
        bb = self.b_bulk.cpu().numpy(); tb = self.t_bulk.cpu().numpy()
        ab = self.a_bulk.cpu().numpy(); xb = self.x0_bulk.cpu().numpy()
        bt = self.b_tail.cpu().numpy(); tt = self.t_tail.cpu().numpy()
        at = self.a_tail.cpu().numpy(); xt = self.x0_tail.cpu().numpy()
        c = self.c.cpu().numpy(); s = self.s.cpu().numpy()
        cos = self.cos_bt.cpu().numpy() if self.cos_bt is not None else np.zeros_like(c)
        print(f"{'Ch':>3} {'c':>9} {'s':>8} | "
              f"{'b_bk':>6} {'t_bk':>6} {'a_bk':>6} {'x0_bk':>8} | "
              f"{'b_tl':>6} {'t_tl':>6} {'a_tl':>6} {'x0_tl':>8} | {'cos':>5}")
        print("-" * 98)
        for j in range(len(bb)):
            print(f"{j:>3d} {c[j]:>9.4f} {s[j]:>8.4f} | "
                  f"{bb[j]:>6.3f} {tb[j]:>6.3f} {ab[j]:>6.3f} {xb[j]:>8.4f} | "
                  f"{bt[j]:>6.3f} {tt[j]:>6.3f} {at[j]:>6.3f} {xt[j]:>8.4f} | "
                  f"{cos[j]:>5.3f}")


# coshGt but fitting on two separate windows 
class Scalar_coshgt_imag:
    """
    Per-channel cosh-tempered Generalized-t (coshGT) potential for wavelet
    coefficients, fitted SEPARATELY on the bulk and on the tails of each
    channel's coefficient histogram.

    Why two fits
    ------------
    A single coshGT must compromise between two regimes that obey different
    laws:
        Body  (|z| small):  p(z) ~ |z|^b                    -> set by  b, x0
        Tail  (|z| large):  p(z) ~ |z|^{-t} e^{-a t |z|}     -> set by  t, a
    Fitting the whole histogram at once lets the dense body dominate the
    likelihood and washes out the (sparse but decisive) tail shape.  We
    therefore run two *weighted* maximum-likelihood fits and expose two
    *windowed* energy terms per channel:

        bulk fit :  weights w_bulk(z) ~ 1 on the body, ~0 on the tail
        tail fit :  weights w_tail(z) ~ 1 on the tail, ~0 on the body

        Phi_bulk(z) = w_bulk(z) * phi_bulk(z)    (active mainly on the body)
        Phi_tail(z) = w_tail(z) * phi_tail(z)    (active mainly on the tail)

    forward() returns (B, 2J) -> [bulk_0..J-1, tail_0..J-1] and
    num_coefficients = 2J.  grad()/v follow the same ordering.

    Non-colinearity (guaranteed by construction)
    --------------------------------------------
    The windows form a smooth partition of unity on |z|:
        w_tail(z) = sigmoid((|z| - c)/s),   w_bulk(z) = 1 - w_tail(z).
    Because w_bulk and w_tail have essentially disjoint support, there is no
    constant lambda with  w_bulk*phi_bulk == lambda * w_tail*phi_tail  for all
    z: where one feature is non-zero the other is ~0, and (1 - w_tail) is not
    proportional to a non-constant sigmoid w_tail.  Hence the two energy terms
    are linearly independent / not colinear.  fit_reference() additionally
    measures the cosine between the two sampled potentials per channel
    (self.cos_bt) and warns if it ever approaches 1.

    Choice of the bulk/tail border c
    --------------------------------
    c is set per channel from a high quantile of |z| (default 0.90): the body
    is the dense central mass, the tail the sparse remainder.  The transition
    width s is set from the spread of |z| between two quantiles
    (default 0.85-0.97) so the switch is gentle and data-adaptive.

    Stored per channel (tensors of shape (J,)), with _bulk / _tail suffixes:
        b, t, a, x0  for each regime, plus the border c, the width s and the
        diagnostic cosine cos_bt.
    """

    def __init__(self, filters, eps_abs=1e-6, eps_scale=1e-6,
                 a_max=5.0, b_bounds=(0.05, 10.0), t_bounds=(0.5, 50.0),
                 bulk_quantile=0.99, trans_quantiles=(0.985, 0.995),
                 min_eff_samples=50.0):
        self.filters           = filters
        self.num_coefficients  = 2 * filters.shape[1]      # bulk + tail per channel
        self.eps_abs           = eps_abs
        self.eps_scale         = eps_scale
        self.a_max             = a_max
        self.b_bounds          = b_bounds
        self.t_bounds          = t_bounds
        self.bulk_quantile     = bulk_quantile
        self.trans_quantiles   = trans_quantiles
        self.min_eff_samples   = min_eff_samples
        # bulk params (J,)
        self.b_bulk = self.t_bulk = self.a_bulk = self.x0_bulk = None
        # tail params (J,)
        self.b_tail = self.t_tail = self.a_tail = self.x0_tail = None
        # window params (J,) and diagnostic
        self.c = self.s = None
        self.cos_bt = None

    @property
    def is_fitted(self):
        return self.a_bulk is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Scalar_coshgt must be fit_reference'd first.")

    # ------------------------------------------------------------------
    # numpy helpers  (fitting only — all in (b, t, a, x0) coords)
    # ------------------------------------------------------------------
    @staticmethod
    def _logcosh_np(z):
        """Numerically stable log cosh(z)."""
        z = np.abs(z)
        return z + np.log1p(np.exp(-2.0 * z)) - np.log(2.0)

    @classmethod
    def _log_u_np(cls, x, b, a, x0):
        """log_u = b * log(g_a(x)/x0),   g_a(x) = |x| cosh(ax)."""
        lx = np.log(np.maximum(np.abs(x), 1e-300))
        return b * (lx + cls._logcosh_np(a * x) - np.log(x0))

    @classmethod
    def _logZ_np(cls, b, t, a, x0):
        """
        log normalisation constant.  Returns np.inf on numerical failure
        so the optimizer sees a large-penalty signal rather than nan/crash.
        """
        try:
            f = lambda s: np.exp(-(t / b) * np.logaddexp(0.0,
                                   cls._log_u_np(s, b, a, x0)))
            I, _ = quad(f, 0.0, np.inf, limit=200)
            if not np.isfinite(I) or I <= 0.0:
                return np.inf
            return np.log(2.0 * I)
        except Exception:
            return np.inf

    @classmethod
    def _logpdf_np(cls, x, b, t, a, x0):
        lZ = cls._logZ_np(b, t, a, x0)
        if not np.isfinite(lZ):
            return np.full_like(x, -np.inf, dtype=float)
        return -(t / b) * np.logaddexp(0.0, cls._log_u_np(x, b, a, x0)) - lZ

    @staticmethod
    def _weighted_median(values, weights):
        """Weighted median of `values` (used for a region-aware robust scale)."""
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if values.size == 0:
            return 0.0
        order = np.argsort(values)
        v, w = values[order], weights[order]
        cw = np.cumsum(w)
        if cw[-1] <= 0:
            return float(np.median(values))
        idx = int(np.searchsorted(cw, 0.5 * cw[-1]))
        idx = min(idx, len(v) - 1)
        return float(v[idx])

    # ------------------------------------------------------------------
    # Per-channel WEIGHTED MAP fit in (b, t, a, x0).
    #
    #   * `weights` (>=0, same shape as h_raw) reweight the log-likelihood so
    #     the fit concentrates on the bulk or on the tail.
    #   * Data is normalised to a region-aware unit scale before fitting and
    #     x0 is rescaled back afterwards — this breaks the a/x0 co-linearity.
    #   * MAP penalty lam*a (half-normal on a, pulls toward pure GT).
    #   * self selection: weighted LR test GT (a=0) vs tempered (a free).
    # ------------------------------------------------------------------
    @classmethod
    def _fit_channel(cls, h_raw, weights=None,
                     b0=1.0, t0=4.0, a0=0.1,
                     lam=1.0, lr_thresh=2.0,
                     b_bounds=(0.05, 10.0), t_bounds=(0.5, 50.0),
                     a_max=5.0, eps_scale=1e-6):
        """Returns (b, t, a, x0) with a in [0, a_max] and x0 in original units."""
        h_raw = np.asarray(h_raw, dtype=float)
        if weights is None:
            weights = np.ones_like(h_raw)
        w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
        if w.sum() <= 0:
            w = np.ones_like(h_raw)

        # --- region-aware robust scale (weighted MAD about 0; coeffs are
        #     zero-mean & symmetric) → normalise the region to ~unit scale ---
        h_scale = float(1.4826 * cls._weighted_median(np.abs(h_raw), w))
        if not (h_scale > 0):
            h_scale = float(np.sqrt((w * h_raw ** 2).sum() / w.sum()))
        h_scale = h_scale or 1.0
        h = h_raw / h_scale
        x0_unit = 1.0
        a_floor = 1e-6

        def unpack(th, free_a):
            b_ = float(np.clip(np.exp(th[0]), *b_bounds))
            t_ = float(np.clip(np.exp(th[1]), *t_bounds))
            a_ = float(np.clip(np.exp(th[2]), a_floor, a_max)) if free_a else a_floor
            return b_, t_, a_

        # ---- full self (a free) ----
        th0_full = np.array([np.log(b0), np.log(t0), np.log(a0)])

        def nll_full(th):
            b_, t_, a_ = unpack(th, free_a=True)
            if not np.isfinite(cls._logZ_np(b_, t_, a_, x0_unit)):
                return 1e12
            ll  = float((w * cls._logpdf_np(h, b_, t_, a_, x0_unit)).sum())
            return -ll + lam * a_

        res_full = minimize(nll_full, th0_full, method="Nelder-Mead",
                            options=dict(xatol=1e-5, fatol=1e-5, maxiter=15000))
        b_f, t_f, a_f = unpack(res_full.x, free_a=True)
        ll_full = float((w * cls._logpdf_np(h, b_f, t_f, a_f, x0_unit)).sum())

        # ---- GT limit (a pinned at floor) ----
        th0_gt = np.array([np.log(b0), np.log(t0)])

        def nll_gt(th):
            b_, t_, a_ = unpack(np.append(th, np.log(a_floor)), free_a=False)
            if not np.isfinite(cls._logZ_np(b_, t_, a_, x0_unit)):
                return 1e12
            return -float((w * cls._logpdf_np(h, b_, t_, a_, x0_unit)).sum())

        res_gt = minimize(nll_gt, th0_gt, method="Nelder-Mead",
                          options=dict(xatol=1e-5, fatol=1e-5, maxiter=15000))
        b_g, t_g, a_g = unpack(np.append(res_gt.x, np.log(a_floor)), free_a=False)
        ll_gt = float((w * cls._logpdf_np(h, b_g, t_g, a_g, x0_unit)).sum())

        # ---- weighted likelihood-ratio self selection ----
        if 2.0 * (ll_full - ll_gt) > lr_thresh:
            b_out, t_out, a_out = b_f, t_f, a_f
        else:
            b_out, t_out, a_out = b_g, t_g, 0.0

        x0_out = h_scale * x0_unit   # |x|_typical=1 in normalised space
        return b_out, t_out, a_out, x0_out

    # ------------------------------------------------------------------
    # Diagnostic: cosine between the two windowed, mean-removed potentials
    # sampled on a |z| grid.  ~0 => well separated, ->1 => colinear.
    # ------------------------------------------------------------------
    @classmethod
    def _potential_cosine(cls, c, s, par_bulk, par_tail, n=1024):
        b_b, t_b, a_b, x0_b = par_bulk
        b_t, t_t, a_t, x0_t = par_tail
        span = c + 10.0 * s
        zg = np.linspace(-span, span, n)
        az = np.abs(zg)
        wt = 1.0 / (1.0 + np.exp(-(az - c) / s))
        wb = 1.0 - wt
        def phi(zz, b, t, a, x0):
            return (t / b) * np.logaddexp(0.0, cls._log_u_np(zz, b, a, x0))
        fb = wb * phi(zg, b_b, t_b, a_b, x0_b)
        ft = wt * phi(zg, b_t, t_t, a_t, x0_t)
        fb = fb - fb.mean(); ft = ft - ft.mean()
        denom = np.linalg.norm(fb) * np.linalg.norm(ft)
        return float(abs(fb @ ft) / denom) if denom > 0 else 0.0

    # ------------------------------------------------------------------
    # fit_reference: fit bulk + tail for all channels from a batch x
    # ------------------------------------------------------------------
    def fit_reference(self, x, lam=1.0, lr_thresh=2.0,
                      bulk_quantile=None, trans_quantiles=None):
        if bulk_quantile is None:
            bulk_quantile = self.bulk_quantile
        if trans_quantiles is None:
            trans_quantiles = self.trans_quantiles

        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).imag.detach().cpu().numpy()

        bb_b, tt_b, aa_b, xx_b = [], [], [], []
        bb_t, tt_t, aa_t, xx_t = [], [], [], []
        cc, ss, cosines = [], [], []

        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            ah = np.abs(h)

            if h.size < 10:
                b_, t_, x0_ = 1.0, 4.0, 1.0
                c_ = float(np.median(ah)) if ah.size else 1.0
                s_ = max(0.1 * c_, self.eps_scale)
                bb_b += [b_]; tt_b += [t_]; aa_b += [0.0]; xx_b += [x0_]
                bb_t += [b_]; tt_t += [max(t_, 1.0)]; aa_t += [0.0]; xx_t += [x0_]
                cc += [c_]; ss += [s_]; cosines += [0.0]
                continue

            # --- bulk/tail border c and transition width s from |z| quantiles ---
            c_ = float(np.quantile(ah, bulk_quantile))
            q_lo, q_hi = trans_quantiles
            band = float(np.quantile(ah, q_hi) - np.quantile(ah, q_lo))
            s_ = band / 4.0                      # ~ +/-2 logistic scales over the band
            s_ = max(s_, 0.05 * (c_ + self.eps_scale), self.eps_scale)

            # --- smooth partition-of-unity windows on |z| ---
            w_tail = 1.0 / (1.0 + np.exp(-(ah - c_) / s_))
            w_bulk = 1.0 - w_tail

            eff_tail = (w_tail.sum() ** 2) / max((w_tail ** 2).sum(), 1e-12)
            if eff_tail < self.min_eff_samples:
                print(f"[coshGT2][ch {j}] tail eff. N={eff_tail:.1f} < "
                      f"{self.min_eff_samples:.0f}: tail fit may be weak "
                      f"(consider lowering bulk_quantile).")

            # --- weighted fits: bulk (body shape) and tail (tail shape) ---
            try:
                b_bk, t_bk, a_bk, x0_bk = self._fit_channel(
                    h, weights=w_bulk, a0=0.05, lam=lam, lr_thresh=lr_thresh,
                    b_bounds=self.b_bounds, t_bounds=self.t_bounds,
                    a_max=self.a_max, eps_scale=self.eps_scale)
            except Exception as e:
                print(f"[coshGT2][ch {j}] bulk fit failed ({e}) -> fallback")
                b_bk, t_bk, a_bk, x0_bk = 1.0, 6.0, 0.0, float(np.std(h) + self.eps_scale)

            try:
                b_tl, t_tl, a_tl, x0_tl = self._fit_channel(
                    h, weights=w_tail, a0=0.10, lam=lam, lr_thresh=lr_thresh,
                    b_bounds=self.b_bounds, t_bounds=self.t_bounds,
                    a_max=self.a_max, eps_scale=self.eps_scale)
            except Exception as e:
                print(f"[coshGT2][ch {j}] tail fit failed ({e}) -> fallback")
                b_tl, t_tl, a_tl, x0_tl = 1.0, 3.0, 0.0, float(np.std(h) + self.eps_scale)

            # --- clip to bounds ---
            b_bk = float(np.clip(b_bk, *self.b_bounds)); t_bk = float(np.clip(t_bk, *self.t_bounds))
            a_bk = float(np.clip(a_bk, 0.0, self.a_max)); x0_bk = float(max(x0_bk, self.eps_scale))
            b_tl = float(np.clip(b_tl, *self.b_bounds)); t_tl = float(np.clip(t_tl, *self.t_bounds))
            a_tl = float(np.clip(a_tl, 0.0, self.a_max)); x0_tl = float(max(x0_tl, self.eps_scale))

            # --- non-colinearity diagnostic ---
            cos_bt = self._potential_cosine(
                c_, s_, (b_bk, t_bk, a_bk, x0_bk), (b_tl, t_tl, a_tl, x0_tl))
            if cos_bt > 0.98:
                print(f"[coshGT2][ch {j}] WARNING: potentials nearly colinear "
                      f"(cos={cos_bt:.3f}); move the bulk/tail border (bulk_quantile) "
                      f"or widen the gap between regimes.")

            print(f"[coshGT2][ch {j}] c={c_:.4f} s={s_:.4f} | "
                  f"bulk(b={b_bk:.3f},t={t_bk:.3f},a={a_bk:.4f},x0={x0_bk:.4f}) | "
                  f"tail(b={b_tl:.3f},t={t_tl:.3f},a={a_tl:.4f},x0={x0_tl:.4f}) | "
                  f"cos={cos_bt:.3f}")

            bb_b += [b_bk]; tt_b += [t_bk]; aa_b += [a_bk]; xx_b += [x0_bk]
            bb_t += [b_tl]; tt_t += [t_tl]; aa_t += [a_tl]; xx_t += [x0_tl]
            cc += [c_]; ss += [s_]; cosines += [cos_bt]

        dtype = x.dtype if x.is_floating_point() else torch.float32
        dev = x.device
        mk = lambda L: torch.tensor(L, dtype=dtype, device=dev)
        self.b_bulk, self.t_bulk, self.a_bulk, self.x0_bulk = mk(bb_b), mk(tt_b), mk(aa_b), mk(xx_b)
        self.b_tail, self.t_tail, self.a_tail, self.x0_tail = mk(bb_t), mk(tt_t), mk(aa_t), mk(xx_t)
        self.c, self.s = mk(cc), mk(ss)
        self.cos_bt = mk(cosines)

    def fit(self, x, **kwargs):
        return self.fit_reference(x, **kwargs)

    # ------------------------------------------------------------------
    # Stable torch building blocks
    # ------------------------------------------------------------------
    @staticmethod
    def _logcosh(z):
        """Numerically stable log cosh(z) in torch."""
        az = z.abs()
        return az + F.softplus(-2.0 * az) - math.log(2.0)

    def _params_bulk(self, device):
        return (self.a_bulk.to(device)[None, :, None], self.b_bulk.to(device)[None, :, None],
                self.t_bulk.to(device)[None, :, None], self.x0_bulk.to(device)[None, :, None])

    def _params_tail(self, device):
        return (self.a_tail.to(device)[None, :, None], self.b_tail.to(device)[None, :, None],
                self.t_tail.to(device)[None, :, None], self.x0_tail.to(device)[None, :, None])

    def _log_u(self, z, a, b, x0):
        """log_u = b * log(g_a(z)/x0),   g_a(z) = |z| cosh(az)."""
        az = torch.sqrt(z ** 2 + self.eps_abs)
        return b * (torch.log(az) + self._logcosh(a * z) - torch.log(x0))

    def _phi(self, z, a, b, t, x0):
        """phi(z) = (t/b) * softplus(log_u)  =  -log p(z) + const."""
        return (t / b) * F.softplus(self._log_u(z, a, b, x0))

    def _dphi(self, z, a, b, t, x0):
        """phi'(z) = t * sigmoid(log_u) * ( z/(z^2+eps) + a*tanh(a z) )."""
        log_u = self._log_u(z, a, b, x0)
        dlog_g = z / (z ** 2 + self.eps_abs) + a * torch.tanh(a * z)
        return t * torch.sigmoid(log_u) * dlog_g

    def _windows(self, z):
        """Smooth partition of unity on |z|:  (w_tail, w_bulk)."""
        c = self.c.to(z.device)[None, :, None]
        s = self.s.to(z.device)[None, :, None]
        az = torch.sqrt(z ** 2 + self.eps_abs)
        w_tail = torch.sigmoid((az - c) / s)
        return w_tail, 1.0 - w_tail

    # ------------------------------------------------------------------
    # forward:  [w_bulk*phi_bulk ; w_tail*phi_tail] averaged over time
    #           -> (B, 2J)   (first J = bulk, last J = tail)
    # ------------------------------------------------------------------
    def forward(self, x, *args):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).imag        # (B, J, T)
        w_tail, w_bulk = self._windows(z)
        a_b, b_b, t_b, x0_b = self._params_bulk(x.device)
        a_t, b_t, t_t, x0_t = self._params_tail(x.device)
        phi_bulk = (w_bulk * self._phi(z, a_b, b_b, t_b, x0_b)).mean(-1)   # (B, J)
        phi_tail = (w_tail * self._phi(z, a_t, b_t, t_t, x0_t)).mean(-1)   # (B, J)
        return torch.cat([phi_bulk, phi_tail], dim=1)                      # (B, 2J)

    # ------------------------------------------------------------------
    # grad:  d/dx of each windowed potential, back-projected through filters.
    #        Product rule:  d/dz [w(z) phi(z)] = w'(z) phi(z) + w(z) phi'(z),
    #        with  w_tail'(z) = w_tail(1-w_tail) * (z/|z|) / s,  w_bulk' = -w_tail'.
    #        Returns (B, 2J, T), or (B, 1, T) if v (length 2J) is given.
    # ------------------------------------------------------------------
    def grad(self, x, v=None, means=None):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).imag        # (B, J, T)
        a_b, b_b, t_b, x0_b = self._params_bulk(x.device)
        a_t, b_t, t_t, x0_t = self._params_tail(x.device)

        c = self.c.to(x.device)[None, :, None]
        s = self.s.to(x.device)[None, :, None]
        az = torch.sqrt(z ** 2 + self.eps_abs)
        w_tail = torch.sigmoid((az - c) / s)
        w_bulk = 1.0 - w_tail
        dw_tail = w_tail * (1.0 - w_tail) * (z / az) / s           # d w_tail / dz
        dw_bulk = -dw_tail

        phi_b = self._phi(z, a_b, b_b, t_b, x0_b); dphi_b = self._dphi(z, a_b, b_b, t_b, x0_b)
        phi_t = self._phi(z, a_t, b_t, t_t, x0_t); dphi_t = self._dphi(z, a_t, b_t, t_t, x0_t)
        D_bulk = dw_bulk * phi_b + w_bulk * dphi_b                 # (B, J, T)
        D_tail = dw_tail * phi_t + w_tail * dphi_t                 # (B, J, T)

        def backproj(D):
            return torch.fft.ifft(torch.fft.fft(D) * filters).imag / x.shape[-1]

        grad_coeff = torch.cat([backproj(D_bulk), backproj(D_tail)], dim=1)  # (B, 2J, T)

        if v is None:
            return grad_coeff
        return (grad_coeff * v[None, :, None]).sum(1)[:, None]     # (B, 1, T)

    # ------------------------------------------------------------------
    def summary(self):
        self._check_fitted()
        bb = self.b_bulk.cpu().numpy(); tb = self.t_bulk.cpu().numpy()
        ab = self.a_bulk.cpu().numpy(); xb = self.x0_bulk.cpu().numpy()
        bt = self.b_tail.cpu().numpy(); tt = self.t_tail.cpu().numpy()
        at = self.a_tail.cpu().numpy(); xt = self.x0_tail.cpu().numpy()
        c = self.c.cpu().numpy(); s = self.s.cpu().numpy()
        cos = self.cos_bt.cpu().numpy() if self.cos_bt is not None else np.zeros_like(c)
        print(f"{'Ch':>3} {'c':>9} {'s':>8} | "
              f"{'b_bk':>6} {'t_bk':>6} {'a_bk':>6} {'x0_bk':>8} | "
              f"{'b_tl':>6} {'t_tl':>6} {'a_tl':>6} {'x0_tl':>8} | {'cos':>5}")
        print("-" * 98)
        for j in range(len(bb)):
            print(f"{j:>3d} {c[j]:>9.4f} {s[j]:>8.4f} | "
                  f"{bb[j]:>6.3f} {tb[j]:>6.3f} {ab[j]:>6.3f} {xb[j]:>8.4f} | "
                  f"{bt[j]:>6.3f} {tt[j]:>6.3f} {at[j]:>6.3f} {xt[j]:>8.4f} | "
                  f"{cos[j]:>5.3f}")


from scipy.optimize import minimize_scalar
from scipy.integrate import IntegrationWarning
import warnings


class Scalar_coshgt_multiregion(Scalar_coshgt):
    """
    Generalization of Scalar_coshgt (bulk/tail, 2 windows) to an arbitrary
    number `n_regions` >= 2 of windowed coshGT regimes ("multi-region coshGT").

    Motivation
    ----------
    Scalar_coshgt fits a coshGT separately on the bulk and tail of |z|. In
    practice two regimes are sometimes not enough to track a histogram across
    its full range (e.g. core / shoulder / tail, or core / near-tail /
    far-tail / extreme-tail). This class partitions |z| into `n_regions`
    contiguous, smoothly-blended windows and fits ONE coshGT per window.
    Everything that doesn't depend on the number of regions (the (b,t,a,x0)
    parametrization, the weighted-MLE channel fit with GT-vs-tempered self
    selection, the cosine non-colinearity diagnostic, the stable torch
    building blocks) is inherited UNCHANGED from Scalar_coshgt.

    Toggle
    ------
    `n_regions=2` recovers exactly Scalar_coshgt's bulk/tail windowing.
    `n_regions=3` or `n_regions=4` are the requested 3-/4-area splits: pick
    one via this single constructor argument. `Scalar_coshgt_3region` and
    `Scalar_coshgt_4region` below are thin convenience subclasses that pin
    this argument, for call sites that want a named class instead of a
    kwarg toggle.

    Smooth partition of unity (telescoping sigmoids)
    --------------------------------------------------
    Let K = n_regions - 1 interior boundaries c_1 < c_2 < ... < c_K on |z|,
    each with its own transition width s_k. Define the "right of boundary k"
    gate
        r_k(z) = sigmoid((|z| - c_k) / s_k),     r_0 := 1,   r_{K+1} := 0 .
    Region i (i = 0 .. n_regions-1, innermost/bulk to outermost/tail) gets
    window
        w_i(z) = r_i(z) - r_{i+1}(z) .
    These telescope to sum_i w_i(z) = r_0 - r_{K+1} = 1 for every z (an exact
    partition of unity), and w_i >= 0 everywhere as long as the boundaries
    are increasing (each gate dominates the next, as in Scalar_coshgt's own
    c). For n_regions=2 this is exactly Scalar_coshgt's
    (w_bulk, w_tail) = (1 - w_tail, w_tail).

    Optimizing the boundaries
    --------------------------
    A boundary's *position* only enters the windows of its two neighbouring
    regions (telescoping cancels it out of every other window). So boundary
    k is optimized by a bounded 1-D line search (`scipy.optimize.
    minimize_scalar`, Brent on a bounded interval) that, at each trial
    position, refits ONLY the two adjacent regions (via the inherited,
    unmodified `Scalar_coshgt._fit_channel`) and scores the trial by their
    total weighted log-likelihood. Boundaries are swept left-to-right for
    `boundary_opt_rounds` coordinate-descent passes, each boundary optimized
    holding all others fixed -- i.e. block coordinate ascent on the joint
    weighted log-likelihood across all regions. Set
    `optimize_boundaries=False` to skip this and just keep the fixed
    `boundary_quantiles` (cheap fallback, in the spirit of Scalar_coshgt's
    fixed `bulk_quantile`).

    Stored per channel (tensors of shape (n_regions, J)):
        b, t, a, x0   -- one coshGT per region, stacked along dim 0.
    Stored per boundary (tensors of shape (n_regions-1, J)):
        c, s          -- boundary position and transition width.
    Diagnostic (n_regions-1, J): cos_adjacent -- cosine between each pair of
    *adjacent* regions' windowed potentials (same role/threshold as
    Scalar_coshgt.cos_bt, computed with the inherited `_potential_cosine`).
    """

    # default boundary-quantile seeds, skewed toward the tail like
    # Scalar_coshgt's own bulk_quantile=0.90 default; only used to *seed*
    # the search when optimize_boundaries=True, and used as-is otherwise.
    _DEFAULT_BOUNDARY_QUANTILES = {
        2: [0.990],
        3: [0.80, 0.95],
        4: [0.55, 0.80, 0.95],
    }

    def __init__(self, filters, n_regions=3, eps_abs=1e-6, eps_scale=1e-6,
                 a_max=5.0, b_bounds=(0.05, 10.0), t_bounds=(0.5, 50.0),
                 boundary_quantiles=None, trans_pad=0.04,
                 optimize_boundaries=False, boundary_opt_rounds=2,
                 boundary_search_bounds=(0.02, 0.98), min_boundary_gap=0.04,
                 min_eff_samples=50.0):
        if n_regions < 2:
            raise ValueError("n_regions must be >= 2 (Scalar_coshgt already "
                              "covers the 2-region case).")
        # NOTE: intentionally does NOT call Scalar_coshgt.__init__ -- that
        # allocates bulk/tail-specific attributes (b_bulk, t_bulk, ...) we
        # replace with stacked (n_regions, J) tensors below. Every method we
        # inherit (the _fit_channel/_logpdf_np/_phi/_dphi/... family) only
        # ever touches self.eps_abs/eps_scale/a_max/b_bounds/t_bounds, which
        # are set here identically to Scalar_coshgt.
        self.filters            = filters
        self.n_regions           = n_regions
        self.num_coefficients    = n_regions * filters.shape[1]
        self.eps_abs             = eps_abs
        self.eps_scale           = eps_scale
        self.a_max               = a_max
        self.b_bounds            = b_bounds
        self.t_bounds            = t_bounds
        self.min_eff_samples     = min_eff_samples
        self.trans_pad           = trans_pad
        self.optimize_boundaries = optimize_boundaries
        self.boundary_opt_rounds = boundary_opt_rounds
        self.boundary_search_bounds = boundary_search_bounds
        self.min_boundary_gap    = min_boundary_gap

        K = n_regions - 1
        if boundary_quantiles is None:
            boundary_quantiles = self._DEFAULT_BOUNDARY_QUANTILES.get(
                n_regions,
                list(np.linspace(0.0, 1.0, n_regions + 1)[1:-1]))
        if len(boundary_quantiles) != K:
            raise ValueError(f"boundary_quantiles must have length "
                              f"n_regions-1={K}, got {len(boundary_quantiles)}")
        self.boundary_quantiles_init = list(boundary_quantiles)

        # region params (n_regions, J) once fitted
        self.b = self.t = self.a = self.x0 = None
        # boundary params (n_regions-1, J) once fitted
        self.c = self.s = None
        self.cos_adjacent = None
        self.boundary_quantiles = None   # final per-channel (K, J), for inspection

    @property
    def is_fitted(self):
        return self.a is not None

    # ------------------------------------------------------------------
    # numpy helpers specific to the multi-region windowing.
    # (_logcosh_np, _log_u_np, _logZ_np, _logpdf_np, _weighted_median,
    #  _fit_channel, _potential_cosine are inherited verbatim from
    #  Scalar_coshgt.)
    # ------------------------------------------------------------------
    def _boundary_c_s(self, ah, q):
        """Boundary center c and transition width s from a quantile q of
        |z|, via the same robust-quantile-band heuristic Scalar_coshgt uses
        for its single bulk/tail border (band of +/-trans_pad quantiles,
        width = band/4)."""
        pad = self.trans_pad
        q_lo = max(q - pad, 1e-4)
        q_hi = min(q + pad, 1.0 - 1e-4)
        c = float(np.quantile(ah, q))
        band = float(np.quantile(ah, q_hi) - np.quantile(ah, q_lo))
        s = band / 4.0
        s = max(s, 0.05 * (c + self.eps_scale), self.eps_scale)
        return c, s

    @staticmethod
    def _telescoped_weights_np(ah, c_list, s_list):
        """Partition of unity on |z| into len(c_list)+1 regions via
        telescoping sigmoid gates (see class docstring)."""
        K = len(c_list)
        r = [np.ones_like(ah)]
        for k in range(K):
            r.append(1.0 / (1.0 + np.exp(-(ah - c_list[k]) / s_list[k])))
        r.append(np.zeros_like(ah))
        return [r[i] - r[i + 1] for i in range(K + 1)]

    def _boundary_neg_ll(self, q_i, i, h, ah, c_list, s_list, lam, lr_thresh):
        """Negative total weighted log-likelihood of the two regions
        adjacent to boundary i (regions i and i+1), as a function of
        boundary i's quantile position -- every other window is unaffected
        by q_i (telescoping), so only these two regions need refitting."""
        c_i, s_i = self._boundary_c_s(ah, q_i)
        c_trial = list(c_list); c_trial[i] = c_i
        s_trial = list(s_list); s_trial[i] = s_i
        weights = self._telescoped_weights_np(ah, c_trial, s_trial)
        ll = 0.0
        with warnings.catch_warnings():
            # the optimizer routinely probes (b,t,a,x0) with a non-convergent
            # tail integral while searching; _logZ_np already turns that into
            # a finite-penalty signal (np.inf -> guarded below), so the
            # IntegrationWarning itself is just noise here.
            warnings.simplefilter('ignore', category=IntegrationWarning)
            for r_idx in (i, i + 1):
                w = weights[r_idx]
                eff = (w.sum() ** 2) / max((w ** 2).sum(), 1e-12)
                if eff < 5.0:           # near-empty window at this trial position
                    return 1e8
                try:
                    b_, t_, a_, x0_ = self._fit_channel(
                        h, weights=w, lam=lam, lr_thresh=lr_thresh,
                        b_bounds=self.b_bounds, t_bounds=self.t_bounds,
                        a_max=self.a_max, eps_scale=self.eps_scale)
                    # guard against a degenerate fitted optimum whose own
                    # log-normalizer is non-finite -- without this, logpdf is
                    # an all -inf array and `0 * -inf = nan` poisons the sum
                    # at every index where w==0 (this was the actual bug:
                    # the nan then propagates through minimize_scalar).
                    if not np.isfinite(self._logZ_np(b_, t_, a_, x0_)):
                        return 1e8
                    ll += float((w * self._logpdf_np(h, b_, t_, a_, x0_)).sum())
                except Exception:
                    return 1e8
        if not np.isfinite(ll):
            return 1e8
        return -ll

    def _optimize_boundaries_channel(self, h, ah, lam, lr_thresh):
        """Block coordinate ascent on the boundary positions: sweep each
        boundary left-to-right with a bounded scalar line search, holding
        the others fixed, for `boundary_opt_rounds` passes."""
        K = self.n_regions - 1
        q = list(self.boundary_quantiles_init)
        c_list = [None] * K; s_list = [None] * K
        for k in range(K):
            c_list[k], s_list[k] = self._boundary_c_s(ah, q[k])

        if K == 0 or not self.optimize_boundaries:
            return q, c_list, s_list

        lo_bound, hi_bound = self.boundary_search_bounds
        gap = self.min_boundary_gap
        for _round in range(self.boundary_opt_rounds):
            for i in range(K):
                q_left  = q[i - 1] if i > 0 else lo_bound
                q_right = q[i + 1] if i < K - 1 else hi_bound
                b_lo = max(q_left + gap, lo_bound)
                b_hi = min(q_right - gap, hi_bound)
                if b_lo >= b_hi:
                    continue        # neighbours too close: keep current position
                res = minimize_scalar(
                    lambda qq: self._boundary_neg_ll(
                        qq, i, h, ah, c_list, s_list, lam, lr_thresh),
                    bounds=(b_lo, b_hi), method='bounded',
                    options=dict(xatol=1e-3))
                if np.isfinite(res.x):     # defensive: never move to a nan/inf position
                    q[i] = float(res.x)
                c_list[i], s_list[i] = self._boundary_c_s(ah, q[i])
        return q, c_list, s_list

    # ------------------------------------------------------------------
    # fit_reference: optimize boundaries, then fit all n_regions regimes
    # for every channel.
    # ------------------------------------------------------------------
    def fit_reference(self, x, lam=1.0, lr_thresh=2.0):
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real.detach().cpu().numpy()

        N, K = self.n_regions, self.n_regions - 1
        all_b  = [[] for _ in range(N)]
        all_t  = [[] for _ in range(N)]
        all_a  = [[] for _ in range(N)]
        all_x0 = [[] for _ in range(N)]
        all_c, all_s, all_q, all_cos = ([[] for _ in range(K)] for _ in range(4))

        for j in range(z.shape[1]):
            h = z[:, j, :].reshape(-1)
            h = h[np.isfinite(h)]
            ah = np.abs(h)

            if h.size < 10:
                # degenerate channel: mirror Scalar_coshgt's tiny-sample branch
                c_med = float(np.median(ah)) if ah.size else 1.0
                for i in range(N):
                    all_b[i] += [1.0]; all_t[i] += [max(4.0 - i, 1.0)]
                    all_a[i] += [0.0]; all_x0[i] += [c_med if c_med > 0 else 1.0]
                for k in range(K):
                    q_fb = self.boundary_quantiles_init[k]
                    all_c[k] += [c_med * (k + 1) / (K + 1)]
                    all_s[k] += [max(0.1 * c_med, self.eps_scale)]
                    all_q[k] += [q_fb]; all_cos[k] += [0.0]
                continue

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=IntegrationWarning)
                q, c_list, s_list = self._optimize_boundaries_channel(h, ah, lam, lr_thresh)

                # final fit of ALL regions at the converged boundaries
                weights = self._telescoped_weights_np(ah, c_list, s_list)
                region_params = []
                for i in range(N):
                    w = weights[i]
                    eff = (w.sum() ** 2) / max((w ** 2).sum(), 1e-12)
                    if eff < self.min_eff_samples:
                        print(f"[coshGT{N}][ch {j}] region {i} eff. N={eff:.1f} < "
                              f"{self.min_eff_samples:.0f}: fit may be weak "
                              f"(consider fewer regions, or widen boundary_search_bounds).")
                    try:
                        b_, t_, a_, x0_ = self._fit_channel(
                            h, weights=w, lam=lam, lr_thresh=lr_thresh,
                            b_bounds=self.b_bounds, t_bounds=self.t_bounds,
                            a_max=self.a_max, eps_scale=self.eps_scale)
                    except Exception as e:
                        print(f"[coshGT{N}][ch {j}] region {i} fit failed ({e}) -> fallback")
                        b_, t_, a_, x0_ = 1.0, max(6.0 - 2 * i, 1.0), 0.0, float(np.std(h) + self.eps_scale)
                    b_  = float(np.clip(b_, *self.b_bounds))
                    t_  = float(np.clip(t_, *self.t_bounds))
                    a_  = float(np.clip(a_, 0.0, self.a_max))
                    x0_ = float(max(x0_, self.eps_scale))
                    region_params.append((b_, t_, a_, x0_))
                    all_b[i] += [b_]; all_t[i] += [t_]; all_a[i] += [a_]; all_x0[i] += [x0_]

            for k in range(K):
                cos_k = self._potential_cosine(
                    c_list[k], s_list[k], region_params[k], region_params[k + 1])
                if cos_k > 0.98:
                    print(f"[coshGT{N}][ch {j}] WARNING: regions {k} & {k + 1} nearly "
                          f"colinear (cos={cos_k:.3f}) across boundary {k}; widen "
                          f"min_boundary_gap or boundary_search_bounds.")
                all_cos[k] += [cos_k]
                all_c[k]   += [c_list[k]]
                all_s[k]   += [s_list[k]]
                all_q[k]   += [q[k]]

            q_str = " ".join(f"{v:.3f}" for v in q)
            reg_str = " | ".join(
                f"r{i}(b={p[0]:.3f},t={p[1]:.3f},a={p[2]:.4f},x0={p[3]:.4f})"
                for i, p in enumerate(region_params))
            print(f"[coshGT{N}][ch {j}] q=[{q_str}] | {reg_str}")

        dtype = x.dtype if x.is_floating_point() else torch.float32
        dev = x.device
        mk = lambda L: torch.tensor(L, dtype=dtype, device=dev)
        self.b  = torch.stack([mk(all_b[i])  for i in range(N)], dim=0)
        self.t  = torch.stack([mk(all_t[i])  for i in range(N)], dim=0)
        self.a  = torch.stack([mk(all_a[i])  for i in range(N)], dim=0)
        self.x0 = torch.stack([mk(all_x0[i]) for i in range(N)], dim=0)
        if K > 0:
            self.c = torch.stack([mk(all_c[k]) for k in range(K)], dim=0)
            self.s = torch.stack([mk(all_s[k]) for k in range(K)], dim=0)
            self.cos_adjacent = torch.stack([mk(all_cos[k]) for k in range(K)], dim=0)
            self.boundary_quantiles = np.stack(
                [np.array(all_q[k]) for k in range(K)], axis=0)
        else:
            self.c = self.s = self.cos_adjacent = None
            self.boundary_quantiles = None

    def fit(self, x, **kwargs):
        return self.fit_reference(x, **kwargs)

    # ------------------------------------------------------------------
    # torch building blocks
    # (_logcosh, _phi, _dphi are inherited verbatim from Scalar_coshgt.)
    # ------------------------------------------------------------------
    def _params_region(self, i, device):
        return (self.a[i].to(device)[None, :, None], self.b[i].to(device)[None, :, None],
                self.t[i].to(device)[None, :, None], self.x0[i].to(device)[None, :, None])

    def _windows(self, z):
        """List of n_regions windows via telescoping sigmoid gates on |z|."""
        az = torch.sqrt(z ** 2 + self.eps_abs)
        K = self.n_regions - 1
        r = [torch.ones_like(az)]
        if K > 0:
            c = self.c.to(z.device); s = self.s.to(z.device)
            for k in range(K):
                ck = c[k][None, :, None]; sk = s[k][None, :, None]
                r.append(torch.sigmoid((az - ck) / sk))
        r.append(torch.zeros_like(az))
        return [r[i] - r[i + 1] for i in range(self.n_regions)]

    # ------------------------------------------------------------------
    # forward: [w_0*phi_0 ; ... ; w_{N-1}*phi_{N-1}] averaged over time
    #          -> (B, N*J)   (region 0 = innermost/bulk .. N-1 = outermost/tail)
    # ------------------------------------------------------------------
    def forward(self, x, *args):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real        # (B, J, T)
        windows = self._windows(z)
        phis = []
        for i in range(self.n_regions):
            a_i, b_i, t_i, x0_i = self._params_region(i, x.device)
            phis.append((windows[i] * self._phi(z, a_i, b_i, t_i, x0_i)).mean(-1))
        return torch.cat(phis, dim=1)                                # (B, N*J)

    # ------------------------------------------------------------------
    # grad: d/dx of each windowed potential, back-projected through filters.
    #       Product rule:  d/dz [w_i(z) phi_i(z)] = w_i'(z) phi_i(z) + w_i(z) phi_i'(z),
    #       with w_i = r_i - r_{i+1} and dr_k/dz = r_k(1-r_k)*(z/|z|)/s_k.
    #       Returns (B, N*J, T), or (B, 1, T) if v (length N*J) is given.
    # ------------------------------------------------------------------
    def grad(self, x, v=None, means=None):
        self._check_fitted()
        filters = self.filters.to(x.device)
        z = torch.fft.ifft(filters * torch.fft.fft(x)).real        # (B, J, T)
        az = torch.sqrt(z ** 2 + self.eps_abs)
        K = self.n_regions - 1

        r = [torch.ones_like(az)]; dr = [torch.zeros_like(az)]
        if K > 0:
            c = self.c.to(x.device); s = self.s.to(x.device)
            for k in range(K):
                ck = c[k][None, :, None]; sk = s[k][None, :, None]
                rk = torch.sigmoid((az - ck) / sk)
                drk = rk * (1.0 - rk) * (z / az) / sk
                r.append(rk); dr.append(drk)
        r.append(torch.zeros_like(az)); dr.append(torch.zeros_like(az))

        windows  = [r[i] - r[i + 1] for i in range(self.n_regions)]
        dwindows = [dr[i] - dr[i + 1] for i in range(self.n_regions)]

        Ds = []
        for i in range(self.n_regions):
            a_i, b_i, t_i, x0_i = self._params_region(i, x.device)
            phi_i  = self._phi(z, a_i, b_i, t_i, x0_i)
            dphi_i = self._dphi(z, a_i, b_i, t_i, x0_i)
            Ds.append(dwindows[i] * phi_i + windows[i] * dphi_i)        # (B, J, T)

        def backproj(D):
            return torch.fft.ifft(torch.fft.fft(D) * filters).real / x.shape[-1]

        grad_coeff = torch.cat([backproj(D) for D in Ds], dim=1)       # (B, N*J, T)

        if v is None:
            return grad_coeff
        return (grad_coeff * v[None, :, None]).sum(1)[:, None]         # (B, 1, T)

    # ------------------------------------------------------------------
    def summary(self):
        self._check_fitted()
        N = self.n_regions
        b = self.b.cpu().numpy(); t = self.t.cpu().numpy()
        a = self.a.cpu().numpy(); x0 = self.x0.cpu().numpy()
        c = self.c.cpu().numpy() if self.c is not None else None
        s = self.s.cpu().numpy() if self.s is not None else None
        cos = self.cos_adjacent.cpu().numpy() if self.cos_adjacent is not None else None
        for j in range(b.shape[1]):
            bnd = (" ".join(f"c{k}={c[k, j]:>8.4f}(s={s[k, j]:>7.4f})" for k in range(N - 1))
                   if c is not None else "")
            reg = " | ".join(
                f"r{i}(b={b[i, j]:>6.3f},t={t[i, j]:>6.3f},a={a[i, j]:>6.3f},x0={x0[i, j]:>8.4f})"
                for i in range(N))
            cs = " ".join(f"cos{k}={cos[k, j]:>5.3f}" for k in range(N - 1)) if cos is not None else ""
            print(f"[ch {j:>3d}] {bnd} | {reg} | {cs}")


class Scalar_coshgt_3region(Scalar_coshgt_multiregion):
    """Scalar_coshgt_multiregion pinned to n_regions=3 (core / shoulder / tail).
    Convenience subclass for call sites that prefer a named class over the
    n_regions kwarg toggle; identical behaviour to
    Scalar_coshgt_multiregion(filters, n_regions=3, ...)."""
    def __init__(self, filters, **kwargs):
        kwargs.pop('n_regions', None)
        super().__init__(filters, n_regions=3, **kwargs)


class Scalar_coshgt_4region(Scalar_coshgt_multiregion):
    """Scalar_coshgt_multiregion pinned to n_regions=4
    (core / near-tail / far-tail / extreme-tail). Convenience subclass;
    identical behaviour to
    Scalar_coshgt_multiregion(filters, n_regions=4, ...)."""
    def __init__(self, filters, **kwargs):
        kwargs.pop('n_regions', None)
        super().__init__(filters, n_regions=4, **kwargs)


