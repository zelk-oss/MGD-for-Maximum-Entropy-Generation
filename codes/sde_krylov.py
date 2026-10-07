"""
SDEKrylov: SDE (codes/sde_routines.py) with the per-step eta / theta solves optionally
replaced by early-stopped Krylov iterations (codes/krylov.py). Test bed: krylov_stopping/.
sde_routines.py is NOT modified; with both solvers 'ridge' and the other options at their
defaults this class calls the parent's compute_eta / compute_theta / compute_rhs_dt_phi_I_t
unchanged (checked bit-identical by krylov_stopping/tests/test_sde_krylov.py).

Sign / units (as in SDE): the corrector solves G(y_k) theta_raw = rhs_k with
rhs_k = phibar(I_{k+1}) - phibar(y_k) = -rho_k in MOMENT units (no 1/(h sigma^2)), and
theta_code = theta_raw / (h sigma^2) = -theta_MGD. The residual G theta_raw - rhs is the
moment-unit residual; signs do not matter for any norm below.

theta_solver:
  'ridge'      parent's solve: Jacobi scaling + regularization * I (0 = direct solve).
  'ito_cg'     CG on the Jacobi-scaled system, stop at N/(2 h s^2) res^T G^-1 res <= chi2_{r,1-alpha}.
  'tol_minres' MINRES whitened by Sigma_tol (negative result, krylov_stopping/RECAP §5).
  'jac_minres' (2026-10-07) A = D^-1/2 G D^-1/2 (D = diag G, live set), c = D^-1/2 rhs,
               kappa = 2 h sigma^2 / N, eps2 = kappa (r + 2 sqrt2 ||A||_F) (finite-N error of one
               predictor step: covariance kappa A, mean kappa r, std kappa sqrt2 ||A||_F).
               ||c||^2 <= eps2 -> theta = 0; else MINRES on A z = c from 0, stop at the first j
               with ||c - A z_j||^2 <= eps2; theta_raw = D^-1/2 z.
eta_solver: 'ridge' (parent) or 'tol_minres'. eta_regularization: if set, the ridge eta solve uses
  this value instead of `regularization` (the theta ridge), e.g. 1e-7 for every method.
g64: accumulate G in float64 (parent: float32 accumulation; needs solve_float64=True).

Monitoring (monitor=True; written to monitor_path every flush_every steps, atomic replace):
  every step  : k, R_k = theta_raw^T G' theta_raw / (2 h sigma^2 d) (corrector displacement over
                Brownian kick), j, ||c||^2/eps2 at j=0, final stat/eps2, max_iter hit, max|y|,
                finite flag;
  every eig_every steps : spectrum of A (float64): mu_min, kappa(A), #mu < 1e-6, the 10 smallest
                weak-direction ratios (u_i^T c)^2 / (kappa mu_i), and mu_min of the Jacobi-scaled
                G accumulated in float32 (only if g64);
  every mis_every steps : whitened mismatch after the corrector (diag Sigma_tol =
                Var_s phi(I_t)/M + Var_i phi(x)/N), measured at the start of the next step.
abort=True raises on: non-finite theta / walkers; R_k > 100 x median(R over the first 100
steps); weak-direction ratios growing for > 500 steps (max ratio, 5 consecutive blocks of 100
steps with increasing means and last/first > 10).
"""
import math
import os
import time

import torch

from codes.sde_routines import SDE
from codes import krylov as K

THETA_SOLVERS = ('ridge', 'ito_cg', 'tol_minres', 'jac_minres')
ETA_SOLVERS = ('ridge', 'tol_minres')
THETA_COLS = ['k', 'r_live', 'j', 'stat', 'thr', 'ito_stat0', 'ito_thr', 'tol_stat0', 'tol_thr', 't_call']
ETA_COLS = ['k', 'r_live', 'j', 'stat', 'thr', 't_call']
MON_COLS = ['k', 'R', 'j', 'c2_over_eps2', 'final_over_eps2', 'hit_max', 'max_abs_y', 'finite']
NAN = float('nan')


class SDEKrylov(SDE):

    def __init__(self, *args, theta_solver='ridge', eta_solver='ridge', tol_weight='full',
                 alpha=0.05, kappa=2.0, var_floor=1e-6, M_eff=None, max_iter=None,
                 diagnostics=True, keep_step_mats=False, eta_regularization=None, g64=False,
                 monitor=False, monitor_path=None, eig_every=10, mis_every=50, flush_every=200,
                 abort=False, **kwargs):
        super().__init__(*args, **kwargs)
        if theta_solver not in THETA_SOLVERS:
            raise ValueError(f'theta_solver must be one of {THETA_SOLVERS}, got {theta_solver!r}')
        if eta_solver not in ETA_SOLVERS:
            raise ValueError(f'eta_solver must be one of {ETA_SOLVERS}, got {eta_solver!r}')
        if tol_weight not in ('diag', 'full', 'shrink'):
            raise ValueError(f"tol_weight must be 'diag', 'full' or 'shrink', got {tol_weight!r}")
        if theta_solver in ('ito_cg', 'jac_minres') and not self.sigma > 0:
            raise ValueError(f"'{theta_solver}' needs sigma > 0 (its error level is 2 h sigma^2 / N)")
        if g64 and not self.solve_float64:
            raise ValueError('g64=True needs solve_float64=True (the parent solve mixes dtypes otherwise)')
        self.theta_solver, self.eta_solver, self.tol_weight = theta_solver, eta_solver, tol_weight
        self.alpha, self.kappa, self.var_floor, self.max_iter = alpha, kappa, var_floor, max_iter
        self.M_eff = M_eff if M_eff is not None else int(self.x_0.shape[0])
        self.diagnostics, self.keep_step_mats = diagnostics, keep_step_mats
        self.eta_regularization, self.g64 = eta_regularization, g64
        self.monitor, self.monitor_path, self.abort = monitor or abort or monitor_path is not None, monitor_path, abort
        self.eig_every, self.mis_every, self.flush_every = eig_every, mis_every, flush_every
        self._klog = {'eta': [], 'theta': []}
        self._khist = {'eta': [], 'theta': []}
        self.step_mats = {'G_theta': [], 'Sigma_tol': []}
        self._mdot_samples = None
        self._mon = {'step': [], 'eig': [], 'mis': []}
        self._jac = None                                  # (j, c2/eps2, final/eps2, hit) of this step

    # ------------------------------------------------------------------ helpers
    def _now(self):
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize()
        return time.perf_counter()

    def _dtype(self, *X):
        return [x.double() for x in X] if self.solve_float64 else list(X)

    def _sigma_tol(self, live, mode):
        parts = [(self._mom_I[:, live], self.M_eff), (self._mom_x[:, live], self._mom_x.shape[0])]
        return K.tol_sigma(parts, mode)

    def _h(self, k):
        return float(self.t[k + 1] - self.t[k])

    def krylov_log(self):
        out = {'theta_cols': THETA_COLS, 'eta_cols': ETA_COLS,
               'theta_solver': self.theta_solver, 'eta_solver': self.eta_solver,
               'tol_weight': self.tol_weight, 'alpha': self.alpha, 'kappa': self.kappa,
               'var_floor': self.var_floor, 'M_eff': self.M_eff}
        for name, cols in (('theta', THETA_COLS), ('eta', ETA_COLS)):
            rows = self._klog[name]
            out[name] = torch.tensor(rows, dtype=torch.float64) if rows else torch.zeros(0, len(cols))
            out[f'{name}_hist'] = self._khist[name]
        if self.keep_step_mats:
            out['G_theta'] = torch.stack(self.step_mats['G_theta'])
            out['Sigma_tol'] = torch.stack(self.step_mats['Sigma_tol'])
        return out

    def monitor_log(self):
        """Monitoring as tensors: step (MON_COLS), eig rows [k, mu_min, cond, n_small, mu_min_f32,
        10 weak ratios], mis rows [k, whitened stat, r_live, max |z|]."""
        t = lambda rows: torch.tensor(rows, dtype=torch.float64) if rows else torch.zeros(0)
        return {'step_cols': MON_COLS, 'step': t(self._mon['step']),
                'eig_cols': ['k', 'mu_min', 'cond', 'n_mu_below_1e-6', 'mu_min_f32'] + [f'weak{i}' for i in range(10)],
                'eig': t(self._mon['eig']), 'mis_cols': ['k', 'whitened_mismatch', 'r_live', 'max_abs_z'],
                'mis': t(self._mon['mis']), 'theta_solver': self.theta_solver, 'sigma': float(self.sigma),
                'n_walkers': int(self.x_k.shape[0]), 'd': int(math.prod(self.x_k.shape[1:])),
                'regularization': float(self.regularization), 'eta_regularization': self.eta_regularization,
                'g64': self.g64}

    def _flush(self, force=False):
        if self.monitor_path is None or not (force or len(self._mon['step']) % self.flush_every == 0):
            return
        try:
            torch.save(self.monitor_log(), str(self.monitor_path) + '.tmp')
            os.replace(str(self.monitor_path) + '.tmp', str(self.monitor_path))
        except OSError as e:                                   # monitoring must never stop the run
            print(f'[monitor] save to {self.monitor_path} failed ({e}); continuing')

    def _fail(self, msg):
        self._flush(force=True)
        raise RuntimeError(f'[monitor abort] {msg}')

    def forward_regularised(self, *args, **kwargs):
        # Theta_reg only reads the stored M_k = G(y_k), b_k and the walker moments, which every theta
        # solver provides (2026-10-07: guard lifted). Early stopping only changes the trajectory.
        out = super().forward_regularised(*args, **kwargs)
        self._flush(force=True)
        return out

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        self._flush(force=True)
        return out

    # ------------------------------------------------------------------ Gram in float64
    def compute_G(self, x):
        if not self.g64:
            return super().compute_G(x)
        bs, n = self.batch_size, x.shape[0]
        G = torch.zeros((self.num_potentials, self.num_potentials), dtype=torch.float64, device=self.device)
        for a in range(0, n, bs):
            g = self.compute_grad_potentials(x[a:a + bs]).double()
            g = g.reshape(g.shape[0], self.num_potentials, -1)
            G += torch.bmm(g, g.transpose(1, 2)).sum(0)
        return G / n

    # ------------------------------------------------------------------ moment velocity
    def compute_rhs_dt_phi_I_t(self, I_k, k):
        if self.eta_solver == 'ridge':
            return super().compute_rhs_dt_phi_I_t(I_k, k)
        I_dot, bs, out = self.gradient_interpolant(k), self.batch_size, []
        for a in range(0, I_k.shape[0], bs):
            g = self.compute_grad_potentials(I_k[a:a + bs])
            B = g.shape[0]
            out.append((g.reshape(B, self.num_potentials, -1) * I_dot[a:a + bs].reshape(B, 1, -1)).sum(-1))
        S = torch.cat(out)
        self._mdot_samples = (k, S)
        return S.mean(0)[:, None]

    # ------------------------------------------------------------------ predictor
    def compute_eta(self, x_k, k):
        if self.monitor and k > 0 and k % self.mis_every == 0 and getattr(self, '_mon_mom_I', None) is not None:
            self._mismatch(x_k, k - 1)
        t0 = self._now()
        if self.eta_solver == 'ridge':
            if self.eta_regularization is None:
                eta = super().compute_eta(x_k, k)
            else:                                          # own eta ridge (e.g. 1e-7 for every method)
                lam_theta, self.regularization = self.regularization, self.eta_regularization
                try:
                    eta = super().compute_eta(x_k, k)
                finally:
                    self.regularization = lam_theta
            self._klog['eta'].append([k, NAN, -1, NAN, NAN, self._now() - t0])
            return eta.to(x_k.dtype)

        I_k = self.compute_interpolant(k)
        cache = getattr(self, '_rhs_dt_cache', None)
        if (cache is not None and cache[0] == k and self._mdot_samples is not None
                and self._mdot_samples[0] == k):
            mdot = cache[1]
        else:
            mdot = self.compute_rhs_dt_phi_I_t(I_k, k)
        G_k = self.compute_G(x_k)
        live = self._live_potentials(G_k, mdot, k, 'eta')
        Gl, bl = self._dtype(G_k[live][:, live], mdot.reshape(-1)[live])
        Gl = (Gl + Gl.T) / 2
        D12 = Gl.diagonal().sqrt()
        self._log_cond('eta', k, Gl / (D12[:, None] * D12[None, :]), D12, live, bl / D12)
        Sigma = K.tol_sigma([(self._mdot_samples[1][:, live], self.M_eff)], self.tol_weight)
        Wih, eps2 = K.tol_weight(Sigma, self.tol_weight, self.var_floor, self.alpha, self.kappa)
        Wih = Wih.to(Gl.dtype)
        Gt, bt = K.whiten(Wih, Gl, bl)
        n = int(live.sum())
        xt, j, hist = K.minres(lambda v: Gt @ v, bt, K.residual_stop(eps2), self.max_iter or n)
        eta = torch.zeros(G_k.shape[0], dtype=x_k.dtype, device=G_k.device)
        eta[live] = K.unwhiten(Wih, xt).to(x_k.dtype)
        self._klog['eta'].append([k, n, j, hist[-1], eps2, self._now() - t0])
        self._khist['eta'].append(torch.tensor(hist))
        return eta

    # ------------------------------------------------------------------ corrector
    def compute_theta(self, y_k, k):
        t0 = self._now()
        h = self._h(k)
        self._jac = None
        if self.theta_solver == 'ridge':
            theta, rhs = super().compute_theta(y_k, k)
            row = [k, NAN, -1, NAN, NAN]
        else:
            theta, rhs, row = self._theta_krylov(y_k, k, h)
        theta = theta.to(y_k.dtype)
        t_call = self._now() - t0
        self._klog['theta'].append(row + self._theta_diagnostics(rhs, h) + [t_call])
        if self.monitor:
            # own reference: forward_regularised sets self._mom_I = None after each step (fixed 10-07,
            # the d = 64 runs of job 742814 have empty mismatch rows because of it)
            self._mon_mom_I = self._mom_I
            self._monitor_step(y_k, k, h, theta, rhs)
        return theta, rhs

    def _theta_krylov(self, y_k, k, h):
        I_k = self.compute_interpolant(k + 1)
        rhs = self.compute_rhs_constraint_correction(y_k, I_k)
        G_k = self.compute_G(y_k)
        self._G_theta = G_k
        live = self._live_potentials(G_k, rhs, k, 'theta')
        self._live_theta = live

        Gl, bl = self._dtype(G_k[live][:, live], rhs[live])
        Gl = (Gl + Gl.T) / 2
        D12 = Gl.diagonal().sqrt()
        As, bs = Gl / (D12[:, None] * D12[None, :]), bl / D12
        self._log_cond('theta', k, As, D12, live, bs, noise=(self._mom_I, self._mom_x))
        n = int(live.sum())
        max_iter = self.max_iter or n
        if self.theta_solver == 'jac_minres':
            kap = 2 * h * self.sigma ** 2 / y_k.shape[0]
            thr = kap * (n + 2 * math.sqrt(2) * float(As.norm()))
            z, j, hist = K.minres(lambda v: As @ v, bs, K.residual_stop(thr), max_iter)
            th = z / D12
            self._jac = (j, hist[0] / thr, hist[-1] / thr, int(j == max_iter and hist[-1] > thr))
        elif self.theta_solver == 'ito_cg':
            thr = K.chi2_threshold(n, self.alpha)
            scale = y_k.shape[0] / (2 * h * self.sigma ** 2)
            z, j, hist = K.cg(lambda v: As @ v, bs, K.ito_stop(As, scale, thr), max_iter)
            th = z / D12
        else:
            Sigma = self._sigma_tol(live, self.tol_weight)
            Wih, thr = K.tol_weight(Sigma, self.tol_weight, self.var_floor, self.alpha, self.kappa)
            Wih = Wih.to(Gl.dtype)
            Gt, bt = K.whiten(Wih, Gl, bl)
            xt, j, hist = K.minres(lambda v: Gt @ v, bt, K.residual_stop(thr), max_iter)
            th = K.unwhiten(Wih, xt)
        theta = torch.zeros(G_k.shape[0], dtype=y_k.dtype, device=G_k.device)
        theta[live] = th.to(y_k.dtype)
        self._khist['theta'].append(torch.tensor(hist))
        return theta, rhs, [k, n, j, hist[-1], thr]

    def _theta_diagnostics(self, rhs, h):
        if not self.diagnostics:
            return [NAN] * 4
        live = self._live_theta
        n = int(live.sum())
        Gl = self._G_theta[live][:, live].double()
        Gl = (Gl + Gl.T) / 2
        bl = rhs[live].double()
        D12 = Gl.diagonal().sqrt()
        ito0 = NAN
        if self.sigma > 0:
            L, info = torch.linalg.cholesky_ex(Gl / (D12[:, None] * D12[None, :]))
            if int(info) == 0:
                z = torch.linalg.solve_triangular(L, (bl / D12)[:, None], upper=False)
                ito0 = self._mom_x.shape[0] / (2 * h * self.sigma ** 2) * float((z * z).sum())
        Sigma = self._sigma_tol(live, self.tol_weight)
        Wih, tol_thr = K.tol_weight(Sigma, self.tol_weight, self.var_floor, self.alpha, self.kappa)
        tol0 = float(K.whiten(Wih, Gl, bl)[1].pow(2).sum())
        if self.keep_step_mats:
            self.step_mats['G_theta'].append(self._G_theta.detach().double().cpu())
            full = self._sigma_tol(torch.ones_like(live), 'full')
            self.step_mats['Sigma_tol'].append(full.detach().cpu())
        return [ito0, K.chi2_threshold(n, self.alpha), tol0, tol_thr]

    # ------------------------------------------------------------------ monitoring
    def _monitor_step(self, y_k, k, h, theta_raw, rhs):
        live = self._live_theta
        G = self._G_theta.double()
        th = theta_raw.double()
        d = math.prod(y_k.shape[1:])
        R = float(th @ G @ th) / (2 * h * self.sigma ** 2 * d)
        finite = bool(torch.isfinite(theta_raw).all()) and bool(torch.isfinite(y_k).all())
        jac = self._jac or (NAN, NAN, NAN, NAN)
        self._mon['step'].append([k, R, jac[0], jac[1], jac[2], jac[3], float(y_k.abs().max()), float(finite)])
        if k % self.eig_every == 0:
            Gl = G[live][:, live]
            Gl = (Gl + Gl.T) / 2
            D12 = Gl.diagonal().sqrt()
            A = Gl / (D12[:, None] * D12[None, :])
            mu, U = torch.linalg.eigh(A)
            c = rhs[live].double() / D12
            kap = 2 * h * self.sigma ** 2 / y_k.shape[0]
            m = min(10, len(mu))
            weak = ((U[:, :m].T @ c) ** 2 / (kap * mu[:m].clamp_min(1e-300))).tolist() + [NAN] * (10 - m)
            mu32 = NAN
            if self.g64:                                   # same Gram, float32 accumulation
                G32 = SDE.compute_G(self, y_k).double()[live][:, live]
                d32 = G32.diagonal().clamp_min(1e-300).sqrt()
                mu32 = float(torch.linalg.eigvalsh((G32 + G32.T) / 2 / (d32[:, None] * d32[None, :]))[0])
            self._mon['eig'].append([k, float(mu[0]), float(mu[-1] / mu[0]) if mu[0] > 0 else float('inf'),
                                     int((mu < 1e-6).sum()), mu32] + weak)
        self._flush()
        if self.abort:
            self._check_abort(k)

    def _mismatch(self, x_k, k_prev):
        """Moment mismatch after the corrector of step k_prev: walkers x_k vs target at t_{k_prev+1},
        whitened by diag Sigma_tol (target and ensemble sampling variances)."""
        mom_x = self.compute_moments(x_k).double()
        mom_I = self._mon_mom_I.double()
        live = self._live_theta if getattr(self, '_live_theta', None) is not None else \
            torch.ones(mom_x.shape[1], dtype=torch.bool, device=mom_x.device)
        res = (mom_x.mean(0) - mom_I.mean(0))[live]
        var = (mom_I.var(0) / mom_I.shape[0] + mom_x.var(0) / mom_x.shape[0])[live].clamp_min(1e-300)
        z = res / var.sqrt()
        self._mon['mis'].append([k_prev, float((z ** 2).sum()), int(live.sum()), float(z.abs().max())])

    def _check_abort(self, k):
        s = self._mon['step'][-1]
        if not s[7]:
            self._fail(f'non-finite theta or walkers at step {k}')
        steps = self._mon['step']
        if len(steps) > 100:
            base = torch.tensor([r[1] for r in steps[:100]]).median().item()
            if base > 0 and s[1] > 100 * base:
                self._fail(f'R_k = {s[1]:.3e} at step {k} > 100 x median of the first 100 steps ({base:.3e})')
        eig = self._mon['eig']
        nb = 100 // self.eig_every                          # eig rows per 100 steps
        if len(eig) >= 5 * nb:
            w = torch.tensor([max(r[5:]) if any(v == v for v in r[5:]) else float('nan') for r in eig[-5 * nb:]])
            blocks = w.reshape(5, nb).mean(1)
            if bool((blocks[1:] > blocks[:-1]).all()) and blocks[-1] > 10 * blocks[0]:
                self._fail(f'weak-direction ratio grew for 500 steps up to step {k} '
                           f'(block means {[f"{b:.2e}" for b in blocks.tolist()]})')
