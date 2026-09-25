"""Time grids for the MGD SDE on t in [0, 1].

Three schedules, chosen per run with ``--schedule`` in run_SDE.py:

``power`` (default, legacy -- unchanged so old runs reproduce)
    t = 1 - (1 - s)^p on a uniform s grid of nt + 1 points, built in float32; the
    caller then drops the trailing points that round to 1.0000 (4 decimals), so the
    run ends at 1 - t ~ 5e-5. Problem found on 2026-09-25: near t = 1 the step h
    shrinks more slowly than the time left 1 - t, so h / (1 - t) grows towards the
    end (sched3: ~3.5e-4 at 1 - t = 1e-2, ~2e-3 at the end), and the runs jump
    (moment error -> 0.3, noisy samples) once it passes ~1.5e-3. Float32 also
    quantises the steps near t = 1 (up to ~30-50% error for sched3 below 1e-4).

``two_phase`` (fixed grid, float64) -- see :func:`two_phase_schedule`
    Phase 1, t in [0, t_switch]: n_bulk uniform steps. Early transport is cheap to
    get right (moment error 1e-4..1e-3 there), so few, large steps.
    Phase 2, t in [t_switch, 1 - gap_end]: every step removes the SAME FRACTION r of
    the time left, 1 - t_{k+1} = (1 - r)(1 - t_k), i.e. h / (1 - t) = r constant.
    r is not an input: it is whatever fits the remaining nt - n_bulk steps between
    1 - t_switch and gap_end, so --nt keeps its meaning (total number of steps, and
    runtime is ~ nt x s/step). Example: nt = 40000, n_bulk = 10000, t_switch = 0.9,
    gap_end = 5e-5  ->  r = 2.5e-4, ~8x below sched3's ratio at the end.

``adaptive`` (grid built during the run, float64) -- see :class:`AdaptiveStepController`
    The step size is chosen from the moment error of the PREVIOUS step, before the
    next step's noise is drawn: no step is ever rejected or redone, so the Brownian
    increments are not selected on (rejecting noisy steps and redrawing would bias
    the SDE). h grows (at most x grow per step) while the error is below tol and
    shrinks when it is above, inside guard rails tied to the time left:
        ratio_min * (1 - t) <= h <= min(h_max, ratio_max * (1 - t)).
    --nt is the CAP on the number of steps. The run ends at 1 - t = gap_end.

All grids stop at t_end = 1 - gap_end: at the default 5e-5 the Cos interpolant's
leftover noise weight is cos(pi t / 2) ~ 7.9e-5, i.e. ~1% of the data spectrum at
the Nyquist frequency (turbulence, M = 256), so the points past it add nothing.
"""

import math

import numpy as np
import torch


# ================================================================================
# Fixed grids
# ================================================================================

def power_schedule(nt, exponent):
    """Legacy grid t = 1 - (1 - linspace(0, 1, nt + 1))^exponent, float32.

    Kept bit-identical to the pre-2026-09-25 code; the caller applies the
    round-to-4-decimals cut (see run_SDE.py).
    """
    return 1 - (1 - torch.linspace(0, 1, nt + 1)) ** exponent


def two_phase_schedule(nt, n_bulk, t_switch=0.9, gap_end=5e-5):
    """Uniform bulk + constant-ratio tail, float64. Returns (t, info).

    Parameters
    ----------
    nt : int
        Total number of steps (len(t) - 1).
    n_bulk : int
        Uniform steps on [0, t_switch]; the remaining nt - n_bulk steps form the tail.
    t_switch : float
        Where the tail starts.
    gap_end : float
        Last grid point is exactly t = 1 - gap_end.

    Returns
    -------
    t : torch.Tensor, float64, shape (nt + 1,)
        Strictly increasing, t[0] = 0, t[-1] = 1 - gap_end.
    info : dict
        h_bulk, tail ratio r = h / (1 - t), and the first/last tail steps (for the log).
    """
    n_tail = nt - n_bulk
    if n_bulk < 1 or n_tail < 1:
        raise ValueError(f'need 1 <= n_bulk < nt, got n_bulk={n_bulk}, nt={nt}')
    if not 0 < gap_end < 1 - t_switch < 1:
        raise ValueError(f'need 0 < gap_end < 1 - t_switch, got {gap_end=}, {t_switch=}')

    bulk = torch.linspace(0, t_switch, n_bulk + 1, dtype=torch.float64)
    # gaps g_j = (1 - t_switch) (1 - r)^j, j = 0..n_tail, with g_{n_tail} = gap_end exactly
    q = (gap_end / (1 - t_switch)) ** (1.0 / n_tail)                 # = 1 - r
    j = torch.arange(1, n_tail + 1, dtype=torch.float64)
    gaps = (1 - t_switch) * q ** j
    gaps[-1] = gap_end                                               # remove rounding drift
    t = torch.cat([bulk, 1 - gaps])

    h = torch.diff(t)
    info = dict(h_bulk=t_switch / n_bulk, tail_ratio=1 - q,
                h_tail_first=float(h[n_bulk]), h_tail_last=float(h[-1]))
    return t, info


# ================================================================================
# Adaptive step size
# ================================================================================

class AdaptiveStepController:
    """Chooses h_{k+1} from the moment error after step k (no rejections).

    Error measure (after the corrector, at t_{k+1}):
        err_k = mean_i |e_i - p_i| / s_i,    s_i = max_{j <= k} |e_i(t_j)|
    e = interpolant (target) moments, p = walker moments, s_i the largest target
    magnitude of moment i seen so far. Normalising by s_i instead of |e_i| keeps
    moments that cross zero from dominating (the plain relative error blows up
    there). Moments with s_i <= scale_floor are ignored.

    Update (first order, err ~ h assumed):
        h_{k+1} = h_k * clip(safety * tol / err_k, shrink, grow)
    then clamped to [ratio_min * (1 - t), min(h_max, ratio_max * (1 - t))] and to
    not step past t_end.

    Guard rails and why:
      ratio_max  -- the explicit step went unstable at h / (1 - t) ~ 1.5e-3 in the
                    2026-09-25 runs; default 1e-3 keeps below it whatever err says.
      ratio_min  -- if the error stops responding to h (e.g. a ridge bias the
                    corrector cannot remove), the controller would otherwise shrink
                    h forever; this floor bounds the step count (a pure-ratio tail
                    from 1e-2 to 5e-5 at 1e-5 would take ~5e5 steps -- the nt cap
                    stops it first). n_at_floor counts steps pinned at the floor.
      grow       -- the error reacts one step late (h is chosen before the noise
                    of the next step), so growth is limited per step.
    """

    def __init__(self, tol=1e-3, h0=1e-5, h_max=1e-3, ratio_max=1e-3, ratio_min=1e-5,
                 grow=1.2, shrink=0.5, safety=0.9, gap_end=5e-5, max_steps=40000,
                 scale_floor=1e-8):
        self.tol, self.h, self.h_max = tol, h0, h_max
        self.ratio_max, self.ratio_min = ratio_max, ratio_min
        self.grow, self.shrink, self.safety = grow, shrink, safety
        self.t_end = 1.0 - gap_end
        self.max_steps = max_steps
        self.scale_floor = scale_floor
        self.scale = None                    # running max |e_i|
        self.errors, self.n_at_floor, self.n_at_cap = [], 0, 0

    def observe_initial(self, e0):
        """Seed the moment scales with the t = 0 target moments."""
        self.scale = e0.detach().abs().double()

    def next_t(self, t):
        """Next grid point from the current t and the current step size."""
        gap = 1.0 - t
        lo, hi = self.ratio_min * gap, min(self.h_max, self.ratio_max * gap)
        h = min(max(self.h, lo), hi)
        self.n_at_floor += h == lo
        self.n_at_cap += h == hi
        self.h = h
        return min(t + h, self.t_end)

    def update(self, e, p):
        """Record the error after the step that just ended; adapt h for the next one."""
        e, p = e.detach().double(), p.detach().double()
        self.scale = torch.maximum(self.scale, e.abs()) if self.scale is not None else e.abs()
        keep = self.scale > self.scale_floor
        err = float(((e - p).abs()[keep] / self.scale[keep]).mean()) if keep.any() else 0.0
        self.errors.append(err)
        factor = self.grow if err == 0 else self.safety * self.tol / err
        self.h *= min(max(factor, self.shrink), self.grow)
        return err

    def done(self, t):
        return t >= self.t_end - 1e-15

    def summary(self):
        e = np.asarray(self.errors)
        return (f'adaptive: {len(e)} steps, err mean {e.mean():.2e} max {e.max():.2e} '
                f'(tol {self.tol:g}); h pinned at ratio_min floor {self.n_at_floor} steps, '
                f'at cap {self.n_at_cap} steps') if len(e) else 'adaptive: no steps'


# ================================================================================
# Construction from run_SDE.py arguments
# ================================================================================

def build_time_grid(args, logger=None):
    """Grid to pass to the SDE for args.schedule, plus the controller (adaptive only).

    power     : legacy float32 grid with the round-to-1.0000 cut (as before).
    two_phase : float64 fixed grid.
    adaptive  : placeholder grid [0, 1 - gap_end]; the SDE builds the real one.
    """
    log = logger.info if logger is not None else print
    schedule = getattr(args, 'schedule', 'power')

    if schedule == 'power':
        t = power_schedule(args.nt, args.schedule_exponent)
        t_final = int((torch.round(t, decimals=4) == 1.0).nonzero(as_tuple=True)[0][0])
        log(f'schedule power^{args.schedule_exponent}: t_final = {t_final}/{len(t)} '
            f'(last t = {t[t_final - 1].item():.6f}, dropping {len(t) - t_final} '
            f'redundant trailing points at 1.0000)')
        return t[:t_final], None

    if schedule == 'two_phase':
        t, info = two_phase_schedule(args.nt, args.n_bulk, args.t_switch, args.gap_end)
        log(f"schedule two_phase: {args.n_bulk} uniform steps on [0, {args.t_switch}] "
            f"(h = {info['h_bulk']:.2e}), {args.nt - args.n_bulk} tail steps at "
            f"h/(1-t) = {info['tail_ratio']:.3e} (h {info['h_tail_first']:.2e} -> "
            f"{info['h_tail_last']:.2e}), ends at 1 - t = {args.gap_end:g}")
        return t, None

    if schedule == 'adaptive':
        ctrl = AdaptiveStepController(
            tol=args.adapt_tol, h0=args.adapt_h0, h_max=args.adapt_h_max,
            ratio_max=args.adapt_ratio_max, ratio_min=args.adapt_ratio_min,
            grow=args.adapt_grow, gap_end=args.gap_end, max_steps=args.nt,
        )
        log(f'schedule adaptive: tol {ctrl.tol:g}, h0 {args.adapt_h0:g}, h_max {ctrl.h_max:g}, '
            f'h/(1-t) in [{ctrl.ratio_min:g}, {ctrl.ratio_max:g}], grow x{ctrl.grow}, '
            f'at most {ctrl.max_steps} steps, ends at 1 - t = {args.gap_end:g}')
        return torch.tensor([0.0, ctrl.t_end], dtype=torch.float64), ctrl

    raise ValueError(f'unknown schedule {schedule!r}')


def add_schedule_args(p):
    """argparse options for the three schedules (shared by the entry points)."""
    g = p.add_argument_group('time grid (codes/time_schedules.py)')
    g.add_argument('--schedule', choices=['power', 'two_phase', 'adaptive'], default='power',
                   help='power: legacy t = 1-(1-s)^schedule_exponent (float32, cut at 1.0000); '
                        'two_phase: n_bulk uniform steps to t_switch, then constant h/(1-t); '
                        'adaptive: h from the previous step\'s moment error (nt = step cap)')
    g.add_argument('--gap_end', type=float, default=5e-5,
                   help='two_phase/adaptive: stop at t = 1 - gap_end')
    g.add_argument('--n_bulk', type=int, default=10000,
                   help='two_phase: uniform steps on [0, t_switch] (tail gets nt - n_bulk)')
    g.add_argument('--t_switch', type=float, default=0.9,
                   help='two_phase: start of the constant-ratio tail')
    g.add_argument('--adapt_tol', type=float, default=1e-3,
                   help='adaptive: target mean moment error, |e - p| / max_t|e| per moment')
    g.add_argument('--adapt_h0', type=float, default=1e-5, help='adaptive: first step')
    g.add_argument('--adapt_h_max', type=float, default=1e-3, help='adaptive: largest step')
    g.add_argument('--adapt_ratio_max', type=float, default=1e-3,
                   help='adaptive: cap on h/(1-t) (runs jumped at ~1.5e-3)')
    g.add_argument('--adapt_ratio_min', type=float, default=1e-5,
                   help='adaptive: floor on h/(1-t) (bounds the step count)')
    g.add_argument('--adapt_grow', type=float, default=1.2,
                   help='adaptive: max growth of h per step')
    return p
