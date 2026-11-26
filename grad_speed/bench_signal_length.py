"""Cost of one MGD step per numerical operation, for a given signal length M and number of scales J (2026-11-20).

Question: where does the time of a step go when M grows (256 -> 512 -> 1024, J = log2 M, J-1, J-2), so that the
turbulence model (arm bicw_allmass_scat34d, the final model) can be run at M = 1024? No SDE trajectory is run: the
operations of one step are timed on fixed walker states (interpolant samples cos(a) x0 + sin(a) x1 at chosen t),
with the real SDE methods, the real fitted GGD states (from a pre-flight run of the same M, J) and the production
settings (sigma 3.5, ridge 1e-4, float64 solves, batch 5000).

Per cell (M, J) it writes <out>/bench_M<M>_J<J>.json and prints:
  1. r (statistics) per potential family.
  2. Per family: forward, full Jacobian, Jacobian contracted with a vector; median ms and peak GPU GiB.
  3. Per SDE operation at t = 0.5 and t = 0.99, for each batch size of --batches (OOM is recorded, not fatal):
     Jacobian assembly, Gram product alone (bmm + sum), compute_G, moment velocity, projection, moments,
     moment mismatch, the r x r solve in float64 and float32, Sigma (float64), copy of M_k and Sigma_k to the CPU and
     write to disk, and the whole step (iteration_step_projection) + the bookkeeping of forward_regularised.
     Everything is also given per walker-step.
  4. Float32 vs float64 solve: eta and theta computed with solve_float64 True and False on the same states
     (t in --times). The difference is compared with the sampling noise of the solution (theta from two halves of
     the walkers), and the extreme eigenvalues of the Jacobi-scaled Gram are reported. Decision rule: the float32
     solve is harmless if its difference is well below the sampling noise (ratio << 1).
  5. Kernel profile of 3 whole steps: GPU time by kernel type (fft / gemm / elementwise / copy / index / reduce /
     other) and the fraction of the wall time the GPU is busy (low = launch-bound).
  6. Real loop (--real_steps, default 100): the real forward_regularised on true walkers from t = 0, production
     arguments, with CUDA-event timers wrapped around the SDE methods and the potentials (nothing in codes/ is
     edited). Reports ms per step per operation, the whole-loop wall time per step (bookkeeping and disk included),
     peak memory, and a fidelity check of the fixed-state version: ratio real/fixed per operation, and the
     precision test of 4 repeated with the real walkers at the last t of the loop. The loop only reaches
     t = steps * 3.6e-4 (about 0.04), so late-time conditioning is still tested on interpolant samples only.

Walkers are interpolant samples, not SDE walkers (their moments are matched to the interpolant, so the statistics of
the Gram are close, not identical). Timing does not depend on the values, except through dead statistics.

Usage (repo root, JZ dev GPU job; needs the fitted_potentials folder of a pre-flight run with the same M, J):
  python -u grad_speed/bench_signal_length.py --M 1024 --J 10 --subseries_len 1024 --target_len 1024 \
         --fitted_dir <run>/fitted_potentials --out grad_speed/results_bench_signal_length
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'codes'))
from filters_bank import return_Filters                         # noqa: E402
from potentials_builder import get_1d_potentials                # noqa: E402
from codes.sde_routines import SDE                              # noqa: E402
from codes.utils import split_periodize_reshape, normalize      # noqa: E402
from codes.time_schedules import two_phase_schedule             # noqa: E402
from codes.ortho_wavelet.ReadyToUseWavelets import DefineWavelet  # noqa: E402
from data.data_loader import load_turbulence_1d                 # noqa: E402

TERMS = ['L_6', 'Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK', 'Scattering_Third_Order_Real',
         'Scattering_Third_Order_Imag_nolp', 'Scattering_Fourth_Order_Mod2_Real_Q1_diag',
         'Scattering_Fourth_Order_Mod2_Imag_Q1']
SIGMA, RIDGE = 3.5, 1e-4


def load_x1(subseries_len, target_len, n1, device):
    """Same data path as turbulence/run_SDE.py (windows, Daubechies-3 coarse-graining, normalisation)."""
    W = DefineWavelet('Db', m=3, device=device)
    data = split_periodize_reshape(load_turbulence_1d().to(device), subseries_len)
    pre_len = data.shape[-1]
    if pre_len > target_len:
        for _ in range(int(np.log2(pre_len / target_len))):
            data = W.decompose(data)[1]
    if data.shape[0] < n1:
        raise ValueError(f'only {data.shape[0]} windows of length {subseries_len}, need n1 = {n1}')
    return normalize(data[:n1]).to(device)


def build_sde(x1, M, J, Q, fitted_dir, batch_size, device):
    filters, filters_Phi = return_Filters(M, J, 1, device=device, include_phi=True)
    filters_Q = return_Filters(M, J, Q, device=device)
    pots = get_1d_potentials(TERMS, J, filters, Q, filters_Q=filters_Q, filters_Phi=filters_Phi,
                             deduplicate_filters=True, dedup_tol=1e-8,
                             ggd_kwargs=dict(tail_quantile=[0.99], tail_mass=True, all_masses=True))
    for name in ('Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK'):
        if not (Path(fitted_dir) / f'{name}.pt').exists():
            raise FileNotFoundError(f'{fitted_dir}/{name}.pt: run the pre-flight of this (M, J) first')
    t = torch.tensor([0.5, 0.5 + 1e-3, 0.5 + 2e-3], dtype=torch.float64)
    return SDE(x1, x1.shape[0], x1.shape[0], t, SIGMA, pots, batch_size, device=device, regularization=RIDGE,
               interpolant='Cos', potentials_save_dir=fitted_dir, solve_float64=True)


def set_time(S, tc):
    """Grid [tc, tc + h, tc + 2h] with the whole step inside (0, 1); step k = 0 is then the step at tc."""
    h = 1e-3 if tc < 0.9 else (1.0 - tc) / 4.0
    S.t = torch.tensor([tc, tc + h, tc + 2 * h], dtype=torch.float64)
    S._rhs_dt_cache = None


def state_at(S, tc, gen):
    """(B, 1, M) samples of the interpolant at tc with fresh noise (stand-in for the walkers)."""
    a = np.pi * tc / 2
    noise = torch.randn(S.x_1.shape, device=S.x_1.device, generator=gen)
    return float(np.cos(a)) * noise + float(np.sin(a)) * S.x_1


def bench(fn, reps, warm):
    """(median ms, peak GiB above the starting allocation); (None, None) on out-of-memory."""
    try:
        with torch.no_grad():
            for _ in range(warm):
                fn()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            fn()
            torch.cuda.synchronize()
            peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
            ts = []
            for _ in range(reps):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                fn()
                b.record()
                torch.cuda.synchronize()
                ts.append(a.elapsed_time(b))
        return statistics.median(ts), peak
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def sde_operations(S, x, y, I0, I1, tmp_file):
    """name -> callable, each one an operation of a step on the walkers x (predictor), y (corrector)."""
    r = S.num_potentials
    eta = torch.randn(r, device=x.device)
    xb = x[:S.batch_size]
    G = S.compute_G(x)
    rhs = S.compute_rhs_dt_phi_I_t(I0, 0)
    D = torch.diag(G).clamp_min(1e-30).sqrt()
    Gs = (G / (D[:, None] * D[None, :])).double()
    Gs = (Gs + Gs.T) / 2 + RIDGE * torch.eye(r, device=x.device, dtype=torch.float64)
    b64 = (rhs / D[:, None]).double()
    Gs32, b32 = Gs.float(), b64.float()
    mom = S.compute_moments(x)

    def sigma64():
        m = mom.double()
        dev = m - m.mean(0)
        return dev.T @ dev / m.shape[0]

    def to_disk():
        torch.save({'M': G.float().cpu(), 'Sigma': sigma64().float().cpu()}, tmp_file)

    def step():
        S._rhs_dt_cache = (0, rhs)                                   # as in the loop: the cache holds rhs at t_k
        return S.iteration_step_projection(x, 0)

    def step_with_bookkeeping():
        out = step()
        mom1 = S.compute_moments(out[0])                              # phi(x_{k+1}) for Sigma and barphi_p
        dev = mom1.double() - mom1.double().mean(0)
        torch.save({'M': S._G_theta.cpu(), 'Sigma': (dev.T @ dev / dev.shape[0]).float().cpu()}, tmp_file)

    return {
        'jacobian_assembly (compute_grad_potentials, one batch)': lambda: S.compute_grad_potentials(xb),
        'compute_G (all walkers)': lambda: S.compute_G(x),
        'moment_velocity (compute_rhs_dt_phi_I_t)': lambda: S.compute_rhs_dt_phi_I_t(I1, 0),
        'projection (compute_grad_phi_projected)': lambda: S.compute_grad_phi_projected(x, eta),
        'moments (compute_moments)': lambda: S.compute_moments(x),
        'moment_mismatch (compute_rhs_constraint_correction)': lambda: S.compute_rhs_constraint_correction(y, I1),
        'live_check (_live_potentials)': lambda: S._live_potentials(G, rhs, 0, 'eta'),
        'solve_float64': lambda: torch.linalg.solve(Gs, b64),
        'solve_float32': lambda: torch.linalg.solve(Gs32, b32),
        'sigma_float64': sigma64,
        'save_system_to_disk (M_k, Sigma_k, float32)': to_disk,
        'step (iteration_step_projection)': step,
        'step + bookkeeping (as forward_regularised)': step_with_bookkeeping,
    }


def gram_product_alone(S, x, reps, warm):
    """bmm + sum on an already assembled (B, r, M) Jacobian block: the Gram GEMM without the assembly."""
    try:
        with torch.no_grad():
            gb = S.compute_grad_potentials(x[:S.batch_size])
        res = bench(lambda: torch.bmm(gb, gb.transpose(1, 2)).sum(0), reps, warm)
        del gb
        torch.cuda.empty_cache()
        return res
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def potential_operations(S, x, reps, warm):
    out = {}
    xb = x[:S.batch_size]
    for name, p in S.potentials.items():
        v = torch.randn(int(p.num_coefficients), device=x.device)
        res = {'num_coefficients': int(p.num_coefficients)}
        for label, fn in (('forward', lambda: p.forward(xb)), ('jacobian', lambda: p.grad(xb)),
                          ('jacobian_contracted', lambda: p.grad(xb, v=v))):
            ms, gib = bench(fn, reps, warm)
            res[label] = {'ms': ms, 'peak_GiB': gib}
        out[name] = res
        print(f'  {name:48s} r {res["num_coefficients"]:4d} | forward {res["forward"]["ms"]} ms | jacobian '
              f'{res["jacobian"]["ms"]} ms ({res["jacobian"]["peak_GiB"]} GiB) | contracted '
              f'{res["jacobian_contracted"]["ms"]} ms', flush=True)
    return out


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def precision_test(S, tc, gen, x=None, y=None):
    """eta and theta with solve_float64 True / False on the same states, vs the sampling noise of the solution.
    x, y: states to use (real walkers); default: interpolant samples at tc."""
    set_time(S, tc)
    x = state_at(S, tc, gen) if x is None else x
    y = state_at(S, tc, gen) if y is None else y
    I0, n = S.compute_interpolant(0), x.shape[0]
    rhs = S.compute_rhs_dt_phi_I_t(I0, 0)
    res = {'t': tc}
    G = S.compute_G(x).double()
    D = torch.diag(G).clamp_min(1e-30).sqrt()
    mu = torch.linalg.eigvalsh((G / (D[:, None] * D[None, :]) + (G / (D[:, None] * D[None, :])).T) / 2)
    res['scaled_gram_mu_min'], res['scaled_gram_mu_max'] = float(mu[0]), float(mu[-1])
    res['condition_with_ridge'] = float((mu[-1] + RIDGE) / (max(float(mu[0]), 0.0) + RIDGE))

    def solve(fn, f64):
        S.solve_float64 = f64
        S._rhs_dt_cache = (0, rhs)
        return fn()

    eta64, eta32 = solve(lambda: S.compute_eta(x, 0), True), solve(lambda: S.compute_eta(x, 0), False)
    th64 = solve(lambda: S.compute_theta(y, 0)[0], True)
    th32 = solve(lambda: S.compute_theta(y, 0)[0], False)
    thA = solve(lambda: S.compute_theta(y[:n // 2], 0)[0], True)
    thB = solve(lambda: S.compute_theta(y[n // 2:], 0)[0], True)
    S.solve_float64 = True
    d_prec, d_noise = (th32.double() - th64.double()), (thA.double() - thB.double())
    res.update({
        'eta_float32_vs_float64_rel': rel(eta32, eta64),
        'theta_float32_vs_float64_rel': rel(th32, th64),
        'theta_halves_rel (sampling noise, 2 x n/2 walkers)': rel(thA, thB),
        'theta_precision_over_noise (norm ratio; << 1 = float32 harmless)': float(d_prec.norm() / d_noise.norm()),
        'theta_fraction_of_statistics_with_precision_diff_above_10pct_of_noise':
            float(((d_prec.abs() > 0.1 * d_noise.abs()).float()).mean()),
    })
    return res


class Timers:
    """CUDA-event timers set as instance attributes on S and its potentials (nothing in codes/ is edited).
    Events are read after one final synchronisation, so the timing does not add syncs to the loop."""

    def __init__(self, S, names):
        self.step, self.rec, self.starts = 0, {}, []
        for n in names:
            self._wrap(S, n, n)
        for pname, p in S.potentials.items():
            for m in ('forward', 'grad'):
                self._wrap(p, m, f'{pname}.{m}')

    def _wrap(self, obj, attr, label):
        f = getattr(obj, attr)
        self.rec[label] = []

        def g(*a, **k):
            if label == 'iteration_step_projection':
                self.step += 1
                self.starts.append(time.perf_counter())
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = f(*a, **k)
            e.record()
            self.rec[label].append((self.step, s, e))
            return out
        setattr(obj, attr, g)

    def summary(self, warm=10):
        """Per operation: calls per step and median (over steps > warm) of the ms spent per step."""
        torch.cuda.synchronize()
        res = {}
        for label, evs in self.rec.items():
            per = {}
            for step, s, e in evs:
                if step > warm:
                    n, ms = per.get(step, (0, 0.0))
                    per[step] = (n + 1, ms + s.elapsed_time(e))
            if per:
                res[label] = {'calls_per_step': statistics.median(v[0] for v in per.values()),
                              'ms_per_step': statistics.median(v[1] for v in per.values())}
        wall = [b - a for a, b in zip(self.starts[warm:], self.starts[warm + 1:])]
        res['wall_per_step (whole loop, incl. bookkeeping and disk)'] = {'ms_per_step': 1e3 * statistics.median(wall)}
        return res


def real_loop(S, steps, nt, n_bulk, out):
    """The real forward_regularised (production arguments) for `steps` steps of the production two_phase grid from
    t = 0, on real walkers, with timers on the methods. The system is written to `out`, as in production."""
    t, _ = two_phase_schedule(nt, n_bulk)
    S.t = t[:steps + 1].clone()
    S._rhs_dt_cache = None
    names = ['iteration_step_projection', 'compute_eta', 'compute_theta', 'compute_G', 'compute_rhs_dt_phi_I_t',
             'compute_grad_phi_projected', 'compute_moments', 'compute_rhs_constraint_correction',
             '_live_potentials']
    tm = Timers(S, names)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    S.forward_regularised(lam=0.0, n_subsample=1, reg_solver='thomas', reg_ridge=0.0,
                          reg_system_path=out / 'real_loop_system.pt', solve_reg=False, reg_mode='moment',
                          reg_schedule='uniform')
    torch.cuda.synchronize()
    res = tm.summary()
    res['_loop_seconds_total (incl. start-up)'] = time.perf_counter() - t0
    res['_peak_GiB_allocated'] = torch.cuda.max_memory_allocated() / 2 ** 30
    res['_GiB_reserved'] = torch.cuda.memory_reserved() / 2 ** 30
    return res


def kernel_profile(fn, n=3):
    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad():
        fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
    cats = {'fft': ('fft',), 'gemm': ('gemm', 'cutlass', 'bmm', 'gemv'), 'copy': ('copy', 'memcpy', 'memset'),
            'index_gather': ('index', 'gather', 'scatter'), 'reduce': ('reduce', 'sum'),
            'elementwise': ('elementwise', 'vectorized')}
    total, by = 0.0, {}
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA:       # GPU kernels / copies only
            continue
        us = getattr(e, 'device_time_total', None)
        if us is None:
            us = getattr(e, 'cuda_time_total', 0.0)
        if not us:
            continue
        name = e.name.lower()
        cat = next((c for c, keys in cats.items() if any(k in name for k in keys)), 'other')
        by[cat] = by.get(cat, 0.0) + us
        total += us
    return {'steps': n, 'wall_s_per_step': wall / n, 'gpu_kernel_s_per_step': total / 1e6 / n,
            'gpu_busy_fraction (includes profiler overhead)': total / 1e6 / wall,
            'share_by_kernel_type': {c: v / total for c, v in sorted(by.items(), key=lambda kv: -kv[1])}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--M', type=int, required=True)
    p.add_argument('--J', type=int, required=True)
    p.add_argument('--Q', type=int, default=3)
    p.add_argument('--subseries_len', type=int, required=True)
    p.add_argument('--target_len', type=int, required=True)
    p.add_argument('--n', type=int, default=5000, help='walkers (one production batch)')
    p.add_argument('--batches', type=int, nargs='+', default=[5000, 2500, 1000])
    p.add_argument('--times', type=float, nargs='+', default=[0.5, 0.9, 0.99, 0.999], help='precision test')
    p.add_argument('--time_ops', type=float, nargs='+', default=[0.5, 0.99])
    p.add_argument('--reps', type=int, default=20)
    p.add_argument('--warm', type=int, default=3)
    p.add_argument('--fitted_dir', required=True)
    p.add_argument('--out', default=str(ROOT / 'grad_speed' / 'results_bench_signal_length'))
    p.add_argument('--skip_profile', action='store_true')
    p.add_argument('--real_steps', type=int, default=100, help='steps of the real loop; 0 = skip')
    p.add_argument('--nt', type=int, default=27500)
    p.add_argument('--n_bulk', type=int, default=2500)
    args = p.parse_args()

    device = 'cuda'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device=device).manual_seed(0)
    torch.manual_seed(0)
    print(f'GPU {torch.cuda.get_device_name()}; torch {torch.__version__}; M {args.M} J {args.J} Q {args.Q} '
          f'n {args.n}', flush=True)

    x1 = load_x1(args.subseries_len, args.target_len, args.n, device)
    if x1.shape[-1] != args.M:
        raise ValueError(f'signal length {x1.shape[-1]} != --M {args.M}')
    S = build_sde(x1, args.M, args.J, args.Q, args.fitted_dir, max(args.batches), device)
    sizes = {k: int(v.num_coefficients) for k, v in S.potentials.items()}
    print(f'r = {S.num_potentials}: {sizes}', flush=True)
    result = {'M': args.M, 'J': args.J, 'Q': args.Q, 'n_walkers': args.n, 'r': S.num_potentials,
              'r_by_potential': sizes, 'gpu': torch.cuda.get_device_name(), 'reps': args.reps}
    path = out / f'bench_M{args.M}_J{args.J}.json'

    def save():
        path.write_text(json.dumps(result, indent=1))

    # 1. per potential family, full batch
    set_time(S, 0.5)
    x = state_at(S, 0.5, gen)
    S.batch_size = max(args.batches)
    print('\n== per potential family', flush=True)
    result['per_potential'] = potential_operations(S, x, args.reps, args.warm)
    save()

    # 2. per SDE operation, per time and batch size
    result['operations'] = {}
    for tc in args.time_ops:
        set_time(S, tc)
        x, y = state_at(S, tc, gen), state_at(S, tc, gen)
        I0, I1 = S.compute_interpolant(0), S.compute_interpolant(1)
        for bs in args.batches:
            S.batch_size = bs
            print(f'\n== operations at t = {tc}, batch {bs}', flush=True)
            res = {}
            ms, gib = gram_product_alone(S, x, args.reps, args.warm)
            res['gram_product_alone (bmm + sum, one batch)'] = {'ms': ms, 'peak_GiB': gib}
            print(f'  {"gram_product_alone (bmm + sum, one batch)":58s} {ms} ms   peak {gib} GiB', flush=True)
            try:
                ops = sde_operations(S, x, y, I0, I1, out / 'tmp_node.pt')
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print('  out of memory while preparing the operations at this batch size', flush=True)
                result['operations'][f't{tc}_batch{bs}'] = {'out_of_memory': True, **res}
                save()
                continue
            for name, fn in ops.items():
                ms, gib = bench(fn, args.reps, args.warm)
                res[name] = {'ms': ms, 'peak_GiB': gib, 'us_per_walker': None if ms is None else 1e3 * ms / args.n}
                print(f'  {name:58s} {ms if ms is None else round(ms, 2)} ms   peak {gib} GiB', flush=True)
            result['operations'][f't{tc}_batch{bs}'] = res
            save()

    # 3. float32 vs float64 solves
    S.batch_size = max(args.batches)
    print('\n== float32 vs float64 solve', flush=True)
    result['precision_test'] = []
    for tc in args.times:
        r = precision_test(S, tc, gen)
        result['precision_test'].append(r)
        print(json.dumps(r), flush=True)
        save()

    # 4. kernel profile of whole steps
    if not args.skip_profile:
        print('\n== kernel profile', flush=True)
        set_time(S, 0.5)
        x, y = state_at(S, 0.5, gen), state_at(S, 0.5, gen)
        I0, I1 = S.compute_interpolant(0), S.compute_interpolant(1)
        try:
            ops = sde_operations(S, x, y, I0, I1, out / 'tmp_node.pt')
            result['kernel_profile'] = kernel_profile(ops['step + bookkeeping (as forward_regularised)'])
            print(json.dumps(result['kernel_profile'], indent=1), flush=True)
        except Exception as e:                                    # the timings above are already saved
            result['kernel_profile'] = f'failed: {type(e).__name__}: {e}'
            print(result['kernel_profile'], flush=True)
        save()

    # 5. real loop (true walkers) and how faithful the fixed-state version is
    if args.real_steps:
        print(f'\n== real loop: {args.real_steps} steps of forward_regularised', flush=True)
        S.batch_size = args.batches[0]
        result['real_loop'] = real_loop(S, args.real_steps, args.nt, args.n_bulk, out)
        for k, v in result['real_loop'].items():
            print(f'  {k:62s} {v}', flush=True)
        save()
        tN = float(S.t[-1])
        x_real = S.x_k.clone()
        h = float(S.t[-1] - S.t[-2])
        y_real = x_real + (2 * h) ** 0.5 * SIGMA * torch.randn_like(x_real)
        S.batch_size = max(args.batches)
        result['fidelity_precision_test'] = {'t': tN, 'interpolant_samples': precision_test(S, tN, gen),
                                             'real_walkers': precision_test(S, tN, gen, x=x_real, y=y_real)}
        print(json.dumps(result['fidelity_precision_test'], indent=1), flush=True)
        # same operation, fixed-state version (t = 0.5, first batch size) vs the real loop (ms per call)
        fixed = result['operations'].get(f't{args.time_ops[0]}_batch{args.batches[0]}', {})
        pairs = {'compute_G': 'compute_G (all walkers)', 'compute_rhs_dt_phi_I_t': 'moment_velocity (compute_rhs_dt_phi_I_t)',
                 'compute_grad_phi_projected': 'projection (compute_grad_phi_projected)',
                 'compute_moments': 'moments (compute_moments)',
                 'compute_rhs_constraint_correction': 'moment_mismatch (compute_rhs_constraint_correction)',
                 'iteration_step_projection': 'step (iteration_step_projection)'}
        fid = {}
        for real_name, fixed_name in pairs.items():
            r = result['real_loop'].get(real_name)
            f = fixed.get(fixed_name, {}).get('ms')
            if r and f:
                per_call = r['ms_per_step'] / r['calls_per_step']
                fid[real_name] = {'fixed_state_ms': f, 'real_loop_ms_per_call': per_call, 'ratio_real_over_fixed': per_call / f}
        result['fidelity_timing'] = fid
        print(json.dumps(fid, indent=1), flush=True)
        save()
    print(f'\nsaved {path}', flush=True)


if __name__ == '__main__':
    main()
