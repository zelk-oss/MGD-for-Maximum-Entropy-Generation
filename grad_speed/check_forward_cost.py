"""Cost of the Scat4 Mod2 forward on the kept triples (1D commit 6733c36, 2D port 2026-11-17) vs the dense version it
replaced: wall time AND peak GPU memory, measured on their own (2026-11-18). Before this check: 1D was only checked for
bit-identity (job 341922; its "~5.9 GB saved" was computed from the array shape, never measured, and no time was
measured); 2D time was measured (job 349196: Mod2 Real forward 11.3 -> 16.8 ms, slower) but not memory.

For each Mod2 Real/Imag class: old (dense: all index combinations, then indexed) vs new (kept triples only), same
input, float32, torch.no_grad (as compute_moments). Reported per batch size:
  - bit-identical (sanity; both checks above found True)
  - median wall time over --reps calls after one warm-up (ms)
  - extra peak GPU memory of one call: max_memory_allocated during the call minus memory allocated before it (GiB),
    i.e. the temporaries plus the output
and, for scale, the time of the forward of ALL potentials of the model (new code) = one compute_moments batch.
The forward runs ~3 times per SDE step on all walkers (corrector on I_k and y_k, then x_{k+1}).

1D: B+Scat4 turbulence and jets 'noscalar' term sets (M 256, J 8, Q 3 / 1, seed-900 GGD fits), batch 5000 (production
    batch_size, as in check_jvp.py), old code = verbatim copy in check_jvp.py.
2D: Mod2 Real/Imag (M 128, J 5, L 4), batches 10 (scat4 runs) and 20 (base_real runs), old code = frozen module
    grad_speed/potentials_2d_before_1117.py.

Usage (repo root, JZ dev GPU job): python -u grad_speed/check_forward_cost.py [--B1d 5000 --B2d 10 20 --reps 10]
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_jvp as c1          # noqa: E402  1D: build(), old_scat4_forward(), TURB, JETS, M
import check_2d_port as c2      # noqa: E402  2D: old_mod, new_mod, make(), return_Filters


def measure(fn, reps):
    """(median ms, extra peak GiB) of fn() under no_grad; (None, None) on out-of-memory."""
    try:
        with torch.no_grad():
            out = fn()                                     # warm-up (cuFFT plans, allocator)
            del out
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            out = fn()
            torch.cuda.synchronize()
            mem = (torch.cuda.max_memory_allocated() - base) / 2**30
            del out
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                out = fn()
                torch.cuda.synchronize()
                ts.append(1e3 * (time.perf_counter() - t0))
                del out
        return statistics.median(ts), mem
    except torch.cuda.OutOfMemoryError:
        return None, None
    finally:
        torch.cuda.empty_cache()


def fmt(t, m):
    return 'OOM' if t is None else f'{t:.1f} ms, peak {m:.2f} GiB'


def row(label, B, eq, old, new):
    (to, mo), (tn, mn) = old, new
    f = lambda v, w: f'{"OOM":>{w}s}' if v is None else f'{v:{w}.{2 if w < 8 else 1}f}'
    su = '     -' if to is None or tn is None else f'{to / tn:6.2f}'
    print(f'{label:44s} {B:5d} {str(eq):>6s} | time ms old {f(to, 8)} new {f(tn, 8)} speed-up {su} '
          f'| peak GiB old {f(mo, 7)} new {f(mn, 7)}', flush=True)


def part_1d(args):
    print(f'\n##### 1D (M {c1.M}, J {c1.J}), batch {args.B1d}, float32, median of {args.reps}', flush=True)
    for label, terms in (('B+Scat4 turbulence', c1.TURB), ('jets noscalar', c1.JETS)):
        pots = c1.build(terms, 'cuda')
        x = torch.randn(args.B1d, 1, c1.M, device='cuda')
        print(f'--- {label}: r = {sum(int(p.num_coefficients) for p in pots.values())}', flush=True)
        for name in ('Scattering_Fourth_Order_Mod2_Real_Q1', 'Scattering_Fourth_Order_Mod2_Imag_Q1'):
            pot, imag = pots[name], 'Imag' in name
            with torch.no_grad():
                eq = torch.equal(pot.forward(x[:64]), c1.old_scat4_forward(pot, x[:64], imag))
            row(f'{name} forward', args.B1d, eq,
                measure(lambda: c1.old_scat4_forward(pot, x, imag), args.reps),
                measure(lambda: pot.forward(x), args.reps))
        t_all, m_all = measure(lambda: [p.forward(x) for p in pots.values()], args.reps)
        print(f'  all {len(pots)} potentials forward (new code, one compute_moments batch): {fmt(t_all, m_all)}',
              flush=True)
        del pots, x
        torch.cuda.empty_cache()


def part_2d(args):
    M, J, L = 128, 5, 4
    f32 = c2.return_Filters(M, J, 1, L=L, device='cuda')
    print(f'\n##### 2D (M {M}, J {J}, L {L}), batches {args.B2d}, float32, median of {args.reps}', flush=True)
    names = ('Scattering_Fourth_Order_Mod2_Real_2d', 'Scattering_Fourth_Order_Mod2_Imag_2d')
    arm = [('L2p_norm_2d', 'p'), ('Scattering_First_Order_2d', 'f'), ('Scattering_Second_Order_2d', 'f'),
           ('Scattering_Third_Order_Real_2d', 'JLf'), ('Scattering_Fourth_Order_Mod2_Real_2d', 'JLf')]
    for B in args.B2d:
        x = torch.randn(B, 1, M, M, device='cuda')
        for name in names:
            o, n = c2.make(c2.old_mod, name, 'JLf', J, L, f32), c2.make(c2.new_mod, name, 'JLf', J, L, f32)
            with torch.no_grad():
                eq = torch.equal(o.forward(x[:2]), n.forward(x[:2]))
            row(f'{name} forward', B, eq, measure(lambda: o.forward(x), args.reps),
                measure(lambda: n.forward(x), args.reps))
        pots = [c2.make(c2.new_mod, nm, k, J, L, f32) for nm, k in arm]
        t_all, m_all = measure(lambda: [p.forward(x) for p in pots], args.reps)
        print(f'  scat4_real arm, all 5 classes forward (new code), B {B}: {fmt(t_all, m_all)}', flush=True)
        del x, pots
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B1d', type=int, default=5000)
    ap.add_argument('--B2d', type=int, nargs='+', default=[10, 20])
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--skip_1d', action='store_true')
    ap.add_argument('--skip_2d', action='store_true')
    args = ap.parse_args()
    torch.manual_seed(0)
    print(f'GPU {torch.cuda.get_device_name()}; torch {torch.__version__}', flush=True)
    if not args.skip_1d:
        part_1d(args)
    if not args.skip_2d:
        part_2d(args)
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
