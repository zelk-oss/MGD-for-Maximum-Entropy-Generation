"""Check of the 2D port of the 1D speed-ups (2026-11-17): new codes/potentials/potentials_2d.py vs the frozen copy
grad_speed/potentials_2d_before_1117.py (commit 3eddb31), same filters, same inputs. Then a timing, float32, at the
batch size of the 2D runs (--B_time 10): old vs new per class and path, and the Gram bmm of the scat4_real arm.

Ported changes and what is expected:
  - Scat3 Real/Imag full Jacobian on the kept pairs (1D b50b9bc)          -> grad(x)   bit-identical (as in 1D)
  - Scat4 Mod2 Real/Imag full Jacobian on the kept triples (1D 32101c3)   -> grad(x)   bit-identical (as in 1D)
  - Scat4 Mod2 Real/Imag forward on the kept triples (1D 6733c36)         -> forward   bit-identical (as in 1D)
  - Scat2 and L_6 projected gradient summed before the inverse FFT (846ab19) -> grad(x, v) round-off only: old and
    new float32 are both compared with the old code in float64 (reference), as in the 1D check.
Every other path (forward/grad of the unchanged classes) is printed too, as a control (must be bit-identical).

Usage (repo root, JZ dev GPU job): python -u grad_speed/check_2d_port.py [--M 128 --J 5 --L 4 --B 2 --B_time 10]
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from filters_bank import return_Filters                      # noqa: E402
import potentials.potentials_2d as new_mod                   # noqa: E402

_spec = importlib.util.spec_from_file_location('potentials_2d_before_1117',
                                               Path(__file__).resolve().parent / 'potentials_2d_before_1117.py')
old_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old_mod)

CLASSES = [  # (name, constructor args kind)
    ('L2p_norm_2d', 'p'),
    ('Scattering_First_Order_2d', 'f'),
    ('Scattering_Second_Order_2d', 'f'),
    ('Scattering_Third_Order_Real_2d', 'JLf'),
    ('Scattering_Third_Order_Imag_2d', 'JLf'),
    ('Scattering_Fourth_Order_Mod2_Real_2d', 'JLf'),
    ('Scattering_Fourth_Order_Mod2_Imag_2d', 'JLf'),
]


def make(mod, name, kind, J, L, filters):
    cls = getattr(mod, name)
    if kind == 'p':
        return cls(3, filters)
    if kind == 'f':
        return cls(filters)
    return cls(J, L, filters)


def rel(a, b):
    """max |a - b| / max |b| (b = reference)."""
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=128)
    ap.add_argument('--J', type=int, default=5)
    ap.add_argument('--L', type=int, default=4)
    ap.add_argument('--B', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--B_time', type=int, default=10, help='batch for the timing (0: no timing); 10 = 2D run batch')
    ap.add_argument('--reps', type=int, default=5)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device {device}, M {args.M}, J {args.J}, L {args.L}, B {args.B}', flush=True)

    f32 = return_Filters(args.M, args.J, 1, L=args.L, device=device)
    f64 = f32.to(torch.complex128) if f32.is_complex() else f32.double()
    g = torch.Generator(device='cpu').manual_seed(args.seed)
    x32 = torch.randn((args.B, 1, args.M, args.M), generator=g).to(device)
    x64 = x32.double()

    n_fail = 0
    print(f"{'class':40s} {'path':10s} {'equal32':>8s} {'rel32':>10s} {'equal64':>8s} {'rel64':>10s} "
          f"{'old32-ref':>10s} {'new32-ref':>10s}", flush=True)
    for name, kind in CLASSES:
        o32, n32 = make(old_mod, name, kind, args.J, args.L, f32), make(new_mod, name, kind, args.J, args.L, f32)
        o64, n64 = make(old_mod, name, kind, args.J, args.L, f64), make(new_mod, name, kind, args.J, args.L, f64)
        r = n32.num_coefficients
        if r != o32.num_coefficients:
            print(f'{name}: num_coefficients differ {r} vs {o32.num_coefficients}')
            n_fail += 1
            continue
        v32 = torch.randn(r, generator=g).to(device)
        v64 = v32.double()
        paths = {
            'forward': lambda m, x, v: m.forward(x),
            'grad': lambda m, x, v: m.grad(x),
            'grad_v': lambda m, x, v: m.grad(x, v=v),
        }
        for pname, fn in paths.items():
            with torch.no_grad():
                a32, b32 = fn(o32, x32, v32), fn(n32, x32, v32)
                try:
                    a64, b64 = fn(o64, x64, v64), fn(n64, x64, v64)
                except RuntimeError as e:
                    # Scat3/Scat4 projected paths build m in float32 whatever v is (old and new code alike: the port
                    # does not touch these paths), so they cannot run in float64. Compare float32 only.
                    if pname != 'grad_v' or 'dtypes match' not in str(e):
                        raise
                    eq32, r32 = torch.equal(a32, b32), rel(b32, a32)
                    print(f'{name:40s} {pname:10s} {str(eq32):>8s} {r32:10.2e} {"n/a":>8s} {"(f32 m)":>10s}', flush=True)
                    if not eq32:
                        n_fail += 1
                    continue
            if a32.shape != b32.shape:
                print(f'{name:40s} {pname:10s} SHAPE MISMATCH old {tuple(a32.shape)} new {tuple(b32.shape)}')
                n_fail += 1
                continue
            eq32, eq64 = torch.equal(a32, b32), torch.equal(a64, b64)
            r32, r64 = rel(b32, a32), rel(b64, a64)
            ro, rn = rel(a32.double(), a64), rel(b32.double(), a64)     # distance of old / new float32 to float64 ref
            print(f'{name:40s} {pname:10s} {str(eq32):>8s} {r32:10.2e} {str(eq64):>8s} {r64:10.2e} '
                  f'{ro:10.2e} {rn:10.2e}', flush=True)
            # bit-identity expected everywhere except the two ported projected paths (round-off, judged vs ref)
            roundoff_ok = name in ('L2p_norm_2d', 'Scattering_Second_Order_2d') and pname == 'grad_v'
            if roundoff_ok:
                if not (rn <= 10 * max(ro, 1e-7) and r64 < 1e-12):
                    n_fail += 1
            elif not (eq32 and eq64):
                n_fail += 1
            del a32, b32, a64, b64
            if device == 'cuda':
                torch.cuda.empty_cache()
    print(f'\nRESULT: {"PASS" if n_fail == 0 else f"FAIL ({n_fail} checks)"}', flush=True)
    if args.B_time > 0:
        timing(args, f32, device)


def timed(fn, device, reps):
    """median wall time (ms) of fn() over reps calls after one warm-up; None if out of memory."""
    import statistics
    import time
    try:
        with torch.no_grad():
            fn()
            if device == 'cuda':
                torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                out = fn()
                if device == 'cuda':
                    torch.cuda.synchronize()
                ts.append(1e3*(time.perf_counter() - t0))
                del out
        return statistics.median(ts)
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        if device == 'cuda':
            torch.cuda.empty_cache()


def timing(args, f32, device):
    """float32, B = B_time (the batch size of the 2D runs): old vs new per class and path, plus the Gram bmm of the
    scat4_real arm (L_6, Scat1, Scat2, Scat3 Real, Scat4 Mod2 Real) for scale."""
    B = args.B_time
    g = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    x = torch.randn((B, 1, args.M, args.M), generator=g).to(device)
    fmt = lambda t: '     OOM' if t is None else f'{t:8.1f}'
    print(f'\nTIMING float32, B {B}, median of {args.reps} (ms); speed-up = old / new', flush=True)
    print(f"{'class':40s} {'path':8s} {'old':>8s} {'new':>8s} {'speed-up':>9s}", flush=True)
    arm = {'L2p_norm_2d', 'Scattering_First_Order_2d', 'Scattering_Second_Order_2d',
           'Scattering_Third_Order_Real_2d', 'Scattering_Fourth_Order_Mod2_Real_2d'}
    tot = {p: [0.0, 0.0] for p in ('forward', 'grad', 'grad_v')}
    r_arm = 0
    for name, kind in CLASSES:
        o, n = make(old_mod, name, kind, args.J, args.L, f32), make(new_mod, name, kind, args.J, args.L, f32)
        v = torch.randn(n.num_coefficients, generator=g).to(device)
        if name in arm:
            r_arm += n.num_coefficients
        for pname, fo, fn in (('forward', lambda: o.forward(x), lambda: n.forward(x)),
                              ('grad', lambda: o.grad(x), lambda: n.grad(x)),
                              ('grad_v', lambda: o.grad(x, v=v), lambda: n.grad(x, v=v))):
            to, tn = timed(fo, device, args.reps), timed(fn, device, args.reps)
            su = '      -' if to is None or tn is None else f'{to/tn:9.2f}'
            print(f'{name:40s} {pname:8s} {fmt(to)} {fmt(tn)} {su}', flush=True)
            if name in arm and to is not None and tn is not None:
                tot[pname][0] += to
                tot[pname][1] += tn
    print(f'\nscat4_real arm (r = {r_arm}), sum over its 5 classes:', flush=True)
    for p, (to, tn) in tot.items():
        print(f'  {p:8s} old {to:8.1f}  new {tn:8.1f}  speed-up {to/max(tn, 1e-9):5.2f}', flush=True)
    # Gram product of compute_G for one batch: (B, r, MN) @ (B, MN, r)
    try:
        J_ = torch.randn((B, r_arm, args.M*args.M), device=device)
        tb = timed(lambda: torch.bmm(J_, J_.transpose(1, 2)).sum(0), device, args.reps)
        print(f'  Gram bmm (B, {r_arm}, {args.M*args.M}): {fmt(tb)} ms (unchanged by the port)', flush=True)
        del J_
    except torch.cuda.OutOfMemoryError:
        print('  Gram bmm: OOM', flush=True)


if __name__ == '__main__':
    main()
