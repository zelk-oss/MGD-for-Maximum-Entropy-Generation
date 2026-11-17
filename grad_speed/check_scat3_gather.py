"""Check of the Scat3 full-Jacobian restriction to the kept pairs (2026-11-17): new grad(x) vs the old code.

Old: P (phase term) and Q (modulus term) evaluated on the full J x (J+1) grid of (j, k), then restricted to the
kept pairs; new: evaluated on the kept pairs only. Same formula, so the difference must be round-off.
Same checks and thresholds as check_scat4_gather.py (Real, Imag, Imag_nolp; float32 + float64; timing).

Usage (repo root, JZ dev GPU job): python -u grad_speed/check_scat3_gather.py [--M 256 --J 8 --B 16 --B_time 500]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_scat4_gather import rel, timed, TOL, return_Filters, get_1d_potentials, abs_eps   # noqa: E402
from potentials.utils_potentials import indices_third_order                                  # noqa: E402

TERMS = ['Scattering_Third_Order_Real', 'Scattering_Third_Order_Imag', 'Scattering_Third_Order_Imag_nolp']


# ---------------------------------------------------------------- old code (verbatim, commit 32101c3)

def _common(self, x):
    filters = self.filters.to(x.device)
    x_fourier = torch.fft.fft(x)
    filters_abs2 = filters*filters.conj()
    x_filtered_2 = torch.fft.ifft(x_fourier*filters_abs2)
    x_filtered = torch.fft.ifft(filters*x_fourier)
    x_filtered_abs = abs_eps(x_filtered)
    x_filtered_over_abs = x_filtered/x_filtered_abs
    return filters, filters_abs2, x_filtered_2, x_filtered_abs, x_filtered_over_abs


def old_real(self, x):
    filters, filters_abs2, x_filtered_2, x_filtered_abs, x_filtered_over_abs = _common(self, x)
    indices = indices_third_order(self.J, 1)
    result_conv_phase = x_filtered_over_abs[:,:-1,None]*torch.real(x_filtered_2)[:,None]
    result_conv_phase = torch.fft.ifft(filters[:,:-1,None].conj()*torch.fft.fft(result_conv_phase))
    result_conv_modulus = torch.fft.ifft(filters_abs2[:,None]*torch.fft.fft(x_filtered_abs)[:,:-1,None])
    output = (result_conv_phase + result_conv_modulus).real
    return output[:, indices[0], indices[1]]/x.shape[-1]


def old_imag(self, x):
    filters, filters_abs2, x_filtered_2, x_filtered_abs, x_filtered_over_abs = _common(self, x)
    filters_abs2_flip = torch.roll(torch.flip(filters_abs2, [-1]), 1, -1)
    indices = self._indices()
    result_conv_phase = x_filtered_over_abs[:,:-1,None]*(-torch.imag(x_filtered_2))[:,None]
    result_conv_phase = torch.fft.ifft(filters[:,:-1,None].conj()*torch.fft.fft(result_conv_phase))
    result_conv_modulus = torch.fft.ifft(filters_abs2_flip[:,None]*torch.fft.fft(x_filtered_abs)[:,:-1,None])
    output = result_conv_phase.real - result_conv_modulus.imag
    return output[:, indices[0], indices[1]]/x.shape[-1]


def build(M, J, dtype, device):
    filters = return_Filters(M, J, 1, device=device)
    if dtype == torch.float64:
        filters = filters.to(torch.complex128) if filters.is_complex() else filters.double()
    return get_1d_potentials(TERMS, J, filters, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--M', type=int, default=256)
    p.add_argument('--J', type=int, default=8)
    p.add_argument('--B', type=int, default=16)
    p.add_argument('--B_time', type=int, default=500)
    p.add_argument('--n_rep', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    device = 'cuda'
    torch.manual_seed(args.seed)
    print(f'M {args.M}, J {args.J}, B {args.B}; GPU {torch.cuda.get_device_name()}; torch {torch.__version__}')

    ok = True
    for dtype in (torch.float32, torch.float64):
        pots = build(args.M, args.J, dtype, device)
        x = torch.randn(args.B, 1, args.M, dtype=dtype, device=device)
        for name, pot in pots.items():
            old = old_imag if 'Imag' in name else old_real
            g_new, g_old = pot.grad(x), old(pot, x)
            if g_new.shape != g_old.shape:
                print(f'FAIL {name} {dtype}: shape new {tuple(g_new.shape)} vs old {tuple(g_old.shape)}')
                ok = False
                continue
            r1 = rel(g_new, g_old)
            r2 = 'not run (projected path float32-only)'
            passed = r1 < TOL[dtype]
            if dtype == torch.float32:
                v = torch.randn(pot.num_coefficients, dtype=dtype, device=device)
                r2v = rel(torch.einsum('k,bkm->bm', v, g_new)[:, None], pot.grad(x, v))
                passed &= r2v < TOL[dtype]
                r2 = f'{r2v:.2e}'
            ok &= passed
            print(f'{"PASS" if passed else "FAIL"} {name:35s} {str(dtype):14s} K {pot.num_coefficients:3d} | '
                  f'new vs old {r1:.2e} | sum v.grad vs grad(v) {r2} | tol {TOL[dtype]:.0e}')

    pots = build(args.M, args.J, torch.float32, device)
    x = torch.randn(args.B_time, 1, args.M, device=device)
    print(f'\ntiming, float32, B {args.B_time}, median of {args.n_rep}:')
    for name, pot in pots.items():
        old = old_imag if 'Imag' in name else old_real
        t_old = timed(lambda: old(pot, x), args.n_rep)
        t_new = timed(lambda: pot.grad(x), args.n_rep)
        print(f'  {name:35s} old {1e3 * t_old:8.2f} ms | new {1e3 * t_new:8.2f} ms | ratio {t_old / t_new:5.2f}')
    print('\nALL PASS' if ok else '\nSOME FAILED')


if __name__ == '__main__':
    main()
