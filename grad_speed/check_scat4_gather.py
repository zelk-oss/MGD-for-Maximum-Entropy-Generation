"""Check of the Scat4 Mod2 full-Jacobian gather (2026-11-17): new grad(x) vs the old dense code.

The old code computed T for all (a, b, c) of the n_Q x n_Q x n_f grid and indexed the kept triples afterwards;
the new one computes only the entries the kept triples need. Same formula, so the difference must be round-off.

Checks, per potential (Mod2 Real, Mod2 Real diag, Mod2 Imag), float32 and float64, on the GPU:
  1. new grad(x) vs old dense grad(x) (verbatim copy below): max |diff| / max |old|;
  2. sum_k v_k grad_k(x) (new) vs grad(x, v) (projected path, unchanged code): max |diff| / max |grad(x, v)|;
  3. wall time of old vs new full grad (float32, B_TIME walkers, median of N_REP after warm-up).
Pass threshold (printed PASS/FAIL): relative diff < 1e-5 in float32, < 1e-12 in float64.

Usage (repo root, JZ dev GPU job): python -u grad_speed/check_scat4_gather.py [--M 256 --J 8 --B 16 --B_time 500]
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from filters_bank import return_Filters                      # noqa: E402
from potentials_builder import get_1d_potentials            # noqa: E402
from potentials.utils_potentials import abs_eps             # noqa: E402

TERMS = ['Scattering_Fourth_Order_Mod2_Real_Q1', 'Scattering_Fourth_Order_Mod2_Real_Q1_diag',
         'Scattering_Fourth_Order_Mod2_Imag_Q1']
TOL = {torch.float32: 1e-5, torch.float64: 1e-12}


# ---------------------------------------------------------------- old dense code (verbatim, commit 3a6f53f)

def dense_real(self, x):
    filters = self.filters.to(x.device)
    filters_Q = self.filters_Q.to(x.device)
    x_filtered_no_LF = torch.fft.ifft(filters_Q*torch.fft.fft(x))
    x_filtered_abs2_no_LF = abs_eps(x_filtered_no_LF)**2
    x_filtered_grad = 2 * x_filtered_no_LF
    x_filtered_abs2_no_LF_filtered_2 = torch.fft.ifft(
        (torch.fft.fft(x_filtered_abs2_no_LF)[:,:,None] * (filters*filters.conj())))
    indices = self.indices
    intermediate_output = x_filtered_grad[:,:,None,None]*torch.real(x_filtered_abs2_no_LF_filtered_2)[:,None]
    output = torch.fft.ifft(filters_Q[:,:,None,None].conj()*torch.fft.fft(intermediate_output)).real
    output = output + torch.transpose(output, 1,2)
    output = output[:, indices[0], indices[1], indices[2]].reshape(x.shape[0], indices.shape[1], x.shape[-1])/x.shape[-1]
    return output / self.norm_indices[:,None].to(x.device).to(x.dtype)


def dense_imag(self, x):
    filters = self.filters.to(x.device)
    filters_tilde = torch.fft.fft(torch.fft.ifft(filters).conj()).conj()
    filters_Q = self.filters_Q.to(x.device)
    filters_Q_tilde = torch.fft.fft(torch.fft.ifft(filters_Q).conj()).conj()
    filters_Q_tilde_adj = filters_Q_tilde if filters_Q.is_complex() else filters_Q_tilde.conj()
    x_filtered_no_LF = torch.fft.ifft(filters_Q*torch.fft.fft(x))
    x_filtered_abs2_no_LF = abs_eps(x_filtered_no_LF)**2
    x_filtered_grad = 2 * x_filtered_no_LF
    x_filtered_abs2_no_LF_filtered_2 = torch.fft.ifft(
        (torch.fft.fft(x_filtered_abs2_no_LF)[:,:,None]*filters.abs()**2))
    x_filtered_abs2_no_LF_filtered_2_tilde = torch.fft.ifft(
        (torch.fft.fft(x_filtered_abs2_no_LF)[:,:,None]*filters_tilde.abs()**2))
    indices = self.indices
    intermediate_output = 0.5*x_filtered_grad[:,:,None,None]*x_filtered_abs2_no_LF_filtered_2[:,None]
    intermediate_output_conj = 0.5*x_filtered_grad[:,:,None,None].conj()*x_filtered_abs2_no_LF_filtered_2[:,None]
    intermediate_output = torch.fft.ifft(filters_Q[:,:,None,None].conj()*torch.fft.fft(intermediate_output))
    intermediate_output_conj = torch.fft.ifft(filters_Q_tilde_adj[:,:,None,None]*torch.fft.fft(intermediate_output_conj))
    output = intermediate_output + intermediate_output_conj
    intermediate_output = 0.5*x_filtered_grad[:,:,None,None]*x_filtered_abs2_no_LF_filtered_2_tilde[:,None]
    intermediate_output_conj = 0.5*x_filtered_grad[:,:,None,None].conj()*x_filtered_abs2_no_LF_filtered_2_tilde[:,None]
    intermediate_output = torch.fft.ifft(filters_Q[:,:,None,None].conj()*torch.fft.fft(intermediate_output))
    intermediate_output_conj = torch.fft.ifft(filters_Q_tilde_adj[:,:,None,None]*torch.fft.fft(intermediate_output_conj))
    output_transpose = intermediate_output + intermediate_output_conj
    output = output + torch.transpose(output_transpose, 1,2)
    output = output[:, indices[0], indices[1], indices[2]].reshape(x.shape[0], indices.shape[1], x.shape[-1])/x.shape[-1]
    return output.imag / self.norm_indices[:,None].to(x.device).to(x.dtype)


# ---------------------------------------------------------------- helpers

def build(M, J, dtype, device):
    filters = return_Filters(M, J, 1, device=device)
    if dtype == torch.float64:
        filters = filters.to(torch.complex128) if filters.is_complex() else filters.double()
    return get_1d_potentials(TERMS, J, filters, 1)


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def timed(fn, n_rep):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


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
            dense = dense_imag if 'Imag' in name else dense_real
            g_new, g_old = pot.grad(x), dense(pot, x)
            if g_new.shape != g_old.shape:
                print(f'FAIL {name} {dtype}: shape new {tuple(g_new.shape)} vs old {tuple(g_old.shape)}')
                ok = False
                continue
            r1 = rel(g_new, g_old)
            r2 = 0.0
            if dtype == torch.float32:      # projected path is float32-only (its m is allocated float32; unchanged code)
                v = torch.randn(pot.num_coefficients, dtype=dtype, device=device)
                g_v = pot.grad(x, v)                                              # (B, 1, M), projected path
                r2 = rel(torch.einsum('k,bkm->bm', v, g_new)[:, None], g_v)
            passed = r1 < TOL[dtype] and r2 < TOL[dtype]
            ok &= passed
            print(f'{"PASS" if passed else "FAIL"} {name:45s} {str(dtype):14s} K {pot.num_coefficients:3d} | '
                  f'new vs old {r1:.2e} | sum v.grad vs grad(v) {r2:.2e} | tol {TOL[dtype]:.0e}')

    pots = build(args.M, args.J, torch.float32, device)
    x = torch.randn(args.B_time, 1, args.M, device=device)
    print(f'\ntiming, float32, B {args.B_time}, median of {args.n_rep}:')
    for name, pot in pots.items():
        dense = dense_imag if 'Imag' in name else dense_real
        t_old = timed(lambda: dense(pot, x), args.n_rep)
        t_new = timed(lambda: pot.grad(x), args.n_rep)
        print(f'  {name:45s} old {1e3 * t_old:8.2f} ms | new {1e3 * t_new:8.2f} ms | ratio {t_old / t_new:5.2f}')
    print('\nALL PASS' if ok else '\nSOME FAILED')


if __name__ == '__main__':
    main()
