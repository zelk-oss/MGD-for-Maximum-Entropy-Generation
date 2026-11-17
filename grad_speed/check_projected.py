"""Check of B3 (2026-11-17): projected gradient sum_k v_k grad phi_k(x) (drift and corrector) computed by summing inside
each channel BEFORE the inverse FFT (linearity), for Scalar_GGD_KRegion, Scattering_Second_Order_1d, L2p_norm (L_6).

Potential sets: B+Scat4 turbulence (real GGD fits of seed 900) and jets 'noscalar', J = 8, M = 256, random x and v.
1. Per changed potential: new and old (verbatim copy) in float32 against a float64 reference (old formula, float64
   inputs and filters): max |diff| / max |ref|. Pass: both < 1e-4. Plus new vs old directly.
2. The whole projected field (SDE.compute_grad_potentials(x, v), all potentials of the set) new vs old-composed.
3. Wall time old vs new per changed potential and for the whole field at B_time walkers.

Usage (repo root, JZ dev GPU job): python -u grad_speed/check_projected.py [--B 64 --B_time 5000]
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_jvp import build, timed, TURB, JETS, M, J, SDE      # noqa: E402
from check_ggd_prealloc import old_ggd_grad                    # noqa: E402

TOL = 1e-4
CHANGED = ('L_6', 'Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK', 'Scattering_Second_Order')


# ---------------------------------------------------------------- old projected paths (verbatim, commit 6733c36)

def old_scat2_v(self, x, v):
    filters = self.filters.to(x.device)
    x_filtered_2 = torch.fft.ifft(torch.fft.fft(x)*(filters*filters.conj()))
    output = x_filtered_2.real.reshape(x_filtered_2.shape[:1]+(-1,x.shape[-1]))
    output = output/self.norm**2
    return (2*output*v[None,:,None]).sum(1)[:,None]/x.shape[-1]


def old_l2p_v(self, x, v):
    filters = self.filters.to(x.device)
    x_filtered = torch.fft.ifft(filters*torch.fft.fft(x))
    output = (x_filtered *(2* self.p) * (torch.abs(x_filtered)**(2*self.p-2)))
    output = torch.fft.ifft(filters.conj()*torch.fft.fft(output)).real
    output = output / self.norm
    return (output*v[None,:,None]).sum(1)[:,None]/x.shape[-1]


def old_v(name, pot, x, v):
    if name.startswith('Scalar_'):
        return old_ggd_grad(pot, x, v)
    if name == 'Scattering_Second_Order':
        return old_scat2_v(pot, x, v)
    if name == 'L_6':
        return old_l2p_v(pot, x, v)
    return pot.grad(x, v=v)                                   # unchanged potentials


def field(pots, x, vec, old):
    out, off = torch.zeros_like(x), 0
    for name, pot in pots.items():
        n = int(pot.num_coefficients)
        out += old_v(name, pot, x, vec[off:off + n]) if old else pot.grad(x, v=vec[off:off + n])
        off += n
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--B', type=int, default=64)
    p.add_argument('--B_time', type=int, default=5000)
    p.add_argument('--n_rep', type=int, default=5)
    p.add_argument('--M', type=int, default=M)
    p.add_argument('--J', type=int, default=J)
    args = p.parse_args()
    if (args.M, args.J) != (M, J):
        raise ValueError(f'the seed-900 fits are for M {M}, J {J}; got M {args.M}, J {args.J}')
    device = 'cuda'
    torch.manual_seed(0)
    print(f'GPU {torch.cuda.get_device_name()}; torch {torch.__version__}')
    ok = True

    for label, terms in (('B+Scat4 (turbulence)', TURB), ('jets noscalar', JETS)):
        pots, pots64 = build(terms, device), build(terms, device, f64=True)
        sizes = {k: int(v.num_coefficients) for k, v in pots.items()}
        r = sum(sizes.values())
        print(f'\n=== {label}: {sizes}, r = {r}')
        x = torch.randn(args.B, 1, M, device=device)
        vec = torch.randn(r, device=device)
        off = 0
        for name, n in sizes.items():
            v = vec[off:off + n]
            off += n
            if name not in CHANGED:
                continue
            ref = old_v(name, pots64[name], x.double(), v.double())
            new, old = pots[name].grad(x, v=v), old_v(name, pots[name], x, v)
            s = ref.abs().max().item()
            e_new = (new.double() - ref).abs().max().item() / s
            e_old = (old.double() - ref).abs().max().item() / s
            d = (new - old).abs().max().item() / s
            good = e_new < TOL and e_old < TOL and new.shape == old.shape
            ok &= good
            print(f'  {"PASS" if good else "FAIL"} {name:26s} K {n:3d} | vs float64 ref: new {e_new:.1e}, old {e_old:.1e}'
                  f' | new vs old {d:.1e} | shape {tuple(new.shape)}')

        ns = SimpleNamespace(potentials=pots, num_potentials=r, signal_dim=1, indices_potentials=
                             [0] + torch.tensor(list(sizes.values())).cumsum(0).tolist())
        f_new = SDE.compute_grad_potentials(ns, x, vec)
        f_old = field(pots, x, vec, old=True)
        d = ((f_new - f_old).abs().max() / f_old.abs().max()).item()
        good = d < TOL
        ok &= good
        print(f'  {"PASS" if good else "FAIL"} whole projected field (all potentials): new vs old {d:.1e}')

        xt = torch.randn(args.B_time, 1, M, device=device)
        print(f'  timing, B {args.B_time}, median of {args.n_rep}:')
        off = 0
        for name, n in sizes.items():
            v = vec[off:off + n]
            off += n
            if name not in CHANGED:
                continue
            t_old = timed(lambda: old_v(name, pots[name], xt, v), args.n_rep)
            t_new = timed(lambda: pots[name].grad(xt, v=v), args.n_rep)
            print(f'    {name:26s} old {1e3 * t_old:7.2f} ms | new {1e3 * t_new:7.2f} ms | ratio {t_old / t_new:5.2f}')
        t_old = timed(lambda: field(pots, xt, vec, old=True), args.n_rep)
        t_new = timed(lambda: field(pots, xt, vec, old=False), args.n_rep)
        print(f'    {"whole field":26s} old {1e3 * t_old:7.2f} ms | new {1e3 * t_new:7.2f} ms | ratio {t_old / t_new:5.2f}')
        del pots, pots64, xt
        torch.cuda.empty_cache()

    print('\nALL PASS' if ok else '\nSOME FAILED')


if __name__ == '__main__':
    main()
