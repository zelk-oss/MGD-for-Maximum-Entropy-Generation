"""RECORD ONLY: ran (job 341922) against the JVP version of SDE.compute_rhs_dt_phi_I_t, which was NOT adopted and has been
removed (it needs SDE._rhs_dt_phi_I_t_jacobian, gone too). Results: JVP as accurate as the Jacobian contraction
(both within 3e-5 of a float64 reference), 1.24x faster on B+Scat4, 0.93x on jets. The Scat4 Mod2 forward on the
kept triples (bit-identical, part 1 below) was adopted.

Check of B1 (2026-11-17): moment velocity d/dt phi_bar(I_t) = (1/N) sum_n <grad phi(I_n), Idot_n> by forward-mode AD
(torch.func.jvp) instead of contracting the full (B, r, M) Jacobian; and the Scat4 Mod2 forward on the kept triples only.

Potential sets: B+Scat4 turbulence (real GGD fits of seed 900) and jets 'noscalar' (L_6, Scat1, Scat2, Scat3 Re/Im,
Scat4 Re/Im), J = 8, M = 256, random I and Idot.
1. Scat4 Mod2 forward, new vs old (verbatim copy): max |diff| / max |old|, and whether bit-identical.
2. rhs, per potential block: new (JVP, float32) and old (Jacobian, float32) against a float64 reference (Jacobian path,
   float64 inputs and filters): which of the two float32 methods is closer. Plus new vs old directly.
   Pass: every block of both methods within 1e-4 relative of the reference (float32 round-off over ~1e6 terms).
3. Whether any potential fell back to the Jacobian path (forward-mode AD unsupported).
4. Wall time and peak GPU memory of old vs new rhs at N = 2 * B_time walkers in batches of B_time (as production).

Usage (repo root, JZ dev GPU job): python -u grad_speed/check_jvp.py [--B 64 --B_time 5000]
"""
import argparse
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_ggd_prealloc import ROOT, RUN, M, J, Q, return_Filters, get_1d_potentials, Scalar_GGD_KRegion, SDE  # noqa
from potentials.utils_potentials import abs_eps                                                                # noqa

TURB = ['L_6', 'Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK',
        'Scattering_Fourth_Order_Mod2_Real_Q1', 'Scattering_Fourth_Order_Mod2_Imag_Q1']
JETS = ['L_6', 'Scattering_First_Order', 'Scattering_Second_Order', 'Scattering_Third_Order_Real',
        'Scattering_Third_Order_Imag', 'Scattering_Fourth_Order_Mod2_Real_Q1', 'Scattering_Fourth_Order_Mod2_Imag_Q1']
TOL = 1e-4


# ---------------------------------------------------------------- old Scat4 Mod2 forward (verbatim, commit abab351)

def old_scat4_forward(self, x, imag):
    filters = self.filters.to(x.device)
    filters_Q = self.filters_Q.to(x.device)
    x_filtered = torch.fft.ifft(filters_Q*torch.fft.fft(x))
    x_filtered = x_filtered/self.norm
    x_filtered_abs2 = abs_eps(x_filtered)**2
    W_Wx = torch.fft.ifft(filters[:, :, None] * torch.fft.fft(x_filtered_abs2)[:, None])
    prod = W_Wx[:, :, :, None] * W_Wx[:, :, None].conj()
    output = (torch.imag(prod) if imag else torch.real(prod)).mean(-1)
    output = output.permute(0, 3, 2, 1)
    indices = self.indices.long()
    output = output[:, indices[0], indices[1], indices[2]]
    return output.reshape(x.shape[0], indices.shape[1])


# ---------------------------------------------------------------- helpers

def build(terms, device, f64=False):
    cast = (lambda f: f.to(torch.complex128) if f.is_complex() else f.double()) if f64 else (lambda f: f)
    filters = cast(return_Filters(M, J, 1, device=device))
    if terms is TURB:
        filters_Q = cast(return_Filters(M, J, Q, device=device))
        pots = get_1d_potentials(terms, J, filters, Q, filters_Q=filters_Q, deduplicate_filters=True, dedup_tol=1e-8)
        for name in ('Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK'):
            pots[name] = Scalar_GGD_KRegion.load_fixed_parameters(ROOT / RUN / f'{name}.pt', pots[name].filters,
                                                                  map_location=device).to(device)
    else:
        pots = get_1d_potentials(terms, J, filters, 1, filters_Q=filters)
    return pots


def sde_like(pots, Idot, batch_size, device):
    ns = SimpleNamespace(potentials=pots, num_potentials=sum(int(p.num_coefficients) for p in pots.values()),
                         signal_dim=1, device=device, batch_size=batch_size)
    ns.gradient_interpolant = lambda k: Idot
    ns.compute_grad_potentials = lambda x, vector=None: SDE.compute_grad_potentials(ns, x, vector)
    return ns


def peak(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30


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
        pots = build(terms, device)
        sizes = {k: int(v.num_coefficients) for k, v in pots.items()}
        print(f'\n=== {label}: {sizes}, r = {sum(sizes.values())}')
        I = torch.randn(args.B, 1, M, device=device)
        Idot = torch.randn(args.B, 1, M, device=device)

        for name in ('Scattering_Fourth_Order_Mod2_Real_Q1', 'Scattering_Fourth_Order_Mod2_Imag_Q1'):
            pot = pots[name]
            new, old = pot.forward(I), old_scat4_forward(pot, I, imag='Imag' in name)
            r = ((new - old).abs().max() / old.abs().max()).item()
            ok &= r < 1e-6
            print(f'  {"PASS" if r < 1e-6 else "FAIL"} forward {name}: new vs old {r:.2e}, '
                  f'bit-identical {torch.equal(new, old)}')

        ns = sde_like(pots, Idot, args.B // 2, device)          # two batches, as in production
        rhs_new = SDE.compute_rhs_dt_phi_I_t(ns, I, 0)[:, 0]
        rhs_old = SDE._rhs_dt_phi_I_t_jacobian(ns, I, 0)[:, 0]
        fell_back = [k for k, v in ns.__dict__.get('_jvp_ok', {}).items() if not v]
        pots64 = build(terms, device, f64=True)
        ns64 = sde_like(pots64, Idot.double(), args.B // 2, device)
        ns64.num_potentials = ns.num_potentials
        ref = torch.cat([(pt.grad(I.double()).reshape(args.B, -1, M) * Idot.double()).sum(-1).sum(0)
                         for pt in pots64.values()]) / args.B
        off = 0
        for name, n in sizes.items():
            sl = slice(off, off + n)
            off += n
            scale = ref[sl].abs().max().item()
            e_new = (rhs_new[sl].double() - ref[sl]).abs().max().item() / scale
            e_old = (rhs_old[sl].double() - ref[sl]).abs().max().item() / scale
            d = (rhs_new[sl] - rhs_old[sl]).abs().max().item() / scale
            good = e_new < TOL and e_old < TOL
            ok &= good
            print(f'  {"PASS" if good else "FAIL"} rhs {name:40s} K {n:3d} | vs float64 ref: JVP {e_new:.1e}, '
                  f'Jacobian {e_old:.1e} | JVP vs Jacobian {d:.1e}')
        print(f'  fell back to the Jacobian path: {fell_back or "none"}')
        ok &= not fell_back

        It = torch.randn(2 * args.B_time, 1, M, device=device)
        Idt = torch.randn(2 * args.B_time, 1, M, device=device)
        nst = sde_like(pots, Idt, args.B_time, device)
        t_old = timed(lambda: SDE._rhs_dt_phi_I_t_jacobian(nst, It, 0), args.n_rep)
        t_new = timed(lambda: SDE.compute_rhs_dt_phi_I_t(nst, It, 0), args.n_rep)
        m_old = peak(lambda: SDE._rhs_dt_phi_I_t_jacobian(nst, It, 0))
        m_new = peak(lambda: SDE.compute_rhs_dt_phi_I_t(nst, It, 0))
        print(f'  timing N {2 * args.B_time} (batches of {args.B_time}), median of {args.n_rep}: Jacobian '
              f'{1e3 * t_old:.1f} ms, JVP {1e3 * t_new:.1f} ms, ratio {t_old / t_new:.2f} | peak GPU memory '
              f'{m_old:.2f} -> {m_new:.2f} GiB')
        del pots, pots64, It, Idt
        torch.cuda.empty_cache()

    print('\nALL PASS' if ok else '\nSOME FAILED')


if __name__ == '__main__':
    main()
