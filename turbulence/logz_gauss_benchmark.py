#!/usr/bin/env python3
"""Entropy check on the Gaussian-with-data-spectrum runs (gauss_spectrum/README.md). Nats, code sign p = exp(+Theta.phi)/Z.

The data law is a stationary Gaussian and the interpolant is Cos, so every marginal p_t is Gaussian and its entropy is
exact: H(p_t) = codes/utils_entropy.compute_gaussian_entropy(x1, interpolant, t). The model is Scattering_Second_Order_Q1
only, so p_Theta ~ exp(Theta . phi) is a Gaussian too and its log Z is exact (fourier_exact, checked against the model phi).

Per seed:
  1. entropy along the transport: MGD bound H_bound(t) = H(p_0) + int_0^t dH_s (the run's dH_t_bound, theta_k inside the
     integral; codes/effective_dimension.h_bound_at) vs exact H(p_t), at the Theta_reg nodes; endpoint t_end = t_reg[-1].
  2. log Z: exact for Theta_reg[-1] (and for theta_t[-1]; nan if not a density) vs the MGD bound
         log Z_bound = H_bound(t_end) + theta_end . m(t_end),   theta_end = Theta_reg[-1]  or  theta_t[-1]
     (m(t_end) = the run's target moments at t_end, aux_moments).
  3. entropy as E[-log p] = log Z - mean(Theta_reg[-1] . phi(x)), x = data x1 (the Gaussian's own law) or the generated
     samples, log Z = exact / bound(Theta_reg) / bound(theta_t); compared with the exact H(p_1).
     With the exact log Z, E_data[-log p] - H(p_1) = KL(data || p_Theta_reg) >= 0: it tests Theta_reg alone.
  4. MGD paper Fig. 2(b) at t = 1: H(p*) of the exact max-ent Gaussian with the data moments (max_ent_gaussian, once per
     group), H(p_1^sigma) of the generated samples, and the bound H_sigma*. NB H(p_1) above is the DATA Gaussian (the
     interpolant at t=1), not p*: H(p*) - H(data) = KL(data || p*) >= 0 is the model error of the 9 band energies.

Usage (JZ, repo root; CPU is enough):
  python turbulence/logz_gauss_benchmark.py --group_dir <group> --results_root <root with saved_results/> --outdir <dir>
      [--theta_reg_dir <root>/saved_results/lagrange_multipliers_regularised] [--lam 5e-4] [--ridge 1e-4]
Outputs in --outdir: logz_benchmark.csv (one row per seed), path_seed<seed>.npz (t_reg, H_bound, H_exact).
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / 'codes', HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import logp_gauss_scat2 as lg                                                    # noqa: E402  (loaders, phi)
from utils_entropy import compute_gaussian_entropy                               # noqa: E402
from codes.effective_dimension import h_bound_at, load_or_recompute_m_t         # noqa: E402

LOGZ = ('exact_reg', 'bound_reg', 'bound_theta_t')


def _pt(path):
    path = Path(path)
    return torch.load(path if path.exists() else path.with_suffix(''), map_location='cpu', weights_only=False)


def fourier_exact_logZ(theta, psi2, M):
    """Exact log Z of p ~ exp(theta . phi) for the order-2 model: theta . phi(x) = (1/M^2) sum_w s(w)|xhat(w)|^2,
    s = theta @ psi2; precision eigenvalues lam(w) = -(s(w) + s(-w))/M; log Z = (M/2) log 2 pi - 1/2 sum log lam.
    nan if some lam(w) <= 0 (not a density). float64."""
    s = np.asarray(theta, dtype=np.float64) @ psi2
    lam = -(s + s[(-np.arange(M)) % M]) / M
    if (lam <= 0).any():
        return np.nan
    return float(0.5 * M * np.log(2 * np.pi) - 0.5 * np.log(lam).sum())


def max_ent_gaussian(x, psi2, M, tol=1e-10, max_iter=300):
    """Exact maximum-entropy density p* with the moments m = mean phi(x) of the order-2 Q1 model (MGD paper Fig. 2(b),
    red line). p_theta ~ exp(theta . phi) is Gaussian, diagonal in the ortho Fourier basis with precision
    lam(w) = -theta . A(w), A_j(w) = (|psi_j(w)|^2 + |psi_j(-w)|^2) / M (same algebra as fourier_exact_logZ), so
      E_theta phi_j = (1/M) sum_w |psi_j(w)|^2 / lam(w),   m_j = (1/M) sum_w |psi_j(w)|^2 C(w),  C = mean |FFT_ortho x|^2.
    theta* minimises the convex dual F = log Z - theta . m (gradient E_theta phi - m); damped Newton in float64 with
    Jacobi scaling (scaling by m_j stalls at cond ~1e20 on an 8-decade spectrum; tested 11-09). H(p*) = F(theta*) = (M/2) log(2 pi e) - 1/2 sum log lam*.
    Returns dict(H_star, iters, moment_rel_err, min_cover) -- H_star nan if not converged."""
    xf = np.fft.fft(x.reshape(-1, M).double().cpu().numpy(), axis=-1, norm='ortho')
    return max_ent_from_spectrum((np.abs(xf) ** 2).mean(0), psi2, M, tol=tol, max_iter=max_iter)


def max_ent_from_spectrum(C, psi2, M, tol=1e-10, max_iter=300):
    """max_ent_gaussian with the moments given through a power spectrum C (ortho FFT): m_j = (1/M) sum_w |psi_j|^2 C(w)."""
    C = np.asarray(C, dtype=np.float64)
    A = (psi2 + psi2[:, (-np.arange(M)) % M]) / M                                # (r, M)
    m = psi2 @ C / M
    D = np.abs(m)
    cover = A.sum(0)                                     # must be > 0 at every w for p* to exist (else lam(w) = 0)

    def F(th):
        lam = -th @ A
        return np.inf if (lam <= 0).any() else float(0.5 * M * np.log(2 * np.pi) - 0.5 * np.log(lam).sum() - th @ m)

    theta = -np.full(len(m), np.median(1.0 / (C * cover)))                       # lam = alpha * cover > 0
    err, it = np.inf, 0
    for it in range(max_iter):
        lam = -theta @ A
        g = 0.5 * A @ (1.0 / lam) - m                                            # E_theta phi - m
        err = float(np.max(np.abs(g) / D))
        if err < tol:
            break
        Hm = 0.5 * (A / lam ** 2) @ A.T                                          # Hessian = Cov_theta(phi)
        s = 1.0 / np.sqrt(np.diag(Hm))                                           # Jacobi scaling
        w, V = np.linalg.eigh(s[:, None] * Hm * s[None, :])
        step = -s * (V @ ((V.T @ (s * g)) / w))
        f0, a = F(theta), 1.0
        while F(theta + a * step) > f0 + 1e-4 * a * float(g @ step) and a > 1e-14:
            a *= 0.5
        theta = theta + a * step
    # at the optimum theta . m = E[theta . phi] = -M/2, so F(theta*) = (M/2) log(2 pi e) - 1/2 sum log lam*
    H = float(0.5 * M * np.log(2 * np.pi * np.e) - 0.5 * np.log(-theta @ A).sum()) if err < 1e-6 else np.nan
    return dict(H_star=H, iters=it, moment_rel_err=err, min_cover=float(cover.min() / cover.max()),
                spectrum_star=1.0 / (-theta @ A), spectrum_data=C)      # ortho-FFT power spectra of p* and of x


def check_quadratic(theta, psi2, pots, M, device):
    """max |Theta.phi(x) (model) - Fourier formula| / std on random x: the exact log Z is valid only if this is ~0."""
    x = torch.randn(300, 1, M, generator=torch.Generator().manual_seed(0))
    q = lg.theta_phi(x, torch.as_tensor(theta, dtype=torch.float64), pots, 300, device)
    xh = torch.fft.fft(x[:, 0].double(), dim=-1)
    qF = ((theta @ psi2)[None, :] * (xh.real ** 2 + xh.imag ** 2).numpy()).sum(1) / M ** 2
    return float(np.abs(q - qF).max() / q.std())


def one_seed(exp_dir, args, x_data, star):
    cfg = json.loads((exp_dir / 'config.json').read_text()); cfg = cfg.get('args', cfg)
    name, seed, M = exp_dir.name, int(cfg['seed']), int(cfg['M'])
    sr = args.results_root / 'saved_results'
    files = sorted(f for f in (args.theta_reg_dir / name).glob(f'lam{args.lam:g}_ridge{args.ridge:g}*.pt') if 'neg' not in f.name)
    if len(files) != 1:
        raise RuntimeError(f'{len(files)} Theta_reg files lam{args.lam:g}_ridge{args.ridge:g}* in {args.theta_reg_dir / name}')
    sol = torch.load(files[0], map_location='cpu', weights_only=False)
    t_reg = sol['t_reg'].double().numpy()
    th_reg = sol['Theta_reg'][-1].double().numpy()
    t = _pt(sr / 'sampling_times' / f'{name}.pt').double().numpy()
    dH = _pt(sr / 'entropy_bounds' / f'{name}.pt').double().numpy()
    th_t = _pt(sr / 'lagrange_multipliers' / f'{name}.pt')
    th_t = (th_t if th_t.dim() == 2 else th_t['theta_t'])[-1].double().numpy()
    pots = lg.build_pots(cfg, args.device)
    psi2 = lg.filters_of(pots)
    if psi2 is None:
        raise RuntimeError(f'model is not Scattering_Second_Order_Q1 only ({list(pots)}): exact log Z does not apply')
    rel = check_quadratic(th_reg, psi2, pots, M, args.device)

    # 1. entropy along the transport
    H_bound = h_bound_at(t, dH, t_reg, M)
    H_ex = compute_gaussian_entropy(x_data.reshape(-1, M).double().cpu(), cfg.get('interpolant') or 'Cos',
                                    torch.as_tensor(np.append(t_reg, 1.0))).numpy()
    H_path, H1 = H_ex[:-1], float(H_ex[-1])
    np.savez(args.outdir / f'path_seed{seed}.npz', t_reg=t_reg, H_bound=H_bound, H_exact=H_path, sigma=float(cfg['sigma']))

    # 2. log Z
    m_end, m_src = load_or_recompute_m_t(sr / 'aux_moments' / f'{name}_aux_moments.pt', t, t_reg[-1:], pots, x_data)
    m_end = m_end.double().numpy()[-1]
    Hb = float(H_bound[-1])
    logZ = dict(exact_reg=fourier_exact_logZ(th_reg, psi2, M), bound_reg=Hb + float(th_reg @ m_end),
                bound_theta_t=Hb + float(th_t @ m_end))
    logZ_exact_t = fourier_exact_logZ(th_t, psi2, M)

    # 3. E[-log p] = log Z - mean(Theta_reg . phi)
    xt = lg.load_samples(args.results_root, name).reshape(-1, 1, M)
    tp = {w: float(lg.theta_phi(x, torch.as_tensor(th_reg), pots, 500, args.device).mean())
          for w, x in (('data', x_data.reshape(-1, 1, M)), ('gen', xt))}

    # 4. MGD paper Fig. 2(b) at t = 1: H(p*) exact (star, same for every seed), H(p_1^sigma) of the generated samples
    #    (quadratic phi -> linear, translation-equivariant drift -> stationary Gaussian output -> 1/2 sum log 2 pi e C_gen),
    #    H_sigma* = bound integrated to the last step (not t_end = t_reg[-1]). Paper: H_sigma* <= H(p_1^sigma) <= H(p*).
    H_gen1 = float(compute_gaussian_entropy(xt.reshape(-1, M).double().cpu(), cfg.get('interpolant') or 'Cos',
                                            torch.tensor([1.0], dtype=torch.float64))[0])
    Hb1 = float(h_bound_at(t, dH, [1.0], M)[0])

    row = dict(seed=seed, sigma=float(cfg['sigma']), nt=int(cfg['nt']), t_end=float(t_reg[-1]), m_source=m_src,
               quad_rel_err=rel, H_exact_tend=float(H_path[-1]), H_exact_1=H1, H_bound=Hb,
               Hbound_minus_Hexact_tend=Hb - float(H_path[-1]),
               **{f'logZ_{k}': v for k, v in logZ.items()}, logZ_exact_theta_t=logZ_exact_t,
               logZ_bound_reg_minus_exact_reg=logZ['bound_reg'] - logZ['exact_reg'],
               logZ_bound_theta_t_minus_exact_reg=logZ['bound_theta_t'] - logZ['exact_reg'],
               logZ_bound_theta_t_minus_exact_theta_t=logZ['bound_theta_t'] - logZ_exact_t,
               **{f'Emlogp_{z}_{w}_minus_H1': logZ[z] - tp[w] - H1 for z in LOGZ for w in tp},
               H_star=star['H_star'], H_gen_1=H_gen1, H_bound_1=Hb1,
               Hstar_minus_Hgen=star['H_star'] - H_gen1, Hstar_minus_Hbound=star['H_star'] - Hb1,
               Hgen_minus_Hbound=H_gen1 - Hb1, Hstar_minus_Hdata=star['H_star'] - H1)
    print(f'\nseed {seed}, sigma {row["sigma"]:g}, nt {row["nt"]}, t_end {row["t_end"]:.6f} ({files[0].name}; m from {m_src})')
    print(f'  Fourier formula vs model phi: max rel err {rel:.1e}' + ('   !! exact log Z invalid' if rel > 1e-3 else ''))
    print(f'  entropy at t_end: H_bound {Hb:.3f} | exact H(p_t_end) {H_path[-1]:.3f} | diff {row["Hbound_minus_Hexact_tend"]:+.3f}'
          f'   (exact H(p_1) {H1:.3f})')
    print(f'  log Z: exact(Theta_reg) {logZ["exact_reg"]:.3f} | bound with Theta_reg {logZ["bound_reg"]:.3f} '
          f'({row["logZ_bound_reg_minus_exact_reg"]:+.3f}) | bound with theta_t {logZ["bound_theta_t"]:.3f} '
          f'({row["logZ_bound_theta_t_minus_exact_reg"]:+.3f}) | exact(theta_t) {logZ_exact_t:.3f}')
    print('  E[-log p] - H(p_1), log Z = ' + ' | '.join(
        f'{z}: data {row[f"Emlogp_{z}_data_minus_H1"]:+.3f}, gen {row[f"Emlogp_{z}_gen_minus_H1"]:+.3f}' for z in LOGZ))
    print(f'  Fig 2(b) at t=1: H(p*) {star["H_star"]:.3f} | H(p_1^sigma) gen {H_gen1:.3f} | H_sigma* {Hb1:.3f} | '
          f'H(data) {H1:.3f}   -> H(p*)-H(gen) {row["Hstar_minus_Hgen"]:+.3f}, H(p*)-H_sigma* '
          f'{row["Hstar_minus_Hbound"]:+.3f}, H(gen)-H_sigma* {row["Hgen_minus_Hbound"]:+.3f} (all >= 0 in the paper)')
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group_dir', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--results_root', type=Path, default=ROOT / 'turbulence')
    p.add_argument('--theta_reg_dir', type=Path, default=None)
    p.add_argument('--lam', type=float, default=5e-4)
    p.add_argument('--ridge', type=float, default=1e-4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    args.theta_reg_dir = args.theta_reg_dir or args.results_root / 'saved_results' / 'lagrange_multipliers_regularised'
    args.outdir.mkdir(parents=True, exist_ok=True)
    seed_dirs = sorted(d for d in args.group_dir.iterdir() if (d / 'config.json').exists())
    cfg0 = json.loads((seed_dirs[0] / 'config.json').read_text()); cfg0 = cfg0.get('args', cfg0)
    x_data = lg.load_data_x1(cfg0, args.device, seed_dirs[0])            # gauss_spectrum runs: redrawn from the saved S
    psi2_0 = lg.filters_of(lg.build_pots(cfg0, args.device))
    if psi2_0 is None:
        raise RuntimeError('model is not Scattering_Second_Order_Q1 only: exact max-ent Gaussian does not apply')
    star = max_ent_gaussian(x_data, psi2_0, int(cfg0['M']))
    print(f'max-ent p*: H(p*) {star["H_star"]:.3f} nats | Newton iters {star["iters"]}, max rel moment err '
          f'{star["moment_rel_err"]:.1e} | filter coverage min/max over w {star["min_cover"]:.1e}')
    rows = []
    for d in seed_dirs:
        try:
            rows.append(one_seed(d, args, x_data, star))
        except Exception as e:
            print(f'\n{d.name}: FAIL {type(e).__name__}: {e}')
    if not rows:
        return
    def f(k):
        v = np.array([r[k] for r in rows], float)
        return f'{np.nanmean(v):+9.3f} +- {np.nanstd(v):.3f}' if np.isfinite(v).any() else '      nan'
    print(f'\n=== {len(rows)} seeds, sigma {rows[0]["sigma"]:g}, nt {rows[0]["nt"]} (mean +- std over seeds, nats)')
    print(f'  H_bound - H(p_t_end) {f("Hbound_minus_Hexact_tend")}')
    print(f'  log Z bound - exact(Theta_reg): theta_end = Theta_reg {f("logZ_bound_reg_minus_exact_reg")} | '
          f'theta_end = theta_t {f("logZ_bound_theta_t_minus_exact_reg")}')
    for w in ('data', 'gen'):
        print(f'  E_{w}[-log p] - H(p_1): ' + ' | '.join(f'{z} {f(f"Emlogp_{z}_{w}_minus_H1")}' for z in LOGZ))
    print(f'  Fig 2(b): H(p*)-H(gen) {f("Hstar_minus_Hgen")} | H(p*)-H_sigma* {f("Hstar_minus_Hbound")} | '
          f'H(gen)-H_sigma* {f("Hgen_minus_Hbound")} | H(p*)-H(data) {f("Hstar_minus_Hdata")}')
    with open(args.outdir / 'logz_benchmark.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f'wrote {args.outdir / "logz_benchmark.csv"} and path_seed*.npz')


if __name__ == '__main__':
    main()
