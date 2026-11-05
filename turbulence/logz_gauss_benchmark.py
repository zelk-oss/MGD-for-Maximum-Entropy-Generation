#!/usr/bin/env python3
"""Benchmark of the three log Z estimates on the Gaussian-with-data-spectrum runs (gauss_spectrum/README.md).

Model = Scattering_Second_Order_Q1 only, so for any Theta
    q(x) = Theta . phi(x) = (1/M^2) sum_w s(w) |xhat(w)|^2 ,   s(w) = sum_j Theta_j |psi_j(w)|^2   (psi2 = |psi_j|^2, float64)
is a quadratic form and p_Theta = exp(q)/Z is an exact stationary Gaussian whenever lam(w) = -(s(w) + s(-w))/M > 0 at
every w (these are the eigenvalues of the precision P = -2B; same exact test as logp_gauss_scat2.existence). Then, EXACTLY
and in float64 (no matrix, no eigendecomposition: v2 of this script, v1 built B in float32 and its small eigenvalues were
noise for a spectrum range of 3.6e8):
    log Z_Theta = (M/2) log 2 pi - (1/2) sum_w log lam(w) ,     H(p_Theta) = log Z_Theta + M/2 ,
    model spectrum S_Theta(w) = M / lam(w) ,                   model moments m_Theta,j = (1/M^2) sum_w psi2_j(w) S_Theta(w).
The formula for q is checked against the model's own phi on random x (quad_rel_err).

The three estimates of log Z (code sign: p = exp(+Theta . phi) / Z), compared with the exact value FOR THE SAME THETA:
    A  Theta_reg inside the entropy integral:  H(p_0) - int Theta_reg . dm  + Theta_reg[-1] . m_1   (h_from_theta_reg)
    B  theta_k inside the integral, Theta_reg[-1] at the end:  H_bound(theta_k) + Theta_reg[-1] . m_1  (logp_entropy.py)
    C  theta_k in both:  H_bound(theta_k) + theta_t[-1] . m_1   (utils_entropy.log_Z_bound, theta_key='theta_t')
A, B vs the exact log Z of Theta_reg[-1]; C vs that of theta_t[-1] (nan if lam <= 0 somewhere: not a density).

Also, per seed:
  - H of the Gaussian with the GENERATED samples' spectrum (periodogram average): the entropy the sampler's law has if it
    is Gaussian-stationary. Compare with H(p_Theta_reg) (max-ent ceiling for the model's moments) and H_bound.
  - NODE SCAN over every Theta_reg node t_k: exact log Z_k, A_k, B_k, errors A_k - log Z_k (and relative to node 0, to
    isolate what accrues along the transport), and the moment mismatch ||m_Theta_k - m_k|| / ||m_k|| (does Theta_reg(t_k)
    reproduce the target moments m_k = barphi_e? this is the moment-matching hypothesis, tested exactly). Saved per seed
    to <outdir>/scan_seed<seed>_lam<lam>.npz.

m_t at the nodes: the run's aux_moments (barphi_e) if saved, else recomputed from the data (effective_dimension).

Usage (JZ, repo root; CPU is enough):
  python turbulence/logz_gauss_benchmark.py --group_dir <group> --results_root <turbulence|jets> \
      --theta_reg_dir <results_root>/saved_results/lagrange_multipliers_regularised --lams 5e-4 --ridge 1e-4 --outdir <dir>
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / 'codes', HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import logp_gauss_scat2 as lg                                                    # noqa: E402  (loaders, true_log_p)
from utils_entropy import entropy_bound, standard_gaussian_entropy               # noqa: E402
from codes.effective_dimension import h_bound_at, h_from_theta_reg, load_or_recompute_m_t   # noqa: E402


def _pt(path):
    path = Path(path)
    return torch.load(path if path.exists() else path.with_suffix(''), map_location='cpu', weights_only=False)


def fourier_exact(theta, psi2, M):
    """Exact Gaussian quantities of p ~ exp(theta . phi); theta (..., r) float64, psi2 (r, M) float64.
    Entries are nan where the precision is not positive definite (some lam(w) <= 0). Returns dict."""
    theta = np.asarray(theta, dtype=np.float64)
    s = theta @ psi2                                                  # (..., M)
    neg = (-np.arange(M)) % M
    lam = -(s + s[..., neg]) / M                                      # eigenvalues of P = -2B, (..., M)
    ok = (lam > 0).all(-1)
    lam_s = np.where(ok[..., None], lam, np.nan)
    logZ = 0.5 * M * np.log(2 * np.pi) - 0.5 * np.log(lam_s).sum(-1)
    S = M / lam_s
    return dict(logZ=logZ, H=logZ + M / 2, min_lam=lam.min(-1), S=S, m=S @ psi2.T / M ** 2, ok=ok)


def check_quadratic(theta, psi2, pots, M, device):
    """max |Theta.phi(x) (model) - (1/M^2) sum_w s |xhat|^2| / std on random x."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(300, 1, M, generator=g)
    q = lg.theta_phi(x, torch.as_tensor(theta, dtype=torch.float64), pots, 300, device)
    xh = torch.fft.fft(x[:, 0].double(), dim=-1)
    qF = ((theta @ psi2)[None, :] * (xh.real ** 2 + xh.imag ** 2).numpy()).sum(1) / M ** 2
    return float(np.abs(q - qF).max() / q.std())


def one_seed(exp_dir, lam, args, x_data, H_data):
    cfg = json.loads((exp_dir / 'config.json').read_text()); cfg = cfg.get('args', cfg)
    name, seed, M = exp_dir.name, int(cfg['seed']), int(cfg['M'])
    sr = args.results_root / 'saved_results'
    sweep_root = args.theta_reg_dir or sr / 'lagrange_multipliers_regularised'
    cdir = sweep_root / name
    files = sorted(f for f in cdir.glob('lam*_ridge*.pt') if 'neg' not in f.name)
    if lam is not None:
        files = [f for f in files if f.name.startswith(f'lam{lam:g}_')]
    if args.ridge is not None:
        files = [f for f in files if f'_ridge{args.ridge:g}' in f.name]
    if len(files) != 1:
        raise RuntimeError(f'{len(files)} Theta_reg files in {cdir} (pin --lams/--ridge): {[f.name for f in files]}')
    sol = torch.load(files[0], map_location='cpu', weights_only=False)
    Theta, t_reg = sol['Theta_reg'].double(), sol['t_reg'].double().numpy()
    th_reg = Theta[-1].numpy()

    t = _pt(sr / 'sampling_times' / f'{name}.pt').double().numpy()
    dH = _pt(sr / 'entropy_bounds' / f'{name}.pt').double().numpy()
    th_t = _pt(sr / 'lagrange_multipliers' / f'{name}.pt')
    th_t = (th_t if th_t.dim() == 2 else th_t['theta_t'])[-1].double().numpy()
    pots = lg.build_pots(cfg, args.device)
    psi2 = lg.filters_of(pots)
    if psi2 is None:
        raise RuntimeError(f'model is not Scattering_Second_Order_Q1 only ({list(pots)}): the Fourier formula does not apply')

    m_nodes, m_src = load_or_recompute_m_t(sr / 'aux_moments' / f'{name}_aux_moments.pt', t, t_reg, pots, x_data)
    m_nodes = m_nodes.double()
    m1 = m_nodes[-1].numpy()
    if abs(t_reg[-1] - t[-1]) > 1e-12:
        print(f'  WARNING t_reg[-1]={t_reg[-1]:.10f} != t[-1]={t[-1]:.10f}: C uses theta_t[-1] at t[-1] with m at t_reg[-1]')

    H_p0 = standard_gaussian_entropy(M)
    H_cons_all = h_from_theta_reg(Theta, m_nodes, M)
    H_pipe_all = h_bound_at(t, dH, t_reg, M)
    H_cons, H_pipe = float(H_cons_all[-1]), float(H_pipe_all[-1])
    eb = entropy_bound({name: dict(xt=torch.zeros(1, 1, M), dH_t_bound=torch.as_tensor(dH), t=torch.as_tensor(t))},
                       name, pots, check_p0=False)
    H_bound = float(eb['H_bound'])
    A = H_cons + float(th_reg @ m1)
    B_ = H_pipe + float(th_reg @ m1)
    C = H_bound + float(th_t @ m1)

    rel = check_quadratic(th_reg, psi2, pots, M, args.device)
    ex_reg, ex_t = fourier_exact(th_reg, psi2, M), fourier_exact(th_t, psi2, M)
    Z_reg, H_reg, Z_t = float(ex_reg['logZ']), float(ex_reg['H']), float(ex_t['logZ'])

    # generated samples: Gaussian-stationary entropy from their periodogram average
    xt = lg.load_samples(args.results_root, name).reshape(-1, M).double()
    S_gen = (torch.fft.fft(xt, dim=-1).abs() ** 2).mean(0).numpy()
    H_gen = float(0.5 * np.log(2 * np.pi * np.e * S_gen / M).sum())

    # ---- node scan
    Th = Theta.numpy()
    mn = m_nodes.numpy()
    fk = fourier_exact(Th, psi2, M)
    thm = (Th * mn).sum(1)
    A_k, B_k = H_cons_all + thm, H_pipe_all + thm
    errA, errB = A_k - fk['logZ'], B_k - fk['logZ']
    m_rel = np.linalg.norm(fk['m'] - mn, axis=1) / np.linalg.norm(mn, axis=1)
    resid_k = thm + M / 2
    np.savez(args.outdir / f'scan_seed{seed}_lam{float(sol["lam"]):g}.npz', t_reg=t_reg, logZ_exact=fk['logZ'], H_exact=fk['H'],
             A_k=A_k, B_k=B_k, errA=errA, errB=errB, m_rel=m_rel, moment_residual=resid_k, min_lam=fk['min_lam'],
             H_cons=H_cons_all, H_pipe=H_pipe_all)

    row = dict(seed=seed, lam=float(sol['lam']), m_source=m_src, M=M,
               quad_rel_err=rel, min_lam_reg=float(ex_reg['min_lam']), min_lam_theta_t=float(ex_t['min_lam']),
               logZ_exact_reg=Z_reg, A=A, B=B_, A_minus_exact=A - Z_reg, B_minus_exact=B_ - Z_reg,
               logZ_exact_theta_t=Z_t, C=C, C_minus_exact=C - Z_t,
               H_exact_reg=H_reg, H_bound=H_bound, H_bound_minus_Hreg=H_bound - H_reg,
               H_data=H_data, H_bound_minus_Hdata=H_bound - H_data, H_gen_gauss=H_gen, H_bound_minus_Hgen=H_bound - H_gen,
               Hreg_minus_Hdata=H_reg - H_data, H_cons=H_cons, H_pipe=H_pipe,
               moment_residual=float(th_reg @ m1 + M / 2), moment_residual_theta_t=float(th_t @ m1 + M / 2),
               m_rel_end=float(m_rel[-1]), nodes_not_normalisable=int((~fk['ok']).sum()), nodes=len(t_reg))
    print(f'\nseed {seed}, lam {row["lam"]:g}  ({files[0].name}; m_t from {m_src})')
    print(f'  q(x) Fourier formula vs model phi: max rel err {rel:.2e}')
    print(f'  normalisable (min lam(w), float64): Theta_reg {row["min_lam_reg"]:+.3e}, theta_t[-1] {row["min_lam_theta_t"]:+.3e}; '
          f'Theta_reg nodes not normalisable: {row["nodes_not_normalisable"]}/{row["nodes"]}')
    print(f'  Theta_reg[-1]: exact log Z {Z_reg:10.3f} | A {A:10.3f} ({A - Z_reg:+.3f}) | B {B_:10.3f} ({B_ - Z_reg:+.3f})')
    print(f'  theta_t[-1]  : exact log Z {Z_t:10.3f} | C {C:10.3f} ({C - Z_t:+.3f})')
    print(f'  entropy: H_bound {H_bound:.3f} | H(p_Theta_reg) exact {H_reg:.3f} | data Gaussian {H_data:.3f} | '
          f'generated (Gaussian fit to their spectrum) {H_gen:.3f}')
    print(f'           H_bound - H(p_Theta_reg) {H_bound - H_reg:+.3f} | H_bound - H_data {H_bound - H_data:+.3f} | '
          f'H_bound - H_gen {H_bound - H_gen:+.3f} | H(p_Theta_reg) - H_data {H_reg - H_data:+.3f}')
    print(f'  moment residual Theta_reg.m_1 + M/2 = {row["moment_residual"]:+.3f} | relative ||m_Theta - m_1||/||m_1|| at t_end {m_rel[-1]:.3e}')
    want = [0.0, 1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0]
    idx = sorted({min(int(np.searchsorted(t_reg, w)), len(t_reg) - 1) for w in want} | {len(t_reg) - 1})
    print('  node scan (exact log Z_k from Theta_reg[k]):')
    print('      t_k      log Z_k    A_k-exact  B_k-exact  (A_k-exact) - node0   ||m_Theta-m||/||m||   Theta.m+M/2   min lam')
    e0 = errA[0]
    for k in idx:
        print(f'   {t_reg[k]:9.5f} {fk["logZ"][k]:11.3f} {errA[k]:11.3f} {errB[k]:10.3f} {errA[k] - e0:18.3f} {m_rel[k]:20.3e} '
              f'{resid_k[k]:13.3f} {fk["min_lam"][k]:10.2e}')
    if rel > 1e-3:
        print('  !! Fourier formula does not reproduce the model phi: exact values invalid')
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group_dir', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--results_root', type=Path, default=ROOT / 'turbulence')
    p.add_argument('--theta_reg_dir', type=Path, default=None)
    p.add_argument('--lams', type=float, nargs='*', default=[None])
    p.add_argument('--ridge', type=float, default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    seed_dirs = sorted(d for d in args.group_dir.iterdir() if (d / 'config.json').exists())
    cfg0 = json.loads((seed_dirs[0] / 'config.json').read_text()); cfg0 = cfg0.get('args', cfg0)
    x_data = lg.load_data_x1(cfg0, args.device, seed_dirs[0])
    S, _ = lg.load_gauss_spectrum(seed_dirs[0])
    nl = -lg.true_log_p(x_data.cpu(), S).numpy()
    H_data = float(nl.mean())
    print(f'{len(seed_dirs)} seeds; data Gaussian entropy (MC over {len(nl)} signals): {H_data:.3f} +- {nl.std() / np.sqrt(len(nl)):.3f}')
    rows = []
    for lam in args.lams:
        for d in seed_dirs:
            try:
                rows.append(one_seed(d, lam, args, x_data, H_data))
            except Exception as e:
                print(f'\n{d.name}: FAIL {type(e).__name__}: {e}')
    if not rows:
        return
    print('\n=== mean +- std over seeds, per lambda (errors = estimate - exact, same theta) ===')
    for lam in sorted({r['lam'] for r in rows}):
        sub = [r for r in rows if r['lam'] == lam]
        def f(k):
            v = np.array([r[k] for r in sub], float)
            return f'{np.nanmean(v):+9.3f} +- {np.nanstd(v):.3f}' if np.isfinite(v).any() else '      nan'
        print(f'lam {lam:g} ({len(sub)} seeds): A-exact {f("A_minus_exact")} | B-exact {f("B_minus_exact")} | '
              f'C-exact {f("C_minus_exact")}\n   H_bound-H(p_Theta_reg) {f("H_bound_minus_Hreg")} | H_bound-H_data {f("H_bound_minus_Hdata")} | '
              f'H_bound-H_gen {f("H_bound_minus_Hgen")} | H(p_Theta_reg)-H_data {f("Hreg_minus_Hdata")} | moment residual {f("moment_residual")}')
    with open(args.outdir / 'logz_benchmark.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f'\nwrote {args.outdir / "logz_benchmark.csv"} and scan_seed*_lam*.npz')


if __name__ == '__main__':
    main()
