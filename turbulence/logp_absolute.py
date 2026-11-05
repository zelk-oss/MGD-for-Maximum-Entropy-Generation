"""-log p_Theta on an ABSOLUTE axis (log Z included) and the entropy H(p_Theta), for one run, seed and Theta_reg.

Convention of the code: p_Theta(x) = exp(+Theta . phi(x)) / Z (paper: theta = -Theta, p = exp(-theta . phi) / Z).
Every potential has phi(0) = 0, so the energy has no free constant and
    -log p_Theta(x) = -Theta . phi(x) + log Z_Theta ,      H(p_Theta) = -Theta . m_Theta + log Z_Theta ,
    mean_data(-log p) - H = Theta . (m_Theta - m_data)          [paper: theta . (m - m_theta)]   <- the gap, reported below.

log Z for the SAME Theta that is plotted (Theta_reg[-1]):
  A. Theta_reg-consistent:  H_cons = H(p_0) - sum 1/2 (Theta_k + Theta_{k+1}) . (m_{k+1} - m_k)   (codes/effective_dimension.h_from_theta_reg)
                            log Z_cons = H_cons + Theta_reg[-1] . m_target
     Exact if p_Theta(t) has the moments m_t at every node; the MGD lower-bound proof (Prop. 4.3) covers the
     per-step theta_k that drove the walkers, NOT Theta_reg -> an estimate here, not a proven bound.
  B. Pipeline (codes/utils_entropy.log_Z_bound): H_bound from the per-step theta_k, plus Theta_reg[-1] . m_target.
     Mixes the two theta paths; printed for comparison (the earlier checks / logp_entropy.py used this).
  C. log_Z_bound with theta = theta_t[-1] (the sampler's own last step): the quantity the proof covers.
The three values are printed side by side; the plot uses A (--logz_source pipeline switches to B).

Normalizability of p_Theta (needs -Theta . phi(x) -> +inf along every ray):
  1. sign of the L_6 coefficients of Theta_reg[-1] (all must be < 0: L_6 = mean |W_j x|^6 is the highest degree);
  2. numerical rays x = r u, r = 1 ... R, for random, pure-tone and Dirac directions u (|u|_rms = 1): the energy
     -Theta . phi(r u) must be positive and growing at large r. Necessary, not sufficient (finite set of rays).

Usage (Jean Zay, repo root; GPU for the data phi matrix):
    python turbulence/logp_absolute.py --label turbulence_nested_M3_bicw_allmass_scat4 --seed 900 --lam 3e-5 [--unconstrained]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data', HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from logp_family_contrib import newest, load_potentials, phi_matrix        # noqa: E402
from check_fit_offline import build_data                                    # noqa: E402
from check_potentials import theta_column_map                               # noqa: E402
from codes.effective_dimension import h_bound_at, h_from_theta_reg          # noqa: E402
from codes.utils_entropy import entropy_bound, standard_gaussian_entropy    # noqa: E402

SR = HERE / 'saved_results'


def _pt(path):
    path = Path(path)
    return torch.load(path if path.exists() else path.with_suffix(''), map_location='cpu', weights_only=False)


def load_solution(config, lam, ridge, eps, unconstrained):
    """Whole Theta_reg solve (all nodes + t_reg) and a description. Same search order as logp_family_contrib.find_theta."""
    scratch = Path(os.environ.get('SCRATCH', '/nonexistent')) / 'MGD-for-Maximum-Entropy-Generation/turbulence'
    l6 = scratch / 'theta_reg_l6neg' / config
    cands = sorted(l6.glob(f'lam{lam:g}_ridge{ridge:g}_moment_mask_neg{eps:g}*.pt'), key=lambda f: f.stat().st_mtime) \
        if l6.exists() and not unconstrained else []
    if cands:
        f, kind = cands[-1], 'L_6<0 constrained'
    else:
        f, kind = SR / 'theta_reg_lamsweep' / config / f'lam{lam:g}_ridge{ridge:g}_moment_mask.pt', 'UNCONSTRAINED'
    sol = torch.load(f, map_location='cpu', weights_only=False)
    assert sol['config'] == config, (sol['config'], config)
    return sol, f'{kind} {f.name}'


# ------------------------------------------------------------------ normalizability ----
def ray_directions(M, n_random=100, n_dirac=16, seed=0):
    g = np.random.default_rng(seed)
    dirs, names = [], []
    for i in range(n_random):
        dirs.append(g.standard_normal(M)); names.append('random')
    n = np.arange(M)
    for k in range(1, M // 2 + 1):
        dirs.append(np.cos(2 * np.pi * k * n / M + g.uniform(0, 2 * np.pi))); names.append('tone')
    for pos in np.linspace(0, M, n_dirac, endpoint=False).astype(int):
        v = np.zeros(M); v[pos] = 1.0; dirs.append(v); names.append('dirac')
    U = np.stack(dirs)
    return U / U.std(1, keepdims=True), np.array(names)          # rms 1 per sample, like standardized data


def ray_test(pots, theta, M, device, radii):
    U, names = ray_directions(M)
    E = np.zeros((len(U), len(radii)))
    th = torch.as_tensor(theta, dtype=torch.float64)
    for j, r in enumerate(radii):
        x = torch.as_tensor(U * r, dtype=torch.float32, device=device).reshape(len(U), 1, M)
        phi = phi_matrix(pots, x)
        E[:, j] = -(phi @ th.numpy())                              # energy = -Theta . phi, E(0) = 0
    return E, names


def main(a):
    exp = newest(f'*/*_seed_{a.seed}_*{a.label}_2*', HERE / 'experiments')
    cfg = json.loads((exp / 'config.json').read_text())
    config = cfg['config_name']
    sol, desc = load_solution(config, a.lam, a.ridge, a.eps, a.unconstrained)
    Theta, t_reg = sol['Theta_reg'].double(), sol['t_reg'].double().numpy()
    theta = Theta[-1].numpy()
    print(f'run {config}\nTheta_reg: {desc}\n  nodes {len(t_reg)}, last t = {t_reg[-1]:.8f}, r = {len(theta)}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    xt = _pt(SR / 'samples' / f'{config}.pt').float()
    xt = xt.reshape(xt.shape[0], 1, -1)
    M = xt.shape[-1]
    d = M
    t = _pt(SR / 'sampling_times' / f'{config}.pt').double().numpy()
    dH = _pt(SR / 'entropy_bounds' / f'{config}.pt')
    theta_t = _pt(SR / 'lagrange_multipliers' / f'{config}.pt')
    barphi = _pt(SR / 'aux_moments' / f'{config}_aux_moments.pt')['barphi_e'].double()
    assert barphi.shape[0] == len(t) - 1, (barphi.shape, len(t))

    # ---- m_t at the Theta_reg nodes (exact subset of the fine grid, as deff_latest_run_check asserts)
    idx = np.clip(np.searchsorted(t[1:], t_reg), 0, len(t) - 2)
    off = np.abs(t[1:][idx] - t_reg).max()
    assert off < 1e-12, f't_reg is not a subset of t[1:] (max offset {off:.2e}): stop'
    m_nodes = barphi[idx]
    m_target = m_nodes[-1].numpy()

    pots = load_potentials(exp, cfg, device)
    ranges = {k: (s, s + w) for k, (s, w) in theta_column_map(pots, len(theta)).items()}
    x1, _ = build_data(int(cfg.get('subseries_len', 512)), int(cfg['M']), int(cfg['n1']), 1)
    phi_d = phi_matrix(pots, x1.to(device))
    phi_g = phi_matrix(pots, xt.to(device))
    m_data, m_gen = phi_d.mean(0), phi_g.mean(0)
    assert phi_d.shape[1] == len(theta), (phi_d.shape, len(theta))

    # ---- log Z for this Theta
    H_p0 = standard_gaussian_entropy(d)
    H_cons_all = h_from_theta_reg(Theta, m_nodes, d)                          # starts at H(p_0) at the FIRST node
    H_pipe_all = h_bound_at(t, dH.double().numpy(), t_reg, d)
    omitted = float(H_pipe_all[0] - H_p0)                                     # entropy change before the first node
    H_cons, H_pipe = float(H_cons_all[-1]), float(H_pipe_all[-1])
    th_dot_m = float(theta @ m_target)
    logZ = {'A_cons': H_cons + th_dot_m, 'B_pipeline': H_pipe + th_dot_m}
    eb = entropy_bound({config: dict(xt=xt, dH_t_bound=dH, t=torch.as_tensor(t))}, config, pots, check_p0=False)
    logZ['B_full_grid'] = float(eb['H_bound'] + th_dot_m)
    th1 = theta_t[-1].double().numpy() if theta_t.dim() == 2 else theta_t['theta_t'][-1].double().numpy()
    logZ['C_theta_t'] = float(eb['H_bound'] + th1 @ m_target)
    src = logZ['A_cons'] if a.logz_source == 'cons' else logZ['B_pipeline']

    nlp_d, nlp_g = src - phi_d @ theta, src - phi_g @ theta                   # -log p, absolute
    H_line = src - float(theta @ m_target)                                    # = H_cons (A) or H_pipe (B)
    gap_data = float(theta @ (m_target - m_data))
    gap_gen = float(theta @ (m_target - m_gen))

    print('\n=== log Z_Theta (nats), Theta = Theta_reg[-1] unless stated ===')
    print(f'  H(p_0)                          {H_p0:12.3f}   (entropy change before first node, from theta_k: {omitted:+.3f})')
    print(f'  H_cons  (Theta_reg in integral)  {H_cons:12.3f}      Theta.m_target {th_dot_m:+.3f}')
    print(f'  H_pipe  (theta_k in integral)    {H_pipe:12.3f}      at t_reg[-1]; full fine grid {eb["H_bound"]:.3f}')
    for k, v in logZ.items():
        print(f'  log Z {k:12s} {v:14.3f}')
    print(f'  A - B (mixing of the two theta paths) = {logZ["A_cons"] - logZ["B_pipeline"]:+.3f} nats')
    print(f'  -log p mean over data  {nlp_d.mean():.3f}  (std {nlp_d.std(ddof=1):.3f}) | over generated {nlp_g.mean():.3f} (std {nlp_g.std(ddof=1):.3f})')
    print(f'  H line = log Z - Theta.m_target = {H_line:.3f}')
    print(f'  gap Theta.(m_target - m_data)      = {gap_data:+.3f}   (SDE target moments vs data moments at t_reg[-1])')
    print(f'  gap Theta.(m_target - <phi>_gen)   = {gap_gen:+.3f}   (sampler moment-matching residual projected on Theta)')
    print(f'  mean_data(-log p) - H_line         = {nlp_d.mean() - H_line:+.3f}  (= -gap_data by construction)')

    # ---- normalizability
    print('\n=== normalizability ===')
    ok_sign = None
    if 'L_6' in ranges:
        s, e = ranges['L_6']
        L = theta[s:e]
        ok_sign = bool((L < 0).all())
        print(f'  L_6 columns {s}:{e}: Theta = {np.array2string(L, precision=3)}\n  all < 0: {ok_sign}  '
              f'(positive: {int((L > 0).sum())}, zero: {int((L == 0).sum())})')
    else:
        print('  no L_6 family in this model: the sign test does not apply')
    radii = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256], float)
    E, names = ray_test(pots, theta, M, device, radii)
    slope = np.log(np.maximum(E[:, -1], 1e-300) / np.maximum(np.abs(E[:, -2]), 1e-300)) / np.log(radii[-1] / radii[-2])
    bad = E[:, -1] <= 0
    for cls in ('random', 'tone', 'dirac'):
        m = names == cls
        print(f'  rays {cls:7s}: {m.sum():4d} | energy<=0 at r={radii[-1]:g}: {int((bad & m).sum())} | '
              f'min energy {E[m, -1].min():.3e} | slope d log E/d log r (median) {np.median(slope[m]):.2f}, min {slope[m].min():.2f}')
    ok_rays = not bad.any()
    print(f'  => rays: {"ALL energies positive at r=%g" % radii[-1] if ok_rays else "FAIL: energy <= 0 on %d rays (p_Theta not normalizable)" % int(bad.sum())}')
    if not (ok_rays and ok_sign in (True, None)):
        print('  !! log Z of a non-normalizable Theta is meaningless: do not read the absolute axis.')

    # ---- plot
    allv = np.concatenate([nlp_d, nlp_g])
    lo, hi = np.percentile(allv, [0.02, 99.98])
    edges = np.linspace(lo, hi, 101)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(np.clip(nlp_d, lo, hi), bins=edges, density=True, color='0.55', alpha=0.6, label='data signals')
    ax.hist(np.clip(nlp_g, lo, hi), bins=edges, density=True, histtype='step', lw=1.6, color='C0', label='generated signals')
    lab_src = ('Theta_reg-consistent log Z (estimate, not covered by the lower-bound proof)' if a.logz_source == 'cons'
               else 'pipeline log Z = H_bound(theta_k) + Theta_reg.m (lower bound for theta_k)')
    ax.axvline(H_line, color='C3', lw=2, label=r'$H(p_\theta)=\mathbb{E}[-\log p_\theta]$ (differential entropy, nats)')
    ax.set_xlabel(r'$-\log p_\theta(x)$  (nats, absolute)', labelpad=8)
    ax.set_ylabel('density')
    ax.set_title(f'{a.label}, seed {a.seed}, {desc.split()[0]}\n{lab_src}', fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.99, 0.55, f'$H$ = {H_line:.1f}\ngap $\\Theta\\cdot(m-m_{{gen}})$ = {gap_gen:+.2f}', transform=ax.transAxes,
            ha='right', fontsize=9)
    fig.tight_layout()
    out = HERE / 'figures' / f'logp_absolute_{a.label}_seed{a.seed}_lam{a.lam:g}{"_unc" if a.unconstrained else ""}.png'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    np.savez(out.with_suffix('.npz'), nlp_data=nlp_d.astype(np.float32), nlp_gen=nlp_g.astype(np.float32), H_line=H_line,
             theta_desc=desc, rays_E=E, ray_radii=radii, ray_class=names, gap_data=gap_data, gap_gen=gap_gen,
             **{f'logZ_{k}': v for k, v in logZ.items()}, H_cons=H_cons, H_pipe=H_pipe, H_p0=H_p0)
    print(f'\nfigure: {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='turbulence_nested_M3_bicw_allmass_scat4')
    ap.add_argument('--seed', type=int, default=900)
    ap.add_argument('--lam', type=float, default=3e-5)
    ap.add_argument('--ridge', type=float, default=1e-4)
    ap.add_argument('--eps', type=float, default=1e-2)
    ap.add_argument('--unconstrained', action='store_true')
    ap.add_argument('--logz_source', choices=['cons', 'pipeline'], default='cons')
    main(ap.parse_args())
