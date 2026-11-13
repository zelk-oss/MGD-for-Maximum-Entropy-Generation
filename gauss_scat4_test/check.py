#!/usr/bin/env python3
"""Checks of the in-family Gaussian test (README.md), per seed, at the cross-seed lam (xseed, window end0.99).
Truth: data ~ p* (make_spectrum.py), so the max-ent model is p* itself: theta(Scat2) = theta_star, theta(L_6) =
theta(order 4) = 0, H(p*) exact. Nats; code sign log p = +Theta . phi + const.

  1. entropy: MGD bound H_bound(t) vs exact H(p_t) of the Gaussian interpolant (Cos) at the Theta_reg nodes, and
     H_bound(1) vs H(p*); Gaussian entropy of the generated samples (an UPPER bound on their entropy: max-ent at fixed
     covariance).
  2. theta: Scat2 part of Theta_reg vs theta_star (relative error); sizes of the terms whose truth is 0, as
     1/2 Var_data(theta_F . phi_F) (nats: the 2nd-order size of their contribution to log p).
  3. Fourier check: Scat2 forward . theta_star == (1/M^2) sum_w (theta_star . psi2)(w) |xhat_w|^2 (validates psi2/theta_star).
  4. finite-data prediction for THIS data realization (all seeds share x1): the exact max-ent fit of the empirical moments
     m_hat is shifted from the truth by dtheta ~ Cov_p*(phi)^-1 (m_hat - m*) (linear response, first order; m*, Cov from
     N_BIG draws of p*), and H_G ~ 1/2 dtheta_G^T Cov_GG dtheta_G for every group (same names as kl_groups/compute_H.py).
     MGD + Theta_reg near this prediction = data noise only; well above = error of the method.
Output: <root>/results/check.csv (one row per seed), pred_H.csv (prediction per group) + printed table.
Usage: python gauss_scat4_test/check.py --root gauss_scat4_test/runs/<TAG> --arm <arm>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data', ROOT / 'turbulence'):
    sys.path.insert(0, str(p))
from logp_family_contrib import load_potentials, phi_matrix                 # noqa: E402
from check_potentials import theta_column_map                               # noqa: E402
from logp_gauss_scat2 import load_samples                                   # noqa: E402
from utils_entropy import compute_gaussian_entropy                          # noqa: E402
from codes.effective_dimension import h_bound_at                            # noqa: E402
from codes.gauss_spectrum import sample_gaussian_spectrum                   # noqa: E402


def _pt(path):
    path = Path(path)
    return torch.load(path if path.exists() else path.with_suffix(''), map_location='cpu', weights_only=False)


N_BIG = 1_000_000


def predict(pots, cmap, phi_data, S, dev):
    """Linear-response prediction of the finite-data shift (see 4. above). Returns (dtheta, rows of pred_H.csv)."""
    xb = sample_gaussian_spectrum(S, N_BIG, seed=777).float()
    phib = np.concatenate([phi_matrix(pots, xb[i:i + 100_000].to(dev)) for i in range(0, N_BIG, 100_000)])
    m_star, cov = phib.mean(0), np.cov(phib, rowvar=False)
    dm = phi_data.mean(0) - m_star
    dth = np.linalg.lstsq(cov, dm, rcond=1e-12)[0]                  # code sign: dm/dtheta = Cov
    h = lambda cols: 0.5 * float(dth[cols] @ cov[np.ix_(cols, cols)] @ dth[cols])
    rows = []
    for k, (s, w) in cmap.items():
        rows.append(dict(group=f'fam:{k}', kind='family', H_pred=h(np.arange(s, s + w))))
        idx = np.asarray(pots[k].indices) if hasattr(pots[k], 'indices') else None
        for i in range(w):
            rows.append(dict(group=f'one:{k}[{i}]', kind='single', H_pred=h(np.array([s + i]))))
        if idx is not None and 'Fourth' in k:
            pairs = {}
            for i, (j, jp, _) in enumerate(idx.T):
                pairs.setdefault((int(j), int(jp)), []).append(s + i)
            rows += [dict(group=f'pair:{k}({j},{jp},all)', kind='pair', H_pred=h(np.array(c))) for (j, jp), c in pairs.items()]
    # sampling error of m* itself (N_BIG draws) enters dm with variance ~ n1 / N_BIG relative to the data's: negligible
    return dth, rows


def main(root, arm, window, spectrum_file):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    xs = torch.load(root / 'results' / f'xseed_{arm}.pt', map_location='cpu', weights_only=False)
    lam = float(xs['selected'][window])
    blob = torch.load(spectrum_file, map_location='cpu', weights_only=False)
    S, th_star, H_star, psi2 = blob['S'].double(), blob['theta_star'].double().numpy(), blob['H_star'], blob['psi2'].double().numpy()
    M = S.shape[-1]
    x_big = sample_gaussian_spectrum(S, 200_000, seed=12345)                # exact law, for H(p_t)
    rows = []
    for config, f in xs['theta_files'][window].items():
        exp = next((root / 'experiments').glob(f'*/{config}'))
        cfg = json.loads((exp / 'config.json').read_text()); cfg = cfg.get('args', cfg)
        sol = torch.load(f, map_location='cpu', weights_only=False)
        assert sol['config'] == config
        th, t_reg = sol['Theta_reg'][-1].double().numpy(), sol['t_reg'].double().numpy()
        sr = root / 'saved_results'
        t, dH = _pt(sr / 'sampling_times' / f'{config}.pt').double().numpy(), _pt(sr / 'entropy_bounds' / f'{config}.pt').double().numpy()
        pots = load_potentials(exp, cfg, dev)
        cmap = theta_column_map(pots, len(th))
        # 3. Fourier check of the Scat2 block and of theta_star
        a2, w2 = cmap['Scattering_Second_Order_Q1']
        xr = torch.randn(300, 1, M, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        q = phi_matrix({'s2': pots['Scattering_Second_Order_Q1']}, xr.float().to(dev)) @ th_star
        xh = torch.fft.fft(xr[:, 0], dim=-1)
        qF = ((th_star @ psi2)[None] * (xh.real ** 2 + xh.imag ** 2).numpy()).sum(1) / M ** 2
        fourier = float(np.abs(q - qF).max() / q.std())
        # 2. theta
        x1 = sample_gaussian_spectrum(S, int(cfg['n1']), seed=int(cfg.get('data_seed') or 0)).float().to(dev)   # = run's x1
        phi = phi_matrix(pots, x1)
        size = {k: 0.5 * float(np.var(phi[:, s:s + w] @ th[s:s + w])) for k, (s, w) in cmap.items()}
        th2 = th[a2:a2 + w2]
        if not rows:                                                 # shared data: one prediction for all seeds
            dth, pred = predict(pots, cmap, phi, S, dev)
            with open(root / 'results' / 'pred_H.csv', 'w', newline='') as fh:
                wr = csv.DictWriter(fh, fieldnames=list(pred[0])); wr.writeheader(); wr.writerows(pred)
            print('finite-data prediction (this x1), 1/2 dtheta^T Cov dtheta per family: '
                  + ', '.join(f'{p["group"][4:]} {p["H_pred"]:.3g}' for p in pred if p['kind'] == 'family'), flush=True)
        # measured shift from the truth (Scat2: theta_star, others 0) vs the predicted one, per family
        truth = np.zeros_like(th); truth[a2:a2 + w2] = th_star
        shift = {k: (float(np.linalg.norm(th[s:s + w] - truth[s:s + w])), float(np.linalg.norm(dth[s:s + w])))
                 for k, (s, w) in cmap.items()}
        print('   |theta - truth| vs predicted |dtheta| per family: '
              + ', '.join(f'{k} {a:.3g} vs {b:.3g}' for k, (a, b) in shift.items()), flush=True)
        # 1. entropy
        Hb = h_bound_at(t, dH, t_reg, M)
        Hb1 = float(h_bound_at(t, dH, [1.0], M)[0])
        He = compute_gaussian_entropy(x_big.reshape(-1, M), cfg.get('interpolant') or 'Cos',
                                      torch.as_tensor(np.append(t_reg, 1.0))).numpy()
        xt = load_samples(root, config).reshape(-1, M).double()
        Hgen = float(compute_gaussian_entropy(xt, cfg.get('interpolant') or 'Cos', torch.tensor([1.0], dtype=torch.float64))[0])
        r = dict(seed=int(cfg['seed']), lam=lam, nt=int(cfg['nt']), sigma=float(cfg['sigma']), t_end=float(t_reg[-1]),
                 H_star=H_star, H_exact_1_check=float(He[-1]), H_bound_1=Hb1, Hbound1_minus_Hstar=Hb1 - H_star,
                 H_exact_tend=float(He[-2]), H_bound_tend=float(Hb[-1]), Hbound_minus_Hexact_tend=float(Hb[-1] - He[-2]),
                 H_gen_gauss=Hgen, Hgen_minus_Hstar=Hgen - H_star,
                 theta2_rel_err=float(np.linalg.norm(th2 - th_star) / np.linalg.norm(th_star)), fourier_check=fourier,
                 **{f'halfvar_{k}': v for k, v in size.items()},
                 **{f'shift_{k}': a for k, (a, _) in shift.items()}, **{f'shift_pred_{k}': b for k, (_, b) in shift.items()})
        rows.append(r)
        print(f'seed {r["seed"]} (lam {lam:g}, t_end {r["t_end"]:.6f}): H(p*) {H_star:.4f} [MC check {r["H_exact_1_check"]:.4f}] | '
              f'H_bound(1) {Hb1:.4f} ({r["Hbound1_minus_Hstar"]:+.4f}) | H_bound(t_end) - H_exact(t_end) '
              f'{r["Hbound_minus_Hexact_tend"]:+.4f} | H_gauss(gen) - H(p*) {r["Hgen_minus_Hstar"]:+.4f}\n'
              f'   theta Scat2 rel err {r["theta2_rel_err"]:.3e} | Fourier check {fourier:.1e} | 1/2 Var_data(theta_F phi_F): '
              + ', '.join(f'{k} {v:.3g}' for k, v in size.items()), flush=True)
    out = root / 'results' / 'check.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'-> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--window', default='end0.99')
    ap.add_argument('--spectrum_file', type=Path, required=True)
    a = ap.parse_args()
    main(a.root, a.arm, a.window, a.spectrum_file)
