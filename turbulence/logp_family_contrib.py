"""log p contribution of each potential family, rare vs typical signals (light version of the
per-family cell of turb_interpretation_rare_typical_final.ipynb).

For a run: C_F(x) = theta_F . phi_F(x) on the data x1, theta = Theta_reg[-1] solved offline
(L_6 < 0 constrained if present, else the unconstrained sweep file, else --theta_t = the
sampler's last theta, which is step noise and only for a quick look). p_theta(x) ~
exp(+theta.phi(x)), so higher C_F = more likely. A signal is 'rare' if its max |lag-1
increment| is above the P-th percentile of x1 (notebook: P = 95; also 99 here).

Prints, per family and per category: mean and std over signals of C_F, then
D_F = mean(rare) - mean(typical) and its share of D_TOTAL. A family with a large |D_F|
is one that separates rare from typical signals: the question for Scat4 is whether its D
is still large once the Scalar families describe the marginals well.

Usage (repo root on Jean Zay):
    python turbulence/logp_family_contrib.py --label turbulence_ggd_mass_L6_2b_scat4
    python turbulence/logp_family_contrib.py --label turbulence_final_coarse_noL2_scat4
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
if __name__ == '__main__':                  # only as a script: an importing notebook keeps its backend
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data', HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from filters_bank import return_Filters                              # noqa: E402
from potentials_builder import get_1d_potentials                     # noqa: E402
from check_potentials import theta_column_map                        # noqa: E402
from check_fit_offline import build_data                             # noqa: E402

SHORT = {'Scattering_Fourth_Order_Mod2_': 'Scat4_', 'Scalar_': 'S_', '_gaussianK': ''}
CHUNK = 500


def short(name):
    for a, b in SHORT.items():
        name = name.replace(a, b)
    return name


def newest(pattern, base):
    files = sorted(Path(base).glob(pattern), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f'nothing matches {base}/{pattern}')
    return files[-1]


def find_theta(config, lam, ridge, eps, use_theta_t, sweep_root, l6neg_root, unconstrained=False):
    """(theta vector, description)."""
    if use_theta_t:
        f = HERE / 'saved_results' / 'lagrange_multipliers' / f'{config}.pt'
        th = torch.load(f, map_location='cpu', weights_only=False)
        th = th[-1] if hasattr(th, 'shape') and th.dim() == 2 else th['theta_t'][-1]
        return th.double().numpy(), f'theta_t[-1] of {f.name} (step noise, quick look only)'
    cands = sorted((l6neg_root / config).glob(f'lam{lam:g}_ridge{ridge:g}_moment_mask_neg{eps:g}*.pt'),
                   key=lambda f: f.stat().st_mtime) if (l6neg_root / config).exists() and not unconstrained else []
    if cands:
        f, kind = cands[-1], 'L_6<0 constrained'
    else:
        f = sweep_root / config / f'lam{lam:g}_ridge{ridge:g}_moment_mask.pt'
        kind = 'UNCONSTRAINED sweep'
        if not f.exists():
            raise FileNotFoundError(f'no Theta_reg for {config}: looked in {l6neg_root / config} and {f}')
    sol = torch.load(f, map_location='cpu', weights_only=False)
    assert sol['config'] == config, (sol['config'], config)
    print(f"  Theta_reg ({kind}): {f.name}  lam={sol['lam']:g}  residual {sol['residual']:.2e}  "
          f"last node t={float(sol['t_reg'][-1]):.8f}")
    return sol['Theta_reg'][-1].double().numpy(), f'{kind} {f.name}'


def load_potentials(exp_dir, cfg, device):
    M, J, Q = cfg['M'], cfg['J'], cfg['Q']
    filters, filters_phi = return_Filters(M, J, 1, device=device, include_phi=True,
                                          wav_type=cfg.get('wav_type', 'morlet'))
    filters_Q = return_Filters(M, J, Q, device=device, wav_type=cfg.get('wav_type', 'morlet'))
    pots = get_1d_potentials(cfg['terms'], J, filters, Q, scalar_param=None, parallel=False,
                             filters_Q=filters_Q, filters_Phi=filters_phi,
                             deduplicate_filters=cfg.get('deduplicate_filters', False),
                             dedup_tol=cfg.get('dedup_tol', 1e-12))
    for name, p in pots.items():
        p.to(device)
        if hasattr(p, 'is_fitted'):
            loaded = type(p).load_fixed_parameters(exp_dir / 'fitted_potentials' / f'{name}.pt',
                                                   p.filters, map_location=device)
            loaded.to(device)
            pots[name] = loaded
    return pots


def phi_matrix(pots, x):
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], CHUNK):
            feats = []
            for p in pots.values():
                f = p(x[i:i + CHUNK])
                feats.append(f.unsqueeze(1) if f.ndim == 1 else f)
            out.append(torch.cat(feats, dim=-1).detach().cpu().double())
    return torch.cat(out).numpy()


def plot_shapes(C, rare, P, title, out):
    """As the notebook's shape cell: histogram of C_F(x) over signals, rare vs typical,
    density (the categories differ in size), log y, one panel per family, TOTAL last."""
    fams = list(C)
    n_cols = 4
    n_rows = int(np.ceil(len(fams) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.4 * n_rows), squeeze=False)
    for i, F in enumerate(fams):
        ax = axes[i // n_cols][i % n_cols]
        lo, hi = np.percentile(C[F], [0.05, 99.95])        # clip extreme outliers from the bin range only
        edges = np.linspace(lo, hi, 61)
        for name, m, col in (('typical', ~rare, 'C0'), ('rare', rare, 'C3')):
            v = C[F][m]
            ax.hist(np.clip(v, lo, hi), bins=edges, density=True, color=col, lw=1.5, log=True,
                    histtype='stepfilled' if name == 'typical' else 'step',
                    alpha=0.35 if name == 'typical' else 1.0,
                    label=f'{name} (n={len(v)}, skew {stats.skew(v):+.2f})')
        ax.set_title(short(F), fontsize=9)
        ax.set_xlabel('theta_F . phi_F(x)  [log p contribution, nats]', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, frameon=False)
    for j in range(len(fams), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis('off')
    fig.suptitle(f'{title}: log p contribution per family, rare (> {P:g}th pct of max |lag-1 increment|) '
                 'vs typical', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130)
    print(f'figure: {out}')


def main(label, seed, lam, ridge, eps, use_theta_t, percentiles, unconstrained=False):
    exp = newest(f'*/*_seed_{seed}_*{label}_2*', HERE / 'experiments')
    cfg = json.loads((exp / 'config.json').read_text())
    config = cfg['config_name']
    print(f'run {config}\n  terms {cfg["terms"]}')
    scratch = Path(os.environ.get('SCRATCH', '/nonexistent')) / 'MGD-for-Maximum-Entropy-Generation/turbulence'
    theta, desc = find_theta(config, lam, ridge, eps, use_theta_t,
                             HERE / 'saved_results' / 'theta_reg_lamsweep', scratch / 'theta_reg_l6neg',
                             unconstrained=unconstrained)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x1, _ = build_data(int(cfg.get('subseries_len', 512)), int(cfg['M']), int(cfg['n1']), 1)
    x1 = x1.to(device)
    pots = load_potentials(exp, cfg, device)
    n_feat = theta.shape[0]
    cmap = theta_column_map(pots, n_feat)
    ranges = {k: (a, a + d) for k, (a, d) in cmap.items()}
    print('  families: ' + ', '.join(f'{short(k)}[{a}:{b}]' for k, (a, b) in ranges.items()))
    phi = phi_matrix(pots, x1)
    assert phi.shape == (x1.shape[0], n_feat), (phi.shape, n_feat)

    C = {k: phi[:, a:b] @ theta[a:b] for k, (a, b) in ranges.items()}
    C['TOTAL'] = phi @ theta
    assert np.allclose(sum(C[k] for k in ranges), C['TOTAL'], rtol=1e-8, atol=1e-6 * np.abs(C['TOTAL']).max())

    inc = (x1[..., 1:] - x1[..., :-1]).abs().amax(dim=(1, 2)).cpu().numpy()
    fams = list(C)
    # seed 900 keeps the original name (read by plot_logp_summary.py, talk_turbulence.ipynb); other seeds
    # get _seed<N>, so they never overwrite it (2026-10-30)
    stag = '' if seed == 900 else f'_seed{seed}'
    npz = HERE / 'figures' / 'data' / f'logp_C_{label}{stag}_lam{lam:g}{"_unc" if unconstrained else ""}.npz'
    np.savez(npz, increment=inc, theta_desc=desc, **C)           # read by plot_logp_summary.py
    print(f'saved family contributions: {npz}')
    for P in percentiles:
        rare = inc > np.percentile(inc, P)
        print(f'\n=== rare = max |lag-1 increment| above the {P}th percentile: '
              f'{int(rare.sum())} rare / {int((~rare).sum())} typical signals   [{desc}]')
        rows = []
        for F in fams:
            r, t = C[F][rare], C[F][~rare]
            rows.append({'family': short(F), 'typ mean': t.mean(), 'typ std': t.std(ddof=1),
                         'rare mean': r.mean(), 'rare std': r.std(ddof=1), 'D = rare-typ': r.mean() - t.mean()})
        df = pd.DataFrame(rows).set_index('family')
        dtot = df.loc['TOTAL', 'D = rare-typ']
        df['D / D_TOTAL'] = df['D = rare-typ'] / dtot
        # bootstrap over the rare signals (typical fixed), as the notebook's table 1
        rng = np.random.default_rng(0)
        idx = np.flatnonzero(rare)
        boot = np.array([[C[F][rng.choice(idx, idx.size)].mean() - C[F][~rare].mean() for F in fams]
                         for _ in range(500)])
        df['SE(D)'] = boot.std(0, ddof=1)
        with pd.option_context('display.float_format', '{:,.4g}'.format, 'display.width', 200):
            print(df.to_string())
        plot_shapes(C, rare, P, label, HERE / 'figures' / f'logp_families_{label}{stag}_lam{lam:g}_p{P:g}.png')
    print('\nstd across signals of each family (all signals), and corr with TOTAL:')
    for F in fams:
        if F != 'TOTAL':
            print(f'  {short(F):22s} std {C[F].std(ddof=1):10.4g}   corr with TOTAL {np.corrcoef(C[F], C["TOTAL"])[0, 1]:+.3f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True, help='experiment label, e.g. turbulence_ggd_mass_L6_2b_scat4')
    ap.add_argument('--seed', type=int, default=900)
    ap.add_argument('--lam', type=float, default=5e-4)
    ap.add_argument('--ridge', type=float, default=1e-4)
    ap.add_argument('--eps', type=float, default=1e-2)
    ap.add_argument('--unconstrained', action='store_true',
                    help='use the unconstrained Theta_reg of theta_reg_lamsweep even if an L_6<0 file exists')
    ap.add_argument('--theta_t', action='store_true', help="use the sampler's last theta instead of Theta_reg")
    ap.add_argument('--percentiles', type=float, nargs='+', default=[95, 99])
    a = ap.parse_args()
    main(a.label, a.seed, a.lam, a.ridge, a.eps, a.theta_t, a.percentiles, a.unconstrained)
