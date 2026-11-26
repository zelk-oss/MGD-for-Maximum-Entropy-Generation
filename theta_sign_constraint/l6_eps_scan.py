"""Helpers for explore_l6_eps_scan.ipynb: does a negative L_6 coefficient, once chosen, change the scalar marginals?

Everything is evaluated locally from the fitted potentials, the turbulence data and the saved Theta_reg of
theta_sign_constraint/results/pin_l6_test (last node). Notation as in the notebook.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TURBULENCE_1D_DATA_PATH', str(ROOT / 'data/data_files/turbulence_1d_period.pt'))
for p in (ROOT, ROOT / 'codes', ROOT / 'turbulence'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from logp_family_contrib import load_potentials          # noqa: E402
from check_potentials import theta_column_map            # noqa: E402
from check_fit_offline import build_data, channel_stats_on_grid   # noqa: E402

RES = ROOT / 'theta_sign_constraint/results/pin_l6_test/pinl6_bscat4'
EXP_ROOT = ROOT / 'turbulence/experiments'
N_L6 = 9                                   # L_6 columns 0:9 = the 8 wavelet scales (fine -> coarse) + the low-pass


def load_case(seed):
    """Fitted potentials, data, free Theta_reg (last node) and column map of one seed."""
    config = json.loads((RES / f'seed{seed}/summary.json').read_text())['config']
    exp = next(q for q in EXP_ROOT.glob(f'*/{config}') if q.is_dir())
    cfg = json.loads((exp / 'config.json').read_text())
    pots = load_potentials(exp, cfg, 'cpu')
    x, _ = build_data(int(cfg.get('subseries_len', 512)), int(cfg['M']), int(cfg['n1']), 1, data_path=cfg.get('data_path'))
    th = np.load(RES / f'seed{seed}/logp_scan.npz')['theta_last|free']
    return dict(pots=pots, x=x.double(), theta_free=th, cmap=theta_column_map(pots, th.size), seed=seed)


def coefficients(pot, x):
    """Complex wavelet coefficients z_c(u) of every channel c: (N, n_channels, T)."""
    f = pot.filters.double()
    return torch.fft.ifft(f * torch.fft.fft(x))


def peak_index(pot):
    return pot.filters[0].abs().argmax(-1).numpy()


def associated_channels(case):
    """For each of the 9 scales, the Scalar_psi (Q=3) channels whose band centre is within half an octave."""
    pk1 = peak_index(case['pots']['Scalar_morlet_gaussianK'])
    pk3 = peak_index(case['pots']['Scalar_psi_gaussianK'])
    out = []
    for p in pk1:
        if p == 0:
            out.append([c for c, q in enumerate(pk3) if q == 0])
        else:
            out.append([c for c, q in enumerate(pk3) if q > 0 and abs(np.log2(q / p)) <= 0.5])
    return out


def scalar_energy(case, name, c, a, sd_u):
    """Scalar confinement energy U(a) = -(V_c(a sd) - V_c(0)) in nats per site, on a = |Re z_c| / sd(Re z_c).

    V_c(z) = sum_s (theta_s / stat_scale_s) g_s(|z|): the active slots of channel c, theta = free Theta_reg."""
    pot = case['pots'][name]
    start = case['cmap'][name][0]
    cols = [i for i, f in enumerate(pot.active_flat.tolist()) if f % pot.J == c]
    coef = case['theta_free'][start + np.array(cols)] / pot.stat_scale[cols].double().numpy()
    g = np.concatenate([[0.0], a * sd_u])
    slots, vals = channel_stats_on_grid(pot, c, g)
    assert len(slots) == len(cols)
    V = coef @ vals
    return -(V[1:] - V[0])


def check_scalar_curve(case, name, c, n=2000):
    """Mean over data of V_c(Re z_c) must equal the channel part of theta . phi from the potential's own forward."""
    pot = case['pots'][name]
    start = case['cmap'][name][0]
    x = case['x'][:n]
    cols = [i for i, f in enumerate(pot.active_flat.tolist()) if f % pot.J == c]
    th = torch.as_tensor(case['theta_free'][start:start + pot.active_flat.numel()])
    ref = (pot.forward(x) .double()[:, cols] * th[cols]).sum(1).mean().item()
    u = coefficients(pot, x)[:, c].real.abs().reshape(-1).numpy()
    coef = case['theta_free'][start + np.array(cols)] / pot.stat_scale[cols].double().numpy()
    _, vals = channel_stats_on_grid(pot, c, u)
    return ref, float((coef @ vals).mean())


def l6_conditional(case, j, c, name_c, edges):
    """g(a) = E[|z_j|^6 | |Re z_c|/sd in bin a] on the data, with the bin counts (z_j: complex Q=1 coefficient)."""
    z6 = coefficients(case['pots']['L_6'], case['x'])[:, j].abs().pow(6).reshape(-1).numpy()
    u = coefficients(case['pots'][name_c], case['x'])[:, c].real.reshape(-1).numpy()
    sd = u.std()
    a = np.abs(u) / sd
    idx = np.digitize(a, edges) - 1
    n = np.bincount(idx.clip(0, len(edges) - 1), minlength=len(edges))[:len(edges) - 1]
    s = np.bincount(idx.clip(0, len(edges) - 1), weights=z6, minlength=len(edges))[:len(edges) - 1]
    g = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return g, n, sd, a


def sd_phi_l6(case):
    """Standard deviation over the data signals of phi_{L_6,j} = <|z_j|^6>_u, j = 0..8."""
    return coefficients(case['pots']['L_6'], case['x']).abs().pow(6).mean(-1).std(0).numpy()


def build_scan(case, edges=np.logspace(-1, 1.3, 41), min_count=30):
    """Everything independent of eps, for every scale j and every channel (Q=1 'morlet', Q=3 'psi' voices)."""
    assoc = associated_channels(case)
    names = {'morlet': 'Scalar_morlet_gaussianK', 'psi': 'Scalar_psi_gaussianK'}
    centres = np.sqrt(edges[:-1] * edges[1:])
    scan = {}
    for j in range(N_L6):
        pairs = [('morlet', j)] + [('psi', c) for c in assoc[j]]
        for fam, c in pairs:
            g, n, sd, a = l6_conditional(case, j, c, names[fam], edges)
            ok = n >= min_count
            Us = scalar_energy(case, names[fam], c, centres, sd)
            scan[(j, fam, c)] = dict(a=centres[ok], g=g[ok], Us=Us[ok], n=n[ok], sd=sd,
                                     a_q999=float(np.quantile(a, 0.999)), a_max=float(a.max()))
    return scan


PSI_COLORS = ['C1', 'C2', 'C3']
A_REF = 1.0                # energies are measured relative to a = A_REF (only increments shape the tails)


def scale_label(case, j):
    pk = int(peak_index(case['pots']['L_6'])[j])
    return 'low-pass' if pk == 0 else f'ψ_{j} (k₀={pk})'


def tail(d):
    """Restrict a scan entry to a >= A_REF and measure both energies from the first bin there."""
    k = np.flatnonzero(d['a'] >= A_REF)
    k0 = k[0]
    return d['a'][k], d['Us'][k] - d['Us'][k0], d['g'][k] - d['g'][k0]


def u6_curve(d, eps, sd_phi_j):
    """L_6 energy increment |theta_L6| (g(a) - g(A_REF)) on the tail, theta_L6 = -eps / sd(phi_L6)."""
    return (eps / sd_phi_j) * tail(d)[2]


def positive_columns(case):
    return [int(j) for j in np.flatnonzero(case['theta_free'][:N_L6] > 0)]


def speed_table(case, scan, j):
    """Regions of the Q=1 scalar potential of scale j: boundaries (in units of sd(Re z)), exponent alpha_k (the potential
    grows like |z|^alpha_k there) and the sign of the fitted coefficient theta_k / stat_scale_k of that region."""
    import pandas as pd
    pot = case['pots']['Scalar_morlet_gaussianK']
    start = case['cmap']['Scalar_morlet_gaussianK'][0]
    sd = scan[(j, 'morlet', j)]['sd']
    cuts = np.concatenate([[0.0], pot.cuts[j].double().numpy() / sd, [np.inf]])
    cols = [(i, f // pot.J) for i, f in enumerate(pot.active_flat.tolist()) if f % pot.J == j]
    coef = {s: case['theta_free'][start + i] / float(pot.stat_scale[i]) for i, s in cols}
    rows = []
    for k in range(pot.K):
        rows.append({'region k': k, 'from (sd)': cuts[k], 'to (sd)': cuts[k + 1], 'α_k': float(pot.alpha[j, k]),
                     'sign θ_k': ('–' if coef[k] < 0 else '+') if k in coef else 'dropped'})
    return pd.DataFrame(rows).set_index('region k')




def plot_scale(case, scan, sdphi, j, eps_grid):
    """Tail energies of scale j on log-log axes: |scalar energy| of every channel (solid where it confines, dotted where
    it raises log p) and the L_6 energy (dashed, Q=1 coefficient) for each eps, light -> dark = small -> large eps."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(8, 5.5))
    n_psi = 0
    for (jj, fam, c), d in scan.items():
        if jj != j:
            continue
        col, lw = ('C0', 2.2) if fam == 'morlet' else (PSI_COLORS[n_psi % 3], 1.3)
        n_psi += fam == 'psi'
        a, Us, _ = tail(d)
        pos, neg = np.where(Us > 0, Us, np.nan), np.where(Us < 0, -Us, np.nan)
        ax.plot(a, pos, color=col, lw=lw)
        ax.plot(a, neg, color=col, lw=lw, ls=':')
    d1 = scan[(j, 'morlet', j)]
    a, _, _ = tail(d1)
    shades = plt.cm.Greys(np.linspace(0.25, 1.0, len(eps_grid)))
    for eps, sh in zip(eps_grid, shades):
        u6 = np.abs(u6_curve(d1, eps, sdphi[j]))
        u6[u6 == 0] = np.nan                         # the increment is 0 at a = 1 by construction
        ax.plot(a, u6, color=sh, ls='--', lw=1.4)
        ax.text(a[-1] * 1.03, u6[-1], f'ε={eps:g}', fontsize=7, va='center')
    ax.axvline(d1['a_q999'], color='grey', ls='-.', lw=1)
    ax.text(d1['a_q999'], 1e-9 * 3, ' 99.9 % of data', fontsize=7, color='grey', rotation=90, va='bottom')
    handles = [Line2D([], [], color='C0', lw=2.2, label='Q=1 scalar'),
               *[Line2D([], [], color=PSI_COLORS[i], lw=1.3, label=f'Q=3 scalar, channel {c}')
                 for i, c in enumerate(c for (jj, fam, c) in scan if jj == j and fam == 'psi')],
               Line2D([], [], color='k', lw=1.3, ls=':', label='scalar, negative (raises log p)'),
               Line2D([], [], color='k', lw=1.4, ls='--', label='L_6 term, one line per ε')]
    ax.legend(handles=handles, fontsize=8, loc='lower right')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(A_REF, a[-1] * 1.6)
    ax.set_ylim(1e-9, 1e3)
    ax.set_xlabel('a = |Re z| / sd(Re z)')
    ax.set_ylabel('|energy increment from a = 1| (nats)')
    ax.set_title(scale_label(case, j))
    fig.tight_layout()
    return fig
