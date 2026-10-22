"""What MGD will be asked to reach, computed before the transport (CPU, offline).

Builds x1 and the Scalar_*_gaussianK potentials EXACTLY as turbulence/run_SDE.py does
(same data preprocessing, filters, dedup, builder, fit: BIC cuts, sigmoid windows,
per-channel |z| floor, pruning), then, for every channel, solves the 1-D MaxEnt whose
moments are the potential's own statistics averaged over x1 -- i.e. the targets MGD
matches -- and compares its quantiles of |z| with the data.

What this leaves out: channels are solved independently (exact only for a near-orthogonal
bank; the overcomplete Q=3 bank couples them), and any other potential (L_6, Scat4).
Comparing this to an MGD run of the same configuration measures that coupling.

Default configuration = the coarse two-bank arm (turbulence_scalar_orthog_2bank.sh):
M=128, J=7, Q=3, n1=3000, morlet, dedup 1e-8.

Usage (repo root):
    TURBULENCE_1D_DATA_PATH=$PWD/data/data_files/turbulence_1d_period.pt \
        python turbulence/check_fit_offline.py [--configs prod tailq mass]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials',
          ROOT / 'data', HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils import normalize, split_periodize_reshape                 # noqa: E402
from ortho_wavelet.ReadyToUseWavelets import DefineWavelet           # noqa: E402
from data.data_loader import load_turbulence_1d                      # noqa: E402
from filters_bank import return_Filters                              # noqa: E402
from potentials_builder import get_1d_potentials                     # noqa: E402
from tail_form_holdout import solve_dual, RESID_TOL                  # noqa: E402

QS = np.array([0.99, 0.999, 0.9999, 0.99995])
CONFIGS = {
    'prod':  {},                                         # production fit
    'tailq': dict(tail_quantile=0.99),                   # last cut forced to q.99
    'mass':  dict(tail_quantile=0.99, tail_mass=True),   # + outer-region mass statistic
}


def build_data(subseries_len, target_len, n1, n_heldout):
    W = DefineWavelet('Db', m=3, device='cpu')
    D = split_periodize_reshape(load_turbulence_1d().cpu(), subseries_len)
    for _ in range(int(np.log2(subseries_len / target_len))):
        D = W.decompose(D)[1]
    x1_raw = D[:n1]
    x1 = normalize(x1_raw)                                               # as run_SDE
    xh = ((D[n1:n1 + n_heldout] - x1_raw.mean()) / x1_raw.std()).to(torch.float32)
    return x1, xh


def channel_stats_on_grid(pot, j, g):
    """The potential's statistics for channel j as functions of |z|, same formulas as
    Scalar_GGD_KRegion._all_slot_forward. Returns (slot indices, values (S, len(g)))."""
    eps = float(pot.eps_ch[j]) if pot.eps_ch is not None else float(pot.eps_abs)
    az = np.sqrt(g ** 2 + eps)
    c, s = pot.cuts[j].double().numpy(), pot.sw[j].double().numpy()
    sig = 1.0 / (1.0 + np.exp(np.clip((az[:, None] - c[None, :]) / s[None, :], -700, 700)))
    K = pot.K
    ws = [sig[:, 0]] + [sig[:, k] - sig[:, k - 1] for k in range(1, K - 1)] + [1.0 - sig[:, K - 2]]
    al = pot.alpha[j].double().numpy()
    vals = [ws[k] * az ** al[k] for k in range(K)]
    if pot.tail_mass:
        vals.append(ws[K - 1])
    slots = [int(f) // pot.J for f in pot.active_flat.tolist() if int(f) % pot.J == j]
    return slots, np.array([vals[k] for k in slots])


def check_potential(name, pot, x1, xh, with_l6=False):
    """with_l6: add L_6 = <|z_j|^6> as one more statistic of channel j. Exact only for
    Scalar_morlet_gaussianK, which uses the same Q=1 bank as L_6 (L2p_norm(3, filters))."""
    with torch.no_grad():
        target_all = pot._all_slot_forward(x1.double()).mean(0).numpy()      # MGD's targets
        z1 = torch.fft.ifft(pot.filters.double() * torch.fft.fft(x1.double())).real.numpy()
        zh = torch.fft.ifft(pot.filters.double() * torch.fft.fft(xh.double())).real.numpy()
    rows = []
    for j in range(pot.J):
        a1, ah = np.abs(z1[:, j]).ravel(), np.abs(zh[:, j]).ravel()
        g = np.concatenate([[0.0], np.geomspace(max(a1[a1 > 0].min(), 1e-14), 20 * a1.max(), 60_000)])
        slots, B = channel_stats_on_grid(pot, j, g)
        if not slots:
            continue
        md = np.array([target_all[k * pot.J + j] for k in slots])
        # replication check: my grid formulas evaluated on x1 must equal the class targets
        _, Bd = channel_stats_on_grid(pot, j, a1)
        rep = float(np.abs(Bd.mean(1) - md).max() / np.abs(md).max())
        if with_l6:
            B = np.vstack([B, g ** 6])
            md = np.append(md, (a1 ** 6).mean())
        scale = np.where(np.abs(md) > 0, np.abs(md), 1.0)
        t, p, res, _ = solve_dual(B / scale[:, None], g, md / scale)
        cdf = np.concatenate([[0.0], np.cumsum((p[1:] + p[:-1]) / 2 * np.diff(g))])
        cdf /= cdf[-1]
        qm = np.interp(QS, cdf, g)
        rows.append(dict(name=name, ch=j, n=len(slots), res=res, rep=rep,
                         r_in=qm / np.quantile(a1, QS), r_ho=qm / np.quantile(ah, QS)))
    return rows


def main(configs, M, J, Q, n1, n_heldout, terms, with_l6):
    print(f'config: M={M} J={J} Q={Q} n1={n1} morlet dedup 1e-8; held-out {n_heldout} windows; '
          f'L_6 in the 1-D solve of Scalar_morlet: {with_l6}')
    x1, xh = build_data(512, M, n1, n_heldout)
    filters, filters_phi = return_Filters(M, J, 1, include_phi=True, wav_type='morlet')
    filters_Q = return_Filters(M, J, Q, wav_type='morlet')
    for cfg in configs:
        pots = get_1d_potentials(terms, J, filters, Q, filters_Q=filters_Q, filters_Phi=filters_phi,
                                 deduplicate_filters=True, dedup_tol=1e-8, ggd_kwargs=CONFIGS[cfg])
        print(f'\n################ {cfg}: {CONFIGS[cfg] or "production defaults"}')
        for name, pot in pots.items():
            pot.verbose = False
            t0 = time.time()
            pot.fit(x1)
            fit_s = time.time() - t0
            import matplotlib.pyplot as plt
            plt.close('all')                           # fit_reference draws one figure per channel
            rows = check_potential(name, pot, x1, xh,
                                   with_l6=with_l6 and name == 'Scalar_morlet_gaussianK')
            print(f'--- {name}: fit {fit_s:.0f} s, {pot.num_coefficients} statistics')
            print(f"{'ch':>3} {'stats':>5} {'resid':>8} {'repl':>8} | in-sample  "
                  + ' '.join(f'{q:>7}' for q in ['q.99', 'q.999', 'q.9999', 'q.99995'])
                  + ' | held-out ' + ' '.join(f'{q:>7}' for q in ['q.9999', 'q.99995']))
            for r in rows:
                flag = '' if r['res'] < RESID_TOL else '  NOT CONVERGED'
                print(f"{r['ch']:>3} {r['n']:>5} {r['res']:8.1e} {r['rep']:8.1e} |            "
                      + ' '.join(f'{v:7.3f}' for v in r['r_in']) + ' |          '
                      + ' '.join(f'{v:7.3f}' for v in r['r_ho'][2:]) + flag)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument('--M', type=int, default=128)
    ap.add_argument('--J', type=int, default=7)
    ap.add_argument('--Q', type=int, default=3)
    ap.add_argument('--n1', type=int, default=3000)
    ap.add_argument('--n_heldout', type=int, default=9000)
    ap.add_argument('--terms', nargs='+', default=['Scalar_morlet_gaussianK', 'Scalar_psi_gaussianK'],
                    choices=['Scalar_morlet_gaussianK', 'Scalar_psi_gaussianK'])
    ap.add_argument('--with_l6', action='store_true',
                    help='add <|z|^6> (L_6) to the 1-D solve of Scalar_morlet channels')
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    main(a.configs, a.M, a.J, a.Q, a.n1, a.n_heldout, a.terms, a.with_l6)
