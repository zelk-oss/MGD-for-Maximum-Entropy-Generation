"""Diagnose Scalar_morlet_gaussianK windows on jets (Modane) or turbulence data, offline CPU.
Usage (repo root): TURBULENCE_1D_DATA_PATH=$PWD/data/data_files/turbulence_1d_period.pt \
  python jets/data_audit/scalar_windows/diag_fit.py --data jets --name default --kw '{}' [--n1 5000] [--solve]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT/'codes', ROOT/'codes/filters', ROOT/'codes/potentials', ROOT/'data', ROOT/'jets', ROOT/'turbulence'):
    sys.path.insert(0, str(p))
from filters_bank import return_Filters
from potentials_builder import get_1d_potentials

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='jets'); ap.add_argument('--name', default='default')
ap.add_argument('--kw', default='{}'); ap.add_argument('--n1', type=int, default=5000)
ap.add_argument('--M', type=int, default=256); ap.add_argument('--J', type=int, default=8)
ap.add_argument('--solve', action='store_true')
a = ap.parse_args()
kw = json.loads(a.kw)
if kw.get('min_window_bic') == 'auto': kw['min_window_bic'] = 1.0   # placeholder, set via --kw numeric normally
NH = 9000
if a.data == 'jets':
    from jet_data import load_modane_windows
    X, _ = load_modane_windows(512, a.M, n=a.n1 + NH, spread=True)
    x1, xh = X[:a.n1], X[a.n1:]
else:
    from check_fit_offline import build_data
    x1, xh = build_data(512, a.M, a.n1, NH)
print('data', a.data, 'x1', tuple(x1.shape), 'xh', tuple(xh.shape), 'kw', kw, flush=True)
filters, filters_phi = return_Filters(a.M, a.J, 1, device='cpu', include_phi=True)
pots = get_1d_potentials(['Scalar_morlet_gaussianK'], a.J, filters, 1, filters_Q=None, filters_Phi=filters_phi,
                         deduplicate_filters=True, dedup_tol=1e-8, ggd_kwargs=kw)
pot = pots['Scalar_morlet_gaussianK']
pot.verbose = True
t0 = time.time(); pot.fit(x1); plt.close('all')
print(f'fit {time.time()-t0:.0f}s', flush=True)
z = torch.fft.ifft(torch.fft.fft(x1.double()) * pot.filters.double()).real.numpy()
zh = torch.fft.ifft(torch.fft.fft(xh.double()) * pot.filters.double()).real.numpy()
J, K = pot.J, pot.K
act = pot.active.numpy()
actflat = set(pot.active_flat.tolist())
print(f'\nSUMMARY {a.name}: J={J} K={K} active stats={pot.num_coefficients} of {pot.n_slots*J}')
for j in range(J):
    ah = np.abs(z[:, j]).ravel(); sd = z[:, j].std()
    c = pot.cuts[j].double().numpy(); e = pot.eps_ch[j].item()**.5 if pot.eps_ch is not None else float('nan')
    edges = np.concatenate([[0], c, [np.inf]])
    cnt = np.array([((ah >= lo) & (ah < hi)).sum() for lo, hi in zip(edges[:-1], edges[1:])])
    cdf = [(ah < ci).mean() for ci in c]
    alive = [k for k in range(pot.n_slots) if k*J+j in actflat]
    print(f'ch{j} std={sd:.3e} floor={e:.2e}({e/sd:.1e}sd) Keff={int(pot.Keff[j])}')
    print('   cuts/sd=' + ' '.join(f'{ci/sd:.2e}' for ci in c) + '  cdf(<cut)=' + ' '.join(f'{p:.4f}' for p in cdf))
    print('   counts=' + ' '.join(str(int(n)) for n in cnt) + '  pi=' + ' '.join(f'{p:.3f}' for p in pot.pi[j].numpy())
          + '  alpha=' + ' '.join(f'{v:.2f}' for v in pot.alpha[j].numpy())
          + '  scale/sd=' + ' '.join(f'{v/sd:.2e}' for v in pot.scale[j].numpy()))
    print('   active-before-prune=' + str([k for k in range(pot.K) if act[j, k] or True][:0]) + f' alive slots after prune={alive}')
if a.solve:
    from check_fit_offline import check_potential, QS
    rows = check_potential(a.name, pot, x1, xh)
    print('\nmodel/data quantile ratios of |z| (MaxEnt of the potential stats), QS=', QS)
    for r in rows:
        print(f"ch{r['ch']} nstat={r['n']} res={r['res']:.1e} in={np.round(r['r_in'],2)} heldout={np.round(r['r_ho'],2)}")
