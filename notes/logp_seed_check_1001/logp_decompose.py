"""Per-signal decomposition of log p = theta . phi(x) on the data x1 of one run.

    python logp_decompose.py <exp_dir> <theta.pt> [--key CONFIG] [--chunk 250]

exp_dir: experiments/.../<config> (config.json + fitted_potentials/)
theta.pt: either a (T, r) theta_t tensor (last row used) or a dict {config: (r,)}.
Rebuilds x1 and the potentials exactly as the rare/typical notebooks do.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
from scipy import stats

ROOT = Path('/home/chiaraz/phd/MGD-for-Maximum-Entropy-Generation')
sys.path.insert(0, str(ROOT / 'codes')); sys.path.insert(0, str(ROOT / 'data'))
os.environ.setdefault('TURBULENCE_1D_DATA_PATH', str(ROOT / 'data/data_files/turbulence_1d_period.pt'))
from sde_routines import *          # noqa  (brings get_1d_potentials, as in the notebooks)
from filters_bank import *          # noqa
from potentials import *            # noqa
from ortho_wavelet import *         # noqa
from utils import normalize, split_periodize_reshape
from check_potentials import theta_column_map
from data_loader import load_turbulence_1d

ap = argparse.ArgumentParser()
ap.add_argument('exp_dir', type=Path); ap.add_argument('theta', type=Path)
ap.add_argument('--chunk', type=int, default=250); ap.add_argument('--save', type=Path)
a = ap.parse_args()
torch.manual_seed(0)
dev = torch.device('cpu')

cfg = json.loads((a.exp_dir / 'config.json').read_text())
cfg = cfg.get('args', cfg)
J, Q, M, n1 = cfg['J'], cfg['Q'], cfg.get('M', 256), cfg['n1']
sub, tgt = cfg.get('subseries_len', 1024), cfg.get('target_len', 256)

W = DefineWavelet('Db', m=3, device=dev)
Data = split_periodize_reshape(load_turbulence_1d().to(dev), sub)
for _ in range(int(np.log2(sub / tgt))):
    Data = W.decompose(Data)[1]
x1 = normalize(Data[:n1])
print('x1', tuple(x1.shape))

filters, filters_Phi = return_Filters(M, J, 1, device=dev, include_phi=True)
filters_Q = return_Filters(M, J, Q, device=dev)
pots = get_1d_potentials(cfg['terms'], J, filters, Q, scalar_param=None, parallel=False,
                         filters_Q=filters_Q, filters_Phi=filters_Phi,
                         deduplicate_filters=bool(cfg.get('deduplicate_filters') or False),
                         dedup_tol=cfg.get('dedup_tol') or 1e-12)
for name, p in pots.items():
    if hasattr(p, 'is_fitted'):
        pots[name] = type(p).load_fixed_parameters(a.exp_dir / 'fitted_potentials' / f'{name}.pt', p.filters,
                                                   map_location=dev)

if a.save and a.save.exists():
    phi = torch.load(a.save).numpy()
else:
    with torch.no_grad():
        phi = torch.cat([torch.cat([(lambda f: f[:, None] if f.ndim == 1 else f)(p(x1[i:i + a.chunk]))
                                    for p in pots.values()], -1).double()
                         for i in range(0, n1, a.chunk)]).numpy()
    if a.save:
        torch.save(torch.as_tensor(phi), a.save)

def _last(path):                            # (T, r) theta_t -> last row; (r,) -> as is
    T = torch.load(path, map_location='cpu', weights_only=False)
    return (T if T.ndim == 1 else T[-1]).double().numpy().reshape(-1)
# a.theta: one file, or a directory of per-seed files -> seed-averaged theta (as the notebooks)
files = sorted(a.theta.glob('*.pt')) if a.theta.is_dir() else [a.theta]
th = np.mean([_last(f) for f in files], axis=0)
print(f'theta: mean over {len(files)} file(s)')
r = th.shape[0]
assert phi.shape == (n1, r), (phi.shape, r)
fam = {n: (s, s + d) for n, (s, d) in theta_column_map(pots, r).items()}
name_of = {}
for n, (s, e) in fam.items():
    for c in range(s, e):
        name_of[c] = f'{n.replace("Scattering_Fourth_Order_Mod2_", "Scat4_").replace("_gaussianK", "")}[{c - s}]'

contrib = phi * th[None, :]                 # (n1, r) per-signal, per-column log p contribution
lp = contrib.sum(1)
med = np.median(lp)
print(f'\nlog p = theta.phi over {n1} data signals: median {med:.4g}, std {lp.std():.4g}, '
      f'IQR {np.subtract(*np.percentile(lp, [75, 25])):.4g}, skew {stats.skew(lp):.3g}, '
      f'excess kurtosis {stats.kurtosis(lp):.3g}, min {lp.min():.4g}, max {lp.max():.4g}')

print('\nper family: std of contribution | excess kurtosis')
for n, (s, e) in fam.items():
    v = contrib[:, s:e].sum(1)
    print(f'  {n:40s} {v.std():10.4g} {stats.kurtosis(v):10.3g}')

sd = contrib.std(0)
print('\ntop 15 columns by std of theta_c * phi_c over signals:')
for c in np.argsort(sd)[::-1][:15]:
    print(f'  col {c:3d} {name_of[c]:26s} theta {th[c]: .4g}  std(phi) {phi[:, c].std():.3g}  '
          f'std(theta*phi) {sd[c]:.4g}')

# near-copy pairs: corr of phi and std of the PAIR's summed contribution
print('\nmost correlated column pairs among the top-30 columns (|corr| > 0.999):')
top = np.argsort(sd)[::-1][:30]
Cc = np.corrcoef(phi[:, top].T)
for i in range(len(top)):
    for j in range(i + 1, len(top)):
        if abs(Cc[i, j]) > 0.999:
            ci, cj = top[i], top[j]
            print(f'  {name_of[ci]:24s} ~ {name_of[cj]:24s} corr {Cc[i, j]:.8f}  '
                  f'std each {sd[ci]:.4g} / {sd[cj]:.4g}  std of the pair sum {contrib[:, [ci, cj]].sum(1).std():.4g}')

# what makes the outliers: deviation from the median signal, per column
dev_ = contrib - np.median(contrib, 0)[None, :]
order = np.argsort(np.abs(lp - med))[::-1]
print('\n10 most extreme signals: log p - median, and the 4 columns contributing most to it')
for i in order[:10]:
    top_c = np.argsort(np.abs(dev_[i]))[::-1][:4]
    print(f'  #{i:5d}  {lp[i] - med:+10.4g} : ' +
          ', '.join(f'{name_of[c]} {dev_[i, c]:+.3g}' for c in top_c))

# leave-one-family-out: does removing a family make log p well-behaved?
print('\nlog p without one family: std, excess kurtosis')
for n, (s, e) in fam.items():
    v = lp - contrib[:, s:e].sum(1)
    print(f'  without {n:40s} std {v.std():10.4g}  kurt {stats.kurtosis(v):10.3g}')
