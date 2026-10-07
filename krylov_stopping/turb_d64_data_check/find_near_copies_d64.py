"""Near-copy statistics of the d = 64 turbulence setup (coarse x3, --n1_spread, J = 6, Q = 3, full term set
with L_6_psi, --deduplicate_filters 1e-8). Fits the potentials on the data as SDE.__init__ does, then looks at
the Jacobi-scaled Gram of the gradients on the data: |A_ij| close to 1 = near-identical gradients.
Run from the repo root:  TURBULENCE_1D_DATA_PATH=$PWD/data/data_files/turbulence_1d_period.pt python <this>"""
import contextlib, io, sys
import numpy as np
import torch
sys.path[:0] = ['.', 'codes', 'data']
from data.data_loader import load_turbulence_1d
from codes.utils import split_periodize_reshape, normalize
from codes.ortho_wavelet.ReadyToUseWavelets import DefineWavelet
from codes.filters_bank import return_Filters
from codes.potentials_builder import get_1d_potentials

TERMS = ['L_6', 'L_6_psi', 'L_2_lowpass', 'Scalar_psi_gaussianK', 'Scalar_morlet_gaussianK',
         'Scattering_Fourth_Order_Mod2_Real_Q1', 'Scattering_Fourth_Order_Mod2_Imag_Q1']
J, Q, N, TOL = 6, 3, 2000, 1e-5

W = DefineWavelet('Db', m=3, device='cpu')
raw = load_turbulence_1d()
x = split_periodize_reshape(raw, 512)
for _ in range(3):
    x = W.decompose(x)[1]
per = x.shape[0] // raw.shape[0]                       # same selection as run_SDE --n1_spread
g = np.random.default_rng(0)
s = g.choice(raw.shape[0], N, replace=False)
x1 = normalize(x[torch.as_tensor(s * per + g.integers(0, per, N))])
M = x1.shape[-1]

filters, filters_Phi = return_Filters(M, J, 1, device='cpu', include_phi=True)
filters_Q = return_Filters(M, J, Q, device='cpu')
with contextlib.redirect_stdout(io.StringIO()):
    pots = get_1d_potentials(TERMS, J, filters, Q, filters_Q=filters_Q, filters_Phi=filters_Phi,
                             scalar_param=None, parallel=False, deduplicate_filters=True, dedup_tol=1e-8)
    for p in pots.values():
        if hasattr(p, 'fit'):
            p.fit(x1)

labels, grads = [], []
for name, p in pots.items():
    gp = p.grad(x1).reshape(N, p.num_coefficients, -1).double()
    grads.append(gp)
    af = getattr(p, 'active_flat', None)
    for i in range(p.num_coefficients):
        if af is not None:                                # region potential: (channel, region) for --drop_stats
            f = int(af[i]); labels.append(f'{name}[{i}] (ch {f % p.J}, region {f // p.J})')
        else:
            labels.append(f'{name}[{i}]')
G = torch.cat(grads, 1)
G = torch.einsum('nrd,nsd->rs', G, G) / N
dd = G.diagonal().clamp_min(1e-300).sqrt()
A = G / torch.outer(dd, dd)
r = A.shape[0]
print('statistics per family:', {n: p.num_coefficients for n, p in pots.items()}, '| total', r)
mu = torch.linalg.eigvalsh(A)
print(f'Jacobi-scaled Gram on the data: mu_min {float(mu[0]):.3e}, cond {float(mu[-1] / mu[0]):.3e}, '
      f'#mu < 1e-6: {int((mu < 1e-6).sum())}')
iu = torch.triu_indices(r, r, 1)
v = A[iu[0], iu[1]].abs()
order = v.argsort(descending=True)
print(f'\npairs with |A_ij| > 1 - {TOL:g} (near-copies):')
for o in order:
    if v[o] <= 1 - TOL:
        break
    print(f'   1-|A| = {1 - float(v[o]):.2e}   {labels[iu[0, o]]}  ~  {labels[iu[1, o]]}')
print('\nnext 10 most similar pairs (below the threshold):')
k = int((v > 1 - TOL).sum())
for o in order[k:k + 10]:
    print(f'   1-|A| = {1 - float(v[o]):.2e}   {labels[iu[0, o]]}  ~  {labels[iu[1, o]]}')
