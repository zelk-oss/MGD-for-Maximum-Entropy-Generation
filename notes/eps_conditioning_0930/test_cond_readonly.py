# cond_every > 0 must not change anything the run produces (same RNG state, same solver state).
import sys, copy, torch, numpy as np
REPO = '/home/chiaraz/phd/MGD-for-Maximum-Entropy-Generation'
sys.path[:0] = [REPO, REPO + '/codes', REPO + '/data']
from codes.sde_routines import SDE
from codes.filters_bank import return_Filters
from codes.potentials_builder import get_1d_potentials
from codes.utils import normalize
torch.manual_seed(0); np.random.seed(0)
M, n1, J, Q, nt = 64, 64, 4, 1, 20
x1 = normalize(torch.randn(n1, 1, M))
filters, filters_Phi = return_Filters(M, J, 1, device='cpu', include_phi=True)
filters_Q = return_Filters(M, J, Q, device='cpu')
pots = get_1d_potentials(['L_2_lowpass', 'Scattering_Fourth_Order_Mod2_Real_Q1', 'Scalar_psi_GGG', 'Scalar_morlet_GGG'],
                         J, filters, Q, filters_Q=filters_Q, filters_Phi=filters_Phi)
t = (1 - (1 - torch.linspace(0, 1, nt + 1)) ** 2)[:-1]
for f64 in (False, True):
    s = SDE(x1, n1, n1, t, 0.3, pots, n1, device='cpu', regularization=1e-4, interpolant='Cos', solve_float64=f64)
    snap = copy.deepcopy(s.__dict__); rng = torch.get_rng_state()
    outs = []
    for ce in (0, 1):
        s.__dict__.clear(); s.__dict__.update(copy.deepcopy(snap)); torch.set_rng_state(rng)
        s.cond_every = ce
        outs.append(s.forward_regularised(lam=1e-3, n_subsample=1, reg_solver='thomas', reg_ridge=1e-4))
        if ce: log = s.cond_log()
    same = all((a is None and b is None) or np.array_equal(np.asarray(a), np.asarray(b)) if not torch.is_tensor(a)
               else torch.equal(a, b) for a, b in zip(*outs))
    print(f"{'PASS' if same else 'FAIL'}  solve_float64={f64}: 8 outputs bit-identical with cond_every=1: {same}; "
          f"logged eta {tuple(log['eta'].shape)}, theta {tuple(log['theta'].shape)}")
