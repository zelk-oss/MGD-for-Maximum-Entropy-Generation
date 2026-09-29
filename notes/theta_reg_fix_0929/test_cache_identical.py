# In one process: run forward_regularised three times from the same Solver state and
# RNG state -- (A) cache ON + interp terms ON (new code), (B) cache OFF (the corrector
# no longer leaves phi(y_k), phi(I_k) behind, so the loop recomputes them as the old
# code did) + interp OFF, (C) cache ON + interp OFF -- and compare everything bit-wise.
import sys, copy, types, torch, numpy as np
sys.argv = ['run_SDE.py', '--n1', '64', '--M', '64', '--J', '4', '--Q', '1', '--nt', '20',
            '--hurst', '0.5', '--n_subsample', '1', '--reg_solver', 'thomas', '--reg_ridge', '1e-4',
            '--save_reg_system', '--reg_system_dir', sys.argv[1] + '/cache_sys', '--outdir',
            sys.argv[1] + '/cache_out', '--label', 'cache', '--seed', '0', '--force_rerun']
sys.path[:0] = ['gaussian_experiment', '.', 'codes', 'data']
import codes.sde_routines as sr
SDE = sr.SDE
orig_fr, orig_rhs = SDE.forward_regularised, SDE.compute_rhs_constraint_correction

def rhs_nocache(self, x_k, I_k):                     # the pre-2026-09-29 body
    return self.compute_moments(I_k).mean(0) - self.compute_moments(x_k).mean(0)

results = {}
def patched(self, **kw):
    snap = copy.deepcopy(self.__dict__)
    rng = torch.get_rng_state(); nprng = np.random.get_state()
    out = None
    for name, cache, interp in [('A', True, True), ('B', False, False), ('C', True, False)]:
        self.__dict__.clear(); self.__dict__.update(copy.deepcopy(snap))
        torch.set_rng_state(rng); np.random.set_state(nprng)
        SDE.compute_rhs_constraint_correction = orig_rhs if cache else rhs_nocache
        path = kw['reg_system_path']
        kw2 = dict(kw, reg_system_path=path.with_name(f'{name}_' + path.name), interp_time_terms=interp)
        res = orig_fr(self, **kw2)
        results[name] = (res, torch.load(kw2['reg_system_path'], weights_only=False))
        out = out or res
    SDE.compute_rhs_constraint_correction = orig_rhs
    return results['A'][0]
SDE.forward_regularised = patched

import run_SDE
from pathlib import Path
for d in ['samples', 'lagrange_multipliers', 'lagrange_multipliers_regularised', 'entropy_bounds',
          'sampling_times', 'sampling_times_regularised', 'aux_moments']:
    Path(sys.argv[sys.argv.index('--outdir') + 1], 'saved_results', d).mkdir(parents=True, exist_ok=True)
try:
    run_SDE.main()
except SystemExit:
    pass

def eq(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a, b)
    if isinstance(a, np.ndarray): return np.array_equal(a, b)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict): return all(eq(a[k], b[k]) for k in a)
    return a == b
names = ['xt', 'barphi_e', 'barphi_p', 'eta_t', 'theta_t', 'dH', 'Theta_reg', 't_reg']
for other in ['B', 'C']:
    ra, sa = results['A']; rb, sb = results[other]
    print(f'A vs {other}: outputs', {n: eq(x, y) for n, x, y in zip(names, ra, rb)})
    print(f'          system  ', {k: eq(sa[k], sb[k]) for k in ['t', 'M', 'G', 'b', 'c', 'm', 'tau_mean']})
sa = results['A'][1]
print('interp keys in A:', [k for k in sa if k.endswith('_I')], '| in B:', [k for k in results['B'][1] if k.endswith('_I')])
t = sa['t'].numpy()
for k in range(0, len(t), 5):
    c, m, cI, mI = sa['c'][k].double(), sa['m'][k], sa['c_I'][k].double(), sa['m_I'][k]
    cos = lambda u, v: float(u @ v / (u.norm() * v.norm()))
    print(f"t={t[k]:.3f} walkers: tau_mean {float(sa['tau_mean'][k]):9.2f} cos(c,m) {cos(c, m):7.4f} | "
          f"interpolant: tau_mean {float(sa['tau_mean_I'][k]):8.2f} cos(c,m) {cos(cI, mI):7.4f}")
