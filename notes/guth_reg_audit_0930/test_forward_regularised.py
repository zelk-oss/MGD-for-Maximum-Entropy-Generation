"""In-process test of SDE.forward_regularised after the 2026-09-30 changes (tiny sizes:
64 walkers, M = 64, nt = 20, CPU, seconds). Builds the SDE directly (no run_SDE.py).

Run from the repo root:  python notes/guth_reg_audit_0930/test_forward_regularised.py OUTDIR

F1  legacy mode: outputs AND saved system bit-identical to commit 4baf4c2 (same state/RNG).
F2  moment mode: the SDE evolution is unchanged (samples, theta_t, eta_t, dH, moments
    bit-identical to the legacy run); saved mdot == compute_rhs_dt_phi_I_t at each node;
    Sigma symmetric PSD; h == the SDE step; live all True here.
F3  codes/resolve_theta_reg.py on the saved moment system reproduces the in-run Theta_reg.
F4  resolve --mode moment on the saved LEGACY system (Sigma = G - m m^T, mdot = FD of
    barphi_e from aux files written as run_experiment does) runs; distance to F3 reported.
F5  resolve --select on the moment system runs and writes a lam path.
F6  n_subsample = 2 in moment mode runs (block averages, live AND).
F7  resolve --mode legacy --mask theta, lam = 0, ridge = regularization reproduces the
    per-step theta_t at every node (float64 Thomas vs float32 per-step solves).
"""
import copy
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path[:0] = [REPO, REPO + '/codes', REPO + '/data']
from codes.sde_routines import SDE                             # noqa: E402
from codes.filters_bank import return_Filters                  # noqa: E402
from codes.potentials_builder import get_1d_potentials          # noqa: E402
from codes.utils import normalize                               # noqa: E402

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ok_all = True


def report(name, ok, msg):
    global ok_all
    ok_all &= bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}")


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return torch.equal(torch.as_tensor(a), torch.as_tensor(b))


old_src = subprocess.run(['git', '-C', REPO, 'show', '4baf4c2:codes/sde_routines.py'],
                         capture_output=True, text=True, check=True).stdout
with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
    f.write(old_src)
spec = importlib.util.spec_from_file_location('old_sde_routines', f.name)
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old); os.unlink(f.name)

torch.manual_seed(0); np.random.seed(0)
M, n1, J, Q, nt = 64, 64, 4, 1, 20
x1 = normalize(torch.randn(n1, 1, M))
filters, filters_Phi = return_Filters(M, J, 1, device='cpu', include_phi=True)
filters_Q = return_Filters(M, J, Q, device='cpu')
terms = ['L_2_lowpass', 'Scattering_Fourth_Order_Mod2_Real_Q1', 'Scalar_psi_GGG', 'Scalar_morlet_GGG']
potentials = get_1d_potentials(terms, J, filters, Q, filters_Q=filters_Q, filters_Phi=filters_Phi)
t = 1 - (1 - torch.linspace(0, 1, nt + 1)) ** 2
t = t[:-1]                                   # as run_SDE: drop the point at t = 1
solver = SDE(x1, n1, n1, t, 0.3, potentials, n1, device='cpu', regularization=1e-1, interpolant='Cos')
snap = copy.deepcopy(solver.__dict__)
rng = torch.get_rng_state()

common = dict(lam=1e-3, n_subsample=1, reg_solver='thomas', reg_ridge=1e-1)


def run(cls, **kw):
    inst = cls.__new__(cls)
    inst.__dict__.update(copy.deepcopy(snap))
    torch.set_rng_state(rng)
    out = inst.forward_regularised(**{**common, **kw})
    return inst, out


# F1
_, out_new = run(SDE, reg_mode='legacy', reg_system_path=OUT / 'legacy_new.pt')
_, out_old = run(old.SDE, reg_mode='legacy', interp_time_terms=False, reg_system_path=OUT / 'legacy_old.pt')
s_new = torch.load(OUT / 'legacy_new.pt', weights_only=False)
s_old = torch.load(OUT / 'legacy_old.pt', weights_only=False)
outs_same = all(same(a, b) for a, b in zip(out_new, out_old))
keys = ['t', 'M', 'G', 'b', 'c', 'm', 'tau_mean']
sys_same = all(same(s_new[k], s_old[k]) for k in keys)
report('F1 legacy == 4baf4c2', outs_same and sys_same,
       f'8 outputs identical: {outs_same}; system keys {keys} identical: {sys_same}')

# F2
inst, out_mom = run(SDE, reg_mode='moment', reg_system_path=OUT / 'moment.pt')
evo_same = all(same(out_mom[i], out_new[i]) for i in range(6))     # xt, barphi_e/p, eta, theta, dH
s_m = torch.load(OUT / 'moment.pt', weights_only=False)
tt = s_m['t'].numpy()
t_full = inst.t.numpy().astype(np.float64)
idx = [int(np.argmin(np.abs(t_full - x))) for x in tt]
mdot_ref = torch.stack([inst.compute_rhs_dt_phi_I_t(inst.compute_interpolant(k), k).reshape(-1) for k in idx])
mdot_err = float((s_m['mdot'] - mdot_ref.double()).abs().max() / mdot_ref.abs().max())
Sig = torch.stack(s_m['Sigma']).double()
sym_err = float((Sig - Sig.transpose(1, 2)).abs().max())
min_eig = float(torch.linalg.eigvalsh((Sig + Sig.transpose(1, 2)) / 2).min())
h_err = float(np.abs(s_m['h'].numpy() - (t_full[idx] - t_full[np.array(idx) - 1])).max())
report('F2 moment mode', evo_same and mdot_err < 1e-6 and sym_err == 0 and min_eig > -1e-6 and h_err < 1e-12
       and bool(s_m['live'].all()),
       f'SDE evolution identical to legacy: {evo_same}; mdot rel err {mdot_err:.1e}; Sigma sym err '
       f'{sym_err:.1e}, min eig {min_eig:.1e}; h err {h_err:.1e}; live all: {bool(s_m["live"].all())}; '
       f'keys {sorted(k for k in s_m if k not in ("meta", "num_potentials"))}; meta {s_m["meta"]}')

# F3
res_dir = OUT / 'resolve'
cmd = [sys.executable, f'{REPO}/codes/resolve_theta_reg.py', str(OUT / 'moment.pt'), '--lams', '1e-3',
       '--ridge', '1e-1', '--outdir', str(res_dir), '--diagnose', '0', '--overwrite']
p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
if p.returncode:
    print(p.stdout, p.stderr)
r3 = torch.load(res_dir / 'moment' / 'lam0.001_ridge0.1_moment_mask.pt', weights_only=False)
d3 = float((r3['Theta_reg'] - out_mom[6]).abs().max())
report('F3 resolve == in-run', p.returncode == 0 and d3 == 0.0, f'max |diff| {d3:.1e} (mode {r3["mode"]}, mask {r3["mask"]})')

# F4: aux files as run_experiment writes them, config = file stem
root = OUT / 'root' / 'saved_results'
for sub in ('aux_moments', 'sampling_times'):
    (root / sub).mkdir(parents=True, exist_ok=True)
torch.save({'barphi_e': out_new[1], 'barphi_p': out_new[2]}, root / 'aux_moments' / 'legacy_new_aux_moments.pt')
torch.save(t, root / 'sampling_times' / 'legacy_new.pt')
cmd = [sys.executable, f'{REPO}/codes/resolve_theta_reg.py', str(OUT / 'legacy_new.pt'), '--lams', '1e-3',
       '--ridge', '1e-1', '--mode', 'moment', '--results_root', str(OUT / 'root'), '--outdir', str(res_dir),
       '--diagnose', '0', '--overwrite']
p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
if p.returncode:
    print(p.stdout, p.stderr)
else:
    r4 = torch.load(res_dir / 'legacy_new' / 'lam0.001_ridge0.1_moment.pt', weights_only=False)
    d4 = float(((r4['Theta_reg'] - r3['Theta_reg']).norm() / r3['Theta_reg'].norm()))
report('F4 moment on a legacy system', p.returncode == 0,
       (f'rel. distance to F3 {d4:.2e} (Sigma at y_k vs x_k+1, FD vs pathwise mdot, no mask); '
        f'inputs: {r4["inputs"]}') if p.returncode == 0 else 'failed')

# F5
cmd = [sys.executable, f'{REPO}/codes/resolve_theta_reg.py', str(OUT / 'moment.pt'), '--lams',
       '1e-5', '1e-3', '1e-1', '10', '--ridge', '1e-1', '--outdir', str(res_dir), '--diagnose', '0',
       '--select', '--overwrite']
p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
print(p.stdout[-900:])
report('F5 --select', p.returncode == 0 and (res_dir / 'moment' / 'lam_path_ridge0.1_moment_mask.pt').exists(),
       'lam path written' if p.returncode == 0 else p.stderr[-500:])

# F6
_, out6 = run(SDE, reg_mode='moment', n_subsample=2, reg_system_path=OUT / 'moment_nsub2.pt')
s6 = torch.load(OUT / 'moment_nsub2.pt', weights_only=False)
report('F6 moment, n_subsample=2', out6[6] is not None and bool(torch.isfinite(out6[6]).all()),
       f'{len(s6["t"])} nodes (vs {len(tt)} at n_subsample 1), Theta finite')

# F7
(root / 'lagrange_multipliers').mkdir(parents=True, exist_ok=True)
torch.save(out_new[4], root / 'lagrange_multipliers' / 'legacy_new.pt')
cmd = [sys.executable, f'{REPO}/codes/resolve_theta_reg.py', str(OUT / 'legacy_new.pt'), '--lams', '0',
       '--ridge', '1e-1', '--mode', 'legacy', '--mask', 'theta', '--results_root', str(OUT / 'root'),
       '--outdir', str(res_dir), '--diagnose', '0', '--overwrite']
p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
if p.returncode:
    print(p.stdout, p.stderr)
r7 = torch.load(res_dir / 'legacy_new' / 'lam0_ridge0.1_mask.pt', weights_only=False)
t7 = r7['t_reg'].numpy()
idx7 = [int(np.argmin(np.abs(t.numpy()[1:] - x))) for x in t7]
th7 = out_new[4].double()[idx7]
d7 = float((r7['Theta_reg'] - th7).norm() / th7.norm())
report('F7 lam=0 reproduces theta_t', p.returncode == 0 and d7 < 1e-4,
       f'rel. diff {d7:.1e} over {len(t7)} nodes')

print('\nALL PASS' if ok_all else '\nSOME TESTS FAILED')
