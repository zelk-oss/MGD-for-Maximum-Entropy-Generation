# Old-format system (no m) + aux_moments/sampling_times -> resolve_theta_reg in all modes;
# check fixed from aux equals a direct solve with the same m.
import os as _os
REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
import sys, types, subprocess, io, contextlib, numpy as np, torch
from pathlib import Path
sys.path[:0] = [REPO,
                REPO + '/codes',
                REPO + '/data']
from codes.sde_routines import SDE
import time; root = Path(sys.argv[1]) / f'e2e_{time.time_ns()}'; torch.manual_seed(0); cfg = 'fakecfg_seed_0'
for d in ['saved_results/aux_moments', 'saved_results/sampling_times', 'sys', 'out']:
    (root / d).mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0); r, nfine = 5, 80
t_fine = np.linspace(0, 0.99, nfine + 1).astype(np.float32)   # float32 grid, like old runs
barphi_p = torch.tensor(rng.standard_normal((nfine, r)))
torch.save({'barphi_e': barphi_p, 'barphi_p': barphi_p}, root / f'saved_results/aux_moments/{cfg}_aux_moments.pt')
torch.save(torch.tensor(t_fine), root / f'saved_results/sampling_times/{cfg}.pt')
keep = np.arange(3, nfine, 2)                                  # system nodes = some fine steps
t_sys = t_fine[1:][keep].astype(np.float64)
spd = lambda: (lambda A: torch.tensor(A @ A.T + np.eye(r), dtype=torch.float32))(rng.standard_normal((r, r)))
sysd = {'t': torch.tensor(t_sys), 'M': [spd() for _ in keep], 'G': [spd() * 10 for _ in keep],
        'b': [torch.randn(r) for _ in keep], 'c': [torch.randn(r) for _ in keep],
        'num_potentials': r, 'meta': {}}
torch.save(sysd, root / 'sys' / f'{cfg}.pt')
for mode, extra in [('legacy', []), ('fixed', []), ('guth', ['--dim', '256'])]:
    out = subprocess.run([sys.executable, 'codes/resolve_theta_reg.py', str(root / 'sys' / f'{cfg}.pt'),
                          '--lams', '0', '1e-3', '--mode', mode, '--results_root', str(root),
                          '--outdir', str(root / 'out'), '--diagnose', '0', *extra],
                         capture_output=True, text=True)
    print(mode, 'exit', out.returncode, '|', [l for l in out.stdout.splitlines() if 'mode=' in l or 'FAIL' in l], out.stderr[-300:])
print(sorted(p.name for p in (root / 'out' / cfg).iterdir()))
res = torch.load(root / 'out' / cfg / 'lam0.001_ridge0_fixed.pt', weights_only=False)
s = types.SimpleNamespace(num_potentials=r)
with contextlib.redirect_stdout(io.StringIO()):
    direct = SDE._solve_regularised_thomas(s, t_sys, sysd['M'], sysd['G'], sysd['b'], sysd['c'], 1e-3,
                                           mode='fixed', m=barphi_p[keep].double())
print('m_source:', res['m_source'], '| fixed via aux == direct:', torch.equal(res['Theta_reg'], direct[1:]))
