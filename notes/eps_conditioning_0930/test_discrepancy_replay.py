# Checks the two facts discrepancy_from_cond.py relies on, on a synthetic problem, using
# SDE._log_cond itself (no run): (1) mu, beta replay the ridge solve exactly for any lam
# (residual and solution norm vs a direct solve); (2) nu = the variance of the theta rhs
# noise along each eigenvector (vs Monte Carlo over independent resamples).
import sys, types, torch, numpy as np
REPO = '/home/chiaraz/phd/MGD-for-Maximum-Entropy-Generation'
sys.path[:0] = [REPO, REPO + '/codes', REPO + '/data']
from codes.sde_routines import SDE
torch.manual_seed(0)
r, B = 12, 4000
fake = types.SimpleNamespace(cond_every=1, num_potentials=r, _live_floor=torch.zeros(r),
                             _cond_log={'theta': []}, _cond_spec={'theta': []})
A = torch.randn(r, r, dtype=torch.float64); C = A @ A.T + 1e-3 * torch.eye(r, dtype=torch.float64)
C[0] *= 1e-4; C[:, 0] *= 1e-4                                  # one badly scaled statistic
live = torch.ones(r, dtype=torch.bool); live[3] = False       # one dead
idx = live.nonzero().flatten()
G = C[idx][:, idx]; D12 = torch.diagonal(G).sqrt()
Gs = G / (D12[:, None] * D12[None, :]); Gs = (Gs + Gs.T) / 2
L = torch.linalg.cholesky(C)
draw = lambda mean: mean + torch.randn(B, r, dtype=torch.float64) @ L.T   # per-sample moments
mI, my = torch.randn(r, dtype=torch.float64), torch.randn(r, dtype=torch.float64)
phiI, phiy = draw(mI), draw(my)
rhs = (phiI.mean(0) - phiy.mean(0))[idx]
SDE._log_cond(fake, 'theta', 0, Gs, D12, live, rhs / D12, noise=(phiI, phiy))
spec = fake._cond_spec['theta'][0].double(); n = int(live.sum())
mu, beta, nu = spec[0, :n], spec[1, :n], spec[2, :n]
ok = True
for lam in (0.0, 1e-4, 1e-2, 1.0):
    th = torch.linalg.solve(Gs + lam * torch.eye(n, dtype=torch.float64), rhs / D12)
    res_direct = (Gs @ th - rhs / D12).norm(); res_replay = torch.sqrt(((lam * beta / (mu + lam)) ** 2).sum())
    sol_replay = torch.sqrt(((beta / (mu + lam)) ** 2).sum())
    e1 = abs(res_direct - res_replay) / max(float(res_direct), 1e-30) if lam else float(res_replay)
    e2 = abs(th.norm() - sol_replay) / th.norm()
    ok &= (e1 < 1e-4) and (e2 < 1e-4)
    print(f'  lam={lam:g}: residual direct {float(res_direct):.6e} replay {float(res_replay):.6e} | '
          f'|theta| direct {float(th.norm()):.6e} replay {float(sol_replay):.6e}')
print(f"{'PASS' if ok else 'FAIL'}  (1) replay == direct solve (float32 storage: rel err < 1e-4)")
# (2) Monte Carlo: resample both sample means R times, project on the logged eigenvectors
_, U = torch.linalg.eigh(Gs)
R = 400
proj = torch.stack([U.T @ ((draw(mI).mean(0) - draw(my).mean(0))[idx] / D12) for _ in range(R)])
nu_mc = proj.var(0)
rel = (nu / nu_mc - 1).abs()
ok2 = bool(rel.max() < 0.35)                                   # sd of a var estimate with R=400 ~ 7%
print(f'  nu / nu_MC: median {float((nu / nu_mc).median()):.3f}, range {float((nu/nu_mc).min()):.3f}-{float((nu/nu_mc).max()):.3f}')
print(f"{'PASS' if ok2 else 'FAIL'}  (2) logged nu matches the Monte Carlo rhs noise variance (R={R})")
