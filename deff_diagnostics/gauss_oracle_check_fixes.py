"""Exact-Gaussian check of the 2026-10-13 fixes in codes/effective_dimension.py (common_noise,
float64, h_from_theta_reg). Reuses the problem set-up of gauss_oracle_deff.py."""
exec(open('gauss_oracle_deff.py').read().split("# ----------------------------------------------------------------------------- (c) cross-check")[0])
from effective_dimension import h_from_theta_reg
t_fine = two_phase_schedule(60000, 10000, 0.9, 5e-5)[0].numpy()
t_dense = t_fine[2:]; s_dense = s_of_t(t_dense)
q = np.linspace(s_dense.min(), s_dense.max(), 240)
ev = np.unique([int(np.argmin(np.abs(s_dense - v))) for v in q])
t = t_dense[ev]; tau = tau_of_t(t)
th, m, logZ = exact(t); ref = oracle(x_np, tau)
pots = {'q': DiagQuad(range(d))}
X32 = torch.tensor(x_np).reshape(n_x, 1, d).float()
bands = [(1e2, 1e8), (1, 1e2), (1e-2, 1), (1e-4, 1e-2), (0, 1e-4)]
def summ(name, a):
    e = np.abs(a - ref)
    print(f'{name:58s} ' + ' '.join(f'{np.median(e[(tau>=lo)&(tau<hi)]):8.2e}/{e[(tau>=lo)&(tau<hi)].max():8.2e}' for lo, hi in bands))
print('ABSOLUTE error median/max per tau band: [1e2,..) [1,1e2) [1e-2,1) [1e-4,1e-2) <1e-4')
g = torch.Generator().manual_seed(5)
H_ex = logZ - (th * m).sum(1)
for nmc in (64, 512):
    summ(f'energy OLD (fresh xi per node, f32 phi), n_mc {nmc}',
         energy_d_eff(X32, torch.tensor(th), t, pots, torch.tensor(m), H_ex, d, n_mc=nmc, generator=g, common_noise=False, float64=False))
    summ(f'energy NEW defaults (common noise, f64), n_mc {nmc}',
         energy_d_eff(X32, torch.tensor(th), t, pots, torch.tensor(m), H_ex, d, n_mc=nmc, generator=g))
summ('denoiser NEW default (f64), n_mc 512', denoiser_d_eff(X32, torch.tensor(th), t, pots, n_mc=512, generator=g))

# h_from_theta_reg on the DENSE grid with exact Theta: log Z should be exact up to a constant
thd, md, logZd = exact(t_dense)
Hc = h_from_theta_reg(thd, md, d)
err = (Hc + (thd * md).sum(1)) - logZd
print(f'\nh_from_theta_reg (exact Theta, dense grid): log Z - exact = const + {np.ptp(err):.2e} (spread over nodes)')
summ('energy NEW with h_from_theta_reg (dense->eval), n_mc 512',
     energy_d_eff(X32, torch.tensor(th), t, pots, torch.tensor(m), Hc[ev], d, n_mc=512, generator=g))

# near-copy pair, |theta| = K
print("\nnear-copy pair +-K: median relative error")
pots2 = {'q': DiagQuad(range(d)), 'a': DiagQuad([0]), 'b': DiagQuad([0], scale=[1 + 1e-3])}
for K in (1e6, 1e8):
    TH2 = torch.cat([torch.tensor(th), torch.full((len(t), 1), K, dtype=torch.float64),
                     torch.full((len(t), 1), -K / (1 + 1e-3), dtype=torch.float64)], 1)
    m2 = torch.cat([torch.tensor(m), torch.tensor(m[:, :1]), torch.tensor(m[:, :1] * (1 + 1e-3))], 1)
    H2 = H_ex - (TH2[:, d:] * m2[:, d:]).sum(1).numpy()
    r = lambda a: np.median(np.abs(a - ref) / ref)
    print(f'  K={K:.0e}: denoiser old {r(denoiser_d_eff(X32, TH2, t, pots2, 256, g, float64=False)):.2e}'
          f' new {r(denoiser_d_eff(X32, TH2, t, pots2, 256, g)):.2e} | energy old '
          f'{r(energy_d_eff(X32, TH2, t, pots2, m2, H2, d, 256, g, common_noise=False, float64=False)):.2e}'
          f' new {r(energy_d_eff(X32, TH2, t, pots2, m2, H2, d, 256, g)):.2e}')
