# Part 2: absolute errors; noise-free energy form (exact E_xi[U]) to isolate grid truncation;
# analytic split (differentiate only the theta.phi piece numerically); denoiser n_mc scaling.
exec(open('gauss_oracle_deff.py').read().split("# ----------------------------------------------------------------------------- (c) cross-check")[0])
t_fine = two_phase_schedule(60000, 10000, 0.9, 5e-5)[0].numpy()
t_dense = t_fine[2:]; s_dense = s_of_t(t_dense)
def eval_nodes(n):
    q = np.linspace(s_dense.min(), s_dense.max(), n)
    return t_dense[np.unique([int(np.argmin(np.abs(s_dense - v))) for v in q])]
bands = [(1e2, 1e8), (1, 1e2), (1e-2, 1), (1e-4, 1e-2), (0, 1e-4)]
def summ(name, a, ref, tau):
    e = np.abs(a - ref); c = []
    for lo, hi in bands:
        mk = (tau >= lo) & (tau < hi)
        c.append(f'{np.median(e[mk]):8.2e}/{e[mk].max():8.2e}' if mk.any() else '      -          ')
    print(f'{name:52s} ' + ' '.join(c))
print('ABSOLUTE error (d=16), median/max per tau band: [1e2,..) [1,1e2) [1e-2,1) [1e-4,1e-2) <1e-4')
pots = {'q': DiagQuad(range(d))}
X32 = torch.tensor(x_np).reshape(n_x, 1, d).float()
g = torch.Generator().manual_seed(2)
for n_eval in (240, 1000):
    t = eval_nodes(n_eval); tau = tau_of_t(t); s = s_of_t(t); a = alpha_of_t(t)
    th, m, logZ = exact(t); ref = oracle(x_np, tau)
    print(f'--- {len(t)} eval nodes uniform in s (ds = {np.abs(np.diff(s)).mean():.3f})')
    for nmc in (512, 4096):
        den = denoiser_d_eff(X32, torch.tensor(th), t, pots, n_mc=nmc, generator=g)
        summ(f'(a) denoiser n_mc {nmc}', den, ref, tau)
    en = energy_d_eff(X32, torch.tensor(th), t, pots, torch.tensor(m), logZ - (th*m).sum(1), d, n_mc=4096, generator=g)
    summ('(b1) energy (code), exact logZ, n_mc 4096', en, ref, tau)
    # noise-free: E_xi[theta.phi(sin a y)] = sum theta_i sin^2 a (x_i^2 + tau)
    Ephi = (th[:, None, :] * np.sin(a)[:, None, None] ** 2 * (x_np[None] ** 2 + tau[:, None, None])).sum(-1)
    U = -Ephi + logZ[:, None] - d * np.log(np.sin(a))[:, None]
    for hw in (2, 5):
        summ(f'(b1) energy, EXACT E[U] (no MC), local poly hw={hw}', d - local_poly_derivative(s, U, half_window=hw), ref, tau)
    # analytic split: logZ and d log sin a differentiated analytically, only E[theta.phi] numerically
    dlogZ_ds = local_poly_derivative(s, logZ) * 0  # placeholder replaced below by exact
    v = m; dt_ds = 1 / np.gradient(s, t)  # only for comparison; use exact chain rule:
    dads = -np.sin(a) * np.cos(a)            # s = log cot a  ->  ds/da = -1/(sin a cos a)
    dv_da = 2 * np.sin(a)[:, None] * np.cos(a)[:, None] * (sig2[None] - 1)
    dlogZ = (0.5 * dv_da / v).sum(1) * dads
    dlogsin = d * (np.cos(a) / np.sin(a)) * dads
    Uphi = -Ephi
    for hw in (2, 5):
        dU = local_poly_derivative(s, Uphi, half_window=hw) + dlogZ[:, None] - dlogsin[:, None]
        summ(f'(b1) energy, EXACT E[theta.phi] + analytic logZ/sin, hw={hw}', d - dU, ref, tau)
