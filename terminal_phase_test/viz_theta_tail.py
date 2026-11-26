"""Figures of explore_theta_tail_m1024.ipynb (2026-11-25). Input: the npz of extract_theta_tail.py.
Coefficients are shown as theta_i * sd(phi_i) [nats]: the change of log p when statistic i moves by one standard
deviation (sd over the final walkers of the first seed), so different statistics share one scale."""
import numpy as np
import matplotlib.pyplot as plt

C_THETA, C_A, C_B = '#8c8c8c', '#257de0', '#f47230'        # theta_t (neutral), Theta_reg A, Theta_reg B
SEED_LS = ['-', '--', ':']
ZOOM = 600                                                  # steps before the planned end shown in the zoom


def load(path):
    return dict(np.load(path, allow_pickle=True))


def short(name):
    return (name.replace('Scattering_Fourth_Order_Mod2_', 'Scat4 ').replace('Scattering_Third_Order_', 'Scat3 ')
            .replace('Scalar_psi_gaussianK', 'Scalar_psi').replace('Scalar_morlet_gaussianK', 'Scalar_morlet'))


def step_index(t_run, t):
    """Index of each time t in the run's time grid (exact match expected)."""
    k = np.searchsorted(t_run, t)
    k = np.clip(k, 0, len(t_run) - 1)
    return k


def fig_steps(d):
    """Step size h_k = t_{k+1} - t_k against the step number, seed 0: the tail steps are 3-5 decades smaller."""
    t, k0 = d['s0_t_run'], int(d['s0_k0'])
    h = np.diff(t)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    k = np.arange(len(h))
    ax.semilogy(k[:k0], h[:k0], color='#333333', lw=1.2, label='planned steps')
    ax.semilogy(k[k0:], h[k0:], color=C_B, lw=1.6, label=f'terminal phase (tail): {len(h) - k0} steps')
    ax.set_xlabel('step number k'); ax.set_ylabel('step size h_k')
    ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8, frameon=False)
    ax.set_title(f'tail spans 1 - t from {1 - t[k0]:.3e} to {1 - t[-1]:.3e}', fontsize=9)
    fig.tight_layout()
    return fig


def fig_coefficient(d, j):
    """One coefficient (index j in d['sel']): left = whole run against 1 - t; right = last steps against the step
    number (each tail step gets the same width although it is 1000x shorter). Seed 0 only."""
    i = int(d['sel'][j]); sd = d['sd'][i]; name = short(str(d['names'][i]))
    t_run, k0 = d['s0_t_run'], int(d['s0_k0'])
    series = [(d['s0_t_theta'], d['s0_theta_t_sel'][:, j], C_THETA, 0.8, 'θ_t of the run (each step)'),
              (d['s0_tA'], d['s0_ThA_sel'][:, j], C_A, 2.0, 'Θ_reg, solve A (with the tail)'),
              (d['s0_tB'], d['s0_ThB_sel'][:, j], C_B, 2.0, 'Θ_reg, solve B (without the tail)')]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True, gridspec_kw=dict(width_ratios=[1.3, 1]))
    for t, y, c, lw, lab in series:
        a1.plot(1 - t, y * sd, color=c, lw=lw, label=lab)
        k = step_index(t_run, t)
        m = k >= k0 - ZOOM
        a2.plot(k[m] - k0, y[m] * sd, color=c, lw=lw, label=lab)
    a1.set_xscale('log'); a1.invert_xaxis()
    a1.axvline(1 - t_run[k0], color='k', ls='--', lw=0.8)
    a1.set_xlabel('1 - t   (log scale, time runs to the right)')
    a1.set_ylabel('θ_i · sd(φ_i)  [nats]')
    a2.axvspan(0, len(t_run) - 1 - k0, color=C_B, alpha=0.08, lw=0)
    a2.axvline(0, color='k', ls='--', lw=0.8)
    a2.set_xlabel('steps after the planned end (shaded = tail; not ∝ time)')
    a2.legend(fontsize=8, frameon=False, loc='best')
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.suptitle(f'{name}  (seed 0; dashed line = end of the planned grid)', fontsize=10)
    fig.tight_layout()
    return fig


def final_values(d):
    """(seeds, r) arrays of theta_i * sd_i at the end: theta_t of the last step, Theta_A(1), Theta_B(1)."""
    n = int(d['n_seeds'])
    return {key: np.stack([d[f's{s}_{key}'] for s in range(n)]) * d['sd'][None]
            for key in ('theta_t_final', 'ThA_final', 'ThB_final')}


LABELS = {'theta_t_final': 'θ_t at the last step', 'ThA_final': 'Θ_reg(1), A (with tail)',
          'ThB_final': 'Θ_reg(1), B (without tail)'}
COLORS = {'theta_t_final': C_THETA, 'ThA_final': C_A, 'ThB_final': C_B}


def fig_final_selected(d):
    """Final value of each selected coefficient, one dot per seed, for the three estimates (side by side)."""
    F = final_values(d)
    sel = d['sel']
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for g, key in enumerate(F):
        for s in range(F[key].shape[0]):
            x = np.arange(len(sel)) + (g - 1) * 0.25
            ax.plot(x, F[key][s, sel], 'o', ms=6, color=COLORS[key], mfc='none' if s else COLORS[key],
                    label=LABELS[key] if s == 0 else None)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(np.arange(len(sel)), [short(str(d['names'][i])) for i in sel], rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('θ_i · sd(φ_i)  [nats]')
    ax.set_title('final value, one dot per seed (filled = seed 0)', fontsize=9)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    return fig


def seed_spread_table(d):
    """Per family: median over its coefficients of the std across seeds of theta_i * sd_i [nats]."""
    import pandas as pd
    F = final_values(d)
    fam = d['family']
    rows = {}
    for key, X in F.items():
        sd_seed = X.std(0, ddof=1)
        rows[LABELS[key]] = {f: np.median(sd_seed[fam == f]) for f in dict.fromkeys(fam)}
    T = pd.DataFrame(rows)
    T.index = [short(str(f)) for f in T.index]
    T.loc['all'] = [np.median(F[k].std(0, ddof=1)) for k in F]
    return T


def load_before_after(path):
    """Per-statistic relative errors at the end of the planned grid (before the tail) and after the tail
    (terminal_phase_errors.py csv): rel = (generated - target) / |target|."""
    import pandas as pd
    T = pd.read_csv(path, comment='#')
    T['family'] = [short(n.rsplit('[', 1)[0]) for n in T['name']]
    return T


def fig_moments_before_after(T, threshold=1e-2):
    """|relative error| of every imposed statistic, before and after the tail, grouped by family (log scale)."""
    fig, ax = plt.subplots(figsize=(12, 3.8))
    x = np.arange(len(T))
    for col, c, lab in (('rel_before', C_B, 'before the tail (end of planned grid)'), ('rel_after', C_A, 'after the tail')):
        v = T[col].abs().values
        nz = v > 0                                        # exact zeros (equal to float precision) cannot go on a log axis
        ax.semilogy(x[nz], v[nz], 'o', ms=3.5, color=c, label=f'{lab}; {int((~nz).sum())} exact zeros not drawn')
    ax.axhline(threshold, color='k', ls='--', lw=0.8)
    ax.text(len(T) - 1, threshold * 1.3, f'{threshold:.0%}', ha='right', va='bottom', fontsize=8)
    edges = np.flatnonzero(T['family'].values[1:] != T['family'].values[:-1]) + 1
    for e in edges:
        ax.axvline(e - 0.5, color='#cccccc', lw=0.8)
    starts = np.r_[0, edges]; ends = np.r_[edges, len(T)]
    ax.set_xticks((starts + ends - 1) / 2, T['family'].values[starts], rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('|generated − target| / |target|')
    ax.legend(fontsize=8, frameon=False, loc='upper left'); ax.grid(alpha=0.3, axis='y', which='major')
    fig.tight_layout()
    return fig


def moments_table(T):
    """Per family: number of statistics with |rel| > 1 % and > 0.1 %, before and after the tail."""
    import pandas as pd
    g = T.groupby('family', sort=False)
    return pd.DataFrame({'n': g.size(),
                         '> 1 % before': g['rel_before'].apply(lambda s: int((s.abs() > 1e-2).sum())),
                         '> 1 % after': g['rel_after'].apply(lambda s: int((s.abs() > 1e-2).sum())),
                         '> 0.1 % before': g['rel_before'].apply(lambda s: int((s.abs() > 1e-3).sum())),
                         '> 0.1 % after': g['rel_after'].apply(lambda s: int((s.abs() > 1e-3).sum())),
                         'max before': g['rel_before'].apply(lambda s: s.abs().max()),
                         'max after': g['rel_after'].apply(lambda s: s.abs().max())})


def fig_seed_spread(d):
    """Std across seeds of theta_i * sd_i, every coefficient, for the three estimates (log scale)."""
    F = final_values(d)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    data = [F[k].std(0, ddof=1) for k in F]
    pos = np.arange(len(data))
    for p, (key, v) in zip(pos, zip(F, data)):
        v = v[v > 0]
        ax.plot(p + np.random.default_rng(0).uniform(-0.18, 0.18, len(v)), v, '.', ms=3, color=COLORS[key], alpha=0.5)
        ax.plot([p - 0.25, p + 0.25], [np.median(v)] * 2, color='k', lw=2)
    ax.set_yscale('log')
    ax.set_xticks(pos, [LABELS[k] for k in F], fontsize=8)
    ax.set_ylabel('std across seeds of θ_i · sd(φ_i)  [nats]')
    ax.set_title('one dot per coefficient; black bar = median', fontsize=9)
    ax.grid(alpha=0.3, axis='y', which='both')
    fig.tight_layout()
    return fig
