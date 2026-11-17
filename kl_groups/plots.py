"""Plots for the kl_groups notebooks (read H_<dataset>.csv / H_vs_N_<dataset>.npz of compute_H.py; nothing is recomputed).
Across seeds: value = mean over seeds, error bar = std over seeds (seed spread); 'boot' = mean bootstrap std within a seed.
All D in nats. D_k = D_KL(p||q_k), A_k = theta_k^T phi_k."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RES = Path(os.environ.get('KL_RESULTS', Path(__file__).resolve().parent / 'results'))
SHORT = {'Scattering_': 'S', 'Fourth_Order_Mod2_': '4', 'Third_Order_': '3', 'Second_Order': '2', 'First_Order': '1',
         '_gaussianK': '', 'Scalar_': 'Sc_'}


def short(s):
    for a, b in SHORT.items():
        s = s.replace(a, b)
    return s


ESS_MIN_ABS = 100    # reliable = many signals share the weight (absolute ESS), not ESS/N: small ESS/N is normal for fat tails
DRIFT_MAX = 2.0      # and H(N) - H(N/2) <= DRIFT_MAX bootstrap std
KHAT_MAX = 0.5       # and Pareto khat < 0.5 (finite variance of the weights e^{A}); used only if the CSV has khat


def load(dataset, res=RES):
    """Per-seed flag RECOMPUTED here (overrides the CSV 'flag' of compute_H.py, which used ESS/N < 0.1 or max w > 1 %):
    lower bound  <=>  ESS = (ess_frac * N) < ESS_MIN_ABS  or  drift > DRIFT_MAX * boot_std."""
    df = pd.read_csv(res / f'H_{dataset}.csv')
    df['ESS'] = df.ess_frac * df.N
    df['flag'] = (df.ESS < ESS_MIN_ABS) | (df.drift > DRIFT_MAX * df.boot_std)
    # CSVs from 2026-11-13 on also carry khat (Pareto tail shape of the weights) and D_lb95 (rigorous 95 % lower bound).
    # khat >= 0.5: infinite-variance weights -> ESS / bootstrap / drift are not valid error measures (d16 Gaussian bench:
    # all such groups rated 'measured' by ESS alone were biased low by 0.1-2 nats).
    if 'khat' in df:
        df['flag'] |= ~(df.khat < KHAT_MAX)
    df['regime'] = np.where(df.flag, 'lower bound', 'measured')
    # Gaussian A: reliable iff N >> e^{D_KL(q_k||p)} = e^{Var(A)/2} = e^{var2} (Chatterjee-Diaconis); the CSV used e^{Var A}
    df['ln_N_needed'] = df.var2
    agg = df.groupby(['group', 'kind', 'family', 'category', 'j', 'jp', 'label', 'n_terms'], dropna=False).agg(
        H=('H', 'mean'), H_seed_std=('H', 'std'), H_min=('H', 'min'), H_max=('H', 'max'),
        boot=('boot_std', 'mean'), var2=('var2', 'mean'), var2_seed_std=('var2', 'std'),
        ess=('ess_frac', 'min'), ESS_min=('ESS', 'min'), wmax=('wmax', 'max'), drift=('drift', 'mean'), flag=('flag', 'any'),
        ln_N_needed=('ln_N_needed', 'mean'), n_seeds=('seed', 'nunique'),
        H_seeds=('H', list), var2_seeds=('var2', list), flag_seeds=('flag', list), N_seeds=('N', list),
        **({'khat_max': ('khat', 'max'), 'D_lb95_min': ('D_lb95', 'min'), 'D_lb_seeds': ('D_lb95', list)}
           if 'khat' in df else {})).reset_index()
    # a group is a lower bound if it is flagged in ANY seed: then D_k >= mean estimate, no error bar
    agg['regime'] = np.where(agg.flag, 'lower bound', 'measured')
    agg['err'] = np.maximum(agg.H_seed_std.fillna(0), agg.boot)
    return df, agg


def _bar_groups(agg):
    """The groups drawn by bars(), in the same order (descending mean D_hat)."""
    d = agg[((agg.kind == 'family') & (agg.category != 'scalar')) | ((agg.kind == 'category') & (agg.family == 'scalar'))]
    return d.sort_values('H', ascending=False)


def bars_table(agg):
    """Numbers behind bars(), same rows and order. ESS = (sum w)^2 / sum w^2 of the weights e^{a - mean} (absolute, out of N),
    worst seed; wmax = largest single-sample share of the weight, worst seed; khat_max = Pareto tail shape, worst seed."""
    d = _bar_groups(agg)
    t = pd.DataFrame({
        'group': [short(g.split(':', 1)[1]) for g in d.group], 'n_terms': d.n_terms.values,
        'D_hat (mean)': d.H.values, '1/2 Var': d.var2.values, 'ESS_min': d.ESS_min.values, 'N': [int(max(n)) for n in d.N_seeds],
        'wmax': d.wmax.values, 'regime': d.regime.values})
    if 'khat_max' in d:
        t['khat_max'] = d.khat_max.values
    return t.set_index('group')


def bars(agg, ax=None, title=''):
    """One column per group: scalar banks summed into ONE group, every other family on its own.
    Per seed: H estimate (filled = measured, open = lower bound), 1/2 Var(A_k) (x)."""
    d = _bar_groups(agg)
    ax = ax or plt.subplots(figsize=(max(5, 0.8 * len(d)), 3.8))[1]
    for i, (_, r) in enumerate(d.iterrows()):
        n = len(r.H_seeds)
        x = i + np.linspace(-0.25, 0.25, n)
        H, f = np.array(r.H_seeds), np.array(r.flag_seeds, bool)
        ax.plot(x[~f], H[~f], 'o', color='C0', ms=5, label='$\\hat D_k$ measured' if i == 0 else None)
        ax.plot(x[f], H[f], 'o', mfc='none', color='C0', ms=5, label='$\\hat D_k$, lower bound ($D_k \\geq$)' if i == 0 else None)
        ax.plot(x, r.var2_seeds, 'x', color='C1', ms=5, label=r'$\frac{1}{2}\mathrm{Var}(A_k)$' if i == 0 else None)
        # spread across seeds of D_hat: mean +- std (ddof 1). For lower bounds this is the spread of the bounds, not an
        # error bar on D_k. Log axis: the lower whisker is clipped at a small positive value when std >= mean.
        m, s = float(np.mean(H)), float(np.std(H, ddof=1)) if n > 1 else 0.0
        ax.errorbar(i + 0.38, m, yerr=[[min(s, 0.999 * m)], [s]], fmt='D', ms=5, color='k', capsize=3,
                    label=r'mean $\pm$ std over seeds of $\hat D_k$' if i == 0 else None)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(d)), [short(g.split(':', 1)[1]) for g in d.group], rotation=45, ha='right')
    ax.set_ylabel('nats'); ax.set_title(f'{title}: one marker per seed')
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1))
    return ax


def _points(ax, x, d, **kw):
    """measured: filled circles with error bars; lower bounds: open up-triangles (D_k >= value), no error bar."""
    m, lb = ~d.flag.values, d.flag.values
    ax.errorbar(x[m], d.H.values[m], d.err.values[m], fmt='o', ms=3, capsize=2, **kw)
    ax.plot(x[lb], d.H.values[lb], '^', mfc='none', ms=6, color=kw.get('color', 'C3'),
            label=None if 'label' in kw else r'$D_k \geq$ (lower bound)')


def singles(agg, family_substr, ax=None, title=''):
    """Every single term of the family, in the order of the potential, NOT sorted by value.
    Scat4 (j, j', l): ordered by j, then j', then l; solid lines separate the j blocks, dotted lines the j' sub-blocks.
    Other families: ordered by the term index [i] of the potential list."""
    d = agg[(agg.kind == 'single') & agg.family.str.contains(family_substr)].copy()
    d['idx'] = d.group.str.extract(r'\[(\d+)\]$')[0].astype(int)
    scat4 = d.label.notna().all() and len(d) > 0
    if scat4:
        jjl = d.label.astype(str).str.split(',', expand=True).astype(int)
        d['J'], d['JP'], d['L'] = jjl[0].values, jjl[1].values, jjl[2].values
        d = d.sort_values(['J', 'JP', 'L'])
    else:
        d = d.sort_values('idx')
    n = len(d)
    ax = ax or plt.subplots(figsize=(min(20, max(6, 0.22 * n)), 3.8))[1]
    x = np.arange(n)
    _points(ax, x, d, color='C0', label=r'$D_k$ measured')
    ax.plot([], [], '^', mfc='none', color='C0', label=r'$D_k \geq$ (lower bound)')
    ax.plot(x, d.var2, 'x', ms=4, color='C1', label=r'$\frac{1}{2}\mathrm{Var}(A_k)$')
    if scat4:
        ax.set_xticks(x, [f'{l}' for l in d.label], rotation=90, fontsize=6)
        jv, jpv = d.J.values, d.JP.values
        for i in range(1, n):
            if jv[i] != jv[i - 1]:
                ax.axvline(i - 0.5, color='k', lw=1)
            elif jpv[i] != jpv[i - 1]:
                ax.axvline(i - 0.5, color='0.6', lw=0.7, ls=':')
        starts = [0] + [i for i in range(1, n) if jv[i] != jv[i - 1]] + [n]
        for a, b in zip(starts[:-1], starts[1:]):
            ax.text((a + b - 1) / 2, 1.01, f'j = {jv[a]}', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=8)
        ax.set_xlabel("single term (j, j', l); solid line = new j, dotted = new j'")
    else:
        ax.set_xticks(x[::5], [str(i) for i in d.idx.values[::5]], rotation=90, fontsize=6)
        ax.set_xlabel('term index in the potential list')
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylabel('nats'); ax.set_yscale('log')
    ax.set_title(f'{title}: {n} single terms, potential order' + ('\n' if scat4 else ''), fontsize=9)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    return ax


def pairs(agg, family_substr, title='', angle='all'):
    d = agg[(agg.kind == 'pair') & agg.family.str.contains(family_substr) & (agg.label == angle)].copy()
    d['dj'] = (d.jp - d.j).abs()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for i, (j, g) in enumerate(d.groupby('j')):
        g = g.sort_values('dj')
        a1.plot(g.dj, g.H, '-', color=f'C{i % 10}', lw=0.8)
        _points(a1, g.dj.values, g, color=f'C{i % 10}', label=f'j = {j}')
    a1.plot([], [], '^', mfc='none', color='k', label=r'$H \geq$ (lower bound)')
    a1.set_yscale('log'); a1.set_xlabel("|j - j'|"); a1.set_ylabel(r"$D_{j,j'}$ (all l), nats")
    a1.set_title(f"{title}: H_(j,j') vs |j - j'|"); a1.legend(fontsize=7, ncol=2)
    J = int(max(d.j.max(), d.jp.max())) + 1
    Hm = np.full((J, J), np.nan)
    for _, r in d.iterrows():
        Hm[int(r.j), int(r.jp)] = r.H
    im = a2.imshow(np.log(Hm), origin='lower', cmap='viridis')
    plt.colorbar(im, ax=a2, label=r"$\ln D_{j,j'}$ (nats)")
    a2.set_xlabel("j'"); a2.set_ylabel('j'); a2.set_title("(j, j') heatmap, all l")
    for _, r in d[d.flag].iterrows():
        a2.text(r.jp, r.j, r'$\geq$', color='r', ha='center', va='center')       # lower bound only
    fig.tight_layout()
    return d


def diagnostics(agg, dataset, res=RES, top=5):
    C = np.load(res / f'H_vs_N_{dataset}.npz')
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5))
    nonsingle = agg[agg.kind != 'single']
    a1.scatter(nonsingle.H, nonsingle.ESS_min, s=8, c=np.where(nonsingle.flag, 'r', 'C0'))
    a1.set_xscale('log'); a1.set_yscale('log'); a1.axhline(ESS_MIN_ABS, ls=':', c='k', label=f'ESS = {ESS_MIN_ABS} (flag below)')
    a1.set_xlabel(r'$\hat D_k$ (nats)'); a1.set_ylabel('ESS (min over seeds)'); a1.legend(fontsize=8)
    a1.set_title(r'weights $e^{A_k}$: effective sample size')
    # curves are stored for the 5 largest groups OF EACH SEED: show the `top` groups stored in most seeds, all their seeds
    names = pd.Series([k.split('|', 1)[1] for k in C.files])
    keep = list(names.value_counts().index[:top])
    for k in C.files:
        seed, g = k.split('|', 1)
        if g not in keep:
            continue
        n, h, s = C[k].T
        i = keep.index(g)
        first = all(not kk.endswith('|' + g) for kk in C.files[:C.files.index(k)])
        a2.errorbar(n, h, s, marker='o', ms=3, capsize=2, color=f'C{i}', lw=0.8,
                    label=short(g.split(':', 1)[1])[:30] + ' (all seeds)' if first else None)
    a2.set_xscale('log'); a2.set_xlabel('N samples'); a2.set_ylabel(r'$\hat D_k$ (nats)')
    a2.set_title('H vs N (should flatten; still rising = biased low)'); a2.legend(fontsize=6)
    fig.tight_layout()


def summary(agg, kinds=('category', 'family', 'pair')):
    d = agg[agg.kind.isin(kinds)].sort_values('H', ascending=False)
    d = d.assign(name=d.group.map(short),
                 D_k=[f'>= {h:.3g}' if f else f'{h:.3g} +- {e:.2g}' for h, e, f in zip(d.H, d.err, d.flag)],
                 ratio=d.H / d.var2)
    # H_seed_std / H_min / H_max: spread of the per-seed estimates (for a lower bound it is NOT an error bar)
    extra = ['khat_max', 'D_lb95_min'] if 'khat_max' in d else []     # D_lb95_min: rigorous 95 % lower bound, worst seed
    return d[['name', 'n_terms', 'D_k', 'regime'] + extra + ['H_seed_std', 'H_min', 'H_max', 'boot', 'var2', 'var2_seed_std',
              'ratio', 'ln_N_needed', 'ESS_min', 'ess', 'wmax', 'drift', 'n_seeds']]


def vs_prediction(agg, pred, title=''):
    """Gaussian in-family test (gauss_scat4_test): measured D_hat_k vs the linear-response prediction of the finite-data
    D_k (pred_H.csv, from the fitted theta - theta_star; truth with infinite data is D_k = 0 for every L_6/Scat4 group).
    One point per group (mean over seeds); filled = measured in all seeds, open = lower bound in some seed.
    Only Scat4 groups: there theta_true = 0, so fitted theta = dtheta and D_k ~ 1/2 dtheta^T Cov dtheta is the prediction.
    For Scat2 pred_H is the size of the NOISE in theta (dtheta), not the D_k of removing Scat2: not comparable."""
    d = agg[agg.family.str.contains('Fourth')].merge(pred[['group', 'H_pred']], on='group')
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for i, kind in enumerate(('family', 'pair', 'single')):
        g = d[d.kind == kind]
        m = ~g.flag.values
        ax.errorbar(g.H_pred[m], g.H[m], g.err[m], fmt='o', ms=4, capsize=2, color=f'C{i}', label=f'{kind}, measured')
        ax.plot(g.H_pred[~m], g.H[~m], 'o', mfc='none', ms=5, color=f'C{i}', label=f'{kind}, lower bound' if (~m).any() else None)
    lo, hi = d[['H_pred', 'H']].min().min() * 0.5, d[['H_pred', 'H']].max().max() * 2
    ax.plot([lo, hi], [lo, hi], 'k:', lw=1, label=r'$\hat D_k$ = prediction')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'predicted finite-data $D_k$ (nats)'); ax.set_ylabel(r'measured $\hat D_k$ (nats, mean over seeds)')
    ax.set_title(title); ax.legend(fontsize=7)
    return d[['group', 'kind', 'n_terms', 'H', 'err', 'H_pred', 'flag']].assign(ratio=d.H / d.H_pred)


def per_seed(df, kinds=('category', 'family')):
    """H of every seed side by side (rows = groups, columns = seeds); '*' = flagged (lower bound) in that seed."""
    d = df[df.kind.isin(kinds)]
    cell = [f'{h:.3g}' + ('*' if f else '') for h, f in zip(d.H, d.flag)]
    return d.assign(name=d.group.map(short), cell=cell).pivot(index='name', columns='seed', values='cell')
