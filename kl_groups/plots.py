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


def load(dataset, res=RES):
    """Per-seed flag RECOMPUTED here (overrides the CSV 'flag' of compute_H.py, which used ESS/N < 0.1 or max w > 1 %):
    lower bound  <=>  ESS = (ess_frac * N) < ESS_MIN_ABS  or  drift > DRIFT_MAX * boot_std."""
    df = pd.read_csv(res / f'H_{dataset}.csv')
    df['ESS'] = df.ess_frac * df.N
    df['flag'] = (df.ESS < ESS_MIN_ABS) | (df.drift > DRIFT_MAX * df.boot_std)
    df['regime'] = np.where(df.flag, 'lower bound', 'measured')
    # Gaussian A: reliable iff N >> e^{D_KL(q_k||p)} = e^{Var(A)/2} = e^{var2} (Chatterjee-Diaconis); the CSV used e^{Var A}
    df['log10_N_needed'] = df.var2 / np.log(10)
    agg = df.groupby(['group', 'kind', 'family', 'category', 'j', 'jp', 'label', 'n_terms'], dropna=False).agg(
        H=('H', 'mean'), H_seed_std=('H', 'std'), H_min=('H', 'min'), H_max=('H', 'max'),
        boot=('boot_std', 'mean'), var2=('var2', 'mean'), var2_seed_std=('var2', 'std'),
        ess=('ess_frac', 'min'), ESS_min=('ESS', 'min'), wmax=('wmax', 'max'), drift=('drift', 'mean'), flag=('flag', 'any'),
        log10_N_needed=('log10_N_needed', 'mean'), n_seeds=('seed', 'nunique'),
        H_seeds=('H', list), var2_seeds=('var2', list), flag_seeds=('flag', list), N_seeds=('N', list)).reset_index()
    # a group is a lower bound if it is flagged in ANY seed: then D_k >= mean estimate, no error bar
    agg['regime'] = np.where(agg.flag, 'lower bound', 'measured')
    agg['err'] = np.maximum(agg.H_seed_std.fillna(0), agg.boot)
    return df, agg


def gauss_cap(var2, N):
    """Gaussian FLOOR of the H estimate: H_hat >= max_i(a_i - mean) - ln N (exact), and for Gaussian A with Var = 2*var2
    max_i(a_i - mean) ~ sigma sqrt(2 ln N). Gaussian A: H_hat ~ var2 if N >> exp(var2), ~ floor if N << exp(var2);
    var2 - floor = (sigma/sqrt2 - sqrt(ln N))^2 >= 0, zero exactly at the reliability threshold N = exp(sigma^2/2)."""
    var2, N = np.asarray(var2, float), np.asarray(N, float)
    return np.sqrt(2 * var2) * np.sqrt(2 * np.log(N)) - np.log(N)


def bars(agg, ax=None, title=''):
    """One column per group: scalar banks summed into ONE group, every other family on its own.
    Per seed: H estimate (filled = measured, open = lower bound), 1/2 Var(A_k) (x), Gaussian floor (-)."""
    d = agg[((agg.kind == 'family') & (agg.category != 'scalar')) | ((agg.kind == 'category') & (agg.family == 'scalar'))]
    d = d.sort_values('H', ascending=False)
    ax = ax or plt.subplots(figsize=(max(5, 0.8 * len(d)), 3.8))[1]
    for i, (_, r) in enumerate(d.iterrows()):
        n = len(r.H_seeds)
        x = i + np.linspace(-0.25, 0.25, n)
        H, f = np.array(r.H_seeds), np.array(r.flag_seeds, bool)
        ax.plot(x[~f], H[~f], 'o', color='C0', ms=5, label='$\\hat D_k$ measured' if i == 0 else None)
        ax.plot(x[f], H[f], 'o', mfc='none', color='C0', ms=5, label='$\\hat D_k$, lower bound ($D_k \\geq$)' if i == 0 else None)
        ax.plot(x, r.var2_seeds, 'x', color='C1', ms=5, label=r'$\frac{1}{2}\mathrm{Var}(A_k)$' if i == 0 else None)
        ax.plot(x, gauss_cap(r.var2_seeds, r.N_seeds), '_', color='0.4', ms=9, mew=1.5,
                label=r'Gaussian floor $\sigma\sqrt{2\ln N}-\ln N$ (one sample dominates)' if i == 0 else None)
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


def singles(agg, family_substr, top=40, ax=None, title=''):
    d = agg[(agg.kind == 'single') & agg.family.str.contains(family_substr)].sort_values('H', ascending=False).head(top)
    ax = ax or plt.subplots(figsize=(max(5, 0.25 * len(d)), 3.5))[1]
    x = np.arange(len(d))
    _points(ax, x, d, color='C0', label=r'$D_k$ measured')
    ax.plot([], [], '^', mfc='none', color='C0', label=r'$D_k \geq$ (lower bound)')
    ax.plot(x, d.var2, 'x', ms=4, color='C1', label=r'$\frac{1}{2}\mathrm{Var}(A_k)$')
    ax.plot(x, gauss_cap(d.var2, [np.mean(n) for n in d.N_seeds]), '_', color='0.4', ms=7, mew=1.5,
            label=r'Gaussian floor $\sigma\sqrt{2\ln N}-\ln N$')
    # Scat4 terms carry (j, j', l); other families only an index: 'one:<family>[i]' -> '<short family>[i]'
    ticks = [f'({l})' if isinstance(l, str) and l else short(g.split(':', 1)[1]) for g, l in zip(d.group, d.label)]
    ax.set_xticks(x, ticks, rotation=90, fontsize=7)
    ax.set_xlabel("single term ((j, j', l) for Scat4)"); ax.set_ylabel('nats'); ax.set_yscale('log')
    ax.set_title(f'{title} top {len(d)} single terms'); ax.legend(fontsize=8)
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
    im = a2.imshow(np.log10(Hm), origin='lower', cmap='viridis')
    plt.colorbar(im, ax=a2, label=r"$\log_{10} D_{j,j'}$ (nats)")
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
    return d[['name', 'n_terms', 'D_k', 'regime', 'H_seed_std', 'H_min', 'H_max', 'boot', 'var2', 'var2_seed_std',
              'ratio', 'log10_N_needed', 'ESS_min', 'ess', 'wmax', 'drift', 'n_seeds']]


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
