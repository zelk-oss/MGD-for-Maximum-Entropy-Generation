"""Figures of the nested-model analysis, from the dicts of logp_entropy.load_all (same layout per model).

    plot_nlogp          -log p on model samples (filled) and on data (line), centred on the model mean
    plot_concentration  std of -log p on model samples per model, mean +- std over seeds
    plot_families       std of C_F = theta_F . phi_F(x) per family, model samples vs data
    sign_table          signs of theta: L_6 and the outer region of each scalar channel (confinement)
Sign convention: p ~ exp(+theta . phi), so confinement needs theta < 0 on the growing statistics.
"""
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _cols(names):
    return {n: f'C{i}' for i, n in enumerate(names)}


def plot_nlogp(res, bins=80, tail_pct=(0.05, 99.95)):
    """One panel per model, pooled over its seeds: histogram of (-log p) - mean(-log p on model samples).
    A model whose name contains 'Gaussian' also gets the exact prediction: y = 2 (-log p - H) + d ~ chi2_d."""
    from scipy import stats
    names = [n for n, r in res.items() if r]
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.6), squeeze=False, sharey=True)
    for ax, n in zip(axes[0], names):
        m = np.concatenate([r['nlp_model'] for r in res[n]])
        dta = np.concatenate([r['nlp_data'] for r in res[n]])
        c = float(np.mean([r['H_sample'] for r in res[n]]))
        lo, hi = np.percentile(np.concatenate([m, dta]) - c, tail_pct)
        edges = np.linspace(lo, hi, bins)
        ax.hist(np.clip(m - c, lo, hi), bins=edges, density=True, log=True, color='C0', alpha=.45,
                label=f'model samples (std {m.std():.1f})')
        ax.hist(np.clip(dta - c, lo, hi), bins=edges, density=True, log=True, histtype='step', color='k', lw=1.5,
                label=f'data (std {dta.std():.0f}, mean {dta.mean() - c:+.0f})')
        if 'Gaussian' in n:
            d = res[n][0]['d']
            z = np.linspace(lo, hi, 400)
            ax.plot(z, 2 * stats.chi2.pdf(2 * z + d, d), color='C3', lw=1.2, label=f'chi2_{d} prediction')
        out = float(((dta - c < lo) | (dta - c > hi)).mean())
        ax.set_title(f'{n}\n{len(res[n])} seed(s); data outside range: {100 * out:.1f}% (clipped)', fontsize=9)
        ax.set_xlabel('-log p  -  H_sample   [nats]')
        ax.legend(fontsize=7, frameon=False)
    axes[0][0].set_ylabel('density')
    fig.tight_layout()
    return fig


def plot_concentration(res):
    names = [n for n, r in res.items() if r]
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 3.4))
    for i, n in enumerate(names):
        v = np.array([r['std_model'] for r in res[n]], float)
        ax.bar(i, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, color='C0', alpha=.7, capsize=4)
        ax.text(i, v.mean(), f' {v.mean():.1f}', ha='center', va='bottom', fontsize=8)
    d = res[names[0]][0]['d']
    ax.axhline(np.sqrt(d / 2), ls='--', color='gray', lw=1, label=f'Gaussian: sqrt(d/2) = {np.sqrt(d / 2):.1f}')
    ax.set_xticks(range(len(names)), names, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel('std of -log p on model samples [nats]')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_families(res):
    """std over samples of C_F per family: model samples (filled) vs data (line). Mean over seeds."""
    names = [n for n, r in res.items() if r]
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 3.4), squeeze=False)
    for ax, n in zip(axes[0], names):
        fams = list(res[n][0]['families'])
        sm = [np.mean([r[f'C_model__{F}'].std() for r in res[n]]) for F in fams]
        sd = [np.mean([r[f'C_data__{F}'].std() for r in res[n]]) for F in fams]
        x = np.arange(len(fams))
        ax.bar(x - .2, sm, .4, label='model samples', color='C0', alpha=.7)
        ax.bar(x + .2, sd, .4, label='data', color='k', alpha=.6)
        ax.set_yscale('log')
        ax.set_xticks(x, [F.replace('Scalar_', 'S_').replace('_gaussianK', '') for F in fams], rotation=20, ha='right', fontsize=8)
        ax.set_title(n, fontsize=9)
        ax.set_ylabel('std of theta_F . phi_F(x)')
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def sign_table(res):
    """Per model: number of seeds, theta_L6 sign counts, outer-region (reg3) and mass-slot (reg4) theta signs of
    the scalar channels, and for the Gaussian every band theta. 'possibly unconfined' = positive L_6 AND positive
    outer-region theta on the same channel in some seed."""
    rows = []
    for n, runs in res.items():
        if not runs:
            continue
        pos_l6, pos_out, unconf, pos_band, n_l6, n_out = 0, 0, 0, 0, 0, 0
        for r in runs:
            th, lab = r['theta'], list(r['labels'])
            l6 = {int(m.group(1)): th[i] for i, l in enumerate(lab) if (m := re.fullmatch(r'L_6\[(\d+)\]', l))}
            out = {int(m.group(1)): th[i] for i, l in enumerate(lab) if (m := re.search(r'ch(\d+),reg3\]', l))}
            pos_l6 += sum(v > 0 for v in l6.values()); n_l6 += len(l6)
            pos_out += sum(v > 0 for v in out.values()); n_out += len(out)
            unconf += sum(1 for c in out if out[c] > 0 and l6.get(c, -1) > 0)
            pos_band += sum(th[i] > 0 for i, l in enumerate(lab) if l.startswith('Scattering_Second'))
        rows.append({'model': n, 'seeds': len(runs), 'theta_L6 > 0': f'{pos_l6} / {n_l6}',
                     'theta_outer(reg3) > 0': f'{pos_out} / {n_out}' if n_out else '-',
                     'channels both > 0 (possibly unconfined)': unconf if n_out else '-',
                     'Scat2 bands theta > 0': pos_band if any('Scattering_Second' in l for l in res[n][0]['labels']) else '-'})
    return pd.DataFrame(rows).set_index('model')


def plot_entropy_evolution(runs, H_gauss, d=256):
    """MGD entropy lower bound along the transport, H_t = H(p_0) + int_0^t dH_t_bound (same trapezoid rule as
    utils_entropy.entropy_bound; the end value is H_bound), against the Gaussian entropy of the data (dashed).
    runs: {model: run_comparison run dict with 't' and 'dH'}."""
    from scipy.integrate import cumulative_trapezoid
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    H0 = 0.5 * d * (np.log(2 * np.pi) + 1)
    for (name, r), c in zip(runs.items(), ['C0', 'C1', 'C2', 'C3', 'C4']):
        if r.get('t') is None or 'dH' not in r:
            continue
        t, dH = r['t'].numpy(), r['dH'].numpy().reshape(-1)
        ax.plot(t[1:], H0 + cumulative_trapezoid(dH, x=t[1:], initial=0), color=c, label=name)
    ax.axhline(H_gauss, ls='--', color='gray', lw=1, label=f'Gaussian, data covariance ({H_gauss:.0f})')
    ax.set_xlabel('t'); ax.set_ylabel('entropy lower bound [nats]')
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def seed_agreement(res):
    """Do the seeds assign the same log p to the same data signals? Pearson and Spearman between the seeds' -log p on
    the data (same x1, point by point): mean and minimum over seed pairs."""
    from scipy import stats
    rows = []
    for n, runs in res.items():
        if len(runs) < 2:
            continue
        L = np.stack([r['nlp_data'] for r in runs]).astype(float)
        pe, sp = [], []
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                pe.append(np.corrcoef(L[i], L[j])[0, 1]); sp.append(stats.spearmanr(L[i], L[j])[0])
        rows.append({'model': n, 'seeds': len(runs), 'Pearson mean': np.mean(pe), 'Pearson min': np.min(pe),
                     'Spearman mean': np.mean(sp), 'Spearman min': np.min(sp)})
    return pd.DataFrame(rows).set_index('model')


def split_flags(res, corr=-0.9, ratio=5.0):
    """Families whose log p contributions C_F cancel each other: corr(C_F, C_G) < corr and std(C_F) > ratio x std of
    the total, on the model samples of the first seed. Their separate contributions are not interpretable."""
    rows = []
    for n, runs in res.items():
        if not runs:
            continue
        r = runs[0]
        fams = [str(F) for F in r['families']]
        C = {F: r[f'C_model__{F}'].astype(float) for F in fams}
        s_tot = sum(C.values()).std()
        bad = [f'{F} vs {G} (corr {np.corrcoef(C[F], C[G])[0, 1]:+.2f}, std {C[F].std():.0f})'
               for i, F in enumerate(fams) for G in fams[i + 1:]
               if np.corrcoef(C[F], C[G])[0, 1] < corr and min(C[F].std(), C[G].std()) > ratio * s_tot]
        rows.append({'model': n, 'std(total)': s_tot, 'ill-defined splits': '; '.join(bad) or 'none'})
    return pd.DataFrame(rows).set_index('model')


def plot_per_scale(res):
    """std over samples of the contribution of each scale (channel) to log p, for the families with statistics per
    channel: model samples (solid) vs data (dashed), mean over seeds. One panel per model and family."""
    panels = [(n, F) for n, runs in res.items() if runs for F in runs[0]['families'] if f'Cch_model__{F}' in runs[0]]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.2), squeeze=False)
    for ax, (n, F) in zip(axes[0], panels):
        ch = res[n][0][f'chans__{F}']
        for kind, ls in (('model', '-'), ('data', '--')):
            s = np.mean([r[f'Cch_{kind}__{F}'].std(0) for r in res[n]], axis=0)
            ax.plot(ch, s, ls, marker='o', ms=3, color='C0' if kind == 'model' else 'k', label=kind)
        ax.set_yscale('log'); ax.set_xlabel('channel (0 = finest)'); ax.set_ylabel('std of the scale\'s log p contribution')
        ax.set_title(f"{n}\n{str(F).replace('Scalar_', 'S_').replace('_gaussianK', '')}", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig
