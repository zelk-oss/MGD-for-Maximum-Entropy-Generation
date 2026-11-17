"""Figures + numbers for notes/dk_gaussian_d16_1114/report.md from the saved bench_exact_D.csv (no recomputation of the bench)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.argv = ['x']
SCR = '/home/chiaraz/phd/MGD-for-Maximum-Entropy-Generation/gauss_scat4_test/bench_exact_D'
exec(open(f'{SCR}/bench_defs.py').read())
OUT = Path('/home/chiaraz/phd/MGD-for-Maximum-Entropy-Generation/notes/dk_gaussian_d16_1114')
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6})
pd.set_option('display.width', 220)
df = pd.read_csv(OUT / 'bench_exact_D.csv', dtype={'terms': str})
fin = df[np.isfinite(df.D_exact)]

print('groups', len(df), '| D = inf:', int((~np.isfinite(df.D_exact)).sum()), '| multi-term:', int((df.n_terms >= 2).sum()),
      'of which inf:', int(((df.n_terms >= 2) & ~np.isfinite(df.D_exact)).sum()))
print('LB95 coverage on finite-D groups: min', fin.LB95_covers.min(), '| n finite-D groups', len(fin))
print('median LB95/Dhat', round(float((fin.LB95_mean / fin.Dhat_mean).median()), 2), '| median Dhat/D', round(float((fin.Dhat_mean / fin.D_exact).median()), 2),
      '| Dhat mean below D in', int((fin.Dhat_mean < fin.D_exact).sum()), 'of', len(fin))
print('\nsingle terms:\n', df[df.n_terms == 1][['terms', 'z_max', 'D_exact', 'Dhat_mean', 'Dhat_q05', 'Dhat_q95', 'LB95_mean', 'khat_mean', 'khat_sd',
                                                'measured_old', 'measured_new', 'bias_old', 'within2boot_old', 'mgd_Dhat']].round(3).to_string(index=False))
print('\nfinite-D multi-term groups:\n', fin[fin.n_terms > 1][['terms', 'z_max', 'D_exact', 'Dhat_mean', 'LB95_mean', 'khat_mean', 'measured_old', 'measured_new',
                                                                'bias_old']].round(3).to_string(index=False))
print('\nold rule: sets rated measured (mean over groups, finite D): bias', round(float(fin.bias_old.mean()), 3), '| within 2 boot', round(float(fin.within2boot_old.mean()), 2))
print('groups with measured_old >= 0.5:', df[df.measured_old >= .5].terms.tolist(), '| their measured_new:', df[df.measured_old >= .5].measured_new.round(2).tolist())
print('max measured_new over all groups', float(df.measured_new.max()), '| khat_mean range', round(float(df.khat_mean.min()), 2), round(float(df.khat_mean.max()), 2))
print('MGD samples vs exact-draw 5-95 band (finite D):', int(((fin.mgd_Dhat >= fin.Dhat_q05) & (fin.mgd_Dhat <= fin.Dhat_q95)).sum()), 'of', len(fin), 'inside;',
      'all', int(((df.mgd_Dhat >= df.Dhat_q05) & (df.mgd_Dhat <= df.Dhat_q95)).sum()), 'of', len(df))

# figure 1: spectrum
C = S / M
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.axhline(1, color=MUTED, lw=1, ls='--', label='white source (C = 1)')
ax.plot(np.arange(M // 2 + 1), C[:M // 2 + 1], 'o-', color=BLUE, ms=5, lw=1.5, label='p*: max-ent Gaussian of the Scat2 family')
ax.set_xlabel('frequency index ω'); ax.set_ylabel('C(ω) = S(ω)/M'); ax.set_ylim(0.8, 1.2); ax.legend(fontsize=7, loc='upper right')
fig.tight_layout(); fig.savefig(OUT / 'fig1_spectrum.png', dpi=130); plt.close(fig)

# figure 2: bench
d = df.reset_index(drop=True)
x = np.arange(len(d)); TOP = 45
fig, ax = plt.subplots(figsize=(10.5, 3.9))
ax.vlines(x, d.Dhat_q05, d.Dhat_q95, color=BLUE, lw=1.3, zorder=2)
ax.plot(x, d.Dhat_mean, 'o', color=BLUE, ms=5, label='D̂ (mean of 200 sets, N = 10⁴); bar = 5–95 %', zorder=3)
ax.plot(x, d.LB95_mean, 'v', color=AQUA, ms=5, label='rigorous 95 % lower bound (mean)', zorder=3)
fm = np.isfinite(d.D_exact)
ax.plot(x[fm], d.D_exact[fm], 'D', color=INK, ms=5, mfc='none', mew=1.4, label='exact D_k', zorder=4)
ax.plot(x[~fm], np.full(int((~fm).sum()), TOP), '^', color=INK, ms=7, label='exact D_k = ∞', zorder=4)
for s in (4, 14, 24, 29):
    ax.axvline(s + 0.5, color=MUTED, lw=0.6)
ax.set_yscale('log'); ax.set_ylim(0.15, 80)
ax.set_xticks(x, d.terms, rotation=90, fontsize=7)
ax.set_xlabel('group = set of removed Scat2 terms (0 = finest filter … 4 = low-pass); vertical lines separate 1, 2, 3, 4, 5 terms')
ax.set_ylabel('nats'); ax.legend(fontsize=7, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.42), frameon=False)
fig.tight_layout(); fig.savefig(OUT / 'fig2_bench_exact_vs_estimate.png', dpi=130, bbox_inches='tight'); plt.close(fig)

# figure 3: Scat4 estimate vs prediction (5 seeds)
sys.path.insert(0, str(ROOT / 'kl_groups'))
import plots as P
RES = ROOT / 'gauss_scat4_test/runs/d16/results'
dd, agg = P.load('gauss4', RES)
pred = pd.read_csv(RES / 'pred_H.csv')
m = agg[agg.family.str.contains('Fourth')].merge(pred[['group', 'H_pred']], on='group')
fig, ax = plt.subplots(figsize=(4.8, 4.2))
for kind, (sym, col) in {'family': ('o', BLUE), 'pair': ('s', ORANGE), 'single': ('^', AQUA)}.items():
    g = m[m.kind == kind]
    ax.errorbar(g.H_pred, g.H, g.err, fmt=sym, ms=5, capsize=2, color=col, label=f'{kind} ({len(g)})')
lo, hi = m[['H_pred', 'H']].min().min() * .6, m[['H_pred', 'H']].max().max() * 1.6
ax.plot([lo, hi], [lo, hi], color=MUTED, ls=':', label='D̂ = prediction')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('predicted finite-data D_k (nats)'); ax.set_ylabel('measured D̂ (mean over 5 seeds)')
ax.legend(fontsize=7, loc='upper left'); fig.tight_layout(); fig.savefig(OUT / 'fig3_scat4_vs_prediction.png', dpi=130); plt.close(fig)
r_ = m.H / m.H_pred
print('\nScat4 D_hat/pred (mean over seeds, per group): n', len(m), '| median', round(float(r_.median()), 2), '| 10-90 %', np.round(np.quantile(r_, [.1, .9]), 2).tolist(),
      '| all measured (flag false):', bool((~m.flag).all()))

# figure 4: lambda plateau
xs = torch.load(RES / 'xseed_g4_M16.pt', map_location='cpu', weights_only=False)
lams = np.array(xs['lams'])
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.0))
for ax, wnd, title in zip(axs, ('whole', 'end0.99'), ('window: whole', 'window: end0.99 (the selection rule)')):
    a = np.asarray(xs['loss'][wnd], float)
    mean = np.nanmean(a.reshape(-1, a.shape[-1]), axis=0)
    ax.loglog(lams[:-1], (mean - mean[-1])[:-1], 'o-', color=BLUE, ms=5)
    ax.set_xlabel('λ'); ax.set_title(title, fontsize=9)
axs[0].set_ylabel('mean cross-seed loss − value at λ = 1')
fig.tight_layout(); fig.savefig(OUT / 'fig4_lambda_plateau.png', dpi=130); plt.close(fig)
print('figures in', OUT)
