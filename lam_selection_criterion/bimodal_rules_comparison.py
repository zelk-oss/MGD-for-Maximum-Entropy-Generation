"""Script version of bimodal_rules_comparison.ipynb (same code cells): reads BM_XSEED_DIR, writes the PNGs to BM_FIG_DIR.
Run as a JZ job with launch_bimodal_rules_fig.sh (matplotlib backend Agg)."""
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import simpson

ROOT = Path.cwd()
while not (ROOT / '.git').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
RESULTS = Path(os.environ.get('BM_XSEED_DIR', ROOT / 'lam_selection_criterion' / 'results' / 'bimodal_xseed2'))
X_RANGE, N_GRID = (-10.0, 10.0), 8001

# label -> (rule name in run['rules'], colour)
RULES = {
    'A  xseed M, t>=0.99':       ('uniform_xseed_end0.99',       '#2a78d6'),
    'B  xseed Sigma_A, t>=0.99': ('uniform_xseed_sigma_end0.99', '#eb6834'),
    'C  Guth, lam=1':            ('guth_lam1',                   '#2f9e44'),
    'oracle (uniform)':          ('uniform_oracle_end',          '#52514e'),
}
INK, MUTED = '#0b0b0b', '#52514e'
FIGDIR = Path(os.environ.get('BM_FIG_DIR', RESULTS))             # where the PNGs go (default: next to the results)
FIGDIR.mkdir(parents=True, exist_ok=True)
print('RESULTS:', RESULTS, '; figures ->', FIGDIR)

def sched_of(rule):
    return 'guth' if rule.startswith('guth') else 'uniform'

def tkey(rule, lam):
    return f'{sched_of(rule)}|{lam:g}'                              # key format of bimodal_lamtest.py

def load_beta_dir(d):
    files = sorted(d.glob('run_*_s*.pt'))
    runs = [torch.load(f, map_location='cpu', weights_only=False) for f in files]
    assert runs, f'no run_*_s*.pt in {d}'
    out = dict(beta=float(runs[0]['beta']), n=len(runs), lams=[float(l) for l in runs[0]['lams']],
               theta_star=runs[0]['theta_star'].double().numpy(), kl_floor=float(runs[0]['kl_floor']),
               cr_bound=float(runs[0]['cr_bound']), logz_true=float(runs[0]['logz_true']),
               small_well_true=float(runs[0]['small_well_true']))
    out['mgd'] = np.array([r['theta_end']['theta_t'].double().numpy() for r in runs])
    out['pick'] = {lab: np.array([float(r['rules'][rule]['lam']) for r in runs]) for lab, (rule, _) in RULES.items()}
    out['theta'] = {lab: np.array([r['theta_end'][tkey(rule, float(r['rules'][rule]['lam']))].double().numpy()
                                   for r in runs]) for lab, (rule, _) in RULES.items()}
    # per-lambda curves (n_runs, n_lams), both weightings
    for sched in ('uniform', 'guth'):
        for name in ('kl_end_true', 'logz', 'small_well'):
            get = (lambda v: float(v['est'])) if name == 'logz' else float      # logz entries are dicts (est, err_true, err_self)
            out[f'{name}_{sched}'] = np.array([[get(r[name][f'{sched}|{l:g}']) for l in out['lams']] for r in runs])
    return out

per_beta = [load_beta_dir(d) for d in sorted(RESULTS.glob('beta_*')) if d.is_dir() and any(d.glob('run_*_s*.pt'))]
assert per_beta, f'no beta_*/run_*_s*.pt under {RESULTS}'
betas = np.array([b['beta'] for b in per_beta])
print(f'{len(per_beta)} beta values: {betas}; runs per beta: {[b["n"] for b in per_beta]}')
for b in per_beta:
    print(f"  beta {b['beta']:.2f}:", ' | '.join(f"{lab.split()[0]}: " + ', '.join(f'{l:g}x{c}' for l, c in zip(*np.unique(b['pick'][lab], return_counts=True)))
                                                 for lab in RULES))

def curve(ax, lams, Y, col, ls, label=None):
    lam, Y = np.array(lams[1:]), Y[:, 1:]
    med = np.nanmedian(np.where(np.isfinite(Y), Y, np.nan), axis=0)
    lo, hi = np.nanpercentile(np.where(np.isfinite(Y), Y, np.nan), [25, 75], axis=0)
    ax.fill_between(lam, lo, hi, color=col, alpha=0.12, lw=0)
    ax.plot(lam, med, ls, color=col, lw=2, label=label)

def pick_markers(ax, b, Ymap):
    for lab, (rule, col) in RULES.items():
        s = sched_of(rule)
        lams = np.array(b['lams'])
        idx = [int(np.argmin(np.abs(np.log10(lams[1:]) - np.log10(max(l, 1e-30))))) + 1 for l in b['pick'][lab]]
        Y = Ymap[s]
        yv = np.nanmedian([Y[i, j] for i, j in enumerate(idx)])
        ax.plot(np.median(b['pick'][lab]), yv, 'o', color=col, ms=11, mfc='white', mec=col, mew=2.2, zorder=5,
                label=lab)

fig, axes = plt.subplots(3, len(per_beta), figsize=(5.2 * len(per_beta), 11), squeeze=False)
for j, b in enumerate(per_beta):
    # row 0: exact KL(p_theta* || p_theta(lam)) at t = 1
    ax = axes[0, j]
    for s, ls, c in (('uniform', '-', '#2a78d6'), ('guth', '--', '#7a5fb8')):
        curve(ax, b['lams'], b[f'kl_end_true_{s}'], c, ls, label=f'{s} weights')
        ax.axhline(np.nanmedian(b[f'kl_end_true_{s}'][:, 0][np.isfinite(b[f'kl_end_true_{s}'][:, 0])]), color=c, ls=':', lw=1.3)
    ax.axhline(b['kl_floor'], color=MUTED, ls='-.', lw=1.4, label='max-likelihood floor')
    pick_markers(ax, b, {'uniform': b['kl_end_true_uniform'], 'guth': b['kl_end_true_guth']})
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title(rf"$\beta={b['beta']:g}$"); ax.set_ylabel(r'$D_{\rm KL}(p_{\theta^*}\,\Vert\,p_{\rm reg})$')
    # row 1: |log Z - log Z(theta*)|
    ax = axes[1, j]
    for s, ls, c in (('uniform', '-', '#2a78d6'), ('guth', '--', '#7a5fb8')):
        curve(ax, b['lams'], np.abs(b[f'logz_{s}'] - b['logz_true']), c, ls)
    pick_markers(ax, b, {s: np.abs(b[f'logz_{s}'] - b['logz_true']) for s in ('uniform', 'guth')})
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylabel(r'$|\log Z - \log Z(\theta^*)|$')
    # row 2: small-well mass P(x<0)
    ax = axes[2, j]
    for s, ls, c in (('uniform', '-', '#2a78d6'), ('guth', '--', '#7a5fb8')):
        curve(ax, b['lams'], b[f'small_well_{s}'], c, ls)
    pick_markers(ax, b, {s: b[f'small_well_{s}'] for s in ('uniform', 'guth')})
    ax.axhline(b['small_well_true'], color=INK, lw=1.6, label='truth')
    ax.set_xscale('log'); ax.set_ylabel(r'small-well mass $P(x<0)$'); ax.set_xlabel(r'$\lambda$')
    for i in range(3):
        axes[i, j].grid(axis='y', color='0.88', lw=0.8); axes[i, j].spines[['top', 'right']].set_visible(False)
h, l = axes[0, 0].get_legend_handles_labels()
h2, l2 = axes[2, 0].get_legend_handles_labels()
by = dict(zip(l + l2, h + h2))
fig.legend(by.values(), by.keys(), loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=11)
plt.tight_layout(rect=(0, 0.04, 1, 1))
plt.savefig(FIGDIR / 'bimodal_rules_vs_lambda.png', dpi=170, bbox_inches='tight')
plt.show()

GRID = np.linspace(*X_RANGE, N_GRID)
PHI = np.stack([GRID, GRID**2 / 2, GRID**3 / 3, GRID**4 / 4], axis=1)

def log_density(thetas):
    thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
    lv = thetas @ PHI.T
    m = lv.max(axis=1, keepdims=True)
    with np.errstate(over='ignore'):
        lv = lv - (m + np.log(simpson(np.exp(lv - m), x=GRID, axis=1))[:, None])
    lv[thetas[:, 3] >= 0] = np.nan                                  # not normalisable on R
    return lv

def kl_rows(log_p_ref, thetas, chunk=200):
    p = np.exp(log_p_ref)
    out = []
    for i in range(0, len(thetas), chunk):
        with np.errstate(invalid='ignore'):
            f = p * (log_p_ref - log_density(thetas[i:i + chunk]))
        out.append(simpson(np.where(p > 0, f, 0.0), x=GRID, axis=1))
    return np.concatenate(out)

# sanity: the quadrature here reproduces the pipeline's kl_end_true for rule A
for b in per_beta:
    lp = log_density(b['theta_star'])[0]
    mine = kl_rows(lp, b['theta']['A  xseed M, t>=0.99'])
    print(f"beta {b['beta']:.2f}: KL(theta*||theta*) = {kl_rows(lp, b['theta_star'][None])[0]: .1e}; rule A mean KL here {np.nanmean(mine):.3e}")

estimators = {'MGD (raw)': INK} | {lab: col for lab, (rule, col) in RULES.items() if lab != 'oracle (uniform)'}
rows = []
for b in per_beta:
    lp_true = log_density(b['theta_star'])[0]
    row = dict(beta=b['beta'], cr=b['cr_bound'], floor=b['kl_floor'], mean={}, true={})
    for name in estimators:
        th = b['mgd'] if name == 'MGD (raw)' else b['theta'][name]
        row['mean'][name] = kl_rows(log_density(th.mean(0))[0], th)
        row['true'][name] = kl_rows(lp_true, th)
    rows.append(row)

with plt.rc_context({'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False}):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, key, ylab in ((axes[0], 'mean', r'$\mathbb{E}\,[D_{\rm KL}(p_{\bar\theta}\Vert p_{\hat\theta})]$'),
                          (axes[1], 'true', r'$\mathbb{E}\,[D_{\rm KL}(p_{\theta^*}\Vert p_{\hat\theta})]$')):
        for name, col in estimators.items():
            vals = [r[key][name][np.isfinite(r[key][name])] for r in rows]
            lo, hi = np.array([np.percentile(v, [25, 75]) for v in vals]).T
            ax.fill_between(betas, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(betas, [v.mean() for v in vals], '-o', color=col, lw=2, ms=8, mfc='white', mew=2, label=name)
        if key == 'mean':
            ax.axhline(rows[0]['cr'], color=MUTED, ls='--', lw=1.6, label='Cramér–Rao bound')
        else:
            ax.plot(betas, [r['floor'] for r in rows], ':', color=MUTED, lw=1.8, label='max-likelihood floor')
        ax.set_yscale('log'); ax.set_xlabel(r'$\beta$'); ax.set_ylabel(ylab)
        ax.grid(axis='y', color='0.88', lw=0.8); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGDIR / 'bimodal_rules_KL_vs_beta.png', dpi=170, bbox_inches='tight')
    plt.show()

print(f"{'beta':>5} {'rule':>28} | {'lam chosen (median)':>19} | {'KL truth: median':>16} {'ratio to oracle':>15} | {'|dlogZ|':>8} | {'P(x<0) (truth)':>16}")
for b in per_beta:
    kl_or = np.array([b['kl_end_true_uniform'][i, :].min() for i in range(b['n'])])
    kl_or = np.minimum(kl_or, [b['kl_end_true_guth'][i, :].min() for i in range(b['n'])])
    for lab, (rule, _) in RULES.items():
        s = sched_of(rule)
        idx = [list(np.round(np.log10(np.maximum(b['lams'], 1e-30)), 6)).index(round(np.log10(max(l, 1e-30)), 6)) if l > 0 else 0
               for l in b['pick'][lab]]
        kl = np.array([b[f'kl_end_true_{s}'][i, k] for i, k in enumerate(idx)])
        dz = np.abs(np.array([b[f'logz_{s}'][i, k] for i, k in enumerate(idx)]) - b['logz_true'])
        sw = np.array([b[f'small_well_{s}'][i, k] for i, k in enumerate(idx)])
        print(f"{b['beta']:5.2f} {lab:>28} | {np.median(b['pick'][lab]):19.1e} | {np.median(kl):16.2e} {np.median(kl / kl_or):15.1f} | "
              f"{np.median(dz):8.3f} | {np.median(sw):7.3f} ({b['small_well_true']:.3f})")
    print()
