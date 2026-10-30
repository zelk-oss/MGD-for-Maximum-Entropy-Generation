#!/usr/bin/env python3
"""Jets log p: distribution, share of each potential family, training vs held-out data; jets_full vs
jets_noscalar, 5 seeds each (2026-10-29).

Per arm x seed: Theta_reg(t=1) at the cross-seed lam (lam_selection_criterion/turb_xseed.py, window end0.99,
as jets/logp_seed_check.py) and the run's own fitted potentials. Per family F (a potential of the run) and
signal x: C_F(x) = Theta_F . phi_F(x), log p(x) = sum_F C_F(x) + const (code sign, p ~ exp(+theta.phi); the const
-log Z is unknown, so only differences between signals under the SAME Theta are exact: spreads across signals,
held-out - training). No rare / typical split (Eulerian jet data has no rare events in the Lagrangian sense, user
2026-10-29): the decomposition is the share of each family in the variance of log p across signals,
Cov(C_F, log p) / Var(log p), which sums to 1 over the families and is free of the constant. Two data sets:
  train   : the run's x1 (10000 spread windows, exactly as run_jet_SDE.py)
  heldout : windows of the record never used in training and more than --gap windows away from every
            training window (consecutive windows are correlated: integral scale ~4 windows of 512 samples),
            standardized with the TRAINING mean / std, as the model sees them.
Caveat (2026-10-29): jets_noscalar has L_6 filter 1 > 0 in 5/5 seeds at lam 1e-2 (density not normalizable along
that direction); its decomposition is shown anyway, for the comparison.

Compute (Jean Zay, GPU node, ~1-2 min per run):  python jets/logp_jets.py
Output: jets/figures/logp_jets/logp_jets.npz (small) -> jets/logp_jets.ipynb (the plot_* functions below).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
if __name__ == '__main__':                                  # only as a script: an importing notebook keeps its backend
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent                      # jets/
ROOT = HERE.parent
for p in (ROOT / 'turbulence', ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data',
          HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ARMS = ['jets_full', 'jets_noscalar']
SETS = ('train', 'heldout')
OUT = HERE / 'figures' / 'logp_jets'
ARM_COLORS = {'jets_full': 'C2', 'jets_noscalar': 'C1', 'jets_scat2_scalar': 'C0', 'jets_scat2': 'C3'}
SHORT = {'Scattering_Fourth_Order_Mod2_': 'Scat4_', 'Scattering_Third_Order_': 'Scat3_',
         'Scattering_Second_Order': 'Scat2', 'Scattering_First_Order': 'Scat1', 'Scalar_': 'S_', '_gaussianK': ''}


def short(name):
    for a, b in SHORT.items():
        name = name.replace(a, b)
    return name


# ------------------------------------------------------------------------ data ----
def train_and_heldout(cfg, gap, n_heldout, seed=1):
    """x1 exactly as run_jet_SDE.py (normalize over the selected windows) and held-out windows standardized with
    x1's raw mean / std. Returns x1, xh (n, 1, T) float32 and the held-out window indices."""
    from jet_data import load_modane_windows, select_windows
    from codes.utils import normalize
    raw, _ = load_modane_windows(cfg['subseries_len'], cfg['target_len'], n=None, normalized=False)
    n_win = raw.shape[0]
    tr = select_windows(n_win, cfg['n1'], bool(cfg.get('n1_spread') or False), 0)
    x1_raw = raw[torch.as_tensor(tr)]
    x1 = normalize(x1_raw)
    dist = np.full(n_win, n_win)                            # windows to the nearest training window
    for d in range(gap + 1):
        for s in (-d, d):
            j = tr + s
            dist[j[(j >= 0) & (j < n_win)]] = np.minimum(dist[j[(j >= 0) & (j < n_win)]], d)
    free = np.flatnonzero(dist > gap)
    ho = np.sort(np.random.default_rng(seed).choice(free, min(n_heldout, free.size), replace=False))
    xh = ((raw[torch.as_tensor(ho)] - x1_raw.mean()) / x1_raw.std()).to(torch.float32)
    print(f'data: {n_win} windows, train {len(tr)}, free (> {gap} windows from training) {free.size}, '
          f'held-out used {len(ho)}')
    return x1, xh, ho


# --------------------------------------------------------------------- compute ----
def compute(args):
    from logp_family_contrib import load_potentials, phi_matrix
    from check_potentials import theta_column_map
    from run_comparison import _find_exp_dir
    from logp_seed_check import theta_file, XSEED
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out, data = {}, None
    for arm in args.arms:
        xs = torch.load(XSEED / f'xseed_{arm}.pt', map_location='cpu', weights_only=False)
        lam_sel, files = xs['selected'][args.window], xs['theta_files'][args.window]
        lam = args.lam if args.lam is not None else lam_sel
        print(f'\n#### {arm}: lam {lam:g} (selected {args.window}: {lam_sel:g})')
        seeds = []
        for config, f_sel in files.items():
            exp = _find_exp_dir(HERE, config)
            assert exp is not None, f'no experiments/ folder for {config}'
            cfg = json.loads((exp / 'config.json').read_text())
            if data is None:
                x1, xh, ho = train_and_heldout(cfg, args.gap, args.n_heldout)
                from jets_compare import reference_data
                assert torch.equal(x1.reshape(x1.shape[0], -1), reference_data(cfg)), 'x1 differs from the run data'
                data = {'train': x1.to(device), 'heldout': xh.to(device)}
                out['heldout_windows'] = ho
            sol = torch.load(theta_file(f_sel, lam_sel, lam), map_location='cpu', weights_only=False)
            assert sol['config'] == config, (sol['config'], config)
            theta = sol['Theta_reg'][-1].double().numpy()
            pots = load_potentials(exp, cfg, device)
            cmap = theta_column_map(pots, theta.shape[0])
            s = int(cfg['seed'])
            seeds.append(s)
            for name, x in data.items():
                phi = phi_matrix(pots, x)
                C = np.stack([phi[:, a:a + d] @ theta[a:a + d] for a, d in cmap.values()])     # (families, n)
                out[f'C__{arm}__{s}__{name}'] = C
            print(f'  seed {s}: ' + ', '.join(f'{short(k)}[{a}:{a + d}]' for k, (a, d) in cmap.items()))
        out[f'families__{arm}'] = np.array([short(k) for k in cmap])
        out[f'seeds__{arm}'] = np.array(seeds)
        out[f'lam__{arm}'] = lam
    out['arms'] = np.array(args.arms)
    out['gap'] = args.gap
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / 'logp_jets.npz', **out)
    print(f'\n-> {args.out / "logp_jets.npz"}')



# ----------------------------------------------------------------- load / stats ----
def load(path=OUT / 'logp_jets.npz'):
    """{arm: {'families': [... , 'TOTAL'], 'seeds': [...], 'C': {set: (seeds, families+1, n)}, 'lam'}}."""
    z = np.load(path, allow_pickle=False)
    runs = {}
    for arm in z['arms']:
        fams = list(z[f'families__{arm}'])
        seeds = list(z[f'seeds__{arm}'])
        C = {}
        for name in SETS:
            c = np.stack([z[f'C__{arm}__{s}__{name}'].astype(np.float64) for s in seeds])
            C[name] = np.concatenate([c, c.sum(1, keepdims=True)], axis=1)            # + TOTAL
        runs[str(arm)] = dict(families=fams + ['TOTAL'], seeds=seeds, C=C, lam=float(z[f'lam__{arm}']))
    return runs


def family_table(runs, data_set='train'):
    """Per arm and family, mean and sd over seeds of: mean C_F over signals (level: depends on where phi_F is
    centred, not comparable between arms), std of C_F across signals, std_norm = std_F / std(log p), share of Var(log p) = Cov(C_F, log p) /
    Var(log p) (sums to 1; > 1 or < 0 = families that cancel each other), corr(C_F, log p). Long DataFrame."""
    import pandas as pd
    rows = []
    for arm, r in runs.items():
        C = r['C'][data_set]                                                         # (S, F+1, n)
        Cc = C - C.mean(2, keepdims=True)
        tot = Cc[:, -1:]
        var_tot = (tot ** 2).mean(2)
        std = C.std(2, ddof=1)
        stats_ = dict(mean=C.mean(2), std_signals=std, std_norm=std / std[:, -1:],  # std_F / std(log p), per seed
                      var_share=(Cc * tot).mean(2) / var_tot,
                      corr=(Cc * tot).mean(2) / np.sqrt((Cc ** 2).mean(2) * var_tot))
        for j, F in enumerate(r['families']):
            row = dict(arm=arm, family=F)
            for k, v in stats_.items():
                row[k], row[f'{k}_sd'] = v[:, j].mean(), v[:, j].std(ddof=1)
            rows.append(row)
    return pd.DataFrame(rows)


def heldout_table(runs):
    """Per arm and family: mean C_F on held-out - on training (same Theta, so the unknown constant cancels), mean
    and sd over seeds; shift_over_std = shift / std of C_F across training signals. Negative TOTAL = held-out data
    less likely than training data (overfitting, or the held-out windows differ from the training ones)."""
    import pandas as pd
    rows = []
    for arm, r in runs.items():
        d = r['C']['heldout'].mean(2) - r['C']['train'].mean(2)                     # (S, F+1)
        sd = r['C']['train'].std(2, ddof=1)
        for j, F in enumerate(r['families']):
            rows.append(dict(arm=arm, family=F, shift=d[:, j].mean(), shift_sd=d[:, j].std(ddof=1),
                             shift_over_std=(d[:, j] / sd[:, j]).mean()))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ plots ----
def show(fig):
    """Notebook display that works with any backend (Agg included) and never twice: close the figure (so the inline
    backend does not show it again at the end of the cell), then display it as an image."""
    from IPython.display import display
    plt.close(fig)
    display(fig)


def _centre(r, s_idx):
    """The seed's TOTAL log p minus its median on the training data (removes the unknown -log Z per seed)."""
    return np.median(r['C']['train'][s_idx, -1])


def plot_distribution(runs, bins=120, data_set='train'):
    """log p (TOTAL) histograms, centred per seed: one panel per arm (1 line per seed), then both arms pooled over
    seeds on the same axis."""
    arms = list(runs)
    fig, axes = plt.subplots(1, len(arms) + 1, figsize=(5.2 * (len(arms) + 1), 4), squeeze=False)
    lp = {a: [r['C'][data_set][i, -1] - _centre(r, i) for i in range(len(r['seeds']))] for a, r in runs.items()}
    allv = np.concatenate([v for a in arms for v in lp[a]])
    edges = np.linspace(*np.percentile(allv, [0.01, 100]), bins + 1)
    for ax, a in zip(axes[0], arms):
        for i, v in enumerate(lp[a]):
            ax.hist(v, edges, histtype='step', color=ARM_COLORS.get(a, 'k'), alpha=0.7, lw=1,
                    label='1 line per seed' if i == 0 else None)
        ax.set_title(f'{a} (lam {runs[a]["lam"]:g})', fontsize=10)
        ax.set_ylabel('count')
    for a in arms:
        axes[0, -1].hist(np.concatenate(lp[a]), edges, histtype='step', lw=1.6, density=True,
                         color=ARM_COLORS.get(a, 'k'), label=a)
    axes[0, -1].set_title('all arms, seeds pooled (density)', fontsize=10)
    for ax in axes[0]:
        ax.set_yscale('log')
        ax.set_xlabel('log p - median (nats)')
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f'log p on the {data_set} data (centred on the training median of each seed)', fontsize=10)
    fig.tight_layout()
    return fig


GROUPS = {'L_6': ('L_6',), 'scalar+Scat1+Scat2': ('S_', 'Scat1', 'Scat2'), 'Scat3+Scat4': ('Scat3', 'Scat4')}


def group_runs(runs, groups=GROUPS):
    """Same structure as load(), with the families summed into groups {name: family-name prefixes} per seed and per
    signal (so std / seed sd are those of the group). Every family must fall in exactly one group; a group with no
    family in an arm (Scat3+Scat4 in scat2*) is left out of that arm."""
    out = {}
    for arm, r in runs.items():
        fams = r['families'][:-1]
        member = {g: [i for i, f in enumerate(fams) if f.startswith(pre)] for g, pre in groups.items()}
        counts = np.bincount([i for m in member.values() for i in m], minlength=len(fams))
        assert (counts == 1).all(), f'{arm}: families not in exactly one group: ' \
                                    f'{[f for f, c in zip(fams, counts) if c != 1]}'
        keep = [g for g in groups if member[g]]
        C = {name: np.stack([c[:, member[g]].sum(1) for g in keep] + [c[:, -1]], axis=1) for name, c in r['C'].items()}
        out[arm] = dict(r, families=keep + ['TOTAL'], C=C)
    return out


def plot_family_bars(runs, data_set='train', groups=None):
    """Bar chart per family (or per group of families, groups = {name: prefixes}, see GROUPS), all arms side by side,
    error bars = sd over the seeds: (1) mean C_F (level only: depends on where phi_F is centred), shaded; (2) std of
    C_F across signals (nats), TOTAL = std of log p; (3) std_F / std(log p) of the same arm, to compare arms (> 1 =
    cancelled by other families). Families missing in an arm are left empty."""
    if groups is not None:
        runs = group_runs(runs, groups)
    T = family_table(runs, data_set)
    fams = [f for f in dict.fromkeys(T['family']) if f != 'TOTAL'] + ['TOTAL']
    arms = list(runs)
    x = np.arange(len(fams))
    w = 0.8 / len(arms)
    panels = (('mean C_F (nats; level only:\ndepends on the centring of phi_F)', 'mean', dict(alpha=0.5)),
              ('std of C_F across signals (nats)\nTOTAL = std of log p', 'std_signals', {}),
              ('std of C_F / std of log p\n(same arm; > 1 = cancelled)', 'std_norm', dict(hatch='//', edgecolor='w')))
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 10), sharex=True)
    for k, a in enumerate(arms):
        t = T[T.arm == a].set_index('family').reindex(fams)
        off = (k - (len(arms) - 1) / 2) * w
        for ax, (title, key, style) in zip(axes, panels):
            ax.bar(x + off, t[key], w, yerr=t[f'{key}_sd'], capsize=3, color=ARM_COLORS.get(a, f'C{k}'), label=a,
                   **style)
            ax.set_ylabel(title, fontsize=9)
    axes[2].axhline(1, color='k', lw=0.6, ls='--')
    for ax in axes:
        ax.axhline(0, color='k', lw=0.6)
        ax.grid(alpha=0.2, axis='y')
        ax.legend(fontsize=8, frameon=False)
    axes[-1].set_xticks(x, fams, rotation=30, ha='right')
    fig.suptitle(f'log p per potential family, {data_set} data; error bars = sd over seeds', fontsize=10)
    fig.tight_layout()
    return fig, T


def plot_heldout(runs, bins=120):
    """Training vs held-out log p under the same Theta (one row per arm, seeds pooled after centring on the training
    median of each seed), and the shift of the mean per family with seed error bars."""
    H = heldout_table(runs)
    arms = list(runs)
    fig, axes = plt.subplots(len(arms), 2, figsize=(13, 3.8 * len(arms)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1, 1.4]))
    for row, a in enumerate(arms):
        r = runs[a]
        v = {name: np.concatenate([r['C'][name][i, -1] - _centre(r, i) for i in range(len(r['seeds']))])
             for name in SETS}
        edges = np.linspace(*np.percentile(np.concatenate(list(v.values())), [0.01, 100]), bins + 1)
        ax = axes[row, 0]
        for name, col in zip(SETS, ('k', ARM_COLORS.get(a, 'C0'))):
            ax.hist(v[name], edges, histtype='step', density=True, lw=1.4, color=col,
                    label=f'{name} (n={r["C"][name].shape[2]} x {len(r["seeds"])} seeds)')
        ax.set_yscale('log')
        ax.set_xlabel('log p - training median (nats)')
        ax.set_title(a, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        t = H[H.arm == a]
        ax = axes[row, 1]
        ax.bar(np.arange(len(t)), t['shift'], yerr=t['shift_sd'], capsize=3, color=ARM_COLORS.get(a, 'C0'))
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(np.arange(len(t)), t['family'], rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('mean held-out - mean training (nats)', fontsize=9)
        ax.grid(alpha=0.2, axis='y')
    fig.suptitle('held-out vs training data, same Theta (the unknown constant cancels); '
                 'error bars = sd over seeds', fontsize=10)
    fig.tight_layout()
    return fig, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=ARMS)
    ap.add_argument('--window', default='end0.99', help='xseed window whose selected lam is used')
    ap.add_argument('--lam', type=float, default=None, help='default: the selected lam of each arm')
    ap.add_argument('--gap', type=int, default=4, help='held-out windows must be > gap windows from training')
    ap.add_argument('--n_heldout', type=int, default=10000)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    compute(args)
    runs = load(args.out / 'logp_jets.npz')
    import pandas as pd
    with pd.option_context('display.float_format', '{:,.4g}'.format, 'display.width', 220):
        T = family_table(runs)
        print('\nper family, training data (mean and sd over seeds):')
        print(T[['arm', 'family', 'var_share', 'var_share_sd', 'corr', 'std_signals', 'std_signals_sd']]
              .to_string(index=False))
        print('\nheld-out - training, mean C_F (nats; shift_over_std = shift / std across training signals):')
        print(heldout_table(runs).to_string(index=False))


if __name__ == '__main__':
    main()
