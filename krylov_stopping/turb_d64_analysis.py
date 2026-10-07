"""Turbulence d = 64 runs (launch_turb_d64.sh, job 742814): ridge 1e-4 vs jac_minres, with / without L_6_psi, 5 seeds.
Reads (pulled from Jean Zay into the repo root): saved_results/aux_moments/*d64*_monitor.pt (all 20 runs, incl. the 3
aborted ones), *_aux_moments.pt, lagrange_multipliers(_regularised)/, sampling_times/, experiments/*d64*/ (fitted potentials).
Run from the repo root: TURBULENCE_1D_DATA_PATH=$PWD/data/data_files/turbulence_1d_period.pt python krylov_stopping/turb_d64_analysis.py
"""
# %% setup
import contextlib, glob, io, re, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().parent
for p in (ROOT, ROOT / 'codes', ROOT / 'data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from codes.time_schedules import two_phase_schedule                                   # noqa: E402

SR, EXP = ROOT / 'saved_results', ROOT / 'experiments'
OUT = ROOT / 'krylov_stopping' / 'results' / 'turb_d64'
OUT.mkdir(parents=True, exist_ok=True)
CONFIGS = ['copies_ridge1e-4', 'copies_jacminres', 'nopsi_ridge1e-4', 'nopsi_jacminres']
SEEDS = [900, 901, 902, 903, 904]
N, D = 2000, 64
T_GRID, _ = two_phase_schedule(10000, 2000, 0.9, 5e-5)                   # the runs' grid (also for aborted runs)
T = np.asarray(T_GRID, dtype=float)[1:]                                    # step k targets t_{k+1}
K_SWITCH = 2000                                                            # first fine-tail step (t = 0.9)


def path(kind, cfg, seed, suffix=''):
    f = glob.glob(str(SR / kind / f'*seed_{seed}_*d64_{cfg}{suffix}*'))
    return Path(f[0]) if f else None


# %% 1. stability from the monitoring files
def stability(cfg, seed):
    m = torch.load(path('aux_moments', cfg, seed, '_monitor'), weights_only=False)
    S, E = m['step'].numpy(), m['eig'].numpy()
    n = len(S)
    th = path('lagrange_multipliers', cfg, seed)
    r = torch.load(th, weights_only=False).shape[1] if th else (142 if cfg.startswith('copies') else 126)
    floor = r / (N * D)
    R = S[:, 1] / floor
    half = n // 2
    slope = np.polyfit(np.arange(half, n), np.log(np.maximum(S[half:, 1], 1e-300)), 1)[0] if n > 100 else np.nan
    weak = np.nanmax(E[:, 5:], axis=1)                                     # max over the 10 weakest directions
    k_eig = E[:, 0].astype(int)
    pre, post = weak[k_eig < K_SWITCH], weak[k_eig >= K_SWITCH]
    j = S[:, 2]
    return dict(n=n, finished=n == 10000, R=R, R_med=np.median(R), R_p99=np.percentile(R, 99), R_max=R.max(),
                spikes=int((R > 10 * np.median(R[:100])).sum()), slope=slope, weak=weak, k_eig=k_eig,
                weak_pre=np.median(pre) if len(pre) else np.nan, weak_post=np.median(post) if len(post) else np.nan,
                mu_min=np.median(E[:, 1]), mu_min_f32=np.nanmedian(E[:, 4]),
                j_mean=np.nanmean(j[j >= 0]) if (j >= 0).any() else np.nan,
                j0=np.mean(j[j >= 0] == 0) if (j >= 0).any() else np.nan, maxy=S[:, 6].max(), r=r)


ST = {c: {s: stability(c, s) for s in SEEDS} for c in CONFIGS}
print('## 1. Stability (monitor; R_k in units of the finite-N floor r/(N d); 5 seeds)\n')
print('| config | finished | R/floor median | p99 | max | spikes (>10x first-100 median) | slope log R, last 50% | '
      'weak ratio median before / after t=0.9 | mu_min (f64 / f32) | j mean, j=0 | max|y| |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for c in CONFIGS:
    v = ST[c]
    f = lambda k: np.array([v[s][k] for s in SEEDS], dtype=float)
    fin = sum(v[s]['finished'] for s in SEEDS)
    print(f'| {c} | {fin}/5 | {np.median(f("R_med")):.3g} | {np.median(f("R_p99")):.3g} | {np.max(f("R_max")):.3g} | '
          f'{int(f("spikes").sum())} | {np.nanmedian(f("slope")):.2e} | {np.nanmedian(f("weak_pre")):.3g} / '
          f'{np.nanmedian(f("weak_post")):.3g} | {np.median(f("mu_min")):.2e} / {np.nanmedian(f("mu_min_f32")):.2e} | '
          f'{np.nanmedian(f("j_mean")):.2f}, {np.nanmedian(f("j0")):.0%} | {np.max(f("maxy")):.3g} |')

# %% 1b. plots: R/floor, weak ratio, j vs t
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt   # noqa: E402
fig, ax = plt.subplots(3, 4, figsize=(17, 9), sharex='col', squeeze=False)
for i, c in enumerate(CONFIGS):
    for s in SEEDS:
        v = ST[c][s]
        n = v['n']
        ax[0, i].plot(1 - T[:n], v['R'], lw=0.6, label=f'{s}' + ('' if v['finished'] else ' (aborted)'))
        ax[1, i].plot(1 - T[v['k_eig']], v['weak'], lw=0.6)
        jj = torch.load(path('aux_moments', c, s, '_monitor'), weights_only=False)['step'][:, 2].numpy()
        if (jj >= 0).any():
            w = 50
            ax[2, i].plot(1 - T[:n][w - 1:], np.convolve(jj, np.ones(w) / w, 'valid'), lw=0.6)
    ax[0, i].set_title(c)
    for a in ax[:, i]:
        a.set_xscale('log'); a.invert_xaxis(); a.axvline(0.1, color='0.5', lw=0.8, ls=':'); a.grid(alpha=0.25, lw=0.5)
    ax[0, i].set_yscale('log'); ax[1, i].set_yscale('log'); ax[1, i].axhline(1, color='0.4', lw=0.8, ls='--')
    ax[2, i].set_xlabel('1 − t (dotted: step-size switch at t = 0.9)')
    ax[0, i].legend(fontsize=7, frameon=False)
ax[0, 0].set_ylabel('R_k / (r / N d)'); ax[1, 0].set_ylabel('max weak-direction ratio'); ax[2, 0].set_ylabel('j_k (50-step mean)')
fig.tight_layout(); fig.savefig(OUT / 'stability.png', dpi=120); plt.close(fig)

# %% 2. final moment mismatch (offline: monitor's mismatch rows were empty, see note)
print('\n## 2. Final moment mismatch, |walkers - target| at the last step / max_t |target| per statistic (finished runs)\n')
print('| config | seeds | median over stats | p90 | max | # stats > 1% |')
print('|---|---|---|---|---|---|')
MOM = {}
for c in CONFIGS:
    rows = []
    for s in SEEDS:
        f = path('aux_moments', c, s, '_aux_moments')
        if f is None:
            continue
        a = torch.load(f, weights_only=False)
        e, p = a['barphi_e'].double(), a['barphi_p'].double()
        rows.append(((p[-1] - e[-1]).abs() / e.abs().amax(0).clamp_min(1e-12)).numpy())
    MOM[c] = np.stack(rows)
    m = MOM[c]
    print(f'| {c} | {len(m)} | {np.median(m):.2e} | {np.percentile(m, 90):.2e} | {m.max():.3g} | '
          f'{np.mean((m > 0.01).sum(1)):.1f} |')

# %% 3. theta: the near-copy pairs and cross-seed spread
PAIRS = [(5, 7 + 14), (6, 7 + 15)]          # L_6[5] ~ L_6_psi[14], L_6[6] ~ L_6_psi[15] (columns of the 'copies' runs)
print('\n## 3. theta at the end: near-copy pairs (copies runs) and cross-seed spread\n')
TH = {}
for c in CONFIGS:
    TH[c] = {}
    for kind, label in (('lagrange_multipliers', 'theta_t'), ('lagrange_multipliers_regularised', 'Theta_reg')):
        L = [torch.load(path(kind, c, s), weights_only=False).double() for s in SEEDS if path(kind, c, s)]
        TH[c][label + '_last'] = torch.stack([x[-1] for x in L])
        if label == 'theta_t':
            TH[c]['theta_t_mean50'] = torch.stack([x[-50:].mean(0) for x in L])
for c in [c for c in CONFIGS if c.startswith('copies')]:
    for est, X in TH[c].items():
        txt = '; '.join(f'L_6[{a}] {X[:, a].mean():+.3g} ± {X[:, a].std():.2g}, L_6_psi[{b - 7}] {X[:, b].mean():+.3g} '
                        f'± {X[:, b].std():.2g}, sum {(X[:, a] + X[:, b]).mean():+.3g} ± {(X[:, a] + X[:, b]).std():.2g}'
                        for a, b in PAIRS)
        print(f'- {c}, {est} ({len(X)} seeds): {txt}')
print('\n| config | estimate | seeds | cross-seed std / |mean|, median over statistics | largest cross-seed std (column) |')
print('|---|---|---|---|---|')
for c in CONFIGS:
    for est, X in TH[c].items():
        rel = (X.std(0) / X.mean(0).abs().clamp_min(1e-12)).numpy()
        k = int(X.std(0).argmax())
        print(f'| {c} | {est} | {len(X)} | {np.median(rel):.3g} | {float(X.std(0)[k]):.3g} (col {k}) |')

# %% 4. log p on the data, agreement across seeds (Theta_reg at the end and theta_t mean of last 50)
from data.data_loader import load_turbulence_1d                                           # noqa: E402
from codes.utils import split_periodize_reshape, normalize                                 # noqa: E402
from codes.ortho_wavelet.ReadyToUseWavelets import DefineWavelet                           # noqa: E402
from codes.filters_bank import return_Filters                                              # noqa: E402
from codes.potentials_builder import get_1d_potentials                                     # noqa: E402

Wv = DefineWavelet('Db', m=3, device='cpu')
raw = load_turbulence_1d()
xx = split_periodize_reshape(raw, 512)
for _ in range(3):
    xx = Wv.decompose(xx)[1]
per = xx.shape[0] // raw.shape[0]
g = np.random.default_rng(0)
sel = g.choice(raw.shape[0], N, replace=False)
x1 = normalize(xx[torch.as_tensor(sel * per + g.integers(0, per, N))])


def phi_on_data(cfg):
    import json
    seed_dir = Path(glob.glob(str(EXP / f'*d64_{cfg}' / f'*seed_900_*'))[0])
    terms = json.load(open(seed_dir / 'config.json'))
    terms = terms.get('args', terms)['terms']
    filters, filters_Phi = return_Filters(D, 6, 1, device='cpu', include_phi=True)
    filters_Q = return_Filters(D, 6, 3, device='cpu')
    with contextlib.redirect_stdout(io.StringIO()):
        pots = get_1d_potentials(terms, 6, filters, 3, filters_Q=filters_Q, filters_Phi=filters_Phi, scalar_param=None,
                                 parallel=False, deduplicate_filters=True, dedup_tol=1e-8)
        for name, pot in list(pots.items()):
            sp = seed_dir / 'fitted_potentials' / f'{name}.pt'
            if hasattr(pot, 'load_fixed_parameters') and sp.exists():
                pots[name] = type(pot).load_fixed_parameters(sp, pot.filters, map_location='cpu')
            elif hasattr(pot, 'fit'):
                pot.fit(x1)
    return torch.cat([p.forward(x1) for p in pots.values()], 1).double()


print('\n## 4. log p = theta . phi(x) on the 2000 data windows: agreement across seeds\n')
print('| config | estimate | seeds | corr between seed pairs: median / worst | Spearman median | '
      'seed spread / signal spread | largest per-signal seed spread (nats) |')
print('|---|---|---|---|---|---|---|')
LOGP = {}
for c in CONFIGS:
    phi = phi_on_data(c)
    for est in ('Theta_reg_last', 'theta_t_mean50'):
        X = TH[c][est]
        if X.shape[1] != phi.shape[1]:
            print(f'| {c} | {est} | column mismatch {X.shape[1]} vs {phi.shape[1]} | | | | |')
            continue
        lp = (phi @ X.T).numpy()                                            # (2000, seeds), code sign: log p = +theta.phi
        LOGP[(c, est)] = lp
        ns = lp.shape[1]
        cs = [np.corrcoef(lp[:, a], lp[:, b])[0, 1] for a in range(ns) for b in range(a + 1, ns)]
        sp = [spearmanr(lp[:, a], lp[:, b]).correlation for a in range(ns) for b in range(a + 1, ns)]
        spread = lp.std(1)
        print(f'| {c} | {est} | {ns} | {np.median(cs):.3f} / {np.min(cs):.3f} | {np.median(sp):.3f} | '
              f'{np.median(spread) / lp.mean(1).std():.3g} | {spread.max():.3g} |')
print(f'\nwrote {OUT / "stability.png"}')
