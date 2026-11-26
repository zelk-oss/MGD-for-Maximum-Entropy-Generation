"""Var/2d of the energy theta.phi per wavelet scale, for Scalar_psi (Q=3) and Scalar_morlet (Q=1) (2026-11-20).

Question: are the coarse scales of Scalar_psi cheap to drop? To first order D_k = KL(p || p without term k)
= 1/2 Var_p(theta_k . phi_k) (D_k >= the information I_k that term k carries), so a small Var/2d of the
coarse channels means their potential adds little. First order only: it underestimates D_k when the energy
is driven by rare events (higher cumulants).

Energy of statistic i of a Scalar_*_gaussianK potential: A_i(x) = theta_i * phi_i(x), theta = Theta_reg(t=1)
of the chosen lambda. Statistics (channel j, magnitude region k) are grouped per channel j. Var over
signals x, evaluated on the data x1 and on the model samples.
    Var/2d = Var(sum_i in channel A_i) / (2 d), d = M = signal length   [nats per dimension; was digits until 2026-11-25]
    nats total = Var/2d * d  (csv has both: var2d, half_var_nats).
"Dropped set" of a period P: all channels with scale >= P (incl. the low-pass); its Var is the Var of the SUM of
the energies (the terms are correlated, the sum of single Var/2 is not the group value).

Usage (Jean Zay, repo root):  python turbulence/var2d_per_scale/var2d_per_scale.py
Outputs (small) in --out: var2d_per_channel.csv, var2d_per_stat.csv, var2d_per_channel.png, var2d_dropped_set.png
"""
# %% imports
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in (ROOT / 'turbulence', ROOT, ROOT / 'codes', ROOT / 'codes/filters', ROOT / 'codes/potentials', ROOT / 'data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_comparison as rc                                         # noqa: E402
from logp_family_contrib import find_theta, load_potentials         # noqa: E402
from check_potentials import theta_column_map                       # noqa: E402

NAMES = {'psi': 'Scalar_psi_gaussianK', 'morlet': 'Scalar_morlet_gaussianK'}
TITLES = {'psi': 'Scalar_psi (Morlet Q=3)', 'morlet': 'Scalar_morlet (Morlet Q=1)'}
CHUNK = 500

# %% arguments
ap = argparse.ArgumentParser()
ap.add_argument('--label', default='turbulence_nested_M3_bicw_allmass_scat4')
ap.add_argument('--seed', type=int, default=900)
ap.add_argument('--lam', type=float, default=1e-3, help='rule A (cross-seed M, t>=0.99) for B+Scat4')
ap.add_argument('--ridge', type=float, default=1e-4)
ap.add_argument('--sweep_dir', type=Path, default=None,
                help='folder with <config>/lam*_ridge*_moment_mask.pt (default: SCRATCH theta_reg_lamsweep_xseed/bscat4)')
ap.add_argument('--theta_file', type=Path, default=None,
                help='Theta_reg file to use (e.g. a pinpos or droplast solve); overrides --lam/--ridge/--sweep_dir lookup')
ap.add_argument('--n_max', type=int, default=None, help='use only the first n signals (smoke test)')
ap.add_argument('--out', type=Path, default=HERE / 'results')
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
if args.sweep_dir is None and args.theta_file is None:
    args.sweep_dir = (Path(os.environ['SCRATCH']) / 'MGD-for-Maximum-Entropy-Generation/turbulence'
                      / 'theta_reg_lamsweep_xseed/bscat4')

# %% run, Theta_reg(1), potentials, signals
config = rc._find_config(rc.ROOT, args.label, args.seed)
assert config is not None, f'no samples for {args.label} seed {args.seed}'
exp = rc._find_exp_dir(rc.ROOT, config)
assert exp is not None, f'no experiments folder for {config}'
run = rc.load_config(config, rc.ROOT, label=args.label, extras=False)
cfg = run['cfg']
M = int(cfg['M'])
# periods are reported in ORIGINAL samples (before coarse graining), so runs at different M / coarse graining
# compare at the same physical scale (user decision 2026-11-25). M=256 runs: 1 grid sample = 2 original samples.
GRID_TO_ORIG = int(cfg['subseries_len']) / int(cfg['target_len'])
if args.theta_file is not None:
    sol = torch.load(args.theta_file, map_location='cpu', weights_only=False)
    assert sol['config'] == config, (sol['config'], config)
    theta, desc = sol['Theta_reg'][-1].double().numpy(), f"{args.theta_file.name} lam={sol['lam']:g}"
    args.lam = float(sol['lam'])
else:
    theta, desc = find_theta(config, args.lam, args.ridge, 1e-2, False, args.sweep_dir,
                             Path('/nonexistent'), unconstrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pots = load_potentials(exp, cfg, device)
cmap = theta_column_map(pots, theta.shape[0])
signals = {'data': rc.reference_data(cfg).double(), 'model': run['xt'].double()}
if args.n_max:
    signals = {k: v[:args.n_max] for k, v in signals.items()}
print(f'run {config}\n  {desc}\n  signals: ' + ', '.join(f'{k} {tuple(v.shape)}' for k, v in signals.items()))


def energies(pot, th, x):
    """A_i(x) = theta_i phi_i(x), (N, n_stat) float64."""
    out = []
    with torch.no_grad():
        for i in range(0, len(x), CHUNK):
            out.append(pot(x[i:i + CHUNK].reshape(-1, 1, M).to(device)).detach().cpu().double())
    return torch.cat(out).numpy() * th[None]


# %% Var of the energy per statistic and per channel
per_dim = 1.0 / (2 * M)                           # Var -> Var/2d  [nats per dimension], d = M (user 2026-11-25)
stat_rows, chan_rows, A_chan, filt = [], [], {}, {}
for key, name in NAMES.items():
    pot = pots[name]
    start, width = cmap[name]
    th = theta[start:start + width]
    flat = pot.active_flat.cpu().tolist()
    assert len(flat) == width, (name, len(flat), width)
    J = int(pot.J)
    slot, chan = np.array([divmod(f, J)[0] for f in flat]), np.array([divmod(f, J)[1] for f in flat])
    f = pot.filters.reshape(-1, M).cpu()
    xi = np.abs(np.fft.fftfreq(M))[f.abs().argmax(-1).numpy()]       # centre frequency, cycles/sample
    period = np.where(xi == 0, np.inf, 1 / np.maximum(xi, 1e-12)) * GRID_TO_ORIG   # original samples
    filt[key] = (f.to(torch.complex128).numpy(), period)
    for src, x in signals.items():
        A = energies(pot, th, x)                                     # (N, n_stat)
        var_stat = A.var(0, ddof=1)
        for i in range(width):
            stat_rows.append(dict(potential=key, source=src, stat=i, channel=int(chan[i]), slot=int(slot[i]),
                                  period=period[chan[i]], theta=th[i], half_var_nats=var_stat[i] / 2))
        for j in range(J):
            cols = np.flatnonzero(chan == j)
            if cols.size == 0:
                continue
            Aj = A[:, cols].sum(1)
            A_chan[(key, src, j)] = Aj
            chan_rows.append(dict(potential=key, source=src, channel=j, period=period[j], n_stats=cols.size,
                                  half_var_nats=Aj.var(ddof=1) / 2, var2d=Aj.var(ddof=1) * per_dim,
                                  var2d_sum_of_singles=var_stat[cols].sum() * per_dim))
    print(f'{key}: {width} statistics in {len(set(chan.tolist()))} of {J} channels')

for fname, rows in (('var2d_per_stat.csv', stat_rows), ('var2d_per_channel.csv', chan_rows)):
    with open(args.out / fname, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)

# %% dropped set = all channels with period >= P (Var of the sum), per potential and source
drop_rows = []
for key in NAMES:
    for src in signals:
        ch = sorted([r for r in chan_rows if r['potential'] == key and r['source'] == src],
                    key=lambda r: -r['period'])                     # coarsest first (low-pass has period inf)
        acc = np.zeros(len(signals[src]))
        for r in ch:
            acc = acc + A_chan[(key, src, r['channel'])]
            drop_rows.append(dict(potential=key, source=src, period_min=r['period'], n_channels=0,
                                  var2d=acc.var(ddof=1) * per_dim, half_var_nats=acc.var(ddof=1) / 2))
        for k, r in enumerate([r for r in drop_rows if r['potential'] == key and r['source'] == src]):
            r['n_channels'] = k + 1
with open(args.out / 'var2d_dropped_set.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(drop_rows[0]))
    wr.writeheader(); wr.writerows(drop_rows)

# %% compensation between the two banks. Each Q=3 channel is assigned to the Q=1 channel with the most similar filter
# (overlap = |<f_psi, f_morlet>| / (|f_psi| |f_morlet|) in Fourier space, 1 = same filter). If theta_psi, theta_morlet are
# large with opposite sign (near-collinear statistics), each group has a large Var but the SUM of the two energies is small.
fp, per_p = filt['psi']
fm, per_m = filt['morlet']
ov = np.abs(fp.conj() @ fm.T) / np.outer(np.linalg.norm(fp, axis=1), np.linalg.norm(fm, axis=1))   # (C_psi, C_mor)
HALF = lambda a: a.var(ddof=1) / 2                                                                  # noqa: E731
# one Q=1 channel against ALL the Q=3 channels whose most similar Q=1 channel it is (the Q=3 filters of one octave
# together span the Q=1 filter of that octave): energy of the group = sum of the energies of its Q=3 channels
best = {j: int(np.argmax(ov[j])) for j in range(ov.shape[0])}
group_rows = []
for src in signals:
    def add(label, psi_chs, mor_chs, per):
        a = sum(A_chan[('psi', src, j)] for j in psi_chs)
        b = sum(A_chan[('morlet', src, m)] for m in mor_chs)
        group_rows.append(dict(source=src, group=label, period_morlet=per, n_psi_channels=len(psi_chs),
                               psi_channels=' '.join(map(str, psi_chs)), half_var_psi=HALF(a), half_var_morlet=HALF(b),
                               corr=np.corrcoef(a, b)[0, 1], half_var_sum=HALF(a + b)))
    for m in sorted(set(best.values())):
        chs = [j for j in sorted(best) if best[j] == m and ('psi', src, j) in A_chan]
        add(f'Q=1 ch {m}', chs, [m], per_m[m])
    # the two coarsest Q=1 octaves of the M=256 report (grid 128 and 256 = 256 and 512 ORIGINAL samples), kept at the
    # same physical scale; coarser octaves (J=9, 10 at M=1024) are separate per-channel rows above
    coarse = [m for m in range(len(per_m)) if 256 / 1.2 <= per_m[m] <= 512 * 1.2]
    if coarse:
        add('periods 256 and 512 orig. samples together (no low-pass)',
            [j for j in sorted(best) if best[j] in coarse], coarse, np.nan)
    # Q=3 band of the M=256 report (grid 21-85 = 42-170 original samples): Var of the SUM of its channels' energies
    band = [j for j in range(len(per_p)) if 42 <= per_p[j] <= 170 and ('psi', src, j) in A_chan]
    if band:
        a = sum(A_chan[('psi', src, j)] for j in band)
        group_rows.append(dict(source=src, group='Q=3 band 42-170 orig. samples (sum)', period_morlet=np.nan,
                               n_psi_channels=len(band), psi_channels=' '.join(map(str, band)), half_var_psi=HALF(a),
                               half_var_morlet=np.nan, corr=np.nan, half_var_sum=HALF(a)))
with open(args.out / 'var2d_bank_groups.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(group_rows[0]))
    wr.writeheader(); wr.writerows(group_rows)
print('\nQ=1 channel vs the group of Q=3 channels around it (data):')
for r in group_rows:
    if r['source'] == 'data':
        print(f"  {r['group'][:44]:44s} period {r['period_morlet']:6.1f}  Q3 ch [{r['psi_channels']:9s}]  half Var Q3 "
              f"{r['half_var_psi']:9.3g}  Q1 {r['half_var_morlet']:9.3g}  corr {r['corr']:+.3f}  sum {r['half_var_sum']:9.3g}")

# %% figures
COL = {'data': 'k', 'model': 'tab:green'}
LAB = {'data': 'data x1', 'model': 'model samples'}


def figure(rows, ycol, ylabel, title, fname, cumulative):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, key in zip(axes, NAMES):
        for src in signals:
            r = [q for q in rows if q['potential'] == key and q['source'] == src]
            per = np.array([q['period' if not cumulative else 'period_min'] for q in r])
            val = np.array([q[ycol] for q in r])
            fin = np.isfinite(per)
            ax.plot(per[fin], val[fin], 'o-', color=COL[src], ms=4, label=LAB[src])
            if (~fin).any():                                          # low-pass: drawn at the right edge, marked
                ax.plot([per[fin].max() * 1.6], val[~fin], 's', color=COL[src], ms=6, mfc='none')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(TITLES[key]); ax.set_xlabel('scale = 1 / centre frequency  [original samples]  (square = low-pass)')
        ax.grid(alpha=0.3, which='both')
    axes[0].set_ylabel(ylabel); axes[0].legend(fontsize=8)
    if cumulative:
        for ax in axes:
            ax.invert_xaxis()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / fname, dpi=150)


figure(chan_rows, 'var2d', 'Var/2d of the channel energy  [nats per dimension]',
       f'Var(theta.phi)/2d per channel, Theta_reg lam={args.lam:g}, seed {args.seed}', 'var2d_per_channel.png', False)
figure(drop_rows, 'var2d', 'Var/2d of the dropped set  [nats per dimension]',
       'Var/2d of the energy of ALL channels with scale >= x (what removing them would cost, 1st order)',
       'var2d_dropped_set.png', True)
print(f'saved in {args.out}: var2d_per_channel.{{csv,png}}, var2d_per_stat.csv, var2d_dropped_set.{{csv,png}}, '
      'var2d_bank_groups.csv')
