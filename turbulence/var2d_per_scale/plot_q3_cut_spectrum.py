"""Where the Q=3 channel cut of report_scale_relevance.ipynb falls on the velocity spectrum of the Lagrangian data.

The report ran at M=256 (512 original samples coarse-grained once, 1 grid sample = 2 original samples) and keeps the
Q=3 channels with centre period <= 16 grid samples = 32 original samples; dropped: >= 42 original samples and the
low-pass. At M=1024 these are channels k >= 13 (the bank gains 3 finer channels, k = 0-2, not in the report).
tau_eta from Li et al. 2024, Table 1 (same DNS): 2.1e-2 time units, sampling step 2.25e-3.
Run from the repo root: python turbulence/var2d_per_scale/plot_q3_cut_spectrum.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'lagr_vs_old_turbulence'))
sys.path.insert(0, str(ROOT))
import compare_lib as c

HERE = Path(__file__).resolve().parent

raw, _ = c.load_raw()
x = c.prepare(raw['new'])                       # (10000, 1024) velocity signals, standardised
L = x.shape[1]
f = np.arange(1, L // 2 + 1) / L
P = (np.abs(np.fft.rfft(x, axis=1)) ** 2).mean(0)[1:]
V = P / P.sum()

r = pd.read_csv(HERE / 'results' / 'var2d_per_channel.csv')
per = r[(r.potential == 'psi') & (r.source == 'data')].period.values
per = np.unique(per[np.isfinite(per)])          # distinct centre periods of the report's Q=3 channels (M=256 grid samples)
# the report ran at M=256 = 512 original samples coarse-grained once: 1 grid sample = 2 original samples. At M=1024
# (original sampling) the same channels sit at twice the period, and the bank gains 3 finer channels (one octave).
new = per[:3]                                   # periods 2-3.2 original samples, not in the report
per = np.concatenate([new, 2 * per])
fc = 1 / per
keep = per <= 32                                # the report keeps periods <= 16 grid samples = 32 original samples
f_cut = 1 / np.sqrt(32 * per[~keep].min())      # between the last kept and the first dropped channel
f_eta = 2.25e-3 / (2 * np.pi * 2.1e-2)          # omega * tau_eta = 1

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axvspan(f_cut, 0.5, color='tab:green', alpha=0.12)
ax.axvspan(f[0], f_cut, color='0.6', alpha=0.15)
ax.loglog(f, V, color='k', lw=2, label='velocity spectrum (data, 1024 samples)')
Vc = np.interp(fc, f, V)
isnew = np.isin(per, new)
ax.plot(fc[keep & ~isnew], Vc[keep & ~isnew], 'o', color='tab:green', ms=8,
        label='Q=3 channels KEPT (period ≤ 32 samples)')
ax.plot(fc[isnew], Vc[isnew], 'o', mfc='white', mec='tab:green', mew=2, ms=8,
        label='Q=3 channels KEPT, new at M=1024 (not in the report)')
ax.plot(fc[~keep], Vc[~keep], 'X', color='tab:red', ms=8, label='Q=3 channels DROPPED (period ≥ 42 samples)')
ax.axvline(f_eta, color='tab:blue', lw=1.5, label=r'$\tau_\eta$ (Kolmogorov time): f = 0.017')
g = np.geomspace(f[0], f_eta, 50)
ax.loglog(g, 3 * V[0] * (g / f[0]) ** -2, color='tab:blue', ls='--', lw=1.2,
          label=r'inertial range (K41, $\propto f^{-2}$)')
ax.text(0.14, 2e-1, 'Q=3 kept', color='tab:green', ha='center', fontsize=10)
ax.text(3.2e-3, 3e-5, 'Q=3 dropped:\nonly Q=1 left', color='0.3', ha='center', fontsize=10)
# the two physical ranges, in a band above the plot
top = ax.get_xaxis_transform()
for lo, hi, txt in ((f[0], f_eta, 'inertial range'), (f_eta, 0.5, 'dissipative range')):
    ax.annotate('', xy=(lo, 1.03), xytext=(hi, 1.03), xycoords=top, textcoords=top,
                arrowprops=dict(arrowstyle='<->', color='tab:blue', lw=1), annotation_clip=False)
    ax.text(np.sqrt(lo * hi), 1.05, txt, transform=top, ha='center', va='bottom', fontsize=9, color='tab:blue')
ax.annotate('coarser channels\n+ low-pass: dropped', xy=(1.1e-3, 3e-1), xytext=(2.2e-3, 2e-3),
            fontsize=8, color='tab:red', arrowprops=dict(arrowstyle='->', color='tab:red'))
for p, fx, vy in zip(per, fc, Vc):
    if p in (2.048, 32, per[~keep].min(), 128):
        ax.annotate(f'{p:.3g}', (fx, vy), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
ax.set_xlabel('frequency f (cycles per original sample)   [period = 1/f]')
ax.set_ylabel('variance share per bin')
ax.set_title('Where the Q=3 cut of the scale-relevance report falls (channel periods in samples)', fontsize=10,
             pad=24)
ax.legend(fontsize=8, loc='lower left')
fig.tight_layout()
fig.savefig(HERE / 'results' / 'q3_cut_on_spectrum.png', dpi=130)
