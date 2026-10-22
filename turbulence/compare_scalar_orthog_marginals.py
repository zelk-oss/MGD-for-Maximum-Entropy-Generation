"""On-own-bank wavelet marginals of the four scalar_orthog arms, side by side.

The automatic histograms.png produced by run_SDE.py uses a FIXED Morlet Q=3 bank for
analysis -- it is the right diagnostic for comparing runs that all share Morlet Q=3,
but NOT for the orthogonality test: the shannon arms constrain moments on the Shannon
Q=1 bank, so the Q=3 Morlet histogram evaluates them on a bank they never saw. This
script re-analyses each arm's generated samples on the SAME bank the arm itself
imposed moments on (morlet Q=1 for the morlet arms, shannon Q=1 for the shannon
arms), and overlays data vs synth per channel. The 4 arms are plotted side by side
in a 7 x 4 grid (J=7 fine-to-coarse rows, one arm per column).

Answers the prof's test: on an orthogonal wavelet basis, imposing scalar GGD-K
moments per channel should recover the marginals (because the full density
factorises). If the shannon_* columns show matched tails and morletQ1_* columns
don't -> orthogonality is the key. If shannon's tails are still too narrow ->
K=3-4 scalar moments per channel are insufficient regardless of orthogonality.

Usage (from the repo root):
    python turbulence/compare_scalar_orthog_marginals.py
    python turbulence/compare_scalar_orthog_marginals.py --seed 900 --out mycompare.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent          # turbulence/
ROOT = HERE.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'data', ROOT / 'codes/filters'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils import normalize, split_periodize_reshape                            # noqa: E402
from ortho_wavelet.ReadyToUseWavelets import DefineWavelet                      # noqa: E402
from data.data_loader import load_turbulence_1d                                 # noqa: E402
from filters_1d import init_band_pass                                           # noqa: E402

ARMS_COARSE = ('morletQ1_L6', 'morletQ1_noL6', 'shannon_L6', 'shannon_noL6')
ARMS_FULL = tuple(a + '_full' for a in ARMS_COARSE)
ARMS_2BANK = ('morletQ1Q3_L6_2b', 'morletQ1Q3_noL6_2b',
              'shannonQ1Q3_L6_2b', 'shannonQ1Q3_noL6_2b')
ARMS_NOSCAT4 = ('noScat4',)                     # the full M=256 noScat4 run
ARMS_FINAL = ('final',)                         # turbulence_final: noScat4 + Scat4

WAV_FOR_ARM = {}
for a in ARMS_COARSE + ARMS_FULL + ARMS_2BANK:
    WAV_FOR_ARM[a] = 'shannon' if a.startswith('shannon') else 'morlet'
WAV_FOR_ARM['noScat4'] = 'morlet'               # noScat4 uses morlet both Q=1 and Q=3
WAV_FOR_ARM['final']   = 'morlet'               # same bank layout as noScat4

# How to turn an arm name into a sample-file glob.
# scalar_orthog* arms live under 'turbulence_scalar_orthog_<label>'; the trailing
# '_2*' anchors the end of the label against the start of the YYYYMMDD timestamp
# (e.g. ..._2026...). noScat4 is the final-run family 'turbulence_final_noScat4'.
# For 'final' (= turbulence_final with Scat4) we must exclude the '_noScat4_'
# variant: '_20*' anchors directly on the timestamp that follows the bare label.
LABEL_GLOB = {}
for a in ARMS_COARSE + ARMS_FULL + ARMS_2BANK:
    LABEL_GLOB[a] = f'*_seed_{{seed}}_*turbulence_scalar_orthog_{a}_2*.pt'
LABEL_GLOB['noScat4'] = '*_seed_{seed}_*turbulence_final_noScat4_2*.pt'
LABEL_GLOB['final']   = '*_seed_{seed}_*turbulence_final_20*.pt'

ARMS_TAILFIX = ('tailfix_noL6', 'tailfix_L6')   # turbulence_ggd_tailfix.sh, M=128 Q=1
ARMS_TAILFIX_FULL = ('tailfix_noL6_full', 'tailfix_L6_full')  # turbulence_ggd_tailfix_full.sh, M=256 Q=1
ARMS_TAILFIX_NOSCAT4 = ('tailfix_noScat4',)                   # noScat4-equivalent + tail fix
ARMS_TAILFIX_FINAL = ('tailfix_final',)                       # final-equivalent (with Scat4) + tail fix
ARMS_MASS_2BANK = ('mass_noL6_2b', 'mass_L6_2b', 'tailq_noL6_2b', 'tailq_L6_2b', 'mass_L6_2b_drop', 'mass_L6_2b_merge',
                   'mass_L6_2b_scat4')  # turbulence_ggd_mass_2bank.sh, M=128 Q=1+3
ARMS_MASS_1BANK = ('mass_L6_1b',)                            # same launcher, M=128 Q=1 only
ARMS_MASS_NOSCAT4 = ('mass_noScat4',)                        # turbulence_ggd_mass_noScat4.sh, M=256 Q=1+3
for a in (ARMS_TAILFIX + ARMS_TAILFIX_FULL + ARMS_TAILFIX_NOSCAT4 + ARMS_TAILFIX_FINAL
          + ARMS_MASS_2BANK + ARMS_MASS_1BANK + ARMS_MASS_NOSCAT4):
    WAV_FOR_ARM[a] = 'morlet'
    LABEL_GLOB[a] = f'*_seed_{{seed}}_*turbulence_ggd_{a}_2*.pt'
# turbulence_final_coarse.sh, ONLY=noL2 WITH_SCAT4=true: M=128 Q=1+3, L_6 + Scalar + Scat4
ARMS_COARSE_FINAL = ('coarse_noL2_scat4',)
WAV_FOR_ARM['coarse_noL2_scat4'] = 'morlet'
LABEL_GLOB['coarse_noL2_scat4'] = '*_seed_{seed}_*turbulence_final_coarse_noL2_scat4_2*.pt'
# M=256 (J=8) arms; the rest are M=128 (J=7)
ARMS_M256 = (ARMS_FULL + ARMS_NOSCAT4 + ARMS_FINAL + ARMS_TAILFIX_FULL + ARMS_TAILFIX_NOSCAT4
             + ARMS_TAILFIX_FINAL + ARMS_MASS_NOSCAT4)
ARMS_Q13 = (ARMS_2BANK + ARMS_MASS_2BANK + ARMS_COARSE_FINAL + ARMS_NOSCAT4 + ARMS_FINAL
            + ARMS_TAILFIX_NOSCAT4 + ARMS_TAILFIX_FINAL + ARMS_MASS_NOSCAT4)


def find_sample(label, seed):
    glob = LABEL_GLOB[label].format(seed=seed)
    files = sorted((HERE / 'saved_results' / 'samples').glob(glob),
                   key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f'no samples for {label} seed {seed} (glob {glob})')
    return files[-1]


def rebuild_data_x1(subseries_len, target_len, n1):
    W = DefineWavelet('Db', m=3, device='cpu')
    Data = load_turbulence_1d().to('cpu')
    Data = split_periodize_reshape(Data, subseries_len)
    for _ in range(int(np.log2(subseries_len / target_len))):
        Data = W.decompose(Data)[1]
    x1 = normalize(Data[:n1])
    return x1.squeeze(1)                                       # (n1, M)


def wavelet_coeffs(x, wav_type, J, Q=1, M=None):
    """FFT-domain convolution with the J bandpass filters of the given family.
    Returns (B, J, M) real coefficients. Mirrors codes/potentials/potentials_1d.py."""
    M = x.shape[-1] if M is None else M
    psi = init_band_pass(wav_type, M, J, Q, high_freq=0.49, wav_norm='l1')
    psi = torch.tensor(np.asarray(psi), dtype=torch.float32)   # (J, M), Fourier
    X = torch.fft.fft(x, dim=-1)                               # (B, M) or (B, 1, M)
    if X.dim() == 3:
        X = X.squeeze(1)
    z = torch.fft.ifft(X.unsqueeze(1) * psi.unsqueeze(0)).real # (B, J, M)
    return z


def plot_grid(x1, xts, wavs, arms, J, out_path, Q=1, q_range=(0.001, 0.999)):
    n_arms = len(xts)
    n_ch = J * Q
    fig, axes = plt.subplots(n_ch, n_arms, figsize=(3.4 * n_arms, 1.6 * n_ch),
                             sharey='row', squeeze=False)
    for col, (arm, xt, wav) in enumerate(zip(arms, xts, wavs)):
        z_data = wavelet_coeffs(x1, wav, J, Q=Q).reshape(n_ch, -1).numpy()
        z_synth = wavelet_coeffs(xt, wav, J, Q=Q).reshape(n_ch, -1).numpy()
        for j in range(n_ch):
            ax = axes[j, col]
            lo = min(np.quantile(z_data[j], q_range[0]), np.quantile(z_synth[j], q_range[0]))
            hi = max(np.quantile(z_data[j], q_range[1]), np.quantile(z_synth[j], q_range[1]))
            if not (lo < hi):
                lo, hi = -1, 1
            bins = np.linspace(lo, hi, 80)
            ax.hist(z_data[j], bins=bins, log=True, histtype='step',
                    label='data', color='C0', lw=1.4)
            ax.hist(z_synth[j], bins=bins, log=True, histtype='step',
                    label='synth', color='C1', lw=1.4)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(f'ch={j}', fontsize=8)
            if j == 0:
                ax.set_title(f'{arm}\n(bank: {wav} Q={Q})', fontsize=9)
            if j == 0 and col == 0:
                ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f'wavelet-coefficient marginals on arm\'s own Q={Q} bank '
                 '(ch=0 finest, ch=J*Q-1 coarsest)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f'figure: {out_path}')


def main(seed, out_base, mode, arms_override=None):
    if arms_override:
        # any registered arms side by side; Q values follow the arms' bank layout
        arms = tuple(arms_override)
        J = 8 if all(a in ARMS_M256 for a in arms) else 7
        Qs = (1, 3) if all(a in ARMS_Q13 for a in arms) else (1,)
    elif mode == 'full':
        arms, J = ARMS_FULL, 8
        Qs = (1,)
    elif mode == 'two_bank':
        arms, J = ARMS_2BANK, 7
        Qs = (1, 3)
    elif mode == 'noScat4':
        # Replay the full-resolution noScat4 run (M=256, J=8, 2 banks of morlet
        # constrained by Scalar_morlet Q=1 + Scalar_psi Q=3 + L_6) with the SAME
        # honest-binning (shared bins, counts, step) used for scalar_orthog, to tell
        # whether the "narrow tails" seen through hist_plot are a plotting artifact
        # (per-hist bins + density=True) or a real model symptom.
        arms, J = ARMS_NOSCAT4, 8
        Qs = (1, 3)
    else:
        arms, J = ARMS_COARSE, 7
        Qs = (1,)
    xts, wavs = [], []
    for arm in arms:
        p = find_sample(arm, seed)
        xt = torch.load(p, map_location='cpu').float()
        if xt.dim() == 3:
            xt = xt.squeeze(1)
        print(f'{arm:24s}  {p.name}  xt shape {tuple(xt.shape)}')
        xts.append(xt)
        wavs.append(WAV_FOR_ARM[arm])

    M = xts[0].shape[-1]
    assert all(x.shape[-1] == M for x in xts), 'arms differ in signal length'
    x1 = rebuild_data_x1(subseries_len=512, target_len=M, n1=xts[0].shape[0])
    print(f'data x1 shape {tuple(x1.shape)}')

    for Q in Qs:
        suffix = f'_Q{Q}' if len(Qs) > 1 else ''
        out = out_base.with_name(out_base.stem + suffix + out_base.suffix)
        # --arms: full range, so the extreme tails are on the plot
        plot_grid(x1, xts, wavs, arms, J, out, Q=Q,
                  q_range=(0.0, 1.0) if arms_override else (0.001, 0.999))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=900)
    ap.add_argument('--full', action='store_true',
                    help='M=256 full-resolution arms (turbulence_scalar_orthog_full.sh)')
    ap.add_argument('--two-bank', action='store_true',
                    help='M=128 2-bank arms (turbulence_scalar_orthog_2bank.sh); '
                         'generates two figures, one per Q value')
    ap.add_argument('--noScat4', action='store_true',
                    help='re-plot turbulence_final_noScat4 (M=256, 2 banks, 3 '
                         'families) with the same honest-binning used here; '
                         'generates two figures, one per Q value')
    ap.add_argument('--arms', nargs='+', default=None, choices=sorted(LABEL_GLOB),
                    help='plot these arms (all M=128 or all M=256) instead of a preset; '
                         'two-bank arms give one figure per Q')
    ap.add_argument('--out', type=Path, default=None,
                    help='figure path (or stem for multi-Q modes); default picks a '
                         'name reflecting the mode')
    a = ap.parse_args()
    picked = sum(bool(x) for x in (a.full, a.two_bank, a.noScat4))
    if picked > 1:
        raise SystemExit('pick at most one of --full / --two-bank / --noScat4')
    if a.full:
        mode = 'full'
    elif a.two_bank:
        mode = 'two_bank'
    elif a.noScat4:
        mode = 'noScat4'
    else:
        mode = 'coarse'
    default_stem = {'coarse':   'scalar_orthog_marginals_own_bank',
                    'full':     'scalar_orthog_marginals_own_bank_full',
                    'two_bank': 'scalar_orthog_marginals_own_bank_2bank',
                    'noScat4':  'noScat4_marginals_own_bank'}[mode]
    if a.arms:
        default_stem = 'marginals_own_bank_' + '_'.join(a.arms)
    out = a.out or (HERE / 'figures' / f'{default_stem}.png')
    main(a.seed, out, mode, arms_override=a.arms)
