"""Numeric agreement data vs synth on wavelet-coefficient marginals.

Rationale (2026-10-20): the per-array binning + density=True in
`check_moments.hist_plot` AND the quantile-clipping in
`compare_scalar_orthog_marginals.plot_grid` BOTH hide tail mismatches that
`marginal_agreement` (moments + quantile ratios + W1) resolves cleanly:
noScat4 at ch=0 Q=1 has q0.9999_ratio=0.50 and m2_ratio=0.67, invisible in
either plot. This script applies the same test to every arm of the 2x2 cube
and prints axis-isolation deltas to locate where the mismatch originates:
intrinsic to the simplest model, resolution-driven, bank-driven, orthogonality-
driven, or L_6-driven.

Design: reuse the arm registry and wavelet-coefficient code from
`compare_scalar_orthog_marginals.py` (same glob logic, same filters, same
Db-decimation of the data) so results are directly comparable to those figures.
Produces no new figures: just one CSV per (arm, bank Q) and one summary txt.

Usage (from the repo root on Jean Zay):
    python turbulence/compute_marginal_agreement.py
    python turbulence/compute_marginal_agreement.py --seed 900 --no-skip-missing
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

HERE = Path(__file__).resolve().parent          # turbulence/
ROOT = HERE.parent
for p in (ROOT, ROOT / 'codes', ROOT / 'data', ROOT / 'codes/filters'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# reuse the arm registry and the EXACT wavelet-coefficient pipeline used by
# the figures. no duplication: if a glob or filter changes there, it propagates.
from compare_scalar_orthog_marginals import (          # noqa: E402
    ARMS_COARSE, ARMS_FULL, ARMS_2BANK, ARMS_NOSCAT4, ARMS_FINAL,
    ARMS_TAILFIX, ARMS_TAILFIX_FULL, ARMS_TAILFIX_NOSCAT4, ARMS_TAILFIX_FINAL, ARMS_MASS_2BANK,
    ARMS_MASS_1BANK, ARMS_MASS_NOSCAT4, ARMS_COARSE_FINAL,
    WAV_FOR_ARM, find_sample, rebuild_data_x1, wavelet_coeffs,
)
ARMS_2BANK = ARMS_2BANK + ARMS_MASS_2BANK + ARMS_COARSE_FINAL   # same M=128, J=7, Q=1+3 setup
ARMS_COARSE = ARMS_COARSE + ARMS_TAILFIX + ARMS_MASS_1BANK   # same M=128, J=7, Q=1 setup
ARMS_FULL = ARMS_FULL + ARMS_TAILFIX_FULL       # same M=256, J=8, Q=1 setup
ARMS_NOSCAT4 = ARMS_NOSCAT4 + ARMS_TAILFIX_NOSCAT4 + ARMS_MASS_NOSCAT4  # same M=256, J=8, Q=1+3 setup
ARMS_FINAL = ARMS_FINAL + ARMS_TAILFIX_FINAL        # same M=256, J=8, Q=1+3, with Scat4

# ────────────────────────────────────────────────────────────────────────────
# which banks each arm imposes moments on (what we evaluate agreement on):
# scalar_orthog arms (coarse + full) only have Q=1; 2bank + noScat4 + final
# have Q=1+3.
ARM_QS = {}
for a in ARMS_COARSE + ARMS_FULL:
    ARM_QS[a] = (1,)
for a in ARMS_2BANK + ARMS_NOSCAT4 + ARMS_FINAL:
    ARM_QS[a] = (1, 3)

# J depends on M (M=128 → J=7, M=256 → J=8); the launchers hard-coded this.
ARM_J = {}
for a in ARMS_COARSE + ARMS_2BANK:
    ARM_J[a] = 7
for a in ARMS_FULL + ARMS_NOSCAT4 + ARMS_FINAL:
    ARM_J[a] = 8

ARM_M = {}
for a in ARMS_COARSE + ARMS_2BANK:
    ARM_M[a] = 128
for a in ARMS_FULL + ARMS_NOSCAT4 + ARMS_FINAL:
    ARM_M[a] = 256


MOMENTS = (2, 4, 6, 8, 10)
QUANTILES = (0.9, 0.99, 0.999, 0.9999)
WASSERSTEIN_N = 200_000


def marginal_agreement(z_d, z_s, seed=0):
    """Return a DataFrame indexed by wavelet channel with:
       - m{p}_ratio   = <|z_s|^p> / <|z_d|^p>  for p in MOMENTS
       - q{a}_ratio   = q_a(|z_s|) / q_a(|z_d|)  for a in QUANTILES
       - W1_over_std  = wasserstein_1(z_d, z_s) / std(z_d), downsampled.
    Perfect agreement = all ratios ≈ 1, W1_over_std ≈ 0."""
    # (B, nch, M) → (nch, B*M) numpy
    def _flat(z):
        return z.permute(1, 0, 2).reshape(z.shape[1], -1).numpy()
    zd, zs = _flat(z_d), _flat(z_s)
    rng = np.random.default_rng(seed)
    rows = []
    for ch in range(zd.shape[0]):
        d, s = zd[ch], zs[ch]
        row = {'ch': ch}
        ad, as_ = np.abs(d), np.abs(s)
        for p in MOMENTS:
            row[f'm{p}_ratio'] = float(np.mean(as_ ** p) / np.mean(ad ** p))
        qd = np.quantile(ad, QUANTILES)
        qs = np.quantile(as_, QUANTILES)
        for a, rd, rs in zip(QUANTILES, qd, qs):
            row[f'q{a}_ratio'] = float(rs / rd)
        n = min(len(d), WASSERSTEIN_N)
        row['W1_over_std'] = float(wasserstein_distance(
            d[rng.choice(len(d), n, replace=False)],
            s[rng.choice(len(s), n, replace=False)],
        ) / d.std())
        rows.append(row)
    return pd.DataFrame(rows).set_index('ch')


# ────────────────────────────────────────────────────────────────────────────
# axis-isolation pair list: (name, armA, armB, Q)
# Delta = B − A on key metrics; by sign convention a NEGATIVE delta on
# q0.9999_ratio means arm B has thinner tails than A (ratio further below 1).
PAIRS = [
    # ── risoluzione: coarse vs full, same wavelet & L_6 setting, Q=1 bank
    ('risoluzione (morlet,  noL6)', 'morletQ1_noL6',       'morletQ1_noL6_full',  1),
    ('risoluzione (morlet,   L6)',  'morletQ1_L6',         'morletQ1_L6_full',    1),
    ('risoluzione (shannon, noL6)', 'shannon_noL6',        'shannon_noL6_full',   1),
    ('risoluzione (shannon,  L6)',  'shannon_L6',          'shannon_L6_full',     1),
    # ── ortogonalità: morlet vs shannon, same M & L_6, Q=1 bank
    ('ortogonalità (M=128, noL6)',  'morletQ1_noL6',       'shannon_noL6',        1),
    ('ortogonalità (M=256, noL6)',  'morletQ1_noL6_full',  'shannon_noL6_full',   1),
    ('ortogonalità (M=128,  L6)',   'morletQ1_L6',         'shannon_L6',          1),
    ('ortogonalità (M=256,  L6)',   'morletQ1_L6_full',    'shannon_L6_full',     1),
    # ── Q=3 aggiunto: at M=128, 1-bank vs 2-bank, compared on the Q=1 bank
    #    (that's the bank present in both configs → clean isolation of the
    #    "adding Q=3 competes with Q=1" effect)
    ('Q=3 aggiunto (morlet,  noL6)',   'morletQ1_noL6',    'morletQ1Q3_noL6_2b',  1),
    ('Q=3 aggiunto (morlet,   L6)',    'morletQ1_L6',      'morletQ1Q3_L6_2b',    1),
    ('Q=3 aggiunto (shannon, noL6)',   'shannon_noL6',     'shannonQ1Q3_noL6_2b', 1),
    ('Q=3 aggiunto (shannon,  L6)',    'shannon_L6',       'shannonQ1Q3_L6_2b',   1),
    # ── L_6: noL6 vs L6, same everything else, Q=1 bank
    ('L_6 (coarse,  morlet)',    'morletQ1_noL6',          'morletQ1_L6',          1),
    ('L_6 (full,    morlet)',    'morletQ1_noL6_full',     'morletQ1_L6_full',     1),
    ('L_6 (coarse,  shannon)',   'shannon_noL6',           'shannon_L6',           1),
    ('L_6 (full,    shannon)',   'shannon_noL6_full',      'shannon_L6_full',      1),
    ('L_6 (2bank,   morlet)',    'morletQ1Q3_noL6_2b',     'morletQ1Q3_L6_2b',     1),
    ('L_6 (2bank,   shannon)',   'shannonQ1Q3_noL6_2b',    'shannonQ1Q3_L6_2b',    1),
    # ── sanity cross-check: cella M=128+2bank+L6 vs cella M=256+2bank+L6+3fam
    #    non isola un singolo asse (confond. M + aggiunta terza famiglia), ma
    #    misura quanto il noScat4 è "fuori" rispetto alla sua immediate meno-
    #    complessa; dichiarato come tale nel report.
    ('tutto insieme (M=128+2b+L6 vs noScat4)', 'morletQ1Q3_L6_2b', 'noScat4',       1),
    # ── Scat4 added: SAME setup as noScat4 (M=256, 2 banks, 3 families, L_6)
    #    + Scattering 4th-order. Isolates whether Scat4 regularizes the marginals.
    ('Scat4 aggiunto (noScat4 vs final)',       'noScat4',          'final',         1),
    # ── GGD tail fix: same arm, last cut at q0.99 + bounded tail alpha
    ('tail fix (noL6)',       'morletQ1_noL6',      'tailfix_noL6',      1),
    ('tail fix (L6)',         'morletQ1_L6',        'tailfix_L6',        1),
    # ── M=256 tail-fix: against the same arm without the fix
    ('tail fix (noL6, M=256)', 'morletQ1_noL6_full', 'tailfix_noL6_full', 1),
    ('tail fix (L6,   M=256)', 'morletQ1_L6_full',   'tailfix_L6_full',   1),
    # ── tail-fix replacing Scat4: tailfix_noScat4 is the noScat4 setup + tail fix
    ('tail fix vs noScat4',   'noScat4',            'tailfix_noScat4',   1),
    ('tail fix vs final',     'final',              'tailfix_noScat4',   1),
    # ── full 2x2 closure {noScat4,final} x {no fix, fix}
    ('Scat4 w/ tail fix (tailfix_noScat4 vs tailfix_final)',
                              'tailfix_noScat4',    'tailfix_final',     1),
    ('tail fix w/ Scat4 (final vs tailfix_final)',
                              'final',              'tailfix_final',     1),
    # ── outer-region mass statistic, coarse two banks, against the production fit
    ('mass stat (2b, noL6)',  'morletQ1Q3_noL6_2b', 'mass_noL6_2b',      1),
    ('mass stat (2b, L6)',    'morletQ1Q3_L6_2b',   'mass_L6_2b',        1),
    ('tail_quantile only (2b, noL6)', 'morletQ1Q3_noL6_2b', 'tailq_noL6_2b', 1),
    ('mass alone (tailq vs mass, 2b, noL6)', 'tailq_noL6_2b', 'mass_noL6_2b', 1),
    ('tail_quantile only (2b, L6)', 'morletQ1Q3_L6_2b', 'tailq_L6_2b', 1),
    ('mass alone (tailq vs mass, 2b, L6)', 'tailq_L6_2b', 'mass_L6_2b', 1),
    # one bank: offline target is exact here (q.9999 1.03/1.05/0.97 on ch 0-2)
    ('mass stat (1b, L6)',    'morletQ1_L6',        'mass_L6_1b',        1),
    ('one vs two banks (mass, L6)', 'mass_L6_2b',   'mass_L6_1b',        1),
    ('drop narrow windows (2b, L6)', 'mass_L6_2b', 'mass_L6_2b_drop',   1),
    ('merge narrow windows (2b, L6)', 'mass_L6_2b', 'mass_L6_2b_merge', 1),
    ('merge vs drop (2b, L6)',  'mass_L6_2b_drop',  'mass_L6_2b_merge', 1),
]

# Pairs that also make sense on the Q=3 bank (presence in both arms).
PAIRS_Q3 = [
    ('ortogonalità Q=3 (M=128, L6)',   'morletQ1Q3_L6_2b', 'shannonQ1Q3_L6_2b', 3),
    ('L_6 Q=3 (2bank morlet)',         'morletQ1Q3_noL6_2b', 'morletQ1Q3_L6_2b', 3),
    ('L_6 Q=3 (2bank shannon)',        'shannonQ1Q3_noL6_2b','shannonQ1Q3_L6_2b', 3),
    ('tutto insieme Q=3 (M=128 vs noScat4)', 'morletQ1Q3_L6_2b', 'noScat4',      3),
    ('Scat4 aggiunto Q=3 (noScat4 vs final)',  'noScat4',          'final',       3),
    ('tail fix Q=3 vs noScat4', 'noScat4', 'tailfix_noScat4', 3),
    ('tail fix Q=3 vs final',   'final',   'tailfix_noScat4', 3),
    ('Scat4 w/ tail fix Q=3', 'tailfix_noScat4', 'tailfix_final', 3),
    ('tail fix w/ Scat4 Q=3', 'final',           'tailfix_final', 3),
    ('mass stat Q=3 (2b, noL6)', 'morletQ1Q3_noL6_2b', 'mass_noL6_2b', 3),
    ('mass stat Q=3 (2b, L6)',   'morletQ1Q3_L6_2b',   'mass_L6_2b',   3),
    ('tail_quantile only Q=3 (2b, noL6)', 'morletQ1Q3_noL6_2b', 'tailq_noL6_2b', 3),
    ('mass alone Q=3 (tailq vs mass, 2b, noL6)', 'tailq_noL6_2b', 'mass_noL6_2b', 3),
    ('tail_quantile only Q=3 (2b, L6)', 'morletQ1Q3_L6_2b', 'tailq_L6_2b', 3),
    ('mass alone Q=3 (tailq vs mass, 2b, L6)', 'tailq_L6_2b', 'mass_L6_2b', 3),
    ('drop narrow windows Q=3 (2b, L6)', 'mass_L6_2b', 'mass_L6_2b_drop', 3),
    ('merge narrow windows Q=3 (2b, L6)', 'mass_L6_2b', 'mass_L6_2b_merge', 3),
    ('merge vs drop Q=3 (2b, L6)', 'mass_L6_2b_drop', 'mass_L6_2b_merge', 3),
    # ── chosen fix with Scat4 (M=128) and at full resolution without Scat4 (M=256)
    ('Scat4 + mass vs Scat4 alone (coarse 2b)',  'coarse_noL2_scat4', 'mass_L6_2b_scat4', 1),
    ('Scat4 + mass vs Scat4 alone Q=3 (coarse 2b)', 'coarse_noL2_scat4', 'mass_L6_2b_scat4', 3),
    ('Scat4 on top of mass (coarse 2b)',  'mass_L6_2b', 'mass_L6_2b_scat4', 1),
    ('Scat4 on top of mass Q=3 (coarse 2b)', 'mass_L6_2b', 'mass_L6_2b_scat4', 3),
    ('mass at M=256 (noScat4 vs mass_noScat4)', 'noScat4', 'mass_noScat4', 1),
    ('mass at M=256 Q=3 (noScat4 vs mass_noScat4)', 'noScat4', 'mass_noScat4', 3),
    ('mass at M=256 vs Scat4 (final vs mass_noScat4)', 'final', 'mass_noScat4', 1),
    ('mass at M=256 vs Scat4 Q=3 (final vs mass_noScat4)', 'final', 'mass_noScat4', 3),
]

FINE_CH = {1: (0, 1, 2),          # Q=1: finest 3 bands
           3: (0, 1, 2, 3, 4, 5)}  # Q=3: finest 2 scales × 3 wavelets each


def summarise_pair(name, dfA, dfB, Q):
    """Return a text block comparing arms A and B on the fine channels."""
    fine = [ch for ch in FINE_CH[Q] if ch in dfA.index and ch in dfB.index]
    if not fine:
        return f'[{name}]  no fine channels in common\n'
    cols = ['q0.9999_ratio', 'q0.999_ratio', 'm2_ratio']
    sub_a = dfA.loc[fine, cols]
    sub_b = dfB.loc[fine, cols]
    delta = (sub_b - sub_a)
    lines = [f'[{name}]  Q={Q}  (fine channels {fine})']
    for col in cols:
        a_mean = sub_a[col].mean()
        b_mean = sub_b[col].mean()
        d_mean = delta[col].mean()
        lines.append(f'    {col:>14s}:  A={a_mean:+.3f}   B={b_mean:+.3f}   Δ(B-A)={d_mean:+.3f}')
    # per-channel detail
    joined = pd.concat({'A': sub_a, 'B': sub_b, 'Δ': delta}, axis=1)
    lines.append(joined.to_string())
    lines.append('')
    return '\n'.join(lines)


def one_arm(arm, Q, seed, x1_cache):
    """Load the sample for `arm`, rebuild its data at the arm's M, compute
    marginal_agreement on bank Q. Returns the DataFrame."""
    M = ARM_M[arm]
    J = ARM_J[arm]
    wav = WAV_FOR_ARM[arm]
    sample_path = find_sample(arm, seed)
    xt = torch.load(sample_path, map_location='cpu').float()
    if xt.dim() == 3:
        xt = xt.squeeze(1)
    # cache the data reconstruction per M (expensive)
    if M not in x1_cache:
        x1_cache[M] = rebuild_data_x1(subseries_len=512, target_len=M,
                                      n1=xt.shape[0])
    x1 = x1_cache[M]
    z_d = wavelet_coeffs(x1, wav, J, Q=Q, M=M)
    z_s = wavelet_coeffs(xt, wav, J, Q=Q, M=M)
    return marginal_agreement(z_d, z_s)


def main(seed, skip_missing):
    out_dir = HERE / 'figures' / 'data'
    if seed != 900:                         # 900 keeps the original location
        out_dir = out_dir / f'seed{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    all_arms = list(dict.fromkeys(
        ARMS_COARSE + ARMS_FULL + ARMS_2BANK + ARMS_NOSCAT4 + ARMS_FINAL))
    dfs = {}        # (arm, Q) -> DataFrame
    x1_cache = {}

    print(f'\n=== compute marginal_agreement for {len(all_arms)} arms, seed {seed} ===')
    for arm in all_arms:
        for Q in ARM_QS[arm]:
            key = (arm, Q)
            try:
                df = one_arm(arm, Q, seed, x1_cache)
            except FileNotFoundError as e:
                msg = f'  skip  {arm:28s} Q={Q}  ({e})'
                print(msg)
                if not skip_missing:
                    raise
                continue
            except Exception as e:  # noqa: BLE001
                print(f'  ERROR {arm:28s} Q={Q}  {type(e).__name__}: {e}')
                traceback.print_exc()
                continue
            dfs[key] = df
            csv = out_dir / f'{arm}_Q{Q}_agreement.csv'
            df.to_csv(csv, float_format='%.4f')
            # one-line summary on the finest channel
            fine_q = df.loc[0, 'q0.9999_ratio']
            fine_m2 = df.loc[0, 'm2_ratio']
            fine_w1 = df.loc[0, 'W1_over_std']
            print(f'  ok    {arm:28s} Q={Q}  '
                  f'ch=0: q0.9999={fine_q:.3f}  m2={fine_m2:.3f}  W1/σ={fine_w1:.3f}'
                  f'  → {csv.name}')

    # axis isolations
    lines = []
    lines.append('=== axis isolations: delta on fine channels ===')
    lines.append('convention: negative Δ(B-A) on q*_ratio = B has thinner tails than A')
    lines.append('            positive Δ(B-A) on m2_ratio moving toward/past 1 = closer/overshooting')
    lines.append('')
    for name, A, B, Q in PAIRS + PAIRS_Q3:
        if (A, Q) not in dfs or (B, Q) not in dfs:
            lines.append(f'[{name}]  missing arm(s): A={A} Q={Q} '
                         f'({"ok" if (A,Q) in dfs else "MISSING"})  '
                         f'B={B} Q={Q} ({"ok" if (B,Q) in dfs else "MISSING"})\n')
            continue
        lines.append(summarise_pair(name, dfs[(A, Q)], dfs[(B, Q)], Q))

    report = '\n'.join(lines)
    print('\n' + report)
    (out_dir / 'axis_isolations.txt').write_text(report)
    print(f'\nreport saved to: {out_dir / "axis_isolations.txt"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=900)
    ap.add_argument('--no-skip-missing', dest='skip_missing',
                    action='store_false', default=True,
                    help='raise instead of skipping arms whose sample is missing')
    a = ap.parse_args()
    main(a.seed, a.skip_missing)
