"""Rule-based dynamical categories of turbulence signals, data vs model samples.

Rebuilt 2026-11-02 from the rule text stored in categories_seed904.json (the original script was lost).
Rule: fine energy = periodic lag-1 increments^2, local profile = 16-pt moving average;
active = profile > data q95; no active patch: quiet if fine_E < data median else rippled;
>=3 patches: multi-burst; else dominant event +-16 pts: step if |net|/total-variation > 0.4,
spike if <=2 direction changes (moves < 20% of the window range ignored), else packet.
Thresholds always come from the DATA, so data and model are labelled with the same cut.
NOT bit-identical to the lost seed-904 script: quiet/rippled/multi-burst identical, overall agreement
97.2% data / 97.1% model on mass_final seed 904 (see --check). Compare models only with labels made by THIS script.

Usage (repo root or anywhere inside it):
    python turbulence/sample_categories/categorize.py --samples <samples .pt> --tag <name>
writes turbulence/sample_categories/{categories,labels}_<tag>.{json,npz}, read by sample_categories.ipynb.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents)
            if (p / 'codes').is_dir() and (p / 'turbulence').is_dir())
for p in (ROOT, ROOT / 'codes', ROOT / 'data'):
    sys.path.insert(0, str(p))
local_data = ROOT / 'data/data_files/turbulence_1d_period.pt'
if local_data.exists():  # otherwise data_loader's default (JZ Lustre path) is used
    os.environ.setdefault('TURBULENCE_1D_DATA_PATH', str(local_data))

CATS = ['quiet', 'rippled', 'step', 'spike', 'packet', 'multi-burst']
WIN, HALF = 16, 16


def load_data(n1=10000, sub_len=512, target_len=256):
    """Data[:n1] exactly as in turbulence/run_SDE.py."""
    from codes.utils import normalize, split_periodize_reshape
    from codes.ortho_wavelet.ReadyToUseWavelets import DefineWavelet
    from data.data_loader import load_turbulence_1d
    dt = split_periodize_reshape(load_turbulence_1d(), sub_len)
    W = DefineWavelet('Db', m=3, device='cpu')
    for _ in range(int(np.log2(sub_len / target_len))):
        dt = W.decompose(dt)[1]
    return normalize(dt[:n1]).numpy()[:, 0], dt.shape[0]


def profile(X):
    e = (np.roll(X, -1, axis=1) - X) ** 2
    k = np.ones(WIN) / WIN
    # circular moving average, centred
    P = np.real(np.fft.ifft(np.fft.fft(e, axis=1) * np.fft.fft(np.roll(np.pad(k, (0, X.shape[1] - WIN)), -WIN // 2))[None], axis=1))
    return e, P


def patches(mask):
    """Contiguous True runs on a periodic axis -> list of index arrays."""
    n = len(mask)
    if mask.all():
        return [np.arange(n)]
    if not mask.any():
        return []
    start = int(np.argmin(mask))  # begin scanning at a False so wrap-around runs stay whole
    idx = (np.arange(n) + start) % n
    out, cur = [], []
    for i in idx:
        if mask[i]:
            cur.append(i)
        elif cur:
            out.append(np.array(cur)); cur = []
    if cur:
        out.append(np.array(cur))
    return out


def direction_changes(w, frac=0.2):
    """Reversals of the signal ignoring moves smaller than frac * (range of w).
    The raw sign-change count of the increments counts every small wiggle; frac=0.2 is the variant
    closest to the lost original (91% agreement on step/spike/packet of the seed-904 data labels)."""
    tol = frac * (w.max() - w.min())
    ch, ref, dirn = 0, w[0], 0
    for v in w[1:]:
        if dirn == 0:
            if v > ref + tol:
                dirn, ref = 1, v
            elif v < ref - tol:
                dirn, ref = -1, v
        elif dirn == 1:
            if v < ref - tol:
                ch, dirn, ref = ch + 1, -1, v
            else:
                ref = max(ref, v)
        else:
            if v > ref + tol:
                ch, dirn, ref = ch + 1, 1, v
            else:
                ref = min(ref, v)
    return ch


def label(X, thr, med):
    e, P = profile(X)
    fineE = e.mean(1)
    n = X.shape[1]
    lab, score = [], []
    for x, p, fe in zip(X, P, fineE):
        pt = patches(p > thr)
        if not pt:
            c = 'quiet' if fe < med else 'rippled'
            s = -fe if c == 'quiet' else fe
        elif len(pt) >= 3:
            c, s = 'multi-burst', len(pt) + p.max() / thr * 1e-3
        else:
            dom = max(pt, key=lambda q: p[q].sum())
            ctr = dom[np.argmax(p[dom])]
            w = x[(np.arange(ctr - HALF, ctr + HALF + 1)) % n]
            d = np.diff(w)
            ratio = abs(w[-1] - w[0]) / np.abs(d).sum()
            nchg = direction_changes(w)
            if ratio > 0.4:
                c, s = 'step', ratio
            elif nchg <= 2:
                c, s = 'spike', p.max()
            else:
                c, s = 'packet', p.max()
        lab.append(c); score.append(s)
    return np.array(lab), np.array(score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True, help='model samples .pt (path absolute or relative to repo root)')
    ap.add_argument('--tag', required=True, help='output suffix, e.g. bicw_allmass_scat4_seed904')
    ap.add_argument('--n_show', type=int, default=8)
    ap.add_argument('--check', help='existing labels npz to compare against (reconstruction test)')
    a = ap.parse_args()

    data, n_traj = load_data()
    spath = Path(a.samples)
    model = torch.load(spath if spath.is_absolute() else ROOT / spath, map_location='cpu').numpy()[:, 0]
    e, P = profile(data)
    thr, med = np.quantile(P, 0.95), np.median(e.mean(1))
    ld, sd = label(data, thr, med)
    lm, sm = label(model, thr, med)

    rng = np.random.default_rng(0)
    clearest, rand, summary = {}, {}, {}
    for c in CATS:
        clearest[c], rand[c] = {}, {}
        for key, L, S in (('data', ld, sd), ('model', lm, sm)):
            ii = np.where(L == c)[0]
            clearest[c][key] = [int(i) for i in ii[np.argsort(-S[ii])][:a.n_show]]
            rand[c][key] = [int(i) for i in rng.permutation(ii)[:a.n_show]]
        summary[c] = {'data': int((ld == c).sum()), 'model': int((lm == c).sum())}
        print(f"{c:>12}  data {summary[c]['data']:5d}  model {summary[c]['model']:5d}")

    rel = spath.resolve().relative_to(ROOT.resolve()) if spath.is_absolute() else spath
    here = ROOT / 'turbulence/sample_categories'
    json.dump({'model_samples': str(rel), 'data': 'Data[:10000] as in turbulence/run_SDE.py (subseries_len 512, target_len 256, normalize)',
               'rule': __doc__.split('Rule: ')[1].split('\n\nUsage')[0].replace('\n', ' '),
               'thresholds': {'profile_q95': float(thr), 'fineE_median': float(med)},
               'categories': CATS, 'summary': summary, 'clearest': clearest, 'random': rand},
              open(here / f'categories_{a.tag}.json', 'w'), indent=1)
    data_traj = np.arange(len(data)) // 3  # 3 windows of 256 per trajectory
    np.savez(here / f'labels_{a.tag}.npz', data=ld, model=lm, data_traj=data_traj)
    print('wrote', here / f'categories_{a.tag}.json', here / f'labels_{a.tag}.npz')

    if a.check:
        old = np.load(a.check)
        for key, L in (('data', ld), ('model', lm)):
            print(f'{key}: agreement with {a.check} = {(old[key] == L).mean():.4f}')


if __name__ == '__main__':
    main()
