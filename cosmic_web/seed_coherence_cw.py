#!/usr/bin/env python3
"""Cross-seed coherence of log p on the 100 DATA maps, as a function of lam.

The data maps are the same for every seed, only Theta (hence log p) differs. Per seed the additive constant of log p is
unknown, so the comparison is a correlation: for each pair of seeds the Pearson correlation of C_F(x_i) over the maps
i, per family F and for the total; the table prints the mean over the 10 pairs. ~1 = the seeds rank the maps alike,
~0 = log p is noise (the std of the difference of two seeds is then sqrt(2) x the std over maps).

Reads results/logp/logp_<arm>.npz (lam of the xseed file, by default 0) and results/logp_lam<lam>/logp_<arm>.npz written by
    for L in 1e-6 1e-5 1e-4 1e-3 1e-2; do python cosmic_web/logp_cw.py --arms <arm> --lam $L \
        --theta_dir $SCRATCH/MGD-for-Maximum-Entropy-Generation/cosmic_web/theta_reg/<arm> --out cosmic_web/results/logp_lam$L; done
Usage:  python cosmic_web/seed_coherence_cw.py --arms base_real scat4_real       (CPU, seconds)
"""
import argparse
import re

import numpy as np

import cw_common as cw


def coherence(C):
    """(families + total,) mean over seed pairs of the Pearson correlation over maps; C = (seeds, families + total, maps)."""
    S = C.shape[0]
    return np.array([np.mean([np.corrcoef(C[a, i], C[b, i])[0, 1] for a in range(S) for b in range(a + 1, S)])
                     for i in range(C.shape[1])])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arms', nargs='+', default=['base_real', 'scat4_real'])
    a = ap.parse_args()
    for arm in a.arms:
        files = {}
        for f in [cw.RESULTS / 'logp' / f'logp_{arm}.npz'] + sorted(cw.RESULTS.glob(f'logp_lam*/logp_{arm}.npz')):
            if f.exists():
                files[float(np.load(f)['lam'])] = f
        print(f'\n=== {arm}: mean Pearson over seed pairs, log p of the {len(cw.SEEDS)}-seed Theta on the same data maps')
        fams = None
        for lam in sorted(files):
            D = np.load(files[lam])
            fams = [str(x) for x in D['families']] + ['TOTAL']
            if lam == min(files):
                print(f'{"lam":>9s} ' + ' '.join(f'{re.sub("Scattering_|_Order", "", f)[:16]:>16s}' for f in fams) + '   std(-log p) of generated / data')
            c = coherence(D['C_data'])
            print(f'{lam:9g} ' + ' '.join(f'{v:16.3f}' for v in c) +
                  f'   {D["C_gen"][:, -1].std(1, ddof=1).mean():8.0f} / {D["C_data"][:, -1].std(1, ddof=1).mean():8.0f}')


if __name__ == '__main__':
    main()
