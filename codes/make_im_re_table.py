#!/usr/bin/env python3
"""Table of the imaginary / real ratio of the Var/2 of the order-3 and order-4 statistics, Lagrangian and Eulerian
(log p evaluated on the GENERATED signals, 5 seeds each; the same inputs as make_kl_tables.py --source generated).

Per seed: Var over the signals of the -log10 p / d contribution of the imaginary category over that of the real one
(the Var/2 factors cancel); mean ± std over the seeds. Orders 3 and 4 are kept separate.
It reuses the loaders of make_kl_tables.py (lagrangian / eulerian) by intercepting their kl_table call; nothing of
that script changes.

Usage (repo root, on Jean Zay in the pytorch-gpu environment, CPU only, seconds):
    python codes/make_im_re_table.py
Output: jets/figures/talk/im_re_ratio_generated.{png,csv} (--outdir DIR: another folder).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'codes'))
import make_kl_tables as mk                                                # noqa: E402  (puts the project folders on sys.path)
import talk_figures as tf                                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label', default='turbulence_nested_M3_bicw_allmass_scat4')
    ap.add_argument('--seeds', type=int, nargs='+', default=[900, 901, 902, 903, 904])
    ap.add_argument('--lam', type=float, default=3e-5)
    ap.add_argument('--jets_arm', default='jets_full')
    ap.add_argument('--outdir', type=Path, default=ROOT / 'jets' / 'figures' / 'talk')
    a = ap.parse_args()

    captured = {}
    orig = tf.kl_table
    def spy(C, widths, d, name=None):                                      # keep the inputs; no csv of the Var/2 table here
        captured[len(captured)] = (C, widths, d)
        return orig(C, widths, d, None)
    tf.kl_table = spy
    try:
        mk.lagrangian(a, 'generated'); mk.eulerian(a, 'generated')
    finally:
        tf.kl_table = orig
    ratios = {}
    for (title, _), (C, widths, d) in zip((('Lagrangian', 0), ('Eulerian', 1)), captured.values()):
        ratios[title] = tf.im_re_ratio(C, widths, d)
        print(title, {k: None if v is None else tuple(round(x, 4) if isinstance(x, float) else x for x in v)
                      for k, v in ratios[title].items()})
    tf.set_output(a.outdir)
    tf.plot_im_re_table(ratios, 'im_re_ratio_generated')


if __name__ == '__main__':
    main()
