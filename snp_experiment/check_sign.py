"""Sign report of the S&P Theta_reg (codes/resolve_theta_reg.py output, one file per (system, lam)), per family.

CODE sign: p ~ exp(Theta . phi) with Theta = -theta_MGD, so Theta < 0 is the confining sign for a statistic that is
a positive-definite function of the amplitude (as for L_6 in theta_sign_constraint/). These models have no L_6; the
candidates that confine at large amplitude are the positive statistics: scat2 (|W|^2, quadratic) and the real scat4
DIAGONAL (squares of the second-layer coefficients, quartic with the |W|^2 envelope). The other families (scat1,
scat3, the scat4 off-diagonal and imaginary parts, the scalar) are indefinite in sign or lower order: reported,
but a positive entry there is not by itself a normalisability problem. REPORT ONLY: nothing is constrained.

Per file and family: live entries (Theta != 0), number > 0, largest value, and the same for t >= 0.9 and t >= 0.99
(where the samples are formed). Columns are located by the model layout (--model M0..M3, J=6), in the order the
builder creates the potentials: scalar (the rest), scat1, scat3 Re, scat3 Im, scat2, scat4 Re (with diagonal), scat4 Im.

Usage (repo root on Jean Zay):
    python snp_experiment/check_sign.py --model M0 <theta_reg_dir>/*seed_900_*/lam*.pt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from potentials.utils_potentials import indices_fourth_order_Q  # noqa: E402


def layout(model, J, l6=False):
    """(name, size) blocks in the order of the terms; the scalar block (M1, M2) takes the rest."""
    s3 = J * (J + 3) // 2
    s4r = indices_fourth_order_Q(J, 1, 0, True).shape[1]
    s4i = indices_fourth_order_Q(J, 1, 1, True, include_lowpass=False).shape[1]
    # ORDER = the order get_1d_potentials (codes/potentials_builder.py) creates the potentials in, NOT the order of
    # --terms: scalar first, then scat1, scat3 Re, scat3 Im, scat2, scat4 Re, scat4 Im (checked by building them, 11-05).
    s1, s2 = ('scat1', J), ('scat2', J + 1)
    # scat3 Im = Scattering_Third_Order_Imag_nolp: without its J identically-zero low-pass columns
    s3r, s3i, s4re, s4im = ('scat3Re', s3), ('scat3Im', s3 - J), ('scat4Re', s4r), ('scat4Im', s4i)
    sca = ('scalar', None)
    blocks = {'M4': [s2],
              'M3': [s1, s2],
              'M2': [sca, s1, s2],
              'M0': [s1, s3r, s3i, s2, s4re, s4im],
              'M1': [sca, s1, s3r, s3i, s2, s4re, s4im]}[model]
    return ([('L6', J + 1)] if l6 else []) + blocks       # L_6 (L2p_norm(3), J+1 columns) is created first


def families(model, J, r, l6=False):
    """name -> column index tensor; scat4Re split into diagonal / off-diagonal."""
    out, start = {}, 0
    blocks = layout(model, J, l6)
    known = sum(size for _, size in blocks if size is not None)
    for name, size in blocks:
        size = r - known if size is None else size          # the scalar block = the rest (its size comes from the fit)
        cols = torch.arange(start, start + size)
        if name == 'scat4Re':
            idx = indices_fourth_order_Q(J, 1, 0, True)
            diag = idx[0] == idx[1]
            out['scat4Re_diag'] = cols[diag]
            out['scat4Re_offdiag'] = cols[~diag]
        else:
            out[name] = cols
        start += size
    assert start == r, f'layout covers {start} columns, Theta has {r}: wrong --model / --J?'
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+')
    p.add_argument('--model', required=True, choices=['M0', 'M1', 'M2', 'M3', 'M4'])
    p.add_argument('--J', type=int, default=6)
    p.add_argument('--l6', action='store_true', help='the model has L_6 (arms snp_M*_L6): columns 0:J+1; Theta_L6 > 0 is the violation')
    args = p.parse_args()
    confining = ('L6', 'scat2', 'scat4Re_diag') if args.l6 else ('scat2', 'scat4Re_diag')

    for f in sorted(args.files):
        d = torch.load(f, map_location='cpu', weights_only=False)
        Th, t = d['Theta_reg'].double(), d['t_reg'].double()
        fam = families(args.model, args.J, Th.shape[1], args.l6)
        print(f'\n{f}\n  lam {d["lam"]:g}, ridge {d.get("ridge")}, nodes {Th.shape[0]}, r {Th.shape[1]}  (code sign: < 0 confines)')
        print(f'  {"family":<16}{"cols":>5}{"live":>8}{"n>0":>7}{"max":>12}{"n>0 t>=.9":>11}{"n>0 t>=.99":>12}')
        viol = 0
        for name, cols in fam.items():
            B = Th[:, cols]
            pos, live = B > 0, B != 0
            late9, late99 = (t >= 0.9)[:, None], (t >= 0.99)[:, None]
            mark = '  <- confining candidate' if name in confining else ''
            print(f'  {name:<16}{len(cols):>5}{int(live.sum()):>8}{int(pos.sum()):>7}{float(B.max()):>12.3e}'
                  f'{int((pos & late9).sum()):>11}{int((pos & late99).sum()):>12}{mark}')
            if name in confining:
                viol += int(pos.sum())
        print('  verdict (' + ('L6, ' if args.l6 else '') + 'scat2 and scat4 real diagonal): ' +
              ('Theta < 0 everywhere' if viol == 0 else f'SIGN VIOLATED: {viol} positive entries'))


if __name__ == '__main__':
    main()
