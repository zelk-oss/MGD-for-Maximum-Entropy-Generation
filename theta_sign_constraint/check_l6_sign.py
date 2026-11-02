"""Sign of the L_6 coefficients in an UNCONSTRAINED offline Theta_reg solve (codes/resolve_theta_reg.py
output, one file per (system, lam)). Question: is L_6 < 0 already, so the bounded solve is unnecessary?

Per file: for each L_6 column (default 0:9 = L_6 first, J=8), the number of nodes with Theta > 0, the
largest Theta, and the earliest t at which it is positive; then the same counts restricted to t >= 0.9
and t >= 0.99 (the end of the trajectory, where the samples are formed). Entries exactly 0 are masked
(dead potential at that node) and counted separately, not as positive.

Usage (repo root on Jean Zay):
    python theta_sign_constraint/check_l6_sign.py turbulence/saved_results/theta_reg_lamsweep/*bicw_rich*/lam3e-05_*.pt
    options: --cols 0:9
"""
import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+')
    p.add_argument('--cols', default='0:9')
    args = p.parse_args()
    a, b = (int(x) for x in args.cols.split(':'))

    for f in sorted(args.files):
        d = torch.load(f, map_location='cpu', weights_only=False)
        Th = d['Theta_reg'].double()                 # (nodes, r)
        t = d['t_reg'].double()
        L = Th[:, a:b]
        print(f'\n{f}\n  lam {d["lam"]:g}, ridge {d.get("ridge")}, nodes {Th.shape[0]}, r {Th.shape[1]}, '
              f'L_6 columns {a}:{b}')
        print(f'  {"col":>4} {"n>0":>7} {"n=0":>7} {"max":>11} {"first t>0":>10} {"n>0,t>=.9":>10} {"n>0,t>=.99":>11}')
        for i in range(L.shape[1]):
            c = L[:, i]
            pos = c > 0
            first = f'{float(t[pos].min()):.4f}' if bool(pos.any()) else '-'
            print(f'  {a + i:>4} {int(pos.sum()):>7} {int((c == 0).sum()):>7} {float(c.max()):>11.3e} {first:>10} '
                  f'{int((pos & (t >= 0.9)).sum()):>10} {int((pos & (t >= 0.99)).sum()):>11}')
        pos = L > 0
        live = L != 0
        print(f'  ALL: {int(pos.sum())} positive of {int(live.sum())} live entries '
              f'({100 * float(pos.sum()) / max(int(live.sum()), 1):.2f} %); '
              f'verdict: {"L_6 < 0 everywhere" if not bool(pos.any()) else "SIGN VIOLATED somewhere"}')


if __name__ == '__main__':
    main()
