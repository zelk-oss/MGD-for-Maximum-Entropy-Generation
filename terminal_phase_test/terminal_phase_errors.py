"""Terminal phase test, per-statistic error BEFORE and AFTER the terminal phase of the SAME run (paired control).

Reads only saved_results/aux_moments/<cfg>_aux_moments.pt and <cfg>_terminal.pt (no samples, no potentials, no data).
aux rows: row j = state after step j (loop index k = j; checked 11-17 on the Gaussian run: rel of statistic 0 / z of
terminal.pt is the same constant, 1.270e-3, on rows 9999..10005 and on the last row). The planned grid is steps 0..n_planned-1,
n_planned = terminal_start, so
  row n_planned - 1 = the run WITHOUT the terminal phase (end of the planned grid, same walkers, same noise),
  last row          = after the terminal phase (shrunk steps + hold).
Per statistic n:  rel_n = |p_n - e_n| / |e_n|  (p = barphi_p walkers, e = barphi_e target of that row).
Prints: counts above 1 % / 0.1 % / 0.01 %, median / p90 / max, the worst statistics before (and what they became),
and the terminal.pt summary (max z in SE units). Writes <out>/<cfg>_before_after.csv (all statistics).

Usage (repo root, CPU only):
    python terminal_phase_test/terminal_phase_errors.py --label turbulence_nested_M3_bicw_allmass_scat4_pf_term \
        --config _sigma3.5_ _nt27500_ --out terminal_phase_test/errors
"""
import argparse
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
THRESH = (1e-2, 1e-3, 1e-4)


def find_cfg(res, label, seed, config):
    cands = sorted((res / 'aux_moments').glob(f'*_seed_{seed}_*{label}_2*_aux_moments.pt'),
                   key=lambda f: f.stat().st_mtime)
    if config:
        cands = [c for c in cands if all(s in c.name for s in config)]
    cands = [c for c in cands if c.name.replace('_aux_moments.pt', '_terminal.pt') in
             {t.name for t in (res / 'aux_moments').glob('*_terminal.pt')}]
    if not cands:
        raise FileNotFoundError(f'no run of {label} seed {seed} config {config} with aux_moments + terminal.pt under {res}')
    return cands[-1].name.replace('_aux_moments.pt', '')


def rel_row(BE, BP, j):
    e, p = BE[j], BP[j]
    return np.abs(p - e) / np.abs(e).clip(1e-300), e, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--seed', type=int, default=900)
    ap.add_argument('--config', nargs='+', default=None)
    ap.add_argument('--root', type=Path, default=ROOT / 'turbulence', help='folder holding saved_results/')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--top', type=int, default=15)
    a = ap.parse_args()
    res = a.root / 'saved_results'
    cfg = find_cfg(res, a.label, a.seed, a.config)
    aux = torch.load(res / 'aux_moments' / f'{cfg}_aux_moments.pt', map_location='cpu')
    term = torch.load(res / 'aux_moments' / f'{cfg}_terminal.pt', map_location='cpu', weights_only=False)
    BE, BP = aux['barphi_e'].double().numpy(), aux['barphi_p'].double().numpy()
    k0 = int(term['terminal_start'])
    n_term = int(term['log'].shape[0])
    names = list(term['labels'])
    print(f'=== {cfg}\n  aux rows {len(BE)}, statistics {BE.shape[1]}, terminal_start (= planned steps) {k0}, '
          f'terminal steps logged {n_term}; last row index {len(BE) - 1} (expected k0 + n_term - 1 = {k0 + n_term - 1})')
    if len(BE) != k0 + n_term:
        raise SystemExit('aux rows != terminal_start + terminal steps: row alignment unverified, stop')
    if len(names) != BE.shape[1]:
        raise SystemExit(f'{len(names)} labels vs {BE.shape[1]} statistics')
    info = term['info']
    print(f"  terminal phase: z at end of planned grid {info['z_end_planned']:.3g} (stat {info['idx_end_planned']}), "
          f"h {info['h_end_planned']:.2e} -> {float(term['log'][-1, 1]):.2e}, max z after the last step "
          f"{float(term['log'][-1, 2]):.3g}, reason: {info['reason']}; params {term['params']}")

    rel0, e0, p0 = rel_row(BE, BP, k0 - 1)
    rel1, e1, p1 = rel_row(BE, BP, len(BE) - 1)
    i0 = int(info['idx_end_planned'])
    c0, c1 = rel0[i0] / info['z_end_planned'], rel1[i0] / float(term['log'][-1, 2]) if int(term['log'][-1, 3]) == i0 else float('nan')
    print(f'  row {k0 - 1} (planned end): sanity: statistic {i0} = {names[i0]}: rel {rel0[i0]:.3e}, z {info["z_end_planned"]:.3g} '
          f'-> rel/z {c0:.3e} (= SE/|e| of that statistic; at the last row the same ratio {c1:.3e} if the worst stat is the same)')

    print(f'\n  stats with rel above threshold: before (row {k0 - 1}, planned end) -> after (last row)')
    for th in THRESH:
        print(f'    > {th:g}: {int((rel0 > th).sum()):4d} -> {int((rel1 > th).sum()):4d}   of {len(rel0)}')
    for tag, r in (('before', rel0), ('after ', rel1)):
        print(f'  {tag}: median {np.median(r):.2e}  p90 {np.quantile(r, .9):.2e}  p99 {np.quantile(r, .99):.2e}  max {r.max():.2e}')

    order = np.argsort(-rel0)[:a.top]
    print(f'\n  worst {a.top} before the terminal phase:')
    print(f"  {'idx':>4} {'statistic':<46}{'rel before':>12}{'rel after':>12}{'factor':>9}{'p/e before':>12}{'p/e after':>11}")
    for i in order:
        f = rel0[i] / rel1[i] if rel1[i] > 0 else np.inf
        print(f'  {i:4d} {names[i]:<46}{rel0[i]:12.3e}{rel1[i]:12.3e}{f:9.1f}{p0[i] / e0[i]:12.4f}{p1[i] / e1[i]:11.4f}')
    worse = np.argsort(-(rel1 - rel0))[:5]
    print('\n  5 statistics whose error GREW most (after - before), to see side effects:')
    for i in worse:
        print(f'  {i:4d} {names[i]:<46}{rel0[i]:12.3e} -> {rel1[i]:.3e}')
    after_top = np.argsort(-rel1)[:5]
    print('\n  5 worst AFTER the terminal phase:')
    for i in after_top:
        print(f'  {i:4d} {names[i]:<46}{rel0[i]:12.3e} -> {rel1[i]:.3e}')

    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        f = a.out / f'{cfg}_before_after.csv'
        with open(f, 'w') as fh:
            fh.write(f'# {cfg}  rows: before = {k0 - 1} (planned end), after = {len(BE) - 1} (last)\n')
            fh.write('index,name,target_before,generated_before,rel_before,target_after,generated_after,rel_after\n')
            for i, nm in enumerate(names):
                fh.write(f'{i},{nm},{e0[i]:.6e},{p0[i]:.6e},{rel0[i]:.6e},{e1[i]:.6e},{p1[i]:.6e},{rel1[i]:.6e}\n')
        print(f'  wrote {f}')


if __name__ == '__main__':
    main()
