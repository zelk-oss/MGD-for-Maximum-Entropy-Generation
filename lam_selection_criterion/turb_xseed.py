"""Cross-seed choice of lam for the time-regularised Theta of a turbulence arm (2026-10-05).

The seeds of an arm share data, potentials and settings; only the random seed differs, so their
noises are independent (user, 2026-10-05). For every ordered pair of seeds (A, B), Theta_A(lam)
(seed A's system solved with lam, by codes/resolve_theta_reg.py -- phase 1) is scored with the
data term of seed B:

    L_AB(lam) = sum_k w_k [ 1/2 Theta_A,k^T (M_B,k + ridge diag M_B,k) Theta_A,k - b_B,k^T Theta_A,k ]

(w_k trapezoid weights in t; entries dead in B at node k are set to 0, as the solver pins them).
B's noise is independent of Theta_A, so up to a lam-independent constant the expectation of L_AB
is the error of Theta_A(lam) in seed B's data metric: no assumption on how the noise of one run is
correlated in time. The losses are pooled over all ordered pairs and lam is the minimiser, on the
whole transport ('whole', what log Z / the entropy integrate) and on t >= e for each --ends value
('end<e>', what log p at t = 1 uses). Same rule as add_xseed_rules in bimodal_lamtest.py, where it
is tested against the exact truth.

Phase 2 only (this script): reads the systems (memory-mapped; only t, M, b, live, meta are touched)
and the phase-1 Theta files, writes <out>/xseed_<arm>.pt and prints the table.

Usage (Jean Zay, after phase 1; launch_turb_xseed.sh does both phases):
  python lam_selection_criterion/turb_xseed.py --systems S900.pt S901.pt ... --sweep_dir DIR \
      --lams 0 1e-6 ... --ridge 1e-4 --arm noSm30_reg1e-4 --out DIR
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'codes', ROOT / 'data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from codes.sde_routines import trapezoid_node_weights      # noqa: E402
from codes.resolve_theta_reg import out_path                # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--systems', type=Path, nargs='+', required=True, help='saved systems, one per seed (>= 2)')
    p.add_argument('--sweep_dir', type=Path, required=True, help='phase-1 output root (resolve_theta_reg --outdir)')
    p.add_argument('--lams', type=float, nargs='+', required=True)
    p.add_argument('--ridge', type=float, required=True, help="the arm's per-step ridge (checked against the meta)")
    p.add_argument('--schedule', default='uniform', choices=['uniform', 'guth'])
    p.add_argument('--ends', type=float, nargs='*', default=[0.99, 0.999])
    p.add_argument('--arm', required=True, help='name used in the output file')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--chunk', type=int, default=256, help='nodes per block of the data-term sum')
    p.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or None)
    return p.parse_args()


def config_of(path):
    return path.name[:-len('.pt')] if path.name.endswith('.pt') else path.name


def load_system(path):
    """t (all nodes), M, b (lists), live (n, r) or None, meta -- memory-mapped, Sigma never read."""
    s = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if s.get('meta', {}).get('reg_mode') != 'moment':
        raise ValueError(f'{path.name}: not a moment-mode system')
    return s


def interp_rows(t_src, Th, t_dst):
    """Linear interpolation in t along axis -2 of Th (..., n_src, r); constant outside."""
    t_src, t_dst = torch.as_tensor(t_src), torch.as_tensor(t_dst)
    j = torch.searchsorted(t_src, t_dst).clamp(1, len(t_src) - 1)
    w = ((t_dst - t_src[j - 1]) / (t_src[j] - t_src[j - 1])).clamp(0, 1)[:, None].to(Th.dtype)
    return (1 - w) * Th[..., j - 1, :] + w * Th[..., j, :]


def data_terms(sysB, Th, ridge, windows, chunk):
    """Th (C, n-1, r): curves on B's nodes 1..n-1 (the phase-1 files drop node 0, as the solver
    output). Returns {window: (C,) loss}."""
    t = sysB['t'].double().numpy()[1:]
    w = torch.as_tensor(trapezoid_node_weights(t))
    live = sysB['live'][1:].bool() if 'live' in sysB else None
    n = len(t)
    q = torch.zeros(Th.shape[0], n, dtype=torch.float64)
    for a in range(0, n, chunk):
        ks = range(a + 1, min(a + chunk, n) + 1)                       # system rows (node 0 dropped)
        M = torch.stack([sysB['M'][k] for k in ks]).double()
        M = (M + M.transpose(1, 2)) / 2
        if ridge:
            M = M + ridge * torch.diag_embed(torch.diagonal(M, dim1=1, dim2=2))
        b = torch.stack([sysB['b'][k].reshape(-1) for k in ks]).double()
        X = Th[:, a:a + len(ks)]
        if live is not None:
            X = X * live[a:a + len(ks)].to(X.dtype)
        q[:, a:a + len(ks)] = 0.5 * torch.einsum('cki,kij,ckj->ck', X, M, X) - torch.einsum('cki,ki->ck', X, b)
    return {name: (q[:, m] * w[m]).sum(1) for name, m in
            ((name, torch.as_tensor(sel(t))) for name, sel in windows.items())}


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if len(args.systems) < 2:
        sys.exit('need at least 2 seeds')
    lams = sorted(set(float(x) for x in args.lams))
    windows = {'whole': lambda t: np.ones(len(t), bool),
               **{f'end{e:g}': (lambda e: (lambda t: t >= e))(e) for e in args.ends}}
    t0 = time.time()

    # Theta_A(lam) of every seed, from phase 1 (Theta_reg = Theta[1:], t_reg = t[1:])
    seeds, Theta, t_ref = [], {}, None
    for path in args.systems:
        s = load_system(path)
        meta = s.get('meta', {})
        reg = meta.get('regularization')
        if reg is not None and not np.isclose(reg, args.ridge):
            sys.exit(f'{path.name}: run ridge {reg:g} != --ridge {args.ridge:g} (the arms differ in ridge)')
        mask = 'system' if 'live' in s else 'none'
        cfg = config_of(path)
        curves = []
        for lam in lams:
            f = out_path(args.sweep_dir, cfg, lam, args.ridge, 'moment', args.schedule, mask)
            if not f.exists():
                sys.exit(f'missing phase-1 file {f}')
            d = torch.load(f, map_location='cpu', weights_only=False)
            curves.append((d['t_reg'].double().numpy(), d['Theta_reg'].double()))
        t_reg = curves[0][0]
        if t_ref is None:
            t_ref = t_reg
        seeds.append(dict(path=path, config=cfg, t=t_reg, mask=mask))
        Theta[cfg] = curves
        del s
    same_grid = all(len(sd['t']) == len(t_ref) and np.abs(sd['t'] - t_ref).max() < 1e-12 for sd in seeds)
    print(f'{len(seeds)} seeds, {len(lams)} lams, nodes {len(t_ref)}, same time grid: {same_grid} '
          f'({time.time() - t0:.0f} s to read the Theta files)', flush=True)

    # L[A, B, lam, window]
    S, L = len(seeds), len(lams)
    loss = {w: np.full((S, S, L), np.nan) for w in windows}
    for b_i, sb in enumerate(seeds):
        t1 = time.time()
        sysB = load_system(sb['path'])
        tB = sysB['t'].double().numpy()[1:]
        others = [a_i for a_i in range(S) if a_i != b_i]
        Th = []
        for a_i in others:
            for t_a, th in Theta[seeds[a_i]['config']]:
                Th.append(th if same_grid else interp_rows(t_a, th, tB))
        Th = torch.stack(Th)                                           # (len(others) * L, n-1, r)
        res = data_terms(sysB, Th, args.ridge, windows, args.chunk)
        for w in windows:
            loss[w][others, b_i, :] = res[w].numpy().reshape(len(others), L)
        del sysB, Th
        print(f'  scored with seed {sb["config"][-40:]} ({time.time() - t1:.0f} s)', flush=True)

    out = dict(arm=args.arm, lams=lams, seeds=[sd['config'] for sd in seeds], ridge=args.ridge,
               schedule=args.schedule, same_grid=same_grid, loss=loss, selected={}, pair_selected={},
               theta_files={})
    print(f'\n######## {args.arm}: cross-seed CV over {S} seeds ({S * (S - 1)} ordered pairs), '
          f'{args.schedule} weights, ridge {args.ridge:g}')
    for w in windows:
        pooled = np.nansum(loss[w], axis=(0, 1))                       # (L,)
        i = int(np.argmin(pooled))
        sel = lams[i]
        pairs = [lams[int(np.argmin(loss[w][a, b]))] for a in range(S) for b in range(S) if a != b]
        out['selected'][w], out['pair_selected'][w] = sel, pairs
        edge = sel in (min(l_ for l_ in lams if l_ > 0), max(lams)) if sel > 0 else False
        counts = {l_: pairs.count(l_) for l_ in sorted(set(pairs))}
        print(f"\n  {w:>9}: selected lam = {sel:g}{'  (GRID EDGE: extend the grid)' if edge else ''}"
              f"{'  (lam = 0: no smoothing preferred)' if sel == 0 else ''}")
        print(f"             per pair: " + ', '.join(f'{l_:g} x{c}' for l_, c in counts.items()))
        print('             pooled loss - min:  ' + '  '.join(f'{l_:g}: {v - pooled[i]:.3g}' for l_, v in zip(lams, pooled)))
        out['theta_files'][w] = {sd['config']: str(out_path(args.sweep_dir, sd['config'], sel, args.ridge, 'moment',
                                                          args.schedule, sd['mask'])) for sd in seeds}
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f'xseed_{args.arm}.pt'
    torch.save(out, dest)
    print(f'\n-> {dest}  (Theta of the selected lam per seed: out["theta_files"][window]); total {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
