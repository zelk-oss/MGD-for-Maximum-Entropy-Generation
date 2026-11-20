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

--sigma_rule (rule B, added 2026-11-18; rule A above is unchanged): cross-seed CV in the walker-covariance metric,
on t >= e for each --ends value only:

    L_AB(lam) = sum_{k: t_k >= e} w_k [ 1/2 Theta_A,k^T Sigma_A,k Theta_A,k - Theta_A,k^T Sigma_A,k theta_B,k ]

Sigma_A,k = Cov_w(phi) of the SCORED seed A (system key 'Sigma', node k; memory-mapped, only the nodes of the
window are read); theta_B = seed B's lam = 0 Theta file (per-step solve with the run's ridge); statistics dead
at node k in A or in B are set to 0 in both curves. Sigma_A, not Sigma_B: Sigma_B and theta_B come from the
same walkers and would correlate in the cross term (user's specification). Up to a lam-independent term
L_AB = 1/2 |Theta_A(lam) - theta_B|^2 in the Sigma_A metric.
Also, as a check of the code, the same loss with seed B's ridged M as the metric (masked with B's live set
only): since M_B theta_B = b_B on the live entries, it must equal rule A's loss to the solve's rounding
(max relative difference printed; the run stops if the argmins differ).
Both rules: per-pair argmins and a leave-one-seed-out jackknife of the pooled argmin (S values). Output
<out>/xseed_<arm>_sigma.pt (the rule-A file xseed_<arm>.pt is not written in this mode).
A pair loss that is NaN stops the run (both modes; it used to be dropped by nansum).

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
    p.add_argument('--tag', default='',
                   help='(2026-11-19) suffix of the phase-1 Theta_reg files after _moment_mask: "" = full C (as before), '
                        '_shr1 = diagonal C, _shrG = shrunk C, _shr1_pinl6 = diagonal + L_6 pinned (resolve_theta_reg.py '
                        '--c_shrink / --pin_l6). Also appended to the output name: xseed_<arm><tag>.pt')
    p.add_argument('--sigma_rule', action='store_true',
                   help='also rule B (Sigma_A metric, on the --ends windows), the M_B check and the jackknife; '
                        'writes xseed_<arm>_sigma.pt instead of xseed_<arm>.pt')
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


def data_terms(sysB, Th, ridge, windows, chunk, T0B=None):
    """Th (C, n-1, r): curves on B's nodes 1..n-1 (the phase-1 files drop node 0, as the solver
    output). Returns {window: (C,) loss}.
    T0B (n-1, r), optional (--sigma_rule check): seed B's lam = 0 curve; then also returns
    {window: (C,) loss} of 1/2 X^T M_B X - X^T M_B T0B (same mask, same ridged M), which must equal the
    rule-A loss to rounding because M_B T0B = b_B on the live entries."""
    t = sysB['t'].double().numpy()[1:]
    w = torch.as_tensor(trapezoid_node_weights(t))
    live = sysB['live'][1:].bool() if 'live' in sysB else None
    n = len(t)
    q = torch.zeros(Th.shape[0], n, dtype=torch.float64)
    qc = torch.zeros(Th.shape[0], n, dtype=torch.float64) if T0B is not None else None
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
        MX = torch.einsum('kij,ckj->cki', M, X)
        q[:, a:a + len(ks)] = 0.5 * (X * MX).sum(-1) - torch.einsum('cki,ki->ck', X, b)
        if qc is not None:
            Y = T0B[a:a + len(ks)]
            if live is not None:
                Y = Y * live[a:a + len(ks)].to(Y.dtype)
            qc[:, a:a + len(ks)] = ((0.5 * X - Y[None]) * MX).sum(-1)    # X^T M Y = (M X) . Y, M symmetric
    masks = {name: torch.as_tensor(sel(t)) for name, sel in windows.items()}
    out = {name: (q[:, m] * w[m]).sum(1) for name, m in masks.items()}
    if qc is None:
        return out
    return out, {name: (qc[:, m] * w[m]).sum(1) for name, m in masks.items()}


def sigma_terms(sysA, ThA, others, ends, chunk):
    """Rule B for the scored seed A. ThA (L, n-1, r): A's curves; others {b_i: (T0, liveB)}: seed B's
    lam = 0 curve (n-1, r) and live mask (n-1, r) bool or None, both on A's nodes. Statistics dead in A or
    in B at a node are set to 0 in both curves. Only the nodes with t >= min(ends) are read from A's Sigma.
    Returns {b_i: {end: (L,) loss}}."""
    t = sysA['t'].double().numpy()[1:]
    w = torch.as_tensor(trapezoid_node_weights(t))
    liveA = sysA['live'][1:].bool() if 'live' in sysA else None
    nodes = np.flatnonzero(t >= min(ends))
    L = ThA.shape[0]
    q = {b_i: torch.zeros(L, len(nodes), dtype=torch.float64) for b_i in others}
    for a in range(0, len(nodes), chunk):
        nd = nodes[a:a + chunk]
        ndt = torch.as_tensor(nd)
        S = torch.stack([sysA['Sigma'][int(k) + 1] for k in nd]).double()   # system rows (node 0 dropped)
        S = (S + S.transpose(1, 2)) / 2
        X0 = ThA[:, ndt]
        for b_i, (T0, liveB) in others.items():
            live = torch.ones(X0.shape[1:], dtype=torch.bool)
            if liveA is not None:
                live &= liveA[ndt]
            if liveB is not None:
                live &= liveB[ndt]
            mask = live.to(X0.dtype)
            X, Y = X0 * mask, T0[ndt] * mask
            SX = torch.einsum('kij,lkj->lki', S, X)
            q[b_i][:, a:a + len(nd)] = ((0.5 * X - Y[None]) * SX).sum(-1)   # X^T S Y = (S X) . Y, S symmetric
    wn = w[torch.as_tensor(nodes)]
    tn = t[nodes]
    return {b_i: {e: (qq[:, torch.as_tensor(tn >= e)] * wn[torch.as_tensor(tn >= e)]).sum(1) for e in ends}
            for b_i, qq in q.items()}


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
            f = out_path(args.sweep_dir, cfg, lam, args.ridge, 'moment', args.schedule, mask, args.tag)
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
    if args.sigma_rule and 0.0 not in lams:
        sys.exit('--sigma_rule needs lam = 0 in --lams (theta_B is the lam = 0 solve)')
    i0 = lams.index(0.0) if 0.0 in lams else None
    end_names = {e: f'end{e:g}' for e in args.ends}
    loss = {w: np.full((S, S, L), np.nan) for w in windows}
    loss_chk = {end_names[e]: np.full((S, S, L), np.nan) for e in args.ends} if args.sigma_rule else None
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
        if args.sigma_rule:
            t_b, T0B = Theta[sb['config']][i0]
            res, chk = data_terms(sysB, Th, args.ridge, windows, args.chunk,
                                  T0B=T0B if same_grid else interp_rows(t_b, T0B, tB))
            for e, wn in end_names.items():
                loss_chk[wn][others, b_i, :] = chk[wn].numpy().reshape(len(others), L)
        else:
            res = data_terms(sysB, Th, args.ridge, windows, args.chunk)
        for w in windows:
            loss[w][others, b_i, :] = res[w].numpy().reshape(len(others), L)
        del sysB, Th
        print(f'  rule A: scored with seed {sb["config"][-40:]} ({time.time() - t1:.0f} s)', flush=True)

    loss_sig = None
    if args.sigma_rule:
        loss_sig = {end_names[e]: np.full((S, S, L), np.nan) for e in args.ends}
        lives = {}
        for b_i, sb in enumerate(seeds):                               # live masks only (small), on B's nodes
            s_ = load_system(sb['path'])
            lives[b_i] = s_['live'][1:].bool().clone() if 'live' in s_ else None
            del s_
        for a_i, sa in enumerate(seeds):
            t1 = time.time()
            sysA = load_system(sa['path'])
            if 'Sigma' not in sysA:
                sys.exit(f'{sa["path"].name}: no Sigma in the system (needed by --sigma_rule)')
            tA = sysA['t'].double().numpy()[1:]
            ThA = torch.stack([th for _, th in Theta[sa['config']]])   # (L, n-1, r), A's own grid
            others = {}
            for b_i, sb in enumerate(seeds):
                if b_i == a_i:
                    continue
                t_b, T0 = Theta[sb['config']][i0]
                lv = lives[b_i]
                if not same_grid:                                      # B's curve and mask onto A's nodes
                    T0 = interp_rows(t_b, T0, tA)
                    j = np.clip(np.searchsorted(t_b, tA), 0, len(t_b) - 1)
                    lv = None if lv is None else lv[torch.as_tensor(j)]
                others[b_i] = (T0, lv)
            res = sigma_terms(sysA, ThA, others, args.ends, args.chunk)
            for b_i, d in res.items():
                for e, wn in end_names.items():
                    loss_sig[wn][a_i, b_i, :] = d[e].numpy()
            del sysA, ThA
            print(f'  rule B: scored seed {sa["config"][-40:]} in its Sigma metric ({time.time() - t1:.0f} s)', flush=True)

    def check_finite(name, arr):
        off = ~np.eye(S, dtype=bool)
        bad = [(a, b) for a in range(S) for b in range(S) if off[a, b] and not np.isfinite(arr[a, b]).all()]
        if bad:
            sys.exit(f'{name}: non-finite pair loss for (A, B) = {bad} (seed order {[sd["config"][-40:] for sd in seeds]})')

    def report(rule, w, arr):
        """Pooled argmin, per-pair argmins, leave-one-seed-out jackknife."""
        check_finite(f'rule {rule} {w}', arr)
        off = ~np.eye(S, dtype=bool)
        pooled = arr[off].sum(0)                                       # (L,)
        i = int(np.argmin(pooled))
        sel = lams[i]
        pairs = [lams[int(np.argmin(arr[a, b]))] for a in range(S) for b in range(S) if a != b]
        jk = []
        for s in range(S):
            keep = off.copy()
            keep[s, :] = keep[:, s] = False
            jk.append(lams[int(np.argmin(arr[keep].sum(0)))] if keep.any() else None)
        edge = sel in (min(l_ for l_ in lams if l_ > 0), max(lams)) if sel > 0 else False
        counts = {l_: pairs.count(l_) for l_ in sorted(set(pairs))}
        print(f"\n  rule {rule} {w:>9}: selected lam = {sel:g}{'  (GRID EDGE: extend the grid)' if edge else ''}"
              f"{'  (lam = 0: no smoothing preferred)' if sel == 0 else ''}")
        print(f"             per pair: " + ', '.join(f'{l_:g} x{c}' for l_, c in counts.items()))
        print(f"             jackknife (leave seed out, in seed order): " + ', '.join('none' if l_ is None else f'{l_:g}' for l_ in jk)
              + ('  (2 seeds: no pair left when one is dropped)' if S == 2 else ''))
        print('             pooled loss - min:  ' + '  '.join(f'{l_:g}: {v - pooled[i]:.3g}' for l_, v in zip(lams, pooled)))
        return sel, pairs, jk, pooled

    out = dict(arm=args.arm, lams=lams, seeds=[sd['config'] for sd in seeds], ridge=args.ridge,
               schedule=args.schedule, same_grid=same_grid, loss=loss, selected={}, pair_selected={},
               jackknife={}, theta_files={})
    print(f'\n######## {args.arm}: cross-seed CV over {S} seeds ({S * (S - 1)} ordered pairs), '
          f'{args.schedule} weights, ridge {args.ridge:g}')
    for w in windows:
        sel, pairs, jk, _ = report('A (M_B metric)', w, loss[w])
        out['selected'][w], out['pair_selected'][w], out['jackknife'][w] = sel, pairs, jk
        out['theta_files'][w] = {sd['config']: str(out_path(args.sweep_dir, sd['config'], sel, args.ridge, 'moment',
                                                          args.schedule, sd['mask'], args.tag)) for sd in seeds}
    if args.sigma_rule:
        out.update(loss_sigma=loss_sig, loss_check_MB=loss_chk, selected_sigma={}, pair_selected_sigma={},
                   jackknife_sigma={}, theta_files_sigma={}, check_MB={})
        print('\n  check: rule A recomputed as 1/2 X^T M_B X - X^T M_B theta_B(0) (must equal rule A to rounding)')
        for wn in loss_chk:
            check_finite(f'check {wn}', loss_chk[wn])
            off = ~np.eye(S, dtype=bool)
            a_, c_ = loss[wn][off], loss_chk[wn][off]
            scale = np.abs(a_).max(axis=-1, keepdims=True).clip(1e-300)
            rel = float((np.abs(a_ - c_) / scale).max())
            same = int(np.argmin(a_.sum(0))) == int(np.argmin(c_.sum(0)))
            out['check_MB'][wn] = dict(max_rel_diff=rel, same_argmin=same)
            print(f'    {wn}: max |A - check| / max|A| per pair = {rel:.2e}, pooled argmin identical: {same}')
            if not same:
                sys.exit(f'CHECK FAILED ({wn}): the M_B form of the loss picks another lam than rule A; '
                         f'rule B results not reported')
        for wn in loss_sig:
            sel, pairs, jk, _ = report('B (Sigma_A metric)', wn, loss_sig[wn])
            out['selected_sigma'][wn], out['pair_selected_sigma'][wn], out['jackknife_sigma'][wn] = sel, pairs, jk
            out['theta_files_sigma'][wn] = {sd['config']: str(out_path(args.sweep_dir, sd['config'], sel, args.ridge,
                                                                       'moment', args.schedule, sd['mask'], args.tag))
                                            for sd in seeds}
        print('\n  side by side (pooled argmin):  ' +
              '  '.join(f"{wn}: A {out['selected'][wn]:g} | B {out['selected_sigma'][wn]:g}" for wn in loss_sig))
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / (f'xseed_{args.arm}{args.tag}_sigma.pt' if args.sigma_rule else f'xseed_{args.arm}{args.tag}.pt')
    torch.save(out, dest)
    print(f'\n-> {dest}  (Theta of the selected lam per seed: out["theta_files"][window]); total {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
