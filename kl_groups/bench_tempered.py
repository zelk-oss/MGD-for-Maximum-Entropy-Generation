"""Test A (method 1) and method 3 on the exact d16 Gaussian bench.  Local CPU only.

  python kl_groups/bench_tempered.py A   --N 10000 --sets 200 --boot 100 [--procs 8]   -> results/tempered/testA_N<N>.csv
  python kl_groups/bench_tempered.py mgd                                               -> results/tempered/testA_mgd.csv
  python kl_groups/bench_tempered.py summary                                           -> results/tempered/testA_summary_N<N>.csv
  python kl_groups/bench_tempered.py M3                                                -> results/tempered/method3.csv
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_d16_exact as bx                                                 # noqa: E402
from tempered import EPS_LIST, dressed_response, tempered_estimate           # noqa: E402

OUT = Path(__file__).resolve().parent / 'results/tempered'
BENCH = bx.load_bench()
SUBSETS = bx.all_subsets(5)


def one_set(args):
    N, s, n_boot = args
    rng = np.random.default_rng([N, s])
    y = bx.draw_y(BENCH, N, rng)
    rows = []
    for gi, G in enumerate(SUBSETS):
        a = bx.A_from_y(BENCH, y, G)
        est = tempered_estimate(a, n_boot=n_boot, seed=s * 1000 + gi)
        for eps, r in est.items():
            rows.append(dict(N=N, set=s, group=bx.label(G), eps=eps, **r))
    return rows


def run_A(N, sets, n_boot, procs):
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(procs) as p:
        rows = [r for chunk in p.imap_unordered(one_set, [(N, s, n_boot) for s in range(sets)]) for r in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f'testA_N{N}.csv', index=False)
    print('wrote', OUT / f'testA_N{N}.csv', len(df), 'rows')


def run_mgd(n_boot=200):
    e = np.load(bx.D16 / 'E_gauss4_seed900.npz', allow_pickle=True)
    E = e['E']                                                              # columns: L_6 0-4, Scat2 5-9, Scat4 Real 10-19, Imag 20-23
    rows = []
    for G in SUBSETS:
        a = -E[:, [5 + k for k in G]].sum(1)
        for eps, r in tempered_estimate(a, n_boot=n_boot, seed=1).items():
            rows.append(dict(N=a.size, group=bx.label(G), eps=eps, **r))
    for name, cols in [('scat4_real', range(10, 20)), ('scat4_imag', range(20, 24)), ('scat4_both', range(10, 24))]:
        a = -E[:, list(cols)].sum(1)
        for eps, r in tempered_estimate(a, n_boot=n_boot, seed=1).items():
            rows.append(dict(N=a.size, group=name, eps=eps, **r))
    pd.DataFrame(rows).to_csv(OUT / 'testA_mgd.csv', index=False)
    print('wrote', OUT / 'testA_mgd.csv')


def summary(N):
    df = pd.read_csv(OUT / f'testA_N{N}.csv')
    rows = []
    for (g, eps), d in df.groupby(['group', 'eps']):
        G = tuple(int(k) for k in g.strip('{}').split(','))
        t_ex = bx.t_exact(BENCH, G, eps)
        D = bx.lam_exact(BENCH, G, 1.0)[0]
        rel = d[d.reliable]
        cover = lambda x: float(((x.t_hat - 2 * x.t_sd <= t_ex) & (t_ex <= x.t_hat + 2 * x.t_sd)).mean()) if len(x) else np.nan
        rows.append(dict(N=N, group=g, n_terms=len(G), eps=eps, D_exact=D, z=bx.z_of(BENCH, G), t_exact=t_ex,
                         t_mean=d.t_hat.mean(), t_sd=d.t_hat.std(ddof=1), bias=d.t_hat.mean() - t_ex,
                         frac_reliable=d.reliable.mean(), frac_ess=d.ess_ok.mean(), frac_khat=d.khat_ok.mean(),
                         frac_drift=d.drift_ok.mean(), frac_censored=d.censored.mean(),
                         n_rel=len(rel), bias_rel=(rel.t_hat.mean() - t_ex) if len(rel) else np.nan,
                         sd_rel=rel.t_hat.std(ddof=1) if len(rel) > 1 else np.nan,
                         coverage_all=cover(d), coverage_rel=cover(rel), mean_boot_sd=d.t_sd.mean()))
    out = pd.DataFrame(rows)
    out['pass_rel'] = (out.n_rel >= 10) & (out.bias_rel.abs() < out.sd_rel) & ((out.coverage_rel - 0.95).abs() < 0.05)
    out.to_csv(OUT / f'testA_summary_N{N}.csv', index=False)
    print('wrote', OUT / f'testA_summary_N{N}.csv')


def run_M3(n_sets=20, N=10000):
    """Method 3: P_G vs exact I_G, exact population P_G; estimated P_cv on exact draws (rest = other Scat2 terms) and on MGD."""
    OUT.mkdir(parents=True, exist_ok=True)
    e = np.load(bx.D16 / 'E_gauss4_seed900.npz', allow_pickle=True)
    E = e['E']
    rng = np.random.default_rng(7)
    ys = [bx.draw_y(BENCH, N, rng) for _ in range(n_sets)]
    rows = []
    for G in SUBSETS:
        rest = [k for k in range(5) if k not in G]
        row = dict(group=bx.label(G), n_terms=len(G), D_exact=bx.lam_exact(BENCH, G, 1.0)[0],
                   half_var_exact=0.5 * float(np.sum(np.where(BENCH['real'], 2.0, 1.0) * bx.coef(BENCH, G) ** 2)),
                   I_exact=bx.I_exact(BENCH, G), P_exact=bx.P_exact(BENCH, G))
        if rest:
            est = []
            for y in ys:
                a = bx.A_from_y(BENCH, y, G)
                Phi = np.stack([bx.A_from_y(BENCH, y, (k,)) for k in rest], 1)
                est.append(dressed_response(a, Phi))
            for key in ('bare', 'P_cv', 'P_in', 'P_ols_in'):
                v = np.array([r[key] for r in est])
                row[f'draw_{key}_mean'], row[f'draw_{key}_sd'] = v.mean(), v.std(ddof=1)
        # MGD seed 900: rest = every other term incl. L_6 and Scat4 (what the real model contains)
        a = -E[:, [5 + k for k in G]].sum(1)
        others = [c for c in range(E.shape[1]) if c not in [5 + k for k in G]]
        m = dressed_response(a, -E[:, others])
        row.update({f'mgd_{k}': v for k, v in m.items()})
        # same, rest restricted to remaining Scat2 terms (comparable with I_exact)
        if rest:
            m2 = dressed_response(a, -E[:, [5 + k for k in rest]])
            row.update({f'mgdS2_{k}': v for k, v in m2.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'method3.csv', index=False)
    print('wrote', OUT / 'method3.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['A', 'mgd', 'summary', 'M3'])
    ap.add_argument('--N', type=int, default=10000)
    ap.add_argument('--sets', type=int, default=200)
    ap.add_argument('--boot', type=int, default=100)
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()
    if a.cmd == 'A':
        run_A(a.N, a.sets, a.boot, a.procs)
    elif a.cmd == 'mgd':
        run_mgd()
    elif a.cmd == 'summary':
        summary(a.N)
    else:
        run_M3()
