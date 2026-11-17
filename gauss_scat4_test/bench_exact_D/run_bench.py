"""Exact-D_k test bench on the d = 16 Gaussian (gauss_scat4_test): for every subset of the 5 Scat2 terms, exact D_k (bench_defs.exact_D) versus
the sample estimator of kl_groups.H_estimate on 200 sets of N = 1e4 exact draws of p* (and on the MGD-generated samples of seed 900).
Writes gauss_scat4_test/bench_exact_D/results/bench_exact_D.csv (about 10 min on CPU). Needs gauss_scat4_test/runs/d16/results/spectrum.pt and the seed-900 samples
(pulled from the cluster beforehand). Reverse KL: kl_reverse.py. Figures and tables: d16_Dk_vs_exact.ipynb (reads results/*.csv)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.argv = ['x']
exec(open(HERE / 'bench_defs.py').read())            # S, a_k, exact_D, draws, A_of, xg, H_estimate, ROOT, M
OUT = ROOT / 'gauss_scat4_test/bench_exact_D/results'
OUT.mkdir(exist_ok=True)


def zmax(a):
    return max(2 * a[w] * S[w] if w in (0, M // 2) else (a[w] + a[M - w]) * S[w] for w in range(M // 2 + 1))


N, R = 10_000, 200
rng = np.random.default_rng(21)
rows = []
for r in range(1, 6):
    for G in itertools.combinations(range(5), r):
        a = a_k[list(G)].sum(0)
        D = exact_D_stable(G)                          # since 2026-11-16: no float64 cancellation (old exact_D gave inf for term 4)
        est = [H_estimate(-A_of(draws(N, rng), a), n_boot=50) for _ in range(R)]    # H_estimate takes the code-sign energy C = -A
        Dh = np.array([e['H'] for e in est]); lb = np.array([e['D_lb95'] for e in est]); kh = np.array([e['khat'] for e in est])
        bt = np.array([e['boot_std'] for e in est])
        old = np.array([e['ess_frac'] * N >= 100 and e['drift'] <= 2 * e['boot_std'] for e in est])     # rule before 11-13
        new = old & (kh < 0.5)                                                                           # + khat < 0.5
        eg = H_estimate(-A_of(xg, a), n_boot=50)
        fin = np.isfinite(D)
        rows.append(dict(terms=''.join(map(str, G)), n_terms=r, z_max=zmax(a), x_min=float(mode_x(G).min()), D_exact=D, Dhat_mean=Dh.mean(), Dhat_q05=np.quantile(Dh, .05),
                         Dhat_q95=np.quantile(Dh, .95), LB95_mean=lb.mean(), LB95_covers=np.mean(lb <= D), khat_mean=kh.mean(), khat_sd=kh.std(),
                         measured_old=old.mean(), measured_new=new.mean(),
                         bias_old=(Dh[old] - D).mean() if (old.any() and fin) else np.nan,
                         within2boot_old=np.mean(np.abs(Dh[old] - D) < 2 * bt[old]) if (old.any() and fin) else np.nan,
                         mgd_Dhat=eg['H'], mgd_LB95=eg['D_lb95'], mgd_khat=eg['khat']))
        print(rows[-1]['terms'], flush=True)
pd.DataFrame(rows).to_csv(OUT / 'bench_exact_D.csv', index=False, float_format='%.5g')
