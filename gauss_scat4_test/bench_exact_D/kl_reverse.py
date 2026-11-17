"""Exact reverse KL D_KL(q_G || p) for the d16 Gaussian bench; q_G has mode variance r*S, r = 1/(1 - z) (z = b S, complex modes;
z = 2 a S, real modes w = 0, M/2).  Complex mode (|xhat|^2 ~ Exp, mean S): KL(Exp(rS)||Exp(S)) = r - 1 - ln r.  Real mode (chi2_1): half of that.
Chatterjee-Diaconis: N ~ exp(D_KL(q||p)) needed for the importance-sampling estimate of Z."""
import sys
import itertools
import numpy as np
import pandas as pd
sys.argv = ['x']
exec(open('/home/chiaraz/phd/MGD-for-Maximum-Entropy-Generation/gauss_scat4_test/bench_exact_D/bench_defs.py').read())


def kl_qp(a):
    out = 0.0
    for w in range(M // 2 + 1):
        real = w in (0, M // 2)
        z = 2 * a[w] * S[w] if real else (a[w] + a[M - w]) * S[w]
        if z >= 1:
            return np.inf
        r = 1 / (1 - z)
        out += (0.5 if real else 1.0) * (r - 1 - np.log(r))
    return out


rows = []
for n in range(1, 6):
    for G in itertools.combinations(range(5), n):
        a = a_k[list(G)].sum(0)
        rows.append((''.join(map(str, G)), exact_D_stable(G), kl_qp_stable(G)))     # stable (no cancellation) since 2026-11-16
df = pd.DataFrame(rows, columns=['terms', 'D_exact', 'KL_q_p'])
print(df[np.isfinite(df.D_exact)].round(2).to_string(index=False))
df.to_csv('/home/chiaraz/phd/MGD-for-Maximum-Entropy-Generation/gauss_scat4_test/bench_exact_D/results/reverse_KL.csv', index=False, float_format='%.5g')
