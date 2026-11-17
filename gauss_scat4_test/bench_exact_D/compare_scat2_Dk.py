"""d16 Gaussian: estimated D_k of the Scat2 groups (from the actual MGD run) versus the EXACT D_k, and the exact refit I_k.

Reads  runs/d16/results/{H_gauss4.csv, E_gauss4_seed900.npz, spectrum.pt}  and the seed-900 generated samples (local only;
the other seeds' Theta_reg files live on JZ and are NOT pulled here, so Theta_reg-based exact values exist for seed 900 only).
Writes bench_exact_D/results/compare_scat2_Dk_{seed900,seeds,sensitivity,Ik}.csv.   Run:  python3 compare_scat2_Dk.py

Setting (all Gaussian, diagonal in the Fourier basis; coefficient c_k(w) of |xhat_w|^2 in the energy of term k):
  complex modes w = 1..7:  |xhat_w|^2 = S(w) Exp(1);  real modes w = 0, 8:  |xhat_w|^2 = S(w) chi2_1.
  D_G = KL(p || q_G) = sum_modes [ -log(1-z) - z ]   (real modes with 1/2),  z = (sum_{k in G} c_k(w) S(w)) (x2 for real modes)
  = +inf if z >= 1 anywhere.  Which law p:
    (a) 'star':  p = p*  (spectrum S), coefficients of theta_star                       (what bench_exact_D.csv uses)
    (b) 'model': the Gaussian law of the fitted model itself, S_Theta(w) = 1 / (sum_k c_k(w))  [x 1/2 for real modes], with the run's
                 Theta_reg Scat2 coefficients (seed 900)  -> the law the SDE is supposed to sample (L_6 / Scat4 terms ignored: their
                 energy contribution is ~1e-3 nats)
    (c) 'emp':   p = Gaussian with the empirical spectrum of the 1e4 generated samples, Theta_reg coefficients
  Exact I_k (refit, Scat2-only Gaussian family): I_k = min over precisions pi(w) = sum_{j not in G} t_j c_j(w) of KL(p* || q_t)
  = S(q*) - S(p*)  (p* in the family; L_6 / Scat4 terms are NOT refit here, so this is the Scat2-only family value)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.argv = ['x']
exec(open(HERE / 'bench_defs.py').read())          # ROOT, S, th (theta_star), psi2, M, a_k (-th psi2/M^2), xg (seed-900 samples), H_estimate
OUT = HERE / 'results'
RES = ROOT / 'gauss_scat4_test/runs/d16/results'
NM = M // 2 + 1                                    # independent modes 0..8
REAL = np.array([w in (0, M // 2) for w in range(NM)])


def coef(a):
    """per-term coefficient of |xhat_w|^2 (w = 0..8): c(w) = a(w) + a(M-w) for complex modes, a(w) for real modes."""
    a = np.atleast_2d(a)
    c = np.empty((a.shape[0], NM))
    for w in range(NM):
        c[:, w] = a[:, w] if REAL[w] else a[:, w] + a[:, M - w]
    return c


def zfun(c, Sp):
    """z(w) = precision removed / precision of p: complex  c S;  real  2 c S."""
    return c * Sp * np.where(REAL, 2.0, 1.0)


def exact_D_c(c, Sp, cfree=None):
    """exact D_G.  If cfree (coefficients of the terms NOT in G) is given and p is the model's own law (so c + cfree = 1 / (Sp * r)), then
    x = 1 - z = cfree / (c + cfree) is computed WITHOUT cancellation (needed: at w = 0 only term 4 acts, 1 - z ~ 1e-35 is below float64
    resolution of 1 - z).  Otherwise x = 1 - z directly (inf if x <= 0)."""
    f = np.where(REAL, 0.5, 1.0)
    if cfree is not None:
        x = cfree / (c + cfree)
        if np.any(x <= 0):
            return np.inf
        z = 1 - x
        return float(np.sum(f * (-np.log(x) - z)))
    z = zfun(c, Sp)
    if np.any(z >= 1):
        return np.inf
    return float(np.sum(f * (-np.log1p(-z) - z)))


def model_spectrum(call):
    """S_Theta(w) of the Gaussian fitted model with total coefficients call(w): precision = c (complex) or 2c (real)."""
    return 1.0 / (call * np.where(REAL, 2.0, 1.0))


# ---- coefficients (a) theta_star, (b)/(c) the run's Theta_reg recovered from the cached energies of seed 900 ------------------------
S9 = S[:NM]
c_star = coef(a_k)                                                  # (5, 9)
z_ = np.load(RES / 'E_gauss4_seed900.npz', allow_pickle=True)
E = z_['E']                                                         # (N, 24) code-sign energies Theta_k phi_k(x_i) on the generated samples
starts, widths, fams = z_['starts'], z_['widths'], list(z_['families'])
print('families', fams, starts, widths)
assert xg.shape[0] == E.shape[0], (xg.shape, E.shape)
xh = np.abs(np.fft.fft(xg, axis=1)) ** 2
X = xh[:, :NM]                                                      # |xhat_w|^2, w = 0..8
# Scat2 block = columns of the Scat2 family in E; if it is not in 'families' (only Scat4 listed) it is the first 5 columns
s2 = int(starts[fams.index('Scattering_Second_Order_Q1')])
A_run = -E[:, s2:s2 + 5]                                            # task-sign energy A_k(x_i), k = 0..4
c_run, *_ = np.linalg.lstsq(X, A_run, rcond=None)                   # (9, 5): exact if A_k is a linear function of |xhat_w|^2
c_run_reg = c_run.T
# Each term is ONE coefficient times a fixed phi_k, so c_run_k(w) = r_k c_star_k(w), r_k = theta_run_k / theta_star_k. Use that exact structure
# (the regression c_run_reg carries ~1e-6 relative noise, which would swamp the 1e-36 filter tails that decide D of the low-pass term).
r_k = (c_run_reg * c_star).sum(1) / (c_star ** 2).sum(1)
c_run = r_k[:, None] * c_star
print('theta_run / theta_star per term:', np.round(r_k, 4), ' structure residual |c_reg - r c_star|/|c_star| =',
      np.round(np.linalg.norm(c_run_reg - c_run, axis=1) / np.linalg.norm(c_star, axis=1), 6))
resid = np.abs(X @ c_run.T - A_run).max(0) / A_run.std(0)
print('regression residual (max |.|/std) per term:', resid)
Semp = X.mean(0) * np.where(REAL, 1.0, 1.0)                         # E|xhat_w|^2 under the generated samples
print('S empirical / S star (w=0..8):', np.round(Semp / S9, 3))
S_mod = model_spectrum(c_run.sum(0))
print('S_Theta / S star (w=0..8):', np.round(S_mod / S9, 3))
rel = np.linalg.norm(c_run - c_star) / np.linalg.norm(c_star)
comp = lambda c, g: c[[j for j in range(5) if j not in g]].sum(0)      # sum over the terms NOT in g (no cancellation)
print('||c_run - c_star|| / ||c_star|| =', rel)

subsets = [G for r in range(1, 6) for G in itertools.combinations(range(5), r)]
name = lambda G: ''.join(map(str, G))


def zmax_of(c, Sp):
    return float(zfun(c, Sp).max())


rows = []
for G in subsets:
    g = list(G)
    cs, cr = c_star[g].sum(0), c_run[g].sum(0)
    Cg = E[:, s2:s2 + 5][:, g].sum(1)                               # code-sign C_G of the run
    est = H_estimate(Cg, n_boot=200)
    ess = est['ess_frac'] * est['N']
    lowb = bool(ess < 100 or est['drift'] > 2 * est['boot_std'] or not (est['khat'] < 0.5))
    rows.append(dict(terms=name(G), n_terms=len(G),
                     D_star=exact_D_c(cs, S9, comp(c_star, g)), zmax_star=zmax_of(cs, S9),
                     D_model=exact_D_c(cr, S_mod, comp(c_run, g)), zmax_model=zmax_of(cr, S_mod),
                     D_emp=exact_D_c(cr, Semp), zmax_emp=zmax_of(cr, Semp),
                     Dhat=est['H'], boot=est['boot_std'], ESS=ess, khat=est['khat'], D_lb95=est['D_lb95'],
                     regime='lower bound' if lowb else 'measured'))
df9 = pd.DataFrame(rows)
df9.to_csv(OUT / 'compare_scat2_Dk_seed900.csv', index=False, float_format='%.5g')

# ---- other seeds: only the CSV of the run (old rule: ESS, drift; no khat / LB stored) ---------------------------------------------
H = pd.read_csv(RES / 'H_gauss4.csv')
rows = []
for sd in sorted(H.seed.unique()):
    d = H[(H.seed == sd) & (H.family == 'Scattering_Second_Order_Q1') & H.kind.isin(['single', 'family'])]
    for _, r in d.iterrows():
        if r.kind == 'family':
            t = '01234'
        else:
            t = str(int(r.group.split('[')[1][:-1]))
        G = [int(ch) for ch in t]
        rows.append(dict(seed=sd, terms=t, Dhat=r.H, ESS=r.ess_frac * r.N, drift=r.drift, boot=r.boot_std,
                         regime_old=('lower bound' if (r.ess_frac * r.N < 100 or r.drift > 2 * r.boot_std) else 'measured'),
                         D_star=exact_D_c(c_star[G].sum(0), S9, comp(c_star, G))))
pd.DataFrame(rows).to_csv(OUT / 'compare_scat2_Dk_seeds.csv', index=False, float_format='%.5g')

# ---- sensitivity: perturb theta_star by a random relative 2 % (norm) and recompute exact D of the single terms ------------------
rng = np.random.default_rng(0)
rows = []
for k in range(5):
    Dn, Dm, zm = [], [], []
    for _ in range(400):
        dlt = rng.standard_normal(th.shape)
        dlt *= 0.02 * np.linalg.norm(th) / np.linalg.norm(dlt)
        a_p = -(th + dlt)[:, None] * psi2 / M ** 2
        cp = coef(a_p)
        Dn.append(exact_D_c(cp[k], S9))                              # p fixed = p*
        Dm.append(exact_D_c(cp[k], model_spectrum(cp.sum(0)), comp(cp, [k])))       # p = the perturbed model's own Gaussian law
        zm.append(zmax_of(cp[k], S9))
    Dn, Dm, zm = np.array(Dn), np.array(Dm), np.array(zm)
    fin = np.isfinite(Dn)
    rows.append(dict(term=k, D_star=exact_D_c(c_star[k], S9, comp(c_star, [k])), zmax_star=zmax_of(c_star[k], S9),
                     frac_inf_pfixed=1 - fin.mean(), D_pfixed_q05=np.quantile(Dn[fin], .05) if fin.any() else np.nan,
                     D_pfixed_med=np.median(Dn[fin]) if fin.any() else np.nan, D_pfixed_q95=np.quantile(Dn[fin], .95) if fin.any() else np.nan,
                     frac_inf_model=1 - np.isfinite(Dm).mean(),
                     D_model_med=np.median(Dm[np.isfinite(Dm)]) if np.isfinite(Dm).any() else np.nan,
                     zmax_pfixed_q05=np.quantile(zm, .05), zmax_pfixed_q95=np.quantile(zm, .95)))
pd.DataFrame(rows).to_csv(OUT / 'compare_scat2_Dk_sensitivity.csv', index=False, float_format='%.5g')


# ---- exact I_k (refit within the Scat2-only Gaussian family) ---------------------------------------------------------------------
def kl_to_family(Sp, cmat, free):
    """min_t KL(p || q_t), x(w) = Sp r(w) sum_{j in free} t_j c_j(w) (> 0): convex in t -> damped Newton from t = 1 (x = free share > 0).
    Returns (KL, t); (inf, nan) if some mode has no free term (x = 0)."""
    f = np.where(REAL, 0.5, 1.0)
    B = (Sp * np.where(REAL, 2.0, 1.0))[None, :] * cmat[free]          # (nfree, 9): x = t @ B
    if np.any(B.sum(0) <= 0):
        return np.inf, np.full(len(free), np.nan)
    t = np.ones(len(free))
    kl = lambda t: float(np.sum(f * ((t @ B) - 1 - np.log(t @ B)))) if np.all(t @ B > 0) else np.inf
    for _ in range(500):
        x = t @ B
        g = B @ (f * (1 - 1 / x))
        Hs = (B * (f / x ** 2)) @ B.T
        step = -np.linalg.solve(Hs + 1e-14 * np.trace(Hs) * np.eye(len(t)), g)
        a, k0 = 1.0, kl(t)
        while kl(t + a * step) > k0 and a > 1e-12:
            a /= 2
        t = t + a * step
        if np.linalg.norm(a * step) < 1e-13 * (1 + np.linalg.norm(t)):
            break
    return kl(t), t


rows = []
for G in subsets:
    free = [j for j in range(5) if j not in G]
    row = dict(terms=name(G), n_terms=len(G), D_star=exact_D_c(c_star[list(G)].sum(0), S9, comp(c_star, list(G))))
    if free:
        Ik, t = kl_to_family(S9, c_star, free)
        # enforce positivity of the precision: report if violated
        pi = t @ c_star[free]
        row.update(I_k=Ik, pi_min=float(pi.min()), t_min=float(t.min()))
    else:
        row.update(I_k=np.nan, pi_min=np.nan, t_min=np.nan)
    rows.append(row)
dI = pd.DataFrame(rows)
dI.to_csv(OUT / 'compare_scat2_Dk_Ik.csv', index=False, float_format='%.5g')

pd.set_option('display.width', 250, 'display.max_columns', 30)
print('\n=== seed 900: estimate vs exact (nats) ===')
print(df9.round(4).to_string(index=False))
print('\n=== other seeds (CSV only) ===')
print(pd.read_csv(OUT / 'compare_scat2_Dk_seeds.csv').round(4).to_string(index=False))
print('\n=== sensitivity of the exact D of single terms to a 2 % relative perturbation of theta_star ===')
print(pd.read_csv(OUT / 'compare_scat2_Dk_sensitivity.csv').round(4).to_string(index=False))
print('\n=== exact I_k (refit, Scat2-only Gaussian family) vs D_k(theta_star) ===')
print(dI.round(5).to_string(index=False))
