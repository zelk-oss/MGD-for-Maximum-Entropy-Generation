"""H_G = D_KL(p || q_G) for groups G of potentials of a trained MGD model, from per-potential energies on samples of p.

Conventions (checked in the code, 2026-11-12):
  * the code has p(x) ~ exp(+Theta . phi(x))  (turbulence/logp_family_contrib.py, talk_logp_generated.py).
    E[:, k] = theta_k * phi_k(x_i) is stored with THAT sign (C_k in the logp_* scripts).
  * task convention p ~ exp(-sum theta phi), A_G = sum_{k in G} theta_k phi_k  =>  A_G = -C_G with C_G = sum_{k in G} E[:, k].
  * q_G ~ p exp(+A_G) = p exp(-C_G): the group's term removed, renormalised.
        H_G = log E_p[exp(A_G - E_p A_G)] = log E_p[exp(-(C_G - E_p C_G))] >= 0          (nats)
  * estimator on N samples x_i ~ p: H = logsumexp(a - mean a) - log N, a = -C_G, float64.
    0.5 Var(A_G) is only the 2nd-cumulant approximation, returned as a check.
  * a group's H is computed on the SUMMED energy of the group, never as a sum of single-term H.
  * L_6 (confinement) is never a group: H is not defined for it (q would not be normalisable).
  * what is known (agreed 2026-11-12):   H_hat <= H_k (Jensen, on average)   and   I_k <= H_k,
    I_k = D_KL(p || q_k*) with the other theta refitted = information of term k. H_hat and I_k are not ordered.
    Flagged groups (degenerate weights) are reported as lower bounds 'H_k >= H_hat', never as values.
"""
import numpy as np
from scipy.special import logsumexp

# flags for degenerate weights w_i ~ exp(a_i): the estimator is then biased LOW (the tail that carries E_p[w] is unsampled)
ESS_MIN = 0.10      # ESS / N below this
WMAX_MAX = 0.01     # largest single normalised weight above this
N_BOOT = 200


def _h(a):
    """H estimate(s) along the last axis; a = A_G - anything (the mean is removed here)."""
    a = a - a.mean(-1, keepdims=True)
    return logsumexp(a, axis=-1) - np.log(a.shape[-1])


def H_estimate(C, n_boot=N_BOOT, seed=0):
    """C: (N,) summed CODE-sign energy C_G = Theta_G . phi_G(x_i) on samples of p. Returns a dict (all in nats)."""
    a = -np.asarray(C, dtype=np.float64)                       # A_G (task sign)
    N = a.size
    H = float(_h(a))
    w = np.exp(a - a.max())
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    rng = np.random.default_rng(seed)
    boot = _h(a[rng.integers(0, N, size=(n_boot, N))])
    half = _h(a[rng.permuted(np.tile(np.arange(N), (20, 1)), axis=1)[:, :N // 2]]).mean()   # mean H over 20 halves
    out = dict(H=H, var2=0.5 * float(a.var()), boot_std=float(boot.std(ddof=1)), N=N,
               ess_frac=float(ess / N), wmax=float(w.max()), H_half=float(half), drift=float(H - half))
    # drift = H(N) - H(N/2) > 0 means H still grows with N: the tail is not converged
    out['flag'] = bool(out['ess_frac'] < ESS_MIN or out['wmax'] > WMAX_MAX or out['drift'] > 2 * out['boot_std'])
    # Gaussian-A rule: ESS ~ N exp(-Var A), so N >> exp(Var A) = exp(2 var2) is needed (log10 to avoid overflow)
    out['log10_N_needed'] = 2 * out['var2'] / np.log(10)
    # flagged: Jensen (E log W_bar <= log E W_bar) -> the estimate is a LOWER bound of H_k, reported as 'H >= value'
    out['regime'] = 'lower bound' if out['flag'] else 'measured'
    return out


def H_groups(E, groups, **kw):
    """E: (N, n_potentials) per-term energies (code sign). groups: {name: column indices}. -> {name: dict}."""
    return {g: dict(H_estimate(E[:, cols].sum(1), **kw), n_terms=len(cols)) for g, cols in groups.items()}


def H_vs_N(C, Ns, n_rep=20, seed=0):
    """Mean +- std of H over n_rep random subsets of size n, for each n in Ns (diagnostic: H(N) should flatten)."""
    a = -np.asarray(C, dtype=np.float64)
    rng = np.random.default_rng(seed)
    res = []
    for n in Ns:
        h = _h(np.stack([rng.choice(a, n, replace=False) for _ in range(n_rep)]))
        res.append((n, float(h.mean()), float(h.std(ddof=1))))
    return np.array(res)


# ---------------------------------------------------------------- group builders (from the model's own indexing)
def family_groups(cmap, exclude=('L6', 'L_6', 'L2p')):
    """cmap = check_potentials.theta_column_map(...): {family: (start, width)}. Confinement families are dropped."""
    return {f: np.arange(s, s + w) for f, (s, w) in cmap.items() if not any(e in f for e in exclude)}


def single_groups(cmap, family, labels=None):
    s, w = cmap[family]
    labels = labels if labels is not None else [str(i) for i in range(w)]
    return {f'{family}[{lab}]': np.array([s + i]) for i, lab in enumerate(labels)}


def scat4_labels(indices):
    """indices = pot.indices (3, n) of Scattering_Fourth_Order_Mod2_{Real,Imag}_1d: rows (j, j', l), j < j' < l.
    phi_{j,j',l} = mean_t Re/Im[(|x*psi_j|^2 * psi_l) conj(|x*psi_j'|^2 * psi_l)] (checked numerically 2026-11-12)."""
    return [tuple(int(v) for v in c) for c in np.asarray(indices).T]


def scale_pair_groups(cmap, family, indices):
    """{(j, j'): columns of all l} for one order-4 family."""
    s, _ = cmap[family]
    groups = {}
    for i, (j, jp, _) in enumerate(scat4_labels(indices)):
        groups.setdefault((j, jp), []).append(s + i)
    return {k: np.array(v) for k, v in groups.items()}
