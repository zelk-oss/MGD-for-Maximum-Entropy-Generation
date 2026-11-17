# Sign of the $L_6$ multipliers, normalisability and the "far bump" — recap 2026-11-17

Model: B+Scat4 turbulence (`turbulence_nested_M3_bicw_allmass_scat4`), $d = M = 256$, seeds 900–904,
$\lambda = 3\cdot10^{-5}$, ridge $10^{-4}$. Results: `theta_sign_constraint/results/pin_l6_test/pinl6_bscat4/`
(job 329564), old method: logs of job 1886491 in `theta_sign_constraint/results/`.

## 1. The model

Code sign (the papers use $\theta_{\rm paper} = -\Theta$):
$$
p_\Theta(x) = \frac{1}{Z_\Theta}\, e^{\Theta\cdot\phi(x)},\qquad
Z_\Theta = \int_{\mathbb R^d} e^{\Theta\cdot\phi(x)}\,dx,\qquad x\in\mathbb R^{256}.
$$
$\phi$ has $r = 341$ statistics in five families:

| family | columns | degree of homogeneity $k$: $\phi(a x) = a^k\phi(x)$ | sign of $\phi$ |
|---|---|---|---|
| $L_6$ (8 Morlet wavelets $J=8,Q=1$ + low-pass) | 0:9 | 6 | $\ge 0$ |
| scalar $\psi$ (GGD-K windows) | 9:141 | not homogeneous; grows at most like $\lvert z\rvert^{\alpha}$, fitted $\alpha\le1.63$ | — |
| scalar Morlet (GGD-K windows) | 141:201 | same | — |
| Scat4 Mod2 Real | 201:285 | 4 | indefinite |
| Scat4 Mod2 Imag | 285:341 | 4 | indefinite |

$\phi_{6,j}(x) \propto \sum_u \lvert (x\star\psi_j)(u)\rvert^6$. The 9 $L_6$ filters cover every Fourier mode
(weakest best response 0.70), so $\sum_j \phi_{6,j}(x) > 0$ for every $x\neq0$.

$L_6$ has no physical role: it is there only so that $Z_\Theta<\infty$.

## 2. Normalisability

Write $x = a\,\hat x$, $\lVert\hat x\rVert = 1$, $a>0$ (a ray). Along a ray
$$
\log p_\Theta(a\hat x) + \log Z_\Theta = \underbrace{S_{\rm low}(a)}_{\text{scalar, }\lesssim a^{1.63}} + S_4(\hat x)\,a^4 + S_6(\hat x)\,a^6,
\qquad S_k(\hat x) = \sum_{i:\,\deg i = k}\Theta_i\,\phi_i(\hat x).
$$
At large $a$ the $a^6$ term wins. Hence
- if $S_6(\hat x) > 0$ on some ray: $\log p\to+\infty$ along it, $Z_\Theta = \infty$;
- if every $\Theta_{6,j} < 0$: $S_6(\hat x) < 0$ on every ray (coverage above), the integrand decays like $e^{-c a^6}$, $Z_\Theta<\infty$.

So "all $L_6$ multipliers negative" is a sufficient condition for normalisability.

## 3. How $\Theta$ is obtained (time-regularised solve, $\Theta_{\rm reg}$)

The MGD SDE transports Gaussian noise ($t=0$) to the model ($t=1$) and gives raw per-step $\theta_t$.
$\Theta_{\rm reg}(t)$ on $n = 54\,999$ time nodes is the minimiser of a quadratic energy
$$
E(\Theta) = \int_0^1\!\Big[\tfrac12\Theta^\top M_t\Theta - b_t^\top\Theta\Big]dt
\;+\;\lambda\int_0^1\!\Big[\tfrac12\dot\Theta^\top\Sigma_t\dot\Theta - \dot m_t^\top\dot\Theta\Big]dt
= \tfrac12\Theta^\top A\,\Theta - f^\top\Theta ,
$$
$M_t = \mathbb E[\nabla\phi\,\nabla\phi^\top]$ on the walkers (the per-time minimiser is the raw $\theta_t$),
$\Sigma_t = \mathrm{Cov}(\phi)$, $\dot m_t = \tfrac{d}{dt}\mathbb E[\phi]$. $A$ is block-tridiagonal, solved
exactly (Thomas). The model used for interpretation is $\Theta := \Theta_{\rm reg}(1)$.

$E$ is defined up to a constant, so only differences $\Delta E$ mean something. If $\Theta_{\rm free}$ is the
unconstrained minimiser, any other $\Theta$ costs exactly
$$
\Delta E = E(\Theta) - E(\Theta_{\rm free}) = \tfrac12\,\delta^\top A\,\delta,\qquad \delta = \Theta-\Theta_{\rm free}.
$$
A small $\Delta E$ allows a large $\delta$ along directions where $A$ has small eigenvalues.

The pinning below is done **only in this offline solve**; the SDE transport and the generated samples are unchanged.

## 4. The three settings for the $L_6$ sign

Per-filter scale: $\varepsilon_j = 10^{-2}/\mathrm{sd}(\phi_{6,j})$ (so $\varepsilon_j\,\phi_{6,j}$ is $O(10^{-2})$ nats on typical data).

| setting | definition | how solved | status |
|---|---|---|---|
| **free** | $\min_\Theta E(\Theta)$ | one Thomas solve | $L_6$ multipliers positive at $t=1$ on every seed (max $+652$ to $+2\cdot10^6$) → $Z=\infty$ |
| **old: bounded (pnewton)** | $\min E(\Theta)$ s.t. $\Theta_{6,j}(t_k)\le-\varepsilon_j\ \forall k,j$ | projected Newton, 20 rounds | NOT converged on all 5 seeds (stalls) |
| **new: pinall** | $\min E(\Theta)$ s.t. $\Theta_{6,j}(t_k) = -\varepsilon_j\ \forall k,j$ | one Thomas solve with the $L_6$ entries fixed | exact, 20–50 min/seed |
| pinall10 | same with $-10\,\varepsilon_j$ | same | seeds 900, 901 only (time budget) |
| pinpos | fix at $-\varepsilon_j$ only the entries that free puts above $-\varepsilon_j$, iterate | repeated solves | never finished (time budget) |

pinall is a feasible point of the bounded problem, so the true bounded minimum satisfies
$E_{\rm free}\le E^*_{\rm bounded}\le E_{\rm pinall}$.

Measured $\Delta E = E - E_{\rm free}$ (units of $E$, a squared drift norm, not nats):

| seed | pinall | pinall10 | pnewton (old) |
|---|---|---|---|
| 900 | 1.54 | 1.00 | 1564 |
| 901 | 1.63 | 1.09 | 4.2 |
| 902 | 1.91 | — | 3.1 |
| 903 | 1.72 | — | 39 107 |
| 904 | 2.03 | — | 5806 |

So the sign constraint costs at most $\approx 2$ in $E$; the old pnewton files are far from the optimum of
their own problem on seeds 900, 903, 904.

## 5. What pinning changes

**Ranking of samples by $-\log p$** (data and generated signals, all seeds, pinall vs free and vs pnewton):
Spearman $\ge 0.99999$, overlap of the 5 % rarest $0.99$–$1.00$. Unchanged.
Reason: the data part of $\Delta E$ is $\tfrac12\int dt\,\mathbb E_{\rm walkers}\lvert\nabla(\delta\cdot\phi)\rvert^2$, so a
small $\Delta E$ makes $\delta\cdot\phi(x)$ almost constant where the walkers are; a constant shift of
$\log p$ does not change the order (it goes into $\log Z$).

**Per-family multipliers** ($\lVert\delta_F\rVert/\lVert\Theta_{{\rm free},F}\rVert$; $r$ = correlation over the data of
$\Theta_F\cdot\phi_F$ pinall vs free):

| family | relative change | $r$ |
|---|---|---|
| scalar $\psi$, Morlet | 0.06–0.8 % | 0.93–1.00 |
| Scat4 Real | 52–393 % | $-0.005$ to $0.68$ |
| Scat4 Imag | 26–280 % | $-0.75$ to $0.98$ |
| $L_6$ | ≈100 % (pinned) | 0.62–0.78 |

Scat4 (degree 4) and $L_6$ (degree 6) trade off along a nearly flat direction of $E$: the total $\log p$
on the data is stable, its split between Scat4 and $L_6$ is not. (Scat4 also moves by up to 395 % between
free and the old pnewton.) Per-family tables (rare/typical shares, Var/2d) for each setting: job 335378, pending.

## 6. The far bump

Keep only the two leading terms on a ray where $S_4 = w_4 > 0$ and $S_6 = -w_6 < 0$:
$$
f(a) = w_4 a^4 - w_6 a^6,\qquad f'(a) = 4w_4a^3 - 6w_6a^5 = 0
\;\Rightarrow\; a_*^2 = \frac{2w_4}{3w_6},\qquad f(a_*) = \frac{4\,w_4^3}{27\,w_6^2}.
$$
- $f$ rises for $a<a_*$ (the $a^4$ term wins), falls for $a>a_*$ (the $a^6$ term wins): normalisable, but
  with a maximum of $\log p$ at $a_*$ times the size of $\hat x$.
- With $\Theta_6 = -\varepsilon$, $w_6\propto\varepsilon$ is small: $a_*\propto\varepsilon^{-1/2}$ (far out),
  height $\propto\varepsilon^{-2}$ (huge).
- $\varepsilon\to10\varepsilon$: $a_*$ moves $\sqrt{10}\approx3.2\times$ closer, height $100\times$ lower.

**Measured** (radial scan: $\log p(a x)-\log p(x)$ for $a\in[1.25,50]$ on 200 data, 200 generated, the 20 rarest
data signals, 25 single-wavelet atoms):

| setting | largest rise | at $a$ |
|---|---|---|
| free | $10^{10}$–$10^{18}$ nats, still rising at $a=50$ | 50 |
| pinall ($-\varepsilon$) | $10^3$–$10^5$ nats (data/gen), up to $9\cdot10^5$ (rarest) | 6–25 |
| pinall10 ($-10\varepsilon$, 2 seeds) | ≤ 320 nats (data/gen), up to $8\cdot10^3$ (rarest) | 2.5–6 |
| pnewton (old) | from none to $8.5\cdot10^4$ nats depending on seed | 2–40 |

Check of the formula, seed 900, same data signal (no. 186): pinall peak at $a=20$, height $4.1\cdot10^4$;
pinall10 peak at $a=6$, height $155$. Ratios 3.3 (predicted 3.2) and 260 (predicted 100, same order;
lower-degree terms and the coarse $a$ grid).

Why it matters: $Z_\Theta$ integrates over all of $\mathbb R^{256}$. A region where the density is
$e^{10^4}$ times that of the data dominates $Z_\Theta$, hence $\log Z$, the entropy and absolute $\log p$.
The ranking (values of $\log p$ at the data, $a=1$) is not affected.

Where the bump comes from: the positive degree-4 part $S_4$, which only a degree-6 wall can stop. In free
there is no wall on some rays (no bump, but $Z=\infty$). pinall also raises to $-\varepsilon_j$ the $L_6$
entries that free had strongly negative, so it weakens walls the data had made strong (part of its bump
is created by the pinning). In the bounded setting those are kept, but every entry sitting at the bound
$-\varepsilon_j$ is again a weak wall: bumps remain wherever $S_4>0$ on such a ray.

## 7. Consequences for absolute quantities and for the papers

- The data carry no information on $p$ at $a\sim10$–$20$ (no samples there): the wall strength is a
  prior, not a measurement. There is no data-defined "true" $Z$ for the free model ($Z=\infty$).
- Gibbs lower bound (extreme-scattering draft): for **any** $\Theta$,
  $\log Z_\Theta \ge H(q) + \Theta\cdot\mathbb E_q\phi$, gap $=\mathrm{KL}(q\,\Vert\,p_\Theta)$. With $q$ the
  transported law and $H_{\rm bound}$: $\log Z_\Theta\ge H_{\rm bound}+\Theta(1)\cdot m_1$. Still valid with a
  bump, but the bump makes it loose (gap large).
- The pinned $\Theta$ is not the max-entropy solution of all constraints ($L_6$ moments are no longer
  matched): it is the max-entropy model of the other statistics with reference measure
  $e^{-\sum_j\varepsilon_j\phi_{6,j}}$. MGD Prop. 4.3 / $H_{\rm bound}$ unaffected; draft Eq. 9
  ($\log Z^\sigma\to\log Z_{\theta^*}$), Eq. 10 / Remark 1, the Cramér–Rao comparison and
  $\mathbb E(H) = \sum_k\theta_k\mathbb E(\phi_k)$ assume max-ent and are biased.

## 8. Open choices

1. **Interpretation only** (ranking, rare/typical by total $-\log p$): pinall at $-\varepsilon$ is enough.
   Per-family statements involving Scat4 / $L_6$ wait for the tables (job 335378).
2. **Absolute values**: choose the wall by a rule, not by hand:
   - $\varepsilon$-scan: smallest $c$ with $\Theta_6=-c\,\varepsilon$ such that the radial scan shows no rise
     beyond the data ($\approx$ 60–80 GPU-h for $c\in\{10,100,1000\}$ × 5 seeds);
   - principled: impose directly that the score points inward at the data,
     $x\cdot\nabla\log p_\Theta(x) = \tfrac{d}{da}\log p_\Theta(ax)\big|_{a=1}\le0$, a linear inequality in
     $\Theta(1)$ per signal (needs a new constrained solver). Consistent with the exact identity
     $\mathbb E_p[x\cdot\nabla\log p(x)] = -d$ for any smooth density.
   A single global $\varepsilon$ strong enough for the worst ray pushes $L_6$ into the data range on the
   others (its share in the tables grows); the scan only checks finitely many rays.
