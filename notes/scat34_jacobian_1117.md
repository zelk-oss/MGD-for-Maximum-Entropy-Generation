# Scat3 / Scat4 potentials: bank fix and cheaper Jacobians (2026-11-17)

Recap for a reader who wants the mathematics behind three code changes made on 2026-11-17
(commits 3a6f53f, 32101c3, and the Scat3 change, uncommitted at the time of writing).
Code: `codes/potentials/potentials_1d.py`, `codes/potentials_builder.py`; checks in `grad_speed/`.

## 0. Notation

- Signal $x \in \mathbb{R}^M$ (one walker), periodic, $M = 256$ in the runs here. DFT $\hat x(\omega)$.
- Q = 1 Morlet bank (`filters`): band-pass wavelets $\psi_0, \dots, \psi_{J-1}$ (fine to coarse) and the
  low-pass $\psi_J$ in last position, i.e. $J+1$ filters; $J = 8$. Filters act by Fourier multiplication:
  $(\psi \star x)^\wedge = \hat\psi\, \hat x$.
- Wavelet coefficients $u_j = W_j x = \psi_j \star x$ (complex), modulus $|u_j|$, squared envelope $E_j = |u_j|^2$.
- $h_k$: the filter with symbol $\hat h_k = |\hat\psi_k|^2$ (real, non-negative; for an analytic Morlet wavelet it
  is not even in $\omega$, so $h_k \star f$ of a real $f$ is complex). Its adjoint has symbol
  $\overline{|\hat\psi_k|^2} = |\hat\psi_k|^2$: $h_k$ is self-adjoint.
- Every statistic is a time average $\frac1M \sum_t (\cdot)$, an empirical mean over the $M$ positions.
- MGD needs, for every walker, the **Jacobian** of the statistic vector $\Phi : \mathbb{R}^M \to \mathbb{R}^K$,
  i.e. the $K$ gradients $\nabla_x \phi_k(x) \in \mathbb{R}^M$. They enter the Gram matrix
  $G = \mathbb{E}[\nabla\Phi\, \nabla\Phi^\top]$ (one $K\times K$ matrix per time step), the costly part of a step.
  A second code path ("projected", argument `v`) returns only the contraction $\sum_k v_k \nabla\phi_k$; it is used
  for the corrector move and was not changed.

## 1. Bug: Scat3 was built on the Q = 3 bank in the turbulence runs (fixed, commit 3a6f53f)

Scat3 statistics (Real part; the Imag part takes Im instead of Re):
$$\phi_{jk}(x) = \frac1M \sum_t \mathrm{Re}\Big[\overline{u_k(t)}\,\big(\psi_k \star |u_j|\big)(t)\Big],
\qquad (j,k) \in \mathcal K_3 = \{ j \in \{0..J-1\},\ k \in \{j..J\} \},$$
$|\mathcal K_3| = \sum_{j=0}^{J-1}(J+1-j) = J(J+3)/2 = 44$ at $J = 8$. $j$ is the scale of the modulus, $k \ge j$ the
filter applied to both $x$ and $|u_j|$ (low-pass included).

The index set is built for a bank of $J+1$ filters ($J$ wavelets plus the low-pass last). The builder passed the
run's `filters_Q` bank. At Q = 1 that is the same tensor as `filters`, so jets and S&P were correct. At Q = 3 (turbulence)
it has $3J+1 = 25$ filters, and the same indices then pick the 9 finest of them (about 2.7 octaves, no low-pass):
not the intended statistics, and no error raised.
Fix (user decision: Scat3 and Scat4 exist on the Q = 1 bank only): the Scat3 terms are always built on `filters`;
the class raises an error for any bank that does not have $J+1$ filters; the unused 1D Scat4 terms on the Q bank were removed.
Consequence: the Scat3 rows of the turbulence moment-reproduction table (job 1867044) were computed on the wrong
bank.

## 2. Scat4 Mod2: Jacobian evaluated only on the kept index set (commit 32101c3)

Statistics (Real; Imag analogous with Im):
$$\phi_{abc}(x) = \frac1M \sum_t \mathrm{Re}\Big[(\psi_c \star E_a)(t)\ \overline{(\psi_c \star E_b)(t)}\Big],
\qquad (a,b,c) \in \mathcal K_4 .$$
By Parseval, $\phi_{abc} = \frac1M\, \mathrm{Re}\,\langle E_a,\ h_c \star E_b\rangle$, a **bilinear form in
$(E_a, E_b)$**, symmetric under $a \leftrightarrow b$ because $h_c$ is self-adjoint.

Index sets (`indices_fourth_order_Q(J, 1, offset=1, lite=True)`): $a < b$ among the $J$ wavelets, $c > b$.
- Real: $c \in \{b+1..J\}$ (low-pass allowed): $K = \sum_{b} b\,(J-b) = 84$ at $J = 8$.
- Imag (`include_lowpass=False`): $c \in \{b+1..J-1\}$: $K = 56$.
- Real with the diagonal $a = b$ (`_diag` variant): $K = 120$.

**Gradient.** With $dE_a = 2\,\mathrm{Re}(\overline{u_a}\, du_a)$ and $du_a = \psi_a \star dx$, the product rule
gives exactly two terms, one per factor:
$$\nabla\phi_{abc} = T_{abc} + T_{bac},\qquad
T_{abc} = \frac{2}{M}\ \mathrm{Re}\, W_a^{*}\Big( u_a \cdot \mathrm{Re}(h_c \star E_b) \Big),$$
where $W_a^{*}$ is the adjoint of $W_a$ (Fourier multiplier $\overline{\hat\psi_a}$). $T_{abc}$ is the derivative
through $E_a$ with $E_b$ fixed. It depends only on its own triple, which costs one forward and one inverse FFT per
walker.

**Old code.** It evaluated $T_{abc}$ on the full grid $a, b \in \{0..J-1\}$, $c \in \{0..J\}$, i.e.
$8 \times 8 \times 9 = 576$ vectors in $\mathbb R^M$ per walker. It then formed $T_{abc} + T_{bac}$ and kept the
entries in $\mathcal K_4$.

**New code.** It evaluates $T$ only on $\mathcal K_4 \cup \mathcal K_4^{\top}$, where
$\mathcal K_4^{\top} = \{(b,a,c) : (a,b,c) \in \mathcal K_4\}$. That is $2K$ terms: 168 for Real. The Imag class
had two such full tensors (one per factor, the second with the frequency-reversed filter), $2 \times 576$, of which
$2 \times 56 = 112$ are used. The function computed is the same; only the number of evaluated terms changed.

**Diagonal variant** (`_diag`, $a \le b$, $K = 120 = 84 + 36$ diagonal triples). A diagonal triple $(a,a,c)$ is its own
partner, so $\nabla\phi_{aac} = 2T_{aac}$. The set of distinct terms needed is therefore $2 \cdot 84 + 36 = 204$, not
$2K = 240$. The code builds the partner list without removing duplicates, so it evaluates the 36 diagonal terms twice:
the result is correct (it forms $T_{aac} + T_{aac}$, as the old code did), with 36 redundant evaluations. This variant
is not used by any current launcher.

**Check** (job 336982, V100, $M = 256$, $J = 8$, $B = 16$ random Gaussian walkers, `grad_speed/check_scat4_gather.py`):
- new vs old (verbatim copy of the old code): max |diff| / max |old| = **0 exactly**, in float32 and float64, for
  Real, Real_diag and Imag. The result is bit-identical.
- independent check: $\sum_k v_k \nabla\phi_k$ (new) vs the projected path, random $v$, float32: 1.0e-7 to 2.0e-7
  relative (float32 round-off). The projected path cannot run in float64 (it allocates a float32 array; pre-existing).
- time of the full Jacobian alone (float32, $B = 500$, median of 10): Real 9.15 -> 4.19 ms (2.2x),
  Real_diag 9.43 -> 5.55 ms (1.7x), Imag 35.11 -> 7.84 ms (4.5x). The Real speed-up is below 576/168 = 3.4 because the
  first layer and the envelope FFTs are shared and still computed in full.
- not measured: the effect on the time of a whole MGD step (depends on the share of this Jacobian in the step).

## 3. Scat3: the same restriction (2026-11-17, check job 337296)

With $\phi_{jk}$ as in section 1, Parseval and the self-adjointness of $h_k$ give
$$\phi_{jk} = \frac1M \sum_t |u_j(t)|\ \mathrm{Re}\big(h_k \star x\big)(t) ,$$
a product of two factors: $|u_j|$ (nonlinear in $x$) and $\mathrm{Re}(h_k \star x)$ (linear in $x$).
The gradient has one term per factor:
$$\nabla\phi_{jk} = P_{jk} + Q_{jk},$$
$$P_{jk} = \frac1M\ \mathrm{Re}\, W_j^{*}\Big( \tfrac{u_j}{|u_j|}\, \mathrm{Re}(h_k \star x) \Big)
\quad\text{(through } |u_j|,\ d|u_j| = \mathrm{Re}(\overline{u_j/|u_j|}\, du_j)\text{; code: result\_conv\_phase)},$$
$$Q_{jk} = \frac1M\ \mathrm{Re}\big( h_k \star |u_j| \big)
\quad\text{(through the linear factor; code: result\_conv\_modulus)}.$$
**Imag class.** $\phi^{\mathrm{Im}}_{jk} = \frac1M \sum_t \mathrm{Im}\big[\overline{u_k}\,(\psi_k\star|u_j|)\big]$.
By Parseval and because $x$ is real, $\phi^{\mathrm{Im}}_{jk} = \frac1M \sum_t x(t)\,\mathrm{Im}(h_k \star |u_j|)(t)$.
Let $\tilde h_k$ be the frequency-reversed filter, $\hat{\tilde h}_k(\omega) = |\hat\psi_k(-\omega)|^2$ (code
`filters_abs2_flip`). For any real $f$, $\tilde h_k \star f = \overline{h_k \star f}$, since
$\overline{\hat f(-\omega)} = \hat f(\omega)$. Hence $\mathrm{Im}(\tilde h_k\star f) = -\mathrm{Im}(h_k\star f)$. Using the
self-adjointness of $h_k$ and $\tilde h_k$, the same statistic can be written
$\phi^{\mathrm{Im}}_{jk} = -\frac1M \sum_t |u_j(t)|\,\mathrm{Im}(h_k\star x)(t)$. Its two gradient terms:
$$P_{jk} = \frac1M\ \mathrm{Re}\, W_j^{*}\Big( \tfrac{u_j}{|u_j|}\, \big(-\mathrm{Im}(h_k \star x)\big) \Big),\qquad
Q_{jk} = -\frac1M\ \mathrm{Im}\big(\tilde h_k \star |u_j|\big) = \frac1M\ \mathrm{Im}\big(h_k \star |u_j|\big).$$
These are the signs in the code: `-torch.imag(x_filtered_2)` in $P$, and `- result_conv_modulus.imag` with
`filters_abs2_flip` in $Q$. I checked this derivation against the Scat3 Imag code. Same two-term structure as the Real class.
(For Scat4 Imag, $\phi^{\mathrm{Im}}_{abc} = \frac1M\sum_t \mathrm{Im}[(\psi_c\star E_a)\overline{(\psi_c\star E_b)}]$ is
antisymmetric under $a \leftrightarrow b$ directly from the definition, since
$\mathrm{Im}(z\bar w) = -\mathrm{Im}(w \bar z)$. That is why its index set uses $a<b$ only, with no diagonal.)

**Differentiability.** $P_{jk}$ contains $u_j/|u_j|$, which is undefined where $u_j(t) = 0$. The code has **no guard**:
`abs_eps` is called with its default $\varepsilon = 0$, i.e. $|u| = \sqrt{(\mathrm{Re}\,u)^2 + (\mathrm{Im}\,u)^2}$. An exact zero
gives $0/0$ = NaN. For walkers with a continuous law, $u_j(t) = 0$ exactly is an event of probability zero; in float32 it
would need both the real and the imaginary part to round to exactly 0. If it happens, the NaN reaches $G$ or the
right-hand side, and `_live_potentials` (`sde_routines.py`) stops the run with a `FloatingPointError` naming the
statistics; it does not continue silently. Scat4 Mod2 has no such point: $E_a = |u_a|^2$ is a polynomial in $x$, and its
gradient $2u_a$ is defined everywhere. The GGD scalar potentials use $\sqrt{z^2 + \varepsilon_j}$ with $\varepsilon_j > 0$. Unlike Scat4 there is no $a \leftrightarrow b$ partner: $\nabla\phi_{jk}$ uses
$P$ and $Q$ at the same $(j,k)$.

Old code: $P$ and $Q$ on the full grid $j \in \{0..J-1\}$, $k \in \{0..J\}$: $2 \times 72 = 144$ terms, then restricted to
$\mathcal K_3$. New code: only $(j,k) \in \mathcal K_3$: $2 \times 44 = 88$ terms (Imag_nolp: $2 \times 36$).
Expected gain at most 144/88 = 1.6x on that part, smaller than Scat4. Scat3 is used in the jets models only.

Check: `grad_speed/check_scat3_gather.py`, same tests and thresholds as for Scat4 (job 337296, V100):
- new vs old: **0 exactly** in float32 and float64 for Real (K = 44), Imag (44) and Imag_nolp (36): bit-identical.
- $\sum_k v_k \nabla\phi_k$ vs the projected path, float32: 1.8e-7 to 2.8e-7 relative.
- time of the full Jacobian alone (float32, $B = 500$, median of 10): Real 2.30 -> 2.02 ms (1.14x), Imag 2.30 -> 2.03 ms
  (1.13x), Imag_nolp 2.34 -> 1.85 ms (1.27x). The gain is small because Scat3 is cheap to begin with: about 2 ms against
  9 to 35 ms for the old Scat4 Jacobians. Its cost is dominated by the shared first layer, not by the 144 vs 88 terms.

## 4. Why restricting the evaluation is exact (the general argument)

Old and new code compute the same set of vectors $\{\nabla\phi_k\}_{k \in \mathcal K}$. Each $\nabla\phi_k$ is a fixed
sum of partial terms ($T_{abc} + T_{bac}$, or $P_{jk} + Q_{jk}$). Each partial term is computed from the shared
first-layer quantities ($u$, $E$, $h\star E$) by element-wise products and FFTs along time only. There is no sum over
the index before the restriction. So a partial term outside $\mathcal K \cup \mathcal K^\top$ never enters any
returned value, and skipping it changes nothing. The terms skipped are not statistics of the model; they are the
index combinations the model excludes:
- Scat4: entries $T_{a'b'c}$ with $a' = b'$ (excluded by `offset=1`), or with $c \le \max(a', b')$. The bound is
  $\max(a', b')$, not $b'$, because the partner $(b,a,c)$ of a kept triple $(a,b,c)$, $a<b<c$, has its first two indices
  swapped. An entry with $a' > b'$ is needed iff $c > a'$.
- Scat3: pairs with $k < j$.

**Mathematics vs numerics.** The argument above shows that the two codes compute the same real numbers: an exact
statement about the functions. That the floating-point results are **bit-identical** (measured difference exactly 0,
jobs 336982 and 337296) is a separate, empirical fact. It holds because cuFFT returns, for each signal in a batch,
a result that does not depend on the batch size or on the other signals in it, and the element-wise operations are
the same. This is observed, not implied by the argument. A different FFT library, GPU or PyTorch version could in
principle break it, without breaking the mathematical equality, which would then hold to round-off.
