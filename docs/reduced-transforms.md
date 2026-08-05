# Reducing spherical-harmonic transforms in $\Delta_\Gamma$

**Summary.** Algorithm 4 costs 12 transforms per matvec (8 syntheses, 4 analyses). A flux-form
reformulation, with weights chosen so that every analyzed field is smooth on $S^2$, evaluates the
same operator in **6 transforms** (4 syntheses, 2 analyses). Notation follows `algos.pdf`.

---

## 1. Where the current cost comes from

| Algorithm 4 line | Work | Transforms |
|---|---|---|
| 1 | $\partial_\theta u,\ \partial_\varphi u$ | 2 $\mathcal{S}$ |
| 5 | analysis of 3 Cartesian components of $\nabla_\Gamma u$ | 3 $\mathcal{A}$ |
| 6 | $\partial_\theta$ and $\partial_\varphi$ of each of those 3 components | 6 $\mathcal{S}$ |
| 8 | final analysis | 1 $\mathcal{A}$ |
| | | **12** |

Two sources of waste:

1. The gradient is carried in **ambient $\mathbb{R}^3$ components** — 3 fields for an intrinsically
   2-dimensional object.
2. Line 6 takes **both** derivatives of **each** component, where the divergence needs only one
   derivative of each of two fluxes.

There is also a possible free win independent of everything below: Algorithm 1 as specified returns
all five derivatives. The Laplacian path needs only $\partial_\theta u$ and $\partial_\varphi u$;
the second-derivative and mixed-derivative machinery is used exclusively by Algorithm 3 (curvature).
If `surface_screened_laplacian` calls Algorithm 1 wholesale, it is doing 5 syntheses where 2 suffice
at lines 1 and 6.

---

## 2. The constraint that shapes the solution

The Cartesian design in Algorithm 4 exists to avoid pole singularities, and it is correct to do so.
The relevant property is smoothness **as a scalar function on $S^2$**, since that is what controls
SH coefficient decay and hence whether $\mathcal{A}$ is meaningful.

| Quantity | Smooth on $S^2$? |
|---|---|
| $\partial_\varphi u$ | yes — exactly band-limited, eq. (2.2) |
| $\sin\theta\,\partial_\theta u$ | yes — exactly band-limited, eq. (2.4) |
| $\partial_\theta u$ | **no** — bounded, but $\varphi$-dependent limit at the poles |
| $g_{\theta\theta},\ g^{\theta\theta},\ V_\theta,\ V_\varphi$ | **no** in general |
| $(\nabla_\Gamma u)_x,\ (\nabla_\Gamma u)_y,\ (\nabla_\Gamma u)_z$ | yes |

Concretely, for the ellipsoid $X = (a\sin\theta\cos\varphi,\ b\sin\theta\sin\varphi,\ c\cos\theta)$,
$|X_\theta|^2 \to a^2\cos^2\varphi + b^2\sin^2\varphi$ as $\theta\to 0$: no limit exists.

Algorithm 4 never analyzes anything in the "no" rows — the non-smooth quantities appear only as
pointwise grid factors. **Any replacement must preserve this property.** The naive flux form
$P = \sqrt{g}\,(g^{\theta\theta}u_\theta + g^{\theta\varphi}u_\varphi)$,
$Q = \sqrt{g}\,(g^{\varphi\theta}u_\theta + g^{\varphi\varphi}u_\varphi)$ does not:
on the round sphere with $u = x$, $Q = -\sin\varphi$, which is not a function on $S^2$.

### The correct weighting

$P$ and $Q$ are $\sqrt{g}$ times the contravariant components of $G := \nabla_\Gamma u$. Using
$\det[X_\theta, X_\varphi, n] = -\sqrt{g}$:

$$P = -\,G\cdot(X_\varphi \times n), \qquad \sin\theta\,Q = -\,G\cdot\big(n \times \sin\theta\,X_\theta\big).$$

Every factor on the right is smooth on $S^2$: $G$ smooth, $n$ smooth, $X_\varphi$ smooth, and
$\sin\theta\,X_\theta$ smooth because it is exactly band-limited by the recurrence already
implemented. Hence $P$ and $\tilde{Q} := \sin\theta\,Q$ are analyzable, on the same footing and for
the same structural reason as the Cartesian gradient components.

The cross products are the smoothness certificate only — they are not needed in the code.

---

## 3. Precompute (once per surface, grid space)

Replaces `_precompute_metric_quantities()`. From the embedding coefficients $\hat{X}^m_\ell$, obtain
$X_\varphi$ and $\sin\theta\,X_\theta$ componentwise via Algorithm 1, where the latter is the
**undivided** output of Algorithm 1 line 4, i.e. $\mathcal{S}(v^m_\ell)$ with no $/\sin\theta$.
Then, pointwise:

$$\tilde{g}_{\theta\theta} := |\sin\theta\,X_\theta|^2, \qquad
  \tilde{g}_{\theta\varphi} := (\sin\theta\,X_\theta)\cdot X_\varphi, \qquad
  g_{\varphi\varphi} := |X_\varphi|^2$$

$$J := \frac{\sqrt{\tilde{g}_{\theta\theta}\,g_{\varphi\varphi} - \tilde{g}_{\theta\varphi}^{\,2}}}{\sin^2\theta}
  \qquad\text{so that } \sqrt{\det g} = J\sin\theta$$

(The radicand is $\sin^2\theta\det g = J^2\sin^4\theta$, so the square root is $J\sin^2\theta$ — hence
$\sin^2\theta$, not $\sin\theta$, in the denominator.)

Store four scalar grid arrays:

$$p_1 = \frac{g_{\varphi\varphi}}{J\sin^2\theta}, \qquad
  p_2 = -\frac{\tilde{g}_{\theta\varphi}}{J\sin^2\theta}, \qquad
  q_2 = \frac{\tilde{g}_{\theta\theta}}{J\sin^2\theta}, \qquad
  r = \frac{1}{J\sin^2\theta}$$

All four are bounded: the $\sin^2\theta$ denominators cancel against vanishing numerators
($\tilde{g}_{\theta\varphi} = O(\sin^2\theta)$, $g_{\varphi\varphi} = O(\sin^2\theta)$), the same
finite limits the current $V_\theta, V_\varphi$ have. Note $p_1, p_2, q_2$ are bounded where
$V_\varphi = O(1/\sin\theta)$ is not.

**Three scalar arrays replace the six components of $V_\theta, V_\varphi$.** The surface
representation is unchanged: $X_\theta, X_\varphi$ still come componentwise from $\hat{X}^m_\ell$
via Algorithm 1.

---

## 4. The per-matvec algorithm

Input $\{u^m_\ell\}$; output $\{(\Delta_\Gamma u)^m_\ell\}$.

| # | Step | Transforms |
|---|---|---|
| 1 | $v^m_\ell \leftarrow \alpha^+(\ell-1,m)u^m_{\ell-1} + \alpha^-(\ell+1,m)u^m_{\ell+1}$; $A \leftarrow \mathcal{S}(v^m_\ell)$ | $\mathcal{S}$ |
| 2 | $B \leftarrow \mathcal{S}(im\,u^m_\ell)$ | $\mathcal{S}$ |
| 3 | $P \leftarrow p_1 A + p_2 B$, $\tilde{Q} \leftarrow p_2 A + q_2 B$ — pointwise | — |
| 4 | $\hat{P} \leftarrow \mathcal{A}(P)$, $\hat{\tilde{Q}} \leftarrow \mathcal{A}(\tilde{Q})$ | 2 $\mathcal{A}$ |
| 5 | $s^m_\ell \leftarrow \alpha^+(\ell-1,m)\hat{P}^m_{\ell-1} + \alpha^-(\ell+1,m)\hat{P}^m_{\ell+1} + im\,\hat{\tilde{Q}}^m_\ell$ | — |
| 6 | $\Delta_\Gamma u \leftarrow r \cdot \mathcal{S}(s^m_\ell)$ | $\mathcal{S}$ |
| 7 | $\{(\Delta_\Gamma u)^m_\ell\} \leftarrow \mathcal{A}(\Delta_\Gamma u)$; zero $\ell \ge L-2$ | $\mathcal{A}$ |

**4 syntheses + 2 analyses = 6**, versus 12.

- $A$ and $B$ are exactly $\sin\theta\,\partial_\theta u$ and $\partial_\varphi u$.
- Steps 1 and 5 use the **same** precomputed $\alpha^\pm$ table; step 5 is the adjoint-style reuse
  of the shift already implemented for step 1. Adding the two flux contributions in coefficient
  space before synthesizing is what saves the final pair of transforms.
- The **only** division by $\sin\theta$ anywhere is folded into $p_1, p_2, q_2, r$ at precompute
  time. The per-matvec path contains none.

---

## 5. Numerical trade-off

Both schemes contain two powers of $\sin\theta$ division in total. What differs is **placement**.

- **Algorithm 4** spends them in separate stages, one before line 5 and one after. The intervening
  analysis suppresses the polar spike: Gauss–Legendre weights give $w_1 = O(L^{-2})$ at the polar
  ring, so a grid error of $\varepsilon L$ there contributes
  $\sim L^{-2}\cdot L^{1/2}\cdot \varepsilon L = \varepsilon L^{-1/2}$ to any coefficient. Stage 2
  then starts from clean coefficients and incurs a *fresh* $\varepsilon L$. The two amplifications
  never multiply. Net grid-space relative error: $\varepsilon L$.
- **The new scheme** has no division at all through step 5, then pays for both powers at once in
  $r = O(L^2)$ at step 6 — one event, with no intervening analysis to break it in half. Net
  grid-space relative error: $\varepsilon L^2$.

The mechanism: with $N \approx L+1$ Gauss–Legendre nodes, $1 - x_1 = O(N^{-2})$ so
$\sin\theta_1 = O(N^{-1})$. Since $s = J\sin^2\theta\,\Delta_\Gamma u$ is $O(L^{-2})$ at the polar
ring but $O(1)$ over the bulk, and synthesis commits roundoff scaled by the field's *global* size at
every node alike, multiplying by $r \sim L^2$ recovers the signal and inflates the noise.

| | divisions | placement | grid-space relative error |
|---|---|---|---|
| Algorithm 4 | $\sin\theta$, $\sin\theta$ | separated by $\mathcal{A}$ | $\varepsilon L$ |
| Six-transform | $\sin^2\theta$ | all at the end | $\varepsilon L^2$ |

**This likely does not reach the returned coefficients.** Step 7's analysis suppresses the spike
exactly as line 5 does today: $L^{-2}\cdot L^{1/2}\cdot\varepsilon L^2 = \varepsilon L^{1/2}$,
comparable to the ordinary $\varepsilon\sqrt{L}$ accumulation of a transform pair — and the new
scheme runs half as many transforms, lowering that baseline. Inside the implicit solve, GMRES sees
only coefficients, so the extra power should be invisible.

It matters only if grid values of $\Delta_\Gamma u$ are consumed directly: a nonlinear reaction
term, max-norm diagnostics, or an adaptive error estimator.

Algorithm 1 line 7 already divides by $\sin^2\theta$, so the code is exposed to $\varepsilon L^2$
today — just on the second-derivative path, which the Laplacian never touches.

---

## 5a. float32 / WebGPU

Target is WebGPU, which is float32-only: $\varepsilon = 2^{-24} \approx 6\times10^{-8}$ (spacing
$2^{-23} \approx 1.2\times10^{-7}$). No float64 fallback exists on device. All estimates in §5 are
linear in $\varepsilon$, so they scale directly:

| | $\varepsilon\sqrt{L}$ (coeffs) | $\varepsilon L$ (Alg. 4 grid) | $\varepsilon L^2$ (new, grid) |
|---|---|---|---|
| $L=64$ | $5\times10^{-7}$ | $4\times10^{-6}$ | $2\times10^{-4}$ |
| $L=128$ | $7\times10^{-7}$ | $8\times10^{-6}$ | $1\times10^{-3}$ |
| $L=256$ | $1\times10^{-6}$ | $1.5\times10^{-5}$ | $4\times10^{-3}$ |

**Coefficient space is fine** (~$10^{-6}$), which is the floor a float32 iterative solve sits at
anyway. **Grid space is not**: 0.1–0.4% relative on the polar rings at $L\ge128$. For a
reaction–diffusion solver this matters only if grid-space $\Delta_\Gamma u$ is consumed outside the
matvec. If the IMEX splitting evaluates $f(u)$ from $u$ on the grid (typical), it never is.

**Two float32-specific arguments in favour of the new scheme:**

- Baseline SHT roundoff accumulates per transform ($\sim\varepsilon\sqrt{L}$ to $\varepsilon L$
  each). Running 6 transforms instead of 12 halves that accumulation. On the coefficient-space error
  GMRES actually sees, this plausibly outweighs the polar term — the new scheme may be *more*
  accurate end-to-end in float32. Not asserted without measurement.
- 3 weight arrays instead of 6 halves per-matvec texture/buffer traffic. On GPU that is often the
  real bottleneck, independent of arithmetic.

**Mitigations available without float64:**

- **Pairwise or blocked summation in the Legendre sum over $\ell$.** The single highest-value
  float32 change, and it benefits the existing code too. See §5b.
- **Double-float (`f32x2`) arithmetic** for the pointwise steps 3 and 6 if needed — cheap, no
  transforms involved. Does not help with transform roundoff, which is the dominant term, so try
  summation order first.
- **CPU precompute in float64.** JS `Number` is float64, so §3 can run on the CPU regardless of
  WebGPU's limits, with float32 weights uploaded. Cost is CPU-side SHTs plus upload, paid once per
  surface update; viable if the surface evolves slowly or is prescribed analytically, likely too
  slow if the metric is rebuilt every timestep. Per the correction below, this is probably
  unnecessary.
- **Cap $L$.** All the error terms grow with $L$; float32 sets a practical ceiling that float64
  would not.

---

## 5b. Summation order in the Legendre transform

This is orthogonal to the 12→6 change, applies equally to the current code, and in float32 is
probably worth more than the transform-count reduction. Do it first and independently, so its effect
can be measured on its own.

**Why.** Every $\varepsilon L$ and $\varepsilon L^2$ in §5 rides on the per-transform roundoff
floor, and in float32 that floor is set by *how the sums are accumulated*, not by the mathematics.
For each $(m, \theta_i)$ the synthesis evaluates

$$u^m(\theta_i) = \sum_{\ell=|m|}^{L} u^m_\ell\,\bar P^m_\ell(\cos\theta_i),$$

an $O(L)$-term sum. Error growth by accumulation strategy, for an $N$-term sum:

| Strategy | Worst case | Typical (random signs) |
|---|---|---|
| Sequential | $\varepsilon N$ | $\varepsilon\sqrt{N}$ |
| Pairwise / tree | $\varepsilon\log_2 N$ | $\varepsilon\sqrt{\log_2 N}$ |
| Kahan compensated | $\varepsilon$ (+ $O(\varepsilon^2 N)$) | $\varepsilon$ |

At $L=256$ in float32 that is the difference between $\sim1.5\times10^{-5}$ and $\sim5\times10^{-7}$
per transform — more than an order of magnitude, for no change in operation count.

**On GPU this may already be partly free.** A workgroup tree reduction over $\ell$ *is* pairwise
summation. The failure mode is a serial `for` loop over $\ell$ inside a single thread, which is the
natural way to write the shader if each thread owns one $(m,\theta_i)$ pair and is exactly the
$\varepsilon N$ row above. Check which shape the kernel has before assuming anything.

**Where it applies.**

- Synthesis $\mathcal{S}$: the sum over $\ell$, as above. The $\varphi$-direction FFT is already
  tree-structured and needs no attention.
- Analysis $\mathcal{A}$: the quadrature sum over latitude nodes $\theta_i$ carries the identical
  problem and the identical fix. It also matters more here, because this is the step relied on in
  §5 to suppress the polar spike — a noisy quadrature sum weakens exactly the mechanism the
  six-transform scheme depends on.

**Practical notes.**

- Blocked summation (accumulate in blocks of 8–32, then combine) captures most of the pairwise
  benefit with a simpler kernel and better register behaviour than a full tree.
- Kahan costs ~4 flops per term and is usually bandwidth-hidden on GPU; worth benchmarking rather
  than assuming it is too expensive.
- For $m>0$ near the poles, $\bar P^m_\ell(\cos\theta)$ spans many orders of magnitude across $\ell$.
  Summing smallest-magnitude-first helps, and is nearly free here because the terms are already
  roughly ordered by $\ell$.
- Standard stable recurrences for $\bar P^m_\ell$ (and guarding their under/overflow in float32's
  narrower exponent range) are a separate prerequisite — no summation strategy rescues inaccurate
  Legendre values.

**Measurement.** Transform a band-limited field forward then back and compare to the input, in
float32, sweeping $L\in\{64,128,256\}$. Sequential accumulation shows error growing roughly linearly
in $L$; pairwise shows near-flat growth. This isolates the transform floor from everything else in
§7 and should be run before the validation gate there, since it sets the baseline that gate is
measured against.

### Measured (2026-08-04, Dawn/Metal, `scripts/sht-accuracy.ts`)

The sweep was run and the summation-order changes tried. Outcome: **withdrawn — the floor here is
not summation-limited.**

| $L$ | grid | rel-$L_2$ roundtrip | worst degree |
|---|---|---|---|
| 63 | 64×128 | $3.4\times10^{-6}$ | $\ell=62$: $4.5\times10^{-6}$ |
| 127 | 128×256 | $4.7\times10^{-6}$ | $\ell=110$: $6.4\times10^{-6}$ |
| 255 | 256×512 | $1.1\times10^{-5}$ | $\ell=246$: $1.4\times10^{-5}$ |

- **The analysis side already sums pairwise.** The quadrature over latitudes is a workgroup
  tree/subgroup reduction (`leg_analys`); only the synthesis has the serial per-thread $\ell$-loop.
- **Kahan is unavailable on WebGPU in practice.** Dawn/Metal compiles WGSL with fast-math: a probe
  kernel evaluates $((10^8 + 1) - 10^8) - 1$ to $0$, so the compensation folds away and Kahan
  compiles to plain summation (bit-identical results, verified).
- **Blocked summation (B=16) in the synthesis $\ell$-loop moved nothing**: $1.128\times10^{-5}
  \to 1.128\times10^{-5}$ at $L=255$ (low digits shift, confirming the reordering was live), while
  costing ~5% per round trip at $L=255$. Reverted.
- **Diagnosis:** the worst error concentrates at the top degrees — the signature of the Legendre
  *recurrence* error (chains of length $\sim\ell$), not of $\ell$-uniform accumulation noise. This
  is the "standard stable recurrences are a separate prerequisite" caveat above: the floor is set
  by the accuracy of the $\bar P^m_\ell$ values themselves, and no summation strategy touches it.
- The measured floor ($\sim\varepsilon L^{0.85}$, $1.1\times10^{-5}$ at $L=255$) is what the §7
  validation gate should be read against.

---

## 6. Code changes

| Location | Change |
|---|---|
| `src/surface_gradient/partial_derivatives` | Expose $\mathcal{S}(v^m_\ell)$ **pre-division** (flag or separate entry point). Needed by both the precompute and step 1. |
| `SurfaceDiffOperator._precompute_metric_quantities()` | Return `p1, p2, q2, r` instead of `V_theta, V_phi`, per §3. |
| `src/surface_screened_laplacian::surface_screened_laplacian()` | Replace body with §4. Both `for i in {x,y,z}` loops disappear. |
| `SurfaceDiffOperator._precompute_curvature()`, Algorithm 3 | **Unchanged.** Still needs $X_{\theta\theta}, X_{\theta\varphi}, X_{\varphi\varphi}$ and the full Algorithm 1. |
| `src/timestepping::make_implicit_op()`, Algorithm 5 | **Unchanged.** Only what line 8 calls changes. |
| `src/real_embedding.py` | **Unchanged.** |

The deprecated `SurfaceDiffOperator` methods for $\Delta_\Gamma$ and $(I + c\Delta_\Gamma)$ are the
natural place to keep the old path as a reference implementation for the validation below.

---

### Correction: precompute conditioning

An earlier draft claimed the polar relative error in $\tilde g_{\theta\theta}$ is $\varepsilon L^2$,
making float64 precompute essential. That was wrong by a factor of $L$, in the safe direction.
$\tilde g_{\theta\theta}$ is not synthesized directly; it is the square of $\sin\theta\,X_\theta$,
which *is* synthesized, is $O(\sin\theta)$ at the poles, and carries absolute error $\varepsilon$ —
so relative error $\varepsilon L$, preserved (up to a factor 2) by squaring. Same for
$g_{\varphi\varphi}$ and $\tilde g_{\theta\varphi}$. The determinant combination is $O(\sin^4\theta)$
and so are both of its terms, so there is no extra cancellation generically; $J$ inherits
$\sim\varepsilon L$, i.e. $\sim10^{-5}$ at $L=128$ in float32. Acceptable.

Caveat: this assumes the difference is not small compared to its terms, which fails if $X_\theta$
becomes nearly parallel to $X_\varphi$ (near-degenerate parametrization). Worth a runtime check on
$\det g$ if the surface can deform that far.

Precompute error is also a *fixed* perturbation, identical every matvec, so it perturbs which
operator is being solved but injects no noise into the Krylov space — GMRES converges normally.

---

## 7. Validation, in order

At $\varepsilon=6\times10^{-8}$ there is no margin for the $\varepsilon\sqrt{L}$ suppression estimate
in §5 to be off by an order of magnitude. Step 2 is a **gate**, not a confirmation.

1. **Smoothness check (do this first).** On a deformed, non-axisymmetric surface, form $P$ and
   $\tilde{Q}$ on the grid and compare their SH coefficient decay against $(\nabla_\Gamma u)_x$ from
   the current code. Matching tails confirm both are genuinely smooth on $S^2$. If this fails,
   nothing else is worth doing. Run this in float64 on CPU — it is a mathematical check, not a
   precision one.
2. **Coefficient-space diff in float32 at production $L$**, against a float64 CPU reference
   implementation of Algorithm 4. Landing near $10^{-6}$ means the suppression argument holds.
   Landing near $10^{-4}$ means the polar spike is surviving the analysis and the polar rings need
   separate handling.
3. **Sweep $L \in \{64,128,256\}$** and fit the growth exponent of (2). Flat-ish confirms
   suppression; growth like $L^2$ means it is not working.
4. **Grid-space max-norm diff near the poles.** Expect the extra power of $L$ here. If only this
   grows and (2) stays flat, the scheme is fine for use inside the implicit solve.
5. **GMRES iteration count and final achieved residual**, float32, versus the current code. The
   operator is the same, so iterations should be unchanged; a stall above tolerance that does not
   occur in float64 indicates the matvec noise floor is binding.

---

## 8. Suggestions considered and withdrawn

- **Splitting $r$ across steps 3 and 6** to keep $\sin^1$ scaling. Does not work: $P/\sin\theta$ is
  not smooth on $S^2$ (round sphere, $u = x$: $P = \sin\theta\cos\theta\cos\varphi$, so
  $P/\sin\theta \to \cos\varphi$ at the pole). That $\sin\theta$ must stay in $r$. Algorithm 4 *can*
  split because its intermediate — the Cartesian gradient — is smooth; that smoothness is precisely
  what the six extra transforms buy.
- **Weighting by $J$ to get an SPD operator and use PCG.** Avoiding the $1/\sqrt{g}$ makes
  $\mathcal{L}$ self-adjoint, but the mass term becomes multiplication by $J$, costing its own
  synthesis/analysis pair. A wash on transform count; worth it only if the CG properties themselves
  are wanted. Discrete symmetry would also hold only to quadrature accuracy unless products are
  dealiased (3/2 rule).
- **Mixed precision (float64 for step 6's synthesis and the $r$ multiply only).** Unavailable:
  WebGPU is float32-only. Superseded by the mitigations in §5a.
- If the $\varepsilon L^2$ ever does bind, the remaining remedies are a shifted or uniform-in-$\theta$
  latitude grid (removing the $O(L^{-2})$ node clustering) or a separate local formula for the polar
  rings — both more work than the transform savings justify without a specific reason. In float32
  the ceiling on $L$ may bind first and be the cheaper accommodation.

## 9. Related: external vector transforms

Steps 1–2 and 4–5 together are a vector/spin-weighted spherical harmonic transform. If SHTns
(`spat_to_SHsphtor`, `SHsphtor_to_spat`) or SPHEREPACK (`gradgs`, `divgs`) can be linked, the
gradient and divergence each become a single library call, the pole divisions are handled internally,
and the hand-rolled $\alpha^\pm$ recurrences on this path are no longer needed.

## 10. Beyond transform count

The other lever is iteration count rather than cost per iteration. Since $M^{-1}A$ approaches
multiplication by $1/J$ at high $\ell$, folding a mean or smoothed $J$ into the preconditioner could
reduce GMRES iterations by more than any of the above reduces transforms.
