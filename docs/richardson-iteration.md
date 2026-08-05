# Notes on this project's Richardson iteration, for readers of `algos.tex`

## Why this exists

`evolving_surface/notes/algos.tex` (Sec. 5, "Implicit timestepping and the
linear solve") specifies the surface diffusion step as backward Euler,
`(I - Δt Δ_Γ) u^{n+1} = u^n`, solved by preconditioned GMRES: the
round-sphere Laplacian `M = I - Δt Δ_S` (diagonal, invertible by eigenvalue)
preconditions the full operator, the real-embedding map `E` puts the
half-spectrum complex coefficients into a real vector space, and restarted
GMRES iterates to a residual tolerance.

turing-surface (this project) solves the *same* split operator with a
different numerical method: a preconditioned Richardson (fixed-point)
iteration, reusing exactly algos.tex's preconditioner `M^{-1}` but with no
Krylov subspace, no orthogonalization, and no adaptive stopping. This note
gives the map between the two, in this project's variable names, and why the
switch.

## Notation map

| algos.tex | this project | meaning |
|---|---|---|
| `u^n`, `u^{n+1}` | `U`/`V` (in), `Un`/`Vn` (out) | spectral state, one array per species |
| `Δ_Γ` | `lap_g` | the surface's Laplace-Beltrami operator |
| `Δ_S` | `lap_s` | the round sphere's operator, eigenvalue `-l(l+1)` |
| — | `dlap` | `lap_g - lap_s`. algos.tex has no name for this because it never splits the operator this way — its GMRES matvec (`surface_screened_laplacian`) applies the *whole* `Δ_Γ` every iteration. |
| `M = I - Δt Δ_S` | `(1 + dt*D*lam)` | the same preconditioner. `lam` holds `+l(l+1)`, not `-l(l+1)`, so it enters as a *sum* — the sign flip is already folded into `lam`. |
| `M^{-1}v` (eq. `preconditioner_inverse`) | `v ./ (1 + dt*D*lam)` | the identical elementwise divide |
| a GMRES iterate | `Un^(k)`, `k = 0..niter` | *not* a Krylov iterate — a fresh, full re-solve of the fixed point below, evaluated at the previous iterate |

## The fixed point this project actually iterates

Same split as algos.tex, `lap_g = lap_s + dlap`, substituted into backward
Euler and rearranged so every occurrence of the unknown is `Un`. Starting
from `(I - dt*D*lap_g) Un = B` and substituting the split:

```
(I - dt*D*(lap_s + dlap)) Un = B
```

Expanding, and moving the `dlap` term to the right so only the exactly
invertible round-sphere part remains on the left:

```
Un - dt*D*lap_s(Un) = B + dt*D*dlap(Un)
```

`lap_s` is diagonal with eigenvalue `-l(l+1)`, and `lam` holds `+l(l+1)`, so
`lap_s(Un) = -lam .* Un` — the left side becomes `Un .* (1 + dt*D*lam)`, and
dividing through gives:

```
Un = (B + dt*D*dlap(Un)) ./ (1 + dt*D*lam)
```

`B` is the explicit-reaction right-hand side — this project's models are
IMEX (explicit reaction, implicit diffusion), where algos.tex's worked
example is the bare heat equation, so `B` here is `u^n` plus a reaction term.
Richardson iteration on this fixed point:

```
Un^(0)   = B ./ (1 + dt*D*lam)                                  [dlap = 0]
Un^(k+1) = (B + dt*D*dlap(Un^(k))) ./ (1 + dt*D*lam)
```

for `k = 0 .. niter-1`. `models/schnakenberg.m`'s `for k = 1:niter` loop *is*
this: `Un^(0)` is the divide computed just before the loop, and each pass
computes `Un^(k+1)` from `Un^(k)`. It is written as a full re-evaluation
rather than an accumulated correction `δ = Un^(k+1) - Un^(k)` on purpose: at
`dlap ≡ 0` (the round sphere), every `Un^(k)` is then bit-for-bit `Un^(0)`,
with no cancellation to round differently — a stronger, and cheaper to
check, statement than "close to the round-sphere answer."

## Convergence, and why it isn't GMRES

Writing `M = I - dt*D*lap_s` and `A = M - dt*D*dlap`, each step is
`Un^(k+1) = M^{-1}(B + dt*D*dlap(Un^(k)))` — a stationary iteration that
converges to the exact solution of `A·Un = B` exactly when the spectral
radius of `M^{-1}(dt*D*dlap)` is below 1: while the geometric correction
stays small against what the round-sphere solve already inverts. Unlike
GMRES, there is no residual check and no adaptive iteration count: `niter` is
fixed before the run starts, so a shape/timestep/diffusivity combination
outside the convergence radius fails silently — the state saturates or
diverges over many steps — rather than being caught the way algos.tex's
`solve_step` catches it (its `info != 0` return, logged when GMRES fails to
reach `tol` within `maxiter`).

**Update (symbol-based preconditioning).** The models now precondition
with `M = I + dt*D*lam/jhat` where `jhat = 2/(muMin + muMax)` is the host's
minimax scale over the eigenvalues of the operator's principal symbol — the
inverse squared principal stretches of the embedding, direction included
(see docs/reduced-transforms.md Sec 10). At high degree the iteration then
contracts at rate `(muMax - muMin)/(muMax + muMin) < 1` on any surface,
where the plain `M` diverges wherever `mu > 2` — which is what used to put
peanut outside the convergence radius at niter >= 2, and what made patterns
drift high-frequency on the ellipsoid as niter or lmax grew. The correction
is also projected onto the band (`.* filt` on `dLu`, algos.tex Algorithm
5's zeroing), without which the top two degrees iterate toward an
undiffused fixed point. The silent-failure caveat above still stands for
what a constant scale cannot capture (strong *spatial* variation of the
symbol at low degree, or `dt*D` beyond the correction's reach), but the
sweep's previously divergent cases all converge now, and `jhat: 1`
reproduces the old behavior for A/B.

That tradeoff is deliberate, not an oversight, and it comes from where the
two projects run. algos.tex's GMRES needs, every iteration: a dot product
across the whole spectral state (Arnoldi orthogonalization) and a residual
norm to test against `tol` — both require reading a scalar back to the host
mid-solve. This project's solver instead records one whole timestep as a
single GPU command buffer, submitted once, with the entire `for k = 1:niter`
loop unrolled at compile time into a fixed sequence of dispatches — there is
no point in that sequence where the host makes a decision, and no path for a
data-dependent stopping rule to plug in. (Recompiling — which changing
`niter` triggers — is the only way this project can change how much work a
step does; see the README's "`for` loops, unrolled".) Richardson iteration is
the cheapest method that still fits that shape: the same preconditioner as
algos.tex, one `dlap` evaluation per iteration, a fixed and
recompile-on-change trip count, in exchange for linear rather than
superlinear convergence.

## One more difference worth flagging

algos.tex maps the half-spectrum complex coefficients through a real vector
space embedding `E` (Sec. 6.4) because GMRES needs one flat, real-linear
operator to hand to a generic solver. This project never needs `E`/`E^{-1}`:
its spectral state is *already* carried as a real "2 x nlm" array — row 0 the
real part, row 1 the imaginary — rather than packed complex, so every step
here, `dlap` included, is already ℝ-linear arithmetic on that layout with no
embedding or un-embedding step at all.
