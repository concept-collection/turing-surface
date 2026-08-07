# turing-surface

Reaction–diffusion systems (Turing patterns) on **closed surfaces given by
spherical-harmonic embeddings**, solved live in the browser with a spectral
method whose transforms run on the GPU via WebGPU.

This is the sibling of
[turing-sphere](https://github.com/concept-collection/turing-sphere), which
solves the same systems on the round sphere. Everything there is here; what is
added is a *surface*.

The geometry is in the operator: the models evaluate the surface
Laplace–Beltrami operator `lap_g` inside the implicit solve, in a **flux form
that costs 5 spherical-harmonic transforms per species per iteration** (plus
one Legendre-free FFT derivative) where the textbook Cartesian-gradient form
needs 12. See
[The geometry in the operator](#the-geometry-in-the-operator) and
[docs/reduced-transforms.md](docs/reduced-transforms.md).

## What a surface is here

A geometry is an embedding of the sphere into R³: three scalar fields x, y, z
over the (θ, φ) parametrization, each carried as spherical-harmonic
coefficients. The unit sphere is the case where all three are pure degree-1
harmonics.

You write one down as MATLAB, in [`geometries/`](geometries/):

```matlab
function [gx, gy, gz] = shape(theta, phi, waist, stretch)
  st = sin(theta);
  r = 1 - waist * (st .^ 2);
  gx = r .* (st .* cos(phi));
  gy = r .* (st .* sin(phi));
  gz = (1 + stretch) * (r .* cos(theta));
end
```

That is ordinary MATLAB. Unlike the models it is not compiled to WGSL: a shape
is evaluated exactly once at build time, so it runs through numbl's CPU
interpreter instead, in f64, with the full MATLAB subset available — loops,
arrays, `min`/`max`, `legendre`, seeded randomness via `rng`/`randn`. The
result is then **analysed into coefficients**, which is the form everything
downstream uses. Two things follow from going through the coefficients rather than keeping
the pointwise values:

- **It is exactly band-limited at lmax.** The surface has as many derivatives as
  the scheme needs and no aliased content the solver cannot see. What the solver
  and the renderer both use is the *synthesis* of the coefficients, so for a
  shape with sharp features the surface being solved on is not quite the one
  that was written down — which is the honest thing for a spectral method to do.
- **It can be evaluated on any grid.** The renderer draws the surface on the
  (possibly finer) display grid by synthesizing the same coefficients there.
  That is exact interpolation, not subdivision — the same argument that lets the
  species fields be oversampled, and it is checked directly in the tests.

Five geometries ship: [sphere](geometries/sphere.m) (the reference case),
[ellipsoid](geometries/ellipsoid.m), [peanut](geometries/peanut.m) — a dumbbell
whose waist is a saddle — [bumpy](geometries/bumpy.m), and one random one:
[blob](geometries/blob.m), surfacefun's blob — the sphere warped by a smooth
random function built from chebfun's `randnfunsphere` construction (random
spherical-harmonic coefficients up to degree ⌊2π/λ⌋, rescaled to [−1, 1]).
It is seeded, so the same seed always gives the same shape; `amp` sets how far
it departs from the sphere, `λ` how fine its lobes are, and **Re-seed shape**
draws another one. Each geometry is
editable in the page, with its own parameters. Changing a shape does not recompile the
solver and does not disturb the run: the geometry is data whose shape in the
bindings depends only on the grid, so a swap is sixteen buffer writes and the
pattern carries straight on.

A **morph** slider blends the drawn surface back to the unit sphere. The
parametrization is the sphere's either way, so sweeping it shows which point
went where.

### Seeding, and `tools/`

A run starts from the uniform steady state plus a small perturbation, and that
perturbation is a *smooth* random field rather than white noise: chebfun's
[`randnfun3`](tools/randnfun3.m) on the surface's bounding box, restricted to
the surface by evaluating it at the grid points — the way surfacefun seeds a
run. Each model's `init` says so itself:

```matlab
function [U, V, u, v] = init(lam3, gx, gy, gz, a, b)
  f = randnfun3(lam3, gx, gy, gz);
  ...
```

A band-limited seed is fully resolved by the grid, where white noise is
whatever the grid happened to alias: the tests measure its energy above degree
20 at 5e-14 of the total, and the flux-form and Algorithm-4 operators now
track each other to 3e-6 through a run instead of 4e-4. The **seed λ** control
sets the field's wavelength; smaller means finer features to grow from. It is
an *absolute* length in the surface's own units, as in chebfun — not a
fraction of the surface's size — so a larger surface draws more modes at the
same λ.

**λ is useful down to about 2π/lmax, and no further.** A field of wavelength λ
on a unit-radius surface carries angular content up to degree ≈ 2π/λ, so at
the default lmax 63 the grid holds everything down to λ ≈ 0.1. Past that,
`init`'s own `analys` discards what the grid cannot represent, and the seed
gets *weaker* rather than finer while costing eight times as much per halving:

| λ | 2π/λ | rms of the resolved seed | energy above l=55 | peak degree |
|---|---|---|---|---|
| 0.5 | 13 | 2.6e-2 | 1e-8 | 8 |
| 0.2 | 31 | 2.6e-2 | 1e-8 | 10 |
| 0.1 | 63 | 2.4e-2 | 0.10 | 44 |
| 0.05 | 126 | 1.5e-2 | 0.20 | 48 |
| 0.03 | 209 | 9.6e-3 | 0.27 | 63 |

Raising lmax moves that floor down, and the seed really does get finer: at
lmax 127 the same λ=0.05 keeps its full amplitude (2.5e-2 against 1.5e-2 at
lmax 63) with its peak at degree 79 instead of pinned to the band edge, and
λ=0.1 becomes *fully* resolved (2e-8 of its energy in the top decile, against
1e-1 at lmax 63 — so even 0.1 is slightly under-resolved on the default grid).

Note that lmax cuts both ways: it quadruples npts, so every λ also costs four
times as much to sum.

**Nothing caps λ but memory and patience.** The mode table grows to whatever
is asked for and the only refusal is a table that could not be built at all,
reported with the mode count it wanted rather than silently truncated. On a
128×256 grid:

| λ | modes | seed time |
|---|---|---|
| 0.05 | 480,431 | 0.26 s |
| 0.03 | 2,094,657 | 0.98 s |
| 0.02 | 6,882,185 | 3.1 s |
| 0.015 | 16,092,829 | 7.4 s |
| 0.01 | 53,574,764 | 25.6 s |

Being slow is the caller's business; **freezing the browser is not**, and at
these times neither half of the work can be left where it was:

- The draw is synchronous interpreter time — 13 s at λ=0.01 — which on the
  main thread stops the page painting and gets it offered up for killing. It
  runs on a worker instead
  ([`randnfun3.worker.ts`](src/mgpu/randnfun3.worker.ts)); it touches no GPU
  and no DOM, so nothing about it needed that thread. Measured during a seed:
  731 animation frames, no stalled sample.
- The GPU sum is split across a fixed 16 dispatches (`randnfun3Chunks`)
  accumulating into the same output, and `submitYielding` ends the submission
  at each one. A browser's GPU process is shared with compositing, so a single
  submission running tens of seconds stops *every* tab painting, and one
  dispatch that long risks the watchdog killing the device outright. Slices
  past the end of a small table exit immediately, so a coarse λ pays nothing.

The device is also asked for the adapter's full storage-buffer limit at
creation ([`src/sht/sht.ts`](src/sht/sht.ts)), so a browser's 128 MB default
is not what decides how fine λ can be. `seed()` is consequently async.

`randnfun3` splits across the CPU/GPU line, and the split is forced rather
than chosen. Drawing the modes needs `randn` and a `sqrt(nnz)` normalization,
neither of which exists in the compiled WGSL dialect, so the draw is MATLAB in
[`tools/randnfun3.m`](tools/randnfun3.m) run by the interpreter — a few
thousand coefficients, ~5 ms. Evaluating is `npts × nmodes` (~6e7 terms at the
default λ), so that is a WGSL kernel
([`src/mgpu/randnfun3.ts`](src/mgpu/randnfun3.ts)) reached as an external
operation, the way `synth` is. The coefficient table is filled in behind the
call, as `synth` hides its Legendre matrices; λ is not hidden, and the plan
records which parameter the `.m` asked with so the host draws from that value.

[`tools/`](tools/) is the shared MATLAB every interpreter run can call, by file
name, as on MATLAB's path — currently `randnfun3` and
[`randnfunsphere`](tools/randnfunsphere.m), which `blob.m` is written on. Both
keep their upstream signatures, including options nothing shipped uses yet
(`randnfunsphere`'s `'monochromatic'`), because the point of a tool is that a
geometry you write next can reach for it. Tools are not available to the
models' *step*, which compiles to WGSL where none of this exists.

## The scheme, and where the geometry enters

It solves the N-species system

```
d(u_k)/dt = D_k*lap_g(u_k) + f_k(t, u_1, ..., u_N),    k = 1, ..., N
```

where `lap_g` is the Laplace–Beltrami operator of the surface. On the round
sphere `lap_g` is diagonal in spherical-harmonic space with eigenvalues
`-l(l+1)`, which is what makes turing-sphere's implicit diffusion a single
divide. On a general surface it is not diagonal, and not even constant-
coefficient, so that divide has to become a solve.

The models split the operator:

```
lap_g = lap_s + dlap
```

with `lap_s` the round-sphere one. `(I - dt*D*lap_s)` is still exactly
invertible, so the implicit step

```
(I - dt*D*lap_g) Unew = B
```

rearranges into a fixed point that keeps the whole geometry on the right-hand
side,

```
Unew = (B + dt*D*dlap(Unew)) ./ (1 + dt*D*lam)
```

and the loop iterates it from the round-sphere answer. That is preconditioned
Richardson, with the operator we can invert exactly as the preconditioner; it
converges while `dt*D*dlap` stays small against `(I - dt*D*lap_s)`, which is
what keeps the cost to a few transforms per step rather than a full elliptic
solve (see [docs/richardson-iteration.md](docs/richardson-iteration.md)). One
species of [`models/schnakenberg.m`](models/schnakenberg.m)'s solve loop:

```matlab
lamJ = lam ./ jhat;             % mean-J preconditioner eigenvalues (below)
...
for k = 1:niter
  Fu = Un .* filt;              % zero the top 2 degrees before differentiating
  vtu = dthetac(Fu);
  vpu = dphic(Fu);
  [Ftu, Fpu] = synth(vtu, vpu); % sin(theta)*dtheta(u), dphi(u) -- smooth on
                                % the sphere, one batched dispatch
  Pu = p1 .* Ftu + p2 .* Fpu;   % the two fluxes, also smooth: the precomputed
  Qu = p2 .* Ftu + q2 .* Fpu;   % weights carry every 1/sin(theta) there is
  PAu = analys(Pu);
  Pcu = PAu .* filt;
  scu = dthetac(Pcu);           % theta part of the divergence, coefficients
  Lu = synth(scu);              % sin(theta) * dtheta(P) on the grid
  dQu = dphig(Qu);              % d/dphi is diagonal in the Fourier index:
                                % two FFT stages, no Legendre work at all
  lapu = r .* (Lu + dQu);       % = lap_g(u) on the grid
  dLu = (analys(lapu) + lamJ .* Un) .* filt;   % dlap, projected onto the band
  Un = (Bu + (dt * D1) * dLu) ./ (1 + (dt * D1) * lamJ);
end
```

On the sphere `dlap` is mathematically zero — `p1 = q2 = 1`, `p2 = 0`,
`r = 1/sin²θ`, `jhat = 1`, and the composition collapses to `lap_s` — so the
sphere case reproduces turing-sphere to fp32 round-off, and the tests assert
the state stays put across 0, 1 and 4 iterations.

**The preconditioner folds in the symbol of the operator.** `jhat` is the
host's minimax scale `2/(μmin + μmax)` over the eigenvalues `μ(x)` of the
operator's principal symbol — the inverse squared principal stretches of
the embedding, direction included, read straight off the flux-metric
arrays (`S = (1/J)·[[p1,p2],[p2,q2]]`). Preconditioning with `lam/jhat`
then contracts every mode *and every direction* at rate
`(μmax − μmin)/(μmax + μmin) < 1` on any surface, where the plain `lam`
diverges wherever `μ > 2` — peanut reaches `μ = 6.2`. A det-based mean of
the area factor (μ's geometric mean, exact only for conformal surfaces) is
not enough: it under-corrects anisotropic stretching and leaves directional
high-degree bands with amplification > 1, which surfaced as patterns going
high-frequency and diverging as `niter` or `lmax` grew. The answer never
depends on `jhat` — the `lamJ` term added inside `dLu` is the term divided
back out — only the convergence rate does.

**The correction is projected onto the band** (`.* filt` on `dLu`,
matching algos.tex Algorithm 5's zeroing of the top coefficients). Without
it the top two degrees iterate toward the *undiffused* `Bu` — each solve
iteration strips a bit more of their implicit diffusion, at species-
dependent rates, which manufactures a spurious Turing band at the band
edge: visible on the round sphere as top-degree energy growing ~3%/step at
`lmax 127, niter 8`. With both fixes the whole niter × geometry sweep
converges, the spectral centroid of the pattern is resolution-independent
(l ≈ 26 at lmax 63 and 127 alike), and `jhat: 1` is kept as the divergent
control in the tests.

### The geometry in the operator

`dlap = lap_g - lap_s` is applied to the current iterate at every solve
iteration, so its transform count is what the whole step's cost scales with.
Two formulations ship:

1. **The flux form** (above, all three models): `lap_g u` as the weighted
   divergence of two weighted fluxes of the sin-scaled derivatives. The
   weights `p2, r, dp1, dq2, jinv` are grid arrays precomputed once per
   surface from the embedding's θ/φ tangents
   ([`src/geom/metric.ts`](src/geom/metric.ts)), chosen so that **every field
   that gets analysed is a smooth function on the sphere** — the property
   that makes spherical-harmonic analysis meaningful, and the entire
   difficulty near the poles. The divergence is split against the round
   sphere: the sphere's share of it is `-jinv .* lap_s(u)`, exact in
   spectral space, so `r ~ 1/sin²θ` multiplies only the geometry deviation.
   Without that split `r` amplifies the polar round-off of the whole flux
   into a static forcing that nucleates a spot at the pole on every seed.
   Cost: **6 Legendre transforms** per species per iteration (4 syntheses —
   two gradient, one divergence, one for the sphere's `-lam .* u`, which
   rides in the gradient's batch — plus 2 analyses; the phi flux never needs
   the Legendre basis, `dphig` differentiates it on the grid with two FFT
   stages, masking m past the top-degree filter, and `dthetac`/`dphic`
   are O(nlm) coefficient shuffles). The derivation, the smoothness
   argument and the fp32 error analysis are in
   [docs/reduced-transforms.md](docs/reduced-transforms.md).
2. **The Cartesian-gradient form** (Algorithm 4 of `docs/algos.pdf`), kept as
   a live reference in
   [`models/schnakenberg_alg4.m`](models/schnakenberg_alg4.m) and selectable
   in the app: the surface gradient carried as three ambient components
   through the inverse metric quantities `Vt*/Vp*`. Cost: **12 transforms**
   per species per iteration. The tests hold both forms to the same answer on
   a curved surface, and both metric formulations are precomputed and
   uploaded for every geometry, so either kind of model runs.

The θ-derivative machinery both forms need — the α± recurrence
(`sin θ ∂θ Y_l^m = α⁺Y_{l+1}^m + α⁻Y_{l-1}^m`) as a coefficient-space shuffle
feeding the existing scalar synthesis — lives in
[`src/sht/deriv.ts`](src/sht/deriv.ts); no Legendre-derivative tables are
required.

### `for` loops, unrolled

A plan is a fixed list of GPU operations with no branching, which is what makes
a timestep pure command recording — one submit, no CPU in the loop. A counted
loop still fits: the planner
([`src/mgpu/plan.ts`](src/mgpu/plan.ts)) unrolls it, planning the body once per
iteration.

Nothing else had to change for that, because numbl gives a variable one cName
for every assignment to it: the buffer an iteration writes is the buffer the
next one reads, which is exactly a loop-carried value. The loop variable gets no
buffer at all — it is bound as a derived scalar to that iteration's literal, so
a kernel reading `k` folds the number in.

Two consequences worth stating:

- **The bounds must be known when the model compiles.** `niter` is supplied as a
  fixed scalar rather than a tunable one, so changing it recompiles — unlike a
  parameter, which is a uniform. A runtime bound is refused at compile time with
  a source position, not silently mis-compiled, and there is a test for that.
- **Fusion survives.** numbl's inline pass recurses into loop bodies, so a line
  inside the loop is still one kernel. It runs there with no protected names,
  though, which means an assignment whose only visible use is later in the same
  body can be elided — correct for a body-local temp, wrong if something outside
  the loop wanted it. [`src/mgpu/compile.ts`](src/mgpu/compile.ts) snapshots what
  each loop body assigns before the pass and refuses the ones that escape, so
  that case is a compile error rather than a stale read.

Unrolling is exactly linear in the trip count: 18 GPU ops per species per
iteration (7 transforms, 3 coefficient shuffles, 8 kernels), asserted in the
tests.

## MATLAB, compiled to WebGPU

Unchanged from turing-sphere. This is the models' path — the geometry files
instead run once through numbl's CPU interpreter, as above. numbl
parses and lowers each function for the concrete argument types of the current
grid; its inline pass folds single-use temps back into their consumer, so one
line of MATLAB becomes one expression tree; and this repo emits one WGSL compute
kernel per element-wise statement
([`src/mgpu/wgsl.ts`](src/mgpu/wgsl.ts)). `synth` / `analys` are external
operations whose type rules numbl learns from a `.mtoc2.js` workspace file, and
which the backend maps onto the spherical-harmonic pipelines. Anything it cannot
express is refused at compile time with a source position.

The Schnakenberg step compiles to 50 GPU operations at one solve iteration:
18 transforms, 6 coefficient-space shuffles, 24 generated kernels, and 2
buffer copies feeding the new state back.

**Transforms batch.** The expensive part of every Legendre stage is
generating the associated Legendre values on the fly by recurrence — work
that depends only on the grid, not on the field. `synth`/`analys` therefore
take multiple fields, and a grouped call runs as one batched dispatch: one
walk of the recurrence, one accumulator lane per field —

```matlab
[Ftu, Fpu, Ftv, Fpv, Su, Sv] = synth(vtu, vpu, vtv, vpv, lam .* Fu, lam .* Fv);
```

The grouping is a promise of independence, never of a lane width: the
planner ([`src/mgpu/plan.ts`](src/mgpu/plan.ts), `materializeTransforms`)
chunks each group into whatever the device supports — one ×4 batch under the
default WebGPU limits, or scalar dispatches with `SHT_BATCH=0` for A/B — so
the same source runs anywhere. Ungrouped transforms that happen to sit on
consecutive independent lines are batched the same way. Per-lane arithmetic
is identical to the scalar kernels', so batched and scalar plans produce
bit-identical states, asserted in the tests along with compile-time refusal
of a group that drops one of its outputs. All 16 Legendre transforms of the step
above land in batches, worth ~25% of the whole step (0.88 vs 1.14 ms/step at
lmax 127, 2 iterations, on bumpy).

Two consequences carried over:

- **The step is synchronous.** WebGPU's encode path is synchronous and every
  pipeline is built once at compile time, so a timestep is pure command
  recording; the only `await` in the loop is the single readback per rendered
  frame.
- **Parameters are uniforms, not constants.** Moving a slider rewrites a small
  buffer instead of triggering a recompile. Editing the MATLAB recompiles;
  changing `dt` does not. `niter` is the deliberate exception, above.

## Provenance

- **turing-sphere**, which this is a fork of: the solver, the transforms
  backend, the compilation path, the benchmarks and the analytic tests.
- **Transforms:** [shtns-webgpu](https://github.com/concept-collection/shtns-webgpu) —
  fp32 spherical harmonic transforms in WGSL compute shaders, modeled on
  [SHTNS](https://nschaeff.bitbucket.io/shtns/). Vendored under
  [`src/sht/`](src/sht/) (CECILL-2.1), including the f64 CPU reference transform
  used for testing.
- **Rendering:** three.js meshes with per-vertex colormaps, adapted from the
  `SphereEmbedding` view in
  [figpack](https://github.com/flatironinstitute/figpack)'s experimental
  extension package ([`src/render/`](src/render/)). That view displays a
  time-varying embedded geometry with fields on it, which is the same picture
  this draws — including its sphere/surface morph, which turing-sphere had
  dropped as having nothing to morph to.

turing-sphere additionally carries a comparison against a native build of
upstream SHTNS ([`bench/shtns/`](https://github.com/concept-collection/turing-sphere/tree/main/bench/shtns)).
That is not duplicated here: the transforms are the same code, and its C-side
transcription of the model would have to be maintained against a step this
project intends to change.

Because the algorithm is compiled to compute shaders, **WebGPU is required** —
there is no CPU fallback (the f64 CPU transform remains, for tests).

## Numerics

- Grid: Gauss–Legendre × equispaced-φ, dealiased for the cubic reactions with
  the `(pdeg+1)` rule: `nlat ≥ ((pdeg+1)·lmax+1)/2`, `nphi ≥ (pdeg+1)·lmax+1`
  (rounded up to a power of two for the GPU FFT path). At the default lmax 63
  that is a 128×256 grid.
- Spectral layout: SHTNS conventions — orthonormal + Condon–Shortley, complex
  coefficients for m ≥ 0, m-major ordering.
- fp32 transforms introduce ~1e-6 relative error per step; for pattern formation
  from a 1e-2 seeded perturbation this is inconsequential. The geometry goes through one
  analysis/synthesis round trip and picks up the same round-off: the unit sphere
  comes back with radius 1 to ~2e-5 under Dawn, ~4e-4 under SwiftShader.
- The shipped analytic geometries are all degree ≤ 5, and the random ones stay
  near degree 13 at their finest slider settings — far below any lmax the app
  offers, so band-limiting removes little to nothing from them. A shape you
  write yourself may not be so lucky — see the note in
  [`geometries/bumpy.m`](geometries/bumpy.m).

## Desktop vs browser

[`scripts/bench.ts`](scripts/bench.ts) runs the same thing the app runs — same
`.m`, same generated WGSL, same transforms — from Node on desktop WebGPU (Google
Dawn), and the app prints the command line that reproduces whatever it is
currently simulating:

```
npm run bench -- --preset schnak-spots --geometry ellipsoid --lmax 63 --niter 1 \
  --steps 2000 --seed 1 --a 0.1 --b 0.9 --D1 0.0004 --D2 0.008 --dt 0.05 \
  --gax 1.5 --gay 1 --gaz 0.6
```

Copy it from under the stats line and compare the `ms/step` it reports with the
app's. Both sides go through the one shared
[`src/bench/runSpec.ts`](src/bench/runSpec.ts) — the app formats a run into that
command, the benchmark parses it back — so there is no second copy of the
defaults for the two runs to drift apart on. Geometry parameters take a `g`
prefix (`--gwaist`) so a shape parameter can never collide with a model one.

The app reports **two** numbers and only the first is comparable to the
benchmark: `solver` is the batch of steps alone, waited for but not read back;
`ms/frame` additionally carries a GPU→CPU readback per species, the
colormapping, and the vertex upload. Those per-frame costs are fixed and do not
shrink when the GPU gets faster, so on a quick GPU a frame can easily cost ten
times the steps inside it. That is expected and is not the solver being slower
in the browser.

To attribute the gap rather than guess at it:

```
node scripts/compare-perf.mjs [--lmax 63] [--steps 300]
```

measures the same solver work in both — batched, nothing read back, no rendering
on either side — and reports each with its CPU-encoding share, the Fourier
stage, and the adapter. It stops you first if the two are not even the same
device, which is a common cause of "the browser is much slower". Both sides
resolve the geometry and the iteration count from the same constants, because
the iteration count is unrolled into the step and a mismatch would compare two
different amounts of work.

The app's **Benchmark** button runs the same measurement in the page, plus the
**ramp** — the first third of the run against the last. GPUs downclock when
idle and an animation-paced loop leaves them idle most of every frame, so a
large ramp means the steady-state number is limited by clocks rather than work.

### Is it really the same computation?

```
node scripts/compare-env.mjs [--lmax 31] [--steps 200] [--preset schnak-spots]
```

runs one identical spec on the desktop and in a real browser and compares the
final spectral state. The pipeline is deterministic given (model source,
geometry, parameters, lmax, niter, seed, steps), so the two should agree to fp32
round-off — not bit for bit, since GPUs differ in fused-multiply-add and other
latitude fp32 allows. It also reports which Fourier stage each side chose, since
FFT and DFT are genuinely different algorithms that round differently.

Desktop WebGPU comes from the `webgpu` package (prebuilt Dawn, ~70 MB), an
optional dependency so that an unsupported platform fails the install of that
package alone. Its binaries need glibc 2.29+. Other flags: `--steps`,
`--warmup`, `--batch`, `--json`, `--help`; `DAWN_FLAGS='backend=vulkan'`
(`;`-separated) passes Dawn options through.

## Tests

There is no second implementation of the solver to diff against, so the `.m`
path is checked against **closed-form answers** and against **exact structural
properties**. Five modules, run in both environments:

[`test/analyticChecks.ts`](test/analyticChecks.ts) — cases whose evolution is
known exactly, run through the whole real pipeline. All three are statements
about the round sphere, so all three build on the sphere geometry:

- **A** — a linear reaction leaves every mode independent, growing by exactly
  `(1 + dt*c) / (1 + dt*D*l(l+1))` per step. Pins the transform round trip, the
  eigenvalue mapping, the IMEX update and the state feedback at once. ~2e-7 over
  20 steps.
- **B** — a nonlinear reaction on a uniform field stays uniform, so each step is
  exactly the scalar ODE map. 1.5e-8 over 25 steps.
- **C** — a 1e-6 perturbation of the Schnakenberg fixed point follows the
  linearized 2×2 IMEX recurrence, and `(l=24, m=7)` is confirmed unstable.
  Looser (~4e-3) because fp32 keeps about four digits of a perturbation that
  small.

[`test/geometryChecks.ts`](test/geometryChecks.ts) — the surface, the loop, and
the seed:

- every geometry evaluates and closes; the sphere has radius 1 everywhere and is
  **exactly degree 1** in the harmonics, which is what makes the reference case
  exact rather than merely accurate;
- the peanut matches its own closed-form radial profile at every grid point, and
  **the same coefficients give the same surface on a 2× grid** — the 2× Gauss
  latitudes share no point with the 1× ones, so agreeing there is agreeing
  everywhere, which is what "rendered exactly, not subdivided" means;
- unrolling is **exactly linear** in the trip count, and on the sphere — where
  the geometric correction is mathematically zero — the state after 20 steps
  stays within fp32 round-off of the 0-iteration one at 1 and 4 iterations;
- a runtime loop bound is refused at compile time;
- swapping the surface mid-run leaves the spectral state untouched;
- the seed field's **WGSL sum matches the same modes summed in f64 on the
  CPU** (1.8e-6 over ~1,400 terms) — a kernel misreading the packed mode table
  would still produce a smooth random-looking field, which no "looks patterned"
  check would catch; the same seed redraws the same field and a different one
  does not; the field is band-limited (5e-14 of its energy above degree 20)
  with λ setting the scale; and a λ finer than the mode table holds is refused
  rather than silently truncated.

[`test/fluxChecks.ts`](test/fluxChecks.ts) — the six-transform flux-form
Laplace-Beltrami scheme
([docs/reduced-transforms.md](docs/reduced-transforms.md)):

- on the sphere, the precomputed weights match their closed form and the
  analysed fluxes are **exactly band-limited** (beyond-band tails at f64
  round-off, ~1e-13), while the deliberately non-smooth control
  `Q̃/sin θ` keeps a fat tail (~1e-2) — the discrimination the whole scheme
  rests on;
- on a non-axisymmetric surface, the flux tails match the Cartesian gradient
  component's, the doc's §7.1 criterion;
- the compiled op sequences add **5 Legendre transforms per species per
  iteration against Algorithm 4's 12**, and a real simulation driven by
  each stays within fp32 accumulation of the other.

[`test/modelChecks.ts`](test/modelChecks.ts) compiles every model the app offers
and asserts **how many kernels it compiles to**, split into the base step and
what one solve iteration adds. That is a fusion guard: if numbl's inline pass
stops folding, the results stay correct while every operator becomes its own
dispatch, which is invisible in the numbers.

[`test/transformChecks.ts`](test/transformChecks.ts) compares the WGSL transforms
against shtns-webgpu's f64 CPU twin, and holds every compiled batch width to
the scalar transforms lane by lane; a model run with `SHT_BATCH=0` must
reproduce the batched run's state exactly.

- `npm run test:node` — under Dawn on the desktop, via `vite-node`. Needs a GPU;
  `--skip-without-gpu` lets a machine without one say so and move on (which is
  what CI does, since the browser suite covers the same modules).
- `npm run test:gpu` — builds and drives headless Chrome, on SwiftShader in CI.
  Also runs the soak. A few geometry tolerances are set by SwiftShader's fp32,
  which is about an order of magnitude looser than Dawn's.

Other commands:

- `npm run bench -- --help` — the desktop benchmark.
- `npm run bench:sht -- --help` — the transforms alone, no solver.
- `npx vite-node scripts/diagnose-sht.ts` — when the transform tests fail on a
  GPU, say *which* stage is wrong.
- `npx vite-node scripts/diagnose-leg.ts [--m 0]` — read the Legendre recurrence
  out of the production shader term by term.
- `npx vite-node scripts/longrun-node.ts [lmax]` — run to t = 100 and confirm the
  pattern saturates rather than decaying or diverging.
- `node scripts/soak.mjs [steps] [lmax]` — drive the demo for many steps,
  sampling JS heap and catching crashes.
- `node scripts/screenshot.mjs out.png [light|dark] [minSteps]` — screenshot the
  demo after a number of steps.
- `node scripts/check-live.mjs [url]` — smoke-check a deployed URL.
- `test.html?soak=<steps>&lmax=<n>` — solver-only soak with no rendering.

### Testing against a reference implementation

Reference solutions live in the sibling
[turing-surface-test-data](https://github.com/concept-collection/turing-surface-test-data)
repo, so an independently-written solver never has to depend on this one.
`cases/schnakenberg-ellipsoid.md` there specifies the one case this repo
currently ships a reference for;
[`docs/ellipsoid-reference-spec.md`](docs/ellipsoid-reference-spec.md)
restates it in this repo's own terms.

`npm run ref -- --in <file>` (`scripts/ref.ts`) loads a reference file, runs
the solver from its exact initial spectral state to the same physical end
time, and reports the relative-L2 and relative-L-infinity (max-norm) error
against its final state (plus a geometry sanity check). `--niter` overrides
the surface-correction iteration count independent of the file, and
`--tolerance`/`--tolerance-linf` each independently turn their metric into a
pass/fail for CI.

The same check runs in the page: **Compare to reference…** picks a `.h5` and
opens the comparison in one step — the file's own settings (its recorded
niter, its band, its dt) as the single variant, paused at the file's exact
initial state, ready to Run. The file defines the whole problem — model,
parameters, geometry, initial state — and the run stops at the file's end
time, measured against one extra static row showing its final state on its
own surface. Watching *where* a variant leaves the reference (rather than
just reading one number per run) is the point. To widen the study, stop
comparing, pick more chips, and press Compare — the file stays loaded, with
the lmax choices floored at its band, since a narrower one could not hold
its initial state. Reading the file uses
[h5wasm](https://github.com/usnistgov/h5wasm)'s wasm build, loaded lazily on
the first file opened.

## Development

```
npm install
npm run dev       # local dev server
npm run build     # type-check + production build to dist/
```

### The numbl dependency

numbl is a local `file:../../numbl` dependency, so a sibling checkout of
[numbl](https://github.com/flatironinstitute/numbl) is required. We use its
compiler internals — parser, lowerer, IR, inline pass — which its package
`exports` map does not publish, so they are reached through the `numbl-src` path
alias in [`vite.config.ts`](vite.config.ts).

The exact surface we depend on is written down in
[`src/mgpu/numbl.d.ts`](src/mgpu/numbl.d.ts) and TypeScript checks against
*that*, not against numbl's sources. This keeps this project's compiler settings
independent of numbl's, and means a change to one of those shapes upstream
breaks the build here with a clear diff rather than deep inside numbl's tree.
The `For` IR node is spelled out there, since the planner now walks it.

CI clones numbl to the sibling path that the `file:` dependency expects, pinned
to a commit, with `--ignore-scripts` (npm runs a linked package's `prepare`
script, and numbl's is husky). numbl's own `node_modules` are not needed: the
slice we import is self-contained TypeScript.

The `scripts/*.ts` entry points that touch the compiler go through `vite-node`,
so they resolve imports exactly as the browser build does. Plain `node` cannot:
numbl's sources import each other as `./foo.js` while the files are `.ts`.

Deployed to GitHub Pages by `.github/workflows/deploy.yml` on push to `main`.

## License

CECILL-2.1 (inherited from SHTNS via shtns-webgpu, whose sources are vendored).
