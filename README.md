# turing-surface

Reaction–diffusion systems (Turing patterns) on **closed surfaces given by
spherical-harmonic embeddings**, solved live in the browser with a spectral
method whose transforms run on the GPU via WebGPU.

This is the sibling of
[turing-sphere](https://github.com/concept-collection/turing-sphere), which
solves the same systems on the round sphere. Everything there is here; what is
added is a *surface*.

> [!NOTE]
> **The geometry is in the operator.** The models solve with the surface's
> Laplace–Beltrami operator, iterated by a fixed-count preconditioned
> Richardson solve ([`solvers/richardson.m`](solvers/richardson.m), applying
> [`lib/dlap.m`](lib/dlap.m)). The iteration count is fixed at compile time
> with no residual check, so a shape/timestep/diffusivity combination outside
> its convergence radius diverges over many steps rather than being caught —
> the tests pin the known cases. See
> [Where the geometry enters the operator](#where-the-geometry-enters-the-operator).

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

That is ordinary element-wise MATLAB and goes through the same compiler and the
same WGSL backend the models do. It is evaluated once on the solver's grid, and
then **analysed into coefficients**, which is the form everything downstream
uses. Two things follow from going through the coefficients rather than keeping
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

Four geometries ship: [sphere](geometries/sphere.m) (the reference case),
[ellipsoid](geometries/ellipsoid.m), [peanut](geometries/peanut.m) — a dumbbell
whose waist is a saddle — and [bumpy](geometries/bumpy.m). Each is editable in
the page, with its own parameters. Changing a shape does not recompile the
solver and does not disturb the run: the geometry is data whose shape in the
bindings depends only on the grid, so a swap is six buffer writes and the
pattern carries straight on.

A **morph** slider blends the drawn surface back to the unit sphere. The
parametrization is the sphere's either way, so sweeping it shows which point
went where.

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
solve.

The pieces of that sentence are separate files, because they are separate
ideas. The **operator** — `dlap` applied to a spectral field — is
[`lib/dlap.m`](lib/dlap.m). The **solver** — the fixed point above, iterated
`niter` times — is [`solvers/richardson.m`](solvers/richardson.m):

```matlab
function X = richardson(B, dtD, lam, filt, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, niter)
  X = B ./ (1 + dtD * lam);
  for k = 1:niter
    dL = dlap(X, filt, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, lam);
    X = (B + dtD * dL) ./ (1 + dtD * lam);
  end
end
```

And a **model** is a reaction plus one solve per species — the whole of
[`models/schnakenberg.m`](models/schnakenberg.m)'s step is:

```matlab
function [Un, Vn, u, v] = step(U, V, lam, filt, gx, gy, gz, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, a, b, D1, D2, dt, niter)
  u = synth(U);
  v = synth(V);
  uuv = u .* u .* v;

  Bu = U + dt * analys(a - u + uuv);
  Bv = V + dt * analys(b - uuv);

  Un = richardson(Bu, dt * D1, lam, filt, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, niter);
  Vn = richardson(Bv, dt * D2, lam, filt, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, niter);
end
```

Trying a different solver against the same operator is a change to those two
call lines: every solver composes from `dlap` (the matvec is
`(1 + dtD.*lam).*x - dtD.*dlap(x)`, the preconditioner the elementwise
divide), and which one a model calls is part of what compiles — swapping
recompiles, like changing `niter` already does. The solver is written as a
full re-evaluation rather than an accumulated correction on purpose: where
`dlap` computes to zero there is no correction to mis-round, and the divide is
turing-sphere's arithmetic unchanged.

Three solvers ship. [`solvers/bicgstab.m`](solvers/bicgstab.m) solves the
same system by preconditioned BiCGSTAB — same `dlap`, same preconditioner, a
Krylov recurrence instead of a stationary one, at two `dlap` evaluations per
iteration instead of one. Its scalars (`rho`, `alpha`, `omega`) never touch
the CPU: `dot` is a GPU reduction into a 1-element buffer, the recurrences on
its results compile to 1-element kernels, and a single-element value
broadcasts into the vector updates. Inner products carry the half-spectrum
weight `wlm` (m > 0 counts twice), making them the real L2 inner products on
the sphere. With no residual test, every ratio `a/b` is written in the
guarded form `a*b/(b*b + 1e-30)`, so a converged (or broken-down) iteration
goes stationary instead of dividing noise by noise. The difference is not
academic: at the app's default lmax, Schnakenberg on the peanut sits outside
the Richardson iteration's convergence radius for `niter ≥ 2` and diverges,
while BiCGSTAB on the identical operator converges monotonically — the tests
pin both behaviors, side by side.

[`solvers/gmres.m`](solvers/gmres.m) is right-preconditioned GMRES(niter) —
one Arnoldi sweep, no restart — with the residual minimized over the whole
Krylov space. Its bookkeeping is what the other solvers never need: a basis
of niter+1 spectral fields, a Hessenberg matrix, Givens rotations, a
triangular back-substitution. The basis lives in a *bank* (`getslab` /
`setslab`: the k-th 2 × nlm field of a wider array), the small matrices are
element-addressed (`getat` / `setat`), and both are functional updates the
planner compiles to static-offset buffer copies — MATLAB's own `H(i,j) = h`
cannot lower, because numbl must prove an indexed write in bounds before the
loop unrolls, and a loop variable has no value yet at that point. Written as
calls, the index resolves at *planning*, where unrolling has made it a
literal. The same resolution lets an inner loop bound depend on the outer
loop's variable, which is what makes the `for i = 1:j` orthogonalization
sweep compile.

### Where the geometry enters the operator

[`lib/dlap.m`](lib/dlap.m) is Algorithm 3 of the evolving-surface notes: the
field's θ/φ derivatives (the `dtheta`/`dphi` transforms,
[`src/sht/deriv.ts`](src/sht/deriv.ts)) are contracted through the inverse
metric quantities into a tangential gradient; each Cartesian component is
re-analysed and differentiated again; the results recombine into the surface
divergence, and `lam .* F` adds back what the round-sphere part already
carries. The metric quantities `Vt*`/`Vp*`
([`src/geom/metric.ts`](src/geom/metric.ts)) are built once from the
embedding's derivatives when the geometry is (re)built — the geometry is
static, so per step they are just six more buffers the kernels read. `filt`
zeroes the top two spectral degrees wherever the operator re-differentiates,
because the derivative recurrences cannot exactly represent a derivative
there.

On the sphere `dlap` computes to (numerical) zero, so any `niter` lands
within transform round-off of the exact round-sphere answer — asserted in the
tests. Off the sphere the correction genuinely moves the answer, and
convergence is a real constraint: the fixed-count loop has no residual check,
so the tests also pin which shape/niter combinations are known to sit outside
the convergence radius and diverge.

### Subroutines

A model file is not limited to `init` and `step`: it can define further
functions and call them, and every model compiles against the shared library
files — [`lib/`](lib/) for operators, [`solvers/`](solvers/) for solvers —
with MATLAB's visibility rules (a file's namesake function is public; a
model-local function of the same name shadows it). numbl specializes each
callee for the argument types at its call sites, and the host then splices
the lowered body into the caller, one clone per call site
([`src/mgpu/inlineCalls.ts`](src/mgpu/inlineCalls.ts)): arguments bind by
renaming rather than copying, and assignments to a callee output become
assignments to the caller's variable, which is what lets a solver iterate its
result in place. Expansion runs before the fusion pass, so a call fuses
exactly as the same code written inline would — the boundary costs nothing,
and `describe()`'s op listing names the expanded internals
(`richardson#1.X`). Recursion cannot unroll into a fixed op sequence and is
refused at compile time, like a runtime loop bound.

### `for` loops, unrolled

A plan is a fixed list of GPU operations with no branching, which is what makes
a timestep pure command recording — one submit, no CPU in the loop. A counted
loop still fits: the planner
([`src/mgpu/plan.ts`](src/mgpu/plan.ts)) unrolls it, planning the body once per
iteration. The loop that matters is `solvers/richardson.m`'s `for k = 1:niter`,
expanded into each model's step at every solve call site.

Nothing else had to change for that, because numbl gives a variable one cName
for every assignment to it: the buffer an iteration writes is the buffer the
next one reads, which is exactly a loop-carried value. The loop variable gets no
buffer at all — it is bound as a derived scalar to that iteration's literal, so
a kernel reading `k` folds the number in.

Two consequences worth stating:

- **The bounds must be known when the model compiles.** `niter` is supplied as a
  fixed scalar rather than a tunable one, so changing it recompiles — unlike a
  parameter, which is a uniform. A bound may also be an enclosing unrolled
  loop's variable (`for i = 1:j` — each unrolled `j` plans its own inner trip
  count, which is how GMRES's triangular sweeps compile). A genuinely runtime
  bound is refused at compile time with a source position, not silently
  mis-compiled, and there is a test for that.
- **Fusion survives.** numbl's inline pass recurses into loop bodies, so a line
  inside the loop is still one kernel. It runs there with no protected names,
  though, which means an assignment whose only visible use is later in the same
  body can be elided — correct for a body-local temp, wrong if something outside
  the loop wanted it. [`src/mgpu/compile.ts`](src/mgpu/compile.ts) snapshots what
  each loop body assigns before the pass and refuses the ones that escape, so
  that case is a compile error rather than a stale read.

Unrolling is exactly linear in the trip count: 26 GPU ops per species per
iteration (the operator's twelve transforms and the solve's kernels), asserted
in the tests.

## MATLAB, compiled to WebGPU

Unchanged from turing-sphere, and it now compiles the geometry files too. numbl
parses and lowers each function for the concrete argument types of the current
grid; user-function calls are expanded into the caller, one clone per call
site; the inline pass folds single-use temps back into their consumer, so one
line of MATLAB becomes one expression tree; and this repo emits one WGSL compute
kernel per element-wise statement
([`src/mgpu/wgsl.ts`](src/mgpu/wgsl.ts)). `synth` / `analys` (and the
derivative pair `dtheta` / `dphi`) are external operations whose type rules
numbl learns from a `.mtoc2.js` workspace file, and which the backend maps onto
the spherical-harmonic pipelines; `dot` is one more, mapped onto a
single-dispatch reduction ([`src/mgpu/reduce.ts`](src/mgpu/reduce.ts)) whose
1-element result stays on the GPU — scalars computed from it become 1-element
kernels, and reading one inside a vector expression broadcasts it. The
indexed-access ops (`getslab`/`setslab`, `getat`/`setat`) compile to
static-offset buffer copies, their indices evaluated at planning time where
the unrolled loop's variable is a literal. Anything it cannot express is
refused at compile time with a source position.

The Schnakenberg step above compiles to 65 GPU operations at one solve
iteration: 28 transforms, 35 generated kernels, and 2 buffer copies feeding the
new state back.

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
  from 1e-2 seeded noise this is inconsequential. The geometry goes through one
  analysis/synthesis round trip and picks up the same round-off: the unit sphere
  comes back with radius 1 to ~2e-5 under Dawn, ~4e-4 under SwiftShader.
- The shipped geometries are all degree ≤ 5, far below any lmax the app offers,
  so band-limiting removes nothing from them. A shape you write yourself may not
  be so lucky — see the note in [`geometries/bumpy.m`](geometries/bumpy.m).

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
properties**. Four modules, run in both environments:

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

[`test/geometryChecks.ts`](test/geometryChecks.ts) — the surface and the loop:

- every geometry compiles and closes; the sphere has radius 1 everywhere and is
  **exactly degree 1** in the harmonics, which is what makes the reference case
  exact rather than merely accurate;
- the peanut matches its own closed-form radial profile at every grid point, and
  **the same coefficients give the same surface on a 2× grid** — the 2× Gauss
  latitudes share no point with the 1× ones, so agreeing there is agreeing
  everywhere, which is what "rendered exactly, not subdivided" means;
- unrolling is **exactly linear** in the trip count; on the sphere the
  geometric correction computes to (numerical) zero, so 0, 1 and 4 iterations
  agree to transform round-off; on the peanut it **measurably moves the
  answer**;
- a niter × geometry sweep stays finite except the combinations **known to sit
  outside the Richardson convergence radius**, which are pinned as diverging —
  and on exactly those combinations **bicgstab and gmres keep converging**,
  also pinned;
- at equal niter, **bicgstab and gmres land far closer to the converged
  answer** than richardson on the same operator;
- a runtime loop bound is refused at compile time;
- swapping the surface mid-run leaves the spectral state untouched.

[`test/modelChecks.ts`](test/modelChecks.ts) compiles every model the app offers
and asserts **how many kernels it compiles to**, split into the base step and
what one solve iteration adds. That is a fusion guard: if numbl's inline pass
stops folding, the results stay correct while every operator becomes its own
dispatch, which is invisible in the numbers. It also compiles a model built of
**user-defined subroutines** — multi-output, scalar-returning, a solver-like
local with its own loop — asserts recursion is refused, and checks the
**reduction and indexing primitives** directly against exact expected values:
the `dot` sum and the `wlm`-weighted inner product against the CPU, scalar
arithmetic on a GPU-resident result bit for bit, the 1-element broadcast, and
element/slab round-trips through `setat`/`getat` and `setslab`/`getslab`.

[`test/transformChecks.ts`](test/transformChecks.ts) compares the WGSL transforms
against shtns-webgpu's f64 CPU twin.

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
