# turing-surface

Reaction–diffusion systems (Turing patterns) on **closed surfaces given by
spherical-harmonic embeddings**, solved live in the browser with a spectral
method whose transforms run on the GPU via WebGPU.

This is the sibling of
[turing-sphere](https://github.com/concept-collection/turing-sphere), which
solves the same systems on the round sphere. Everything there is here; what is
added is a *surface*.

> [!WARNING]
> **The geometry is rendered, not yet solved on.** The Laplace–Beltrami
> operator in the models is still the round sphere's — the term that carries
> the shape is a placeholder that is identically zero. On anything but the
> sphere you are looking at the sphere's pattern painted onto that surface, not
> the pattern that surface would grow. Everything the correction needs in order
> to be dropped in — the embedding, the split of the operator, the iterative
> solve, the unrolled loop — is built and tested. See
> [The geometry is not in the operator yet](#the-geometry-is-not-in-the-operator-yet).

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
what would keep the cost to a few transforms per step rather than a full
elliptic solve. Written out, the whole of
[`models/schnakenberg.m`](models/schnakenberg.m)'s step is:

```matlab
function [Un, Vn, u, v] = step(U, V, lam, gx, gy, gz, a, b, D1, D2, dt, niter)
  u = synth(U);
  v = synth(V);
  uuv = u .* u .* v;

  Bu = U + dt * analys(a - u + uuv);
  Bv = V + dt * analys(b - uuv);

  Un = Bu ./ (1 + (dt * D1) * lam);
  Vn = Bv ./ (1 + (dt * D2) * lam);

  for k = 1:niter
    dLu = 0 * Un;                                    % <- the placeholder
    dLv = 0 * Vn;
    Un = (Bu + (dt * D1) * dLu) ./ (1 + (dt * D1) * lam);
    Vn = (Bv + (dt * D2) * dLv) ./ (1 + (dt * D2) * lam);
  end
end
```

Written this way rather than as a residual correction on purpose: with `dlap`
zero, every iterate is *bit for bit* the first line, with no cancellation to
round differently. So the sphere case is not "close to" turing-sphere, it is
the same arithmetic, and the tests assert exactly that — the state after 20
steps is identical at 0, 1 and 4 iterations.

### The geometry is not in the operator yet

What belongs where `dLu` is now is `dlap = lap_g - lap_s` applied to the current
iterate. Getting it needs two things this repo does not have:

1. **The induced metric**, `g_ij = ∂_i X · ∂_j X` for `X = (gx, gy, gz)`. The
   geometry is static and low-degree, so this is a one-off precomputation, not
   per-step work — but it needs θ- and φ-derivatives of the embedding.
2. **Surface derivatives of the field**, per iteration. In the round frame this
   is the spheroidal transform pair — SHTNS's `SHsph_to_spat` and
   `spat_to_SHsph`, i.e. `grad_s` and `div_s` — which lets the operator be
   written as `div_s(A grad_s f)` with `A` built from the metric, with no
   explicit `1/sin θ` to go singular at the poles.

Both need Legendre *derivative* tables, which the vendored WGSL transforms under
[`src/sht/`](src/sht/) do not implement — they are scalar synthesis and analysis
only. That is the missing piece, and it is a substantial addition to the
transforms rather than a change to the models. Until it lands, the models take
`gx, gy, gz` (the surface on the grid) and `Gx, Gy, Gz` (the same surface as
coefficients) as arguments and do not use them, and the app says so.

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

Unrolling is exactly linear in the trip count: 2 GPU ops per species per
iteration, asserted in the tests.

## MATLAB, compiled to WebGPU

Unchanged from turing-sphere, and it now compiles the geometry files too. numbl
parses and lowers each function for the concrete argument types of the current
grid; its inline pass folds single-use temps back into their consumer, so one
line of MATLAB becomes one expression tree; and this repo emits one WGSL compute
kernel per element-wise statement
([`src/mgpu/wgsl.ts`](src/mgpu/wgsl.ts)). `synth` / `analys` are external
operations whose type rules numbl learns from a `.mtoc2.js` workspace file, and
which the backend maps onto the spherical-harmonic pipelines. Anything it cannot
express is refused at compile time with a source position.

The Schnakenberg step above compiles to 17 GPU operations at one solve
iteration: 4 transforms, 11 generated kernels, and 2 buffer copies feeding the
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
- unrolling is **exactly linear** in the trip count, and the state after 20 steps
  is **bit-identical** at 0, 1 and 4 iterations;
- a runtime loop bound is refused at compile time;
- swapping the surface mid-run leaves the spectral state untouched.

[`test/modelChecks.ts`](test/modelChecks.ts) compiles every model the app offers
and asserts **how many kernels it compiles to**, split into the base step and
what one solve iteration adds. That is a fusion guard: if numbl's inline pass
stops folding, the results stay correct while every operator becomes its own
dispatch, which is invisible in the numbers.

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
