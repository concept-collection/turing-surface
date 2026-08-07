# Reference test case: Schnakenberg on a triaxial ellipsoid

For validating this repo's Laplace-Beltrami surface correction against an
independently-implemented reference solver, comparing final state in
spherical-harmonic (SH) coefficient space.

## Geometry

A triaxial ellipsoid, `gx = ax·sinθ·cosφ`, `gy = ay·sinθ·sinφ`,
`gz = az·cosθ` (`geometries/ellipsoid.m`; defaults `ax=1.5, ay=1.0, az=0.6`).

**Important**: the solver does not run on this analytic surface — it
band-limits it first, analysing `(gx, gy, gz)` into SH coefficients truncated
at degree `lmax` and re-synthesizing before use (`src/geom/geometry.ts`). To
remove geometry-representation error as a confound, a reference file
supplies these coefficients directly as `/geometry/Gx`, `/geometry/Gy`,
`/geometry/Gz`. **The reference solver must reconstruct its surface (and
induced metric) by synthesizing these coefficients, not by evaluating the
analytic formula above.**

## Equations

Schnakenberg reaction-diffusion with the true surface Laplace-Beltrami
operator `Δ_g` (`models/schnakenberg.m`):

```
du/dt = D1·Δ_g(u) + a - u + u²v
dv/dt = D2·Δ_g(v) + b     - u²v
```

This repo's internal discretization (`niter` Richardson-iteration count for
its own `Δ_g` approximation, `dt` for its IMEX-Euler timestep) is not part of
the equations being tested — the reference solver may use any consistent
method for `Δ_g` and any timestep. It only needs to reach the same physical
end time `T = steps · dt`.

## Initial condition

Loaded from the reference file's `/initial/U` / `/initial/V` (t=0,
immediately after seeding, before any step), not regenerated — avoids
needing to reimplement this repo's PRNG (`src/mgpu/noise.ts`) to get a
matching initial condition.

## Output convention (must match exactly, from `src/sht/layout.ts`)

- Orthonormal spherical harmonics **including Condon-Shortley phase**.
- Real field ⇒ complex coefficients stored for `m ≥ 0` only:
  `Q_{l,-m} = (-1)^m · conj(Q_lm)`; `m=0` coefficients have zero imaginary part.
- **m-major ordering**: for `m = 0..lmax`, for `l = m..lmax`.
  `index(l,m) = m·(lmax+1) − m·(m−1)/2 + (l−m)`.
- Flat array, length `2·nlm` with `nlm = (lmax+1)(lmax+2)/2`, `[re,im]`
  interleaved per coefficient (`qlm[2·index(l,m)]`, `qlm[2·index(l,m)+1]`).

The reference solver must project its final `u`, `v` onto this same
convention/truncation and report flat `2·nlm` arrays, to diff directly
against the reference file's `/final/U` / `/final/V`.

## HDF5 file layout

Each reference file is one `.h5` file per run (written with
[h5wasm](https://github.com/usnistgov/h5wasm); readable from Python with
`h5py.File(path, "r")`). Coefficient datasets are `float32`, each of length
`2·nlm` in the convention above. Metadata is stored as attributes, grouped by
what it describes rather than as a single flat namespace:

```
/                             (attrs: command, model, species)
├─ backend/                   (attrs: adapter, runtime, precision)
├─ spec/                      (attrs: preset, geometry, lmax, seed, steps, warmup, niter)
│  ├─ params/                 (attrs: the model's own params, e.g. a, b, D1, D2, dt)
│  └─ geometry_params/        (attrs: the geometry's own params, e.g. ax, ay, az)
├─ grid/                      (attrs: lmax, mmax, nlat, nphi, nlm)
├─ geometry/
│  ├─ Gx                      dataset, float32[2·nlm]
│  ├─ Gy                      dataset, float32[2·nlm]
│  └─ Gz                      dataset, float32[2·nlm]
├─ initial/                   one dataset per species (e.g. U, V), float32[2·nlm] each
└─ final/                     one dataset per species (e.g. U, V), float32[2·nlm] each
```

`species` (root attribute) names which datasets live under `initial/` and
`final/` — `["U", "V"]` for Schnakenberg. `command` is the equivalent
`npm run bench --` invocation, for reproducing the run exactly.

## Parameters

| name | meaning | default |
|---|---|---|
| `a`, `b` | Schnakenberg kinetics | 0.1, 0.9 |
| `D1`, `D2` | diffusion coefficients | 4e-4, 8e-3 |
| `ax`, `ay`, `az` | ellipsoid semi-axes | 1.5, 1.0, 0.6 |
| `lmax` | SH truncation degree | 63 |
| `T = steps·dt` | physical end time | e.g. 2000·0.05 = 100 |
| `seed` | provenance only — IC supplied as coefficients | 1 |

## Checking a run against a reference file

`npm run ref -- --in <file>` loads a reference file, runs this
repo's own solver from its exact initial condition to the same physical end
time, and reports the relative-L2 and relative-L-infinity (max-norm) error of
the resulting state against the file's final state (and, as a sanity check,
of the regenerated geometry against the file's own geometry coefficients —
this should be ~0 unless geometry construction itself has changed). `--niter
<n>` overrides the solve's own iteration count for the surface correction,
independent of what the reference file was generated with — useful for seeing
how much that correction term actually matters for a given run. `--tolerance
<n>` and `--tolerance-linf <n>` each independently turn their metric into a
pass/fail (nonzero exit code on failure), for use in CI.

The browser demo runs the same check visually: **Compare → Reference file…**
loads a reference file into the convergence study, seeds every variant from
its exact initial state, runs them side by side to its end time, and shows
its final state as one extra static row — on the file's own surface, with
each variant's relative-L2 distance to it updating live. Both readers share
one parser (`src/compare/referenceCase.ts`), so the layout above is
interpreted identically on the CLI and in the page.

Note the files record only the two endpoint states (`initial/`, `final/`) —
no intermediate snapshots — so the comparison is meaningful at the end time;
the live Δ before that reads as "distance still to the final state".

## Caveat

This repo runs fp32 on GPU; expect ~1e-4–1e-6 relative floating-point noise
on top of any genuine numerical-method disagreement between solvers.
