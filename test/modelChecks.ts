/**
 * Every model the app offers: that it compiles, what it compiles to, and that it
 * runs stably and produces a pattern.
 *
 * Numerical correctness of the pipeline is analyticChecks.ts's job. This file is
 * about the models themselves and about the compilation staying as intended — in
 * particular the kernel count, which is a fusion guard: numbl's lowering emits
 * one statement per *operator*, and its inline pass folds those back into
 * per-line expression trees. If that stops happening the results stay correct
 * but every operator becomes its own dispatch, which is invisible except here.
 */
import { ModelSession } from '../src/mgpu/session.ts';
import { mModels, mModelByKey, defaultParams } from '../src/mgpu/registry.ts';
import { eigenvalues, weightMask } from '../src/mgpu/model.ts';
import { modelLibs } from '../src/mgpu/libs.ts';
import {
  formatCommand,
  parseArgs,
  BENCH_COMMAND,
  type RunSpec,
} from '../src/bench/runSpec.ts';
import type { Check, Log } from './analyticChecks.ts';

/**
 * Kernels each model's step compiles to outside its solve loop — one per
 * element-wise line, where the argument of a transform counts as its own line
 * (it cannot fuse into an external call).
 */
const EXPECTED_KERNELS: Record<string, number> = {
  schnakenberg: 11,
  brusselator: 11,
  allencahn: 5,
  'schnakenberg-alg4': 8,
};

/**
 * What one unrolled iteration of the solve loop adds — 9 kernels per
 * species with the factored flux-form operator: 8 in lib/dlap.m (the
 * filtered input, the sphere term lam.*G, the two flux combinations, the
 * filtered theta-flux, the r-scaled sphere-split divergence, the band-
 * projected correction, and the lamJ divisor recomputed per inlined call),
 * plus the solver's update divide. Each species' correction is the
 * flux-form Laplace-Beltrami matvec of docs/reduced-transforms.md Sec 4 —
 * see lib/dlap.m, solvers/richardson.m and docs/richardson-iteration.md.
 * `schnakenberg-alg4` keeps the original Cartesian-gradient form
 * (Algorithm 3/4 of evolving_surface/notes/algos.tex) as a live,
 * self-contained reference, with its original counts.
 */
const KERNELS_PER_ITERATION: Record<string, number> = {
  schnakenberg: 18,
  brusselator: 18,
  allencahn: 9,
  'schnakenberg-alg4': 32,
};

const LMAX = 31;
const STEPS = 40;
const NITER = 1;

export async function modelChecks(
  device: GPUDevice,
  check: Check,
  log: Log,
): Promise<void> {
  check('models: registry populated', mModels.length === 4, `${mModels.length} models`);

  // The app formats the run it is showing into a `npm run bench` command and
  // the benchmark parses it back. That is only worth anything if the round
  // trip is lossless — a knob that formatCommand forgets is a knob the desktop
  // run would silently take a default for, and the two runs would differ while
  // claiming to be the same. Every field of the spec, through both directions.
  {
    const spec: RunSpec = {
      preset: 'schnak-fine',
      lmax: 127,
      seed: 12345,
      steps: 777,
      warmup: 13,
      params: { a: 0.11, b: 0.91, D1: 5e-4, D2: 9e-3, dt: 0.04 },
      geometry: 'peanut',
      geometryParams: { waist: 0.45, stretch: 1.25 },
      niter: 3,
      solver: 'bicgstab',
    };
    const command = formatCommand(spec);
    const back = parseArgs(command.slice(BENCH_COMMAND.length).trim().split(/\s+/));
    const same = JSON.stringify(back) === JSON.stringify(spec);
    check(
      'runSpec: the benchmark command round-trips every field',
      same,
      same ? command.slice(BENCH_COMMAND.length + 1) : `got ${JSON.stringify(back)}`,
    );
  }

  for (const model of mModels) {
    const session = await ModelSession.create({
      device,
      model,
      params: defaultParams(model),
      lmax: LMAX,
      niter: NITER,
    });

    const plan = session.describe();
    const kernels = plan.step.filter((l) => l.startsWith('kernel')).length;
    const xforms = plan.step.filter(
      (l) => l.startsWith('synth') || l.startsWith('analys'),
    ).length;
    const expected = EXPECTED_KERNELS[model.key] + NITER * KERNELS_PER_ITERATION[model.key];
    log(
      `  ${model.key}.m -> ${plan.step.length} ops/step ` +
        `(${kernels} generated kernels, ${xforms} transforms, ${NITER} solve iter)`,
    );
    check(
      `${model.key}: element-wise lines fused into one kernel each`,
      kernels === expected,
      `${kernels} kernels (expected ${expected})`,
    );

    await session.seed(1);
    session.step(STEPS);

    // Every rendered field must be finite and have developed some contrast.
    for (const field of model.species) {
      const values = await session.read(field);
      let lo = Infinity;
      let hi = -Infinity;
      let finite = true;
      for (const v of values) {
        if (!Number.isFinite(v)) finite = false;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      check(
        `${model.key}: '${field}' is finite and patterned after ${STEPS} steps`,
        finite && hi - lo > 1e-6,
        finite
          ? `range [${lo.toFixed(5)}, ${hi.toFixed(5)}]`
          : 'contains NaN or Infinity',
      );
    }

    session.destroy();
  }

  // Batched transforms are an encoding of the same arithmetic, so a run with
  // batching disabled (SHT_BATCH=0 compiles scalar-only plans) must reproduce
  // the default run to shader-compiler latitude, and the default run must
  // actually be batching (the describe() lines say so). This is the guard
  // that the planner's adjacency grouping rewires buffers correctly — a lane
  // bound to the wrong field would miss by O(1), not O(1e-6).
  {
    const model = mModelByKey('schnakenberg')!;
    const params = defaultParams(model);
    const states: Float32Array[] = [];
    let batchedLanes = 0;
    for (const batch of [undefined, 0]) {
      const g = globalThis as Record<string, unknown>;
      if (batch !== undefined) g.SHT_BATCH = batch;
      try {
        const session = await ModelSession.create({
          device, model, params, lmax: LMAX, niter: NITER,
        });
        if (batch === undefined) {
          batchedLanes = session
            .describe()
            .step.filter((l) => l.includes('[batch lane')).length;
        }
        await session.seed(1);
        session.step(STEPS);
        states.push(await session.read('U'));
        session.destroy();
      } finally {
        delete g.SHT_BATCH;
      }
    }
    // Every batchable run at one solve iteration: the u/v syntheses and the
    // reaction analyses outside the loop (2 + 2), plus each species' solve
    // running lib/dlap.m once — the gradient synthesis group with the
    // round-sphere term riding along (3 lanes per species; the theta-flux
    // analysis, divergence synthesis and final analysis are single
    // transforms with nothing adjacent to batch against, and the phi flux
    // goes through dphig, which has no Legendre stage at all). Lane counts
    // are batch-width invariant: a x4 run is one batch at K = 4 and two at
    // K = 2, but the lanes annotated are the same 10 either way.
    check(
      'batch: the compiled step batches every adjacent transform group',
      batchedLanes === 10,
      `${batchedLanes} batched transform lanes (expected 10)`,
    );
    let worst = 0;
    for (let i = 0; i < states[0].length; i++) {
      worst = Math.max(worst, Math.abs(states[0][i] - states[1][i]));
    }
    check(
      'batch: batched and scalar plans agree through a real run',
      worst < 1e-4,
      `max |U_batched - U_scalar| = ${worst.toExponential(2)} after ${STEPS} steps`,
    );
  }

  // Misusing the grouped-transform syntax is refused at compile time with a
  // message that says how to write it, not silently mis-planned: every input
  // must get an output (each one costs a transform), whether the mismatch is
  // an under-bound assignment or an ignored slot. The grouped call lives in
  // lib/dlap.m now, so the misuse arrives through libSources — the same
  // route the page's editor takes for the shared files.
  {
    const model = mModelByKey('allencahn')!;
    const dlapSource = modelLibs.find((f) => f.name === 'dlap.m')!.source;
    const cases: [string, string, string][] = [
      [
        'a single output bound to a grouped call',
        'Ft = synth(vt, vp, S0);',
        'bind each one',
      ],
      [
        'an ignored output slot',
        // Fp is reassigned so the only error left is the dropped slot
        // itself, which the planner refuses (numbl would otherwise catch
        // the undefined 'Fp' first, masking the check under test).
        '[Ft, ~, S] = synth(vt, vp, S0);\n  Fp = Ft;',
        'must be bound',
      ],
    ];
    for (const [what, bad, expect] of cases) {
      const source = dlapSource.replace('[Ft, Fp, S] = synth(vt, vp, S0);', bad);
      if (source === dlapSource) throw new Error('grouped-call fixture no longer matches lib/dlap.m');
      let message = '';
      try {
        const session = await ModelSession.create({
          device, model, params: defaultParams(model), lmax: LMAX, niter: 1,
          libSources: { 'dlap.m': source },
        });
        session.destroy();
      } catch (e) {
        message = e instanceof Error ? e.message : String(e);
      }
      check(
        `batch: ${what} is refused at compile time`,
        message.includes(expect),
        message ? `refused: ${message.slice(0, 76)}…` : 'compiled anyway',
      );
    }
  }

  // User-defined subroutines: a .m may define its own functions (and call the
  // shared solver/operator library), and each call is expanded into the caller
  // at compile time (src/mgpu/inlineCalls.ts). This model exercises the
  // shapes the shipped models do not: a multi-output function, a
  // scalar-returning function, a function reassigning its own parameter, and
  // a solver-like local whose loop bound arrives as the `niter` argument.
  {
    const model = mModels.find((m) => m.key === 'allencahn')!;
    const source = `
function [U, u] = init(noise)
  U = analys(noise);
  u = synth(U);
end

function [Un, u] = step(U, lam, eps2, dt, niter)
  u = synth(U);
  [p, q] = react(u, dt);
  s = gain(eps2, dt);
  Bu = U + s * analys(p - q);
  Un = solveid(Bu, lam, dt, niter);
end

function [p, q] = react(x, c)
  p = x + c * (x .* x);
  q = c * (x .* x);
end

function y = gain(a, b)
  y = a + 2 * b;
end

function X = solveid(B, lam, c, n)
  X = B ./ (1 + c * lam);
  for k = 1:n
    X = (B + c * (0 * X)) ./ (1 + c * lam);
  end
end
`;
    const session = await ModelSession.create({
      device, model, params: defaultParams(model), lmax: LMAX, source, niter: 2,
    });
    session.seed(1);
    session.step(STEPS);
    const values = await session.read('u');
    let finite = true;
    for (const v of values) if (!Number.isFinite(v)) finite = false;
    check(
      'subroutines: a model composed of user functions compiles and runs',
      finite,
      `${session.describe().step.length} ops/step after expansion`,
    );
    session.destroy();
  }

  // The reduction op and GPU-resident scalars: `dot` runs as a single
  // reduction dispatch into a 1-element buffer, scalars computed from its
  // result compile to 1-element kernels, and a single-element value
  // broadcasts into element-wise expressions as `in[0]`. These are the
  // primitives the Krylov solver is made of, checked directly against the
  // CPU here so a solver-level failure has somewhere smaller to point.
  {
    const model = mModels.find((m) => m.key === 'allencahn')!;
    const source = `
function [U, u] = init(noise)
  U = analys(noise);
  u = synth(U);
end

function [Un, u] = step(U, lam, wlm, eps2, dt, niter)
  u = synth(U);
  s = dot(U, U);
  Uw = U .* wlm;
  sw = dot(Uw, lam);
  s2 = 2 * s;
  s3 = s2 - s;
  Un = (s * U) ./ s;
end
`;
    const session = await ModelSession.create({
      device, model, params: defaultParams(model), lmax: LMAX, source, niter: 1,
    });
    session.seed(1);
    session.step(1);
    const U = await session.read('U');
    const nlm = U.length / 2;
    const cfg = session.cfg;

    let cpuS = 0;
    for (let i = 0; i < U.length; i++) cpuS += U[i] * U[i];
    const gpuS = (await session.read('s'))[0];
    check(
      'dot: matches the CPU sum',
      Math.abs(gpuS - cpuS) <= 1e-5 * Math.abs(cpuS),
      `gpu ${gpuS.toExponential(6)} vs cpu ${cpuS.toExponential(6)}`,
    );

    const wlm = weightMask(cfg, nlm);
    const lam = eigenvalues(cfg, nlm);
    let cpuSw = 0;
    for (let i = 0; i < U.length; i++) cpuSw += U[i] * wlm[i] * lam[i];
    const gpuSw = (await session.read('sw'))[0];
    check(
      'dot: the wlm-weighted inner product matches the CPU',
      Math.abs(gpuSw - cpuSw) <= 1e-5 * Math.abs(cpuSw),
      `gpu ${gpuSw.toExponential(6)} vs cpu ${cpuSw.toExponential(6)}`,
    );

    // 2s - s is exact in any IEEE arithmetic, so the whole scalar chain
    // (reduction -> 1-element kernels -> readback) must return s's bits.
    const gpuS3 = (await session.read('s3'))[0];
    check('dot: scalar arithmetic on the result is exact', gpuS3 === gpuS,
      `s3 ${gpuS3.toExponential(6)} vs s ${gpuS.toExponential(6)}`);

    const Un = await session.read('Un');
    let worst = 0;
    let scale = 0;
    for (let i = 0; i < U.length; i++) {
      worst = Math.max(worst, Math.abs(Un[i] - U[i]));
      scale = Math.max(scale, Math.abs(U[i]));
    }
    check(
      'dot: a 1-element value broadcasts into an element-wise kernel',
      worst <= 1e-6 * scale,
      `(s*U)./s vs U: worst |d| = ${worst.toExponential(2)}`,
    );
    session.destroy();
  }

  // The indexed-access ops (getslab/setslab on a bank of spectral fields,
  // getat/setat on a small matrix): functional updates the planner compiles
  // to static-offset buffer copies. Everything below has an exact expected
  // value, so the offsets themselves are what is being checked.
  {
    const model = mModels.find((m) => m.key === 'allencahn')!;
    const source = `
function [U, u] = init(noise)
  U = analys(noise);
  u = synth(U);
end

function [Un, u] = step(U, lam, eps2, dt, nlm, niter)
  u = synth(U);
  A = zeros(2, 2);
  s1 = dot(U, U);
  s2 = 2 * s1;
  A = setat(A, s1, 1, 1);
  A = setat(A, s2, 2, 2);
  a11 = getat(A, 1, 1);
  a22 = getat(A, 2, 2);
  a21 = getat(A, 2, 1);
  chk = a22 - 2 * a11 + a21;
  VB = zeros(2, nlm * 2);
  VB = setslab(VB, U, 2);
  U2 = getslab(VB, 2);
  Z1 = getslab(VB, 1);
  Un = U2 + Z1;
end
`;
    const session = await ModelSession.create({
      device, model, params: defaultParams(model), lmax: LMAX, source, niter: 1,
    });
    session.seed(1);
    session.step(1);
    // a22 - 2*a11 + a21 = 2*s - 2*s + 0, exactly, if every element landed
    // where its indices say.
    const chk = (await session.read('chk'))[0];
    check('indexing: matrix elements round-trip through setat/getat', chk === 0,
      `a22 - 2*a11 + a21 = ${chk}`);
    // The slab written at 2 must come back; the slab at 1 must still be zero.
    const U = await session.read('U');
    const Un = await session.read('Un');
    let same = U.length === Un.length;
    for (let i = 0; same && i < U.length; i++) if (Un[i] !== U[i]) same = false;
    check('indexing: a spectral field round-trips through setslab/getslab', same,
      same ? 'getslab(setslab(VB, U, 2), 2) + zeros = U, element for element' : 'mismatch');
    session.destroy();
  }

  // A recursive function cannot unroll into a fixed op sequence, and must be
  // refused with a message that says so, not hang the compiler.
  {
    const model = mModels.find((m) => m.key === 'allencahn')!;
    const source = `
function [U, u] = init(noise)
  U = analys(noise);
  u = synth(U);
end

function [Un, u] = step(U, lam, eps2, dt, niter)
  u = synth(U);
  Un = f(U);
end

function y = f(x)
  y = f(x) + 1;
end
`;
    let message = '';
    try {
      const session = await ModelSession.create({
        device, model, params: defaultParams(model), lmax: LMAX, source, niter: 1,
      });
      session.destroy();
    } catch (e) {
      message = e instanceof Error ? e.message : String(e);
    }
    check(
      'subroutines: recursion is refused at compile time',
      message.includes('recursion'),
      message ? `refused: ${message.slice(0, 72)}…` : 'compiled anyway',
    );
  }

  // The oversampled readback: readSpecies must be the state synthesized on the
  // display grid. Comparing against the display plan's own upload path
  // (read the state back, synth it from the CPU) exercises the GPU-to-GPU
  // coefficient copy against a known-good route through the same kernels.
  {
    const model = mModels.find((m) => m.key === 'allencahn')!;
    const session = await ModelSession.create({
      device,
      model,
      params: defaultParams(model),
      lmax: LMAX,
      oversample: 2,
    });
    await session.seed(1);
    session.step(STEPS);

    const fine = await session.readSpecies(0);
    const { nlat, nphi } = session.viewSht.cfg;
    check(
      'oversample: species field is on the 2x display grid',
      nlat === 2 * session.cfg.nlat &&
        nphi === 2 * session.cfg.nphi &&
        fine.length === nlat * nphi,
      `render ${nlat}×${nphi}, ${fine.length} values`,
    );

    const qlm = await session.read('U');
    const expected = await session.viewSht.synth(qlm);
    let maxDiff = 0;
    for (let i = 0; i < fine.length; i++) {
      const d = Math.abs(fine[i] - expected[i]);
      if (d > maxDiff) maxDiff = d;
    }
    check(
      'oversample: readSpecies matches synth of the read-back state',
      maxDiff <= 1e-6,
      `max |diff| = ${maxDiff.toExponential(2)}`,
    );

    // A timing burst must be invisible: the state is snapshotted and restored
    // around it, and model time does not advance.
    const tBefore = session.t;
    const stepsBefore = session.steps;
    const ms = await session.measure(8);
    const after = await session.read('U');
    let identical = qlm.length === after.length;
    if (identical) {
      for (let i = 0; i < qlm.length; i++) {
        if (qlm[i] !== after[i]) {
          identical = false;
          break;
        }
      }
    }
    check(
      'measure: a timing burst leaves state, t and steps untouched',
      identical && session.t === tBefore && session.steps === stepsBefore,
      identical
        ? `state identical, t = ${session.t.toFixed(3)}, ${ms.toFixed(3)} ms/step`
        : 'state changed',
    );

    // Changing the oversampling in place is display-only: the state survives
    // and the render grid drops back to the solver's.
    await session.setOversample(1);
    const qlmAfterSwap = await session.read('U');
    let stateSurvived = qlmAfterSwap.length === after.length;
    if (stateSurvived) {
      for (let i = 0; i < after.length; i++) {
        if (qlmAfterSwap[i] !== after[i]) {
          stateSurvived = false;
          break;
        }
      }
    }
    session.step(1); // recompute the view fields on the solver grid
    const coarse = await session.readSpecies(0);
    check(
      'setOversample: swaps the render grid without touching the state',
      stateSurvived &&
        session.viewSht === session.sht &&
        coarse.length === session.cfg.nlat * session.cfg.nphi,
      stateSurvived
        ? `state survived, render back to ${session.cfg.nlat}×${session.cfg.nphi}`
        : 'state changed',
    );

    session.destroy();
  }
}
