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
import { mModels, defaultParams } from '../src/mgpu/registry.ts';
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
  schnakenberg: 7,
  brusselator: 7,
  allencahn: 3,
};

/**
 * And what one unrolled iteration of the solve loop adds, per species: the
 * placeholder line that will become the geometry correction, and the update
 * that consumes it. Two rather than one because the correction does not fuse
 * into its consumer — which is right, since the operator that replaces it will
 * be transforms and kernels of its own, not an expression.
 */
const KERNELS_PER_ITERATION = 2;

const LMAX = 31;
const STEPS = 40;
const NITER = 1;

export async function modelChecks(
  device: GPUDevice,
  check: Check,
  log: Log,
): Promise<void> {
  check('models: registry populated', mModels.length === 3, `${mModels.length} models`);

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
    const expected =
      EXPECTED_KERNELS[model.key] +
      NITER * KERNELS_PER_ITERATION * model.species.length;
    log(
      `  ${model.key}.m -> ${plan.step.length} ops/step ` +
        `(${kernels} generated kernels, ${xforms} transforms, ${NITER} solve iter)`,
    );
    check(
      `${model.key}: element-wise lines fused into one kernel each`,
      kernels === expected,
      `${kernels} kernels (expected ${expected})`,
    );

    session.seed(1);
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
    session.seed(1);
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
