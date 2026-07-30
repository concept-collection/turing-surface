/**
 * The two things this project adds to turing-sphere: a surface, and a `for`
 * loop in the compiled step.
 *
 * The surface is checked against what it is supposed to be — the sphere really
 * is the unit sphere and really is degree 1, a deformed shape really has the
 * radius profile its .m says, and the coefficients really do evaluate to the
 * same surface on a finer grid.
 *
 * The loop is checked for the property the whole design rests on: it is
 * unrolled into the fixed op sequence, so more iterations means more GPU ops —
 * and, while the geometry correction inside it is identically zero, the answer
 * must be *bit for bit* independent of how many times it runs. That is a
 * stronger statement than "close enough": if the placeholder were ever
 * something that merely rounds to zero, or if the loop were miscompiled to
 * read a stale buffer, these would differ in the last bits and this fails.
 */
import { ShtPlan } from '../src/sht/sht.ts';
import { gridForLmax, lmIndex } from '../src/sht/layout.ts';
import { ModelSession } from '../src/mgpu/session.ts';
import { mModelByKey, defaultParams } from '../src/mgpu/registry.ts';
import { Geometry } from '../src/geom/geometry.ts';
import {
  mGeometries,
  mGeometryByKey,
  defaultGeometryParams,
  SPHERE_KEY,
} from '../src/geom/registry.ts';
import { ModelCompileError } from '../src/mgpu/errors.ts';
import type { Check, Log } from './analyticChecks.ts';

const LMAX = 31;
const STEPS = 20;

/** Build one geometry on its own transform plan, for inspection. */
async function buildGeometry(device: GPUDevice, key: string) {
  const g = mGeometryByKey(key)!;
  const { nlat, nphi } = gridForLmax(LMAX, 3);
  const cfg = { lmax: LMAX, mmax: LMAX, nlat, nphi };
  const sht = await ShtPlan.create(device, cfg);
  const geometry = await Geometry.create({
    device,
    sht,
    cfg,
    source: g.source,
    paramNames: g.params.map((p) => p.key),
    params: defaultGeometryParams(g),
  });
  return { g, sht, cfg, geometry };
}

export async function geometryChecks(
  device: GPUDevice,
  check: Check,
  log: Log,
): Promise<void> {
  // ---- every geometry compiles and closes ---------------------------------
  for (const spec of mGeometries) {
    const { sht, geometry } = await buildGeometry(device, spec.key);
    let finite = true;
    for (const a of [geometry.x, geometry.y, geometry.z]) {
      for (const v of a) if (!Number.isFinite(v)) finite = false;
    }
    const { lo, hi } = geometry.radiusRange();
    check(
      `geometry: ${spec.key}.m evaluates to a finite surface`,
      finite && lo > 1e-3,
      `radius ${lo.toFixed(4)}–${hi.toFixed(4)}`,
    );
    sht.destroy();
  }

  // ---- the sphere is the unit sphere, exactly, and is degree 1 ------------
  {
    const { sht, geometry } = await buildGeometry(device, SPHERE_KEY);

    let maxRadiusErr = 0;
    for (let i = 0; i < geometry.x.length; i++) {
      const r = Math.hypot(geometry.x[i], geometry.y[i], geometry.z[i]);
      maxRadiusErr = Math.max(maxRadiusErr, Math.abs(r - 1));
    }
    // Tolerance is fp32 through a full analysis/synthesis round trip, not the
    // geometry: the exact answer is representable, and what is measured here
    // is the transforms' own round-off. It is set by the loosest stack this
    // runs on — SwiftShader in CI is an order of magnitude worse than Dawn on
    // real hardware (4e-4 against 2e-5). A geometry that was actually wrong
    // would miss by O(1), so the slack costs nothing.
    check(
      'geometry: sphere.m has radius 1 everywhere',
      maxRadiusErr < 2e-3,
      `max |r - 1| = ${maxRadiusErr.toExponential(2)}`,
    );

    // x, y, z of the unit sphere are the three degree-1 harmonics and nothing
    // else, so analysing them must leave every other coefficient at zero.
    // This is what makes the sphere case exact rather than merely accurate:
    // there is no content for the band limit to throw away.
    const degreeOne = new Set([
      lmIndex(LMAX, 1, 0),
      lmIndex(LMAX, 1, 1),
    ]);
    let leak = 0;
    for (const coeffs of [geometry.X, geometry.Y, geometry.Z]) {
      for (let i = 0; i < coeffs.length / 2; i++) {
        if (degreeOne.has(i)) continue;
        leak = Math.max(leak, Math.abs(coeffs[2 * i]), Math.abs(coeffs[2 * i + 1]));
      }
    }
    check(
      'geometry: sphere.m is exactly degree 1 in the harmonics',
      leak < 1e-3,
      `max |coefficient| outside l = 1 is ${leak.toExponential(2)}`,
    );
    sht.destroy();
  }

  // ---- a deformed surface matches its own formula, on any grid ------------
  {
    const { g, sht, cfg, geometry } = await buildGeometry(device, 'peanut');
    const p = defaultGeometryParams(g);

    // peanut.m written out: r = 1 - waist*sin(theta)^2 scales the unit sphere,
    // and z is then stretched, so the distance from the origin depends on
    // theta alone. Checking every point against this closed form checks the
    // whole path at once — the compiled shape kernel, the analysis into
    // coefficients, the synthesis back — and, because the formula has no phi
    // in it, that the surface really is a surface of revolution.
    const peanutRadius = (ct: number): number => {
      const st2 = Math.max(0, 1 - ct * ct);
      const r = 1 - p.waist * st2;
      return r * Math.hypot(Math.sqrt(st2), (1 + p.stretch) * ct);
    };

    const onGrid = (
      cosTheta: Float64Array,
      nlat: number,
      nphi: number,
      at: (i: number) => number,
    ): number => {
      let worst = 0;
      for (let i = 0; i < nlat; i++) {
        const want = peanutRadius(cosTheta[i]);
        for (let j = 0; j < nphi; j++) {
          worst = Math.max(worst, Math.abs(at(i * nphi + j) - want));
        }
      }
      return worst;
    };

    const coarse = onGrid(sht.cosTheta, cfg.nlat, cfg.nphi, (k) =>
      Math.hypot(geometry.x[k], geometry.y[k], geometry.z[k]),
    );
    check(
      'geometry: peanut.m matches its own radial formula on the solver grid',
      coarse < 1e-3,
      `max |dr| = ${coarse.toExponential(2)}`,
    );

    // And the same on a finer grid, from the same coefficients. This is what
    // "the rendered surface is the surface being solved on" means: display
    // oversampling evaluates the embedding at more points, it does not
    // subdivide or smooth it. The 2x Gauss latitudes share no point with the
    // 1x ones, so agreeing here is agreeing everywhere, not at samples.
    const fine = await ShtPlan.create(device, {
      lmax: cfg.lmax,
      mmax: cfg.mmax,
      nlat: 2 * cfg.nlat,
      nphi: 2 * cfg.nphi,
    });
    const finePos = await geometry.positionsOn(fine);
    const refined = onGrid(fine.cosTheta, 2 * cfg.nlat, 2 * cfg.nphi, (k) =>
      Math.hypot(finePos[3 * k], finePos[3 * k + 1], finePos[3 * k + 2]),
    );
    check(
      'geometry: the same coefficients give the same surface on a 2x grid',
      refined < 1e-3,
      `max |dr| = ${refined.toExponential(2)} at ${2 * cfg.nlat}×${2 * cfg.nphi} points`,
    );
    fine.destroy();
    sht.destroy();
  }

  // ---- the unrolled loop: more ops, identical answer ----------------------
  {
    const model = mModelByKey('schnakenberg')!;
    const params = defaultParams(model);
    const counts = [0, 1, 4];
    const ops: number[] = [];
    const states: Float32Array[] = [];

    for (const niter of counts) {
      const session = await ModelSession.create({
        device, model, params, lmax: LMAX, niter,
      });
      ops.push(session.describe().step.length);
      session.seed(1);
      session.step(STEPS);
      states.push(await session.read('U'));
      session.destroy();
    }

    log(`  schnakenberg.m ops/step by solve iterations: ${
      counts.map((n, i) => `${n} -> ${ops[i]}`).join(', ')
    }`);
    check(
      'loop: each solve iteration adds GPU operations',
      ops[0] < ops[1] && ops[1] < ops[2],
      `${ops.join(' < ')} ops for ${counts.join(', ')} iterations`,
    );
    // Unrolling has to be exactly linear in the trip count: the body planned
    // once per iteration, no more and no less. Two dispatches per species per
    // iteration — the placeholder line and the update that reads it.
    const perIteration = ops[1] - ops[0];
    const want = 2 * model.species.length;
    check(
      'loop: unrolling is exactly linear in the trip count',
      perIteration === want && ops[2] - ops[0] === 4 * perIteration,
      `${perIteration} ops per iteration (expected ${want}), ` +
        `${ops[2] - ops[0]} for 4 iterations`,
    );

    let identical = true;
    let worst = 0;
    for (let k = 1; k < states.length; k++) {
      if (states[k].length !== states[0].length) identical = false;
      for (let i = 0; i < states[0].length; i++) {
        if (states[k][i] !== states[0][i]) identical = false;
        worst = Math.max(worst, Math.abs(states[k][i] - states[0][i]));
      }
    }
    check(
      'loop: the geometry correction is exactly zero, so the answer does not move',
      identical,
      identical
        ? `bit-identical after ${STEPS} steps at ${counts.join('/')} iterations`
        : `states differ by up to ${worst.toExponential(2)}`,
    );
  }

  // ---- a loop whose length is not known at compile time is refused --------
  {
    const model = mModelByKey('allencahn')!;
    // `dt` is a tunable parameter, so it reaches the compiler with no value:
    // the plan cannot know how many iterations to emit.
    const bad = model.source.replace('for k = 1:niter', 'for k = 1:dt');
    let message = '';
    try {
      const session = await ModelSession.create({
        device, model, params: defaultParams(model), lmax: LMAX, source: bad, niter: 1,
      });
      session.destroy();
    } catch (e) {
      message = e instanceof ModelCompileError ? e.message : `wrong error type: ${e}`;
    }
    check(
      'loop: a runtime loop bound is refused at compile time',
      message.includes('known when the model is compiled'),
      message ? `refused: ${message.slice(0, 72)}…` : 'compiled anyway',
    );
  }

  // ---- swapping the surface leaves the simulation alone ------------------
  {
    const model = mModelByKey('schnakenberg')!;
    const session = await ModelSession.create({
      device, model, params: defaultParams(model), lmax: LMAX,
    });
    session.seed(1);
    session.step(STEPS);
    const before = await session.read('U');

    const peanut = mGeometryByKey('peanut')!;
    await session.setGeometry(peanut, defaultGeometryParams(peanut));
    const after = await session.read('U');

    let survived = before.length === after.length;
    for (let i = 0; survived && i < before.length; i++) {
      if (before[i] !== after[i]) survived = false;
    }
    const { lo, hi } = session.geometry.radiusRange();
    check(
      'geometry: swapping the surface mid-run does not disturb the state',
      survived && session.geometryModel.key === 'peanut' && hi - lo > 0.1,
      survived
        ? `state identical, now on ${session.geometryModel.key} (radius ${lo.toFixed(3)}–${hi.toFixed(3)})`
        : 'state changed',
    );
    session.destroy();
  }
}

/** Index of the entry minimizing `score`, over the first `n` entries. */
function argMin(
  xs: Float64Array | Float32Array,
  n: number,
  score: (v: number) => number,
): number {
  let best = 0;
  let bestScore = Infinity;
  for (let i = 0; i < n; i++) {
    const s = score(xs[i]);
    if (s < bestScore) {
      bestScore = s;
      best = i;
    }
  }
  return best;
}
