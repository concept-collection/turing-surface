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
 * unrolled into the fixed op sequence, so more iterations means more GPU ops.
 * On the sphere, where the surface Laplace-Beltrami correction is
 * mathematically zero (lap_g = lap_s exactly), the answer must stay close
 * across niter to fp32 tolerance — not bit-identical, since the correction is
 * now a real (if numerically near-zero) computation rather than the literal
 * `0 * Un` placeholder, so the op sequence differs even though the answer
 * shouldn't move much. On a genuinely curved surface the correction must
 * actually change the answer, and — since the Richardson iteration only
 * converges while the correction stays small relative to what the
 * round-sphere solve inverts (docs/richardson-iteration.md) — a niter/dt/
 * geometry combination outside that radius is expected to diverge. The
 * niter x geometry sweep below documents which shipped combinations that
 * currently affects, so a regression that makes a *currently-healthy*
 * combination diverge is caught without this file silently asserting away a
 * real, known numerical limit.
 */
import { ShtPlan } from '../src/sht/sht.ts';
import { DerivPlan } from '../src/sht/deriv.ts';
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
/** The app's actual default lmax (README: "at the default lmax 63 that is a
 *  128x256 grid"), used for the niter/geometry sweep below and the peanut
 *  check next to it -- the divergence they're both about is a real, lmax-
 *  dependent numerical property of the Richardson iteration, not one this
 *  file's other, smaller LMAX happens to reproduce. */
const SWEEP_LMAX = 63;

/** Build one geometry on its own transform plan, for inspection. */
async function buildGeometry(device: GPUDevice, key: string) {
  const g = mGeometryByKey(key)!;
  const { nlat, nphi } = gridForLmax(LMAX, 3);
  const cfg = { lmax: LMAX, mmax: LMAX, nlat, nphi };
  const sht = await ShtPlan.create(device, cfg);
  const deriv = await DerivPlan.create(device, sht);
  const geometry = await Geometry.create({
    device,
    sht,
    cfg,
    source: g.source,
    paramNames: g.params.map((p) => p.key),
    params: defaultGeometryParams(g),
    deriv,
  });
  return { g, sht, deriv, cfg, geometry };
}

export interface GeometryCheckOptions {
  /**
   * Run the niter x geometry sweep at the end. On by default, and nearly free
   * on desktop Dawn (~3 s for all 20 combinations), but in a browser it
   * dominates the whole suite: every session recompiles its unrolled step from
   * scratch — there is no pipeline cache across sessions — so the sweep costs
   * ~6 minutes on software WebGPU against ~30 s for every other check here put
   * together. The browser page therefore leaves it out unless asked (?sweep=1),
   * which is what keeps CI short.
   */
  sweep?: boolean;
}

export async function geometryChecks(
  device: GPUDevice,
  check: Check,
  log: Log,
  opts: GeometryCheckOptions = {},
): Promise<void> {
  const runSweep = opts.sweep ?? true;
  // ---- every geometry compiles and closes ---------------------------------
  for (const spec of mGeometries) {
    const { sht, deriv, geometry } = await buildGeometry(device, spec.key);
    let finite = true;
    for (const a of [geometry.x, geometry.y, geometry.z]) {
      for (const v of a) if (!Number.isFinite(v)) finite = false;
    }
    for (const a of [geometry.Vtx, geometry.Vty, geometry.Vtz, geometry.Vpx, geometry.Vpy, geometry.Vpz]) {
      for (const v of a) if (!Number.isFinite(v)) finite = false;
    }
    const { lo, hi } = geometry.radiusRange();
    check(
      `geometry: ${spec.key}.m evaluates to a finite surface`,
      finite && lo > 1e-3,
      `radius ${lo.toFixed(4)}–${hi.toFixed(4)}`,
    );
    deriv.destroy();
    sht.destroy();
  }

  // ---- the sphere is the unit sphere, exactly, and is degree 1 ------------
  {
    const { sht, deriv, geometry } = await buildGeometry(device, SPHERE_KEY);

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

    // The inverse metric quantities have a closed form on the unit sphere:
    // V_theta = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta)),
    // V_phi = (-sin(phi)/sin(theta), cos(phi)/sin(theta), 0). Checking these
    // pins the sign convention of computeMetric (src/geom/metric.ts) before
    // it is buried under the Laplace-Beltrami operator built on top of it.
    //
    // Split by latitude, because the accuracy available here is not uniform.
    // computeMetric divides by det = g_tt*g_pp - g_tp^2, and on the sphere
    // g_pp and det are both O(sin^2 theta) -- 1.4e-3 at the outermost Gauss
    // latitude of this grid. The transforms deliver X_theta/X_phi with an
    // *absolute* fp32 error of ~1e-6, which is a large *relative* error once
    // it is squared into quantities that small, so the error in both V's grows
    // like 1/sin^2 theta toward the poles. That is conditioning, not a wrong
    // formula: measured worst cases are
    //
    //             sin(theta) >= 0.2    sin(theta) < 0.2 (8 of 64 latitudes)
    //   Dawn            3.2e-4               1.7e-3
    //   SwiftShader     1.5e-3               8.8e-3
    //
    // and a sign or formula error would be O(1) in either band, so tolerances
    // a few times the looser stack still catch one.
    const POLE_SIN = 0.2;
    let maxMetricErr = 0;
    let maxPoleErr = 0;
    for (let i = 0; i < sht.cfg.nlat; i++) {
      const ct = sht.cosTheta[i];
      const st = Math.sqrt(Math.max(0, 1 - ct * ct));
      for (let j = 0; j < sht.cfg.nphi; j++) {
        const phi = (2 * Math.PI * j) / sht.cfg.nphi;
        const k = i * sht.cfg.nphi + j;
        const cphi = Math.cos(phi);
        const sphi = Math.sin(phi);
        const wantVtx = ct * cphi;
        const wantVty = ct * sphi;
        const wantVtz = -st;
        const wantVpx = -sphi / st;
        const wantVpy = cphi / st;
        const wantVpz = 0;
        const worst = Math.max(
          Math.abs(geometry.Vtx[k] - wantVtx),
          Math.abs(geometry.Vty[k] - wantVty),
          Math.abs(geometry.Vtz[k] - wantVtz),
          Math.abs(geometry.Vpx[k] - wantVpx),
          Math.abs(geometry.Vpy[k] - wantVpy),
          Math.abs(geometry.Vpz[k] - wantVpz),
        );
        if (st < POLE_SIN) maxPoleErr = Math.max(maxPoleErr, worst);
        else maxMetricErr = Math.max(maxMetricErr, worst);
      }
    }
    check(
      'geometry: sphere.m has the closed-form inverse metric quantities',
      maxMetricErr < 4e-3,
      `max |V - closed form| = ${maxMetricErr.toExponential(2)} ` +
        `away from the poles (sin theta >= ${POLE_SIN})`,
    );
    check(
      'geometry: the polar caps stay within their conditioning',
      maxPoleErr < 2e-2,
      `max |V - closed form| = ${maxPoleErr.toExponential(2)} at sin theta < ${POLE_SIN}`,
    );

    deriv.destroy();
    sht.destroy();
  }

  // ---- a deformed surface matches its own formula, on any grid ------------
  {
    const { g, sht, deriv, cfg, geometry } = await buildGeometry(device, 'peanut');
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
    deriv.destroy();
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
    // once per iteration, no more and no less. Per species per iteration: 8
    // dtheta/dphi + 4 analys transforms (Algorithm 3's cost, applied to the
    // field and to each of its three Cartesian gradient components) plus 15
    // generated kernels -- see test/modelChecks.ts's KERNELS_PER_ITERATION,
    // which counts the kernels alone; this counts every op, transforms
    // included.
    const perIteration = ops[1] - ops[0];
    const want = 54;
    check(
      'loop: unrolling is exactly linear in the trip count',
      perIteration === want && ops[2] - ops[0] === 4 * perIteration,
      `${perIteration} ops per iteration (expected ${want}), ` +
        `${ops[2] - ops[0]} for 4 iterations`,
    );

    // On the sphere lap_g = lap_s exactly, so the correction should compute
    // (numerically) close to zero regardless of niter -- not bit-identical
    // (it is a real computation now, through 8+ chained fp32 transforms per
    // iteration, not the literal `0 * Un` placeholder that used to make this
    // exact), but close. The tolerance is set by that chain's fp32 roundoff,
    // not by the scheme: a real geometry-correction bug would miss by orders
    // of magnitude more than this.
    let worst = 0;
    for (let k = 1; k < states.length; k++) {
      for (let i = 0; i < states[0].length; i++) {
        worst = Math.max(worst, Math.abs(states[k][i] - states[0][i]));
      }
    }
    check(
      'loop: on the sphere, the correction stays near zero across niter',
      worst < 2e-3,
      `states differ by up to ${worst.toExponential(2)} after ${STEPS} steps at ${counts.join('/')} iterations`,
    );
  }

  // ---- on a curved surface, the correction actually changes the answer ----
  {
    const model = mModelByKey('schnakenberg')!;
    const params = defaultParams(model);
    const peanut = mGeometryByKey('peanut')!;
    const peanutParams = defaultGeometryParams(peanut);
    // niter 0 vs 1 only -- deliberately not the 4/8 the sweep below already
    // documents as outside the Richardson iteration's convergence radius on
    // this geometry. The point here is just that the correction is not a
    // no-op, which a much smaller, still-converging niter already shows.
    const states: Float32Array[] = [];
    for (const niter of [0, 1]) {
      const session = await ModelSession.create({
        device, model, params, lmax: SWEEP_LMAX,
        geometry: peanut, geometryParams: peanutParams, niter,
      });
      session.seed(1);
      session.step(STEPS);
      states.push(await session.read('U'));
      session.destroy();
    }
    let worst = 0;
    for (let i = 0; i < states[0].length; i++) {
      worst = Math.max(worst, Math.abs(states[1][i] - states[0][i]));
    }
    check(
      'loop: on peanut, the correction measurably changes the answer',
      worst > 1e-4 && states[1].every((v) => Number.isFinite(v)),
      `states differ by ${worst.toExponential(2)} after ${STEPS} steps at niter 0 vs 1`,
    );
  }

  // ---- niter x geometry sweep: catch a "doesn't run" regression early -----
  // This is what actually turned up the two real issues found while building
  // the correction: peanut diverging at niter >= 4 with schnak-spots'
  // shipped default dt (a genuine Richardson-convergence-radius limit, not a
  // bug -- see docs/richardson-iteration.md), and a since-fixed compiler bug
  // where a loop-body statement could silently reuse a *different*
  // statement's compiled kernel (test/modelChecks.ts's pipeline-cache check
  // guards that one directly). Every shipped geometry x every niter the
  // app's <select> actually offers, so a regression anywhere in that grid is
  // caught -- without asserting away the one combination already known to be
  // outside the convergence radius.
  //
  // SWEEP_LMAX cannot be lowered to make this cheaper: at lmax 31 or 15 the
  // peanut/2, /4 and /8 combinations below all stay finite, so a smaller grid
  // would quietly turn KNOWN_DIVERGENT into three failures and stop testing
  // the thing this sweep exists to pin down.
  if (!runSweep) {
    log(
      '  sweep: skipped — run `npm run test:node` (desktop Dawn, ~3 s) or ' +
        '`npm run test:gpu -- --sweep` for the niter x geometry sweep.',
    );
  } else {
    const model = mModelByKey('schnakenberg')!;
    const params = defaultParams(model);
    const SWEEP_NITER = [0, 1, 2, 4, 8];
    const KNOWN_DIVERGENT = new Set(['peanut/2', 'peanut/4', 'peanut/8']);

    for (const geomSpec of mGeometries) {
      for (const niter of SWEEP_NITER) {
        const session = await ModelSession.create({
          device, model, params, lmax: SWEEP_LMAX,
          geometry: geomSpec, geometryParams: defaultGeometryParams(geomSpec),
          niter,
        });
        session.seed(1);
        session.step(STEPS);
        const values = await session.read('u');
        const finite = values.every((v) => Number.isFinite(v));
        session.destroy();

        const key = `${geomSpec.key}/${niter}`;
        const expectDivergent = KNOWN_DIVERGENT.has(key);
        check(
          expectDivergent
            ? `sweep: ${key} is known to diverge (outside the Richardson convergence radius)`
            : `sweep: ${key} stays finite after ${STEPS} steps`,
          expectDivergent ? !finite : finite,
          expectDivergent
            ? finite
              ? 'now finite -- the convergence radius may have improved; update KNOWN_DIVERGENT'
              : 'diverged as expected'
            : finite
              ? 'finite'
              : 'NOT FINITE -- unexpected divergence, investigate before treating this as another known case',
        );
      }
    }
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
