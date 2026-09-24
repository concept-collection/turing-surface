/**
 * The exact solver (solvers/exact.m, src/mgpu/exact.ts): a dense LU of the
 * implicit operator, assembled and factored on the GPU.
 *
 * What is checked is the chain end to end, against double precision: the
 * matrix the host assembled (read back from the GPU) is factored again here
 * with numbl's LAPACK translation (dgetrf, f64), and the step's own solve
 * must land on that solution to fp32 accuracy and leave a residual of the
 * same order — on the peanut, after a parameter change (which refactors),
 * after a surface swap (which reassembles), and on the sphere, where the
 * matrix collapses to the diagonal and the answer is the plain divide. The
 * iterative solvers are placed against it: their error to the same f64
 * solution must exceed the exact solver's, which is the whole point of
 * having it. The compile-time refusals and the compare-mode bookkeeping
 * (an exact variant has no iteration count) are pinned too.
 */
import { dgetrf } from 'numbl-src/ts-lapack/src/SRC/dgetrf.ts';
import { ModelSession } from '../src/mgpu/session.ts';
import { mModelByKey, defaultParams } from '../src/mgpu/registry.ts';
import { eigenvalues, filterMask } from '../src/mgpu/model.ts';
import { mGeometryByKey, defaultGeometryParams } from '../src/geom/registry.ts';
import { lmIndex } from '../src/sht/layout.ts';
import { crossProduct, mostResolved, variantLabel, labelParts } from '../src/compare/variants.ts';
import type { Check, Log } from './analyticChecks.ts';

const LMAX = 15;

/** Solve A x = b in double precision, A column-major n x n. */
function solveF64(A: Float64Array, n: number, b: ArrayLike<number>): Float64Array {
  const lu = new Float64Array(A);
  const ipiv = new Int32Array(n);
  const info = dgetrf(n, n, lu, n, ipiv);
  if (info !== 0) throw new Error(`dgetrf: info ${info}`);
  const x = Float64Array.from(b);
  for (let i = 0; i < n; i++) {
    const p = ipiv[i] - 1;
    if (p !== i) {
      const t = x[i];
      x[i] = x[p];
      x[p] = t;
    }
  }
  for (let j = 0; j < n; j++) {
    const xj = x[j];
    for (let i = j + 1; i < n; i++) x[i] -= lu[i + j * n] * xj;
  }
  for (let j = n - 1; j >= 0; j--) {
    x[j] /= lu[j + j * n];
    const xj = x[j];
    for (let i = 0; i < j; i++) x[i] -= lu[i + j * n] * xj;
  }
  return x;
}

/** The matrix the exact solver factors, from the operator matrix it built:
 *  A = I - dtD*K on the band, the diagonal M = 1 + dtD*lam/jhat on the
 *  filtered top degrees (src/mgpu/exact.ts, formWGSL). */
function formA(
  K: Float32Array,
  n: number,
  dtD: number,
  jhat: number,
  lam: Float32Array,
  filt: Float32Array,
): Float64Array {
  const A = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) A[i + j * n] = -dtD * K[i + j * n];
    A[j + j * n] += 1 + dtD * (1 - filt[j]) * lam[j] / jhat;
  }
  return A;
}

const relDiff = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let d = 0;
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    d += (a[i] - b[i]) ** 2;
    s += b[i] ** 2;
  }
  return Math.sqrt(d / s);
};

const residual = (A: Float64Array, n: number, x: ArrayLike<number>, b: ArrayLike<number>): number => {
  const r = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const xj = x[j];
    for (let i = 0; i < n; i++) r[i] += A[i + j * n] * xj;
  }
  let d = 0;
  let s = 0;
  for (let i = 0; i < n; i++) {
    d += (r[i] - b[i]) ** 2;
    s += b[i] ** 2;
  }
  return Math.sqrt(d / s);
};

/** The state plus a seeded normal draw of amplitude `amp` in every mode
 *  (imaginary parts at m = 0 stay zero, as a real field's must). */
function roughen(
  state: Record<string, Float32Array>,
  lmax: number,
  amp: number,
): Record<string, Float32Array> {
  let seed = 0x9e3779b9;
  const uniform = (): number => {
    // mulberry32
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = (): number =>
    Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
  const out: Record<string, Float32Array> = {};
  for (const [name, q] of Object.entries(state)) {
    const r = new Float32Array(q);
    for (let m = 0; m <= lmax; m++) {
      for (let l = m; l <= lmax; l++) {
        const i = lmIndex(lmax, l, m);
        r[2 * i] += amp * normal();
        if (m > 0) r[2 * i + 1] += amp * normal();
      }
    }
    out[name] = r;
  }
  return out;
}

const allFinite = (v: Float32Array): boolean => {
  for (const x of v) if (!Number.isFinite(x)) return false;
  return true;
};

export async function exactChecks(device: GPUDevice, check: Check, log: Log): Promise<void> {
  log('--- exact solver (dense LU on the GPU) ---');
  const model = mModelByKey('schnakenberg')!;
  const params = defaultParams(model);
  const peanut = mGeometryByKey('peanut')!;
  const ellipsoid = mGeometryByKey('ellipsoid')!;

  const t0 = performance.now();
  const session = await ModelSession.create({
    device,
    model,
    params,
    lmax: LMAX,
    geometry: peanut,
    geometryParams: defaultGeometryParams(peanut),
    solver: 'exact',
    niter: 0,
  });
  const buildMs = performance.now() - t0;
  const nlm = session.sht.nlm;
  const n = 2 * nlm;
  const cfg = session.cfg;
  const lam = eigenvalues(cfg, nlm);
  const filt = filterMask(cfg, nlm);
  log(`  schnakenberg on peanut, lmax ${LMAX}: n = ${n}, compiled and assembled in ${buildMs.toFixed(0)} ms`);

  const ops = session.describe().step;
  const solves = ops.filter((l) => l.startsWith('lusolve')).length;
  check(
    'exact: each species solve is one lusolve dispatch, nothing iterated',
    solves === 2 && ops.length < 30,
    `${solves} lusolve ops in ${ops.length} ops/step`,
  );

  await session.seed(1);
  session.step(20);
  check('exact: the run stays finite', allFinite(await session.read('u')), '20 steps');

  /** The exact solver's answer for species U against the f64 solve of the
   *  matrix it built: the relative error, the residual, and both for
   *  reference. `state` is loaded first so different solvers face one
   *  right-hand side. */
  const solveCheck = async (
    s: ModelSession,
    what: string,
  ): Promise<{ err: number; res: number }> => {
    s.step(1);
    const Bu = await s.read('Bu');
    const Un = await s.read('Un');
    const K = await s.gpu.exactOperator!.readMatrix();
    const dtD = Math.fround(Math.fround(s.params.dt) * Math.fround(s.params.D1));
    const A = formA(K, n, dtD, Math.fround(s.geometry.Jhat), lam, filt);
    const x = solveF64(A, n, Bu);
    const err = relDiff(Un, x);
    const res = residual(A, n, Un, Bu);
    log(`  ${what}: |X - X64|/|X64| = ${err.toExponential(2)}, |A X - B|/|B| = ${res.toExponential(2)}`);
    return { err, res };
  };

  const state = await session.readState();
  const peanutSolve = await solveCheck(session, 'peanut, default dt');
  check(
    'exact: the GPU solve matches the f64 solve of its own matrix to fp32',
    peanutSolve.err < 1e-4 && peanutSolve.res < 1e-4,
    `error ${peanutSolve.err.toExponential(2)}, residual ${peanutSolve.res.toExponential(2)}`,
  );

  // A parameter change moves dtD, which refactors before the next step.
  session.setParams({ ...params, dt: params.dt / 3 });
  const refactored = await solveCheck(session, 'peanut, dt/3 (refactored)');
  check(
    'exact: a changed dt refactors, and the new solve is exact too',
    refactored.err < 1e-4 && refactored.res < 1e-4,
    `error ${refactored.err.toExponential(2)}, residual ${refactored.res.toExponential(2)}`,
  );
  session.setParams(params);

  // A surface swap reassembles the operator.
  const Kpeanut = await session.gpu.exactOperator!.readMatrix();
  await session.setGeometry(ellipsoid, defaultGeometryParams(ellipsoid));
  const Kell = await session.gpu.exactOperator!.readMatrix();
  const swapped = await solveCheck(session, 'ellipsoid (reassembled)');
  check(
    'exact: a surface swap reassembles the operator and the solve stays exact',
    relDiff(Kell, Kpeanut) > 1e-2 && swapped.err < 1e-4 && swapped.res < 1e-4,
    `|K_ell - K_peanut|/|K_peanut| = ${relDiff(Kell, Kpeanut).toExponential(2)}, ` +
      `error ${swapped.err.toExponential(2)}, residual ${swapped.res.toExponential(2)}`,
  );

  // The iterative solvers, from the same state, against the same f64
  // solution: an exact answer is what they converge toward. At the model's
  // own dt*D the implicit system at this band is nearly the identity and two
  // Richardson iterations already converge below the dense factorization's
  // fp32 rounding (one ulp), so the comparison is made where the solve
  // matters: dt*D*lmax*(lmax+1) ~ 10, on a state roughened with a seeded
  // draw across every mode, so that the geometric part of the operator is
  // large and a few iterations are visibly not enough.
  await session.setGeometry(peanut, defaultGeometryParams(peanut));
  const hard = { ...params, D1: 1, D2: 1, dt: 10 / (cfg.lmax * (cfg.lmax + 1)) };
  const A0 = formA(
    Kpeanut, n,
    Math.fround(Math.fround(hard.dt) * Math.fround(hard.D1)),
    Math.fround(session.geometry.Jhat), lam, filt,
  );
  const rough = roughen(state, cfg.lmax, 0.05);
  session.setParams(hard);
  session.loadState(rough);
  session.step(1);
  const Bu = await session.read('Bu');
  const x64 = solveF64(A0, n, Bu);
  const errs: Record<string, number> = { exact: relDiff(await session.read('Un'), x64) };
  for (const [solver, niter] of [['gmres', 2], ['richardson', 2], ['gmres', 8]] as const) {
    const it = await ModelSession.create({
      device, model, params: hard, lmax: LMAX,
      geometry: peanut, geometryParams: defaultGeometryParams(peanut),
      solver, niter,
    });
    it.loadState(rough);
    it.step(1);
    const same = relDiff(await it.read('Bu'), Bu);
    if (same > 1e-6) throw new Error(`the iterative session saw a different right-hand side (${same})`);
    errs[`${solver}(${niter})`] = relDiff(await it.read('Un'), x64);
    it.destroy();
  }
  log(
    `  error to the f64 solution: ` +
      Object.entries(errs).map(([k, v]) => `${k} ${v.toExponential(2)}`).join(', '),
  );
  check(
    'exact: closer to the f64 solution than every iterative solver at 2 iterations',
    errs.exact < errs['gmres(2)'] && errs.exact < errs['richardson(2)'],
    `exact ${errs.exact.toExponential(2)} vs gmres(2) ${errs['gmres(2)'].toExponential(2)}, ` +
      `richardson(2) ${errs['richardson(2)'].toExponential(2)}`,
  );
  check(
    'exact: gmres closes in on it as niter grows',
    errs['gmres(8)'] < errs['gmres(2)'] && errs.exact < errs['gmres(8)'],
    `gmres(2) ${errs['gmres(2)'].toExponential(2)} -> gmres(8) ${errs['gmres(8)'].toExponential(2)} ` +
      `-> exact ${errs.exact.toExponential(2)}`,
  );
  session.destroy();

  // On the round sphere the operator is zero, the matrix diagonal, and the
  // exact answer the same divide every solver starts from.
  {
    const sphere = await ModelSession.create({
      device, model, params, lmax: LMAX, solver: 'exact', niter: 0,
    });
    await sphere.seed(1);
    sphere.step(5);
    const Bu = await sphere.read('Bu');
    const Un = await sphere.read('Un');
    const dtD = Math.fround(Math.fround(params.dt) * Math.fround(params.D1));
    const divide = new Float64Array(n);
    for (let i = 0; i < n; i++) divide[i] = Bu[i] / (1 + dtD * lam[i] / sphere.geometry.Jhat);
    const err = relDiff(Un, divide);
    check(
      'exact: on the sphere it reproduces the round-sphere divide',
      err < 1e-5,
      `|X - B./M|/|B./M| = ${err.toExponential(2)}`,
    );
    sphere.destroy();
  }

  // Refusals at compile time: a dtD the host cannot evaluate, and an
  // operator argument that is not the app's own array.
  for (const [what, edit, expect] of [
    [
      'a dtD computed on the GPU',
      (src: string) => src.replace(
        'Un = solve(Bu, dt * D1,',
        'sB = dot(Bu, Bu);\n  Un = solve(Bu, dt * D1 * (sB / sB),',
      ),
      "computable from the model's parameters",
    ],
    [
      'a modified operator array',
      (src: string) => src.replace(
        'Un = solve(Bu, dt * D1, lam,',
        'lam2 = 2 * lam;\n  Un = solve(Bu, dt * D1, lam2,',
      ),
      "must be 'lam' itself",
    ],
  ] as const) {
    const source = edit(model.source);
    if (source === model.source) throw new Error(`exact refusal fixture no longer matches schnakenberg.m (${what})`);
    let message = '';
    try {
      const s = await ModelSession.create({
        device, model, params, lmax: LMAX, source, solver: 'exact', niter: 0,
        geometry: peanut, geometryParams: defaultGeometryParams(peanut),
      });
      s.destroy();
    } catch (e) {
      message = e instanceof Error ? e.message : String(e);
    }
    check(
      `exact: ${what} is refused at compile time`,
      message.includes(expect),
      message ? `refused: ${message.slice(0, 90)}…` : 'compiled anyway',
    );
  }

  // Compare-mode bookkeeping: an exact variant carries no iteration count,
  // so the niter axis does not multiply it, its label omits niter, and it
  // is the most-resolved choice at its band.
  {
    const vs = crossProduct(['richardson', 'exact'], [4, 8], [15], [1]);
    const exact = vs.filter((v) => v.solver === 'exact');
    const labels = vs.map((v) => variantLabel(v, labelParts(vs)));
    check(
      'exact: one compare variant per band, listed last and chosen as the reference',
      exact.length === 1 && vs[vs.length - 1].solver === 'exact' &&
        vs[mostResolved(vs)].solver === 'exact' && labels[labels.length - 1] === 'exact · lmax 15',
      labels.join(' | '),
    );
    const mixed = crossProduct(['exact', 'gmres'], [8], [15, 31], [1]);
    check(
      'exact: a finer band still outranks it for the reference',
      vs.length === 3 && mixed[mostResolved(mixed)].lmax === 31,
      mixed.map((v) => variantLabel(v, labelParts(mixed))).join(' | '),
    );
  }
}
