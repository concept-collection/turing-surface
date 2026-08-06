/**
 * Browser validation, in the environment the demo actually ships to.
 *
 * Runs the same four check modules as `npm run test:node` — so both GPU stacks
 * (Dawn on the desktop, the browser's own here) get the same guarantees — plus a
 * long soak that only makes sense in a page.
 *
 * Results are posted to window.__RESULTS__ for the headless runner.
 */
import { requestShtDevice, describeAdapter } from '../src/sht/sht.ts';
import { ModelSession } from '../src/mgpu/session.ts';
import { mModels, defaultParams } from '../src/mgpu/registry.ts';
import { digestOf, formatDigest, type StateDigest } from '../src/mgpu/digest.ts';
import {
  parseArgs,
  modelForSpec,
  geometryForSpec,
  formatCommand,
  DEFAULT_NITER,
} from '../src/bench/runSpec.ts';
import {
  mGeometryByKey,
  defaultGeometryParams,
  DEFAULT_GEOMETRY_KEY,
} from '../src/geom/registry.ts';
import { transformChecks } from './transformChecks.ts';
import { analyticChecks } from './analyticChecks.ts';
import { modelChecks } from './modelChecks.ts';
import { geometryChecks } from './geometryChecks.ts';
import { fluxChecks } from './fluxChecks.ts';
import { compareChecks } from './compareChecks.ts';

declare global {
  interface Window {
    __RESULTS__?: { ok: boolean; fatal?: string; lines: string[] };
    /** Set by the ?state= mode, for scripts/compare-env.mjs. */
    __STATE__?: { digest: StateDigest; state: number[] };
    /** Set by the ?soak= mode, for scripts/compare-perf.mjs. */
    __SOAK__?: {
      lmax: number;
      steps: number;
      batch: number;
      solverMsPerStep: number;
      encodeMsPerStep: number;
      adapter: string;
      fourier: 'fft' | 'dft';
    };
  }
}

const logEl = document.getElementById('log')!;
const lines: string[] = [];
let failures = 0;

function log(s: string): void {
  lines.push(s);
  logEl.textContent = lines.join('\n');
  console.log(s);
}

function check(name: string, ok: boolean, detail: string): void {
  log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`);
  if (!ok) failures++;
}

/**
 * Solver-only soak, selected with ?soak=<steps>&lmax=<n>.
 *
 * No three.js at all, so this is the browser's honest solver rate: the same
 * batched, no-readback measurement the desktop benchmark reports. If this number
 * matches the benchmark's but the app's frame cost does not, the difference is
 * the readback and competing with the renderer for the GPU, not the computation.
 */
async function soak(steps: number, lmax: number): Promise<void> {
  const device = await requestShtDevice();
  const model = mModels[0];
  // The desktop benchmark this is compared against resolves its geometry and
  // iteration count from the same two constants. They have to agree: the
  // iteration count is unrolled into the step, so a mismatch would compare
  // two different amounts of work and call the difference "the browser".
  const geometry = mGeometryByKey(DEFAULT_GEOMETRY_KEY)!;
  const session = await ModelSession.create({
    device,
    model,
    params: defaultParams(model),
    lmax,
    geometry,
    geometryParams: defaultGeometryParams(geometry),
    niter: DEFAULT_NITER,
  });
  session.seed(5);
  log(
    `soak: ${steps} steps at lmax ${lmax} ` +
      `(grid ${session.cfg.nlat}x${session.cfg.nphi}, ${geometry.key}, ` +
      `${DEFAULT_NITER} solve iter, ${session.describe().step.length} ops/step), solver only`,
  );

  const BATCH = 25;
  // Timed separately from the sampling: `solverMs` counts only submitted steps
  // waited for, never read back, so it is comparable to `npm run bench`.
  let solverMs = 0;
  let solverSteps = 0;
  let encodeMs = 0;
  const t0 = performance.now();
  for (let s = 0; s < steps; s += BATCH) {
    const n = Math.min(BATCH, steps - s);
    const b0 = performance.now();
    session.step(n);
    // CPU-side command encoding, separated from GPU execution: in a browser each
    // WebGPU call crosses Blink's bindings and Dawn's validation, so on a fast
    // GPU the encoding can be what actually limits the step rate.
    const b1 = performance.now();
    encodeMs += b1 - b0;
    await session.sync();
    solverMs += performance.now() - b0;
    solverSteps += n;
    const u = await session.read(model.species[0]);
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of u) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if ((s + BATCH) % 100 === 0) {
      const mem = (performance as Performance & { memory?: { usedJSHeapSize: number } })
        .memory;
      log(
        `  step ${session.steps}  u in [${lo.toFixed(4)}, ${hi.toFixed(4)}]` +
          (mem ? `  heap ${(mem.usedJSHeapSize / 1048576).toFixed(1)} MB` : ''),
      );
      // yield so the page stays responsive and the runner can poll
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  const ms = (performance.now() - t0) / steps;

  const final = await session.read(model.species[0]);
  let finite = true;
  for (const v of final) if (!Number.isFinite(v)) finite = false;
  const solverPerStep = solverMs / solverSteps;
  const encodePerStep = encodeMs / solverSteps;
  check(
    `soak: ${steps} steps survived`,
    finite,
    `solver ${solverPerStep.toFixed(2)} ms/step (batches of ${BATCH}, no readback), ` +
      `of which ${encodePerStep.toFixed(3)} ms/step CPU encoding, ` +
      `${ms.toFixed(2)} ms/step incl. sampling readback`,
  );
  log(
    `  compare 'solver' with the ms/step from \`npm run bench -- --lmax ${lmax}\`:\n` +
      `  same .m, same kernels, no rendering on either side.`,
  );

  window.__SOAK__ = {
    lmax,
    steps,
    batch: BATCH,
    solverMsPerStep: solverPerStep,
    encodeMsPerStep: encodePerStep,
    adapter: await describeAdapter(device),
    fourier: session.sht.fourierMode,
  };
  session.destroy();
  window.__RESULTS__ = { ok: failures === 0, lines };
  log(failures === 0 ? 'ALL PASS' : `${failures} FAILURE(S)`);
}

/**
 * Run one exact spec and post its final state, for scripts/compare-env.mjs to
 * compare against the same spec run on the desktop. Query parameters map
 * straight onto the benchmark's flags — `?state=1&lmax=31&steps=200` — and go
 * through the same parseArgs, so neither side can quietly use different
 * defaults.
 */
async function dumpState(q: URLSearchParams): Promise<void> {
  const argv: string[] = [];
  for (const [k, v] of q) {
    if (k === 'state') continue;
    argv.push(`--${k}`, v);
  }
  const spec = parseArgs(argv);
  const model = modelForSpec(spec);

  const device = await requestShtDevice();
  const adapter = await describeAdapter(device);
  const session = await ModelSession.create({
    device,
    model,
    params: spec.params,
    lmax: spec.lmax,
    geometry: geometryForSpec(spec),
    geometryParams: spec.geometryParams,
    niter: spec.niter,
  });
  session.seed(spec.seed);
  session.step(spec.steps);
  await session.sync();
  const state = await session.read(model.state[0]);
  const digest = digestOf(state, session.sht.fourierMode, adapter);

  log(`${formatCommand(spec)}\n`);
  log(`state after ${spec.steps} steps from seed ${spec.seed}:`);
  log(`  ${formatDigest(digest)}`);
  log(`  adapter: ${adapter}`);
  window.__STATE__ = { digest, state: [...state] };
  session.destroy();
}

async function main(): Promise<void> {
  const q = new URLSearchParams(location.search);
  if (q.has('state')) return dumpState(q);
  if (q.has('soak')) {
    return soak(Number(q.get('soak')) || 500, Number(q.get('lmax')) || 63);
  }
  const device = await requestShtDevice();

  await transformChecks(device, check, log);
  await analyticChecks(device, check, log);
  await modelChecks(device, check, log);
  // The sweep is opt-in here (?sweep=1): it is a few seconds on desktop Dawn
  // but minutes in a browser, where each session recompiles its unrolled step.
  await geometryChecks(device, check, log, { sweep: q.has('sweep') });
  await fluxChecks(device, check, log, { ab: q.has('sweep') });
  await compareChecks(device, check, log);

  window.__RESULTS__ = { ok: failures === 0, lines };
  log(failures === 0 ? 'ALL PASS' : `${failures} FAILURE(S)`);
}

main().catch((e) => {
  const msg = e instanceof Error ? `${e.message}\n${e.stack ?? ''}` : String(e);
  log(`fatal: ${msg}`);
  window.__RESULTS__ = { ok: false, fatal: msg, lines };
});
