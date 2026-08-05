/**
 * Command-line benchmark: run exactly what the browser runs — the same .m
 * models, lowered by numbl and compiled to the same WGSL kernels, over the same
 * transforms — on desktop WebGPU (Google Dawn, via the optional `webgpu`
 * package), and report ms/step. The app prints the matching command under its
 * stats line; copy it and run it here for an apples-to-apples comparison.
 *
 *   npm run bench -- --preset schnak-spots --lmax 63 --steps 2000 --seed 1 \
 *     --a 0.1 --b 0.9 --D1 0.0004 --D2 0.008 --dt 0.05
 *
 * The only thing missing here is the rendering: this is the solver alone.
 *
 * Two numbers are reported, because they answer different questions:
 *  - throughput: a batch of steps submitted together, awaited once. This is how
 *    the app runs, and what keeping the state in GPU buffers is for.
 *  - latency: one step per submit, each awaited. Comparable to a design that
 *    reads back every step, and the only way to get a per-step distribution.
 */
import { requestShtDevice, describeAdapter } from '../src/sht/sht.ts';
import { ModelSession } from '../src/mgpu/session.ts';
import { presets } from '../src/mgpu/registry.ts';
import { mGeometries, DEFAULT_GEOMETRY_KEY } from '../src/geom/registry.ts';
import { DEFAULT_SOLVER, solverKeys } from '../src/mgpu/libs.ts';
import {
  parseArgs,
  modelForSpec,
  resolvePreset,
  geometryForSpec,
  formatCommand,
  BENCH_COMMAND,
  DEFAULT_LMAX,
  DEFAULT_NITER,
  DEFAULT_SEED,
  DEFAULT_STEPS,
  DEFAULT_WARMUP,
  type RunSpec,
} from '../src/bench/runSpec.ts';
import { digestOf, formatDigest } from '../src/mgpu/digest.ts';
import { installWebGpu, errMsg, NO_ADAPTER_HINT } from './nodeWebGpu.ts';
import { writeFileSync } from 'node:fs';

const USAGE = `usage: ${BENCH_COMMAND} [options]

  --preset <key>    ${presets.map((p) => p.key).join(' | ')}
                    (default ${presets[0].key})
  --geometry <key>  ${mGeometries.map((g) => g.key).join(' | ')}
                    (default ${DEFAULT_GEOMETRY_KEY})
  --lmax <n>        spherical harmonic truncation (default ${DEFAULT_LMAX})
  --niter <n>       iterations of the implicit solve, unrolled into the compiled
                      step (default ${DEFAULT_NITER})
  --solver <key>    ${solverKeys.join(' | ')} — which solver answers the
                      models' solve(...) call (default ${DEFAULT_SOLVER})
  --steps <n>       timed steps (default ${DEFAULT_STEPS})
  --warmup <n>      untimed steps first (default ${DEFAULT_WARMUP})
  --seed <n>        initial-noise seed (default ${DEFAULT_SEED})
  --batch <n>       steps per submit for the throughput number (default 16)
  --digest          after timing, re-run exactly --steps steps from the seed and
                      print a digest of the final state
  --dump-state <f>  like --digest, and write the state to <f> as JSON, for
                      scripts/compare-env.mjs to compare against a browser run
  --<param> <v>     any parameter of the preset's model, e.g. --dt 0.05
  --g<param> <v>    any parameter of the geometry, e.g. --gwaist 0.6
  --json            machine-readable output
  --help

The browser app shows the command for whatever it is currently simulating;
copy it from under the stats line to compare the same run here.`;

function fail(msg: string, code = 1): never {
  console.error(`bench: ${msg}`);
  process.exit(code);
}

// ---------------------------------------------------------------- arguments
const argv = process.argv.slice(2);
if (argv.includes('--help') || argv.includes('-h')) {
  console.log(USAGE);
  process.exit(0);
}
const wantJson = argv.includes('--json');
let batch = 16;
let dumpState: string | null = null;
let wantDigest = false;
const rest: string[] = [];
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--json') continue;
  if (a === '--digest') {
    wantDigest = true;
    continue;
  }
  const valued = (name: string): string | null => {
    if (a === `--${name}`) return argv[++i];
    if (a.startsWith(`--${name}=`)) return a.slice(name.length + 3);
    return null;
  };
  const b = valued('batch');
  if (b !== null) {
    batch = Number(b);
    continue;
  }
  const d = valued('dump-state');
  if (d !== null) {
    dumpState = d;
    wantDigest = true;
    continue;
  }
  rest.push(a);
}
if (!Number.isInteger(batch) || batch < 1) fail(`--batch must be an integer >= 1`, 2);

let spec: RunSpec;
try {
  spec = parseArgs(rest);
} catch (e) {
  fail(`${errMsg(e)}\n\n${USAGE}`, 2);
}

// ---------------------------------------------------------------- statistics
interface Timing {
  meanMs: number;
  medianMs: number;
  p05Ms: number;
  p95Ms: number;
  minMs: number;
  totalMs: number;
  stepsPerSec: number;
}

function timing(samples: Float64Array): Timing {
  const sorted = Float64Array.from(samples).sort();
  const q = (p: number): number =>
    sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
  let total = 0;
  for (const v of samples) total += v;
  const mean = total / samples.length;
  return {
    meanMs: mean,
    medianMs: q(0.5),
    p05Ms: q(0.05),
    p95Ms: q(0.95),
    minMs: sorted[0],
    totalMs: total,
    stepsPerSec: 1000 / mean,
  };
}

function fieldRange(v: ArrayLike<number>): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < v.length; i++) {
    if (v[i] < min) min = v[i];
    if (v[i] > max) max = v[i];
  }
  return { min, max };
}

// ---------------------------------------------------------------- run
const model = modelForSpec(spec);
const { preset } = resolvePreset(spec.preset);
const geometry = geometryForSpec(spec);

let device: GPUDevice | null = null;
let session: ModelSession | null = null;

try {
  const runtime = await installWebGpu();
  device = await requestShtDevice().catch((e: unknown) => {
    throw new Error(`${errMsg(e)}\n${NO_ADAPTER_HINT}`);
  });
  const adapter = await describeAdapter(device);

  session = await ModelSession.create({
    device,
    model,
    params: spec.params,
    lmax: spec.lmax,
    geometry,
    geometryParams: spec.geometryParams,
    niter: spec.niter,
    solver: spec.solver,
  });
  session.seed(spec.seed);

  const plan = session.describe();
  const kernels = plan.step.filter((l) => l.startsWith('kernel')).length;
  const cfg = session.cfg;

  if (!wantJson) {
    console.log(`turing-surface bench — solver only, no rendering\n`);
    console.log(`  preset    ${preset.label}  (models/${model.key}.m: ${model.species.join(', ')})`);
    console.log(
      `  params    ${model.params.map((p) => `${p.key}=${spec.params[p.key]}`).join('  ')}`,
    );
    const radius = session.geometry.radiusRange();
    console.log(
      `  geometry  ${geometry.label}  (geometries/${geometry.key}.m` +
        (geometry.params.length
          ? `: ${geometry.params.map((p) => `${p.key}=${spec.geometryParams[p.key]}`).join('  ')})`
          : ')') +
        `  radius ${radius.lo.toFixed(3)}–${radius.hi.toFixed(3)}`,
    );
    console.log(
      `  grid      lmax ${cfg.lmax} · ${cfg.nlat}×${cfg.nphi} · nlm ${session.sht.nlm.toLocaleString()} · ` +
        `${spec.niter} solve iteration${spec.niter === 1 ? '' : 's'} of ${spec.solver}`,
    );
    console.log(`  compiled  ${plan.step.length} GPU ops/step (${kernels} generated kernels)`);
    console.log(`  fourier   ${session.sht.fourierMode.toUpperCase()} stage`);
    console.log(`  backend   WebGPU fp32${adapter ? ` — ${adapter}` : ''}\n            ${runtime}`);
    console.log(`  run       ${spec.warmup} warmup + ${spec.steps} timed steps, seed ${spec.seed}\n`);
  }

  const done = (): Promise<undefined> => device!.queue.onSubmittedWorkDone();

  session.step(spec.warmup);
  await done();

  // --- throughput: batches submitted together, awaited once each ---
  const batches = Math.max(1, Math.ceil(spec.steps / batch));
  const progress = !wantJson && process.stderr.isTTY;
  let lastReport = performance.now();
  const tp0 = performance.now();
  let stepsRun = 0;
  let encodeMs = 0;
  for (let b = 0; b < batches; b++) {
    const n = Math.min(batch, spec.steps - stepsRun);
    const e0 = performance.now();
    session.step(n);
    encodeMs += performance.now() - e0;
    await done();
    stepsRun += n;
    if (progress && performance.now() - lastReport > 1000) {
      const so_far = (performance.now() - tp0) / stepsRun;
      process.stderr.write(
        `\r\x1b[K  ${stepsRun}/${spec.steps} steps · ${so_far.toFixed(2)} ms/step`,
      );
      lastReport = performance.now();
    }
  }
  const throughputMs = (performance.now() - tp0) / stepsRun;
  const encodePerStep = encodeMs / stepsRun;
  if (progress) process.stderr.write('\r\x1b[K');

  // --- latency: one step per submit, for the distribution ---
  const latencySteps = Math.min(spec.steps, 200);
  const samples = new Float64Array(latencySteps);
  for (let s = 0; s < latencySteps; s++) {
    const t0 = performance.now();
    session.step(1);
    await done();
    samples[s] = performance.now() - t0;
  }
  const t = timing(samples);

  const field = await session.read(model.species[0]);
  const range = fieldRange(field);
  let finite = true;
  for (const v of field) if (!Number.isFinite(v)) finite = false;

  // A reproducible state to compare across machines: exactly `--steps` steps
  // from the seed, separate from the timed runs above (which step a different
  // number of times to measure throughput and latency).
  let digest = null;
  let state: Float32Array | null = null;
  if (wantDigest) {
    session.seed(spec.seed);
    session.step(spec.steps);
    await done();
    state = await session.read(model.state[0]);
    digest = digestOf(state, session.sht.fourierMode, adapter);
  }

  if (wantJson) {
    console.log(
      JSON.stringify(
        {
          command: formatCommand(spec),
          spec,
          model: model.key,
          backend: { adapter, runtime, precision: 'fp32' },
          grid: { lmax: cfg.lmax, nlat: cfg.nlat, nphi: cfg.nphi, nlm: session.sht.nlm },
          compiled: { opsPerStep: plan.step.length, kernels },
          digest,
          throughput: {
            batch,
            msPerStep: throughputMs,
            stepsPerSec: 1000 / throughputMs,
            encodeMsPerStep: encodePerStep,
          },
          latency: t,
          state: {
            t: session.t,
            steps: session.steps,
            species: model.species[0],
            min: range.min,
            max: range.max,
            contrast: range.max - range.min,
            finite,
          },
        },
        null,
        2,
      ),
    );
  } else {
    console.log(
      `  ${throughputMs.toFixed(2)} ms/step   ${(1000 / throughputMs).toFixed(1)} steps/s   ` +
        `${(spec.params.dt * (1000 / throughputMs)).toFixed(2)} model time/s` +
        `   (batches of ${batch})`,
    );
    console.log(
      `  of which CPU command encoding: ${encodePerStep.toFixed(3)} ms/step ` +
        `(${((100 * encodePerStep) / throughputMs).toFixed(0)}% — the rest is the GPU)`,
    );
    console.log(
      `  one step per submit: ${t.meanMs.toFixed(2)} ms mean · median ${t.medianMs.toFixed(2)} · ` +
        `p05 ${t.p05Ms.toFixed(2)} · p95 ${t.p95Ms.toFixed(2)} · min ${t.minMs.toFixed(2)}`,
    );
    console.log(
      `  after ${session.steps} steps: t = ${session.t.toFixed(2)}, ` +
        `${model.species[0]} ∈ [${range.min.toFixed(4)}, ${range.max.toFixed(4)}] ` +
        `(contrast ${(range.max - range.min).toFixed(4)})${finite ? '' : '  — NOT FINITE'}`,
    );
    if (digest) {
      console.log(`\n  state after ${spec.steps} steps from seed ${spec.seed}:`);
      console.log(`    ${formatDigest(digest)}`);
    }
    console.log(
      `\n  The app's stats line reports the same solver number (batched steps,\n` +
        `  nothing read back) plus a separate ms/frame that carries the readback\n` +
        `  and the rendering. Compare solver with solver.`,
    );
  }

  if (dumpState && state && digest) {
    writeFileSync(
      dumpState,
      JSON.stringify({ command: formatCommand(spec), spec, digest, state: [...state] }),
    );
    if (!wantJson) console.log(`\n  wrote ${dumpState}`);
  }

  session.destroy();
  device.destroy();
  process.exit(finite ? 0 : 1);
} catch (e) {
  session?.destroy();
  device?.destroy();
  fail(errMsg(e));
}
