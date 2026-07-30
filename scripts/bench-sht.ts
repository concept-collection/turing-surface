/**
 * The transforms alone, on desktop WebGPU — the number to put next to upstream
 * SHTNS.
 *
 *   npm run bench:sht -- --lmax 63 --steps 2000
 *
 * `npm run bench` measures a whole timestep of a .m model. This measures one
 * spectral -> grid -> spectral round trip and nothing else, which is what
 * bench/shtns/shtbench{,_gpu} --mode transform measures on the other side. The
 * solver does one of these per species per step, and profiling of the reference
 * implementation puts them at ~96% of its compute, so this is the comparison
 * that actually decides how fast the solver can be.
 *
 * The grid comes from the same rule the app uses, through the same
 * parseArgs/configForSpec as `npm run bench`, so --preset and --lmax mean here
 * exactly what they mean there. Nothing about the model is used beyond its
 * dealiasing degree.
 *
 * Like the solver benchmark it reports throughput (a batch of round trips
 * submitted together, awaited once) and latency (one per submit, for the
 * distribution).
 */
import { ShtPlan, requestShtDevice, describeAdapter, type ShtBinding } from '../src/sht/sht.ts';
import { lmIndex } from '../src/sht/layout.ts';
import { makeRand } from '../src/mgpu/noise.ts';
import { digestOf, formatDigest, relL2 } from '../src/mgpu/digest.ts';
import {
  parseArgs,
  configForSpec,
  modelForSpec,
  DEFAULT_LMAX,
  DEFAULT_SEED,
  DEFAULT_STEPS,
  DEFAULT_WARMUP,
  type RunSpec,
} from '../src/bench/runSpec.ts';
import { presets } from '../src/mgpu/registry.ts';
import { installWebGpu, errMsg, NO_ADAPTER_HINT } from './nodeWebGpu.ts';
import { writeFileSync } from 'node:fs';

const BENCH_SHT_COMMAND = 'npm run bench:sht --';

const USAGE = `usage: npm run bench:sht -- [options]

  --lmax <n>        spherical harmonic truncation (default ${DEFAULT_LMAX})
  --steps <n>       timed round trips (default ${DEFAULT_STEPS})
  --warmup <n>      untimed round trips first (default ${DEFAULT_WARMUP})
  --seed <n>        seed of the initial spectrum (default ${DEFAULT_SEED})
  --batch <n>       round trips per submit for the throughput number (default 16)
  --preset <key>    only for its dealiasing degree, so the grid matches the
                      solver benchmark's: ${presets.map((p) => p.key).join(' | ')}
                      (default ${presets[0].key})
  --fourier <mode>  auto | fft | dft (default auto)
  --digest          after timing, re-run exactly --steps round trips from the
                      seed and print a digest of the final spectrum
  --dump-state <f>  like --digest, and write the spectrum to <f> as JSON, for
                      scripts/compare-native.mjs to diff
  --json            machine-readable output
  --help

The native counterpart is
  bench/shtns/shtbench     --mode transform --lmax <n> --steps <n>   (CPU, fp64)
  bench/shtns/shtbench_gpu --mode transform --lmax <n> --steps <n>   (CUDA, fp32)`;

function fail(msg: string, code = 1): never {
  console.error(`bench:sht: ${msg}`);
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
let fourier: 'auto' | 'fft' | 'dft' = 'auto';
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
  const f = valued('fourier');
  if (f !== null) {
    if (f !== 'auto' && f !== 'fft' && f !== 'dft') fail(`--fourier must be auto|fft|dft`, 2);
    fourier = f;
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
if (!Number.isInteger(batch) || batch < 1) fail('--batch must be an integer >= 1', 2);

let spec: RunSpec;
try {
  spec = parseArgs(rest);
} catch (e) {
  fail(`${errMsg(e)}\n\n${USAGE}`, 2);
}
const cfg = configForSpec(spec);

// -------------------------------------------------------------- the spectrum
/**
 * A seeded starting spectrum, uniform in [-1, 1). Deliberately the plainest
 * thing both sides can agree on bit for bit: mulberry32 only, no transcendental
 * functions, so a difference in the result is a difference in the transforms and
 * not in the input. The m = 0 imaginary parts are zeroed, since a real field has
 * none and the two libraries need not treat a coefficient that cannot occur
 * alike. Mirrors shtb_seeded_spectrum() in bench/shtns/spec.h.
 */
function seededSpectrum(lmax: number, mmax: number, nlm: number, seed: number): Float32Array {
  const rand = makeRand(seed);
  const qlm = new Float32Array(2 * nlm);
  for (let m = 0; m <= mmax; m++) {
    for (let l = m; l <= lmax; l++) {
      const lm = lmIndex(lmax, l, m);
      qlm[2 * lm] = 2 * rand() - 1;
      const im = 2 * rand() - 1;
      qlm[2 * lm + 1] = m === 0 ? 0 : im;
    }
  }
  return qlm;
}

// ---------------------------------------------------------------------- run
let device: GPUDevice | null = null;
let plan: ShtPlan | null = null;

try {
  const runtime = await installWebGpu();
  device = await requestShtDevice().catch((e: unknown) => {
    throw new Error(`${errMsg(e)}\n${NO_ADAPTER_HINT}`);
  });
  const adapter = await describeAdapter(device);
  plan = await ShtPlan.create(device, cfg, { fourier });
  const nlm = plan.nlm;
  const npts = cfg.nlat * cfg.nphi;

  // Two spectral buffers and one spatial one, so a round trip needs no copy:
  // round trips alternate direction, A -> spat -> B then B -> spat -> A.
  const mk = (label: string, size: number) =>
    device!.createBuffer({
      label,
      size,
      usage:
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  const qlm: [GPUBuffer, GPUBuffer] = [mk('sht-bench-qa', 8 * nlm), mk('sht-bench-qb', 8 * nlm)];
  const spat = mk('sht-bench-spat', 4 * npts);
  const readback = device.createBuffer({
    label: 'sht-bench-readback',
    size: 8 * nlm,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  // Built once, at plan time — a bind group per round trip would be measuring
  // bind-group creation.
  const synth: [ShtBinding, ShtBinding] = [
    plan.createSynthBinding(qlm[0], spat),
    plan.createSynthBinding(qlm[1], spat),
  ];
  const analys: [ShtBinding, ShtBinding] = [
    plan.createAnalysBinding(spat, qlm[1]),
    plan.createAnalysBinding(spat, qlm[0]),
  ];

  let cur = 0;
  /** Record `n` round trips into one submission. Returns nothing; the result is
   *  in qlm[cur] once the queue has drained. */
  const submit = (n: number): void => {
    const enc = device!.createCommandEncoder({ label: 'sht-bench' });
    const pass = enc.beginComputePass({ label: 'sht-bench' });
    for (let i = 0; i < n; i++) {
      plan!.encodeSynthInto(pass, synth[cur]);
      plan!.encodeAnalysInto(pass, analys[cur]);
      cur ^= 1;
    }
    pass.end();
    device!.queue.submit([enc.finish()]);
  };
  /** `submit` in chunks, so an arbitrary round-trip count does not build one
   *  command buffer with tens of thousands of dispatches in it. */
  const submitAll = (n: number, chunk = batch): void => {
    for (let done = 0; done < n; done += chunk) submit(Math.min(chunk, n - done));
  };
  const seed = (): void => {
    cur = 0;
    const q0 = seededSpectrum(cfg.lmax, cfg.mmax, nlm, spec.seed);
    device!.queue.writeBuffer(qlm[0], 0, q0 as Float32Array<ArrayBuffer>);
  };
  const readSpectrum = async (): Promise<Float32Array> => {
    const enc = device!.createCommandEncoder({ label: 'sht-bench-read' });
    enc.copyBufferToBuffer(qlm[cur], 0, readback, 0, 8 * nlm);
    device!.queue.submit([enc.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return out;
  };
  const done = (): Promise<undefined> => device!.queue.onSubmittedWorkDone();

  // What every report of this run says about itself, whether it succeeded, failed
  // early, or is being written to a state file for compare-native.mjs to diff.
  const identity = {
    mode: 'transform',
    spec: {
      preset: spec.preset,
      lmax: spec.lmax,
      seed: spec.seed,
      steps: spec.steps,
      warmup: spec.warmup,
    },
    backend: { library: 'shtns-webgpu (src/sht)', runtime, adapter, precision: 'fp32' },
    grid: { lmax: cfg.lmax, nlat: cfg.nlat, nphi: cfg.nphi, nlm },
    fourier: plan.fourierMode,
  };

  if (!wantJson) {
    console.log('turing-surface bench:sht — transforms only, no solver, no rendering\n');
    console.log(
      `  grid      lmax ${cfg.lmax} · ${cfg.nlat}×${cfg.nphi} · nlm ${nlm.toLocaleString()}` +
        `  (dealiased for ${modelForSpec(spec).key}, pdeg ${modelForSpec(spec).pdeg})`,
    );
    console.log(`  step      1 synthesis + 1 analysis (one round trip)`);
    console.log(`  fourier   ${plan.fourierMode.toUpperCase()} stage`);
    console.log(`  backend   WebGPU fp32${adapter ? ` — ${adapter}` : ''}\n            ${runtime}`);
    console.log(
      `  run       ${spec.warmup} warmup + ${spec.steps} timed round trips, seed ${spec.seed}\n`,
    );
  }

  /*
   * Before timing anything: does one round trip work on this device?
   *
   * analys(synth(q)) is the identity for a band-limited q — exact Gauss
   * quadrature, nphi past the aliasing limit — so a single round trip should
   * return the input to fp32 round-off, and this is the sharpest check the
   * transforms are working at all here. Doing it separately means a bad result
   * says *which* it is: broken from the first transform, or drifted over the
   * thousands of iterations the timing run does. Otherwise that is a manual
   * bisection on --steps.
   */
  seed();
  const input = seededSpectrum(cfg.lmax, cfg.mmax, nlm, spec.seed);
  submit(1);
  await done();
  const afterOne = await readSpectrum();
  const firstFinite = afterOne.every((v) => Number.isFinite(v));
  const firstRelL2 = firstFinite ? relL2(afterOne, input) : NaN;
  if (!wantJson) {
    console.log(
      `  one round trip: ${
        firstFinite
          ? `back to the input to ${firstRelL2.toExponential(2)} relative L2`
          : 'NOT FINITE'
      }`,
    );
  }
  if (!firstFinite || !(firstRelL2 < 1e-3)) {
    // Report it the same way a good run reports itself, so a caller reading
    // --json learns what went wrong and on which device rather than having to
    // scrape stderr. Then say it in prose and stop: timing a transform that does
    // not transform is a waste of minutes.
    if (wantJson) {
      console.log(
        JSON.stringify(
          {
            ...identity,
            firstRoundTrip: { finite: firstFinite, relL2: firstRelL2 },
            throughput: null,
            latency: null,
            digest: null,
            input: null,
            state: { min: null, max: null, finite: firstFinite },
          },
          null,
          2,
        ),
      );
    }
    const detail = firstFinite
      ? `it came back ${firstRelL2.toExponential(3)} away from the input, which is far\n` +
        `  outside fp32 round-off (~1e-7)`
      : `it came back with no finite values at all`;
    fail(
      `a single spectral -> grid -> spectral round trip does not round-trip on this\n` +
        `  device: ${detail}.\n\n` +
        `  That is a correctness problem in the transforms here, not a benchmarking one,\n` +
        `  so there is nothing worth timing yet. Adapter: ${adapter || '(unknown)'};\n` +
        `  Fourier stage: ${plan.fourierMode.toUpperCase()}.\n\n` +
        `  Worth trying, in order:\n` +
        `    npm run test:node                   the repo's own transform check against\n` +
        `                                        its f64 CPU twin, on this device\n` +
        `    ${BENCH_SHT_COMMAND} --fourier dft   the other Fourier stage; if this works,\n` +
        `                                        the WGSL FFT is the problem\n` +
        `    ${BENCH_SHT_COMMAND} --lmax 15       does it depend on the grid size?`,
    );
  }

  seed();
  submitAll(spec.warmup);
  await done();

  // --- throughput: batches submitted together, awaited once each ---
  const batches = Math.max(1, Math.ceil(spec.steps / batch));
  const tp0 = performance.now();
  let stepsRun = 0;
  let encodeMs = 0;
  for (let b = 0; b < batches; b++) {
    const n = Math.min(batch, spec.steps - stepsRun);
    const e0 = performance.now();
    submit(n);
    encodeMs += performance.now() - e0;
    await done();
    stepsRun += n;
  }
  const throughputMs = (performance.now() - tp0) / stepsRun;
  const encodePerStep = encodeMs / stepsRun;

  // --- latency: one round trip per submit ---
  const latencySteps = Math.min(spec.steps, 200);
  const samples = new Float64Array(latencySteps);
  for (let s = 0; s < latencySteps; s++) {
    const t0 = performance.now();
    submit(1);
    await done();
    samples[s] = performance.now() - t0;
  }
  const sorted = Float64Array.from(samples).sort();
  const q = (p: number): number => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
  let latTotal = 0;
  for (const v of samples) latTotal += v;
  const latency = {
    meanMs: latTotal / samples.length,
    medianMs: q(0.5),
    p05Ms: q(0.05),
    p95Ms: q(0.95),
    minMs: sorted[0],
  };

  // --- a reproducible spectrum to compare across implementations ---
  let digest = null;
  let inputDigest = null;
  let state: Float32Array | null = null;
  if (wantDigest) {
    inputDigest = digestOf(input, plan.fourierMode, adapter);
    seed();
    submitAll(spec.steps);
    await done();
    state = await readSpectrum();
    digest = digestOf(state, plan.fourierMode, adapter);
  }

  const current = await readSpectrum();
  let finite = true;
  let min = Infinity;
  let max = -Infinity;
  for (const v of current) {
    if (!Number.isFinite(v)) finite = false;
    if (v < min) min = v;
    if (v > max) max = v;
  }

  if (wantJson) {
    console.log(
      JSON.stringify(
        {
          ...identity,
          firstRoundTrip: { finite: firstFinite, relL2: firstRelL2 },
          throughput: {
            batch,
            msPerStep: throughputMs,
            stepsPerSec: 1000 / throughputMs,
            encodeMsPerStep: encodePerStep,
          },
          latency,
          digest,
          input: inputDigest,
          state: { min, max, finite },
        },
        null,
        2,
      ),
    );
  } else {
    console.log(
      `  ${throughputMs.toFixed(3)} ms/round trip   ` +
        `${(1000 / throughputMs).toFixed(1)} round trips/s   (batches of ${batch})`,
    );
    console.log(`  i.e. ${(throughputMs / 2).toFixed(3)} ms per single transform`);
    console.log(
      `  of which CPU command encoding: ${encodePerStep.toFixed(3)} ms/round trip ` +
        `(${((100 * encodePerStep) / throughputMs).toFixed(0)}% — the rest is the GPU)`,
    );
    console.log(
      `  one round trip per submit: ${latency.meanMs.toFixed(3)} ms mean · ` +
        `median ${latency.medianMs.toFixed(3)} · p05 ${latency.p05Ms.toFixed(3)} · ` +
        `p95 ${latency.p95Ms.toFixed(3)} · min ${latency.minMs.toFixed(3)}`,
    );
    if (!finite) console.log('  — NOT FINITE');
    if (digest) {
      console.log(`\n  spectrum after ${spec.steps} round trips from seed ${spec.seed}:`);
      console.log(`    ${formatDigest(digest)}`);
    }
    console.log(
      `\n  The native counterpart is bench/shtns/shtbench{,_gpu} --mode transform;\n` +
        `  scripts/compare-native.mjs runs both and lines the numbers up.`,
    );
  }

  if (dumpState && state && digest) {
    writeFileSync(
      dumpState,
      JSON.stringify({
        ...identity,
        digest,
        input: inputDigest,
        state: [...state],
      }),
    );
    if (!wantJson) console.log(`\n  wrote ${dumpState}`);
  }

  for (const b of [qlm[0], qlm[1], spat, readback]) b.destroy();
  plan.destroy();
  device.destroy();
  process.exit(finite ? 0 : 1);
} catch (e) {
  plan?.destroy();
  device?.destroy();
  fail(errMsg(e));
}
