/**
 * Import a reference HDF5 file — geometry, initial and final spherical-
 * harmonic coefficients for a run of this repo's solver, in the format
 * documented alongside the sibling test-data repo's case files (see
 * ../turing-surface-test-data/cases/) — run this repo's own solver from that
 * file's exact initial condition, and report the numerical error against
 * its final state.
 *
 * This is the regression check for the surface Laplace-Beltrami correction:
 * replay a saved-off run and see how far this repo's own output has drifted
 * (or use --niter to probe how much the correction term itself matters).
 *
 *   npm run ref -- --in data/schnak-spots.h5
 *   npm run ref -- --in data/schnak-spots.h5 --niter 0
 */
import { requestShtDevice, describeAdapter } from '../src/sht/sht.ts';
import { ModelSession } from '../src/mgpu/session.ts';
import { extractReferenceCase, type H5Node } from '../src/compare/referenceCase.ts';
import { relL2, relLinf } from '../src/mgpu/digest.ts';
import { installWebGpu, errMsg, NO_ADAPTER_HINT } from './nodeWebGpu.ts';
import * as h5wasm from 'h5wasm/node';

const USAGE = `usage: npm run ref -- --in <file> [options]

  --in <file>            the reference HDF5 file to check against (required)
  --niter <n>            override the solve iteration count (default: the file's own)
  --tolerance <n>        if given, exit 1 when any reported relL2 meets or exceeds it
  --tolerance-linf <n>   if given, exit 1 when any reported relLinf meets or exceeds it
  --json                 machine-readable output
  --help

Runs this repo's solver from the file's exact initial spectral state, to the
same physical end time, and reports the relative-L2 and relative-L-infinity
(max-norm) error of the resulting state against the file's final state (and,
as a sanity check, of the regenerated geometry against the file's own
geometry coefficients). --tolerance and --tolerance-linf gate independently:
either can fail the run on its own.`;

function fail(msg: string, code = 1): never {
  console.error(`ref: ${msg}`);
  process.exit(code);
}

const argv = process.argv.slice(2);
if (argv.includes('--help') || argv.includes('-h')) {
  console.log(USAGE);
  process.exit(0);
}
let inFile: string | null = null;
let niterOverride: number | null = null;
let tolerance: number | null = null;
let toleranceLinf: number | null = null;
const wantJson = argv.includes('--json');
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--json') continue;
  const valued = (name: string): string | null => {
    if (a === `--${name}`) return argv[++i];
    if (a.startsWith(`--${name}=`)) return a.slice(name.length + 3);
    return null;
  };
  const inv = valued('in');
  if (inv !== null) {
    inFile = inv;
    continue;
  }
  const niterv = valued('niter');
  if (niterv !== null) {
    niterOverride = Number(niterv);
    if (!Number.isInteger(niterOverride) || niterOverride < 0) {
      fail(`--niter must be an integer >= 0 (got '${niterv}')`, 2);
    }
    continue;
  }
  const tolLinfv = valued('tolerance-linf');
  if (tolLinfv !== null) {
    toleranceLinf = Number(tolLinfv);
    if (!Number.isFinite(toleranceLinf)) fail(`--tolerance-linf must be a number (got '${tolLinfv}')`, 2);
    continue;
  }
  const tolv = valued('tolerance');
  if (tolv !== null) {
    tolerance = Number(tolv);
    if (!Number.isFinite(tolerance)) fail(`--tolerance must be a number (got '${tolv}')`, 2);
    continue;
  }
  fail(`unrecognized argument '${a}'\n\n${USAGE}`, 2);
}
if (!inFile) fail(`--in <file> is required\n\n${USAGE}`, 2);

let device: GPUDevice | null = null;
let session: ModelSession | null = null;
let h5file: InstanceType<typeof h5wasm.File> | null = null;

try {
  await h5wasm.ready;
  h5file = new h5wasm.File(inFile, 'r');
  const rc = extractReferenceCase(h5file as H5Node, inFile);
  h5file.close();
  h5file = null;

  const { model, geometry: geometryModel, params, geometryParams, lmax, steps } = rc;
  const niter = niterOverride ?? rc.niter;
  const fileGeom = rc.geometryCoeffs;
  const fileInitial = rc.initial;
  const fileFinal = rc.final;

  const runtime = await installWebGpu();
  device = await requestShtDevice().catch((e: unknown) => {
    throw new Error(`${errMsg(e)}\n${NO_ADAPTER_HINT}`);
  });
  const adapter = await describeAdapter(device);

  session = await ModelSession.create({
    device,
    model,
    params,
    lmax,
    geometry: geometryModel,
    geometryParams,
    niter,
  });

  const errorOf = (a: Float32Array, b: Float32Array) => ({ relL2: relL2(a, b), relLinf: relLinf(a, b) });

  const geometryError = {
    Gx: errorOf(session.geometry.X, fileGeom.X),
    Gy: errorOf(session.geometry.Y, fileGeom.Y),
    Gz: errorOf(session.geometry.Z, fileGeom.Z),
  };

  session.loadState(fileInitial);
  session.step(steps);

  const stateError: Record<string, { relL2: number; relLinf: number }> = {};
  for (const name of model.state) {
    // Sequential: GpuModel.read() shares one staging buffer across calls.
    const ours = await session.read(name);
    stateError[name] = errorOf(ours, fileFinal[name]);
  }

  const allErrors = [...Object.values(geometryError), ...Object.values(stateError)];
  const worstL2 = Math.max(...allErrors.map((e) => e.relL2));
  const worstLinf = Math.max(...allErrors.map((e) => e.relLinf));
  const passL2 = tolerance === null ? null : worstL2 < tolerance;
  const passLinf = toleranceLinf === null ? null : worstLinf < toleranceLinf;
  const checks = [passL2, passLinf].filter((p): p is boolean => p !== null);
  const pass = checks.length === 0 ? null : checks.every(Boolean);

  if (wantJson) {
    console.log(
      JSON.stringify(
        {
          in: inFile,
          model: model.key,
          geometry: geometryModel.key,
          grid: { lmax, nlm: session.sht.nlm },
          niter,
          steps,
          dt: params.dt,
          T: steps * (params.dt ?? 0),
          backend: { adapter, runtime, precision: 'fp32' },
          geometryError,
          stateError,
          worstL2,
          worstLinf,
          tolerance,
          toleranceLinf,
          passL2,
          passLinf,
          pass,
        },
        null,
        2,
      ),
    );
  } else {
    console.log(`ref: ${inFile}`);
    console.log(
      `  model      ${model.label}  (${model.state.join(', ')})\n` +
        `  geometry   ${geometryModel.label}  ` +
        geometryModel.params.map((p) => `${p.key}=${geometryParams[p.key]}`).join(' ') +
        `\n  grid       lmax ${lmax} · nlm ${session.sht.nlm}\n` +
        `  niter      ${niter}${niterOverride !== null ? ` (file: ${rc.niter})` : ''}\n` +
        `  run        ${steps} steps, dt=${params.dt}  (T=${(steps * (params.dt ?? 0)).toFixed(2)})\n`,
    );
    const fmtErr = (v: { relL2: number; relLinf: number }) =>
      `relL2 ${v.relL2.toExponential(3)}  relLinf ${v.relLinf.toExponential(3)}`;
    console.log(`  geometry check (regenerated vs file):`);
    for (const [k, v] of Object.entries(geometryError)) console.log(`    ${k}  ${fmtErr(v)}`);
    console.log(`\n  final state (this run vs file):`);
    for (const [k, v] of Object.entries(stateError)) console.log(`    ${k}  ${fmtErr(v)}`);
    if (tolerance !== null) {
      console.log(
        `\n  worst relL2   ${worstL2.toExponential(3)} vs tolerance ${tolerance.toExponential(3)}: ` +
          (passL2 ? 'PASS' : 'FAIL'),
      );
    }
    if (toleranceLinf !== null) {
      console.log(
        `  worst relLinf ${worstLinf.toExponential(3)} vs tolerance-linf ${toleranceLinf.toExponential(3)}: ` +
          (passLinf ? 'PASS' : 'FAIL'),
      );
    }
  }

  session.destroy();
  device.destroy();
  process.exit(pass === false ? 1 : 0);
} catch (e) {
  h5file?.close();
  session?.destroy();
  device?.destroy();
  fail(errMsg(e));
}
