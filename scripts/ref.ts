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
import { mModelByKey, defaultParams, type Params } from '../src/mgpu/registry.ts';
import { mGeometryByKey, defaultGeometryParams } from '../src/geom/registry.ts';
import { relL2 } from '../src/mgpu/digest.ts';
import { installWebGpu, errMsg, NO_ADAPTER_HINT } from './nodeWebGpu.ts';
import * as h5wasm from 'h5wasm/node';

const USAGE = `usage: npm run ref -- --in <file> [options]

  --in <file>       the reference HDF5 file to check against (required)
  --niter <n>       override the solve iteration count (default: the file's own)
  --tolerance <n>   if given, exit 1 when any reported relL2 meets or exceeds it
  --json            machine-readable output
  --help

Runs this repo's solver from the file's exact initial spectral state, to the
same physical end time, and reports the relative-L2 error of the resulting
state against the file's final state (and, as a sanity check, of the
regenerated geometry against the file's own geometry coefficients).`;

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
  const tolv = valued('tolerance');
  if (tolv !== null) {
    tolerance = Number(tolv);
    if (!Number.isFinite(tolerance)) fail(`--tolerance must be a number (got '${tolv}')`, 2);
    continue;
  }
  fail(`unrecognized argument '${a}'\n\n${USAGE}`, 2);
}
if (!inFile) fail(`--in <file> is required\n\n${USAGE}`, 2);

const attrsOf = (entity: { attrs: Record<string, { value: unknown }> }): Record<string, unknown> =>
  Object.fromEntries(Object.entries(entity.attrs).map(([k, v]) => [k, v.value]));

const numberAttrs = (entity: { attrs: Record<string, { value: unknown }> }): Params =>
  Object.fromEntries(
    Object.entries(attrsOf(entity)).map(([k, v]) => [k, Number(v)]),
  );

let device: GPUDevice | null = null;
let session: ModelSession | null = null;
let h5file: InstanceType<typeof h5wasm.File> | null = null;

try {
  await h5wasm.ready;
  h5file = new h5wasm.File(inFile, 'r');

  const rootAttrs = attrsOf(h5file);
  const modelKey = String(rootAttrs.model);
  const model = mModelByKey(modelKey);
  if (!model) fail(`unknown model '${modelKey}' in ${inFile}`);

  const specGroup = h5file.get('spec') as InstanceType<typeof h5wasm.Group>;
  const specAttrs = attrsOf(specGroup);
  const geometryKey = String(specAttrs.geometry);
  const geometryModel = mGeometryByKey(geometryKey);
  if (!geometryModel) fail(`unknown geometry '${geometryKey}' in ${inFile}`);

  const lmax = Number(specAttrs.lmax);
  const steps = Number(specAttrs.steps);
  const niter = niterOverride ?? Number(specAttrs.niter);

  const params: Params = {
    ...defaultParams(model),
    ...numberAttrs(specGroup.get('params') as InstanceType<typeof h5wasm.Group>),
  };
  const geometryParams: Params = {
    ...defaultGeometryParams(geometryModel),
    ...numberAttrs(specGroup.get('geometry_params') as InstanceType<typeof h5wasm.Group>),
  };

  const geomGroup = h5file.get('geometry') as InstanceType<typeof h5wasm.Group>;
  const fileGeom = {
    X: (geomGroup.get('Gx') as InstanceType<typeof h5wasm.Dataset>).value as Float32Array,
    Y: (geomGroup.get('Gy') as InstanceType<typeof h5wasm.Dataset>).value as Float32Array,
    Z: (geomGroup.get('Gz') as InstanceType<typeof h5wasm.Dataset>).value as Float32Array,
  };

  const initialGroup = h5file.get('initial') as InstanceType<typeof h5wasm.Group>;
  const finalGroup = h5file.get('final') as InstanceType<typeof h5wasm.Group>;
  const fileInitial: Record<string, Float32Array> = {};
  const fileFinal: Record<string, Float32Array> = {};
  for (const name of model.state) {
    fileInitial[name] = (initialGroup.get(name) as InstanceType<typeof h5wasm.Dataset>)
      .value as Float32Array;
    fileFinal[name] = (finalGroup.get(name) as InstanceType<typeof h5wasm.Dataset>)
      .value as Float32Array;
  }

  h5file.close();
  h5file = null;

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

  const geometryError = {
    Gx: relL2(session.geometry.X, fileGeom.X),
    Gy: relL2(session.geometry.Y, fileGeom.Y),
    Gz: relL2(session.geometry.Z, fileGeom.Z),
  };

  session.loadState(fileInitial);
  session.step(steps);

  const stateError: Record<string, number> = {};
  for (const name of model.state) {
    // Sequential: GpuModel.read() shares one staging buffer across calls.
    const ours = await session.read(name);
    stateError[name] = relL2(ours, fileFinal[name]);
  }

  const allErrors = [...Object.values(geometryError), ...Object.values(stateError)];
  const worst = Math.max(...allErrors);
  const pass = tolerance === null ? null : worst < tolerance;

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
          tolerance,
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
        `  niter      ${niter}${niterOverride !== null ? ` (file: ${specAttrs.niter})` : ''}\n` +
        `  run        ${steps} steps, dt=${params.dt}  (T=${(steps * (params.dt ?? 0)).toFixed(2)})\n`,
    );
    console.log(`  geometry check (regenerated vs file, relL2):`);
    for (const [k, v] of Object.entries(geometryError)) console.log(`    ${k}  ${v.toExponential(3)}`);
    console.log(`\n  final state (this run vs file, relL2):`);
    for (const [k, v] of Object.entries(stateError)) console.log(`    ${k}  ${v.toExponential(3)}`);
    if (tolerance !== null) {
      console.log(
        `\n  worst relL2 ${worst.toExponential(3)} vs tolerance ${tolerance.toExponential(3)}: ` +
          (pass ? 'PASS' : 'FAIL'),
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
