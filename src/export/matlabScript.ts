/**
 * The run on screen, as one standalone MATLAB script.
 *
 * The models and geometries are already MATLAB; what the app supplies around
 * them — the transforms, the geometry weights, the seeded field, the driver —
 * exists only as TypeScript and WGSL. This module assembles a single function
 * file carrying all of it: the current model and geometry sources verbatim as
 * local functions, double-precision MATLAB ports of the host-provided
 * operations (support.m), and a generated driver with the run's settings
 * baked in.
 *
 * Fidelity is method-for-method, not bit-for-bit: the ports run in f64 where
 * the GPU path is f32, and random draws use MATLAB's own rng, so a seed value
 * selects a different member of the same random ensemble than the same value
 * in the app. The script's results file uses the app's reference-run layout
 * (docs/ellipsoid-reference-spec.md), so a MATLAB run can be loaded back into
 * the page or checked with `npm run ref`.
 */
import supportSource from './support.m?raw';
import randnfun3Source from '../../tools/randnfun3.m?raw';
import randnfunsphereSource from '../../tools/randnfunsphere.m?raw';
import type { MModel, Params } from '../mgpu/registry.ts';
import type { MGeometry } from '../geom/registry.ts';

/** The generated function's name — and therefore the file name to save as. */
export const MATLAB_SCRIPT_NAME = 'turing_surface_run';

export interface MatlabExportSpec {
  model: MModel;
  /** Model source as running — the editor's working copy when edited. */
  modelSource: string;
  params: Params;
  geometry: MGeometry;
  geometrySource: string;
  geometryParams: Params;
  lmax: number;
  niter: number;
  /** Wavelength of the seeded random field. */
  lam3: number;
  seed: number;
  /** Preset key, recorded in the results file's /spec. */
  preset: string;
  /** The equivalent `npm run bench` command, recorded for provenance. */
  command: string;
  /** The generated script's own run controls; app defaults when omitted. */
  controls?: { nsteps?: number; plotEvery?: number; outFile?: string };
}

interface Signature {
  outputs: string[];
  params: string[];
}

/** First `function [outs] = name(args)` line in a .m — the same contract the
 *  compiler applies, minus everything it checks later. */
function parseSignature(source: string, name: string, file: string): Signature {
  const re = new RegExp(
    String.raw`^[ \t]*function\s+(?:\[([^\]]*)\]|([A-Za-z]\w*))\s*=\s*${name}\s*\(([^)]*)\)`,
    'm',
  );
  const m = re.exec(source);
  if (!m) {
    throw new Error(`cannot export: ${file} defines no function named '${name}'`);
  }
  const split = (s: string): string[] =>
    s.split(',').map((t) => t.trim()).filter((t) => t.length > 0);
  return {
    outputs: m[1] !== undefined ? split(m[1]) : [m[2]],
    params: split(m[3]),
  };
}

/** A number as MATLAB source. JS stringification round-trips doubles exactly
 *  and every form it produces (0.0004, 1e-21, -3) is a MATLAB literal. */
const num = (v: number): string => (Number.isFinite(v) ? String(v) : '0');

/** A string as a MATLAB char literal. */
const str = (s: string): string => `'${s.replace(/'/g, "''")}'`;

const banner = (title: string): string => {
  const line = `% ${'='.repeat(72)}`;
  return `${line}\n% ${title}\n${line}`;
};

export function generateMatlabScript(spec: MatlabExportSpec): string {
  const { model, geometry } = spec;
  const controls = {
    nsteps: spec.controls?.nsteps ?? 2000,
    plotEvery: spec.controls?.plotEvery ?? 10,
    outFile: spec.controls?.outFile ?? `${MATLAB_SCRIPT_NAME}.h5`,
  };

  const init = parseSignature(spec.modelSource, 'init', `models/${model.key}.m`);
  const step = parseSignature(spec.modelSource, 'step', `models/${model.key}.m`);
  const shape = parseSignature(spec.geometrySource, 'shape', `geometries/${geometry.key}.m`);

  // The driver defines every host-provided name the .m may ask for (lam,
  // filt, the geometry fields, jhat, niter, ...) under its canonical name,
  // so a model call is its own signature read back. Only the tunable
  // parameters live elsewhere — in the mp/gp structs, where the person
  // running the script edits them — so those names are mapped.
  const modelParamKeys = new Set(model.params.map((p) => p.key));
  const modelArg = (a: string): string => (modelParamKeys.has(a) ? `mp.${a}` : a);
  const shapeArg = (a: string): string =>
    a === 'theta' || a === 'phi' ? a : `gp.${a}`;

  const stateOuts = [...model.state, ...model.species];
  const outs = `[${stateOuts.join(', ')}]`;
  const initCall = `${outs} = init(${init.params.map(modelArg).join(', ')});`;
  const stepCall = `${outs} = step(${step.params.map(modelArg).join(', ')});`;
  const shapeCall = `[gxr, gyr, gzr] = shape(${shape.params.map(shapeArg).join(', ')});`;

  const speciesCell = `{${model.species.join(', ')}}`;
  const namesCell = `{${model.species.map((s) => str(s)).join(', ')}}`;

  // `noise` is the plain seeded grid perturbation, for a .m that takes it
  // instead of calling randnfun3 (none of the shipped models do).
  const takesNoise = init.params.includes('noise') || step.params.includes('noise');

  const mpBlock = model.params
    .map((p) => `mp.${p.key} = ${num(spec.params[p.key] ?? p.value)};`)
    .join('\n');
  const gpBlock = geometry.params
    .map((p) => `gp.${p.key} = ${num(spec.geometryParams[p.key] ?? p.value)};`)
    .join('\n');

  const driver = `function ${MATLAB_SCRIPT_NAME}()
% ${model.label} on ${geometry.label} -- a run captured from the
% turing-surface app as one standalone MATLAB script.
%
% The model and geometry .m below are the app's own, verbatim; around them
% this file carries double-precision MATLAB ports of everything the app
% provides from the host side: the spherical-harmonic transforms and their
% derivative shuffles, the metric weights of the surface Laplace-Beltrami
% operator, the seeded random field, and the run loop (src/sht and src/geom
% in the repository). The scheme is the app's: IMEX Euler, implicit
% diffusion preconditioned on the round sphere, the geometric correction
% iterated niter times per step.
%
% Two deliberate differences from the page. Everything here runs in double
% precision, where the app's GPU path is single. And random draws use
% MATLAB's own rng, so a seed value selects a different member of the same
% random ensemble than the same value in the app.
%
% Save as ${MATLAB_SCRIPT_NAME}.m and run it. The run plots live, and the
% final state is written to an HDF5 file in the app's reference-run layout
% (docs/ellipsoid-reference-spec.md in the repository), so it can be loaded
% back into the page ("Compare against uploaded data") or checked on the
% desktop with \`npm run ref -- --in ${controls.outFile}\`.
% Needs base MATLAB, R2020b or newer; no toolboxes.

% ---- run controls --------------------------------------------------------
nsteps = ${controls.nsteps};  % timesteps to run
plot_every = ${controls.plotEvery};  % live-plot interval, in steps; 0 disables plotting
out_file = ${str(controls.outFile)};  % results file; '' disables
seed = ${num(spec.seed)};  % rng seed for the initial condition

% ---- captured from the app -----------------------------------------------
lmax = ${spec.lmax};    % spherical-harmonic truncation degree
niter = ${spec.niter};  % iterations of the implicit solve's geometric correction
lam3 = ${num(spec.lam3)};  % wavelength of the seeded random field
${model.params.length ? `% ${model.label} parameters\n${mpBlock}` : `% ${model.label} has no parameters`}
${geometry.params.length ? `% ${geometry.label} parameters\n${gpBlock}` : `% ${geometry.label} has no parameters`}

% ---- grid and transforms -------------------------------------------------
% nlat/nphi follow lmax by the app's dealiasing rule (src/sht/layout.ts) for
% a reaction of polynomial degree pdeg.
pdeg = ${model.pdeg};
mmax = lmax;
nlat = 2 * ceil(max(lmax + 1, ((pdeg + 1) * lmax + 1) / 2) / 2);
nphi = 2 ^ nextpow2((pdeg + 1) * lmax + 1);
npts = nlat * nphi;
sht_tables(sht_setup(lmax, mmax, nlat, nphi));
S = sht_tables();
nlm = S.nlm;
lam = S.lam;
filt = S.filt;
theta = S.theta;
phi = S.phi;

% ---- the surface ---------------------------------------------------------
${shapeCall}
% A constant coordinate comes back scalar; spread it over the grid.
gxr = gxr + zeros(npts, 1);
gyr = gyr + zeros(npts, 1);
gzr = gzr + zeros(npts, 1);
G = surface_tables(gxr, gyr, gzr);
gx = G.gx;   gy = G.gy;   gz = G.gz;
Gx = G.Gx;   Gy = G.Gy;   Gz = G.Gz;
p1 = G.p1;   p2 = G.p2;   q2 = G.q2;   r = G.r;
dp1 = G.dp1; dq2 = G.dq2; jinv = G.jinv;
Vtx = G.Vtx; Vty = G.Vty; Vtz = G.Vtz;
Vpx = G.Vpx; Vpy = G.Vpy; Vpz = G.Vpz;
jhat = G.Jhat;
radius = sqrt(gx.^2 + gy.^2 + gz.^2);
fprintf('grid %d x %d, nlm %d, radius %.3f-%.3f, Jhat %.3f\\n', ...
  nlat, nphi, nlm, min(radius), max(radius), jhat);

% ---- initial condition ---------------------------------------------------
rng(seed);
${takesNoise ? `noise = ${num(model.seedAmp)} * randn(npts, 1);\n` : ''}${initCall}
${model.state.map((s) => `${s}0 = ${s};`).join('\n')}

% ---- time loop -----------------------------------------------------------
if plot_every > 0
  ph = plot_setup(gx, gy, gz, ${speciesCell}, ${namesCell});
  plot_update(ph, ${speciesCell}, 0, 0, nsteps);
end
report_every = max(1, round(nsteps / 10));
t = 0;
tstart = tic;
for k = 1:nsteps
  ${stepCall}
  t = t + mp.dt;
  if plot_every > 0 && (mod(k, plot_every) == 0 || k == nsteps)
    plot_update(ph, ${speciesCell}, t, k, nsteps);
  end
  if mod(k, report_every) == 0 || k == nsteps
    fprintf('step %d/%d   t = %.3f   (%.1f s)\\n', k, nsteps, t, toc(tstart));
  end
end

% ---- results file --------------------------------------------------------
% The app's reference-run layout, plus a /fields group with the final grid
% fields, the surface and the grid angles (each field stored nphi x nlat,
% ring by ring from the north pole).
if ~isempty(out_file)
  if exist(out_file, 'file') == 2
    delete(out_file);
  end
${['Gx', 'Gy', 'Gz']
  .map((c) => `  write_coeffs(out_file, '/geometry/${c}', ${c});`)
  .join('\n')}
${model.state
  .map((s) => `  write_coeffs(out_file, '/initial/${s}', ${s}0);`)
  .join('\n')}
${model.state
  .map((s) => `  write_coeffs(out_file, '/final/${s}', ${s});`)
  .join('\n')}
${[...model.species.map((s) => [s, s] as const), (['x', 'gx'] as const), (['y', 'gy'] as const), (['z', 'gz'] as const)]
  .map(
    ([name, v]) =>
      `  h5create(out_file, '/fields/${name}', [nphi nlat]);\n` +
      `  h5write(out_file, '/fields/${name}', reshape(${v}, nphi, nlat));`,
  )
  .join('\n')}
  h5create(out_file, '/fields/theta', nlat);
  h5write(out_file, '/fields/theta', acos(min(1, max(-1, S.ct))));
  h5create(out_file, '/fields/phi', nphi);
  h5write(out_file, '/fields/phi', 2*pi*(0:nphi-1)'/nphi);
  make_group(out_file, '/backend');
  make_group(out_file, '/spec');
  make_group(out_file, '/spec/params');
  make_group(out_file, '/spec/geometry_params');
  make_group(out_file, '/grid');
  h5writeatt(out_file, '/', 'model', ${str(model.key)});
  h5writeatt(out_file, '/', 'species', [${model.state.map((s) => `"${s}"`).join(' ')}]);
  h5writeatt(out_file, '/', 'command', ${str(spec.command)});
  h5writeatt(out_file, '/backend', 'runtime', 'matlab');
  h5writeatt(out_file, '/backend', 'adapter', ['MATLAB ' version]);
  h5writeatt(out_file, '/backend', 'precision', 'double');
  h5writeatt(out_file, '/spec', 'preset', ${str(spec.preset)});
  h5writeatt(out_file, '/spec', 'geometry', ${str(geometry.key)});
  h5writeatt(out_file, '/spec', 'lmax', lmax);
  h5writeatt(out_file, '/spec', 'seed', seed);
  h5writeatt(out_file, '/spec', 'steps', nsteps);
  h5writeatt(out_file, '/spec', 'warmup', 0);
  h5writeatt(out_file, '/spec', 'niter', niter);
  h5writeatt(out_file, '/spec', 'lam3', lam3);
${model.params
  .map((p) => `  h5writeatt(out_file, '/spec/params', ${str(p.key)}, mp.${p.key});`)
  .join('\n')}
${geometry.params
  .map((p) => `  h5writeatt(out_file, '/spec/geometry_params', ${str(p.key)}, gp.${p.key});`)
  .join('\n')}
  h5writeatt(out_file, '/grid', 'lmax', lmax);
  h5writeatt(out_file, '/grid', 'mmax', mmax);
  h5writeatt(out_file, '/grid', 'nlat', nlat);
  h5writeatt(out_file, '/grid', 'nphi', nphi);
  h5writeatt(out_file, '/grid', 'nlm', nlm);
  fprintf('wrote %s\\n', out_file);
end
end`;

  // tools/randnfun3.m verbatim, renamed: the models call the app's builtin
  // `randnfun3(lam3, gx, gy, gz)`, which support.m provides as a dispatcher
  // over this mode draw.
  const modesSource = randnfun3Source.replace(
    /function\s*\[\s*k\s*,\s*c\s*\]\s*=\s*randnfun3\s*\(/,
    'function [k, c] = randnfun3_modes(',
  );
  if (modesSource === randnfun3Source) {
    throw new Error('cannot export: tools/randnfun3.m no longer matches the expected signature');
  }

  const usesSphere = /\brandnfunsphere\b/.test(spec.geometrySource + spec.modelSource);

  const parts = [
    driver,
    banner(`models/${model.key}.m -- the model, verbatim`),
    spec.modelSource.trim(),
    banner(`geometries/${geometry.key}.m -- the surface, verbatim`),
    spec.geometrySource.trim(),
    banner('tools/randnfun3.m -- the random-field mode draw, verbatim'),
    modesSource.trim(),
    ...(usesSphere
      ? [banner('tools/randnfunsphere.m -- verbatim'), randnfunsphereSource.trim()]
      : []),
    banner('host-provided operations, ported from src/sht and src/geom'),
    supportSource.trim(),
  ];
  return parts.join('\n\n') + '\n';
}
