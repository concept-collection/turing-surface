/**
 * One run, described in a single object shared by the browser app and the
 * command-line benchmark. The app formats the run it is currently showing into a
 * `npm run bench` command; the benchmark parses that command back into the same
 * object and compiles the same .m with it. Neither side keeps its own copy of
 * the defaults, so the two runs cannot drift apart — and both execute the same
 * MATLAB through the same pipeline, so the comparison is like for like.
 */
import {
  mModels,
  presets,
  defaultParams,
  type MModel,
  type Params,
  type Preset,
} from '../mgpu/registry.ts';
import {
  mGeometries,
  mGeometryByKey,
  defaultGeometryParams,
  DEFAULT_GEOMETRY_KEY,
  type MGeometry,
} from '../geom/registry.ts';
import { gridForLmax, type ShtConfig } from '../sht/layout.ts';

export interface RunSpec {
  /** Preset key from the registry; fixes the model, params may still be edited. */
  preset: string;
  lmax: number;
  /** Seed of the initial noise. */
  seed: number;
  /** Timed steps (the app runs forever; the benchmark stops here). */
  steps: number;
  /** Untimed steps run first, so shader/pipeline warm-up is not measured. */
  warmup: number;
  /** Full parameter set of the preset's model, as edited. */
  params: Params;
  /** Geometry key from the geometry registry. */
  geometry: string;
  /** Full parameter set of that geometry, as edited. Written on the command
   *  line with a `g` prefix (`--gwaist`) so a shape parameter can never
   *  collide with a model one. */
  geometryParams: Params;
  /** Iterations of the .m's implicit solve. Structural: it is unrolled into
   *  the compiled step, so it belongs to the spec rather than to the params. */
  niter: number;
}

/** Iterations of the implicit solve, everywhere that does not say otherwise:
 *  the app's `solve iters` control, `npm run bench`, and the soak. */
export const DEFAULT_NITER = 8;

/** Geometry + starting parameters of a geometry key. */
export function resolveGeometry(key: string): { geometry: MGeometry; params: Params } {
  const geometry = mGeometryByKey(key);
  if (!geometry) {
    throw new Error(
      `unknown geometry '${key}' (have: ${mGeometries.map((g) => g.key).join(', ')})`,
    );
  }
  return { geometry, params: defaultGeometryParams(geometry) };
}

export function geometryForSpec(spec: RunSpec): MGeometry {
  return resolveGeometry(spec.geometry).geometry;
}

/** The command the app displays and the benchmark answers to. Goes through npm
 *  because the benchmark runs under vite-node, which is what resolves numbl's
 *  compiler sources and the `?raw` model imports. */
export const BENCH_COMMAND = 'npm run bench --';
export const DEFAULT_LMAX = 63;
export const DEFAULT_SEED = 1;
/** Long enough that clock ramp-up and the occasional scheduling hiccup wash
 *  out: ~10 s of GPU stepping at lmax 63. */
export const DEFAULT_STEPS = 2000;
export const DEFAULT_WARMUP = 100;

/** Model + starting parameters of a preset, for the app's dropdown and the
 *  benchmark's --preset flag. */
export function resolvePreset(key: string): {
  preset: Preset;
  model: MModel;
  params: Params;
} {
  const preset = presets.find((p) => p.key === key);
  if (!preset) {
    throw new Error(
      `unknown preset '${key}' (have: ${presets.map((p) => p.key).join(', ')})`,
    );
  }
  const model = mModels.find((m) => m.key === preset.modelKey);
  if (!model) throw new Error(`preset '${key}' names unknown model '${preset.modelKey}'`);
  return { preset, model, params: { ...defaultParams(model), ...preset.params } };
}

export function modelForSpec(spec: RunSpec): MModel {
  return resolvePreset(spec.preset).model;
}

/** Transform configuration implied by the spec (same rule as the app). */
export function configForSpec(spec: RunSpec): ShtConfig {
  const { nlat, nphi } = gridForLmax(spec.lmax, modelForSpec(spec).pdeg);
  return { lmax: spec.lmax, mmax: spec.lmax, nlat, nphi };
}

/** The command line that reproduces this run. Every knob the app exposes is
 *  written out explicitly, so the command stays valid if a preset changes. */
export function formatCommand(spec: RunSpec): string {
  const model = modelForSpec(spec);
  const geometry = geometryForSpec(spec);
  const parts = [
    BENCH_COMMAND,
    `--preset ${spec.preset}`,
    `--geometry ${spec.geometry}`,
    `--lmax ${spec.lmax}`,
    `--niter ${spec.niter}`,
    `--steps ${spec.steps}`,
    `--seed ${spec.seed}`,
    ...model.params.map((p) => `--${p.key} ${String(spec.params[p.key])}`),
    ...geometry.params.map((p) => `--g${p.key} ${String(spec.geometryParams[p.key])}`),
  ];
  if (spec.warmup !== DEFAULT_WARMUP) parts.push(`--warmup ${spec.warmup}`);
  return parts.join(' ');
}

/** Inverse of formatCommand: `--key value` or `--key=value`, in any order.
 *  Throws with a usable message on anything it does not recognize. */
export function parseArgs(argv: string[]): RunSpec {
  const flags = new Map<string, string>();
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith('--')) throw new Error(`unexpected argument '${arg}'`);
    const eq = arg.indexOf('=');
    const key = eq >= 0 ? arg.slice(2, eq) : arg.slice(2);
    const value = eq >= 0 ? arg.slice(eq + 1) : argv[++i];
    if (value === undefined) throw new Error(`--${key} needs a value`);
    if (!key) throw new Error(`bad option '${arg}'`);
    flags.set(key, value);
  }
  const take = (key: string): string | undefined => {
    const v = flags.get(key);
    flags.delete(key);
    return v;
  };
  const number = (key: string, dflt: number): number => {
    const raw = take(key);
    if (raw === undefined) return dflt;
    const v = Number(raw);
    if (!Number.isFinite(v)) throw new Error(`--${key} must be a number (got '${raw}')`);
    return v;
  };
  const count = (key: string, dflt: number, min: number): number => {
    const v = number(key, dflt);
    if (!Number.isInteger(v) || v < min) {
      throw new Error(`--${key} must be an integer >= ${min} (got '${v}')`);
    }
    return v;
  };

  const presetKey = take('preset') ?? presets[0].key;
  const { model, params } = resolvePreset(presetKey);
  const geometryKey = take('geometry') ?? DEFAULT_GEOMETRY_KEY;
  const { geometry, params: geometryParams } = resolveGeometry(geometryKey);
  const spec: RunSpec = {
    preset: presetKey,
    lmax: count('lmax', DEFAULT_LMAX, 1),
    seed: number('seed', DEFAULT_SEED),
    steps: count('steps', DEFAULT_STEPS, 1),
    warmup: count('warmup', DEFAULT_WARMUP, 0),
    params,
    geometry: geometryKey,
    geometryParams,
    niter: count('niter', DEFAULT_NITER, 0),
  };
  const readInto = (into: Params, key: string, flag: string): void => {
    const raw = take(flag);
    if (raw === undefined) return;
    const v = Number(raw);
    if (!Number.isFinite(v)) throw new Error(`--${flag} must be a number (got '${raw}')`);
    into[key] = v;
  };
  for (const p of model.params) readInto(params, p.key, p.key);
  for (const p of geometry.params) readInto(geometryParams, p.key, `g${p.key}`);
  if (flags.size) {
    throw new Error(
      `unknown option(s): ${[...flags.keys()].map((k) => `--${k}`).join(', ')}\n` +
        `parameters of ${model.label}: ${model.params.map((p) => `--${p.key}`).join(' ')}\n` +
        `parameters of ${geometry.label}: ` +
        (geometry.params.length
          ? geometry.params.map((p) => `--g${p.key}`).join(' ')
          : '(none)'),
    );
  }
  return spec;
}
