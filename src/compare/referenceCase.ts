/**
 * Reading a reference HDF5 file into the pieces a replay needs.
 *
 * A reference file is a saved run from an independently-implemented solver —
 * geometry, initial and final spherical-harmonic coefficients, and the run's
 * parameters — in the layout documented in docs/ellipsoid-reference-spec.md.
 * Two things read it: the `npm run ref` CLI (through `h5wasm/node`) and the
 * browser's compare mode (through `h5wasm`, lazily loaded — see
 * referenceFile.ts). Both hand this module the same object shape, so the
 * format knowledge lives once.
 */
import { mModelByKey, defaultParams, type MModel, type Params } from '../mgpu/registry.ts';
import { mGeometryByKey, defaultGeometryParams, type MGeometry } from '../geom/registry.ts';
import { nlmCalc } from '../sht/layout.ts';

/** The slice of h5wasm's File/Group/Dataset API this reader touches — enough
 *  that the node and browser builds both satisfy it structurally. */
export interface H5Node {
  attrs: Record<string, { value: unknown }>;
  get(name: string): unknown;
}

export interface ReferenceCase {
  /** Where it came from — the file name, for labels and messages. */
  label: string;
  model: MModel;
  geometry: MGeometry;
  /** The model's defaults overlaid with the file's own — `dt` included, so
   *  `steps * params.dt` is the file's end time. */
  params: Params;
  geometryParams: Params;
  lmax: number;
  /** The solve-iteration count recorded in the file — the replay's default. */
  niter: number;
  /** Steps at `params.dt` from the initial state to the final one. */
  steps: number;
  /** The band-limited surface's own coefficients, [re, im] per (l, m). The
   *  reference solver ran on this exact surface, not the analytic shape. */
  geometryCoeffs: { X: Float32Array; Y: Float32Array; Z: Float32Array };
  /** Spectral state per species (keyed by `model.state` name) at t = 0. */
  initial: Record<string, Float32Array>;
  /** The same, at the end time. */
  final: Record<string, Float32Array>;
}

const attrsOf = (node: H5Node): Record<string, unknown> =>
  Object.fromEntries(Object.entries(node.attrs).map(([k, v]) => [k, v.value]));

/** Attributes as numbers — h5wasm hands back number or BigInt by dtype. */
const numberAttrs = (node: H5Node): Params =>
  Object.fromEntries(Object.entries(attrsOf(node)).map(([k, v]) => [k, Number(v)]));

function groupOf(node: H5Node, name: string): H5Node {
  const g = node.get(name) as H5Node | null;
  if (!g || typeof g.get !== 'function') {
    throw new Error(`no '${name}/' group — is this a reference file?`);
  }
  return g;
}

function coeffsOf(group: H5Node, groupName: string, name: string, nlm: number): Float32Array {
  const v = (group.get(name) as { value?: unknown } | null)?.value;
  if (!(v instanceof Float32Array)) {
    throw new Error(`'${groupName}/${name}' is not a float32 dataset`);
  }
  if (v.length !== 2 * nlm) {
    throw new Error(`'${groupName}/${name}' has ${v.length} values, expected 2*nlm = ${2 * nlm}`);
  }
  return v;
}

/** Read an open reference file. Throws with a plain message on anything the
 *  replay could not act on — unknown model or geometry, missing or misshapen
 *  coefficients — so both the CLI and the page can just show it. */
export function extractReferenceCase(file: H5Node, label: string): ReferenceCase {
  const modelKey = String(attrsOf(file).model);
  const model = mModelByKey(modelKey);
  if (!model) throw new Error(`unknown model '${modelKey}'`);

  const spec = groupOf(file, 'spec');
  const specAttrs = attrsOf(spec);
  const geometryKey = String(specAttrs.geometry);
  const geometry = mGeometryByKey(geometryKey);
  if (!geometry) throw new Error(`unknown geometry '${geometryKey}'`);

  const lmax = Number(specAttrs.lmax);
  const steps = Number(specAttrs.steps);
  const niter = Number(specAttrs.niter);
  if (!Number.isInteger(lmax) || lmax < 1) throw new Error(`bad lmax '${String(specAttrs.lmax)}'`);
  if (!Number.isInteger(steps) || steps < 1) throw new Error(`bad steps '${String(specAttrs.steps)}'`);
  if (!Number.isInteger(niter) || niter < 0) throw new Error(`bad niter '${String(specAttrs.niter)}'`);
  const nlm = nlmCalc(lmax, lmax);

  const params: Params = {
    ...defaultParams(model),
    ...numberAttrs(groupOf(spec, 'params')),
  };
  if (!(params.dt! > 0)) throw new Error(`bad dt '${params.dt}'`);
  const geometryParams: Params = {
    ...defaultGeometryParams(geometry),
    ...numberAttrs(groupOf(spec, 'geometry_params')),
  };

  const geom = groupOf(file, 'geometry');
  const geometryCoeffs = {
    X: coeffsOf(geom, 'geometry', 'Gx', nlm),
    Y: coeffsOf(geom, 'geometry', 'Gy', nlm),
    Z: coeffsOf(geom, 'geometry', 'Gz', nlm),
  };

  const initialGroup = groupOf(file, 'initial');
  const finalGroup = groupOf(file, 'final');
  const initial: Record<string, Float32Array> = {};
  const final: Record<string, Float32Array> = {};
  for (const name of model.state) {
    initial[name] = coeffsOf(initialGroup, 'initial', name, nlm);
    final[name] = coeffsOf(finalGroup, 'final', name, nlm);
  }

  return {
    label, model, geometry, params, geometryParams,
    lmax, niter, steps, geometryCoeffs, initial, final,
  };
}
