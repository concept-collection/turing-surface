/**
 * The MATLAB export (src/export/matlabScript.ts) is string assembly, so these
 * checks are cheap and need no GPU: every preset x geometry combination must
 * generate, the assembled file must keep its local-function namespace free of
 * collisions, and the driver's calls must match the signatures the .m files
 * declare. Whether the generated MATLAB actually reproduces a run is checked
 * against MATLAB itself, not here: a run exported at defaults and executed in
 * MATLAB R2026b lands within fp32 accumulation error of the app's own replay
 * (relL2 ~1e-7 over 60 steps via `npm run ref`), and the flux and Algorithm-4
 * exports track each other to ~3e-10 in f64.
 */
import { generateMatlabScript, MATLAB_SCRIPT_NAME } from '../src/export/matlabScript.ts';
import { presets, mModelByKey } from '../src/mgpu/registry.ts';
import { mGeometries } from '../src/geom/registry.ts';
import { formatCommand, resolvePreset, DEFAULT_WARMUP } from '../src/bench/runSpec.ts';

type Check = (name: string, ok: boolean, detail: string) => void;
type Log = (s: string) => void;

export function matlabExportChecks(check: Check, log: Log): void {
  log('--- MATLAB export ---');
  for (const preset of presets) {
    const { model, params } = resolvePreset(preset.key);
    for (const geometry of mGeometries) {
      const geometryParams = Object.fromEntries(
        geometry.params.map((p) => [p.key, p.value]),
      );
      const spec = {
        preset: preset.key,
        lmax: 63,
        seed: 1,
        steps: 2000,
        warmup: DEFAULT_WARMUP,
        params,
        geometry: geometry.key,
        geometryParams,
        niter: 8,
        solver: 'gmres' as const,
      };
      const name = `matlab-export ${preset.key} on ${geometry.key}`;
      let text: string;
      try {
        text = generateMatlabScript({
          model,
          modelSource: model.source,
          params,
          geometry,
          geometrySource: geometry.source,
          geometryParams,
          lmax: 63,
          niter: 8,
          solver: 'gmres',
          lam3: 0.5,
          seed: 1,
          preset: preset.key,
          command: formatCommand(spec),
        });
      } catch (e) {
        check(name, false, e instanceof Error ? e.message : String(e));
        continue;
      }

      // One file, one namespace: every local function name must be unique,
      // or MATLAB silently shadows one definition with another.
      const fnNames = [...text.matchAll(/^[ \t]*function\s+(?:\[[^\]]*\]|\w+)\s*=\s*(\w+)\s*\(/gm)]
        .map((m) => m[1]);
      const dupes = fnNames.filter((n, i) => fnNames.indexOf(n) !== i);

      // The driver must define what it calls: the state it steps, the model
      // call mapped through the mp struct, and the transform setup.
      // A model that hands its solve to solve(...) must carry the shim, the
      // selected solver and the operator; a self-contained model (alg4) must
      // not drag them in.
      const usesSolve = /\bsolve\s*\(/.test(model.source);
      const wants = [
        `function ${MATLAB_SCRIPT_NAME}()`,
        'sht_tables(sht_setup(lmax, mmax, nlat, nphi));',
        `= init(`,
        `= step(${model.state.join(', ')}, `,
        'surface_tables(gxr, gyr, gzr)',
        `'/final/${model.state[0]}'`,
        ...(usesSolve
          ? ['function X = solve(', 'function X = gmres(', 'function dL = dlap(']
          : []),
      ];
      const missing = wants.filter((w) => !text.includes(w));
      const dragged = usesSolve
        ? []
        : ['function X = solve(', 'function X = gmres('].filter((w) => text.includes(w));

      // randnfunsphere rides along exactly when the geometry draws on it.
      const wantsSphereTool = /\brandnfunsphere\b/.test(geometry.source);
      const carriesSphereTool = /function f = randnfunsphere\(/.test(text);

      const problems = [
        ...(dupes.length ? [`duplicate local functions: ${[...new Set(dupes)].join(', ')}`] : []),
        ...(missing.length ? [`missing: ${missing.join(' | ')}`] : []),
        ...(dragged.length ? [`included needlessly: ${dragged.join(' | ')}`] : []),
        ...(wantsSphereTool !== carriesSphereTool
          ? [`randnfunsphere ${wantsSphereTool ? 'missing' : 'included needlessly'}`]
          : []),
      ];
      check(name, problems.length === 0, problems.join('; ') || `${fnNames.length} local functions`);
    }
  }

  // The exact solver rides along like the others: its .m and the support
  // layer's lusolve, which assembles and factors the matrix in MATLAB.
  {
    const { model, params } = resolvePreset(presets[0].key);
    const geometry = mGeometries[0];
    const text = generateMatlabScript({
      model,
      modelSource: model.source,
      params,
      geometry,
      geometrySource: geometry.source,
      geometryParams: Object.fromEntries(geometry.params.map((p) => [p.key, p.value])),
      lmax: 15,
      niter: 0,
      solver: 'exact',
      lam3: 0.5,
      seed: 1,
      preset: presets[0].key,
      command: '',
    });
    const wants = ['function X = solve(', 'X = exact(B,', 'function X = exact(', 'function X = lusolve('];
    const missing = wants.filter((w) => !text.includes(w));
    check(
      'matlab-export carries the exact solver and its lusolve',
      missing.length === 0,
      missing.length ? `missing: ${missing.join(' | ')}` : wants.join(', '),
    );
  }

  // An edited working copy that dropped a required function is refused with a
  // message naming the file, not exported broken.
  const { model, params } = resolvePreset(presets[0].key);
  const geometry = mGeometries[0];
  try {
    generateMatlabScript({
      model,
      modelSource: '% nothing here',
      params,
      geometry,
      geometrySource: geometry.source,
      geometryParams: {},
      lmax: 63,
      niter: 8,
      solver: 'richardson',
      lam3: 0.5,
      seed: 1,
      preset: presets[0].key,
      command: '',
    });
    check('matlab-export refuses a source without init', false, 'no error thrown');
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    check(
      'matlab-export refuses a source without init',
      msg.includes("'init'") && msg.includes(model.key),
      msg,
    );
  }
  // Guard the assumption the model registry makes for stateFor: state names
  // are used as `<name>0` initial-capture variables, which must not collide
  // with the species names.
  for (const p of presets) {
    const m = mModelByKey(p.modelKey)!;
    const all = new Set([...m.state, ...m.species]);
    check(
      `matlab-export names disjoint for ${m.key}`,
      all.size === m.state.length + m.species.length,
      [...all].join(', '),
    );
  }
}
