/**
 * Shared .m files every model compiles against: the operator library and the
 * solvers. A model calls these by name (`richardson(...)`, `dlap(...)`) the
 * way it calls `synth` — except these are ordinary MATLAB, compiled through
 * the same pipeline and expanded into the caller at compile time
 * (src/mgpu/inlineCalls.ts), so a call costs exactly what writing the body
 * inline would.
 *
 * MATLAB file-visibility rules apply: only a file's namesake function is
 * callable from other files, and a function defined in the model shadows a
 * lib of the same name.
 */
import dlapSource from '../../lib/dlap.m?raw';
import richardsonSource from '../../solvers/richardson.m?raw';
import bicgstabSource from '../../solvers/bicgstab.m?raw';
import gmresSource from '../../solvers/gmres.m?raw';

export interface LibFile {
  name: string;
  source: string;
}

/** The operator: dlap = lap_g - lap_s applied to a spectral field. */
export const operatorLibs: LibFile[] = [{ name: 'dlap.m', source: dlapSource }];

/** The solvers for (I - dtD*lap_g) X = B. All share dlap's operator. */
export const solverLibs: LibFile[] = [
  { name: 'richardson.m', source: richardsonSource },
  { name: 'bicgstab.m', source: bicgstabSource },
  { name: 'gmres.m', source: gmresSource },
];

/** Everything a model may call. */
export const modelLibs: LibFile[] = [...operatorLibs, ...solverLibs];

/** Where each shared file lives on disk, for display. */
export const libPath = (name: string): string =>
  operatorLibs.some((f) => f.name === name) ? `lib/${name}` : `solvers/${name}`;

export type SolverKey = 'richardson' | 'bicgstab' | 'gmres';
export const solverKeys: SolverKey[] = ['richardson', 'bicgstab', 'gmres'];
export const DEFAULT_SOLVER: SolverKey = 'richardson';

/** What each solver actually takes: richardson needs no inner products, so
 *  no weights; only gmres sizes a basis bank, so only it takes nlm. */
const SOLVE_FORWARD: Record<SolverKey, string> = {
  richardson: 'richardson(B, dtD, lam, filt, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, niter)',
  bicgstab: 'bicgstab(B, dtD, lam, filt, wlm, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, niter)',
  gmres: 'gmres(B, dtD, lam, filt, wlm, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, nlm, niter)',
};

/**
 * The one-line shim behind the models' `solve(...)` call. The models pass
 * every argument any solver could want, and this host-generated file
 * forwards to the chosen one — so which solver runs is a compile-time choice
 * the app's solver control makes (swapping recompiles, like changing niter),
 * while the models stay identical across solvers. A model may also bypass
 * the shim and call a solver by name.
 */
export function solveShim(solver: SolverKey): LibFile {
  return {
    name: 'solve.m',
    source:
      `% Host-generated: forwards the models' solve(...) call to the solver\n` +
      `% selected in the app. See solvers/${solver}.m.\n` +
      `function X = solve(B, dtD, lam, filt, wlm, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, nlm, niter)\n` +
      `  X = ${SOLVE_FORWARD[solver]};\n` +
      `end\n`,
  };
}
