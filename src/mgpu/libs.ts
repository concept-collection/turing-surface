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
