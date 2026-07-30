/**
 * The two spherical-harmonic transforms, as external operations the .m can
 * call: `synth` (spectral -> grid) and `analys` (grid -> spectral).
 *
 * numbl needs only their *type rule* in order to lower a call site. It gets
 * that from a `.mtoc2.js` workspace file — numbl's sanctioned extension point
 * for a JS-defined builtin (see `mtoc2UserFunctionsByName` in numbl's
 * LoweringContext). The file is evaluated in a bare CommonJS sandbox with no
 * imports available, so `transfer` builds numbl `Type` objects as plain
 * literals, and the grid sizes are baked in by the generator below (a grid
 * change recompiles anyway).
 *
 * The `emit`/`cBody` exports exist only because the loader's contract requires
 * them; we never emit C. The actual implementation is supplied by the WGSL
 * backend, which turns each of these calls into an ShtPlan encode.
 *
 * Spectral fields are carried as REAL 2 x nlm arrays (row 0 real part, row 1
 * imaginary), matching the interleaved layout the GPU buffers already use.
 * The IMEX update is real-linear, so no complex arithmetic is needed.
 */

export interface GridSizes {
  /** Grid points, nlat*nphi. Grid fields are npts x 1 column vectors. */
  npts: number;
  /** Spectral coefficients. Spectral fields are 2 x nlm. */
  nlm: number;
}

const numericType = (rows: number, cols: number): string =>
  `{ kind: "Numeric", elem: "double", isComplex: false, ` +
  `dims: [${dim(rows)}, ${dim(cols)}], shape: [${rows}, ${cols}], sign: "unknown" }`;

// numbl's tensorDouble() canonicalizes an extent of 1 to its shared DIM_ONE
// singleton; mirror that so types compare equal to host-built ones.
const dim = (n: number): string =>
  n === 1 ? `{ kind: "exact", value: 1 }` : `{ kind: "exact", value: ${n} }`;

/** Source for one transform's `.mtoc2.js`. */
function transformSource(
  name: string,
  inRows: number,
  inCols: number,
  outRows: number,
  outCols: number,
): string {
  return `
exports.name = ${JSON.stringify(name)};

exports.transfer = function (argTypes, nargout) {
  if (argTypes.length !== 1) {
    throw new Error("${name} takes exactly one argument, got " + argTypes.length);
  }
  if (nargout > 1) {
    throw new Error("${name} returns one value, but " + nargout + " were requested");
  }
  var a = argTypes[0];
  if (!a || a.kind !== "Numeric" || a.isComplex) {
    throw new Error("${name} requires a real numeric array");
  }
  var s = a.shape;
  if (!s || s.length !== 2 || s[0] !== ${inRows} || s[1] !== ${inCols}) {
    throw new Error(
      "${name} requires a ${inRows}x${inCols} array, got " +
        (s ? s.join("x") : "unknown shape")
    );
  }
  return [${numericType(outRows, outCols)}];
};

// Never called: this project executes the IR on WebGPU and emits no C.
exports.emit = function () {
  throw new Error("${name}: no C backend (this transform runs on WebGPU)");
};
exports.cBody = function () {
  return "";
};
`;
}

/** Workspace files that make `synth` / `analys` resolvable during lowering. */
export function externalOpFiles(g: GridSizes): { name: string; source: string }[] {
  return [
    {
      name: 'synth.mtoc2.js',
      source: transformSource('synth', 2, g.nlm, g.npts, 1),
    },
    {
      name: 'analys.mtoc2.js',
      source: transformSource('analys', g.npts, 1, 2, g.nlm),
    },
  ];
}

/** Names the WGSL backend must implement as GPU encodes rather than kernels. */
export const EXTERNAL_OPS = new Set(['synth', 'analys']);
