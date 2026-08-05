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

/** Source for one transform's `.mtoc2.js`. With `multi`, the op maps each of
 *  N inputs to its own output — `[a, b] = synth(x, y)` — so the backend can
 *  run the group as one batched Legendre dispatch (or split it to whatever
 *  lane width the device supports; the syntax promises grouping intent, not
 *  a width). */
function transformSource(
  name: string,
  inRows: number,
  inCols: number,
  outRows: number,
  outCols: number,
  multi = false,
): string {
  return `
exports.name = ${JSON.stringify(name)};

exports.transfer = function (argTypes, nargout) {
  ${
    multi
      ? `if (argTypes.length < 1) {
    throw new Error("${name} takes at least one argument");
  }
  if (nargout > 1 && nargout !== argTypes.length) {
    throw new Error(
      "${name}: each input produces one output, so " + argTypes.length +
        " input(s) return " + argTypes.length + " output(s), but " +
        nargout + " were requested -- write [a, b] = ${name}(x, y)"
    );
  }
  if (nargout <= 1 && argTypes.length !== 1) {
    throw new Error(
      "${name}: " + argTypes.length + " inputs produce " + argTypes.length +
        " outputs -- bind each one: [a, b] = ${name}(x, y)"
    );
  }`
      : `if (argTypes.length !== 1) {
    throw new Error("${name} takes exactly one argument, got " + argTypes.length);
  }
  if (nargout > 1) {
    throw new Error("${name} returns one value, but " + nargout + " were requested");
  }`
  }
  for (var i = 0; i < argTypes.length; i++) {
    var a = argTypes[i];
    if (!a || a.kind !== "Numeric" || a.isComplex) {
      throw new Error("${name} requires real numeric arrays (argument " + (i + 1) + ")");
    }
    var s = a.shape;
    if (!s || s.length !== 2 || s[0] !== ${inRows} || s[1] !== ${inCols}) {
      throw new Error(
        "${name} requires ${inRows}x${inCols} arrays, argument " + (i + 1) +
          " is " + (s ? s.join("x") : "unknown shape")
      );
    }
  }
  var out = [];
  for (var k = 0; k < Math.max(1, nargout); k++) {
    out.push(${numericType(outRows, outCols)});
  }
  return out;
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

/**
 * Workspace files that make `synth` / `analys` / `dtheta` / `dphi` /
 * `dthetac` / `dphic` resolvable during lowering. `dtheta` and `dphi` (the
 * surface's first partial derivatives, coefficients -> grid — see
 * src/sht/deriv.ts) have exactly `synth`'s shape rule: both take spectral
 * coefficients and produce a grid field. `dthetac` and `dphic` are their
 * coefficient-space halves alone — the alpha^+/alpha^- shift and the i*m
 * multiply, spectral -> spectral — which the six-transform Laplace-Beltrami
 * scheme (docs/reduced-transforms.md) applies twice per
 * matvec: to the field (gradient side) and to the analysed fluxes
 * (divergence side, the same shift, not its transpose).
 */
export function externalOpFiles(g: GridSizes): { name: string; source: string }[] {
  return [
    {
      name: 'synth.mtoc2.js',
      source: transformSource('synth', 2, g.nlm, g.npts, 1, true),
    },
    {
      name: 'analys.mtoc2.js',
      source: transformSource('analys', g.npts, 1, 2, g.nlm, true),
    },
    {
      name: 'dtheta.mtoc2.js',
      source: transformSource('dtheta', 2, g.nlm, g.npts, 1),
    },
    {
      name: 'dphi.mtoc2.js',
      source: transformSource('dphi', 2, g.nlm, g.npts, 1),
    },
    {
      name: 'dthetac.mtoc2.js',
      source: transformSource('dthetac', 2, g.nlm, 2, g.nlm),
    },
    {
      name: 'dphic.mtoc2.js',
      source: transformSource('dphic', 2, g.nlm, 2, g.nlm),
    },
    {
      // Grid-space phi-derivative: two Fourier stages and an i*m multiply,
      // no Legendre work (d/dphi is diagonal in the Fourier index). What
      // lets the flux-form divergence skip the Q-flux's spherical-harmonic
      // analysis.
      name: 'dphig.mtoc2.js',
      source: transformSource('dphig', g.npts, 1, g.npts, 1),
    },
  ];
}

/** Names the WGSL backend must implement as GPU encodes rather than kernels. */
export const EXTERNAL_OPS = new Set([
  'synth', 'analys', 'dtheta', 'dphi', 'dthetac', 'dphic', 'dphig',
]);
