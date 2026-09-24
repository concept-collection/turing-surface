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

/** Source for `dot`'s `.mtoc2.js`: two same-shape real arrays -> a 1x1
 *  scalar. The result is GPU-resident (a 1-element buffer written by a
 *  reduction dispatch, src/mgpu/reduce.ts), which is what lets a solver's
 *  alpha/omega recurrences run without the CPU in the loop. */
function dotSource(): string {
  return `
exports.name = "dot";

exports.transfer = function (argTypes, nargout) {
  if (argTypes.length !== 2) {
    throw new Error("dot takes exactly two arguments, got " + argTypes.length);
  }
  if (nargout > 1) {
    throw new Error("dot returns one value, but " + nargout + " were requested");
  }
  for (var i = 0; i < 2; i++) {
    var a = argTypes[i];
    if (!a || a.kind !== "Numeric" || a.isComplex) {
      throw new Error("dot requires real numeric arrays");
    }
  }
  var s0 = argTypes[0].shape;
  var s1 = argTypes[1].shape;
  if (!s0 || !s1 || s0.length !== s1.length ||
      !s0.every(function (d, i) { return d === s1[i]; })) {
    throw new Error(
      "dot requires two arrays of the same shape, got " +
        (s0 ? s0.join("x") : "unknown") + " and " + (s1 ? s1.join("x") : "unknown")
    );
  }
  return [${numericType(1, 1)}];
};

// Never called: this project executes the IR on WebGPU and emits no C.
exports.emit = function () {
  throw new Error("dot: no C backend (this reduction runs on WebGPU)");
};
exports.cBody = function () {
  return "";
};
`;
}

/**

 * Sources for the indexed-access ops a Krylov solver's bookkeeping needs:
 *
 *   Vk = getslab(VB, k)      k-th [2, nlm] field in a bank VB = [2, nlm*K]
 *   VB = setslab(VB, W, k)   the bank with field k replaced
 *   h  = getat(A, i)         element of a small matrix (1- or 2-index,
 *   h  = getat(A, i, j)        column-major)
 *   A  = setat(A, h, i)      the matrix with that element replaced
 *   A  = setat(A, h, i, j)
 *
 * All four are functional updates at the MATLAB level — `A(i,j) = h` cannot
 * lower, because numbl's JIT must prove an indexed write in bounds at
 * lowering time and a loop variable has no value there. Written as calls,
 * lowering just types them; the *planner* resolves each index when the
 * unrolled loop makes it a literal, and compiles every one of these to a
 * static-offset buffer copy (src/mgpu/plan.ts) — an in-place write when the
 * result is assigned back over the base, as a solver's loop does.
 */
function indexOpFiles(nlm: number): { name: string; source: string }[] {
  const shared = `
function isRealNumeric(t) {
  return t && t.kind === "Numeric" && !t.isComplex;
}
function isScalarIndex(t) {
  return isRealNumeric(t) && (!t.shape || t.shape.every(function (d) { return d === 1; }));
}
/** The base's type, with any exact (constant-folded) value stripped — the
 *  result differs from the base in one element, so it is not that constant. */
function baseType(t) {
  return {
    kind: "Numeric", elem: t.elem, isComplex: false,
    dims: t.dims, shape: t.shape, sign: "unknown",
  };
}
function checkBank(name, t) {
  if (!isRealNumeric(t) || !t.shape || t.shape.length !== 2 || t.shape[0] !== 2 ||
      t.shape[1] % ${nlm} !== 0) {
    throw new Error(
      name + " requires a 2 x (k*${nlm}) bank of spectral fields, got " +
        (t && t.shape ? t.shape.join("x") : "unknown shape")
    );
  }
}
`;
  const scalar11 = numericType(1, 1);
  const slab = numericType(2, nlm);
  return [
    {
      name: 'getslab.mtoc2.js',
      source: `${shared}
exports.name = "getslab";
exports.transfer = function (argTypes, nargout) {
  if (argTypes.length !== 2) throw new Error("getslab takes (bank, k), got " + argTypes.length + " arguments");
  if (nargout > 1) throw new Error("getslab returns one value");
  checkBank("getslab", argTypes[0]);
  if (!isScalarIndex(argTypes[1])) throw new Error("getslab's index must be a real scalar");
  return [${slab}];
};
exports.emit = function () { throw new Error("getslab: no C backend"); };
exports.cBody = function () { return ""; };
`,
    },
    {
      name: 'setslab.mtoc2.js',
      source: `${shared}
exports.name = "setslab";
exports.transfer = function (argTypes, nargout) {
  if (argTypes.length !== 3) throw new Error("setslab takes (bank, field, k), got " + argTypes.length + " arguments");
  if (nargout > 1) throw new Error("setslab returns one value");
  checkBank("setslab", argTypes[0]);
  var f = argTypes[1];
  if (!isRealNumeric(f) || !f.shape || f.shape.length !== 2 || f.shape[0] !== 2 || f.shape[1] !== ${nlm}) {
    throw new Error("setslab's field must be 2 x ${nlm}, got " + (f && f.shape ? f.shape.join("x") : "unknown shape"));
  }
  if (!isScalarIndex(argTypes[2])) throw new Error("setslab's index must be a real scalar");
  return [baseType(argTypes[0])];
};
exports.emit = function () { throw new Error("setslab: no C backend"); };
exports.cBody = function () { return ""; };
`,
    },
    {
      name: 'getat.mtoc2.js',
      source: `${shared}
exports.name = "getat";
exports.transfer = function (argTypes, nargout) {
  if (argTypes.length !== 2 && argTypes.length !== 3) {
    throw new Error("getat takes (A, i) or (A, i, j), got " + argTypes.length + " arguments");
  }
  if (nargout > 1) throw new Error("getat returns one value");
  if (!isRealNumeric(argTypes[0]) || !argTypes[0].shape) throw new Error("getat's base must be a real array of known shape");
  for (var k = 1; k < argTypes.length; k++) {
    if (!isScalarIndex(argTypes[k])) throw new Error("getat's indices must be real scalars");
  }
  return [${scalar11}];
};
exports.emit = function () { throw new Error("getat: no C backend"); };
exports.cBody = function () { return ""; };
`,
    },
    {
      name: 'setat.mtoc2.js',
      source: `${shared}
exports.name = "setat";
exports.transfer = function (argTypes, nargout) {
  if (argTypes.length !== 3 && argTypes.length !== 4) {
    throw new Error("setat takes (A, v, i) or (A, v, i, j), got " + argTypes.length + " arguments");
  }
  if (nargout > 1) throw new Error("setat returns one value");
  if (!isRealNumeric(argTypes[0]) || !argTypes[0].shape) throw new Error("setat's base must be a real array of known shape");
  if (!isScalarIndex(argTypes[1])) throw new Error("setat's value must be a real scalar");
  for (var k = 2; k < argTypes.length; k++) {
    if (!isScalarIndex(argTypes[k])) throw new Error("setat's indices must be real scalars");
  }
  return [baseType(argTypes[0])];
};
exports.emit = function () { throw new Error("setat: no C backend"); };
exports.cBody = function () { return ""; };
`,
    },
  ];
}

/**
 * Source for `lusolve`'s `.mtoc2.js`, the exact solver's call
 * (solvers/exact.m, src/mgpu/exact.ts):
 *
 *   X = lusolve(B, dtD, lam, filt, jhat, p2, r, dp1, dq2, jinv)
 *
 * The right-hand side and the result are spectral fields; dtD and jhat are
 * scalars the host reads to know which factorization the step needs; the
 * rest are the operator's grid arrays, from which the host assembles the
 * matrix. The planner checks that those really are the app's own buffers.
 */
function lusolveSource(g: GridSizes): string {
  return `
exports.name = "lusolve";

exports.transfer = function (argTypes, nargout) {
  var names = ["B", "dtD", "lam", "filt", "jhat", "p2", "r", "dp1", "dq2", "jinv"];
  if (argTypes.length !== names.length) {
    throw new Error(
      "lusolve takes (" + names.join(", ") + "), got " + argTypes.length + " argument(s)"
    );
  }
  if (nargout > 1) {
    throw new Error("lusolve returns one value, but " + nargout + " were requested");
  }
  function shapeOf(t) { return t && t.shape ? t.shape.join("x") : "unknown shape"; }
  function want(i, rows, cols) {
    var a = argTypes[i];
    if (!a || a.kind !== "Numeric" || a.isComplex) {
      throw new Error("lusolve requires real numeric arrays (" + names[i] + ")");
    }
    var s = a.shape;
    if (!s || s.length !== 2 || s[0] !== rows || s[1] !== cols) {
      throw new Error(
        "lusolve's " + names[i] + " must be " + rows + "x" + cols + ", not " + shapeOf(a)
      );
    }
  }
  want(0, 2, ${g.nlm});
  want(1, 1, 1);
  want(2, 2, ${g.nlm});
  want(3, 2, ${g.nlm});
  want(4, 1, 1);
  for (var i = 5; i < names.length; i++) want(i, ${g.npts}, 1);
  return [${numericType(2, g.nlm)}];
};

// Never called: this project executes the IR on WebGPU and emits no C.
exports.emit = function () {
  throw new Error("lusolve: no C backend (this solve runs on WebGPU)");
};
exports.cBody = function () {
  return "";
};
`;
}

/**
 * Workspace files that make `synth` / `analys` / `dtheta` / `dphi` /
 * `dthetac` / `dphic` / `dphig` / `randnfun3` / `dot` (and the
 * indexed-access ops above) resolvable during lowering. `dtheta` and `dphi`
 * (the surface's first partial derivatives, coefficients -> grid — see
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
    {
      // The seeded random field a model's `init` starts from
      // (src/mgpu/randnfun3.ts): a wavelength and the three surface
      // coordinates in, one value per grid point out.
      name: 'randnfun3.mtoc2.js',
      source: randnfun3Source(g),
    },
    {
      name: 'dot.mtoc2.js',
      source: dotSource(),
    },
    {
      // The exact solver's dense direct solve (src/mgpu/exact.ts).
      name: 'lusolve.mtoc2.js',
      source: lusolveSource(g),
    },
    ...indexOpFiles(g.nlm),
  ];
}

/** Source for `randnfun3`'s `.mtoc2.js`: `f = randnfun3(lambda, x, y, z)`. */
function randnfun3Source(g: GridSizes): string {
  return `
exports.name = "randnfun3";

exports.transfer = function (argTypes, nargout) {
  if (argTypes.length !== 4) {
    throw new Error(
      "randnfun3 takes a wavelength and the three surface coordinates -- " +
        "randnfun3(lambda, gx, gy, gz) -- got " + argTypes.length + " argument(s)"
    );
  }
  if (nargout > 1) {
    throw new Error("randnfun3 returns one value, but " + nargout + " were requested");
  }
  var lam = argTypes[0];
  if (!lam || lam.kind !== "Numeric" || lam.isComplex) {
    throw new Error("randnfun3's wavelength must be a real number");
  }
  var ls = lam.shape;
  if (!ls || ls.length !== 2 || ls[0] !== 1 || ls[1] !== 1) {
    throw new Error(
      "randnfun3's wavelength must be a single number, not a " +
        (ls ? ls.join("x") : "unknown shape") + " array"
    );
  }
  var names = ["gx", "gy", "gz"];
  for (var i = 1; i < 4; i++) {
    var a = argTypes[i];
    if (!a || a.kind !== "Numeric" || a.isComplex) {
      throw new Error("randnfun3 requires real numeric arrays (" + names[i - 1] + ")");
    }
    var s = a.shape;
    if (!s || s.length !== 2 || s[0] !== ${g.npts} || s[1] !== 1) {
      throw new Error(
        "randnfun3 evaluates on the grid, so " + names[i - 1] +
          " must be ${g.npts}x1, not " + (s ? s.join("x") : "unknown shape")
      );
    }
  }
  return [${numericType(g.npts, 1)}];
};

// Never called: this project executes the IR on WebGPU and emits no C.
exports.emit = function () {
  throw new Error("randnfun3: no C backend (this runs on WebGPU)");
};
exports.cBody = function () {
  return "";
};
`;
}

/** Names the WGSL backend must implement as GPU encodes rather than kernels,
 *  each with the argument counts it accepts. */
export const EXTERNAL_OPS = new Map<string, { minArgs: number; maxArgs: number }>([
  ['synth', { minArgs: 1, maxArgs: 1 }],
  ['analys', { minArgs: 1, maxArgs: 1 }],
  ['dtheta', { minArgs: 1, maxArgs: 1 }],
  ['dphi', { minArgs: 1, maxArgs: 1 }],
  ['dthetac', { minArgs: 1, maxArgs: 1 }],
  ['dphic', { minArgs: 1, maxArgs: 1 }],
  ['dphig', { minArgs: 1, maxArgs: 1 }],
  ['randnfun3', { minArgs: 4, maxArgs: 4 }],
  ['dot', { minArgs: 2, maxArgs: 2 }],
  ['getslab', { minArgs: 2, maxArgs: 2 }],
  ['setslab', { minArgs: 3, maxArgs: 3 }],
  ['getat', { minArgs: 2, maxArgs: 3 }],
  ['setat', { minArgs: 3, maxArgs: 4 }],
  ['lusolve', { minArgs: 10, maxArgs: 10 }],
]);
