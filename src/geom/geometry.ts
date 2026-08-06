/**
 * The surface: a .m shape file, compiled and evaluated into spherical-harmonic
 * coefficients.
 *
 * A geometry file is ordinary MATLAB defining one function,
 *
 *   function [gx, gy, gz] = shape(theta, phi, <parameters>)
 *
 * over the solver's (theta, phi) grid. Unlike the models it is *not* compiled
 * to WGSL: a model's step runs every frame and must lower to a fixed sequence
 * of GPU dispatches, but a shape is evaluated exactly once at build time and
 * survives only as coefficients. So it runs through numbl's CPU interpreter
 * instead, which buys the full MATLAB subset — loops, arrays, reductions,
 * `legendre`, seeded randomness via `rng`/`randn` — and f64 evaluation, where
 * the step dialect is element-wise f32. The result is then *analysed*: the
 * canonical geometry this project carries is the three sets of coefficients
 * `X`, `Y`, `Z`, one per Cartesian component of the embedding.
 *
 * Going through the coefficients rather than keeping the pointwise values is
 * what makes the geometry usable by a spectral method, for two reasons:
 *
 *  - it is exactly band-limited at lmax afterwards, so the surface has as many
 *    derivatives as the scheme needs and no aliased content the solver cannot
 *    see. `x`, `y`, `z` below are the synthesis of the coefficients, not the
 *    raw output of the .m — the shape actually being solved on, which for a
 *    shape with sharp features is not quite the shape that was written down.
 *  - it can be evaluated on any grid. The renderer draws the surface on the
 *    (possibly finer) display grid by synthesizing the same coefficients
 *    there, which is exact interpolation rather than subdivision — the same
 *    argument that lets the species fields be oversampled.
 *
 * The unit sphere is the case where `x`, `y`, `z` are pure degree-1 harmonics
 * and everything downstream reduces to turing-sphere.
 */
import { parseMFile, type FunctionStmt } from 'numbl-src/numbl-core/parser/index.ts';
import { executeCode } from 'numbl-src/numbl-core/executeCode.ts';
import {
  RuntimeTensor,
  isRuntimeTensor,
  type RuntimeValue,
} from 'numbl-src/numbl-core/runtime/types.ts';
import { ShtPlan } from '../sht/sht.ts';
import type { ShtConfig } from '../sht/layout.ts';
import type { DerivPlan } from '../sht/deriv.ts';
import { computeMetric, computeFluxMetric } from './metric.ts';
import { toolFiles } from '../tools.ts';
import { inFunction, inModel, ModelCompileError } from '../mgpu/errors.ts';
import type { ModelParams } from '../mgpu/model.ts';

/** The function a geometry file must define. */
export const SHAPE_FN = 'shape';

export interface GeometryOptions {
  /** The solver's transform plan — the grid the shape is evaluated on. */
  sht: ShtPlan;
  cfg: ShtConfig;
  /** Geometry source (.m text). */
  source: string;
  /** Parameter names the .m may take beyond `theta` and `phi`. */
  paramNames: string[];
  params: ModelParams;
  /** Computes the theta/phi derivatives the inverse metric quantities need. */
  deriv: DerivPlan;
}

export class Geometry {
  /** Coordinates on the solver grid, npts each — synthesis of the coefficients. */
  readonly x: Float32Array;
  readonly y: Float32Array;
  readonly z: Float32Array;
  /** Their spherical-harmonic coefficients, 2 x nlm each. */
  readonly X: Float32Array;
  readonly Y: Float32Array;
  readonly Z: Float32Array;
  /**
   * Inverse metric quantities (src/geom/metric.ts), grid space, npts each.
   * Depend only on the geometry, so — like x,y,z,X,Y,Z above — these are a
   * one-off computed here, not per-solve-step work. Used by the Algorithm-4
   * (12-transform) Laplace-Beltrami path.
   */
  readonly Vtx: Float32Array;
  readonly Vty: Float32Array;
  readonly Vtz: Float32Array;
  readonly Vpx: Float32Array;
  readonly Vpy: Float32Array;
  readonly Vpz: Float32Array;
  /**
   * Flux-form metric weights (src/geom/metric.ts computeFluxMetric), grid
   * space, npts each — the six-transform Laplace-Beltrami scheme's
   * replacement for the six V arrays (docs/reduced-transforms.md
   * Sec 3). Both sets are carried so either operator formulation can run.
   */
  readonly p1: Float32Array;
  readonly p2: Float32Array;
  readonly q2: Float32Array;
  readonly r: Float32Array;
  /**
   * The same flux weights with the round sphere subtracted off, plus the
   * bounded 1/J — what lets a model evaluate lap_g without ever multiplying
   * the *whole* flux divergence by r ~ 1/sin^2(theta). Writing p1 = 1 + dp1,
   * q2 = 1 + dq2 (p2 is already a pure deviation, zero on the sphere) splits
   * the divergence into a round-sphere part, whose cancelling bracket
   * sin(theta) dtheta(A) + dphi(B) = -sin^2(theta) lap_s u is known exactly in
   * spectral space, and a remainder:
   *
   *   lap_g u = -jinv * lap_s u + r * (sin(theta) dtheta(P') + dphi(Q'))
   *
   * with P' = dp1*A + p2*B, Q' = p2*A + dq2*B. Only the remainder meets the
   * concentrated division, so the polar roundoff gain drops by |P'|/|P|
   * instead of applying to the full flux. Subtracting 1 in f64 here is the
   * point: on a near-sphere dp1 is the small quantity, and forming it as an
   * f32 difference in the .m would lose it. See docs/reduced-transforms.md
   * Sec 5 and models/schnakenberg.m.
   */
  readonly dp1: Float32Array;
  readonly dq2: Float32Array;
  readonly jinv: Float32Array;
  /**
   * Preconditioner scale for the implicit solve (docs/reduced-transforms.md
   * Sec 10). At high degree the Richardson iteration's per-mode factor is
   * governed by the operator's principal symbol: in the orthonormal frame
   * the surface symbol matrix is S = (1/J)[[p1, p2], [p2, q2]], whose
   * eigenvalues mu(x) are the inverse squared principal stretches of the
   * embedding — the round sphere has mu = 1. Preconditioning with lam/Jhat
   * contracts every mode and every direction iff Jhat*mu stays in (0, 2),
   * so the minimax constant is the harmonic mean of the symbol extremes,
   *
   *   Jhat = 2/(muMin + muMax),  rate = (muMax - muMin)/(muMax + muMin) < 1.
   *
   * The direction dependence is the point: a det-based mean of the area
   * factor J (mu's geometric mean, exact only for conformal surfaces)
   * under-corrects anisotropic stretching — on the shipped ellipsoid it
   * leaves a band of directional high-degree modes with amplification > 1,
   * which inflates the pattern's spectrum at moderate niter/lmax and
   * diverges at larger ones. The plain scheme (Jhat = 1) diverges wherever
   * muMax > 2. The solve's fixed point never depends on Jhat; only the
   * convergence rate does.
   */
  readonly Jhat: number;
  /** Symbol-eigenvalue range over the grid (see Jhat), for diagnostics. */
  readonly muMin: number;
  readonly muMax: number;
  /** Area-factor range over the grid, for diagnostics. */
  readonly Jmin: number;
  readonly Jmax: number;

  private constructor(init: {
    x: Float32Array; y: Float32Array; z: Float32Array;
    X: Float32Array; Y: Float32Array; Z: Float32Array;
    Vtx: Float32Array; Vty: Float32Array; Vtz: Float32Array;
    Vpx: Float32Array; Vpy: Float32Array; Vpz: Float32Array;
    p1: Float32Array; p2: Float32Array; q2: Float32Array; r: Float32Array;
    dp1: Float32Array; dq2: Float32Array; jinv: Float32Array;
    Jhat: number; muMin: number; muMax: number; Jmin: number; Jmax: number;
  }) {
    this.x = init.x;
    this.y = init.y;
    this.z = init.z;
    this.X = init.X;
    this.Y = init.Y;
    this.Z = init.Z;
    this.Vtx = init.Vtx;
    this.Vty = init.Vty;
    this.Vtz = init.Vtz;
    this.Vpx = init.Vpx;
    this.Vpy = init.Vpy;
    this.Vpz = init.Vpz;
    this.p1 = init.p1;
    this.p2 = init.p2;
    this.q2 = init.q2;
    this.r = init.r;
    this.dp1 = init.dp1;
    this.dq2 = init.dq2;
    this.jinv = init.jinv;
    this.Jhat = init.Jhat;
    this.muMin = init.muMin;
    this.muMax = init.muMax;
    this.Jmin = init.Jmin;
    this.Jmax = init.Jmax;
  }

  /**
   * Evaluate the shape file once on the solver grid and reduce it to
   * coefficients. Everything here happens at build time — a geometry never
   * takes part in the timestep — so the .m runs on the CPU (see
   * `evaluateShape`) and only the analysis onward touches the GPU.
   */
  static async create(opts: GeometryOptions): Promise<Geometry> {
    const { sht, cfg, source, paramNames, params, deriv } = opts;
    const npts = cfg.nlat * cfg.nphi;

    const { theta, phi } = gridAngles(sht, cfg);
    const raw = evaluateShape(source, paramNames, params, theta, phi, npts);

    // Coefficients first, then back to the grid: what the solver and the
    // renderer both see is the band-limited surface, not the raw .m output.
    const [X, Y, Z] = [
      await sht.analys(raw[0]),
      await sht.analys(raw[1]),
      await sht.analys(raw[2]),
    ];
    const [x, y, z] = [
      await sht.synth(X),
      await sht.synth(Y),
      await sht.synth(Z),
    ];

    // Inverse metric quantities (algos.tex Algorithm 2): theta/phi
    // derivatives of the embedding's coefficients, contracted through the
    // inverse first fundamental form. Depends only on the geometry, so
    // this is a one-off alongside x,y,z above, not per-step work.
    const Xt = await deriv.dtheta(X);
    const Xp = await deriv.dphi(X);
    const Yt = await deriv.dtheta(Y);
    const Yp = await deriv.dphi(Y);
    const Zt = await deriv.dtheta(Z);
    const Zp = await deriv.dphi(Z);
    const { Vtx, Vty, Vtz, Vpx, Vpy, Vpz } = computeMetric(npts, Xt, Xp, Yt, Yp, Zt, Zp);

    // Flux-form metric weights for the six-transform scheme, built from the
    // *undivided* theta tangents sin(theta)*X_theta (smooth on the sphere,
    // unlike X_theta itself) and the same X_phi as above. Also a one-off;
    // the f64 combination happens on the CPU, rounded to f32 for upload.
    const sXtx = await deriv.sinDtheta(X);
    const sXty = await deriv.sinDtheta(Y);
    const sXtz = await deriv.sinDtheta(Z);
    const flux = computeFluxMetric(npts, sXtx, sXty, sXtz, Xp, Yp, Zp);

    // The preconditioner scale — see the Jhat field comment. The symbol
    // matrix in the orthonormal frame is S = (1/J)[[p1,p2],[p2,q2]] with
    // 1/J = r sin^2(theta); its entries are the bounded quantities
    // g^tt, sin g^tp, sin^2 g^pp, so the eigenvalue extremes are clean to
    // take over the grid. det S = 1/J^2, so the area factor comes along
    // for free. f64 throughout.
    let muMin = Infinity;
    let muMax = 0;
    let Jmin = Infinity;
    let Jmax = 0;
    // The sphere-subtracted weights ride along on this loop: 1/J is already
    // being formed here, and dp1/dq2 want the same f64 arithmetic.
    const dp1 = new Float32Array(npts);
    const dq2 = new Float32Array(npts);
    const jinv = new Float32Array(npts);
    for (let i = 0; i < cfg.nlat; i++) {
      const ct = sht.cosTheta[i];
      const st2 = Math.max(0, 1 - ct * ct);
      for (let j = 0; j < cfg.nphi; j++) {
        const k = i * cfg.nphi + j;
        const invJ = flux.r[k] * st2;
        dp1[k] = flux.p1[k] - 1;
        dq2[k] = flux.q2[k] - 1;
        jinv[k] = invJ;
        const s11 = flux.p1[k] * invJ;
        const s12 = flux.p2[k] * invJ;
        const s22 = flux.q2[k] * invJ;
        const mean = (s11 + s22) / 2;
        const disc = Math.sqrt(((s11 - s22) / 2) ** 2 + s12 * s12);
        if (mean - disc < muMin) muMin = mean - disc;
        if (mean + disc > muMax) muMax = mean + disc;
        const J = 1 / invJ;
        if (J < Jmin) Jmin = J;
        if (J > Jmax) Jmax = J;
      }
    }
    const Jhat = 2 / (muMin + muMax);

    return new Geometry({
      x, y, z, X, Y, Z, Vtx, Vty, Vtz, Vpx, Vpy, Vpz,
      p1: new Float32Array(flux.p1),
      p2: new Float32Array(flux.p2),
      q2: new Float32Array(flux.q2),
      r: new Float32Array(flux.r),
      dp1, dq2, jinv,
      Jhat, muMin, muMax, Jmin, Jmax,
    });
  }

  /**
   * The surface evaluated on another plan's grid, as interleaved xyz vertex
   * positions (nlat * nphi * 3) — for rendering at display resolution. Exact
   * interpolation: the same coefficients, more evaluation points.
   */
  async positionsOn(view: ShtPlan): Promise<Float32Array> {
    const [x, y, z] = [
      await view.synth(this.X),
      await view.synth(this.Y),
      await view.synth(this.Z),
    ];
    const out = new Float32Array(x.length * 3);
    for (let i = 0; i < x.length; i++) {
      out[3 * i] = x[i];
      out[3 * i + 1] = y[i];
      out[3 * i + 2] = z[i];
    }
    return out;
  }

  /** How far the surface departs from the unit sphere, as min/max radius. */
  radiusRange(): { lo: number; hi: number } {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < this.x.length; i++) {
      const r = Math.hypot(this.x[i], this.y[i], this.z[i]);
      if (r < lo) lo = r;
      if (r > hi) hi = r;
    }
    return { lo, hi };
  }
}

/** The (theta, phi) of every grid point, flattened phi-fastest as the fields
 *  are — in f64, the precision the shape is evaluated at. */
function gridAngles(
  sht: ShtPlan,
  cfg: ShtConfig,
): { theta: Float64Array; phi: Float64Array } {
  const { nlat, nphi } = cfg;
  const theta = new Float64Array(nlat * nphi);
  const phi = new Float64Array(nlat * nphi);
  for (let i = 0; i < nlat; i++) {
    const th = Math.acos(Math.max(-1, Math.min(1, sht.cosTheta[i])));
    for (let j = 0; j < nphi; j++) {
      theta[i * nphi + j] = th;
      phi[i * nphi + j] = (2 * Math.PI * j) / nphi;
    }
  }
  return { theta, phi };
}

/**
 * Evaluate the shape file on the grid, through numbl's CPU interpreter.
 *
 * The .m keeps the same contract it had as a compiled model: it names the
 * arguments it wants — `theta`, `phi`, and any of the registry's parameters —
 * and the host supplies them by name, so their order in the signature is the
 * .m's own business. A one-line driver script calls `shape` with exactly the
 * arguments its signature declares, with those names pre-bound in the
 * driver's workspace.
 */
function evaluateShape(
  source: string,
  paramNames: string[],
  params: ModelParams,
  theta: Float64Array,
  phi: Float64Array,
  npts: number,
): [Float32Array, Float32Array, Float32Array] {
  const file = `${SHAPE_FN}.m`;
  const ast = inModel(() => parseMFile(source, file));
  const fn = ast.body.find(
    (s): s is FunctionStmt =>
      s.type === 'Function' && (s as FunctionStmt).name === SHAPE_FN,
  );
  if (!fn) {
    throw new ModelCompileError(
      `the geometry defines no function named '${SHAPE_FN}'`,
    );
  }
  if (fn.outputs.length !== 3) {
    throw new ModelCompileError(
      `'${SHAPE_FN}' must return three outputs [gx, gy, gz], not ${fn.outputs.length}`,
      { fn: SHAPE_FN, start: fn.span.start, end: fn.span.end },
    );
  }
  const known = new Set(['theta', 'phi', ...paramNames]);
  for (const p of fn.params) {
    if (!known.has(p)) {
      throw new ModelCompileError(
        `'${SHAPE_FN}' takes an argument '${p}' that is neither the grid ` +
          `(theta, phi) nor one of this geometry's parameters` +
          (paramNames.length ? ` (${paramNames.join(', ')})` : ''),
        { fn: SHAPE_FN, start: fn.span.start, end: fn.span.end },
      );
    }
  }

  const vars: Record<string, RuntimeValue> = {
    theta: new RuntimeTensor(theta, [npts, 1]),
    phi: new RuntimeTensor(phi, [npts, 1]),
  };
  for (const name of paramNames) {
    const v = params[name];
    // Missing parameters read as 0, as ModelPlan.setParams has it.
    vars[name] = Number.isFinite(v) ? v : 0;
  }

  const driver = `[gx__, gy__, gz__] = ${SHAPE_FN}(${fn.params.join(', ')});`;
  const result = inFunction(SHAPE_FN, () =>
    executeCode(
      driver,
      { initialVariableValues: vars, displayResults: false, implicitCwdPath: null },
      [...toolFiles, { name: file, source }],
      'geometry-driver.m',
    ),
  );

  return [
    toGridField(result.variableValues['gx__'], fn.outputs[0], npts),
    toGridField(result.variableValues['gy__'], fn.outputs[1], npts),
    toGridField(result.variableValues['gz__'], fn.outputs[2], npts),
  ];
}

/** One returned coordinate → npts values, rounded to the transforms' f32. */
function toGridField(
  value: RuntimeValue | undefined,
  name: string,
  npts: number,
): Float32Array {
  // A constant coordinate stays scalar in MATLAB; spread it over the grid.
  if (typeof value === 'number') return new Float32Array(npts).fill(value);
  if (value !== undefined && isRuntimeTensor(value)) {
    if (value.imag) {
      throw new ModelCompileError(
        `the geometry's '${name}' is complex; coordinates must be real`,
        { fn: SHAPE_FN },
      );
    }
    // A vector of npts values, either orientation. A 2-D reshape is refused
    // rather than reordered: the tensor's column-major layout would not match
    // the grid's phi-fastest rows.
    if (value.data.length === npts && value.shape.every((d) => d === 1 || d === npts)) {
      return new Float32Array(value.data);
    }
    throw new ModelCompileError(
      `the geometry's '${name}' is ${value.shape.join(' x ')}, but the grid ` +
        `wants one value per point (${npts} x 1)`,
      { fn: SHAPE_FN },
    );
  }
  throw new ModelCompileError(`the geometry's '${name}' is not numeric`, {
    fn: SHAPE_FN,
  });
}
