/**
 * The surface: a .m shape file, compiled and evaluated into spherical-harmonic
 * coefficients.
 *
 * A geometry file is ordinary MATLAB defining one function,
 *
 *   function [gx, gy, gz] = shape(theta, phi, <parameters>)
 *
 * over the solver's (theta, phi) grid — the same element-wise MATLAB the models
 * are written in, compiled by the same backend into the same kind of WGSL
 * kernel. It is evaluated once, on the CPU's behalf, and then *analysed*: the
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
import { ShtPlan } from '../sht/sht.ts';
import type { ShtConfig } from '../sht/layout.ts';
import type { DerivPlan } from '../sht/deriv.ts';
import { computeMetric, computeFluxMetric } from './metric.ts';
import { HostBuffers, ModelPlan } from '../mgpu/plan.ts';
import { CompiledModel, type Binding } from '../mgpu/compile.ts';
import { inFunction, inFunctionAsync, inModel } from '../mgpu/errors.ts';
import type { ModelParams } from '../mgpu/model.ts';

/** The function a geometry file must define. */
export const SHAPE_FN = 'shape';

export interface GeometryOptions {
  device: GPUDevice;
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
    this.Jhat = init.Jhat;
    this.muMin = init.muMin;
    this.muMax = init.muMax;
    this.Jmin = init.Jmin;
    this.Jmax = init.Jmax;
  }

  /**
   * Compile the shape file, evaluate it once on the solver grid, and reduce it
   * to coefficients. Everything here happens at build time — a geometry never
   * takes part in the timestep — so it reads back through the CPU freely.
   */
  static async create(opts: GeometryOptions): Promise<Geometry> {
    const { device, sht, cfg, source, paramNames, params, deriv } = opts;
    const npts = cfg.nlat * cfg.nphi;
    const nlm = sht.nlm;

    const bindings: Record<string, Binding> = {
      theta: { kind: 'tensor', shape: [npts, 1] },
      phi: { kind: 'tensor', shape: [npts, 1] },
      npts: { kind: 'const', value: npts },
    };
    for (const p of paramNames) bindings[p] = { kind: 'param' };

    const compiled = inModel(() => new CompiledModel(source, bindings, { npts, nlm }));
    const fn = inFunction(SHAPE_FN, () => compiled.specialize(SHAPE_FN, 3));
    compiled.finish();

    const host = new HostBuffers(device);
    host.ensure('theta', npts);
    host.ensure('phi', npts);

    const plan = await inFunctionAsync(SHAPE_FN, () =>
      // Nothing feeds back: the three outputs are read once and the plan is
      // thrown away.
      ModelPlan.create(device, sht, { fn, feedback: [null, null, null] }, host),
    );

    try {
      const { theta, phi } = gridAngles(sht, cfg);
      host.upload('theta', theta);
      host.upload('phi', phi);
      plan.setParams(params);

      const enc = device.createCommandEncoder({ label: 'geometry-shape' });
      plan.encodeSteps(enc, 1);
      device.queue.submit([enc.finish()]);

      const raw = await Promise.all(
        fn.outputs.map((out) => readBuffer(device, plan, out.name, npts)),
      );
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
      for (let i = 0; i < cfg.nlat; i++) {
        const ct = sht.cosTheta[i];
        const st2 = Math.max(0, 1 - ct * ct);
        for (let j = 0; j < cfg.nphi; j++) {
          const k = i * cfg.nphi + j;
          const invJ = flux.r[k] * st2;
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
        Jhat, muMin, muMax, Jmin, Jmax,
      });
    } finally {
      plan.destroy();
      host.destroy();
    }
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

/** The (theta, phi) of every grid point, flattened phi-fastest as the fields are. */
function gridAngles(
  sht: ShtPlan,
  cfg: ShtConfig,
): { theta: Float32Array; phi: Float32Array } {
  const { nlat, nphi } = cfg;
  const theta = new Float32Array(nlat * nphi);
  const phi = new Float32Array(nlat * nphi);
  for (let i = 0; i < nlat; i++) {
    const th = Math.acos(Math.max(-1, Math.min(1, sht.cosTheta[i])));
    for (let j = 0; j < nphi; j++) {
      theta[i * nphi + j] = th;
      phi[i * nphi + j] = (2 * Math.PI * j) / nphi;
    }
  }
  return { theta, phi };
}

async function readBuffer(
  device: GPUDevice,
  plan: ModelPlan,
  name: string,
  count: number,
): Promise<Float32Array> {
  const buffer = plan.buffer(name);
  if (!buffer) {
    throw new Error(`the geometry never assigns '${name}'`);
  }
  const staging = device.createBuffer({
    label: `geometry-read-${name}`,
    size: 4 * count,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  try {
    const enc = device.createCommandEncoder({ label: `geometry-read-${name}` });
    enc.copyBufferToBuffer(buffer, 0, staging, 0, 4 * count);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    return out;
  } finally {
    staging.destroy();
  }
}
