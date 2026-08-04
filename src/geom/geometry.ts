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
import { computeMetric } from './metric.ts';
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
   * one-off computed here, not per-solve-step work.
   */
  readonly Vtx: Float32Array;
  readonly Vty: Float32Array;
  readonly Vtz: Float32Array;
  readonly Vpx: Float32Array;
  readonly Vpy: Float32Array;
  readonly Vpz: Float32Array;

  private constructor(init: {
    x: Float32Array; y: Float32Array; z: Float32Array;
    X: Float32Array; Y: Float32Array; Z: Float32Array;
    Vtx: Float32Array; Vty: Float32Array; Vtz: Float32Array;
    Vpx: Float32Array; Vpy: Float32Array; Vpz: Float32Array;
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

      return new Geometry({ x, y, z, X, Y, Z, Vtx, Vty, Vtz, Vpx, Vpy, Vpz });
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
