/**
 * A .m model, compiled and running on the GPU.
 *
 * A model file is ordinary MATLAB: it defines an `init` function that builds the
 * initial spectral state and a `step` function that advances it one timestep.
 * Each is specialized for the current grid and compiled into a ModelPlan, and
 * both operate on the same state buffers (see HostBuffers).
 *
 * Both functions return the new state followed by the grid fields the app
 * renders, so their signatures say exactly what they produce:
 *
 *   function [U, V, u, v] = init(noise, a, b)
 *   function [U, V, u, v] = step(U, V, lam, a, b, D1, D2, dt)
 *
 * The host supplies the things that are precomputation rather than algorithm:
 * the grid, the Laplace-Beltrami eigenvalues, the seeded initial noise, and the
 * parameter values. Each argument is matched to the .m's declared parameter
 * name, so the file documents its own interface.
 */
import { ShtPlan } from '../sht/sht.ts';
import type { DerivPlan } from '../sht/deriv.ts';
import { lmIndex, type ShtConfig } from '../sht/layout.ts';
import { HostBuffers, ModelPlan, type Randnfun3Lambda } from './plan.ts';
import { MODE_BUFFER } from './randnfun3.ts';
import { inFunction, inFunctionAsync, inModel } from './errors.ts';
import { CompiledModel, type Binding } from './compile.ts';

export interface ModelParams {
  [key: string]: number;
}

export interface GpuModelOptions {
  device: GPUDevice;
  sht: ShtPlan;
  cfg: ShtConfig;
  /** Model source (.m text). */
  source: string;
  /** Parameter names the .m may take as arguments. */
  paramNames: string[];
  /** Spectral state names, in order (e.g. ['U', 'V']). */
  state: string[];
  /** Grid fields to render, in order (e.g. ['u', 'v']). */
  view: string[];
  /**
   * The surface, as the .m may ask for it: `gx`, `gy`, `gz` are the embedding's
   * Cartesian coordinates on the grid, and `Gx`, `Gy`, `Gz` the spherical-
   * harmonic coefficients they were synthesized from. Omitted for a bare unit
   * sphere, where the .m has no geometry to take.
   */
  geometry?: GeometryBuffers;
  /** Computes `dtheta`/`dphi` for the .m's surface Laplace-Beltrami
   *  correction. Omitted for a bare unit sphere, same as `geometry`. */
  deriv?: DerivPlan;
  /**
   * Iterations of the implicit solve the .m's `for` loop runs. A fixed scalar
   * rather than a tunable one: the loop is unrolled into the op sequence, so
   * the count is part of what compiles and changing it recompiles.
   */
  niter?: number;
}

/** Host-supplied surface fields, in the layout the .m sees them. */
export interface GeometryBuffers {
  /** Grid coordinates, npts each. */
  x: Float32Array;
  y: Float32Array;
  z: Float32Array;
  /** Their spherical-harmonic coefficients, 2 x nlm each. */
  X: Float32Array;
  Y: Float32Array;
  Z: Float32Array;
  /** Inverse metric quantities (src/geom/metric.ts), grid space, npts each —
   *  the Algorithm-4 (12-transform) Laplace-Beltrami path. */
  Vtx: Float32Array;
  Vty: Float32Array;
  Vtz: Float32Array;
  Vpx: Float32Array;
  Vpy: Float32Array;
  Vpz: Float32Array;
  /** Flux-form metric weights (src/geom/metric.ts computeFluxMetric), grid
   *  space, npts each — the six-transform Laplace-Beltrami scheme of
   *  docs/reduced-transforms.md. */
  p1: Float32Array;
  p2: Float32Array;
  q2: Float32Array;
  r: Float32Array;
  /** The same weights with the round sphere subtracted (Geometry.dp1/dq2/jinv)
   *  — the sphere-split form of the flux divergence, which keeps r off the
   *  round-sphere part of the operator. */
  dp1: Float32Array;
  dq2: Float32Array;
  jinv: Float32Array;
  /** Mean-J preconditioner scale (Geometry.Jhat): folded into every
   *  setParams upload as the 'jhat' uniform, so a .m that takes jhat is
   *  never left with the zero a missing parameter would default to. An
   *  explicit jhat in the params wins (jhat: 1 pins the plain round-sphere
   *  preconditioner, for A/B). */
  Jhat: number;
}

/** Names the .m may take for the grid coordinates and for their coefficients. */
export const GEOMETRY_GRID_NAMES = ['gx', 'gy', 'gz'] as const;
export const GEOMETRY_SPECTRAL_NAMES = ['Gx', 'Gy', 'Gz'] as const;
/** Names the .m may take for the inverse metric quantities (Algorithm 4). */
export const METRIC_GRID_NAMES = ['Vtx', 'Vty', 'Vtz', 'Vpx', 'Vpy', 'Vpz'] as const;
/** Names the .m may take for the flux-form metric weights (six-transform
 *  scheme). A model asks for whichever set its loop uses; both are uploaded. */
export const FLUX_METRIC_GRID_NAMES = ['p1', 'p2', 'q2', 'r', 'dp1', 'dq2', 'jinv'] as const;

/** Laplace-Beltrami eigenvalues l(l+1), duplicated across re/im so the array
 *  matches the 2 x nlm spectral layout element for element. */
export function eigenvalues(cfg: ShtConfig, nlm: number): Float32Array {
  const lam = new Float32Array(2 * nlm);
  for (let m = 0; m <= cfg.mmax; m++) {
    for (let l = m; l <= cfg.lmax; l++) {
      const i = lmIndex(cfg.lmax, l, m);
      lam[2 * i] = l * (l + 1);
      lam[2 * i + 1] = l * (l + 1);
    }
  }
  return lam;
}

/**
 * 1 where l < lmax-2, else 0, duplicated across re/im like `lam`. The
 * theta/phi derivative recurrences (src/sht/derivCoeffs.ts) cannot exactly
 * represent a derivative at the top two degrees of the band limit, so the
 * surface Laplace-Beltrami correction filters them out wherever it
 * re-differentiates a field (evolving_surface/notes/algos.tex Sec 6,
 * "Miscellaneous implementation details").
 */
export function filterMask(cfg: ShtConfig, nlm: number): Float32Array {
  const filt = new Float32Array(2 * nlm);
  for (let m = 0; m <= cfg.mmax; m++) {
    for (let l = m; l <= cfg.lmax; l++) {
      const i = lmIndex(cfg.lmax, l, m);
      const keep = l < cfg.lmax - 2 ? 1 : 0;
      filt[2 * i] = keep;
      filt[2 * i + 1] = keep;
    }
  }
  return filt;
}

export class GpuModel {
  readonly paramNames: string[];
  readonly state: string[];
  readonly view: string[];
  readonly npts: number;
  readonly nlm: number;

  #device: GPUDevice;
  #host: HostBuffers;
  #initPlan: ModelPlan;
  #stepPlan: ModelPlan;
  /** Current geometry's mean-J scale; 1 with no geometry (the sphere). */
  #jhat = 1;
  #readback: GPUBuffer;
  /** Scratch holding a copy of the whole spectral state; see snapshotState. */
  #stash: GPUBuffer;
  /** Which function wrote the state most recently; see `read`. */
  #lastRan: 'init' | 'step' = 'init';
  #stashedRan: 'init' | 'step' = 'init';
  #destroyed = false;

  private constructor(init: {
    device: GPUDevice;
    host: HostBuffers;
    initPlan: ModelPlan;
    stepPlan: ModelPlan;
    readback: GPUBuffer;
    stash: GPUBuffer;
    paramNames: string[];
    state: string[];
    view: string[];
    npts: number;
    nlm: number;
  }) {
    this.#device = init.device;
    this.#host = init.host;
    this.#initPlan = init.initPlan;
    this.#stepPlan = init.stepPlan;
    this.#readback = init.readback;
    this.#stash = init.stash;
    this.paramNames = init.paramNames;
    this.state = init.state;
    this.view = init.view;
    this.npts = init.npts;
    this.nlm = init.nlm;
  }

  static async create(opts: GpuModelOptions): Promise<GpuModel> {
    const { device, sht, cfg, source, paramNames, state, view, geometry, deriv } = opts;
    const npts = cfg.nlat * cfg.nphi;
    const nlm = sht.nlm;
    const niter = opts.niter ?? 0;

    // What the .m may ask for by parameter name. Spectral state, the
    // eigenvalues and the top-mode filter are 2 x nlm; the seeded
    // perturbation is a grid field.
    const bindings: Record<string, Binding> = {
      lam: { kind: 'tensor', shape: [2, nlm] },
      filt: { kind: 'tensor', shape: [2, nlm] },
      noise: { kind: 'tensor', shape: [npts, 1] },
      npts: { kind: 'const', value: npts },
      nlm: { kind: 'const', value: nlm },
      niter: { kind: 'const', value: niter },
    };
    if (geometry) {
      for (const g of GEOMETRY_GRID_NAMES) bindings[g] = { kind: 'tensor', shape: [npts, 1] };
      for (const g of GEOMETRY_SPECTRAL_NAMES) bindings[g] = { kind: 'tensor', shape: [2, nlm] };
      for (const g of METRIC_GRID_NAMES) bindings[g] = { kind: 'tensor', shape: [npts, 1] };
      for (const g of FLUX_METRIC_GRID_NAMES) bindings[g] = { kind: 'tensor', shape: [npts, 1] };
      // Mean-J preconditioner scale (Geometry.Jhat): a uniform, not a const,
      // so swapping the surface updates it with no recompile. The session
      // folds the current geometry's value into every setParams call.
      bindings['jhat'] = { kind: 'param' };
      // The wavelength of the seeded random field (src/mgpu/randnfun3.ts).
      // A uniform like jhat, not a const: changing it redraws the field
      // without recompiling the step.
      bindings['lam3'] = { kind: 'param' };
    }
    for (const s of state) bindings[s] = { kind: 'tensor', shape: [2, nlm] };
    for (const p of paramNames) bindings[p] = { kind: 'param' };

    // Parsing belongs to the file, not to either function.
    const compiled = inModel(() => new CompiledModel(source, bindings, { npts, nlm }));
    // Both functions return the new state first, then the rendered grid fields.
    const nargout = state.length + view.length;
    const initFn = inFunction('init', () => compiled.specialize('init', nargout));
    const stepFn = inFunction('step', () => compiled.specialize('step', nargout));
    compiled.finish();

    // Only the state outputs feed back into the argument buffers; the grid
    // fields are read for display and then overwritten next call.
    const feedback = [...state, ...view.map(() => null)];

    const host = new HostBuffers(device);
    // The host owns the state and the inputs it uploads, whether or not a given
    // function happens to take them as arguments — `init` does not read `U`, but
    // it writes it, and `step` reads it back.
    for (const s of state) host.ensure(s, 2 * nlm);
    host.ensure('lam', 2 * nlm);
    host.ensure('filt', 2 * nlm);
    host.ensure('noise', npts);
    if (geometry) {
      for (const g of GEOMETRY_GRID_NAMES) host.ensure(g, npts);
      for (const g of GEOMETRY_SPECTRAL_NAMES) host.ensure(g, 2 * nlm);
      for (const g of METRIC_GRID_NAMES) host.ensure(g, npts);
      for (const g of FLUX_METRIC_GRID_NAMES) host.ensure(g, npts);
    }

    const initPlan = await inFunctionAsync('init', () =>
      ModelPlan.create(device, sht, { fn: initFn, feedback }, host, deriv),
    );
    const stepPlan = await inFunctionAsync('step', () =>
      ModelPlan.create(device, sht, { fn: stepFn, feedback }, host, deriv),
    );

    host.upload('lam', eigenvalues(cfg, nlm));
    host.upload('filt', filterMask(cfg, nlm));
    if (geometry) {
      host.upload('gx', geometry.x);
      host.upload('gy', geometry.y);
      host.upload('gz', geometry.z);
      host.upload('Gx', geometry.X);
      host.upload('Gy', geometry.Y);
      host.upload('Gz', geometry.Z);
      host.upload('Vtx', geometry.Vtx);
      host.upload('Vty', geometry.Vty);
      host.upload('Vtz', geometry.Vtz);
      host.upload('Vpx', geometry.Vpx);
      host.upload('Vpy', geometry.Vpy);
      host.upload('Vpz', geometry.Vpz);
      host.upload('p1', geometry.p1);
      host.upload('p2', geometry.p2);
      host.upload('q2', geometry.q2);
      host.upload('r', geometry.r);
      host.upload('dp1', geometry.dp1);
      host.upload('dq2', geometry.dq2);
      host.upload('jinv', geometry.jinv);
    }

    const readback = device.createBuffer({
      label: 'mgpu-readback',
      size: 4 * Math.max(npts, 2 * nlm),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const stash = device.createBuffer({
      label: 'mgpu-state-stash',
      size: 4 * state.length * 2 * nlm,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const gpu = new GpuModel({
      device, host, initPlan, stepPlan, readback, stash,
      paramNames, state, view, npts, nlm,
    });
    if (geometry) gpu.#jhat = geometry.Jhat;
    return gpu;
  }

  setParams(params: ModelParams): void {
    const merged = { jhat: this.#jhat, ...params };
    this.#initPlan.setParams(merged);
    this.#stepPlan.setParams(merged);
  }

  /**
   * Write a host-owned value directly — the spectral state, or one of the input
   * fields. Lets a test set up an exact initial condition (a single spherical-
   * harmonic mode, say) instead of going through `init`.
   */
  upload(name: string, data: Float32Array): void {
    this.#host.upload(name, data);
  }

  /**
   * Swap the surface under a running model. The geometry is data, not code —
   * its shape in the bindings depends only on the grid — so changing it is six
   * buffer writes and needs no recompile, and the simulation carries straight
   * on. Only meaningful if the .m took the geometry as an argument.
   */
  uploadGeometry(geometry: GeometryBuffers): void {
    const fields: [string, Float32Array][] = [
      ['gx', geometry.x], ['gy', geometry.y], ['gz', geometry.z],
      ['Gx', geometry.X], ['Gy', geometry.Y], ['Gz', geometry.Z],
      ['Vtx', geometry.Vtx], ['Vty', geometry.Vty], ['Vtz', geometry.Vtz],
      ['Vpx', geometry.Vpx], ['Vpy', geometry.Vpy], ['Vpz', geometry.Vpz],
      ['p1', geometry.p1], ['p2', geometry.p2],
      ['q2', geometry.q2], ['r', geometry.r],
      ['dp1', geometry.dp1], ['dq2', geometry.dq2], ['jinv', geometry.jinv],
    ];
    for (const [name, data] of fields) {
      if (this.#host.get(name)) this.#host.upload(name, data);
    }
    // The new surface's preconditioner scale takes effect on the next
    // setParams (the session re-applies its params after a swap).
    this.#jhat = geometry.Jhat;
  }

  /** The wavelength this model's `init` asked `randnfun3` for, or null if it
   *  seeds some other way. The session resolves it and draws the modes. */
  get randnfun3Lambda(): Randnfun3Lambda | null {
    return this.#initPlan.randnfun3Lambda;
  }

  /**
   * Upload the seeded initial data and run `init`.
   *
   * Both inputs are optional in the sense that a .m uses one or the other:
   * `modes` is the random field's coefficient table for a model that calls
   * `randnfun3`, `noise` the plain grid field for one that takes `noise`
   * directly (the analytic test models inject exact initial conditions that
   * way). Only what the plan actually bound is uploaded.
   */
  async init(noise: Float32Array, modes: Float32Array | null): Promise<void> {
    if (this.#host.get('noise')) this.#host.upload('noise', noise);
    // Sized to the wavelength, so this may reallocate and rebind.
    if (modes) this.#initPlan.uploadRandnfun3Table(this.#host, modes);
    // Submitted in pieces: a fine seed wavelength makes the mode sum long
    // enough that one submission would stall the browser's compositor.
    await this.#initPlan.submitYielding('mgpu-init');
    this.#lastRan = 'init';
  }

  /**
   * Copy the spectral state aside, so a batch of steps can run — to be timed —
   * and then be undone with restoreState, leaving the simulation exactly where
   * it was. Only the state is stashed: the grid view fields keep whatever the
   * batch last wrote until a subsequent step recomputes them, so step before
   * reading a view after a restore.
   */
  snapshotState(): void {
    this.#stashedRan = this.#lastRan;
    this.#copyState('save');
  }

  restoreState(): void {
    this.#copyState('restore');
    this.#lastRan = this.#stashedRan;
  }

  #copyState(dir: 'save' | 'restore'): void {
    // A restore can land after a rebuild destroyed the buffers mid-await;
    // there is nothing left to protect, so do not submit into destroyed state.
    if (this.#destroyed) return;
    const enc = this.#device.createCommandEncoder({ label: `mgpu-state-${dir}` });
    let offset = 0;
    for (const name of this.state) {
      const slot = this.#host.get(name);
      if (!slot) throw new Error(`state '${name}' has no host buffer`);
      const bytes = 4 * slot.count;
      if (dir === 'save') {
        enc.copyBufferToBuffer(slot.buffer, 0, this.#stash, offset, bytes);
      } else {
        enc.copyBufferToBuffer(this.#stash, offset, slot.buffer, 0, bytes);
      }
      offset += bytes;
    }
    this.#device.queue.submit([enc.finish()]);
  }

  /**
   * Advance `steps` timesteps. Synchronous — this only records commands and
   * submits them; nothing is read back and nothing is awaited.
   */
  step(steps = 1): void {
    const enc = this.#device.createCommandEncoder({ label: 'mgpu-step' });
    this.#stepPlan.encodeSteps(enc, steps);
    this.#device.queue.submit([enc.finish()]);
    this.#lastRan = 'step';
  }

  /**
   * The buffer currently holding a named value. Grid fields like `u` are
   * produced by both functions, into separate buffers (only the spectral state
   * is shared), so this resolves to whichever function ran most recently —
   * which is what makes the first frame show the initial state rather than an
   * unwritten buffer.
   */
  #locate(name: string): { buffer: GPUBuffer; count: number } | null {
    const [first, second] =
      this.#lastRan === 'init'
        ? [this.#initPlan, this.#stepPlan]
        : [this.#stepPlan, this.#initPlan];
    const buffer = first.buffer(name) ?? second.buffer(name);
    const count = first.elementCount(name) ?? second.elementCount(name);
    if (!buffer || count === undefined) return null;
    return { buffer, count };
  }

  /** The GPU buffer a named value would be read from right now — for encoding
   *  further GPU work against it (e.g. a display-grid synthesis of the state)
   *  without a CPU round trip. */
  valueBuffer(name: string): GPUBuffer | null {
    return this.#locate(name)?.buffer ?? null;
  }

  /** Read a named value back to the CPU. The only await in the whole loop. */
  async read(name: string): Promise<Float32Array> {
    const located = this.#locate(name);
    if (!located) {
      throw new Error(`read: the model has no value named '${name}'`);
    }
    const { buffer, count } = located;
    const enc = this.#device.createCommandEncoder({ label: `mgpu-read-${name}` });
    enc.copyBufferToBuffer(buffer, 0, this.#readback, 0, 4 * count);
    this.#device.queue.submit([enc.finish()]);
    await this.#readback.mapAsync(GPUMapMode.READ, 0, 4 * count);
    const out = new Float32Array(this.#readback.getMappedRange(0, 4 * count).slice(0));
    this.#readback.unmap();
    return out;
  }

  /** What the .m compiled to, for display. */
  describe(): { init: string[]; step: string[] } {
    return { init: this.#initPlan.describe(), step: this.#stepPlan.describe() };
  }

  destroy(): void {
    this.#destroyed = true;
    this.#initPlan.destroy();
    this.#stepPlan.destroy();
    this.#host.destroy();
    this.#readback.destroy();
    this.#stash.destroy();
  }
}
