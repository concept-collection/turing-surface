/**
 * One running model: grid, transforms, compiled .m, seeded state.
 *
 * Everything that is not rendering. The app, the desktop benchmark and the
 * tests all go through this, so there is one place that decides how a model is
 * turned into something running on the GPU — and nothing about it is
 * browser-specific beyond needing a GPUDevice.
 */
import { ShtPlan } from '../sht/sht.ts';
import { DerivPlan } from '../sht/deriv.ts';
import { gridForLmax, type ShtConfig } from '../sht/layout.ts';
import { GpuModel, type ModelParams } from './model.ts';
import { seededNoise } from './noise.ts';
import type { MModel } from './registry.ts';
import { Geometry } from '../geom/geometry.ts';
import { mGeometryByKey, defaultGeometryParams, SPHERE_KEY, type MGeometry } from '../geom/registry.ts';

export interface ModelSessionOptions {
  device: GPUDevice;
  model: MModel;
  params: ModelParams;
  lmax: number;
  /** Override the model source — the editor's working copy. */
  source?: string;
  /** Linear render oversampling: read the species fields on a grid this many
   *  times finer than the solver's in each direction (default 1). The state is
   *  band-limited at lmax, so the finer evaluation is exact interpolation. */
  oversample?: number;
  /** The surface to solve on. Defaults to the unit sphere. */
  geometry?: MGeometry;
  geometryParams?: ModelParams;
  /** Override the geometry source — the editor's working copy. */
  geometrySource?: string;
  /**
   * Iterations of the .m's implicit solve. Structural, not tunable: the loop
   * is unrolled into the op sequence, so a change recompiles.
   */
  niter?: number;
}

export class ModelSession {
  readonly device: GPUDevice;
  readonly model: MModel;
  readonly cfg: ShtConfig;
  readonly sht: ShtPlan;
  readonly gpu: GpuModel;
  readonly npts: number;
  /** Iterations of the implicit solve compiled into the step. */
  readonly niter: number;

  /** The surface being solved on, as spherical-harmonic coefficients. */
  #geometry: Geometry;
  #geometryModel: MGeometry;
  /** Computes the theta/phi derivatives a geometry's metric quantities need. */
  #deriv: DerivPlan;

  /** Model time and step count since the last seeding. */
  t = 0;
  steps = 0;

  #params: ModelParams;
  /** Display-only transforms on the oversampled grid; null at 1x. */
  #displaySht: ShtPlan | null;
  #oversample: number;

  private constructor(init: {
    device: GPUDevice;
    model: MModel;
    cfg: ShtConfig;
    sht: ShtPlan;
    displaySht: ShtPlan | null;
    gpu: GpuModel;
    params: ModelParams;
    oversample: number;
    geometry: Geometry;
    geometryModel: MGeometry;
    deriv: DerivPlan;
    niter: number;
  }) {
    this.device = init.device;
    this.model = init.model;
    this.cfg = init.cfg;
    this.sht = init.sht;
    this.gpu = init.gpu;
    this.npts = init.cfg.nlat * init.cfg.nphi;
    this.#oversample = init.oversample;
    this.#params = init.params;
    this.#displaySht = init.displaySht;
    this.#geometry = init.geometry;
    this.#geometryModel = init.geometryModel;
    this.#deriv = init.deriv;
    this.niter = init.niter;
  }

  get geometry(): Geometry {
    return this.#geometry;
  }

  get geometryModel(): MGeometry {
    return this.#geometryModel;
  }

  /** Linear render oversampling factor (1 = read on the solver grid). */
  get oversample(): number {
    return this.#oversample;
  }

  static async create(opts: ModelSessionOptions): Promise<ModelSession> {
    const { device, model, params, lmax } = opts;
    const oversample = Math.max(1, Math.round(opts.oversample ?? 1));
    const niter = Math.max(0, Math.round(opts.niter ?? 1));
    const geometryModel = opts.geometry ?? mGeometryByKey(SPHERE_KEY)!;
    const geometryParams = opts.geometryParams ?? defaultGeometryParams(geometryModel);
    const { nlat, nphi } = gridForLmax(lmax, model.pdeg);
    const cfg = { lmax, mmax: lmax, nlat, nphi };
    const sht = await ShtPlan.create(device, cfg);
    let displaySht: ShtPlan | null = null;
    let deriv: DerivPlan | null = null;
    try {
      // The display plan shares nothing with the solver's beyond the
      // coefficients copied into it per readback; its grid is the solver's
      // scaled by the oversampling factor, so nphi stays a power of two (the
      // FFT path) for power-of-two factors.
      if (oversample > 1) {
        displaySht = await ShtPlan.create(device, {
          lmax,
          mmax: lmax,
          nlat: oversample * nlat,
          nphi: oversample * nphi,
        });
      }
      // Computes the theta/phi derivatives the geometry's metric quantities
      // (and, per step, the surface Laplace-Beltrami correction) need.
      deriv = await DerivPlan.create(device, sht);
      // The surface is built before the model, because the model takes it as
      // an argument. It is a one-off: compiled, evaluated, read back, and its
      // plan discarded — nothing of it survives into the timestep but sixteen
      // buffers of numbers (the embedding, and both metric formulations built
      // on it: the inverse metric quantities and the flux-form weights).
      const geometry = await Geometry.create({
        device,
        sht,
        cfg,
        source: opts.geometrySource ?? geometryModel.source,
        paramNames: geometryModel.params.map((p) => p.key),
        params: geometryParams,
        deriv,
      });
      const gpu = await GpuModel.create({
        device,
        sht,
        cfg,
        source: opts.source ?? model.source,
        paramNames: model.params.map((p) => p.key),
        state: model.state,
        view: model.species,
        geometry,
        deriv,
        niter,
      });
      gpu.setParams(params);
      return new ModelSession({
        device, model, cfg, sht, displaySht, gpu, params, oversample,
        geometry, geometryModel, deriv, niter,
      });
    } catch (e) {
      // The transform plans own GPU buffers; do not leak them on a compile error.
      deriv?.destroy();
      displaySht?.destroy();
      sht.destroy();
      throw e;
    }
  }

  /**
   * Vertex positions for the current render grid: the surface synthesized on
   * `viewSht`, interleaved xyz. Exact interpolation of the same coefficients
   * the solver sees, so the drawn surface is the one being solved on however
   * finely it is sampled.
   */
  renderPositions(): Promise<Float32Array> {
    return this.#geometry.positionsOn(this.viewSht);
  }

  /**
   * Change the surface in place, without recompiling or disturbing the run.
   * The geometry's shape in the bindings depends only on the grid, so the
   * compiled step does not change — only the numbers it reads. The caller
   * still has to rebuild the mesh from `renderPositions()`.
   */
  async setGeometry(
    geometryModel: MGeometry,
    params: ModelParams,
    source?: string,
  ): Promise<void> {
    const next = await Geometry.create({
      device: this.device,
      sht: this.sht,
      cfg: this.cfg,
      source: source ?? geometryModel.source,
      paramNames: geometryModel.params.map((p) => p.key),
      params,
      deriv: this.#deriv,
    });
    this.#geometry = next;
    this.#geometryModel = geometryModel;
    this.gpu.uploadGeometry(next);
    // The new surface brings a new preconditioner scale (GpuModel folds its
    // current geometry's jhat into every params upload).
    this.gpu.setParams(this.#params);
  }

  /** The plan whose grid `readSpecies` samples on — the display plan when
   *  oversampling, otherwise the solver's. Its cosTheta/nphi define the mesh. */
  get viewSht(): ShtPlan {
    return this.#displaySht ?? this.sht;
  }

  /**
   * Change the display oversampling in place. Display-only: the simulation
   * state, time and parameters are untouched, so the run continues seamlessly
   * on the new render grid. The caller must not have a readSpecies in flight —
   * its readback maps a buffer of the plan being destroyed.
   */
  async setOversample(oversample: number): Promise<void> {
    const os = Math.max(1, Math.round(oversample));
    await this.setDisplayGrid(os * this.cfg.nlat, os * this.cfg.nphi);
  }

  /**
   * Point the display plan at an arbitrary grid, rather than an integer
   * multiple of the solver's. Same contract as setOversample — display-only,
   * no readback may be in flight — and the same exactness argument, which does
   * not care about the ratio: the state is band-limited at lmax, so
   * synthesizing it anywhere is evaluation, not resampling. What this adds is a
   * grid that need not be *finer*: several sessions at different lmax can be
   * put on one common grid, which is what makes their fields directly
   * comparable point by point and lets one mesh serve all of them.
   */
  async setDisplayGrid(nlat: number, nphi: number): Promise<void> {
    const view = this.viewSht.cfg;
    if (nlat === view.nlat && nphi === view.nphi) return;
    const onSolverGrid = nlat === this.cfg.nlat && nphi === this.cfg.nphi;
    const next = onSolverGrid
      ? null
      : await ShtPlan.create(this.device, {
          lmax: this.cfg.lmax,
          mmax: this.cfg.mmax,
          nlat,
          nphi,
        });
    const old = this.#displaySht;
    this.#displaySht = next;
    this.#oversample = nlat / this.cfg.nlat;
    old?.destroy();
  }

  /** Run `init` from a seeded perturbation, resetting model time. */
  seed(seed: number): void {
    this.seedWith(seededNoise(this.npts, this.model.seedAmp, seed));
  }

  /**
   * Run `init` from a caller-supplied perturbation field, resetting model time.
   * `seed()` is this with the field the host's RNG produces on this session's
   * grid; supplying the field instead is how several sessions on *different*
   * grids can be started from the same band-limited initial condition, which is
   * the only way a comparison across lmax compares one problem rather than two
   * (see src/compare/sharedStart.ts).
   */
  seedWith(noise: Float32Array): void {
    if (noise.length !== this.npts) {
      throw new Error(`seedWith: noise must have length ${this.npts} (got ${noise.length})`);
    }
    this.gpu.init(noise);
    this.t = 0;
    this.steps = 0;
  }

  setParams(params: ModelParams): void {
    this.#params = params;
    this.gpu.setParams(params);
  }

  /** Advance `n` steps. Synchronous: records and submits, nothing read back. */
  step(n = 1): void {
    this.gpu.step(n);
    this.t += n * (this.#params.dt ?? 0);
    this.steps += n;
  }

  /**
   * Wait for the submitted steps to finish, without reading anything back.
   * This is the honest way to time the solver: a readback would add a GPU->CPU
   * round trip, which in a browser also crosses a process boundary and can cost
   * more than the steps themselves.
   */
  sync(): Promise<undefined> {
    return this.device.queue.onSubmittedWorkDone();
  }

  /**
   * Time a batch of `n` steps and return ms/step, leaving the simulation
   * exactly where it was: the spectral state is snapshotted before the batch
   * and restored after, and `t`/`steps` do not advance. One sync amortized
   * over the batch — the same measurement the desktop benchmark makes. The
   * grid view fields hold the batch's output until the next real step, so
   * step before reading them.
   */
  async measure(n: number): Promise<number> {
    this.gpu.snapshotState();
    const t0 = performance.now();
    this.gpu.step(n);
    await this.sync();
    const ms = (performance.now() - t0) / n;
    this.gpu.restoreState();
    return ms;
  }

  /** Read a named value (a grid field or the spectral state). */
  read(name: string): Promise<Float32Array> {
    return this.gpu.read(name);
  }

  /**
   * Read species `k` at render resolution (`viewSht`'s grid). Without
   * oversampling this is the grid field the .m returned. With oversampling the
   * spectral state is synthesized on the finer grid instead — the same field,
   * since the models define each species as synth of its state, evaluated
   * exactly on more points.
   */
  readSpecies(k: number): Promise<Float32Array> {
    if (!this.#displaySht) return this.read(this.model.species[k]);
    const state = this.model.state[k];
    const buf = this.gpu.valueBuffer(state);
    if (!buf) throw new Error(`readSpecies: no buffer for state '${state}'`);
    return this.#displaySht.synthFrom(buf);
  }

  describe(): { init: string[]; step: string[] } {
    return this.gpu.describe();
  }

  destroy(): void {
    this.gpu.destroy();
    this.#deriv.destroy();
    this.#displaySht?.destroy();
    this.sht.destroy();
  }
}
