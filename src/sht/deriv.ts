/**
 * First derivatives of a scalar field, coefficients -> grid: dtheta and dphi
 * (evolving_surface/notes/algos.tex Algorithm 1, theta/phi branches only --
 * the Laplace-Beltrami operator built on these never needs the second-
 * derivative/curvature branches, so they are not ported).
 *
 * Both derivatives start with a shuffle in coefficient space (the theta
 * branch's +-1 index gather via the alpha recurrence, the phi branch's i*m
 * row-swap) and then reuse the *existing* Legendre+Fourier synthesis
 * pipeline (ShtPlan.createSynthBinding/encodeSynthInto) unchanged -- neither
 * derivative touches the Legendre recurrence stage itself. dtheta
 * additionally divides by sin(theta) on the grid afterwards.
 *
 * The two shuffles are also exposed on their own, coefficients -> coefficients
 * (`dthetac`, `dphic`), because the six-transform Laplace-Beltrami operator of
 * docs/reduced-transforms.md needs them apart from a
 * synthesis, and needs them twice: once on the field (steps 1-2) and once on
 * the two fluxes (step 5, which is the *same* alpha^+/alpha^- gather, not its
 * transpose). Everything above is then a composition of them:
 *
 *   dphi(U)   == synth(dphic(U))
 *   dtheta(U) == synth(dthetac(U)) / sin(theta)
 */
import type { ShtPlan, ShtBinding } from './sht.ts';
import { derivCoeffs } from './derivCoeffs.ts';
import { dthetaShuffleWGSL, dphiShuffleWGSL, divideSinThetaWGSL } from './wgsl/deriv.ts';

const WG = 64;

async function makePipeline(
  device: GPUDevice,
  code: string,
  entryPoint: string,
): Promise<GPUComputePipeline> {
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ code, label: entryPoint });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) {
    throw new Error(
      `WGSL compile error in ${entryPoint}:\n` +
        errors.map((e) => `  ${e.lineNum}:${e.linePos} ${e.message}`).join('\n'),
    );
  }
  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module, entryPoint },
    label: entryPoint,
  });
  const err = await device.popErrorScope();
  if (err) throw new Error(`pipeline ${entryPoint}: ${err.message}`);
  return pipeline;
}

/** Bindings for one dtheta/dphi call against caller-supplied buffers. */
export interface DerivBinding {
  readonly shuffle: GPUBindGroup;
  readonly sht: ShtBinding;
  /** Only present for dtheta: the post-synthesis divide by sin(theta). */
  readonly divide?: GPUBindGroup;
}

export class DerivPlan {
  private device: GPUDevice;
  private sht: ShtPlan;
  private nlm: number;
  private npts: number;

  private bufAPlus!: GPUBuffer;
  private bufAMinus!: GPUBuffer;
  private bufMOf!: GPUBuffer;
  private bufSinTheta!: GPUBuffer;
  /** Scratch coefficient buffer for the shuffled input to synth -- shared
   *  sequentially like ShtPlan's fmBuf, since ops within one pass execute
   *  in submission order. */
  private scratch!: GPUBuffer;

  private pipeDtheta!: GPUComputePipeline;
  private pipeDphi!: GPUComputePipeline;
  private pipeDivide!: GPUComputePipeline;

  private constructor(device: GPUDevice, sht: ShtPlan) {
    this.device = device;
    this.sht = sht;
    this.nlm = sht.nlm;
    this.npts = sht.cfg.nlat * sht.cfg.nphi;
  }

  static async create(device: GPUDevice, sht: ShtPlan): Promise<DerivPlan> {
    const plan = new DerivPlan(device, sht);
    await plan.init();
    return plan;
  }

  private async init(): Promise<void> {
    const { nlat, nphi } = this.sht.cfg;
    const dev = this.device;

    const { aPlus, aMinus, mOf } = derivCoeffs(this.sht.cfg.lmax, this.sht.cfg.mmax);
    const sinTheta = new Float32Array(nlat);
    for (let i = 0; i < nlat; i++) {
      const ct = this.sht.cosTheta[i];
      sinTheta[i] = Math.sqrt(Math.max(0, 1 - ct * ct));
    }

    const mk = (label: string, size: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) =>
      dev.createBuffer({ label, size, usage });
    this.bufAPlus = mk('deriv-aplus', 4 * this.nlm);
    this.bufAMinus = mk('deriv-aminus', 4 * this.nlm);
    this.bufMOf = mk('deriv-mof', 4 * this.nlm);
    this.bufSinTheta = mk('deriv-sintheta', 4 * nlat);
    this.scratch = mk(
      'deriv-scratch',
      8 * this.nlm,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    );

    dev.queue.writeBuffer(this.bufAPlus, 0, new Float32Array(aPlus));
    dev.queue.writeBuffer(this.bufAMinus, 0, new Float32Array(aMinus));
    dev.queue.writeBuffer(this.bufMOf, 0, mOf as Uint32Array<ArrayBuffer>);
    dev.queue.writeBuffer(this.bufSinTheta, 0, sinTheta);

    const [pDtheta, pDphi, pDivide] = await Promise.all([
      makePipeline(dev, dthetaShuffleWGSL({ nlm: this.nlm }), 'dtheta_shuffle'),
      makePipeline(dev, dphiShuffleWGSL({ nlm: this.nlm }), 'dphi_shuffle'),
      makePipeline(dev, divideSinThetaWGSL({ nlat, nphi }), 'divide_sin_theta'),
    ]);
    this.pipeDtheta = pDtheta;
    this.pipeDphi = pDphi;
    this.pipeDivide = pDivide;
  }

  /**
   * Bind group for the coefficient-space half of dtheta on its own:
   * v_l^m = alpha^+(l-1,m) u_{l-1}^m + alpha^-(l+1,m) u_{l+1}^m, the
   * coefficients of sin(theta) * dtheta(u). Input and output must be
   * different buffers -- WebGPU forbids binding one buffer as both readable
   * and writable storage in a dispatch, and the gather reads l+-1 anyway.
   */
  createDthetacBinding(qlmIn: GPUBuffer, qlmOut: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.pipeDtheta.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bufAPlus } },
        { binding: 1, resource: { buffer: this.bufAMinus } },
        { binding: 2, resource: { buffer: qlmIn } },
        { binding: 3, resource: { buffer: qlmOut } },
      ],
    });
  }

  /** Bind group for the coefficient-space half of dphi on its own:
   *  (dphi u)_l^m = i*m*u_l^m. Same buffer restriction as dthetac. */
  createDphicBinding(qlmIn: GPUBuffer, qlmOut: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.pipeDphi.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bufMOf } },
        { binding: 1, resource: { buffer: qlmIn } },
        { binding: 2, resource: { buffer: qlmOut } },
      ],
    });
  }

  /** Record the bare alpha^+/alpha^- shift into an existing compute pass. */
  encodeDthetacInto(pass: GPUComputePassEncoder, bindGroup: GPUBindGroup): void {
    pass.setPipeline(this.pipeDtheta);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.nlm / WG));
  }

  /** Record the bare i*m multiply into an existing compute pass. */
  encodeDphicInto(pass: GPUComputePassEncoder, bindGroup: GPUBindGroup): void {
    pass.setPipeline(this.pipeDphi);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.nlm / WG));
  }

  /** Bindings for dtheta(qlmIn) -> spatOut, against caller-owned buffers. */
  createDthetaBinding(qlmIn: GPUBuffer, spatOut: GPUBuffer): DerivBinding {
    const shuffle = this.createDthetacBinding(qlmIn, this.scratch);
    const sht = this.sht.createSynthBinding(this.scratch, spatOut);
    const divide = this.device.createBindGroup({
      layout: this.pipeDivide.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bufSinTheta } },
        { binding: 1, resource: { buffer: spatOut } },
      ],
    });
    return { shuffle, sht, divide };
  }

  /** Bindings for dphi(qlmIn) -> spatOut, against caller-owned buffers. */
  createDphiBinding(qlmIn: GPUBuffer, spatOut: GPUBuffer): DerivBinding {
    const shuffle = this.createDphicBinding(qlmIn, this.scratch);
    const sht = this.sht.createSynthBinding(this.scratch, spatOut);
    return { shuffle, sht };
  }

  /** Record dtheta into an existing compute pass. */
  encodeDthetaInto(pass: GPUComputePassEncoder, b: DerivBinding): void {
    this.encodeSinDthetaInto(pass, b);
    pass.setPipeline(this.pipeDivide);
    pass.setBindGroup(0, b.divide!);
    pass.dispatchWorkgroups(Math.ceil(this.npts / WG));
  }

  /** Record dtheta *without* its final division: the grid values of
   *  sin(theta) * dtheta(u), which unlike dtheta(u) itself is a smooth
   *  function on the sphere. Takes a dtheta binding and simply stops early. */
  encodeSinDthetaInto(pass: GPUComputePassEncoder, b: DerivBinding): void {
    this.encodeDthetacInto(pass, b.shuffle);
    this.sht.encodeSynthInto(pass, b.sht);
  }

  /** Record dphi into an existing compute pass. */
  encodeDphiInto(pass: GPUComputePassEncoder, b: DerivBinding): void {
    this.encodeDphicInto(pass, b.shuffle);
    this.sht.encodeSynthInto(pass, b.sht);
  }

  /** CPU convenience: qlm (interleaved [re,im], length 2*nlm) -> grid field. */
  async dtheta(qlm: Float32Array): Promise<Float32Array> {
    return this.#runToGrid(qlm, 'dtheta');
  }

  /** CPU convenience: sin(theta) * dtheta(u) on the grid, the undivided
   *  synthesis of the alpha shift. What the flux-form metric precompute
   *  (src/geom/metric.ts) is built from. */
  async sinDtheta(qlm: Float32Array): Promise<Float32Array> {
    return this.#runToGrid(qlm, 'sinDtheta');
  }

  /** CPU convenience: qlm (interleaved [re,im], length 2*nlm) -> grid field. */
  async dphi(qlm: Float32Array): Promise<Float32Array> {
    return this.#runToGrid(qlm, 'dphi');
  }

  async #runToGrid(
    qlm: Float32Array,
    mode: 'dtheta' | 'sinDtheta' | 'dphi',
  ): Promise<Float32Array> {
    if (qlm.length !== 2 * this.nlm) throw new Error(`qlm must have length ${2 * this.nlm}`);
    const dev = this.device;
    const qlmIn = dev.createBuffer({
      label: 'deriv-qlm-in',
      size: 8 * this.nlm,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const spatOut = dev.createBuffer({
      label: 'deriv-spat-out',
      size: 4 * this.npts,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const stage = dev.createBuffer({
      label: 'deriv-stage',
      size: 4 * this.npts,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    try {
      dev.queue.writeBuffer(qlmIn, 0, qlm as Float32Array<ArrayBuffer>);
      const binding =
        mode === 'dphi'
          ? this.createDphiBinding(qlmIn, spatOut)
          : this.createDthetaBinding(qlmIn, spatOut);
      const enc = dev.createCommandEncoder({ label: 'deriv-run' });
      const pass = enc.beginComputePass({ label: 'deriv-run' });
      if (mode === 'dtheta') this.encodeDthetaInto(pass, binding);
      else if (mode === 'sinDtheta') this.encodeSinDthetaInto(pass, binding);
      else this.encodeDphiInto(pass, binding);
      pass.end();
      enc.copyBufferToBuffer(spatOut, 0, stage, 0, 4 * this.npts);
      dev.queue.submit([enc.finish()]);
      await stage.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(stage.getMappedRange().slice(0));
      stage.unmap();
      return out;
    } finally {
      qlmIn.destroy();
      spatOut.destroy();
      stage.destroy();
    }
  }

  destroy(): void {
    for (const b of [
      this.bufAPlus, this.bufAMinus, this.bufMOf, this.bufSinTheta, this.scratch,
    ]) b?.destroy();
  }
}
