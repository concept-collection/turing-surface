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

  /** Bindings for dtheta(qlmIn) -> spatOut, against caller-owned buffers. */
  createDthetaBinding(qlmIn: GPUBuffer, spatOut: GPUBuffer): DerivBinding {
    const shuffle = this.device.createBindGroup({
      layout: this.pipeDtheta.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bufAPlus } },
        { binding: 1, resource: { buffer: this.bufAMinus } },
        { binding: 2, resource: { buffer: qlmIn } },
        { binding: 3, resource: { buffer: this.scratch } },
      ],
    });
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
    const shuffle = this.device.createBindGroup({
      layout: this.pipeDphi.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bufMOf } },
        { binding: 1, resource: { buffer: qlmIn } },
        { binding: 2, resource: { buffer: this.scratch } },
      ],
    });
    const sht = this.sht.createSynthBinding(this.scratch, spatOut);
    return { shuffle, sht };
  }

  /** Record dtheta into an existing compute pass. */
  encodeDthetaInto(pass: GPUComputePassEncoder, b: DerivBinding): void {
    pass.setPipeline(this.pipeDtheta);
    pass.setBindGroup(0, b.shuffle);
    pass.dispatchWorkgroups(Math.ceil(this.nlm / WG));
    this.sht.encodeSynthInto(pass, b.sht);
    pass.setPipeline(this.pipeDivide);
    pass.setBindGroup(0, b.divide!);
    pass.dispatchWorkgroups(Math.ceil(this.npts / WG));
  }

  /** Record dphi into an existing compute pass. */
  encodeDphiInto(pass: GPUComputePassEncoder, b: DerivBinding): void {
    pass.setPipeline(this.pipeDphi);
    pass.setBindGroup(0, b.shuffle);
    pass.dispatchWorkgroups(Math.ceil(this.nlm / WG));
    this.sht.encodeSynthInto(pass, b.sht);
  }

  /** CPU convenience: qlm (interleaved [re,im], length 2*nlm) -> grid field. */
  async dtheta(qlm: Float32Array): Promise<Float32Array> {
    return this.#runToGrid(qlm, true);
  }

  /** CPU convenience: qlm (interleaved [re,im], length 2*nlm) -> grid field. */
  async dphi(qlm: Float32Array): Promise<Float32Array> {
    return this.#runToGrid(qlm, false);
  }

  async #runToGrid(qlm: Float32Array, withDivide: boolean): Promise<Float32Array> {
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
      const binding = withDivide
        ? this.createDthetaBinding(qlmIn, spatOut)
        : this.createDphiBinding(qlmIn, spatOut);
      const enc = dev.createCommandEncoder({ label: 'deriv-run' });
      const pass = enc.beginComputePass({ label: 'deriv-run' });
      if (withDivide) this.encodeDthetaInto(pass, binding);
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
