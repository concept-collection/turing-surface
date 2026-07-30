/**
 * WebGPU spherical harmonic transform plan (scalar transforms, fp32).
 *
 * Mirrors the structure of the SHTNS CUDA backend (sht_gpu.cu):
 * host-side f64 precomputation of grid + recurrence coefficients, shader
 * source generated with sizes baked in (SHTNS uses NVRTC; WGSL is always
 * runtime-compiled), then per-transform: Legendre stage + Fourier stage.
 */
import { gaussNodesWeights } from './gauss.ts';
import { legendreCoeffs } from './coeffs.ts';
import { nlmCalc, validateConfig, isPowerOfTwo, type ShtConfig } from './layout.ts';
import { legSynthWGSL, legAnalysWGSL } from './wgsl/leg.ts';
import {
  fftSynthWGSL,
  fftAnalysWGSL,
  fftSynthRealWGSL,
  fftAnalysRealWGSL,
  dftSynthWGSL,
  dftAnalysWGSL,
  fftThreads,
} from './wgsl/fourier.ts';

export type FourierMode = 'auto' | 'fft' | 'dft';

/** The two bind groups (Legendre stage, Fourier stage) of one transform. */
export interface ShtBinding {
  readonly bgLeg: GPUBindGroup;
  readonly bgFour: GPUBindGroup;
}

const bgEntries = (bufs: GPUBuffer[]) =>
  bufs.map((buffer, binding) => ({ binding, resource: { buffer } }));

export interface ShtOptions {
  /** Fourier stage implementation.  'auto' picks fft when nphi is a power of two that fits in workgroup memory. */
  fourier?: FourierMode;
}

const WG_SYNTH = 64;

/**
 * Tuning knob, for A/B-ing a change without editing code. Reads globalThis
 * first (set it before creating a plan, as scripts/_ab.ts does), then the
 * environment, so `SHT_SUBGROUPS=0 npm run bench:sht` works too. `process` is
 * absent in the browser, where only the globalThis form applies.
 */
function tuning(name: string): unknown {
  const g = (globalThis as Record<string, unknown>)[name];
  if (g !== undefined) return g;
  const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.[name];
  if (env === undefined || env === '') return undefined;
  if (env === '1' || env === 'true') return true;
  if (env === '0' || env === 'false') return false;
  const n = Number(env);
  return Number.isFinite(n) ? n : env;
}

/**
 * Workgroup size for the analysis Legendre reduction. The right answer differs
 * between the two reduction strategies, so it is chosen per strategy.
 *
 * Measured on an RTX PRO 6000 Blackwell (analysis, us). Shared-memory tree,
 * where each doubling of wgAnalys costs another barrier per l-pair:
 *
 *          wgAnalys:   16     32     64    128    256
 *   nlat=128          43.2   40.6   42.6   46.7   52.9   -> 32
 *   nlat=256         104.0   85.6   87.8   89.9  100.0   -> 32
 *   nlat=512         408.8  216.9  184.4  190.8  204.8   -> 64
 *
 * i.e. max(32, nlat/8). A flat 32 would be worse than the old default of 256 at
 * nlat=512, so it cannot be fitted on one grid. With subgroupAdd the barrier
 * count stops growing with wgAnalys and the picture inverts: threads in flight,
 * (mmax+1) * wgAnalys, becomes binding, since analysis dispatches only mmax+1
 * workgroups. 128 then wins at every grid (round trip, us):
 *
 *   128x256  37.8 (vs 38.3),  256x512  66.6 (vs 74.7),  512x1024  131.9 (vs 156.6)
 */
function defaultWgAnalys(nlat: number, limit: number, subgroups: boolean): number {
  if (subgroups) {
    // capped at nlat so small grids do not launch threads with no latitude to own
    let cap = 1;
    while (cap < nlat) cap *= 2;
    return Math.min(128, limit, cap);
  }
  const target = Math.max(32, nlat / 8);
  let wg = 1;
  while (wg < target) wg *= 2; // the tree reduction halves, so a power of two
  return Math.min(wg, limit);
}

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

export class ShtPlan {
  readonly cfg: ShtConfig;
  readonly nlm: number;
  readonly fourierMode: 'fft' | 'dft';
  /** Latitudes leg_synth walks: nlat/2 when parity folding. */
  readonly legLat: number = 0;
  /** Colatitudes theta_i (f64, increasing: north to south). */
  readonly theta: Float64Array;
  readonly cosTheta: Float64Array;
  readonly gaussWeights: Float64Array;

  private device: GPUDevice;
  private bufAb!: GPUBuffer;
  private bufAmm!: GPUBuffer;
  private bufCtstw!: GPUBuffer;
  private bufTrig!: GPUBuffer;
  /** Spectral input (synthesis) — write with queue.writeBuffer or use synth(). */
  readonly qlmIn!: GPUBuffer;
  /** Spectral output (analysis). */
  readonly qlmOut!: GPUBuffer;
  /** Fourier-space intermediate [(m)*nlat + ilat], complex f32. COPY_SRC so the
   *  stage boundary is observable: a transform is Legendre-then-Fourier, and
   *  scripts/diagnose-sht.ts tells the two apart by reading this. */
  readonly fmBuf!: GPUBuffer;
  /** Spatial field [ilat*nphi + iphi], f32. */
  readonly spatBuf!: GPUBuffer;
  private stageSpat!: GPUBuffer;
  private stageQ!: GPUBuffer;

  private pipeLegSynth!: GPUComputePipeline;
  private pipeLegAnalys!: GPUComputePipeline;
  private pipeFourSynth!: GPUComputePipeline;
  private pipeFourAnalys!: GPUComputePipeline;
  private bgLegSynth!: GPUBindGroup;
  private bgLegAnalys!: GPUBindGroup;
  private bgFourSynth!: GPUBindGroup;
  private bgFourAnalys!: GPUBindGroup;

  private constructor(device: GPUDevice, cfg: ShtConfig, fourierMode: 'fft' | 'dft') {
    this.device = device;
    this.cfg = cfg;
    this.nlm = nlmCalc(cfg.lmax, cfg.mmax);
    this.fourierMode = fourierMode;
    const { x, w } = gaussNodesWeights(cfg.nlat);
    this.cosTheta = x;
    this.gaussWeights = w;
    this.theta = new Float64Array(cfg.nlat);
    for (let i = 0; i < cfg.nlat; i++) this.theta[i] = Math.acos(x[i]);
  }

  static async create(device: GPUDevice, cfg: ShtConfig, opts: ShtOptions = {}): Promise<ShtPlan> {
    validateConfig(cfg);
    const want = opts.fourier ?? 'auto';
    const fftFits =
      isPowerOfTwo(cfg.nphi) &&
      16 * cfg.nphi <= device.limits.maxComputeWorkgroupStorageSize &&
      fftThreads(cfg.nphi) <= device.limits.maxComputeInvocationsPerWorkgroup;
    if (want === 'fft' && !fftFits) {
      throw new Error(
        `fourier:'fft' requires power-of-two nphi with 16*nphi <= maxComputeWorkgroupStorageSize ` +
          `(nphi=${cfg.nphi}, limit=${device.limits.maxComputeWorkgroupStorageSize})`,
      );
    }
    const mode: 'fft' | 'dft' = want === 'dft' ? 'dft' : fftFits ? 'fft' : 'dft';
    const plan = new ShtPlan(device, cfg, mode);
    await plan.init();
    return plan;
  }

  private async init(): Promise<void> {
    const { lmax, mmax, nlat, nphi } = this.cfg;
    const dev = this.device;
    const self = this as {
      -readonly [k in keyof ShtPlan]: ShtPlan[k];
    };

    // --- host precomputation (f64), then downcast to f32 for upload ---
    const { amm, ab } = legendreCoeffs(lmax, mmax);
    const ctstw = new Float32Array(3 * nlat);
    for (let i = 0; i < nlat; i++) {
      ctstw[i] = this.cosTheta[i];
      ctstw[nlat + i] = Math.sqrt(1 - this.cosTheta[i] * this.cosTheta[i]);
      ctstw[2 * nlat + i] = this.gaussWeights[i] * ((2 * Math.PI) / nphi);
    }
    // twiddle/phase table in f64 (device sin/cos is too inaccurate: ~2^-11 under Vulkan)
    const trig = new Float32Array(2 * nphi);
    for (let k = 0; k < nphi; k++) {
      trig[2 * k] = Math.cos((2 * Math.PI * k) / nphi);
      trig[2 * k + 1] = Math.sin((2 * Math.PI * k) / nphi);
    }

    const mkBuf = (label: string, size: number, usage: GPUBufferUsageFlags) =>
      dev.createBuffer({ label, size, usage });
    this.bufAb = mkBuf('sht-ab', 8 * this.nlm, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.bufAmm = mkBuf('sht-amm', 4 * (mmax + 1), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.bufCtstw = mkBuf('sht-ctstw', 4 * 3 * nlat, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.bufTrig = mkBuf('sht-trig', 8 * nphi, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    self.qlmIn = mkBuf('sht-qlm-in', 8 * this.nlm, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    self.qlmOut = mkBuf('sht-qlm-out', 8 * this.nlm, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    self.fmBuf = mkBuf('sht-fm', 8 * (mmax + 1) * nlat, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    self.spatBuf = mkBuf('sht-spat', 4 * nlat * nphi, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.stageSpat = mkBuf('sht-stage-spat', 4 * nlat * nphi, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    this.stageQ = mkBuf('sht-stage-q', 8 * this.nlm, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

    dev.queue.writeBuffer(this.bufAb, 0, new Float32Array(ab));
    dev.queue.writeBuffer(this.bufAmm, 0, new Float32Array(amm));
    dev.queue.writeBuffer(this.bufCtstw, 0, ctstw);
    dev.queue.writeBuffer(this.bufTrig, 0, trig);

    // --- shaders / pipelines ---
    const subgroups = tuning('SHT_SUBGROUPS') !== false && dev.features.has('subgroups');
    // parity folding needs an equator-symmetric grid; Gauss nodes are, if nlat is even
    const parity = tuning('SHT_PARITY') !== false && nlat % 2 === 0;
    const wgAnalys =
      (tuning('SHT_WG_ANALYS') as number | undefined) ??
      defaultWgAnalys(nlat, dev.limits.maxComputeInvocationsPerWorkgroup, subgroups);
    const legP = {
      lmax,
      mmax,
      nlat,
      wgSynth: WG_SYNTH,
      wgAnalys,
      subgroups,
      spanPairs: tuning('SHT_SPAN_PAIRS') as number | undefined,
      parity,
    };
    (this as { legLat: number }).legLat = parity ? nlat / 2 : nlat;
    const fourP = { mmax, nlat, nphi, radix: (tuning('SHT_RADIX') as number | undefined) ?? 4 };
    // The spatial field is real (layout.ts stores m >= 0 only), so the Fourier
    // stage can run an nphi/2-point complex FFT plus a recombination instead of
    // a full nphi-point one: half the arithmetic and half the workgroup storage.
    // The complex kernels remain for a future complex-valued field, and are what
    // SHT_REAL_FFT=0 selects.
    const realFft =
      this.fourierMode === 'fft' && nphi % 2 === 0 && tuning('SHT_REAL_FFT') !== false;
    const fftS = realFft ? fftSynthRealWGSL : fftSynthWGSL;
    const fftA = realFft ? fftAnalysRealWGSL : fftAnalysWGSL;
    const [pLegS, pLegA, pFourS, pFourA] = await Promise.all([
      makePipeline(dev, legSynthWGSL(legP), 'leg_synth'),
      makePipeline(dev, legAnalysWGSL(legP), 'leg_analys'),
      makePipeline(
        dev,
        this.fourierMode === 'fft' ? fftS(fourP) : dftSynthWGSL(fourP),
        this.fourierMode === 'fft' ? 'fft_synth' : 'dft_synth',
      ),
      makePipeline(
        dev,
        this.fourierMode === 'fft' ? fftA(fourP) : dftAnalysWGSL(fourP),
        this.fourierMode === 'fft' ? 'fft_analys' : 'dft_analys',
      ),
    ]);
    this.pipeLegSynth = pLegS;
    this.pipeLegAnalys = pLegA;
    this.pipeFourSynth = pFourS;
    this.pipeFourAnalys = pFourA;

    const entries = bgEntries;
    this.bgLegSynth = dev.createBindGroup({
      layout: pLegS.getBindGroupLayout(0),
      entries: entries([this.bufAb, this.bufAmm, this.bufCtstw, this.qlmIn, this.fmBuf]),
    });
    this.bgLegAnalys = dev.createBindGroup({
      layout: pLegA.getBindGroupLayout(0),
      entries: entries([this.bufAb, this.bufAmm, this.bufCtstw, this.fmBuf, this.qlmOut]),
    });
    this.bgFourSynth = dev.createBindGroup({
      layout: pFourS.getBindGroupLayout(0),
      entries: entries([this.fmBuf, this.spatBuf, this.bufTrig]),
    });
    this.bgFourAnalys = dev.createBindGroup({
      layout: pFourA.getBindGroupLayout(0),
      entries: entries([this.spatBuf, this.fmBuf, this.bufTrig]),
    });
  }

  /**
   * Bind groups for one transform against caller-supplied spectral/spatial
   * buffers, so a transform can read and write buffers it does not own (the
   * .m-driven executor keeps a buffer per IR variable). Build these once at
   * plan time, not per step. `fmBuf` stays internal scratch: passes and
   * dispatches within a submission execute in order, so sequential transforms
   * can share it.
   */
  createSynthBinding(qlmIn: GPUBuffer, spatOut: GPUBuffer): ShtBinding {
    return {
      bgLeg: this.device.createBindGroup({
        layout: this.pipeLegSynth.getBindGroupLayout(0),
        entries: bgEntries([this.bufAb, this.bufAmm, this.bufCtstw, qlmIn, this.fmBuf]),
      }),
      bgFour: this.device.createBindGroup({
        layout: this.pipeFourSynth.getBindGroupLayout(0),
        entries: bgEntries([this.fmBuf, spatOut, this.bufTrig]),
      }),
    };
  }

  createAnalysBinding(spatIn: GPUBuffer, qlmOut: GPUBuffer): ShtBinding {
    return {
      bgFour: this.device.createBindGroup({
        layout: this.pipeFourAnalys.getBindGroupLayout(0),
        entries: bgEntries([spatIn, this.fmBuf, this.bufTrig]),
      }),
      bgLeg: this.device.createBindGroup({
        layout: this.pipeLegAnalys.getBindGroupLayout(0),
        entries: bgEntries([this.bufAb, this.bufAmm, this.bufCtstw, this.fmBuf, qlmOut]),
      }),
    };
  }

  /** Record synthesis into an existing compute pass. */
  encodeSynthInto(pass: GPUComputePassEncoder, b: ShtBinding): void {
    const { mmax, nlat, nphi } = this.cfg;
    pass.setPipeline(this.pipeLegSynth);
    pass.setBindGroup(0, b.bgLeg);
    pass.dispatchWorkgroups(Math.ceil(this.legLat / WG_SYNTH), mmax + 1);
    pass.setPipeline(this.pipeFourSynth);
    pass.setBindGroup(0, b.bgFour);
    if (this.fourierMode === 'fft') {
      pass.dispatchWorkgroups(nlat);
    } else {
      pass.dispatchWorkgroups(Math.ceil(nphi / 64), nlat);
    }
  }

  /** Record analysis into an existing compute pass. */
  encodeAnalysInto(pass: GPUComputePassEncoder, b: ShtBinding): void {
    const { mmax, nlat } = this.cfg;
    pass.setPipeline(this.pipeFourAnalys);
    pass.setBindGroup(0, b.bgFour);
    if (this.fourierMode === 'fft') {
      pass.dispatchWorkgroups(nlat);
    } else {
      pass.dispatchWorkgroups(Math.ceil((mmax + 1) / 64), nlat);
    }
    pass.setPipeline(this.pipeLegAnalys);
    pass.setBindGroup(0, b.bgLeg);
    pass.dispatchWorkgroups(mmax + 1);
  }

  /** Record the synthesis (spectral qlmIn -> spatial spatBuf) into an encoder. */
  encodeSynth(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: 'sht-synth' });
    this.encodeSynthInto(pass, { bgLeg: this.bgLegSynth, bgFour: this.bgFourSynth });
    pass.end();
  }

  /**
   * Diagnostics: encode one stage alone, in its own pass, so a timestamp query
   * can measure just that kernel. The solver wants both stages in a shared pass
   * and should use encodeSynthInto/encodeAnalysInto; this exists because
   * inferring per-kernel cost by subtracting trivially-sized runs is unreliable.
   */
  encodeStage(
    encoder: GPUCommandEncoder,
    stage: 'legSynth' | 'fourSynth' | 'fourAnalys' | 'legAnalys',
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void {
    const { mmax, nlat, nphi } = this.cfg;
    const fft = this.fourierMode === 'fft';
    const pass = encoder.beginComputePass({ label: `sht-${stage}`, timestampWrites });
    switch (stage) {
      case 'legSynth':
        pass.setPipeline(this.pipeLegSynth);
        pass.setBindGroup(0, this.bgLegSynth);
        pass.dispatchWorkgroups(Math.ceil(this.legLat / WG_SYNTH), mmax + 1);
        break;
      case 'fourSynth':
        pass.setPipeline(this.pipeFourSynth);
        pass.setBindGroup(0, this.bgFourSynth);
        if (fft) pass.dispatchWorkgroups(nlat);
        else pass.dispatchWorkgroups(Math.ceil(nphi / 64), nlat);
        break;
      case 'fourAnalys':
        pass.setPipeline(this.pipeFourAnalys);
        pass.setBindGroup(0, this.bgFourAnalys);
        if (fft) pass.dispatchWorkgroups(nlat);
        else pass.dispatchWorkgroups(Math.ceil((mmax + 1) / 64), nlat);
        break;
      case 'legAnalys':
        pass.setPipeline(this.pipeLegAnalys);
        pass.setBindGroup(0, this.bgLegAnalys);
        pass.dispatchWorkgroups(mmax + 1);
        break;
    }
    pass.end();
  }

  /** Record the analysis (spatial spatBuf -> spectral qlmOut) into an encoder. */
  encodeAnalys(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: 'sht-analys' });
    this.encodeAnalysInto(pass, { bgLeg: this.bgLegAnalys, bgFour: this.bgFourAnalys });
    pass.end();
  }

  /**
   * Spectral -> spatial.  qlm: interleaved [re, im], SHTNS LM ordering,
   * length 2*nlm.  Returns the spatial field, length nlat*nphi.
   */
  async synth(qlm: Float32Array): Promise<Float32Array> {
    const { nlat, nphi } = this.cfg;
    if (qlm.length !== 2 * this.nlm) throw new Error(`qlm must have length ${2 * this.nlm}`);
    this.device.queue.writeBuffer(this.qlmIn, 0, qlm as Float32Array<ArrayBuffer>);
    const enc = this.device.createCommandEncoder();
    this.encodeSynth(enc);
    enc.copyBufferToBuffer(this.spatBuf, 0, this.stageSpat, 0, 4 * nlat * nphi);
    this.device.queue.submit([enc.finish()]);
    await this.stageSpat.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.stageSpat.getMappedRange().slice(0));
    this.stageSpat.unmap();
    return out;
  }

  /**
   * Spectral -> spatial, with the coefficients read from a caller-owned GPU
   * buffer (interleaved [re, im], 8*nlm bytes, COPY_SRC) instead of uploaded
   * from the CPU. This is how a field already on the device — a model's
   * spectral state — is evaluated on this plan's grid, e.g. a finer display
   * grid than the one the coefficients were produced on.
   */
  async synthFrom(qlmSrc: GPUBuffer): Promise<Float32Array> {
    const { nlat, nphi } = this.cfg;
    const enc = this.device.createCommandEncoder({ label: 'sht-synth-from' });
    enc.copyBufferToBuffer(qlmSrc, 0, this.qlmIn, 0, 8 * this.nlm);
    this.encodeSynth(enc);
    enc.copyBufferToBuffer(this.spatBuf, 0, this.stageSpat, 0, 4 * nlat * nphi);
    this.device.queue.submit([enc.finish()]);
    await this.stageSpat.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.stageSpat.getMappedRange().slice(0));
    this.stageSpat.unmap();
    return out;
  }

  /** Spatial -> spectral.  spat: length nlat*nphi.  Returns interleaved qlm, length 2*nlm. */
  async analys(spat: Float32Array): Promise<Float32Array> {
    const { nlat, nphi } = this.cfg;
    if (spat.length !== nlat * nphi) throw new Error(`spat must have length ${nlat * nphi}`);
    this.device.queue.writeBuffer(this.spatBuf, 0, spat as Float32Array<ArrayBuffer>);
    const enc = this.device.createCommandEncoder();
    this.encodeAnalys(enc);
    enc.copyBufferToBuffer(this.qlmOut, 0, this.stageQ, 0, 8 * this.nlm);
    this.device.queue.submit([enc.finish()]);
    await this.stageQ.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.stageQ.getMappedRange().slice(0));
    this.stageQ.unmap();
    return out;
  }

  destroy(): void {
    for (const b of [
      this.bufAb, this.bufAmm, this.bufCtstw, this.bufTrig, this.qlmIn, this.qlmOut,
      this.fmBuf, this.spatBuf, this.stageSpat, this.stageQ,
    ]) b?.destroy();
  }
}

/** Best-effort human-readable adapter name, so it is clear which GPU (or
 *  software rasterizer) is actually running the transforms. */
export async function describeAdapter(device: GPUDevice): Promise<string> {
  const fmt = (info: GPUAdapterInfo | undefined): string => {
    if (!info) return '';
    const parts = [info.description, info.device, info.vendor].filter(
      (s): s is string => !!s && s.length > 0,
    );
    const name = parts[0] ?? '';
    return info.architecture && !name.includes(info.architecture)
      ? `${name} (${info.architecture})`.trim()
      : name;
  };
  const own = fmt((device as GPUDevice & { adapterInfo?: GPUAdapterInfo }).adapterInfo);
  if (own) return own;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return fmt(adapter?.info);
  } catch {
    return '';
  }
}

/** Request an adapter/device suitable for the transforms. */
export async function requestShtDevice(): Promise<GPUDevice> {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter available');
  // ask for a larger workgroup storage if the adapter offers it (bigger FFTs)
  const wgStorage = Math.min(adapter.limits.maxComputeWorkgroupStorageSize, 32768);
  // `subgroups` lets the analysis reduction use subgroupAdd instead of a
  // shared-memory tree (2 barriers per l-pair instead of 1 + log2(wgAnalys)).
  // Optional: ShtPlan falls back to the tree when it is not available.
  const features: GPUFeatureName[] = [];
  if (adapter.features.has('subgroups')) features.push('subgroups');
  // timestamp-query is only used by the profiling scripts, but it has to be
  // requested at device creation, and asking costs nothing when unused.
  if (adapter.features.has('timestamp-query')) features.push('timestamp-query');
  return adapter.requestDevice({
    requiredFeatures: features,
    requiredLimits: { maxComputeWorkgroupStorageSize: wgStorage },
  });
}
