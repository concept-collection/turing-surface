/**
 * The exact solver: (I - dtD*lap_g) X = B by dense LU on the GPU.
 *
 * Every other solver applies the operator implicitly, through lib/dlap.m,
 * and iterates. This one writes the operator down. The matvec every solver
 * shares is
 *
 *   A x = M.*x - dtD*dlap(x),   M = 1 + dtD*lam./jhat,
 *
 * and since dlap(x) = (analys(lap_g x) + (lam./jhat).*x).*filt, the jhat
 * term cancels on the band and A is affine in dtD:
 *
 *   A = I - dtD*K,   K = the matrix of x -> dlap(x) - (lam./jhat).*x.*filt,
 *
 * with K zero on the top (filtered) degrees, where A is just the diagonal M.
 * K depends on the surface and the band only — not on dt, D or jhat — so it
 * is assembled once per surface: 2*nlm unit vectors pushed through the very
 * same compiled dlap the iterative solvers run (an `opcol` program the host
 * compiles against the same lib/dlap.m, editor copy included), each result
 * copied into its column. That is the expensive part, 6 transforms per
 * column, and it is what makes this a small-lmax solver: the matrix is
 * (2*nlm)^2 floats, 0.3 MB at lmax 15, 4.5 MB at lmax 31, 69 MB at lmax 63.
 *
 * Each solve call site (one per species) then owns a factorization of its
 * own A: formed from K, lam, filt and the current (dtD, jhat) by one kernel,
 * factored in place by unblocked right-looking LU with partial pivoting —
 * per column, one single-workgroup dispatch finds the pivot, swaps the rows
 * and scales the column, and one 2-D dispatch applies the rank-1 update to
 * the trailing block — and finished by a kernel that composes the pivot
 * sequence into a permutation. All of it is command recording with no
 * readback, so it is encoded ahead of the step in the same submission the
 * moment the host sees dtD or jhat change (a slider move refactors; at
 * lmax 63 that is ~1 s of GPU time per change, at lmax 31 milliseconds).
 *
 * The per-step solve is one dispatch of one workgroup: gather P*B, forward
 * substitution with the unit-lower factor, back substitution with the
 * upper one, blocked so that the sequential part runs on a diagonal block
 * in workgroup memory and global memory is touched once per block of
 * columns. Sub-millisecond at small lmax; the whole factor is read twice
 * per species per step, which is what bounds it at large lmax.
 *
 * Everything is fp32, like the rest of the pipeline. The backward error of
 * a pivoted fp32 LU is a small multiple of fp32 epsilon, so the answer is
 * the discrete operator's solution to fp32 round-off — the same precision
 * the iterative solvers' matvec has — which is what makes this the natural
 * reference for how far they have converged.
 */
import type { ModelPlan } from './plan.ts';

/** Threads in the single-workgroup kernels (pivot search, triangular solve). */
const WG = 256;
/** Tile edge of the 2-D dispatches (matrix formation, rank-1 update). */
const TILE = 16;
/** Columns per block of the triangular solves: the bs x bs diagonal block
 *  lives in workgroup memory, 16 KB at 64, so 64 needs more than WebGPU's
 *  16 KB default (the device asks the adapter for up to 32 KB) and 32 is
 *  the fallback. */
const blockSize = (device: GPUDevice): number =>
  device.limits.maxComputeWorkgroupStorageSize >= 4 * (64 * 64 + 64) + 1024 ? 64 : 32;
/** Columns assembled per submission while building K, so the queue is handed
 *  back between chunks and a long build never stalls the compositor. */
const BUILD_CHUNK = 64;
/** WebGPU's default minimum alignment of a dynamic uniform offset. */
const UNIFORM_STRIDE = 256;

/** Bytes of the dense n x n matrix the exact solver keeps per copy. */
export const exactMatrixBytes = (n: number): number => 4 * n * n;

const formWGSL = (n: number): string => `
@group(0) @binding(0) var<storage, read_write> lu: array<f32>;
@group(0) @binding(1) var<storage, read> kmat: array<f32>;
@group(0) @binding(2) var<storage, read> lam: array<f32>;
@group(0) @binding(3) var<storage, read> filt: array<f32>;
// x: dtD, y: jhat
@group(0) @binding(4) var<uniform> sc: vec4<f32>;

// A = I - dtD*K on the band; on the filtered top degrees, where K is zero,
// the diagonal is the preconditioner's M = 1 + dtD*lam/jhat, exactly what
// the iterative solvers' matvec does there. Column-major, like the plan's
// 2 x nlm layout: element (i, j) sits at i + j*n.
@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  if (i >= ${n}u || j >= ${n}u) { return; }
  let idx = i + j * ${n}u;
  var a = -sc.x * kmat[idx];
  if (i == j) {
    a = a + 1.0 + sc.x * (1.0 - filt[i]) * lam[i] / sc.y;
  }
  lu[idx] = a;
}
`;

const pivotWGSL = (n: number): string => `
@group(0) @binding(0) var<storage, read_write> lu: array<f32>;
@group(0) @binding(1) var<storage, read_write> piv: array<u32>;
@group(0) @binding(2) var<uniform> col: vec4<u32>;

var<workgroup> mv: array<f32, ${WG}>;
var<workgroup> mi: array<u32, ${WG}>;

// Column k of the factorization: the pivot row is the first maximum of
// |A(i, k)| over i >= k (a strided scan per thread, then a tree that
// prefers the larger value and, on a tie, the lower index — LAPACK's
// idamax order, so the pivot sequence is deterministic). Rows k and p are
// then swapped across every column, L's included, and the column below the
// pivot is scaled by it. A zero pivot leaves the column unscaled, as
// dgetrf does; the singular system shows up as non-finite state.
@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let k = col.x;
  let t = lid.x;
  var best = -1.0;
  var bi = k;
  var i = k + t;
  loop {
    if (i >= ${n}u) { break; }
    let v = abs(lu[i + k * ${n}u]);
    if (v > best) { best = v; bi = i; }
    i = i + ${WG}u;
  }
  mv[t] = best;
  mi[t] = bi;
  workgroupBarrier();
  var stride = ${WG / 2}u;
  loop {
    if (stride == 0u) { break; }
    if (t < stride) {
      let ov = mv[t + stride];
      let oi = mi[t + stride];
      if (ov > mv[t] || (ov == mv[t] && oi < mi[t])) {
        mv[t] = ov;
        mi[t] = oi;
      }
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let p = mi[0];
  if (t == 0u) { piv[k] = p; }
  if (p != k) {
    var j = t;
    loop {
      if (j >= ${n}u) { break; }
      let a = lu[k + j * ${n}u];
      lu[k + j * ${n}u] = lu[p + j * ${n}u];
      lu[p + j * ${n}u] = a;
      j = j + ${WG}u;
    }
  }
  storageBarrier();
  workgroupBarrier();
  let d = lu[k + k * ${n}u];
  if (d != 0.0) {
    var r = k + 1u + t;
    loop {
      if (r >= ${n}u) { break; }
      lu[r + k * ${n}u] = lu[r + k * ${n}u] / d;
      r = r + ${WG}u;
    }
  }
}
`;

const updateWGSL = (n: number): string => `
@group(0) @binding(0) var<storage, read_write> lu: array<f32>;
@group(0) @binding(2) var<uniform> col: vec4<u32>;

// The rank-1 update of the trailing block after column k: A(i, j) -=
// L(i, k) * U(k, j) for i, j > k. Reads only row k and column k, which this
// dispatch never writes. Threads run down i, so each warp reads and writes
// a contiguous run of a column.
@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = col.x;
  let i = k + 1u + gid.x;
  let j = k + 1u + gid.y;
  if (i >= ${n}u || j >= ${n}u) { return; }
  lu[i + j * ${n}u] = lu[i + j * ${n}u] - lu[i + k * ${n}u] * lu[k + j * ${n}u];
}
`;

const permWGSL = (n: number): string => `
@group(0) @binding(0) var<storage, read> piv: array<u32>;
@group(0) @binding(1) var<storage, read_write> perm: array<u32>;

// The pivot sequence, applied to the identity in order, gives the row
// permutation as one gather: (P b)[i] = b[perm[i]]. Sequential by nature,
// once per factorization, one thread.
@compute @workgroup_size(1)
fn main() {
  for (var i = 0u; i < ${n}u; i++) { perm[i] = i; }
  for (var k = 0u; k < ${n}u; k++) {
    let p = piv[k];
    if (p != k) {
      let t = perm[k];
      perm[k] = perm[p];
      perm[p] = t;
    }
  }
}
`;

/**
 * The triangular solves, one dispatch per block of `bs` columns (forward)
 * and again per block on the way back. Substitution is sequential in the
 * column index by nature, so the block's own bs x bs solve is done by one
 * workgroup on a diagonal block held in workgroup memory; what is parallel
 * — the previous block's rank-bs update of every row still open, and the
 * initial gather of P*b — is spread over all workgroups of the same
 * dispatch, workgroup 0 taking the block's own rows and then continuing
 * into the block solve, so nothing waits on a separate update dispatch.
 * Each dispatch reads the columns of the previous block once, coalesced
 * down each column, with a row's loads issued together.
 */
const solveWGSL = (n: number, bs: number, dir: 'fwd' | 'bwd'): string => `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> lu: array<f32>;
@group(0) @binding(3) var<storage, read> perm: array<u32>;
// x: the block's first column k0
@group(0) @binding(4) var<uniform> col: vec4<u32>;

var<workgroup> blk: array<f32, ${bs * bs}>;
var<workgroup> xs: array<f32, ${bs}>;

@compute @workgroup_size(${WG})
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let t = lid.x;
  let g = wid.x;
  let k0 = col.x;
  let nblk = min(${bs}u, ${n}u - k0);
  ${
    dir === 'fwd'
      ? `// Rows: workgroup 0 owns the block's own, the rest the rows below it.
  var r = k0 + t;
  var own = t < nblk;
  if (g != 0u) {
    r = k0 + nblk + (g - 1u) * ${WG}u + t;
    own = r < ${n}u;
  }
  if (own) {
    var v: f32;
    if (k0 == 0u) {
      v = b[perm[r]];
    } else {
      // The previous block is full-width: only the last block can be short.
      let p0 = k0 - ${bs}u;
      v = x[r];
      for (var j = 0u; j < ${bs}u; j++) {
        v = v - lu[r + (p0 + j) * ${n}u] * x[p0 + j];
      }
    }
    x[r] = v;
  }`
      : `// Rows: workgroup 0 owns the block's own, the rest the rows above it.
  var r = k0 + t;
  var own = t < nblk;
  if (g != 0u) {
    r = (g - 1u) * ${WG}u + t;
    own = r < k0;
  }
  let p0 = k0 + ${bs}u;
  if (own && p0 < ${n}u) {
    let pblk = min(${bs}u, ${n}u - p0);
    var v = x[r];
    for (var j = 0u; j < pblk; j++) {
      v = v - lu[r + (p0 + j) * ${n}u] * x[p0 + j];
    }
    x[r] = v;
  }`
  }
  if (g != 0u) { return; }
  storageBarrier();
  workgroupBarrier();

  // Workgroup 0: the block's own solve, on its diagonal block in workgroup
  // memory (element (i, j) at i*bs + j).
  var e = t;
  loop {
    if (e >= ${bs * bs}u) { break; }
    let bi = e / ${bs}u;
    let bj = e % ${bs}u;
    if (bi < nblk && bj < nblk) { blk[e] = lu[(k0 + bi) + (k0 + bj) * ${n}u]; }
    e = e + ${WG}u;
  }
  if (t < nblk) { xs[t] = x[k0 + t]; }
  workgroupBarrier();
  ${
    dir === 'fwd'
      ? `// Unit lower: no divide.
  for (var j = 0u; j < nblk; j++) {
    let v = xs[j];
    let i = j + 1u + t;
    if (i < nblk) { xs[i] = xs[i] - blk[i * ${bs}u + j] * v; }
    workgroupBarrier();
  }`
      : `// Upper: divide by the diagonal, then eliminate above.
  for (var jj = 0u; jj < nblk; jj++) {
    let j = nblk - 1u - jj;
    let v = xs[j] / blk[j * ${bs}u + j];
    workgroupBarrier();
    if (t == 0u) { xs[j] = v; }
    if (t < j) { xs[t] = xs[t] - blk[t * ${bs}u + j] * v; }
    workgroupBarrier();
  }`
  }
  if (t < nblk) { x[k0 + t] = xs[t]; }
}
`;

interface Kernels {
  form: GPUComputePipeline;
  formLayout: GPUBindGroupLayout;
  pivot: GPUComputePipeline;
  update: GPUComputePipeline;
  luLayout: GPUBindGroupLayout;
  perm: GPUComputePipeline;
  permLayout: GPUBindGroupLayout;
  fwd: GPUComputePipeline;
  bwd: GPUComputePipeline;
  solveLayout: GPUBindGroupLayout;
  /** Columns per triangular-solve block. */
  bs: number;
}

async function makePipeline(
  device: GPUDevice,
  code: string,
  label: string,
  layout: GPUBindGroupLayout,
): Promise<GPUComputePipeline> {
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) {
    throw new Error(
      `WGSL compile error in ${label}:\n` +
        errors.map((e) => `  ${e.lineNum}:${e.linePos} ${e.message}`).join('\n'),
    );
  }
  const pipeline = await device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint: 'main' },
    label,
  });
  const err = await device.popErrorScope();
  if (err) throw new Error(`pipeline ${label}: ${err.message}`);
  return pipeline;
}

async function makeKernels(device: GPUDevice, n: number): Promise<Kernels> {
  const entry = (
    binding: number,
    type: GPUBufferBindingType,
    hasDynamicOffset = false,
  ): GPUBindGroupLayoutEntry => ({
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type, hasDynamicOffset },
  });
  const formLayout = device.createBindGroupLayout({
    label: 'exact-form',
    entries: [
      entry(0, 'storage'),
      entry(1, 'read-only-storage'),
      entry(2, 'read-only-storage'),
      entry(3, 'read-only-storage'),
      entry(4, 'uniform'),
    ],
  });
  const luLayout = device.createBindGroupLayout({
    label: 'exact-lu',
    entries: [entry(0, 'storage'), entry(1, 'storage'), entry(2, 'uniform', true)],
  });
  const permLayout = device.createBindGroupLayout({
    label: 'exact-perm',
    entries: [entry(0, 'read-only-storage'), entry(1, 'storage')],
  });
  const solveLayout = device.createBindGroupLayout({
    label: 'exact-solve',
    entries: [
      entry(0, 'storage'),
      entry(1, 'read-only-storage'),
      entry(2, 'read-only-storage'),
      entry(3, 'read-only-storage'),
      entry(4, 'uniform', true),
    ],
  });
  const bs = blockSize(device);
  const [form, pivot, update, perm, fwd, bwd] = await Promise.all([
    makePipeline(device, formWGSL(n), `exact-form-${n}`, formLayout),
    makePipeline(device, pivotWGSL(n), `exact-pivot-${n}`, luLayout),
    makePipeline(device, updateWGSL(n), `exact-update-${n}`, luLayout),
    makePipeline(device, permWGSL(n), `exact-perm-${n}`, permLayout),
    makePipeline(device, solveWGSL(n, bs, 'fwd'), `exact-fwd-${n}`, solveLayout),
    makePipeline(device, solveWGSL(n, bs, 'bwd'), `exact-bwd-${n}`, solveLayout),
  ]);
  return { form, formLayout, pivot, update, luLayout, perm, permLayout, fwd, bwd, solveLayout, bs };
}

export interface ExactOperatorOptions {
  device: GPUDevice;
  /** The compiled column program: `O = dlap(ecol, ...) - lam.*ecol.*filt`. */
  plan: ModelPlan;
  /** Its input buffer (the host binding `ecol`) and its output `O`. */
  unit: GPUBuffer;
  column: GPUBuffer;
  /** The eigenvalue and top-degree filter buffers the models take. */
  lam: GPUBuffer;
  filt: GPUBuffer;
  /** Matrix dimension, 2*nlm. */
  n: number;
}

/**
 * The geometric operator's matrix K, shared by every solve call site of a
 * model, and the factory for their factorizations.
 */
export class ExactOperator {
  readonly n: number;
  /** Bumped when K is rebuilt; a site whose factorization predates it refactors. */
  version = 0;

  #device: GPUDevice;
  #plan: ModelPlan;
  #unit: GPUBuffer;
  #column: GPUBuffer;
  #lam: GPUBuffer;
  #filt: GPUBuffer;
  #K: GPUBuffer;
  /** A single 1.0f, copied into the unit vector's slot per column. */
  #one: GPUBuffer;
  #kernels: Kernels;
  #chain: Promise<void> = Promise.resolve();
  #destroyed = false;

  private constructor(opts: ExactOperatorOptions, kernels: Kernels) {
    const { device, n } = opts;
    this.n = n;
    this.#device = device;
    this.#plan = opts.plan;
    this.#unit = opts.unit;
    this.#column = opts.column;
    this.#lam = opts.lam;
    this.#filt = opts.filt;
    this.#kernels = kernels;
    this.#K = device.createBuffer({
      label: 'exact-K',
      size: exactMatrixBytes(n),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.#one = device.createBuffer({
      label: 'exact-one',
      size: 4,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.#one, 0, new Float32Array([1]));
  }

  static async create(opts: ExactOperatorOptions): Promise<ExactOperator> {
    const kernels = await makeKernels(opts.device, opts.n);
    return new ExactOperator(opts, kernels);
  }

  /**
   * Assemble K from the current surface: one column per unit vector through
   * the compiled operator. Serialized, so a second call during a build waits
   * for the first. The columns land in a staging matrix and are copied over
   * K in one command at the end — a step submitted meanwhile factors from the
   * complete old K rather than a half-written new one, and refactors on its
   * next step because `version` moves only after the copy is queued.
   */
  build(): Promise<void> {
    this.#chain = this.#chain.then(() => this.#build());
    return this.#chain;
  }

  async #build(): Promise<void> {
    if (this.#destroyed) return;
    const device = this.#device;
    const n = this.n;
    const staging = device.createBuffer({
      label: 'exact-K-staging',
      size: exactMatrixBytes(n),
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    for (let j0 = 0; j0 < n; j0 += BUILD_CHUNK) {
      if (this.#destroyed) {
        staging.destroy();
        return;
      }
      const enc = device.createCommandEncoder({ label: 'exact-build' });
      const j1 = Math.min(n, j0 + BUILD_CHUNK);
      for (let j = j0; j < j1; j++) {
        enc.clearBuffer(this.#unit);
        enc.copyBufferToBuffer(this.#one, 0, this.#unit, 4 * j, 4);
        this.#plan.encodeSteps(enc, 1);
        enc.copyBufferToBuffer(this.#column, 0, staging, 4 * n * j, 4 * n);
      }
      device.queue.submit([enc.finish()]);
      if (j1 < n) {
        // Let the queue drain and the event loop turn between chunks.
        await device.queue.onSubmittedWorkDone();
        await new Promise((r) => setTimeout(r, 0));
      }
    }
    if (this.#destroyed) {
      staging.destroy();
      return;
    }
    const enc = device.createCommandEncoder({ label: 'exact-build-commit' });
    enc.copyBufferToBuffer(staging, 0, this.#K, 0, exactMatrixBytes(n));
    device.queue.submit([enc.finish()]);
    this.version++;
    await device.queue.onSubmittedWorkDone();
    staging.destroy();
  }

  /** One solve call site's factorization and solve kernel, bound to its
   *  right-hand side and result buffers. */
  createSite(B: GPUBuffer, X: GPUBuffer, label: string): ExactSite {
    return new ExactSite(this.#device, this, this.#kernels, {
      B, X, K: this.#K, lam: this.#lam, filt: this.#filt, label,
    });
  }

  /** K back on the CPU, column-major — for the tests. */
  async readMatrix(): Promise<Float32Array> {
    const bytes = exactMatrixBytes(this.n);
    const device = this.#device;
    const readback = device.createBuffer({
      label: 'exact-K-readback',
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(this.#K, 0, readback, 0, bytes);
    device.queue.submit([enc.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    readback.destroy();
    return out;
  }

  destroy(): void {
    this.#destroyed = true;
    this.#plan.destroy();
    this.#K.destroy();
    this.#one.destroy();
  }
}

/**
 * One `X = lusolve(B, dtD, ...)` call site: the LU of its own A, refactored
 * when (dtD, jhat) or the operator changes, and the kernel that applies it.
 */
export class ExactSite {
  #device: GPUDevice;
  #op: ExactOperator;
  #kernels: Kernels;
  #n: number;
  #lu: GPUBuffer;
  #piv: GPUBuffer;
  #perm: GPUBuffer;
  /** `n` copies of the column index, one per UNIFORM_STRIDE, read through a
   *  dynamic offset so every column's dispatch shares one bind group. */
  #cols: GPUBuffer;
  /** (dtD, jhat) for the formation kernel. */
  #scal: GPUBuffer;
  #formGroup: GPUBindGroup;
  #luGroup: GPUBindGroup;
  #permGroup: GPUBindGroup;
  #solveGroup: GPUBindGroup;
  #factored: { dtD: number; jhat: number; version: number } | null = null;
  readonly label: string;

  constructor(
    device: GPUDevice,
    op: ExactOperator,
    kernels: Kernels,
    bufs: { B: GPUBuffer; X: GPUBuffer; K: GPUBuffer; lam: GPUBuffer; filt: GPUBuffer; label: string },
  ) {
    const n = op.n;
    this.#device = device;
    this.#op = op;
    this.#kernels = kernels;
    this.#n = n;
    this.label = bufs.label;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.#lu = device.createBuffer({ label: 'exact-LU', size: exactMatrixBytes(n), usage: storage });
    this.#piv = device.createBuffer({ label: 'exact-piv', size: 4 * n, usage: storage });
    this.#perm = device.createBuffer({ label: 'exact-perm', size: 4 * n, usage: storage });
    this.#cols = device.createBuffer({
      label: 'exact-cols',
      size: UNIFORM_STRIDE * n,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const cols = new Uint32Array((UNIFORM_STRIDE / 4) * n);
    for (let k = 0; k < n; k++) cols[(UNIFORM_STRIDE / 4) * k] = k;
    device.queue.writeBuffer(this.#cols, 0, cols);
    this.#scal = device.createBuffer({
      label: 'exact-scalars',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bind = (layout: GPUBindGroupLayout, entries: GPUBufferBinding[]): GPUBindGroup =>
      device.createBindGroup({
        layout,
        entries: entries.map((resource, binding) => ({ binding, resource })),
      });
    const b = (buffer: GPUBuffer): GPUBufferBinding => ({ buffer });
    this.#formGroup = bind(kernels.formLayout, [
      b(this.#lu), b(bufs.K), b(bufs.lam), b(bufs.filt), b(this.#scal),
    ]);
    this.#luGroup = bind(kernels.luLayout, [
      b(this.#lu),
      b(this.#piv),
      { buffer: this.#cols, size: 16 },
    ]);
    this.#permGroup = bind(kernels.permLayout, [b(this.#piv), b(this.#perm)]);
    this.#solveGroup = bind(kernels.solveLayout, [
      b(bufs.X), b(bufs.B), b(this.#lu), b(this.#perm), { buffer: this.#cols, size: 16 },
    ]);
  }

  /** Whether the factorization on the GPU is the one for these scalars and
   *  the current operator. */
  isFactoredFor(dtD: number, jhat: number): boolean {
    const f = this.#factored;
    return !!f && f.dtD === dtD && f.jhat === jhat && f.version === this.#op.version;
  }

  /**
   * Record the formation and factorization of A for (dtD, jhat) unless the
   * factors already hold it. Must precede the step's solve in the same
   * submission (or an earlier one); `encoder` is outside any pass.
   */
  ensureFactored(encoder: GPUCommandEncoder, dtD: number, jhat: number): void {
    if (this.isFactoredFor(dtD, jhat)) return;
    const n = this.#n;
    const k = this.#kernels;
    this.#device.queue.writeBuffer(this.#scal, 0, new Float32Array([dtD, jhat, 0, 0]));
    const pass = encoder.beginComputePass({ label: 'exact-factor' });
    pass.setPipeline(k.form);
    pass.setBindGroup(0, this.#formGroup);
    const tiles = Math.ceil(n / TILE);
    pass.dispatchWorkgroups(tiles, tiles);
    for (let col = 0; col < n; col++) {
      pass.setPipeline(k.pivot);
      pass.setBindGroup(0, this.#luGroup, [col * UNIFORM_STRIDE]);
      pass.dispatchWorkgroups(1);
      const rest = n - col - 1;
      if (rest > 0) {
        pass.setPipeline(k.update);
        pass.setBindGroup(0, this.#luGroup, [col * UNIFORM_STRIDE]);
        const t = Math.ceil(rest / TILE);
        pass.dispatchWorkgroups(t, t);
      }
    }
    pass.setPipeline(k.perm);
    pass.setBindGroup(0, this.#permGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    this.#factored = { dtD, jhat, version: this.#op.version };
  }

  /** Dispatches one solve records: two sweeps of one per column block. */
  get solveDispatches(): number {
    return 2 * Math.ceil(this.#n / this.#kernels.bs);
  }

  /** The solve itself, inside the step's pass: the forward sweep over the
   *  column blocks, then the backward one (see solveWGSL). The dynamic
   *  offset selects the block's first column from the same slots the
   *  factorization's per-column dispatches use. */
  encodeSolve(pass: GPUComputePassEncoder): void {
    const n = this.#n;
    const { bs, fwd, bwd } = this.#kernels;
    const nb = Math.ceil(n / bs);
    pass.setPipeline(fwd);
    for (let kb = 0; kb < nb; kb++) {
      const k0 = kb * bs;
      const below = n - Math.min(n, k0 + bs);
      pass.setBindGroup(0, this.#solveGroup, [k0 * UNIFORM_STRIDE]);
      pass.dispatchWorkgroups(1 + Math.ceil(below / WG));
    }
    pass.setPipeline(bwd);
    for (let kb = nb - 1; kb >= 0; kb--) {
      const k0 = kb * bs;
      pass.setBindGroup(0, this.#solveGroup, [k0 * UNIFORM_STRIDE]);
      pass.dispatchWorkgroups(1 + Math.ceil(k0 / WG));
    }
  }

  destroy(): void {
    this.#lu.destroy();
    this.#piv.destroy();
    this.#perm.destroy();
    this.#cols.destroy();
    this.#scal.destroy();
  }
}
