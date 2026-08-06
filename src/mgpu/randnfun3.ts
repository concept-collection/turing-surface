/**
 * `randnfun3` — a smooth random function in 3D, evaluated at the surface.
 *
 * chebfun's randnfun3 is a random trig series on a box: a few thousand
 * Fourier modes with independent normal coefficients, confined to a ball for
 * isotropy and normalized to unit variance. Restricting it to a surface is
 * just evaluating it at the surface's points, which is what a model's `init`
 * wants for a seeded initial condition (surfacefun seeds exactly this way).
 *
 * The work splits in two, and the split is forced rather than chosen:
 *
 *  - **Drawing the modes needs `randn`**, which the compiled WGSL dialect has
 *    no counterpart for, and `sqrt(nnz)` normalization, which is a reduction.
 *    Both are a few lines of MATLAB, so the draw lives in
 *    `tools/randnfun3.m` and runs in numbl's interpreter — a few thousand
 *    numbers, ~5 ms.
 *  - **Evaluating is npts x nmodes**, ~6e7 terms at the default lambda. That
 *    is the whole cost, and it is what this file's kernel does on the GPU.
 *
 * So the .m calls `f = randnfun3(lambda, gx, gy, gz)` — chebfun's signature,
 * lambda in and values out — and the coefficient table is filled in behind it
 * by the host, the way `synth` hides its Legendre matrices. lambda is not
 * decorative: the plan records which parameter the .m passed, and the host
 * draws the table from *that* parameter's value (src/mgpu/plan.ts,
 * `randnfun3Lambda`), so changing it in the .m changes the field.
 */
import { executeCode } from 'numbl-src/numbl-core/executeCode.ts';
import { isRuntimeTensor } from 'numbl-src/numbl-core/runtime/types.ts';
import { toolFiles } from '../tools.ts';

/**
 * Dispatches the mode sum is split across.
 *
 * lambda is an absolute length and the mode count goes as its inverse cube,
 * so halving lambda costs eight times the work — there is no natural ceiling
 * to put on that, and nothing in the method breaks as it grows. It just gets
 * slower, which is the caller's business. What is *not* the caller's business
 * is a browser's GPU-process watchdog, which kills the device outright when a
 * single dispatch runs too long; a fine wavelength would otherwise turn "this
 * takes a while" into "device lost".
 *
 * So the sum is split into a fixed number of dispatches, each covering its own
 * slice of the table and accumulating into the same output. The count is fixed
 * at plan time (the op sequence has no runtime branching) and the slice bounds
 * come from the table's header, so one plan serves any wavelength. Slices that
 * fall past the end of a small table exit immediately, which is why a coarse
 * wavelength pays nothing for the split.
 */
const CHUNKS = 16;

/** Floats the table needs for `nmodes` modes. */
export const modeTableLength = (nmodes: number): number =>
  HEADER + STRIDE * nmodes;

/** Modes a table holds, from its header. */
export const modeCount = (table: Float32Array): number => table[0];

/**
 * Largest table this will try to build, in f32. Not a policy about how fine a
 * wavelength is sensible — that is the caller's call, and a fine one is
 * merely slow — but the point past which the draw would fail anyway: the
 * host-side Float32Array alone would be 8 GB. The device's own
 * storage-buffer limit is checked separately, when the buffer is allocated.
 */
const MAX_TABLE_FLOATS = 2 ** 31;

/** What the table starts at, before any seed has been drawn. Big enough for
 *  the default wavelength on the shipped surfaces, so the common case never
 *  reallocates. */
export const INITIAL_MODES = 4096;

/** Wavelength of the seeded field when the app names none. Fine enough to
 *  give a Turing pattern plenty to grow from, coarse enough that the draw is
 *  ~1,400 modes rather than the ~11,500 of the slider's finest setting. */
export const DEFAULT_LAMBDA = 0.5;

/** Floats before the first mode: `[nmodes, 0, 0, 0]`. The count travels in
 *  the buffer rather than a second binding, so the kernel needs one storage
 *  buffer and the host one write. */
const HEADER = 4;
/** Floats per mode: kx, ky, kz, real, imag. */
const STRIDE = 5;

/** The name the coefficient buffer takes in the plan's HostBuffers. */
export const MODE_BUFFER = 'randnfun3_modes';

/** How many dispatches `randnfun3WGSL` must be planned as. */
export const randnfun3Chunks = CHUNKS;

/**
 * One thread per surface point, summing this chunk's slice of the modes.
 *
 * The inner loop is a dot product, a cos, a sin and two multiply-adds, over a
 * table small enough (~1,400 modes at the default lambda) to sit in cache for
 * every thread. Chunk 0 initializes the output and the rest accumulate onto
 * it; dispatches within one compute pass are ordered, so the reads see the
 * previous chunk's writes. Nothing here is per-step work: `init` runs once a
 * seed.
 */
export function randnfun3WGSL(npts: number, chunk: number): string {
  return `
@group(0) @binding(0) var<storage, read_write> outf: array<f32>;
@group(0) @binding(1) var<storage, read> px: array<f32>;
@group(0) @binding(2) var<storage, read> py: array<f32>;
@group(0) @binding(3) var<storage, read> pz: array<f32>;
// [nmodes, _, _, _], then kx, ky, kz, re, im per mode.
@group(0) @binding(4) var<storage, read> modes: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${npts}u) { return; }
  let n = u32(modes[0]);
  // This chunk's slice. Ceiling division, so the last slices are the short
  // ones and an empty slice costs a single comparison.
  let per = (n + ${CHUNKS}u - 1u) / ${CHUNKS}u;
  let lo = min(${chunk}u * per, n);
  let hi = min(lo + per, n);
  var acc = 0.0;
  if (lo < hi) {
    let x = px[i];
    let y = py[i];
    let z = pz[i];
    for (var m = lo; m < hi; m = m + 1u) {
      let b = ${HEADER}u + m * ${STRIDE}u;
      let t = modes[b] * x + modes[b + 1u] * y + modes[b + 2u] * z;
      acc = acc + modes[b + 3u] * cos(t) - modes[b + 4u] * sin(t);
    }
  }
${chunk === 0 ? '  outf[i] = acc;' : '  outf[i] = outf[i] + acc;'}
}
`;
}

/**
 * Modes a wavelength will draw on a box, without drawing them: chebfun's
 * cube size, times the fraction its isotropy ball keeps (pi/6 of a cube,
 * approached from below at small m). Used to price a wavelength up front.
 */
function plannedModes(lambda: number, box: BoundingBox): number {
  const side = (w: number): number => 2 * Math.round((1.2 * w) / lambda + 2) + 1;
  const cube =
    side(box.x1 - box.x0) * side(box.y1 - box.y0) * side(box.z1 - box.z0);
  return Math.ceil((Math.PI / 6) * cube);
}

/** The box a random field is drawn over: the surface's own bounding box. */
export interface BoundingBox {
  x0: number; x1: number;
  y0: number; y1: number;
  z0: number; z1: number;
}

/** The bounding box of a surface, as `Geometry` holds its coordinates. */
export function boundingBox(
  x: Float32Array,
  y: Float32Array,
  z: Float32Array,
): BoundingBox {
  const box = {
    x0: Infinity, x1: -Infinity,
    y0: Infinity, y1: -Infinity,
    z0: Infinity, z1: -Infinity,
  };
  for (let i = 0; i < x.length; i++) {
    if (x[i] < box.x0) box.x0 = x[i];
    if (x[i] > box.x1) box.x1 = x[i];
    if (y[i] < box.y0) box.y0 = y[i];
    if (y[i] > box.y1) box.y1 = y[i];
    if (z[i] < box.z0) box.z0 = z[i];
    if (z[i] > box.z1) box.z1 = z[i];
  }
  return box;
}

/**
 * Draw a field's modes and pack them for the GPU: `tools/randnfun3.m` run
 * through the interpreter, seeded, then interleaved into the buffer layout
 * above. Column-major out of MATLAB, interleaved on the way in.
 */
export function drawModes(
  lambda: number,
  box: BoundingBox,
  seed: number,
  /** Points the field will be summed at, for the cost budget. */
  npts: number,
): Float32Array {
  if (!(lambda > 0) || !Number.isFinite(lambda)) {
    throw new Error(`randnfun3: lambda must be a positive number, got ${lambda}`);
  }
  // The mode count follows from lambda and the box alone, so a table that
  // cannot be built is refused before anything is drawn. The only ceiling is
  // what fits: how slow a fine wavelength is, is the caller's to decide.
  const planned = plannedModes(lambda, box);
  if (modeTableLength(planned) > MAX_TABLE_FLOATS) {
    throw new Error(
      `randnfun3: lambda ${lambda} needs about ` +
        `${planned.toLocaleString()} Fourier modes on this surface, a ` +
        `${((4 * modeTableLength(planned)) / 1e9).toFixed(1)} GB table. ` +
        `lambda is an absolute length, so a larger surface needs more modes ` +
        `for the same value, and halving it costs eight times as many.`,
    );
  }
  const result = executeCode(
    'rng(seed); [k, c] = randnfun3(lambda, [x0 x1 y0 y1 z0 z1]);',
    {
      initialVariableValues: { lambda, seed, ...box },
      displayResults: false,
      implicitCwdPath: null,
    },
    toolFiles,
    'randnfun3-driver.m',
  );
  const k = result.variableValues['k'];
  const c = result.variableValues['c'];
  if (!k || !c || !isRuntimeTensor(k) || !isRuntimeTensor(c)) {
    throw new Error("randnfun3: tools/randnfun3.m did not return [k, c] arrays");
  }
  const nmodes = k.shape[0];
  const out = new Float32Array(modeTableLength(nmodes));
  out[0] = nmodes;
  for (let i = 0; i < nmodes; i++) {
    const b = HEADER + STRIDE * i;
    out[b] = k.data[i];                  // kx
    out[b + 1] = k.data[nmodes + i];     // ky
    out[b + 2] = k.data[2 * nmodes + i]; // kz
    out[b + 3] = c.data[i];              // real
    out[b + 4] = c.data[nmodes + i];     // imag
  }
  return out;
}

/**
 * `drawModes` on a worker thread, so a fine wavelength does not freeze the
 * page (src/mgpu/randnfun3.worker.ts).
 *
 * Falls back to drawing in place where there is no `Worker` — the node test
 * runner and the desktop benchmark, neither of which has an event loop it
 * would matter to. Failures surface as a rejection either way, so a caller
 * never has to know which path ran.
 */
export function drawModesAsync(
  lambda: number,
  box: BoundingBox,
  seed: number,
  npts: number,
): Promise<Float32Array> {
  if (typeof Worker === 'undefined') {
    try {
      return Promise.resolve(drawModes(lambda, box, seed, npts));
    } catch (e) {
      return Promise.reject(e instanceof Error ? e : new Error(String(e)));
    }
  }
  const w = drawWorker();
  const id = nextDrawId++;
  return new Promise((resolve, reject) => {
    pendingDraws.set(id, { resolve, reject });
    w.postMessage({ id, lambda, box, seed, npts });
  });
}

let worker: Worker | null = null;
let nextDrawId = 1;
const pendingDraws = new Map<
  number,
  { resolve: (t: Float32Array) => void; reject: (e: Error) => void }
>();

/** The draw worker, started on first use and kept for the session — starting
 *  one re-parses numbl, which costs more than a coarse draw does. */
function drawWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL('./randnfun3.worker.ts', import.meta.url), {
    type: 'module',
  });
  worker.onmessage = (e: MessageEvent<{ id: number; table?: Float32Array; error?: string }>): void => {
    const waiting = pendingDraws.get(e.data.id);
    if (!waiting) return;
    pendingDraws.delete(e.data.id);
    if (e.data.error !== undefined) waiting.reject(new Error(e.data.error));
    else waiting.resolve(e.data.table!);
  };
  worker.onerror = (e: ErrorEvent): void => {
    // A worker that died takes every outstanding draw with it.
    for (const [, waiting] of pendingDraws) {
      waiting.reject(new Error(`randnfun3 draw worker failed: ${e.message}`));
    }
    pendingDraws.clear();
    worker?.terminate();
    worker = null;
  };
  return worker;
}
