/**
 * WGSL Fourier-stage kernels (the role cuFFT/VkFFT plays in SHTNS).
 *
 * Real fields, band-limited to |m| <= mmax < nphi/2:
 *  - synthesis: assemble a Hermitian spectrum from F_m (m >= 0) and do an
 *    inverse complex FFT along phi; take the real part.
 *  - analysis: forward complex FFT of the (real) row; keep m = 0..mmax.
 *
 * Two implementations, selected at plan creation:
 *  - 'fft': radix-2 Stockham in workgroup memory, one workgroup per
 *    latitude row.  Requires nphi a power of two and
 *    2 * 8 * nphi bytes <= maxComputeWorkgroupStorageSize.
 *  - 'dft': direct band-limited trigonometric summation, O(nphi * mmax)
 *    per row.  Works for any nphi; also useful as a cross-check.
 *
 * All trigonometric factors come from a host-precomputed (f64 -> f32)
 * table trig[k] = (cos, sin)(2*pi*k/nphi): device sin/cos is only
 * guaranteed to ~2^-11 absolute error under Vulkan, which would dominate
 * the fp32 transform error.
 */

export interface FourierParams {
  mmax: number;
  nlat: number;
  nphi: number;
  /** 2 or 4; radix-4 uses log4(n) barrier stages instead of log2(n). */
  radix?: number;
}

const TRIG_BINDING = /* wgsl */ `
@group(0) @binding(2) var<storage, read> trig: array<vec2f>;  // (cos,sin)(2*pi*k/NPHI), k < NPHI
`;

/**
 * @param n     transform length (bufA/bufB are this long)
 * @param scale trig-table stride multiplier: the table holds
 *              (cos,sin)(2*pi*k/NPHI), so an n-point transform needs NPHI/n.
 */
function stockham(n: number, threads: number, sign: number, scale = 1): string {
  const nphi = n;
  const log2n = Math.log2(n);
  if (!Number.isInteger(log2n)) throw new Error('fft requires power-of-two nphi');
  // twiddle for pass with half-block ns: w = e^{sign*i*pi*j/ns} = T[j * (N/(2*ns))]^sign
  return /* wgsl */ `
var<workgroup> bufA: array<vec2f, ${nphi}>;
var<workgroup> bufB: array<vec2f, ${nphi}>;

fn cmul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn ld(sel: u32, i: u32) -> vec2f {
  if (sel == 0u) { return bufA[i]; }
  return bufB[i];
}
fn st_(sel: u32, i: u32, v: vec2f) {
  if (sel == 0u) { bufA[i] = v; } else { bufB[i] = v; }
}

// radix-2 Stockham, natural order in and out; data starts in bufA (sel 0)
// and ends in sel = LOG2N % 2.  Unnormalized: X_k = sum_j x_j e^{s*2*pi*i*jk/N}.
fn fft_inplace(lid: u32) {
  for (var p = 0u; p < ${log2n}u; p++) {
    workgroupBarrier();
    let ns = 1u << p;
    let sel = p & 1u;
    let stride = ${(nphi / 2) * scale}u >> p;   // (NPHI/n) * n/(2*ns)
    for (var t = lid; t < ${nphi / 2}u; t += ${threads}u) {
      let j = t & (ns - 1u);
      let tw = trig[j * stride];
      let w = vec2f(tw.x, ${sign > 0 ? '' : '-'}tw.y);
      let u = ld(sel, t);
      let v = cmul(ld(sel, t + ${nphi / 2}u), w);
      let idst = 2u * (t - j) + j;
      st_(1u - sel, idst, u + v);
      st_(1u - sel, idst + ns, u - v);
    }
  }
  workgroupBarrier();
}
const FFT_OUT_SEL: u32 = ${log2n % 2}u;
`;
}

/**
 * Radix-4 Stockham. Same interface and conventions as stockham(), but log4(n)
 * stages instead of log2(n) -- each stage carries a workgroupBarrier, and
 * barriers are what these kernels are actually bound by. When log2(n) is odd a
 * single radix-2 stage runs first, so n = 128 costs 1 + 3 stages rather than 7.
 *
 * Butterfly, with w = e^{s 2 pi i / 4}:
 *   a = x0 + x2,  b = x0 - x2,  c = x1 + x3,  d = s i (x1 - x3)
 *   y = (a + c, b + d, a - c, b - d)
 */
function stockham4(n: number, threads: number, sign: number, scale = 1): string {
  const log2n = Math.log2(n);
  if (!Number.isInteger(log2n)) throw new Error('fft requires power-of-two nphi');
  const needR2 = log2n % 2 === 1;
  const stages4 = Math.floor(log2n / 2);
  const total = (needR2 ? 1 : 0) + stages4;
  const negY = sign > 0 ? '' : '-';
  return /* wgsl */ `
var<workgroup> bufA: array<vec2f, ${n}>;
var<workgroup> bufB: array<vec2f, ${n}>;

fn cmul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
fn ld(sel: u32, i: u32) -> vec2f {
  if (sel == 0u) { return bufA[i]; }
  return bufB[i];
}
fn st_(sel: u32, i: u32, v: vec2f) {
  if (sel == 0u) { bufA[i] = v; } else { bufB[i] = v; }
}
fn tw(i: u32) -> vec2f {
  let t = trig[i];
  return vec2f(t.x, ${negY}t.y);
}

fn fft_inplace(lid: u32) {
  var sel = 0u;
  var ns = 1u;
${
  needR2
    ? `  // leading radix-2 (ns = 1, so the twiddle is 1 and is skipped)
  workgroupBarrier();
  for (var t = lid; t < ${n / 2}u; t += ${threads}u) {
    let u = ld(sel, t);
    let v = ld(sel, t + ${n / 2}u);
    st_(1u - sel, 2u * t, u + v);
    st_(1u - sel, 2u * t + 1u, u - v);
  }
  sel = 1u - sel;
  ns = 2u;`
    : ''
}
  for (var p = 0u; p < ${stages4}u; p++) {
    workgroupBarrier();
    let s4 = ${(n * scale) / 4}u / ns;   // trig unit: NPHI / (4 * ns)
    for (var t = lid; t < ${n / 4}u; t += ${threads}u) {
      let j = t & (ns - 1u);
      let x0 = ld(sel, t);
      var x1 = ld(sel, t + ${n / 4}u);
      var x2 = ld(sel, t + ${n / 2}u);
      var x3 = ld(sel, t + ${(3 * n) / 4}u);
      if (ns > 1u) {
        x1 = cmul(x1, tw(j * s4));
        x2 = cmul(x2, tw(2u * j * s4));
        x3 = cmul(x3, tw(3u * j * s4));
      }
      let a = x0 + x2;
      let b = x0 - x2;
      let c = x1 + x3;
      let e = x1 - x3;
      let d = vec2f(${sign > 0 ? '-e.y, e.x' : 'e.y, -e.x'});   // s * i * e
      let idst = 4u * (t - j) + j;
      st_(1u - sel, idst, a + c);
      st_(1u - sel, idst + ns, b + d);
      st_(1u - sel, idst + 2u * ns, a - c);
      st_(1u - sel, idst + 3u * ns, b - d);
    }
    sel = 1u - sel;
    ns = ns * 4u;
  }
  workgroupBarrier();
}
const FFT_OUT_SEL: u32 = ${total % 2}u;
`;
}

/** Choose FFT workgroup size: enough threads for the butterflies, capped at 256. */
export function fftThreads(nphi: number): number {
  return Math.max(32, Math.min(256, nphi / 2));
}

export function fftSynthWGSL(p: FourierParams): string {
  const T = fftThreads(p.nphi);
  return /* wgsl */ `
const MMAX: u32 = ${p.mmax}u;
const NLAT: u32 = ${p.nlat}u;
const NPHI: u32 = ${p.nphi}u;
@group(0) @binding(0) var<storage, read> fm: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> spat: array<f32>;
${TRIG_BINDING}
${stockham(p.nphi, T, +1)}

@compute @workgroup_size(${T})
fn fft_synth(@builtin(local_invocation_id) lid3: vec3u,
             @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  let ilat = wid.x;
  // assemble Hermitian spectrum: X[0] = Re F_0, X[m] = F_m, X[N-m] = conj(F_m)
  for (var k = lid; k < NPHI; k += ${T}u) {
    var v = vec2f(0.0);
    if (k == 0u) {
      v = vec2f(fm[ilat].x, 0.0);
    } else if (k <= MMAX) {
      v = fm[k * NLAT + ilat];
    } else if (k >= NPHI - MMAX) {
      let c = fm[(NPHI - k) * NLAT + ilat];
      v = vec2f(c.x, -c.y);
    }
    bufA[k] = v;
  }
  fft_inplace(lid);
  for (var k = lid; k < NPHI; k += ${T}u) {
    spat[ilat * NPHI + k] = ld(FFT_OUT_SEL, k).x;
  }
}
`;
}

export function fftAnalysWGSL(p: FourierParams): string {
  const T = fftThreads(p.nphi);
  return /* wgsl */ `
const MMAX: u32 = ${p.mmax}u;
const NLAT: u32 = ${p.nlat}u;
const NPHI: u32 = ${p.nphi}u;
@group(0) @binding(0) var<storage, read> spat: array<f32>;
@group(0) @binding(1) var<storage, read_write> fm: array<vec2f>;
${TRIG_BINDING}
${stockham(p.nphi, T, -1)}

@compute @workgroup_size(${T})
fn fft_analys(@builtin(local_invocation_id) lid3: vec3u,
              @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  let ilat = wid.x;
  for (var k = lid; k < NPHI; k += ${T}u) {
    bufA[k] = vec2f(spat[ilat * NPHI + k], 0.0);
  }
  fft_inplace(lid);
  for (var m = lid; m <= MMAX; m += ${T}u) {
    fm[m * NLAT + ilat] = ld(FFT_OUT_SEL, m);
  }
}
`;
}

export function dftSynthWGSL(p: FourierParams): string {
  return /* wgsl */ `
const MMAX: u32 = ${p.mmax}u;
const NLAT: u32 = ${p.nlat}u;
const NPHI: u32 = ${p.nphi}u;
@group(0) @binding(0) var<storage, read> fm: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> spat: array<f32>;
${TRIG_BINDING}

@compute @workgroup_size(64)
fn dft_synth(@builtin(global_invocation_id) gid: vec3u) {
  let iphi = gid.x;
  let ilat = gid.y;
  if (iphi >= NPHI) { return; }
  var v: f32 = fm[ilat].x;   // m = 0: real part
  for (var m = 1u; m <= MMAX; m++) {
    let w = trig[(m * iphi) % NPHI];     // e^{+i m phi}
    let c = fm[m * NLAT + ilat];
    v += 2.0 * (c.x * w.x - c.y * w.y);
  }
  spat[ilat * NPHI + iphi] = v;
}
`;
}

export function dftAnalysWGSL(p: FourierParams): string {
  return /* wgsl */ `
const MMAX: u32 = ${p.mmax}u;
const NLAT: u32 = ${p.nlat}u;
const NPHI: u32 = ${p.nphi}u;
@group(0) @binding(0) var<storage, read> spat: array<f32>;
@group(0) @binding(1) var<storage, read_write> fm: array<vec2f>;
${TRIG_BINDING}

@compute @workgroup_size(64)
fn dft_analys(@builtin(global_invocation_id) gid: vec3u) {
  let m = gid.x;
  let ilat = gid.y;
  if (m > MMAX) { return; }
  var acc = vec2f(0.0);
  for (var j = 0u; j < NPHI; j++) {
    let w = trig[(m * j) % NPHI];        // conj => e^{-i m phi}
    let f = spat[ilat * NPHI + j];
    acc += f * vec2f(w.x, -w.y);
  }
  fm[m * NLAT + ilat] = acc;
}
`;
}

/**
 * Real-field Fourier stage: half the arithmetic and half the shared memory of
 * the complex path, which transforms N points to get a Hermitian result.
 *
 * A length-N real transform is an N/2-point complex FFT wrapped in a
 * recombination. Writing H = N/2 and taking the unnormalized conventions of the
 * complex kernels above (synthesis e^{+i}, analysis e^{-i}):
 *
 *   synthesis  Z[k] = (X[k] + conj(X[H-k])) + i e^{+2pi i k/N} (X[k] - conj(X[H-k]))
 *              z = FFT_H^{+}(Z),  then x[2m] = Re z[m], x[2m+1] = Im z[m]
 *   analysis   z[m] = x[2m] + i x[2m+1],  Z = FFT_H^{-}(z)
 *              Xe = (Z[k] + conj(Z[H-k]))/2,  Xo = -i (Z[k] - conj(Z[H-k]))/2
 *              X[k] = Xe + e^{-2pi i k/N} Xo
 *
 * The factors of 2 in the synthesis direction cancel against the 1/2 in Xe/Xo,
 * which is why none appear there. The complex kernels are kept: they are what a
 * complex-valued spatial field would use, and stockham() is shared by both.
 */
export function fftSynthRealWGSL(p: FourierParams): string {
  const H = p.nphi / 2;
  const T = fftThreads(H);
  return /* wgsl */ `
const MMAX: u32 = ${p.mmax}u;
const NLAT: u32 = ${p.nlat}u;
const NPHI: u32 = ${p.nphi}u;
const H: u32 = ${H}u;
@group(0) @binding(0) var<storage, read> fm: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> spat: array<f32>;
${TRIG_BINDING}
${(p.radix ?? 4) === 4 ? stockham4(H, T, +1, p.nphi / H) : stockham(H, T, +1, p.nphi / H)}

// X[k] of the Hermitian spectrum, for 0 <= k <= H. mmax < H, so the
// upper-conjugate branch of the complex kernel cannot be reached here.
fn spec(ilat: u32, k: u32) -> vec2f {
  if (k == 0u) { return vec2f(fm[ilat].x, 0.0); }
  if (k <= MMAX) { return fm[k * NLAT + ilat]; }
  return vec2f(0.0);
}

@compute @workgroup_size(${T})
fn fft_synth(@builtin(local_invocation_id) lid3: vec3u,
             @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  let ilat = wid.x;
  for (var k = lid; k < H; k += ${T}u) {
    let xk = spec(ilat, k);
    let xh = spec(ilat, H - k);
    let cj = vec2f(xh.x, -xh.y);
    let b = cmul(xk - cj, trig[k]);          // e^{+2 pi i k / N}
    bufA[k] = (xk + cj) + vec2f(-b.y, b.x);  // + i * b
  }
  fft_inplace(lid);
  for (var m = lid; m < H; m += ${T}u) {
    let z = ld(FFT_OUT_SEL, m);
    spat[ilat * NPHI + 2u * m] = z.x;
    spat[ilat * NPHI + 2u * m + 1u] = z.y;
  }
}
`;
}

export function fftAnalysRealWGSL(p: FourierParams): string {
  const H = p.nphi / 2;
  const T = fftThreads(H);
  return /* wgsl */ `
const MMAX: u32 = ${p.mmax}u;
const NLAT: u32 = ${p.nlat}u;
const NPHI: u32 = ${p.nphi}u;
const H: u32 = ${H}u;
@group(0) @binding(0) var<storage, read> spat: array<f32>;
@group(0) @binding(1) var<storage, read_write> fm: array<vec2f>;
${TRIG_BINDING}
${(p.radix ?? 4) === 4 ? stockham4(H, T, -1, p.nphi / H) : stockham(H, T, -1, p.nphi / H)}

@compute @workgroup_size(${T})
fn fft_analys(@builtin(local_invocation_id) lid3: vec3u,
              @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  let ilat = wid.x;
  for (var m = lid; m < H; m += ${T}u) {
    bufA[m] = vec2f(spat[ilat * NPHI + 2u * m], spat[ilat * NPHI + 2u * m + 1u]);
  }
  fft_inplace(lid);
  for (var k = lid; k <= MMAX; k += ${T}u) {
    let zk = ld(FFT_OUT_SEL, k);
    let zh = ld(FFT_OUT_SEL, (H - k) % H);   // Z[H] == Z[0]
    let cj = vec2f(zh.x, -zh.y);
    let xe = 0.5 * (zk + cj);
    let d = 0.5 * (zk - cj);
    let xo = vec2f(d.y, -d.x);                       // -i * d
    let w = vec2f(trig[k].x, -trig[k].y);            // e^{-2 pi i k / N}
    fm[k * NLAT + ilat] = xe + cmul(xo, w);
  }
}
`;
}
