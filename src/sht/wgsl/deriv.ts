/**
 * WGSL kernels for the coefficient-space step of the theta/phi first-
 * derivative algorithm (evolving_surface/notes/algos.tex, Algorithm 1, theta
 * and phi branches only -- the Laplace-Beltrami operator built on these never
 * needs the second-derivative/curvature branches).
 *
 * Both derivatives are a shuffle across nearby spectral coefficients --
 * independent of latitude/longitude, so neither touches the Legendre
 * recurrence or Fourier stages in leg.ts/fourier.ts -- followed by the
 * *existing*, unchanged scalar synthesis pipeline. dtheta additionally
 * divides the synthesized grid field by sin(theta) afterwards.
 */

const WG = 64;

export interface DerivCoeffParams {
  nlm: number;
}

/**
 * v_l^m = alpha^+(l-1,m) * u_{l-1}^m + alpha^-(l+1,m) * u_{l+1}^m, the
 * coefficients of sin(theta) * dtheta(u) (algos.tex eq. v_coeffs). aPlus/
 * aMinus are precomputed zero at each m-block's boundary
 * (src/sht/derivCoeffs.ts), so the multiply is always mathematically
 * correct; the bounds checks below exist only to avoid reading past the ends
 * of the qlm array (an m-block-internal +-1 step never leaves the array, so
 * this is the only place it could).
 */
export function dthetaShuffleWGSL(p: DerivCoeffParams): string {
  return /* wgsl */ `
const NLM: u32 = ${p.nlm}u;

@group(0) @binding(0) var<storage, read> aPlus: array<f32>;
@group(0) @binding(1) var<storage, read> aMinus: array<f32>;
@group(0) @binding(2) var<storage, read> qlmIn: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> vOut: array<vec2f>;

@compute @workgroup_size(${WG})
fn dtheta_shuffle(@builtin(global_invocation_id) gid: vec3u) {
  let lm = gid.x;
  if (lm >= NLM) { return; }
  var v = vec2f(0.0);
  if (lm > 0u) { v += aPlus[lm] * qlmIn[lm - 1u]; }
  if (lm + 1u < NLM) { v += aMinus[lm] * qlmIn[lm + 1u]; }
  vOut[lm] = v;
}
`;
}

/**
 * (dphi u)_l^m = i*m*u_l^m: in the [re, im] row layout this swaps and
 * negates, re' = -m*im, im' = m*re (algos.tex eq. dYdphi).
 */
export function dphiShuffleWGSL(p: DerivCoeffParams): string {
  return /* wgsl */ `
const NLM: u32 = ${p.nlm}u;

@group(0) @binding(0) var<storage, read> mOf: array<u32>;
@group(0) @binding(1) var<storage, read> qlmIn: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> vOut: array<vec2f>;

@compute @workgroup_size(${WG})
fn dphi_shuffle(@builtin(global_invocation_id) gid: vec3u) {
  let lm = gid.x;
  if (lm >= NLM) { return; }
  let m = f32(mOf[lm]);
  let c = qlmIn[lm];
  vOut[lm] = vec2f(-m * c.y, m * c.x);
}
`;
}

export interface DivideParams {
  nlat: number;
  nphi: number;
}

/**
 * Elementwise divide by sin(theta): the grid-space finish of Algorithm 1's
 * dtheta branch (dtheta(u) = synth(v_l^m) / sin(theta)). Gauss nodes never
 * sit at the poles, so this never divides by zero.
 */
export function divideSinThetaWGSL(p: DivideParams): string {
  const npts = p.nlat * p.nphi;
  return /* wgsl */ `
const NLAT: u32 = ${p.nlat}u;
const NPHI: u32 = ${p.nphi}u;
const NPTS: u32 = ${npts}u;

@group(0) @binding(0) var<storage, read> sinTheta: array<f32>;
@group(0) @binding(1) var<storage, read_write> spat: array<f32>;

@compute @workgroup_size(${WG})
fn divide_sin_theta(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= NPTS) { return; }
  let ilat = i / NPHI;
  spat[i] = spat[i] / sinTheta[ilat];
}
`;
}

export interface FmDphiParams {
  mmax: number;
  nlat: number;
  nphi: number;
  /** Highest m kept; modes above are zeroed (mirrors the l-space filt). */
  mcut: number;
}

/**
 * The Fourier-space middle of the grid-space phi-derivative `dphig`:
 * fm holds the unnormalized DFT modes of each latitude row (what the
 * Fourier analysis stage produces), so d/dphi is fm[m] *= i*m/NPHI --
 * the 1/NPHI undoes the unnormalized analysis+synthesis round trip.
 * Modes above MCUT are zeroed: the Fourier analysis stage already
 * truncated m > mmax for free, and MCUT additionally mirrors the
 * top-degree filt so the differentiated field carries no content the
 * l-space route would not have kept.
 */
export function fmDphiWGSL(p: FmDphiParams): string {
  const count = (p.mmax + 1) * p.nlat;
  return /* wgsl */ `
const NLAT: u32 = ${p.nlat}u;
const COUNT: u32 = ${count}u;
const MCUT: u32 = ${Math.max(0, p.mcut)}u;
const INV_NPHI: f32 = ${1 / p.nphi};

@group(0) @binding(0) var<storage, read_write> fm: array<vec2f>;

@compute @workgroup_size(${WG})
fn fm_dphi(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= COUNT) { return; }
  let m = i / NLAT;
  var k: f32 = 0.0;
  if (m <= MCUT) { k = f32(m) * INV_NPHI; }
  let c = fm[i];
  fm[i] = vec2f(-k * c.y, k * c.x);
}
`;
}
