/**
 * WGSL Legendre-transform kernels, modeled on leg_m_kernel / ileg_m_kernel
 * in SHT/cuda_legendre.gen.cu (non-Ishioka fp32 path: SHTNS disables the
 * Ishioka recurrence for fp32 because it loses too much accuracy).
 *
 * Synthesis:  F_m(theta_i) = sum_{l=m..lmax} Q_lm * ytilde_l^m(theta_i)
 *   - one thread per latitude, one workgroup row per m (workgroup_id.y).
 * Analysis:   Q_lm = sum_i w_i * G_m(theta_i) * ytilde_l^m(theta_i)
 *   - one workgroup per m; threads own latitudes (strided); per-l pair
 *     workgroup tree reduction (portable stand-in for the CUDA warp
 *     shuffles).
 *
 * The associated Legendre functions are generated on the fly by the
 * standard 3-term recurrence over l (coefficients a,b precomputed on the
 * host in f64), with the SHTNS fp32 rescaling scheme for sin(theta)^m
 * underflow (see common.ts).
 */
import { RESCALE_WGSL } from './common.ts';

export interface LegParams {
  lmax: number;
  mmax: number;
  nlat: number;
  wgSynth: number; // workgroup size for synthesis (threads over latitude)
  wgAnalys: number; // workgroup size for analysis (power of two)
  /** Use subgroup reductions in the analysis kernel (needs the `subgroups` feature). */
  subgroups?: boolean;
  /** l-pairs accumulated before the span is reduced (subgroup path only). */
  spanPairs?: number;
  /**
   * Fold north/south latitude pairs onto one recurrence (halves Legendre work).
   * Needs an equator-symmetric grid with even nlat, which the Gauss grid is.
   */
  parity?: boolean;
}

const BINDINGS = /* wgsl */ `
@group(0) @binding(0) var<storage, read> ab: array<vec2f>;     // (a_l^m, b_l^m) per lm
@group(0) @binding(1) var<storage, read> amm: array<f32>;      // seed per m
@group(0) @binding(2) var<storage, read> ctstw: array<f32>;    // [ct | st | w], each NLAT
`;

export function legSynthWGSL(p: LegParams): string {
  const half = p.parity === true;
  return /* wgsl */ `
${RESCALE_WGSL}
const LMAX: u32 = ${p.lmax}u;
const NLAT: u32 = ${p.nlat}u;
const NLAT_2: u32 = ${p.nlat / 2}u;
${BINDINGS}
@group(0) @binding(3) var<storage, read> qlm: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> fm: array<vec2f>;  // [(m)*NLAT + ilat]

@compute @workgroup_size(${p.wgSynth})
fn leg_synth(@builtin(global_invocation_id) gid: vec3u,
             @builtin(workgroup_id) wid: vec3u) {
  let ilat = gid.x;
  let m = wid.y;
  if (ilat >= ${half ? 'NLAT_2' : 'NLAT'}) { return; }

  let ct = ctstw[ilat];
  let st = ctstw[NLAT + ilat];
  let base = m * (LMAX + 1u) - (m * (m - 1u)) / 2u;   // lm index of (l=m, m)

  var seed = sinpow_rescaled(st, m);
  var y0 = seed.y0 * amm[m];
  var ny = seed.ny;
  var y1: f32 = 0.0;
  if (m < LMAX) {
    y1 = ab[base + 1u].x * ct * y0;
  }

${
    half
      ? `  // Parity folding: ytilde_l^m(-x) = (-1)^(l-m) ytilde_l^m(x) and the Gauss
  // grid is symmetric, so one recurrence serves a north/south pair. y0 always
  // carries even (l-m) and y1 odd, so summing them apart gives
  //   F_m(north) = accE + accO,  F_m(south) = accE - accO.
  var accE = vec2f(0.0);
  var accO = vec2f(0.0);`
      : `  var acc = vec2f(0.0);`
  }
  var l = m;
  loop {
    if (ny == 0) {
${
    half
      ? `      accE += y0 * qlm[base + (l - m)];
      if (l + 1u <= LMAX) {
        accO += y1 * qlm[base + (l + 1u - m)];
      }`
      : `      acc += y0 * qlm[base + (l - m)];
      if (l + 1u <= LMAX) {
        acc += y1 * qlm[base + (l + 1u - m)];
      }`
  }
    } else if (abs(y0) > RESCALE_THR) {
      ny += 1;
      y0 *= INV_SCALE;
      y1 *= INV_SCALE;
    }
    if (l + 2u > LMAX) { break; }
    // Advance (y_l, y_{l+1}) to (y_{l+2}, y_{l+3}).
    //
    // Written in exactly the shape leg_analys uses below — both coefficients
    // fetched unconditionally, the new y0 carried in a temporary rather than
    // assigned and then read back by the y1 update. The shorter form,
    //
    //   let c0 = ab[base + (l + 2u - m)];
    //   y0 = c0.x * ct * y1 + c0.y * y0;
    //   if (l + 3u <= LMAX) { ... y1 = c1.x * ct * y0 + c1.y * y1; }
    //
    // says the same thing and is what this was, but NVIDIA's Vulkan compiler
    // (driver 590.48, Blackwell) mis-compiles it: c0 reads as (0, 0) on the
    // first iteration, so y_{l+2} comes out exactly zero and every later term
    // follows a different solution of the recurrence, reaching ~1e11 by l = 63.
    // leg_analys, doing the same arithmetic in this shape, was correct on the
    // same driver. See scripts/diagnose-leg.ts, which is how that was found.
    let a0 = ab[base + (l + 2u - m)];
    var a1 = vec2f(0.0);
    if (l + 3u <= LMAX) {
      a1 = ab[base + (l + 3u - m)];
    }
    let t0 = a0.x * ct * y1 + a0.y * y0;
    y1 = a1.x * ct * t0 + a1.y * y1;
    y0 = t0;
    l += 2u;
  }
${
    half
      ? `  fm[m * NLAT + ilat] = accE + accO;
  fm[m * NLAT + (NLAT - 1u - ilat)] = accE - accO;`
      : `  fm[m * NLAT + ilat] = acc;`
  }
}
`;
}

export function legAnalysWGSL(p: LegParams): string {
  const half = p.parity === true;
  // parity folding leaves only the northern half of the grid to walk
  const K = Math.ceil((half ? p.nlat / 2 : p.nlat) / p.wgAnalys);
  // With subgroups, the per-l-pair reduction is one subgroupAdd plus a combine
  // across subgroups: 2 barriers instead of 1 + log2(wgAnalys). This is what
  // SHTNS's CUDA kernel does with warp shuffles. `red` then holds one partial
  // per subgroup; WebGPU guarantees subgroup size >= 4, so wgAnalys/4 is a safe
  // upper bound on how many there can be.
  const sg = p.subgroups === true;
  // Reduce once per span of l-pairs rather than once per pair. The l-loop is
  // serial, so its barriers are the critical path: at lmax=127 the m=0
  // workgroup paid 2 of them 64 times over. SHTNS amortizes the same way
  // (LSPAN_A = 16, or 32 for fp32), staging a whole span before reducing.
  // Partials for the span live in registers and are combined in one batch.
  const nsubMax = Math.max(1, p.wgAnalys / 4); // WebGPU guarantees subgroup size >= 4
  // 16 pairs = 32 l-values, which is what SHTNS uses for fp32 (LSPAN_A). Clamped
  // so `red` stays within 8 KB of workgroup storage, since nsubMax has to assume
  // the smallest legal subgroup and would otherwise oversize it badly.
  const pairs = sg
    ? Math.max(1, Math.min(p.spanPairs ?? 16, Math.floor(8192 / (nsubMax * 16))))
    : 1;
  const redLen = sg ? nsubMax * pairs : p.wgAnalys;
  return /* wgsl */ `${sg ? 'enable subgroups;\n' : ''}
${RESCALE_WGSL}
const LMAX: u32 = ${p.lmax}u;
const NLAT: u32 = ${p.nlat}u;
const WG: u32 = ${p.wgAnalys}u;
const K: u32 = ${K}u;
const NLAT_2: u32 = ${p.nlat / 2}u;
const PAIRS: u32 = ${pairs}u;
${BINDINGS}
@group(0) @binding(3) var<storage, read> fm: array<vec2f>;        // [(m)*NLAT + ilat]
@group(0) @binding(4) var<storage, read_write> qout: array<vec2f>;

var<workgroup> red: array<vec4f, ${redLen}>;

@compute @workgroup_size(${p.wgAnalys})
fn leg_analys(@builtin(local_invocation_id) lid3: vec3u,
              @builtin(workgroup_id) wid: vec3u${
                sg
                  ? ',\n              @builtin(subgroup_size) sgSize: u32,\n              @builtin(subgroup_invocation_id) sgLane: u32'
                  : ''
              }) {
  let lid = lid3.x;
  let m = wid.x;
  let base = m * (LMAX + 1u) - (m * (m - 1u)) / 2u;

  // per-thread recurrence state for K latitudes
  var y0v: array<f32, ${K}>;
  var y1v: array<f32, ${K}>;
  var nyv: array<i32, ${K}>;
  var ctv: array<f32, ${K}>;
${
    half
      ? `  // Transpose of the synthesis folding: splitting the latitude sum into
  // hemispheres gives Q_lm = sum_north w_i * ytilde * (G_north +/- G_south),
  // with + for even (l-m) and - for odd -- which the loop already routes
  // through y0 and y1 respectively.
  var wpv: array<vec2f, ${K}>;
  var wmv: array<vec2f, ${K}>;`
      : `  var wfv: array<vec2f, ${K}>;`
  }

  for (var k = 0u; k < K; k++) {
    let lat = lid + k * WG;
    var ct: f32 = 0.0;
    var st: f32 = 0.0;
${
    half
      ? `    var wp = vec2f(0.0);
    var wm = vec2f(0.0);
    if (lat < NLAT_2) {
      ct = ctstw[lat];
      st = ctstw[NLAT + lat];
      let w = ctstw[2u * NLAT + lat];                    // Gauss weight (incl. 2*pi/nphi)
      let gN = fm[m * NLAT + lat];
      let gS = fm[m * NLAT + (NLAT - 1u - lat)];
      wp = (gN + gS) * w;
      wm = (gN - gS) * w;
    }`
      : `    var wf = vec2f(0.0);
    if (lat < NLAT) {
      ct = ctstw[lat];
      st = ctstw[NLAT + lat];
      wf = fm[m * NLAT + lat] * ctstw[2u * NLAT + lat];  // Gauss weight (incl. 2*pi/nphi)
    }`
  }
    ctv[k] = ct;
    let seed = sinpow_rescaled(st, m);
    y0v[k] = seed.y0 * amm[m];
    nyv[k] = seed.ny;
    y1v[k] = 0.0;
    if (m < LMAX) {
      y1v[k] = ab[base + 1u].x * ct * y0v[k];
    }
${half ? '    wpv[k] = wp;\n    wmv[k] = wm;' : '    wfv[k] = wf;'}
  }

  var l = m;
${
    sg
      ? `  // Accumulate up to PAIRS l-pairs into registers, then reduce the whole span
  // at once: 2 barriers per span instead of 2 per pair.
  loop {
    let lstart = l;
    var npairs = 0u;
    var last = false;
    let sub = lid / sgSize;
    for (var jj = 0u; jj < PAIRS; jj++) {
      var c0 = vec2f(0.0);
      var c1 = vec2f(0.0);
      for (var k = 0u; k < K; k++) {
        if (nyv[k] == 0) {
${
      half
        ? `          c0 += wpv[k] * y0v[k];   // even (l-m): hemispheres add
          c1 += wmv[k] * y1v[k];   // odd  (l-m): hemispheres subtract`
        : `          c0 += wfv[k] * y0v[k];
          c1 += wfv[k] * y1v[k];`
    }
        } else if (abs(y0v[k]) > RESCALE_THR) {
          nyv[k] += 1;
          y0v[k] *= INV_SCALE;
          y1v[k] *= INV_SCALE;
        }
      }
      // subgroupAdd needs no barrier, so the per-subgroup partial can go
      // straight to shared memory; only the cross-subgroup combine below has
      // to wait, and it waits once for the whole span.
      let part = subgroupAdd(vec4f(c0, c1));
      if (sgLane == 0u) { red[sub * PAIRS + jj] = part; }
      npairs = jj + 1u;
      if (l + 2u > LMAX) { last = true; break; }
      let a0 = ab[base + (l + 2u - m)];
      var a1 = vec2f(0.0);
      if (l + 3u <= LMAX) {
        a1 = ab[base + (l + 3u - m)];
      }
      for (var k = 0u; k < K; k++) {
        let t0 = a0.x * ctv[k] * y1v[k] + a0.y * y0v[k];
        y0v[k] = t0;
        y1v[k] = a1.x * ctv[k] * t0 + a1.y * y1v[k];
      }
      l += 2u;
    }

    workgroupBarrier();
    if (lid == 0u) {
      let nsub = (WG + sgSize - 1u) / sgSize;
      for (var jj = 0u; jj < npairs; jj++) {
        var tot = vec4f(0.0);
        for (var i = 0u; i < nsub; i++) { tot += red[i * PAIRS + jj]; }
        let ll = lstart + 2u * jj;
        qout[base + (ll - m)] = tot.xy;
        if (ll + 1u <= LMAX) {
          qout[base + (ll + 1u - m)] = tot.zw;
        }
      }
    }
    workgroupBarrier();   // red is reused by the next span

    if (last) { break; }
  }`
      : `  loop {
    var c0 = vec2f(0.0);
    var c1 = vec2f(0.0);
    for (var k = 0u; k < K; k++) {
      if (nyv[k] == 0) {
${
          half
            ? `        c0 += wpv[k] * y0v[k];   // even (l-m): hemispheres add
        c1 += wmv[k] * y1v[k];   // odd  (l-m): hemispheres subtract`
            : `        c0 += wfv[k] * y0v[k];
        c1 += wfv[k] * y1v[k];`
        }
      } else if (abs(y0v[k]) > RESCALE_THR) {
        nyv[k] += 1;
        y0v[k] *= INV_SCALE;
        y1v[k] *= INV_SCALE;
      }
    }
    // workgroup tree reduction of (c0, c1)
    red[lid] = vec4f(c0, c1);
    workgroupBarrier();
    var s = WG / 2u;
    while (s > 0u) {
      if (lid < s) { red[lid] += red[lid + s]; }
      workgroupBarrier();
      s = s >> 1u;
    }
    if (lid == 0u) {
      qout[base + (l - m)] = red[0].xy;
      if (l + 1u <= LMAX) {
        qout[base + (l + 1u - m)] = red[0].zw;
      }
    }
    if (l + 2u > LMAX) { break; }
    let a0 = ab[base + (l + 2u - m)];
    var a1 = vec2f(0.0);
    if (l + 3u <= LMAX) {
      a1 = ab[base + (l + 3u - m)];
    }
    for (var k = 0u; k < K; k++) {
      let t0 = a0.x * ctv[k] * y1v[k] + a0.y * y0v[k];
      y0v[k] = t0;
      y1v[k] = a1.x * ctv[k] * t0 + a1.y * y1v[k];
    }
    l += 2u;
  }`
  }
}
`;
}
