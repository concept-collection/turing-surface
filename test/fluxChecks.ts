/**
 * The flux-form (six-transform) Laplace-Beltrami scheme of
 * docs/reduced-transforms.md, against the two things that can
 * silently go wrong with it:
 *
 * 1. The smoothness claim (doc Sec 2, validation Sec 7.1). The whole scheme
 *    rests on the analysed fluxes P and Qtilde being smooth functions on the
 *    sphere — that is a mathematical property of the p1/p2/q2 weighting, so it
 *    is checked in f64 on the CPU, where a failure is a wrong formula and not
 *    round-off. The fields are synthesized and re-analysed on a grid with
 *    twice the band limit: content beyond the band is exactly the non-smooth
 *    residue the weighting is supposed to remove.
 *
 *    Two surfaces split the claim's two halves. On the *round sphere* the
 *    correctly weighted fluxes are exactly band-limited, so their beyond-band
 *    tail is f64 round-off, while the doc's Sec 8 counterexample
 *    Qtilde/sin(theta) — bounded but with a phi-dependent polar limit — keeps
 *    an algebraically decaying tail orders of magnitude above it: the
 *    decisive smooth-vs-non-smooth discrimination, plus the closed-form check
 *    p1 = q2 = 1, p2 = 0, r = 1/sin^2(theta). On *bumpy* (non-axisymmetric,
 *    so the off-diagonal p2 does real work) nothing is band-limited and every
 *    smooth field's tail is set by the weights' own spectral decay, so the
 *    check there is the doc's relative one: P and Qtilde must sit on the same
 *    footing as the Cartesian gradient component Algorithm 4 analyses.
 *
 * 2. The operator identity (validation Sec 7.2/7.4). The flux form and the
 *    Cartesian-gradient form (models/schnakenberg.m vs
 *    models/schnakenberg_alg4.m) are the same operator, so a real simulation
 *    driven by one must track the other to fp32 accumulation — checked on a
 *    non-axisymmetric surface, where the off-diagonal weight p2 actually does
 *    something. The headline transform count (6 vs 12 per species per
 *    iteration) is asserted from the compiled op sequences, not the doc.
 */
import { ShtPlan } from '../src/sht/sht.ts';
import { DerivPlan } from '../src/sht/deriv.ts';
import { ShtReference } from '../src/sht/reference.ts';
import { gridForLmax, lmIndex, nlmCalc, type ShtConfig } from '../src/sht/layout.ts';
import { ModelSession } from '../src/mgpu/session.ts';
import { mModelByKey, defaultParams } from '../src/mgpu/registry.ts';
import { Geometry } from '../src/geom/geometry.ts';
import { mGeometryByKey, defaultGeometryParams } from '../src/geom/registry.ts';
import { computeFluxMetric } from '../src/geom/metric.ts';
import type { Check, Log } from './analyticChecks.ts';

/** Band limit of the test surface and field. */
const LMAX = 24;
/** Band limit of the oversampled analysis grid the tails are measured on. */
const LMAX_HI = 63;
/** Degrees at and above this count as "beyond-band tail": LMAX+1 is the last
 *  degree with direct content, and the smooth-but-not-band-limited metric
 *  weights spread it upward with (their own) exponentially decaying spectra,
 *  so the window starts well above the band edge. */
const TAIL_START = 44;

/** The part of a transform layout the spectral helpers need. */
interface Band {
  lmax: number;
  mmax: number;
}

/** Per-degree spectral amplitude: E(l) = sqrt(sum_m |q_l^m|^2). */
function degreeEnergy(band: Band, qlm: ArrayLike<number>): Float64Array {
  const E = new Float64Array(band.lmax + 1);
  for (let m = 0; m <= band.mmax; m++) {
    for (let l = m; l <= band.lmax; l++) {
      const i = lmIndex(band.lmax, l, m);
      E[l] += qlm[2 * i] ** 2 + qlm[2 * i + 1] ** 2;
    }
  }
  for (let l = 0; l <= band.lmax; l++) E[l] = Math.sqrt(E[l]);
  return E;
}

/** max E(l) over l >= TAIL_START, relative to max E(l) overall. */
function tailRel(band: Band, qlm: ArrayLike<number>): number {
  const E = degreeEnergy(band, qlm);
  let bulk = 0;
  let tail = 0;
  for (let l = 0; l <= band.lmax; l++) {
    if (E[l] > bulk) bulk = E[l];
    if (l >= TAIL_START && E[l] > tail) tail = E[l];
  }
  return tail / Math.max(bulk, 1e-300);
}

/** Re-index coefficients from the lo layout into the hi layout (zero-padded). */
function padSpectrum(qlo: ArrayLike<number>, lo: Band, hi: Band): Float64Array {
  const out = new Float64Array(2 * nlmCalc(hi.lmax, hi.mmax));
  for (let m = 0; m <= lo.mmax; m++) {
    for (let l = m; l <= lo.lmax; l++) {
      const src = lmIndex(lo.lmax, l, m);
      const dst = lmIndex(hi.lmax, l, m);
      out[2 * dst] = qlo[2 * src];
      out[2 * dst + 1] = qlo[2 * src + 1];
    }
  }
  return out;
}

/** Deterministic random band-limited spectrum with O(1) coefficients. */
function flatSpectrum(band: Band, seed: number): Float64Array {
  const nlm = nlmCalc(band.lmax, band.mmax);
  const q = new Float64Array(2 * nlm);
  let s = seed >>> 0;
  const rnd = () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return (s / 4294967296) * 2 - 1;
  };
  for (let k = 0; k < 2 * nlm; k++) q[k] = rnd();
  for (let l = 0; l <= band.lmax; l++) q[2 * lmIndex(band.lmax, l, 0) + 1] = 0;
  return q;
}

export interface FluxCheckOptions {
  /**
   * Run the live flux-vs-Algorithm-4 A/B (4 sessions at lmax 63). On by
   * default, but — like geometryChecks' sweep, and for the same reason — a
   * browser recompiles every session's unrolled step from scratch on software
   * WebGPU, so the page leaves it out unless asked (?sweep=1) to keep CI
   * short. The f64 smoothness checks always run; they are CPU work.
   */
  ab?: boolean;
}

export async function fluxChecks(
  device: GPUDevice,
  check: Check,
  log: Log,
  opts: FluxCheckOptions = {},
): Promise<void> {
  // ---- 1a. round sphere: closed-form weights, decisive discrimination -----
  {
    const hiGrid = gridForLmax(LMAX_HI, 1);
    const hi = { lmax: LMAX_HI, mmax: LMAX_HI, nlat: hiGrid.nlat, nphi: hiGrid.nphi };
    const ref = new ShtReference(hi);
    const npts = hi.nlat * hi.nphi;

    // The unit sphere needs no GPU build: analyse the closed-form embedding
    // on the fine grid directly, in f64.
    const xg = new Float64Array(npts);
    const yg = new Float64Array(npts);
    const zg = new Float64Array(npts);
    for (let i = 0; i < hi.nlat; i++) {
      const ct = ref.ct[i];
      const st = ref.st[i];
      for (let j = 0; j < hi.nphi; j++) {
        const phi = (2 * Math.PI * j) / hi.nphi;
        const k = i * hi.nphi + j;
        xg[k] = st * Math.cos(phi);
        yg[k] = st * Math.sin(phi);
        zg[k] = ct;
      }
    }
    const X = ref.analys(xg);
    const Y = ref.analys(yg);
    const Z = ref.analys(zg);

    const sXt = [ref.sinDtheta(X), ref.sinDtheta(Y), ref.sinDtheta(Z)];
    const Xp = [ref.dphi(X), ref.dphi(Y), ref.dphi(Z)];
    const { p1, p2, q2, r } = computeFluxMetric(
      npts, sXt[0], sXt[1], sXt[2], Xp[0], Xp[1], Xp[2],
    );

    // On the sphere the weights have a closed form: p1 = q2 = 1, p2 = 0,
    // r = 1/sin^2(theta) — the flux-form counterpart of geometryChecks'
    // closed-form V check, pinning computeFluxMetric before it is buried
    // under the operator. f64 throughout, so the tolerance is conditioning
    // at the polar rings, not fp32.
    let worst = 0;
    for (let i = 0; i < hi.nlat; i++) {
      const st2 = ref.st[i] * ref.st[i];
      for (let j = 0; j < hi.nphi; j++) {
        const k = i * hi.nphi + j;
        worst = Math.max(
          worst,
          Math.abs(p1[k] - 1),
          Math.abs(p2[k]),
          Math.abs(q2[k] - 1),
          Math.abs(r[k] * st2 - 1),
        );
      }
    }
    check(
      'flux: sphere weights match the closed form (p1 = q2 = 1, p2 = 0, r = 1/sin^2)',
      worst < 1e-9,
      `max deviation ${worst.toExponential(2)} in f64`,
    );

    // Flat random u, band-limited at LMAX. The properly weighted fluxes are
    // then *exactly* band-limited (P = sin(theta) dtheta u, Qtilde = dphi u),
    // so their beyond-band tails are pure round-off; the Sec 8 control
    // Qtilde/sin(theta) is not a function on the sphere and keeps a fat tail.
    const band = { lmax: LMAX, mmax: LMAX };
    const u = padSpectrum(flatSpectrum(band, 777), band, hi);
    const A = ref.sinDtheta(u);
    const B = ref.dphi(u);
    const P = new Float64Array(npts);
    const Qt = new Float64Array(npts);
    const control = new Float64Array(npts);
    for (let i = 0; i < hi.nlat; i++) {
      const st = ref.st[i];
      for (let j = 0; j < hi.nphi; j++) {
        const k = i * hi.nphi + j;
        P[k] = p1[k] * A[k] + p2[k] * B[k];
        Qt[k] = p2[k] * A[k] + q2[k] * B[k];
        control[k] = Qt[k] / st;
      }
    }
    const tails = {
      P: tailRel(hi, ref.analys(P)),
      Qt: tailRel(hi, ref.analys(Qt)),
      control: tailRel(hi, ref.analys(control)),
    };
    log(
      `  flux smoothness on the sphere (f64, band ${LMAX}, analysed to ${LMAX_HI}, ` +
        `tail l >= ${TAIL_START}): P ${tails.P.toExponential(2)}, ` +
        `Qt ${tails.Qt.toExponential(2)}, control ${tails.control.toExponential(2)}`,
    );
    check(
      'flux: on the sphere the fluxes are band-limited and the non-smooth control is not',
      tails.P < 1e-10 && tails.Qt < 1e-10 &&
        tails.control > 1e3 * Math.max(tails.P, tails.Qt, 1e-14),
      `P ${tails.P.toExponential(2)}, Qt ${tails.Qt.toExponential(2)}, ` +
        `control ${tails.control.toExponential(2)}`,
    );
  }

  // ---- 1b. bumpy: the fluxes sit on the Cartesian gradient's footing ------
  {
    // The surface: bumpy, the one shipped geometry that is genuinely
    // non-axisymmetric (g_thetaphi != 0), so the off-diagonal weight p2 is
    // exercised. Built by the real pipeline at LMAX, then everything below is
    // CPU f64 from its band-limited coefficients.
    const g = mGeometryByKey('bumpy')!;
    const { nlat, nphi } = gridForLmax(LMAX, 3);
    const cfg = { lmax: LMAX, mmax: LMAX, nlat, nphi };
    const sht = await ShtPlan.create(device, cfg);
    const deriv = await DerivPlan.create(device, sht);
    const geometry = await Geometry.create({
      device, sht, cfg,
      source: g.source,
      paramNames: g.params.map((p) => p.key),
      params: defaultGeometryParams(g),
      deriv,
    });
    deriv.destroy();
    sht.destroy();

    const hiGrid = gridForLmax(LMAX_HI, 1);
    const hi = { lmax: LMAX_HI, mmax: LMAX_HI, nlat: hiGrid.nlat, nphi: hiGrid.nphi };
    const ref = new ShtReference(hi);
    const npts = hi.nlat * hi.nphi;

    // Embedding and test field, zero-padded into the fine layout. Both are
    // band-limited at LMAX, so on the fine grid every derived field's content
    // beyond the band is genuinely the non-band-limited part of the weights —
    // the thing being measured — and not aliasing.
    const X = padSpectrum(geometry.X, cfg, hi);
    const Y = padSpectrum(geometry.Y, cfg, hi);
    const Z = padSpectrum(geometry.Z, cfg, hi);
    const u = padSpectrum(flatSpectrum(cfg, 777), cfg, hi);

    // Tangents, both weightings, all f64.
    const sXt = [ref.sinDtheta(X), ref.sinDtheta(Y), ref.sinDtheta(Z)];
    const Xp = [ref.dphi(X), ref.dphi(Y), ref.dphi(Z)];
    const Xt = [ref.dtheta(X), ref.dtheta(Y), ref.dtheta(Z)];
    const { p1, p2, q2 } = computeFluxMetric(
      npts, sXt[0], sXt[1], sXt[2], Xp[0], Xp[1], Xp[2],
    );

    const A = ref.sinDtheta(u); // sin(theta) dtheta u
    const B = ref.dphi(u);      // dphi u

    // The two fluxes. No non-smooth control here: on a deformed surface
    // every smooth field's beyond-band tail is set by the weights' own
    // (slowly decaying) spectra, which swamps a pole singularity at this
    // resolution — the sphere block above is where the discrimination has
    // teeth. This block asserts the doc's relative criterion instead.
    const P = new Float64Array(npts);
    const Qt = new Float64Array(npts);
    for (let k = 0; k < npts; k++) {
      P[k] = p1[k] * A[k] + p2[k] * B[k];
      Qt[k] = p2[k] * A[k] + q2[k] * B[k];
    }

    // The known-smooth yardstick (doc Sec 2): the x component of the
    // Cartesian surface gradient, built the Algorithm-4 way from the inverse
    // metric quantities, in f64.
    const gradx = new Float64Array(npts);
    {
      const ut = ref.dtheta(u);
      const up = ref.dphi(u);
      for (let k = 0; k < npts; k++) {
        const gtt = Xt[0][k] ** 2 + Xt[1][k] ** 2 + Xt[2][k] ** 2;
        const gtp = Xt[0][k] * Xp[0][k] + Xt[1][k] * Xp[1][k] + Xt[2][k] * Xp[2][k];
        const gpp = Xp[0][k] ** 2 + Xp[1][k] ** 2 + Xp[2][k] ** 2;
        const det = gtt * gpp - gtp * gtp;
        const Vtx = (gpp * Xt[0][k] - gtp * Xp[0][k]) / det;
        const Vpx = (gtt * Xp[0][k] - gtp * Xt[0][k]) / det;
        gradx[k] = ut[k] * Vtx + up[k] * Vpx;
      }
    }

    const tails = {
      P: tailRel(hi, ref.analys(P)),
      Qt: tailRel(hi, ref.analys(Qt)),
      gradx: tailRel(hi, ref.analys(gradx)),
    };
    log(
      `  flux smoothness on bumpy (f64, band ${LMAX}, analysed to ${LMAX_HI}, ` +
        `tail l >= ${TAIL_START}): P ${tails.P.toExponential(2)}, ` +
        `Qt ${tails.Qt.toExponential(2)}, gradx ${tails.gradx.toExponential(2)}`,
    );
    // "Matching tails" (Sec 7.1): same footing as the Cartesian component,
    // with an order of magnitude of headroom on top of it. A wrong weighting
    // (a missing sin factor, say) puts genuinely non-smooth content into P or
    // Qtilde and the tail lands at O(bulk), far above this.
    const ceiling = Math.max(30 * tails.gradx, 1e-10);
    check(
      'flux: on bumpy, P and Qtilde tails match the Cartesian gradient component',
      tails.P < ceiling && tails.Qt < ceiling,
      `P ${tails.P.toExponential(2)}, Qt ${tails.Qt.toExponential(2)} vs ` +
        `ceiling ${ceiling.toExponential(2)}`,
    );
  }

  // ---- 2. flux form vs Algorithm 4, live, on a curved surface -------------
  if (!(opts.ab ?? true)) {
    log(
      '  flux A/B: skipped — run `npm run test:node` (desktop Dawn) or ' +
        '`npm run test:gpu -- --sweep` for the flux-vs-Algorithm-4 comparison.',
    );
  } else {
    const geometry = mGeometryByKey('bumpy')!;
    const geometryParams = defaultGeometryParams(geometry);
    const LMAX_AB = 63;
    const STEPS = 20;
    const states: Float32Array[] = [];
    const xformsPerIter: number[] = [];

    for (const key of ['schnakenberg', 'schnakenberg-alg4']) {
      const model = mModelByKey(key)!;
      const params = defaultParams(model);
      // Real transforms added by one solve iteration: synth/analys ops plus
      // dtheta/dphi (each of which contains a synthesis); the coefficient-
      // space dthetac/dphic shuffles are O(nlm) index gathers, not transforms.
      const counts: number[] = [];
      for (const niter of [0, 1]) {
        const session = await ModelSession.create({
          device, model, params, lmax: LMAX_AB,
          geometry, geometryParams, niter,
        });
        counts.push(
          session.describe().step.filter((l) =>
            l.startsWith('synth') || l.startsWith('analys') ||
            l.startsWith('dtheta ') || l.startsWith('dphi '),
          ).length,
        );
        if (niter === 1) {
          session.seed(1);
          session.step(STEPS);
          states.push(await session.read('U'));
        }
        session.destroy();
      }
      xformsPerIter.push(counts[1] - counts[0]);
    }

    // The headline number, from the compiled op sequences: 6 transforms per
    // species per iteration against Algorithm 4's 12 (2 species here).
    check(
      'flux: 6 transforms per species per iteration, versus 12',
      xformsPerIter[0] === 12 && xformsPerIter[1] === 24,
      `flux form adds ${xformsPerIter[0]} transforms/iteration, ` +
        `Algorithm 4 adds ${xformsPerIter[1]}`,
    );

    // Same operator, same discretization, different arithmetic path: after
    // STEPS steps the two states may differ only by fp32 accumulation. A
    // formulation error (wrong weight, wrong shift, missing sin) would show
    // up at O(1), not O(1e-3). Identical states would mean the A/B compared
    // one path to itself.
    let worst = 0;
    let identical = true;
    let finite = true;
    for (let i = 0; i < states[0].length; i++) {
      const d = Math.abs(states[0][i] - states[1][i]);
      if (d > worst) worst = d;
      if (states[0][i] !== states[1][i]) identical = false;
      if (!Number.isFinite(states[0][i]) || !Number.isFinite(states[1][i])) finite = false;
    }
    check(
      'flux: tracks the Algorithm-4 reference through a real simulation',
      finite && !identical && worst < 5e-3,
      `max |U_flux - U_alg4| = ${worst.toExponential(2)} after ${STEPS} steps ` +
        `on bumpy at lmax ${LMAX_AB}`,
    );
  }
}
