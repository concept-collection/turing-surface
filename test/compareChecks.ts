/**
 * The two things a side-by-side comparison of solver settings rests on.
 *
 * Both are silent when broken: the panels still animate, the difference norm
 * still produces a number, and the number is simply wrong — it reports a
 * disagreement between two runs that were never solving the same problem, or
 * that were never at the same time. Neither failure looks like a failure, which
 * is exactly why they are pinned here.
 *
 *   1. One initial condition, in both of the ways a model can seed. A model
 *      that calls `randnfun3` — every shipped one does — gets a field in space,
 *      so one table drawn once is one field on every grid; the control is the
 *      per-session draw the study must not do. A model that takes `noise` gets
 *      one deviate per grid point, which sharedNoise has to project; the
 *      control there is starker, since the same integer seed on two grids is
 *      simply two unrelated fields.
 *
 *   2. One clock. dt varies by a power-of-two divisor, so `steps * dt` is
 *      bit-identical across variants and no comparison is ever made across a
 *      fraction of a timestep.
 *
 * Deliberately small — pairs of sessions at niter 1, lmax 31 and 63 — because a
 * session compiles its whole unrolled step and this suite has to stay short.
 */
import { ModelSession } from '../src/mgpu/session.ts';
import { mModelByKey, defaultParams, type MModel, type ParamSpec } from '../src/mgpu/registry.ts';
import { prolongCoeffs, sharedNoise, sharedModes } from '../src/compare/sharedStart.ts';
import linearSource from './models/linear.m?raw';
import { lmIndex, nlmCalc } from '../src/sht/layout.ts';
import { crossProduct, mostResolved } from '../src/compare/variants.ts';
import { floorRange } from '../src/render/colorbar.ts';

type Check = (name: string, ok: boolean, detail: string) => void;
type Log = (line: string) => void;

const COARSE = 31;
const FINE = 63;

export async function compareChecks(
  device: GPUDevice,
  check: Check,
  log: Log,
): Promise<void> {
  log('\ncompare mode (convergence study):');

  // ---- prolongCoeffs: every (l, m) lands on itself -------------------------
  {
    const src = new Float32Array(2 * nlmCalc(COARSE, COARSE));
    for (let m = 0; m <= COARSE; m++) {
      for (let l = m; l <= COARSE; l++) {
        const i = 2 * lmIndex(COARSE, l, m);
        src[i] = l + m / 100;
        src[i + 1] = -l - m / 100;
      }
    }
    const out = prolongCoeffs(src, COARSE, FINE);
    let moved = 0;
    let leaked = 0;
    for (let m = 0; m <= FINE; m++) {
      for (let l = m; l <= FINE; l++) {
        const j = 2 * lmIndex(FINE, l, m);
        if (l <= COARSE && m <= COARSE) {
          const i = 2 * lmIndex(COARSE, l, m);
          if (out[j] !== src[i] || out[j + 1] !== src[i + 1]) moved++;
        } else if (out[j] !== 0 || out[j + 1] !== 0) {
          leaked++;
        }
      }
    }
    check(
      'compare: prolongation puts every coefficient at its own (l, m)',
      moved === 0 && leaked === 0,
      `${moved} misplaced, ${leaked} non-zero above the source band ` +
        `(${nlmCalc(COARSE, COARSE)} -> ${nlmCalc(FINE, FINE)} coefficients)`,
    );
  }

  // ---- a file's exact state loads onto every grid --------------------------
  // What a reference-file study does instead of seeding: the file's spectral
  // state pushed into each variant by loadState, prolonged into its band. The
  // load is a plain upload, so the state must come back bit-exact; and read on
  // one shared grid the variants must then show one field, because synthesis
  // of the same band-limited coefficients is evaluation, not resampling.
  {
    const model = mModelByKey('allencahn')!;
    const params = defaultParams(model);
    const sessions: ModelSession[] = [];
    try {
      for (const lmax of [COARSE, FINE]) {
        sessions.push(await ModelSession.create({ device, model, params, lmax, niter: 0 }));
      }
      const [coarse, fine] = sessions;
      // A deterministic band-limited state, decaying like a real spectrum;
      // m = 0 imaginary parts stay zero (the state is a real field).
      const q = new Float32Array(2 * nlmCalc(COARSE, COARSE));
      for (let m = 0; m <= COARSE; m++) {
        for (let l = m; l <= COARSE; l++) {
          const i = 2 * lmIndex(COARSE, l, m);
          const amp = Math.exp(-l / 6);
          q[i] = amp * Math.sin(1 + 3 * l + 7 * m);
          q[i + 1] = m === 0 ? 0 : amp * Math.cos(2 + 5 * l + 11 * m);
        }
      }
      coarse.loadState({ U: q });
      fine.loadState({ U: prolongCoeffs(q, COARSE, FINE) });

      const back = await coarse.read('U');
      let exact = back.length === q.length;
      if (exact) {
        for (let i = 0; i < q.length; i++) {
          if (back[i] !== q[i]) {
            exact = false;
            break;
          }
        }
      }
      check(
        'compare: loadState puts the exact coefficients in the state',
        exact,
        `${q.length} float32 values round-tripped bit-exact at lmax ${COARSE}`,
      );

      // The coarse session's own solver grid, so its display plan is the
      // solver's — the branch a crowded study lands on.
      for (const s of sessions) await s.setDisplayGrid(64, 128);
      const cu = await coarse.readSpecies(0);
      const fu = await fine.readSpecies(0);
      let maxd = 0;
      let scale = 0;
      for (let i = 0; i < cu.length; i++) {
        maxd = Math.max(maxd, Math.abs(cu[i] - fu[i]));
        scale = Math.max(scale, Math.abs(cu[i]));
      }
      check(
        'compare: one loaded state reads back as one field on a shared grid',
        maxd < 1e-4 * scale,
        `max |du| = ${maxd.toExponential(2)} vs max |u| = ${scale.toExponential(2)} ` +
          `across lmax ${COARSE} vs ${FINE}`,
      );
    } finally {
      for (const s of sessions) s.destroy();
    }
  }

  // ---- one random field across lmax: the shipped models' seeding -----------
  {
    const model = mModelByKey('schnakenberg')!;
    const params = defaultParams(model);
    const sessions: ModelSession[] = [];
    try {
      for (const lmax of [COARSE, FINE]) {
        sessions.push(await ModelSession.create({ device, model, params, lmax, niter: 1 }));
      }
      const [coarse, fine] = sessions;

      // What the study does: one coefficient table, drawn once, summed on each
      // variant's own grid points. The residual is the coarse grid's analysis of
      // a field with a little content above its band, not a difference in the
      // field -- so it is bounded by the perturbation, not by |U|, which is why
      // compareStates measures against the non-constant part.
      const noise = await sharedNoise(sessions, model.seedAmp, 1);
      const modes = await sharedModes(fine, 1);
      check(
        'compare: a randnfun3 model seeds every variant from one drawn table',
        modes !== null,
        modes ? `${modes[0]} Fourier modes, one table for both grids` : 'no table drawn',
      );
      for (let i = 0; i < sessions.length; i++) await sessions[i].seedWith(noise[i], modes);
      const shared = compareStates(
        prolongCoeffs(await coarse.read('U'), COARSE, FINE),
        await fine.read('U'),
      );
      check(
        'compare: one shared random field gives both grids the same state',
        shared.rel < 1e-3,
        `max |dU| = ${shared.abs.toExponential(2)} ` +
          `(${(100 * shared.rel).toFixed(3)}% of the perturbation, max ` +
          `${shared.scale.toExponential(2)}) across lmax ${COARSE} vs ${FINE}`,
      );

      // The control, and the reason `sharedModes` exists: left to seed itself
      // each session draws over *its own* bounding box, and a box is grid
      // samples of the surface, so the two draws are near neighbours rather than
      // one field. A tolerance is not what separates them — the shared table is
      // simply closer, and would be however either number moved.
      await coarse.seed(1);
      await fine.seed(1);
      const own = compareStates(
        prolongCoeffs(await coarse.read('U'), COARSE, FINE),
        await fine.read('U'),
      );
      check(
        'compare: control — a per-session draw is not the same field',
        own.abs > shared.abs,
        `per-session draws differ by ${own.abs.toExponential(2)}, ` +
          `${(own.abs / Math.max(shared.abs, 1e-30)).toFixed(1)}x the shared table's ` +
          `${shared.abs.toExponential(2)}`,
      );
    } finally {
      for (const s of sessions) s.destroy();
    }
  }

  // ---- one grid-point perturbation across lmax, and the control ------------
  // The other way a model can seed: `init(noise)` takes the host's field
  // directly, one deviate per grid point (the test models here, and any .m
  // edited to do it). Nothing about it is a function of space, so this is the
  // case sharedNoise's projection is for — and the case where the same integer
  // seed on two grids gives two entirely unrelated initial conditions.
  {
    const params = { c: 0, D: 1e-3, dt: 0.05 };
    const model = noiseModel();
    const sessions: ModelSession[] = [];
    try {
      for (const lmax of [COARSE, FINE]) {
        sessions.push(await ModelSession.create({ device, model, params, lmax, niter: 1 }));
      }
      const [coarse, fine] = sessions;

      const noise = await sharedNoise(sessions, model.seedAmp, 1);
      for (let i = 0; i < sessions.length; i++) await sessions[i].seedWith(noise[i]);
      const shared = compareStates(
        prolongCoeffs(await coarse.read('U'), COARSE, FINE),
        await fine.read('U'),
      );
      check(
        'compare: one shared perturbation gives both grids the same state',
        shared.rel < 5e-5,
        `max |dU| = ${shared.abs.toExponential(2)} ` +
          `(${(100 * shared.rel).toFixed(4)}% of max |U| = ${shared.scale.toExponential(2)}) ` +
          `across lmax ${COARSE} vs ${FINE}`,
      );

      await coarse.seed(1);
      await fine.seed(1);
      const plain = compareStates(
        prolongCoeffs(await coarse.read('U'), COARSE, FINE),
        await fine.read('U'),
      );
      check(
        'compare: control — the same integer seed alone does not do it',
        plain.abs > 20 * shared.abs,
        `per-grid seeding differs by ${plain.abs.toExponential(2)}, ` +
          `${(plain.abs / Math.max(shared.abs, 1e-30)).toExponential(1)}x the shared start's ` +
          `(seed amplitude ${model.seedAmp})`,
      );
      log(
        `  shared start: ${shared.abs.toExponential(2)}, ` +
          `per-grid seeds: ${plain.abs.toExponential(2)}`,
      );
    } finally {
      for (const s of sessions) s.destroy();
    }
  }

  // ---- one clock: steps * dt is bit-identical across the divisors ----------
  {
    const divisors = [1, 2, 4, 8];
    const steps = 4;
    let worst = 0;
    const cases: string[] = [];
    for (const model of ['schnakenberg', 'brusselator', 'allencahn']) {
      const dt = defaultParams(mModelByKey(model)!).dt;
      for (const div of divisors) {
        // A variant at dt/div takes div times as many steps to cover the same
        // span. Powers of two only touch the exponent, so both the divide and
        // the multiply back are exact and the two spans are the same float.
        const span = (steps * div) * (dt / div);
        const ulps = Math.abs(span - steps * dt);
        if (ulps > worst) worst = ulps;
        if (div === divisors[divisors.length - 1]) {
          cases.push(`${model} dt ${dt} -> ${dt / div}`);
        }
      }
    }
    check(
      'compare: a power-of-two dt divisor keeps every variant on one clock',
      worst === 0,
      `exact for every shipped dt x ${divisors.join('/')} (${cases.join(', ')})`,
    );
  }

  // ---- a uniform field is drawn uniform, on every grid --------------------
  // Schnakenberg seeds v as a literal constant (`vs * ones(...)`), so its whole
  // spread is the fp32 residue of the analys/synth round trip -- pole-localized
  // and grid-dependent, so scaled to its own extremes it paints two unrelated
  // pictures of the same constant, which is what a broken seeding would look
  // like. The spans below are measured (worst |deviation| x 2, on vs = 0.9):
  // lmax 63, 127, 255. See floorRange for where they come from.
  {
    const vs = 0.9;
    const spans = [5.2e-5, 1.8e-4, 4.4e-4];
    // Each must end up a small slice of the drawn range rather than all of it.
    const shares = spans.map((sp) => sp / (floorRange(vs - sp / 2, vs + sp / 2).hi -
      floorRange(vs - sp / 2, vs + sp / 2).lo));
    // ...while real structure keeps its own range exactly. v once the spots
    // have formed spans ~0.03 on the same 0.9, two orders above the residue.
    const real = floorRange(0.895, 0.924);
    check(
      'compare: fp32 residue on a constant field does not become a picture',
      shares.every((s) => s < 0.1) && real.lo === 0.895 && real.hi === 0.924,
      `residue uses ${shares.map((s) => `${(100 * s).toFixed(1)}%`).join(', ')} ` +
        `of the colormap at lmax 63/127/255; real pattern ` +
        `[${real.lo}, ${real.hi}] left untouched`,
    );
  }

  // ---- the variant grid and its reference ---------------------------------
  {
    const variants = crossProduct([1, 4], [31, 63], [1, 2]);
    const ref = variants[mostResolved(variants)];
    check(
      'compare: the reference is the most-resolved corner of the grid',
      variants.length === 8 &&
        ref.niter === 4 && ref.lmax === 63 && ref.dtDiv === 2 &&
        new Set(variants.map((v) => `${v.niter}/${v.lmax}/${v.dtDiv}`)).size === 8,
      `${variants.length} distinct variants, reference niter ${ref.niter} · ` +
        `lmax ${ref.lmax} · dt/${ref.dtDiv}`,
    );
  }
}

/**
 * The one-species linear test model, seeded from `noise` rather than from a
 * random field — `init(noise)`, so the host's grid-point field is what reaches
 * the state (test/models/linear.m). Never stepped here; the parameters exist
 * because the .m names them.
 */
function noiseModel(): MModel {
  const param = (key: string): ParamSpec => ({
    key, label: key, value: 0, min: -1e9, max: 1e9, step: 1,
  });
  return {
    key: 'linear',
    label: 'linear',
    blurb: '',
    species: ['u'],
    state: ['U'],
    params: ['c', 'D', 'dt'].map(param),
    pdeg: 1,
    seedAmp: 1e-2,
    source: linearSource,
  };
}

/**
 * Max absolute difference of two equal-length spectral states, and that
 * difference relative to the scale of the reference's *non-constant* part —
 * every coefficient but (l, m) = (0, 0), which is index 0 in either layout.
 *
 * Normalizing against the whole state would hide the question. A model seeded as
 * a perturbation of a uniform steady state puts that state in (0, 0) alone, two
 * orders above everything else, so |dU| / max |U| would report a comfortable
 * fraction of the *background* however unrelated the two perturbations were —
 * including when no perturbation arrived at all, which is what a table that
 * never reaches a session looks like. Against the perturbation, that failure
 * reads as a ratio of 1.
 */
function compareStates(
  a: Float32Array,
  b: Float32Array,
): { abs: number; rel: number; scale: number } {
  let abs = 0;
  let scale = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    abs = Math.max(abs, Math.abs(a[i] - b[i]));
    if (i >= 2) scale = Math.max(scale, Math.abs(b[i]));
  }
  return { abs, rel: scale > 0 ? abs / scale : Infinity, scale };
}
