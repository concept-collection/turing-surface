/**
 * The two things a side-by-side comparison of solver settings rests on.
 *
 * Both are silent when broken: the panels still animate, the difference norm
 * still produces a number, and the number is simply wrong — it reports a
 * disagreement between two runs that were never solving the same problem, or
 * that were never at the same time. Neither failure looks like a failure, which
 * is exactly why they are pinned here.
 *
 *   1. One initial condition. Sessions at different lmax seeded through
 *      sharedNoise hold the *same* spectral state, zero-padded — and the
 *      control shows what that is owed to: seeded the ordinary per-grid way,
 *      the same integer seed gives two unrelated fields.
 *
 *   2. One clock. dt varies by a power-of-two divisor, so `steps * dt` is
 *      bit-identical across variants and no comparison is ever made across a
 *      fraction of a timestep.
 *
 * Deliberately small — two sessions at niter 1, lmax 31 and 63 — because a
 * session compiles its whole unrolled step and this suite has to stay short.
 */
import { ModelSession } from '../src/mgpu/session.ts';
import { mModelByKey, defaultParams } from '../src/mgpu/registry.ts';
import { prolongCoeffs, sharedNoise } from '../src/compare/sharedStart.ts';
import { lmIndex, nlmCalc } from '../src/sht/layout.ts';
import { crossProduct, mostResolved } from '../src/compare/variants.ts';

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

  // ---- one initial condition across lmax, and the control ------------------
  {
    const model = mModelByKey('schnakenberg')!;
    const params = defaultParams(model);
    const sessions: ModelSession[] = [];
    try {
      for (const lmax of [COARSE, FINE]) {
        sessions.push(await ModelSession.create({ device, model, params, lmax, niter: 1 }));
      }
      const [coarse, fine] = sessions;

      // What the study does: one band-limited field, evaluated on each grid.
      const noise = await sharedNoise(sessions, model.seedAmp, 1);
      sessions.forEach((s, i) => s.seedWith(noise[i]));
      const shared = compareStates(
        prolongCoeffs(await coarse.read('U'), COARSE, FINE),
        await fine.read('U'),
      );
      check(
        'compare: one shared perturbation gives both grids the same state',
        shared.rel < 5e-5,
        `max |dU| = ${shared.abs.toExponential(2)} ` +
          `(${(100 * shared.rel).toFixed(4)}% of max |U| = ${shared.scale.toFixed(3)}) ` +
          `across lmax ${COARSE} vs ${FINE}`,
      );

      // The control: the ordinary per-grid seeding these two would otherwise
      // get. One deviate per grid point, and the grids differ, so the same
      // integer seed is two different initial conditions -- comparing runs
      // started this way would report a difference that is entirely the seed.
      sessions.forEach((s) => s.seed(1));
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

/** Max absolute difference of two equal-length spectral states, and that
 *  difference relative to the scale of the reference. */
function compareStates(
  a: Float32Array,
  b: Float32Array,
): { abs: number; rel: number; scale: number } {
  let abs = 0;
  let scale = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    abs = Math.max(abs, Math.abs(a[i] - b[i]));
    scale = Math.max(scale, Math.abs(b[i]));
  }
  return { abs, rel: scale > 0 ? abs / scale : Infinity, scale };
}
