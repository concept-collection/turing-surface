/**
 * One initial condition, on every variant's grid.
 *
 * The host's seeded perturbation is one normal deviate per *grid point*
 * (src/mgpu/noise.ts), so two sessions at different lmax seeded from the same
 * integer do not start from the same field — they start from unrelated fields
 * that merely share a random seed. Comparing them would compare two different
 * problems, and every number the comparison produced would be meaningless.
 *
 * So the field is built once, band-limited at the *coarsest* variant's lmax,
 * and evaluated on each variant's own grid:
 *
 *   1. white noise on the coarsest grid
 *   2. analysed there -> coefficients up to lmax_min
 *   3. zero-padded into each variant's coefficient layout
 *   4. synthesized on that variant's grid
 *
 * Steps 3 and 4 are exact: the field is band-limited at lmax_min, and every
 * variant's band contains that, so each one receives the *same function*
 * sampled where it needs it. Running each model's own `init` on it then leaves
 * every session holding the identical spectral state (zero-padded), which is
 * what makes a pointwise comparison at later times mean something.
 *
 * The coarsest variant gets the projected field too, not the raw white noise
 * it was analysed from — otherwise it alone would start somewhere slightly
 * different from the others.
 *
 * A model whose `init` calls `randnfun3` (all of the shipped ones do) draws its
 * perturbation from a Fourier series on the surface's bounding box instead, and
 * that needs no projection: it is a function of space, evaluated wherever it is
 * asked, so one coefficient table *is* one field on every variant's grid. It
 * still has to be drawn once rather than per session — see `sharedModes`.
 */
import { lmIndex, nlmCalc } from '../sht/layout.ts';
import { seededNoise } from '../mgpu/noise.ts';
import type { ModelSession } from '../mgpu/session.ts';

/**
 * Re-index coefficients from a band limit into a wider one's layout, zero-
 * filling the degrees the source does not have. Both layouts are SHTNS
 * m-major with mmax = lmax, so nothing but the index mapping changes.
 */
export function prolongCoeffs(
  q: Float32Array,
  lmaxFrom: number,
  lmaxTo: number,
): Float32Array {
  if (lmaxTo === lmaxFrom) return q;
  if (lmaxTo < lmaxFrom) {
    throw new Error(`prolongCoeffs: cannot widen ${lmaxFrom} into a smaller ${lmaxTo}`);
  }
  const out = new Float32Array(2 * nlmCalc(lmaxTo, lmaxTo));
  for (let m = 0; m <= lmaxFrom; m++) {
    for (let l = m; l <= lmaxFrom; l++) {
      const from = 2 * lmIndex(lmaxFrom, l, m);
      const to = 2 * lmIndex(lmaxTo, l, m);
      out[to] = q[from];
      out[to + 1] = q[from + 1];
    }
  }
  return out;
}

/**
 * The same band-limited perturbation, sampled on each session's grid. Order
 * follows `sessions`. Nothing may be in flight on any session's transform
 * plan — the one-off analys/synth here use the plan's own scratch buffers.
 */
export async function sharedNoise(
  sessions: ModelSession[],
  amp: number,
  seed: number,
): Promise<Float32Array[]> {
  let base = sessions[0];
  for (const s of sessions) if (s.cfg.lmax < base.cfg.lmax) base = s;
  const coeffs = await base.sht.analys(seededNoise(base.npts, amp, seed));
  const out: Float32Array[] = [];
  for (const s of sessions) {
    out.push(await s.sht.synth(prolongCoeffs(coeffs, base.cfg.lmax, s.cfg.lmax)));
  }
  return out;
}

/**
 * The random field every variant seeds from, drawn once — from `reference`,
 * whose numbers the study quotes — or null for a model that does not call
 * `randnfun3`.
 *
 * One table for all of them is not merely an economy (the draw is interpreter
 * time, and at a fine wavelength seconds of it). Each session would otherwise
 * draw from *its own* bounding box, and a box comes from grid samples of the
 * surface: at different lmax those differ in the last digits, and the draw is
 * sensitive to the box — a different mode count consumes the RNG differently
 * and the fields stop being the same one. Drawing once removes the question.
 */
export function sharedModes(
  reference: ModelSession,
  seed: number,
): Promise<Float32Array | null> {
  return reference.drawSeedModes(seed);
}
