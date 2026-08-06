/**
 * One point of a convergence study: a choice of the three knobs that decide
 * *how well* the same problem is being solved, rather than what the problem is.
 *
 *   niter   iterations of the implicit solve (structural — it unrolls into the
 *           compiled step, so each value is its own compiled session)
 *   lmax    the spectral band, and with it the grid (also structural)
 *   dtDiv   the timestep, as an integer divisor of the model's own dt
 *
 * dt is a *divisor* rather than a free value on purpose, and it is the whole
 * reason the comparison can be trusted: variants have to be compared at the
 * same model time, and with dt = dtBase/K every variant lands exactly on the
 * same t after K times as many steps — no rounding, no drift, no interpolation
 * in time. A free dt would put each variant on its own timeline and every
 * difference reported would be part real and part "these are 0.003 apart".
 */

export interface Variant {
  /** Iterations of the implicit solve. */
  niter: number;
  /** Spectral band limit. */
  lmax: number;
  /** Timestep divisor: this variant runs at dtBase / dtDiv. */
  dtDiv: number;
}

/** Stable identity of a variant, for keying maps and the reference <select>. */
export const variantKey = (v: Variant): string => `${v.niter}/${v.lmax}/${v.dtDiv}`;

/** Human label. The dt term is dropped when nothing varies it, so the common
 *  case (niter x lmax) reads as just those two. */
export const variantLabel = (v: Variant, showDt: boolean): string =>
  `niter ${v.niter} · lmax ${v.lmax}` + (showDt ? ` · dt/${v.dtDiv}` : '');

/**
 * Every combination of the selected values, in a stable order: coarsest first,
 * so the grid reads top-to-bottom from least to most resolved and the
 * reference (the last row) is the one everything is measured against.
 */
export function crossProduct(
  niters: number[],
  lmaxes: number[],
  dtDivs: number[],
): Variant[] {
  const out: Variant[] = [];
  for (const lmax of [...lmaxes].sort((a, b) => a - b)) {
    for (const dtDiv of [...dtDivs].sort((a, b) => a - b)) {
      for (const niter of [...niters].sort((a, b) => a - b)) {
        out.push({ niter, lmax, dtDiv });
      }
    }
  }
  return out;
}

/**
 * Index of the most-resolved variant: the natural reference, since it is the
 * one every other choice is an approximation of. Finer band first (it bounds
 * what can be represented at all), then more solve iterations, then smaller
 * timestep.
 */
export function mostResolved(variants: Variant[]): number {
  let best = 0;
  for (let i = 1; i < variants.length; i++) {
    const a = variants[i];
    const b = variants[best];
    if (
      a.lmax > b.lmax ||
      (a.lmax === b.lmax && a.niter > b.niter) ||
      (a.lmax === b.lmax && a.niter === b.niter && a.dtDiv > b.dtDiv)
    ) {
      best = i;
    }
  }
  return best;
}

/** Distinguishable line/label colors, one per variant row. */
export const VARIANT_COLORS = [
  '#0969da', '#bf8700', '#1a7f37', '#cf222e', '#8250df', '#0f7c8a',
];
