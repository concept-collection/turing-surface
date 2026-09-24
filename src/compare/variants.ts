/**
 * One point of a convergence study: a choice of the four knobs that decide
 * *how well* the same problem is being solved, rather than what the problem is.
 *
 *   solver  which method answers the models' solve(...) call (structural — it
 *           is a compile-time choice, so each value is its own compiled session)
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
import { isDirectSolver, solverKeys, type SolverKey } from '../mgpu/libs.ts';

export interface Variant {
  /** The solver behind solve(...). */
  solver: SolverKey;
  /** Iterations of the implicit solve. 0 for a direct solver, which has no
   *  iterations: the niter axis does not multiply it, and its label omits
   *  the count. */
  niter: number;
  /** Spectral band limit. */
  lmax: number;
  /** Timestep divisor: this variant runs at dtBase / dtDiv. */
  dtDiv: number;
}

/** Which parts of a label are worth printing: an axis nothing varies along
 *  is noise in every row, so it is dropped and the common case (niter x lmax)
 *  reads as just those two. */
export interface LabelParts {
  showSolver: boolean;
  showDt: boolean;
}

/** The parts to show for a given set of variants: each axis, iff it varies. */
export const labelParts = (variants: Variant[]): LabelParts => ({
  showSolver: variants.some((v) => v.solver !== variants[0].solver),
  showDt: variants.some((v) => v.dtDiv !== variants[0].dtDiv),
});

/** Stable identity of a variant, for keying maps and the reference <select>. */
export const variantKey = (v: Variant): string => `${v.solver}/${v.niter}/${v.lmax}/${v.dtDiv}`;

/** Human label. A direct solver has no iteration count to print, and its
 *  name is always shown: "niter 0" would say the opposite of what it does. */
export const variantLabel = (v: Variant, parts: LabelParts): string =>
  isDirectSolver(v.solver)
    ? `${v.solver} · lmax ${v.lmax}` + (parts.showDt ? ` · dt/${v.dtDiv}` : '')
    : (parts.showSolver ? `${v.solver} · ` : '') +
      `niter ${v.niter} · lmax ${v.lmax}` +
      (parts.showDt ? ` · dt/${v.dtDiv}` : '');

/**
 * Where a solver stands among the four, for choosing a reference: gmres
 * minimises the residual over the same Krylov space that bicgstab and
 * richardson search, so at equal iterations its iterate is never worse than
 * theirs; bicgstab in turn converges in fewer iterations than richardson on
 * the operators here; and exact is what all three converge toward. This is
 * the order the solvers/ files are listed in.
 */
const solverRank = (s: SolverKey): number => solverKeys.indexOf(s);

/** Iterations for ranking purposes: a direct solver has, in effect, all of
 *  them, so it outranks any iteration count of an iterative one. */
const effectiveNiter = (v: Variant): number => (isDirectSolver(v.solver) ? Infinity : v.niter);

/**
 * Every combination of the selected values, in a stable order: coarsest first,
 * so the grid reads top-to-bottom from least to most resolved and the
 * reference (the last row) is the one everything is measured against.
 */
export function crossProduct(
  solvers: SolverKey[],
  niters: number[],
  lmaxes: number[],
  dtDivs: number[],
): Variant[] {
  const out: Variant[] = [];
  for (const lmax of [...lmaxes].sort((a, b) => a - b)) {
    for (const dtDiv of [...dtDivs].sort((a, b) => a - b)) {
      for (const niter of [...niters].sort((a, b) => a - b)) {
        for (const solver of [...solvers].sort((a, b) => solverRank(a) - solverRank(b))) {
          if (isDirectSolver(solver)) continue;
          out.push({ solver, niter, lmax, dtDiv });
        }
      }
      // A direct solver has no niter axis: one variant per band and dt,
      // after every iterative one it is the limit of.
      for (const solver of [...solvers].sort((a, b) => solverRank(a) - solverRank(b))) {
        if (isDirectSolver(solver)) out.push({ solver, niter: 0, lmax, dtDiv });
      }
    }
  }
  return out;
}

/**
 * Index of the most-resolved variant: the natural reference, since it is the
 * one every other choice is an approximation of. Finer band first (it bounds
 * what can be represented at all), then more solve iterations, then the
 * stronger solver at those iterations, then smaller timestep. More iterations
 * of any solver rank ahead of a better solver at fewer, because the two are
 * not comparable across methods; the user can always pick otherwise.
 */
export function mostResolved(variants: Variant[]): number {
  let best = 0;
  for (let i = 1; i < variants.length; i++) {
    const a = variants[i];
    const b = variants[best];
    const cmp =
      a.lmax - b.lmax ||
      cmpNiter(effectiveNiter(a), effectiveNiter(b)) ||
      solverRank(a.solver) - solverRank(b.solver) ||
      a.dtDiv - b.dtDiv;
    if (cmp > 0) best = i;
  }
  return best;
}

/** Infinity - Infinity is NaN, so iteration counts compare by sign. */
const cmpNiter = (a: number, b: number): number => (a === b ? 0 : a < b ? -1 : 1);

/** Distinguishable line/label colors, one per variant row. */
export const VARIANT_COLORS = [
  '#0969da', '#bf8700', '#1a7f37', '#cf222e', '#8250df', '#0f7c8a',
];
