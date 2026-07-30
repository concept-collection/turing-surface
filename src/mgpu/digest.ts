/**
 * A run's final state, in a form two different machines can be compared on.
 *
 * The pipeline is deterministic given (model source, parameters, lmax, seed,
 * steps): the perturbation comes from a seeded PRNG, and everything after it is
 * fixed arithmetic. So the same spec run anywhere should land on the same state —
 * not bit for bit, since GPUs differ in fused-multiply-add and other latitude the
 * fp32 rules allow, but far closer than any real difference in what is being
 * computed would be.
 *
 * That makes a cross-environment comparison a genuine check that the browser and
 * the desktop are running the same computation, rather than something that merely
 * looks similar.
 */

export interface StateDigest {
  /** Element count, so a shape mismatch is caught before the values are read. */
  n: number;
  min: number;
  max: number;
  mean: number;
  /** Root mean square — sensitive to every element, unlike min/max. */
  rms: number;
  /** Which Fourier stage the transform plan chose. FFT and DFT are different
   *  algorithms and round differently, so a mismatch here explains a difference
   *  in the values rather than being a symptom of one. */
  fourier: 'fft' | 'dft';
  /** Informational: the GPU the numbers came from. */
  adapter: string;
}

export function digestOf(
  values: ArrayLike<number>,
  fourier: 'fft' | 'dft',
  adapter: string,
): StateDigest {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumsq = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    sumsq += v * v;
  }
  const n = values.length;
  return {
    n,
    min,
    max,
    mean: sum / n,
    rms: Math.sqrt(sumsq / n),
    fourier,
    adapter,
  };
}

/** Relative L2 difference of two states of equal length. */
export function relL2(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return Math.sqrt(num / Math.max(den, 1e-300));
}

export function formatDigest(d: StateDigest): string {
  const g = (v: number): string => v.toPrecision(9);
  return (
    `n=${d.n} min=${g(d.min)} max=${g(d.max)} mean=${g(d.mean)} rms=${g(d.rms)} ` +
    `fourier=${d.fourier}`
  );
}

/** Worst relative disagreement between two digests' scalar summaries. */
export function digestDrift(a: StateDigest, b: StateDigest): number {
  const rel = (x: number, y: number): number =>
    Math.abs(x - y) / Math.max(Math.abs(x), Math.abs(y), 1e-30);
  return Math.max(
    rel(a.min, b.min),
    rel(a.max, b.max),
    rel(a.mean, b.mean),
    rel(a.rms, b.rms),
  );
}
