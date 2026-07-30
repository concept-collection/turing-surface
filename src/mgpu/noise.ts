/**
 * The seeded perturbation a model's `init` starts from.
 *
 * Host-side rather than in the .m, so a run is reproducible from an integer
 * seed and the same field can be handed to any model.
 */

/**
 * Seeded uniform deviates in [0, 1): mulberry32.
 *
 * Integer arithmetic and one division by 2^32, so any faithful port of it
 * produces bit-identical values — which is what lets the native benchmark under
 * bench/shtns/ seed the same run.
 */
export function makeRand(seed: number): () => number {
  let s = seed >>> 0;
  return (): number => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Seeded normal deviates: mulberry32 + Box-Muller. */
export function makeRandn(seed: number): () => number {
  const rand = makeRand(seed);
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    while (u === 0) u = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    const th = 2 * Math.PI * rand();
    spare = r * Math.sin(th);
    return r * Math.cos(th);
  };
}

/** `amp`-scaled normal deviates, one per grid point, in index order. */
export function seededNoise(npts: number, amp: number, seed: number): Float32Array {
  const randn = makeRandn(seed);
  const out = new Float32Array(npts);
  for (let i = 0; i < npts; i++) out[i] = amp * randn();
  return out;
}
