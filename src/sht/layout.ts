/**
 * Grid and spectral layout definitions, following SHTNS conventions:
 *
 * - Spectral coefficients Q_lm are complex, stored for m >= 0 only (real
 *   fields), interleaved [re, im], with SHTNS "m-major" ordering:
 *   for m = 0..mmax: for l = m..lmax.  Index of (l, m) is lm(l, m).
 * - Spatial fields are real, phi-contiguous: spat[ilat * nphi + iphi],
 *   with ilat ordered by increasing colatitude theta (north to south)
 *   and iphi covering [0, 2*pi) uniformly.
 * - Normalization: orthonormal spherical harmonics INCLUDING the
 *   Condon-Shortley phase (SHTNS default: sht_orthonormal).
 *   A real field is f = sum_{l,m>=0} Q_lm Y_lm + c.c.(m>0), i.e.
 *   Q_{l,-m} = (-1)^m conj(Q_lm) is implied.  m=0 coefficients must
 *   have zero imaginary part.
 */

export interface ShtConfig {
  lmax: number;
  mmax: number;
  nlat: number;
  nphi: number;
}

export function nlmCalc(lmax: number, mmax: number): number {
  // sum over m=0..mmax of (lmax - m + 1)
  return (mmax + 1) * (lmax + 1) - (mmax * (mmax + 1)) / 2;
}

/** Index of coefficient (l, m) in the spectral array (SHTNS LM ordering). */
export function lmIndex(lmax: number, l: number, m: number): number {
  return m * (lmax + 1) - (m * (m - 1)) / 2 + (l - m);
}

export function validateConfig(cfg: ShtConfig): void {
  const { lmax, mmax, nlat, nphi } = cfg;
  if (!Number.isInteger(lmax) || lmax < 1) throw new Error(`lmax must be an integer >= 1 (got ${lmax})`);
  if (!Number.isInteger(mmax) || mmax < 0 || mmax > lmax)
    throw new Error(`mmax must be an integer in [0, lmax] (got ${mmax})`);
  if (!Number.isInteger(nlat) || nlat <= lmax)
    throw new Error(`nlat must be an integer > lmax for exact Gauss quadrature (got nlat=${nlat}, lmax=${lmax})`);
  if (!Number.isInteger(nphi) || nphi < 2 * mmax + 1)
    throw new Error(`nphi must be an integer >= 2*mmax+1 to avoid aliasing (got nphi=${nphi}, mmax=${mmax})`);
}

export function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

/**
 * Re-index coefficients from one (lmax, mmax) layout into another, dropping
 * the modes the target cannot hold and zero-filling the ones the source does
 * not have. Both truncation and padding are exact spectral operations —
 * truncation is orthogonal projection onto the smaller band, padding changes
 * nothing — so a surface carried as coefficients can be moved onto any plan.
 */
export function relayoutCoeffs(
  q: Float32Array,
  from: { lmax: number; mmax: number },
  to: { lmax: number; mmax: number },
): Float32Array {
  if (from.lmax === to.lmax && from.mmax === to.mmax) return q;
  const out = new Float32Array(2 * nlmCalc(to.lmax, to.mmax));
  const mTop = Math.min(from.mmax, to.mmax);
  const lTop = Math.min(from.lmax, to.lmax);
  for (let m = 0; m <= mTop; m++) {
    for (let l = m; l <= lTop; l++) {
      const src = 2 * lmIndex(from.lmax, l, m);
      const dst = 2 * lmIndex(to.lmax, l, m);
      out[dst] = q[src];
      out[dst + 1] = q[src + 1];
    }
  }
  return out;
}

/** Grid sizes for a given lmax, dealiased for a reaction of polynomial degree
 *  `pdeg` (the rule from websph's reference implementation):
 *  nlat >= ((pdeg+1)*lmax+1)/2, nphi >= (pdeg+1)*lmax+1. nphi is rounded up to
 *  a power of two to keep the GPU FFT path. */
export function gridForLmax(lmax: number, pdeg: number): { nlat: number; nphi: number } {
  const minLat = Math.max(lmax + 1, ((pdeg + 1) * lmax + 1) / 2);
  const nlat = 2 * Math.ceil(minLat / 2);
  let nphi = 1;
  while (nphi < (pdeg + 1) * lmax + 1) nphi *= 2;
  return { nlat, nphi };
}
