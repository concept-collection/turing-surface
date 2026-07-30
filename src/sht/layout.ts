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
