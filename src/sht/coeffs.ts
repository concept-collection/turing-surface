/**
 * Recurrence coefficients for orthonormal associated Legendre functions
 * ytilde_l^m(theta) (spherical-harmonic normalized, Condon-Shortley phase
 * included), matching SHTNS legendre_precomp() with norm=sht_orthonormal:
 *
 *   ytilde_m^m(theta)   = amm * sin(theta)^m
 *   ytilde_{m+1}^m      = a_{m+1}^m * cos(theta) * ytilde_m^m
 *   ytilde_l^m          = a_l^m * cos(theta) * ytilde_{l-1}^m + b_l^m * ytilde_{l-2}^m
 *
 * with (cf. sht_legendre.c lines 442-447):
 *   a_{m+1}^m = sqrt(2m+3)
 *   a_l^m     = sqrt( (2l+1)(2l-1) / ((l+m)(l-m)) )
 *   b_l^m     = -sqrt( (2l+1)/(2l-3) * ((l-1+m)(l-1-m)) / ((l+m)(l-m)) )
 *   amm       = cs^m * sqrt( 1/(4pi) * prod_{k=1..m} (2k+1)/(2k) )
 *
 * With this normalization, Y_lm(theta,phi) = ytilde_l^m(theta) e^{i m phi}
 * and integral |Y_lm|^2 dOmega = 1.
 */
import { lmIndex, nlmCalc } from './layout.ts';

export interface LegendreCoeffs {
  /** amm[m]: seed value (includes Condon-Shortley phase (-1)^m). */
  amm: Float64Array;
  /** ab[2*lm], ab[2*lm+1] = (a_l^m, b_l^m); entries at l=m unused (0), b at l=m+1 unused (0). */
  ab: Float64Array;
}

export function legendreCoeffs(lmax: number, mmax: number): LegendreCoeffs {
  const nlm = nlmCalc(lmax, mmax);
  const amm = new Float64Array(mmax + 1);
  const ab = new Float64Array(2 * nlm);

  let t = 1.0 / (4.0 * Math.PI);
  amm[0] = Math.sqrt(t);
  for (let m = 1; m <= mmax; m++) {
    t *= (2 * m + 1) / (2 * m);
    amm[m] = -Math.sqrt(t); // (-1)^m accumulates: Condon-Shortley phase
    if (m % 2 === 0) amm[m] = -amm[m];
  }

  for (let m = 0; m <= mmax; m++) {
    if (m + 1 <= lmax) {
      const lm = lmIndex(lmax, m + 1, m);
      ab[2 * lm] = Math.sqrt(2 * m + 3); // a_{m+1}^m
      ab[2 * lm + 1] = 0;
    }
    for (let l = m + 2; l <= lmax; l++) {
      const lm = lmIndex(lmax, l, m);
      const t1 = (l + m) * (l - m);
      const t2 = (l - 1 + m) * (l - 1 - m);
      ab[2 * lm] = Math.sqrt(((2 * l + 1) * (2 * l - 1)) / t1);
      ab[2 * lm + 1] = -Math.sqrt(((2 * l + 1) / (2 * l - 3)) * (t2 / t1));
    }
  }
  return { amm, ab };
}

/**
 * Evaluate ytilde_l^m(theta) for l = m..lmax at one point, in f64.
 * ct = cos(theta), st = sin(theta).  Plain (unscaled) recurrence: fine in
 * f64 for the moderate lmax this library targets (underflow of st^m only
 * matters for m of several hundred very close to the poles).
 */
export function legendreRow(
  coeffs: LegendreCoeffs,
  lmax: number,
  m: number,
  ct: number,
  st: number,
  out: Float64Array, // length lmax - m + 1
): void {
  let y0 = coeffs.amm[m] * Math.pow(st, m);
  out[0] = y0;
  if (m === lmax) return;
  const base = lmIndex(lmax, m, m);
  let y1 = coeffs.ab[2 * (base + 1)] * ct * y0;
  out[1] = y1;
  for (let l = m + 2; l <= lmax; l++) {
    const lm = base + (l - m);
    const y2 = coeffs.ab[2 * lm] * ct * y1 + coeffs.ab[2 * lm + 1] * y0;
    y0 = y1;
    y1 = y2;
    out[l - m] = y2;
  }
}
