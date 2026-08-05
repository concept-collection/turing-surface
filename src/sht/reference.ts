/**
 * Double-precision reference implementation of the scalar spherical
 * harmonic transform, by direct summation.  Slow (O(nlat*nlm) Legendre +
 * O(nlat*nphi*mmax) Fourier) but simple, and serves as ground truth for
 * validating the fp32 WebGPU implementation.
 *
 * Conventions are identical to the GPU path (see layout.ts).
 */
import { gaussNodesWeights } from './gauss.ts';
import { legendreCoeffs, legendreRow, type LegendreCoeffs } from './coeffs.ts';
import { alphaPlus, alphaMinus } from './derivCoeffs.ts';
import { lmIndex, nlmCalc, validateConfig, type ShtConfig } from './layout.ts';

export class ShtReference {
  readonly cfg: ShtConfig;
  readonly nlm: number;
  readonly ct: Float64Array;
  readonly st: Float64Array;
  readonly wg: Float64Array; // Gauss weights (for integral over cos(theta))
  readonly coeffs: LegendreCoeffs;

  constructor(cfg: ShtConfig) {
    validateConfig(cfg);
    this.cfg = cfg;
    this.nlm = nlmCalc(cfg.lmax, cfg.mmax);
    const { x, w } = gaussNodesWeights(cfg.nlat);
    this.ct = x;
    this.wg = w;
    this.st = new Float64Array(cfg.nlat);
    for (let i = 0; i < cfg.nlat; i++) this.st[i] = Math.sqrt(1 - x[i] * x[i]);
    this.coeffs = legendreCoeffs(cfg.lmax, cfg.mmax);
  }

  /**
   * Legendre stage of the synthesis: F_m(theta_i) = sum_l Q_lm ytilde_l^m(theta_i).
   * Returns complex array indexed [m * nlat + ilat], interleaved re/im.
   */
  legendreSynth(qlm: ArrayLike<number>): Float64Array {
    const { lmax, mmax, nlat } = this.cfg;
    const fm = new Float64Array(2 * (mmax + 1) * nlat);
    const row = new Float64Array(lmax + 1);
    for (let i = 0; i < nlat; i++) {
      for (let m = 0; m <= mmax; m++) {
        legendreRow(this.coeffs, lmax, m, this.ct[i], this.st[i], row);
        let re = 0, im = 0;
        const base = lmIndex(lmax, m, m);
        for (let l = m; l <= lmax; l++) {
          const y = row[l - m];
          re += y * qlm[2 * (base + l - m)];
          im += y * qlm[2 * (base + l - m) + 1];
        }
        const o = 2 * (m * nlat + i);
        fm[o] = re;
        fm[o + 1] = im;
      }
    }
    return fm;
  }

  /** Full synthesis: spectral -> spatial grid [ilat * nphi + iphi]. */
  synth(qlm: ArrayLike<number>): Float64Array {
    const { mmax, nlat, nphi } = this.cfg;
    const fm = this.legendreSynth(qlm);
    const spat = new Float64Array(nlat * nphi);
    for (let i = 0; i < nlat; i++) {
      for (let j = 0; j < nphi; j++) {
        const phi = (2 * Math.PI * j) / nphi;
        let v = fm[2 * (0 * nlat + i)]; // m=0: real part (imag must be 0)
        for (let m = 1; m <= mmax; m++) {
          const o = 2 * (m * nlat + i);
          const c = Math.cos(m * phi);
          const s = Math.sin(m * phi);
          v += 2 * (fm[o] * c - fm[o + 1] * s);
        }
        spat[i * nphi + j] = v;
      }
    }
    return spat;
  }

  /** Full analysis: spatial grid -> spectral coefficients (interleaved re/im). */
  analys(spat: ArrayLike<number>): Float64Array {
    const { lmax, mmax, nlat, nphi } = this.cfg;
    const qlm = new Float64Array(2 * this.nlm);
    const row = new Float64Array(lmax + 1);
    // forward Fourier: G_m(theta_i) = (2*pi/nphi) * sum_j f_ij e^{-i m phi_j}
    const gm = new Float64Array(2 * (mmax + 1) * nlat);
    for (let i = 0; i < nlat; i++) {
      for (let m = 0; m <= mmax; m++) {
        let re = 0, im = 0;
        for (let j = 0; j < nphi; j++) {
          const phi = (2 * Math.PI * j) / nphi;
          const f = spat[i * nphi + j];
          re += f * Math.cos(m * phi);
          im -= f * Math.sin(m * phi);
        }
        const o = 2 * (m * nlat + i);
        const norm = (2 * Math.PI) / nphi;
        gm[o] = re * norm;
        gm[o + 1] = im * norm;
      }
    }
    // Legendre stage with Gauss quadrature: Q_lm = sum_i w_i ytilde_l^m(theta_i) G_m(theta_i)
    for (let m = 0; m <= mmax; m++) {
      const base = lmIndex(lmax, m, m);
      for (let i = 0; i < nlat; i++) {
        legendreRow(this.coeffs, lmax, m, this.ct[i], this.st[i], row);
        const o = 2 * (m * nlat + i);
        const wr = this.wg[i] * gm[o];
        const wi = this.wg[i] * gm[o + 1];
        for (let l = m; l <= lmax; l++) {
          const y = row[l - m];
          qlm[2 * (base + l - m)] += y * wr;
          qlm[2 * (base + l - m) + 1] += y * wi;
        }
      }
    }
    return qlm;
  }

  /**
   * Coefficient-space theta shift, f64: v_l^m = alpha^+(l-1,m) u_{l-1}^m +
   * alpha^-(l+1,m) u_{l+1}^m (algos.tex eq. v_coeffs) — the coefficients of
   * sin(theta) * dtheta(u). The same shift serves the divergence side of the
   * six-transform Laplace-Beltrami scheme (step 5 of
   * docs/reduced-transforms.md), which is why it is exposed
   * apart from the synthesis.
   */
  dthetac(qlm: ArrayLike<number>): Float64Array {
    const { lmax, mmax } = this.cfg;
    const v = new Float64Array(2 * this.nlm);
    for (let m = 0; m <= mmax; m++) {
      for (let l = m; l <= lmax; l++) {
        const lm = lmIndex(lmax, l, m);
        let re = 0;
        let im = 0;
        if (l - 1 >= m) {
          const lm1 = lmIndex(lmax, l - 1, m);
          const a = alphaPlus(l - 1, m);
          re += a * qlm[2 * lm1];
          im += a * qlm[2 * lm1 + 1];
        }
        if (l + 1 <= lmax) {
          const lm2 = lmIndex(lmax, l + 1, m);
          const a = alphaMinus(l + 1, m);
          re += a * qlm[2 * lm2];
          im += a * qlm[2 * lm2 + 1];
        }
        v[2 * lm] = re;
        v[2 * lm + 1] = im;
      }
    }
    return v;
  }

  /** Coefficient-space phi derivative, f64: (dphi u)_l^m = i*m*u_l^m. */
  dphic(qlm: ArrayLike<number>): Float64Array {
    const { lmax, mmax } = this.cfg;
    const v = new Float64Array(2 * this.nlm);
    for (let m = 0; m <= mmax; m++) {
      for (let l = m; l <= lmax; l++) {
        const lm = lmIndex(lmax, l, m);
        v[2 * lm] = -m * qlm[2 * lm + 1];
        v[2 * lm + 1] = m * qlm[2 * lm];
      }
    }
    return v;
  }

  /** sin(theta) * dtheta(u) on the grid, f64: the undivided synthesis of the
   *  theta shift. Smooth on the sphere, unlike dtheta(u) itself. */
  sinDtheta(qlm: ArrayLike<number>): Float64Array {
    return this.synth(this.dthetac(qlm));
  }

  /** Theta-derivative, f64: synth(dthetac(u)) / sin(theta). */
  dtheta(qlm: ArrayLike<number>): Float64Array {
    const { nlat, nphi } = this.cfg;
    const grid = this.sinDtheta(qlm);
    for (let i = 0; i < nlat; i++) {
      const st = this.st[i];
      for (let j = 0; j < nphi; j++) grid[i * nphi + j] /= st;
    }
    return grid;
  }

  /** Phi-derivative, f64: (dphi u)_l^m = i*m*u_l^m, then synthesize. */
  dphi(qlm: ArrayLike<number>): Float64Array {
    return this.synth(this.dphic(qlm));
  }
}

/** Random band-limited spectrum for testing (m=0 imaginary parts zeroed). */
export function randomSpectrum(cfg: ShtConfig, seed = 12345): Float32Array {
  const nlm = nlmCalc(cfg.lmax, cfg.mmax);
  const q = new Float32Array(2 * nlm);
  let s = seed >>> 0;
  const rnd = () => {
    // xorshift32
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return (s / 4294967296) * 2 - 1;
  };
  for (let k = 0; k < 2 * nlm; k++) q[k] = rnd();
  for (let l = 0; l <= cfg.lmax; l++) q[2 * lmIndex(cfg.lmax, l, 0) + 1] = 0; // m=0 real
  return q;
}
