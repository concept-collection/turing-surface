/**
 * Inverse metric quantities V_theta, V_phi of a surface embedding X=(x,y,z)
 * (evolving_surface/notes/algos.tex Algorithm 2 / SurfaceDiffOperator.
 * _precompute_metric_quantities, clear_denominators=False branch): six grid
 * scalar fields depending only on the geometry, used by the surface
 * Laplace-Beltrami operator (Algorithm 3) to contract a field's theta/phi
 * derivatives into a tangential gradient/divergence.
 *
 *   g_tt = Xt.Xt,  g_tp = Xt.Xp,  g_pp = Xp.Xp     (first fundamental form)
 *   det  = g_tt*g_pp - g_tp^2
 *   V_theta = ( g_pp*Xt - g_tp*Xp ) / det
 *   V_phi   = ( g_tt*Xp - g_tp*Xt ) / det
 */

export interface MetricFields {
  /** V_theta, Cartesian components, npts each. */
  Vtx: Float32Array;
  Vty: Float32Array;
  Vtz: Float32Array;
  /** V_phi, Cartesian components, npts each. */
  Vpx: Float32Array;
  Vpy: Float32Array;
  Vpz: Float32Array;
}

/**
 * Xt/Xp (etc) are the theta/phi derivatives of each Cartesian embedding
 * component, grid space, npts each -- the tangent vectors X_theta, X_phi of
 * algos.tex Sec 4.1, one component per array.
 */
export function computeMetric(
  npts: number,
  Xt: Float32Array,
  Xp: Float32Array,
  Yt: Float32Array,
  Yp: Float32Array,
  Zt: Float32Array,
  Zp: Float32Array,
): MetricFields {
  const Vtx = new Float32Array(npts);
  const Vty = new Float32Array(npts);
  const Vtz = new Float32Array(npts);
  const Vpx = new Float32Array(npts);
  const Vpy = new Float32Array(npts);
  const Vpz = new Float32Array(npts);

  for (let i = 0; i < npts; i++) {
    const xt = Xt[i];
    const xp = Xp[i];
    const yt = Yt[i];
    const yp = Yp[i];
    const zt = Zt[i];
    const zp = Zp[i];

    const gtt = xt * xt + yt * yt + zt * zt;
    const gtp = xt * xp + yt * yp + zt * zp;
    const gpp = xp * xp + yp * yp + zp * zp;
    const det = gtt * gpp - gtp * gtp;

    Vtx[i] = (gpp * xt - gtp * xp) / det;
    Vty[i] = (gpp * yt - gtp * yp) / det;
    Vtz[i] = (gpp * zt - gtp * zp) / det;
    Vpx[i] = (gtt * xp - gtp * xt) / det;
    Vpy[i] = (gtt * yp - gtp * yt) / det;
    Vpz[i] = (gtt * zp - gtp * zt) / det;
  }

  return { Vtx, Vty, Vtz, Vpx, Vpy, Vpz };
}

/**
 * Flux-form metric weights p1, p2, q2, r of the six-transform Laplace-Beltrami
 * scheme (docs/reduced-transforms.md Sec 3). Built from the
 * *sin-weighted* theta tangent sin(theta)*X_theta — the undivided synthesis of
 * the alpha shift, DerivPlan.sinDtheta — and X_phi, both smooth on the sphere:
 *
 *   gtt~ = |sin(theta) X_theta|^2     (= sin^2(theta) g_tt)
 *   gtp~ = (sin(theta) X_theta).X_phi (= sin(theta)   g_tp)
 *   gpp  = |X_phi|^2
 *   D    = sqrt(gtt~ gpp - gtp~^2)    (= sin^2(theta) sqrt(det g) / sin(theta)
 *                                      = J sin^2(theta), with J = sqrt(det g)/sin(theta))
 *
 *   p1 = gpp / D,  p2 = -gtp~ / D,  q2 = gtt~ / D,  r = 1 / D.
 *
 * With these, for A = sin(theta) dtheta(u) and B = dphi(u), the two fluxes
 *
 *   P = p1*A + p2*B,   Qtilde = p2*A + q2*B
 *
 * equal sqrt(det g) g^{theta j} u_j and sin(theta) sqrt(det g) g^{phi j} u_j —
 * both smooth on the sphere — and Delta_Gamma u = r * (sin(theta) dtheta(P) +
 * dphi(Qtilde)). p1, p2, q2 are bounded (the sin^2 in D cancels against the
 * vanishing numerators); r ~ 1/sin^2(theta) is finite at the Gauss nodes and
 * is the scheme's one concentrated division (Sec 5 of the doc).
 *
 * All arithmetic is f64 (JS numbers) regardless of the input arrays' storage
 * type; results are rounded to f32 only on upload. That is the doc's "CPU
 * precompute in float64" mitigation, inherited for free.
 */
export interface FluxMetricFields {
  p1: Float64Array;
  p2: Float64Array;
  q2: Float64Array;
  r: Float64Array;
}

/**
 * sXt* are the Cartesian components of sin(theta)*X_theta, Xp* those of
 * X_phi, all grid space, npts each. No sin(theta) input is needed: every
 * division the scheme performs is by D, which the sin-weighted inputs build
 * directly.
 */
export function computeFluxMetric(
  npts: number,
  sXtx: ArrayLike<number>,
  sXty: ArrayLike<number>,
  sXtz: ArrayLike<number>,
  Xpx: ArrayLike<number>,
  Xpy: ArrayLike<number>,
  Xpz: ArrayLike<number>,
): FluxMetricFields {
  const p1 = new Float64Array(npts);
  const p2 = new Float64Array(npts);
  const q2 = new Float64Array(npts);
  const r = new Float64Array(npts);

  for (let i = 0; i < npts; i++) {
    const xt = sXtx[i];
    const yt = sXty[i];
    const zt = sXtz[i];
    const xp = Xpx[i];
    const yp = Xpy[i];
    const zp = Xpz[i];

    const gtt = xt * xt + yt * yt + zt * zt; // sin^2 g_tt
    const gtp = xt * xp + yt * yp + zt * zp; // sin   g_tp
    const gpp = xp * xp + yp * yp + zp * zp; //       g_pp
    const D = Math.sqrt(gtt * gpp - gtp * gtp); // J sin^2(theta)

    p1[i] = gpp / D;
    p2[i] = -gtp / D;
    q2[i] = gtt / D;
    r[i] = 1 / D;
  }

  return { p1, p2, q2, r };
}
