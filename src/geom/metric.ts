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
