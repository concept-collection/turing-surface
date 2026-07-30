/**
 * Gauss-Legendre quadrature nodes and weights, computed in double
 * precision by Newton iteration on P_n (cf. gauss_nodes() in SHTNS
 * sht_legendre.c).
 *
 * Returns nodes x_i = cos(theta_i) in DECREASING order (theta increasing,
 * north pole first), and weights w_i for integration over x in [-1, 1]:
 *   integral f(x) dx  ~=  sum_i w_i f(x_i),  exact for polynomials of
 *   degree <= 2n - 1.
 */
export function gaussNodesWeights(n: number): { x: Float64Array; w: Float64Array } {
  const x = new Float64Array(n);
  const w = new Float64Array(n);
  const m = (n + 1) >> 1;
  for (let i = 0; i < m; i++) {
    // initial guess (Tricomi-like), then Newton
    let z = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5));
    let pp = 0;
    for (let iter = 0; iter < 100; iter++) {
      // evaluate P_n(z) and P_{n-1}(z) by recurrence
      let p1 = 1.0;
      let p2 = 0.0;
      for (let j = 1; j <= n; j++) {
        const p3 = p2;
        p2 = p1;
        p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
      }
      pp = (n * (z * p1 - p2)) / (z * z - 1.0);
      const dz = p1 / pp;
      z -= dz;
      if (Math.abs(dz) < 1e-15 * Math.abs(z) + 1e-300) {
        // one extra iteration for full convergence
        let q1 = 1.0, q2 = 0.0;
        for (let j = 1; j <= n; j++) {
          const q3 = q2; q2 = q1;
          q1 = ((2 * j - 1) * z * q2 - (j - 1) * q3) / j;
        }
        pp = (n * (z * q1 - q2)) / (z * z - 1.0);
        z -= q1 / pp;
        break;
      }
    }
    x[i] = z; // largest roots first => theta increasing
    x[n - 1 - i] = -z;
    const wi = 2.0 / ((1.0 - z * z) * pp * pp);
    w[i] = wi;
    w[n - 1 - i] = wi;
  }
  if (n & 1) x[m - 1] = 0.0; // exact for odd n
  return { x, w };
}
