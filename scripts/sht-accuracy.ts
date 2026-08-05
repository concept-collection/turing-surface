/**
 * The fp32 transform round-trip floor, swept in lmax — the measurement
 * docs/reduced-transforms.md Sec 5b prescribes before and
 * after any change to summation order in the Legendre kernels.
 *
 *   npx vite-node scripts/sht-accuracy.ts [--lmax 63,127,255] [--seed 42]
 *
 * For a band-limited spectrum q, analys(synth(q)) = q exactly (Gauss
 * quadrature is exact for the band), so the relative round-trip error is the
 * transforms' own fp32 round-off with no reference implementation in the
 * loop. Sequential accumulation over l in the synthesis shows this floor
 * growing roughly linearly in lmax; pairwise/compensated accumulation shows
 * it near-flat. The per-degree profile says *where* the error lives (the
 * high-l coefficients are the ones the alpha shifts and the l(l+1)
 * eigenvalues amplify).
 */
import { ShtPlan, requestShtDevice, describeAdapter } from '../src/sht/sht.ts';
import { gridForLmax, lmIndex, nlmCalc } from '../src/sht/layout.ts';
import { randomSpectrum } from '../src/sht/reference.ts';
import { installWebGpu, errMsg, NO_ADAPTER_HINT } from './nodeWebGpu.ts';

const arg = (name: string): string | undefined => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : undefined;
};
const LMAXES = (arg('lmax') ?? '63,127,255').split(',').map(Number);
const SEED = Number(arg('seed') ?? 42);

let runtime: string;
try {
  runtime = await installWebGpu();
} catch (e) {
  console.error(`sht-accuracy: ${errMsg(e)}\n${NO_ADAPTER_HINT}`);
  process.exit(1);
}
const device = await requestShtDevice();
console.log(`sht-accuracy — ${runtime}, ${await describeAdapter(device)}\n`);
console.log('  lmax   grid        rel L2 roundtrip   worst degree (rel)');

for (const lmax of LMAXES) {
  const { nlat, nphi } = gridForLmax(lmax, 1);
  const cfg = { lmax, mmax: lmax, nlat, nphi };
  const plan = await ShtPlan.create(device, cfg);
  const q = randomSpectrum(cfg, SEED);

  const grid = await plan.synth(q);
  const back = await plan.analys(grid);

  // Overall relative L2, and the same per degree — errors concentrate in l.
  let num = 0;
  let den = 0;
  const nlm = nlmCalc(lmax, lmax);
  const errL = new Float64Array(lmax + 1);
  const magL = new Float64Array(lmax + 1);
  for (let m = 0; m <= lmax; m++) {
    for (let l = m; l <= lmax; l++) {
      const i = lmIndex(lmax, l, m);
      const dr = back[2 * i] - q[2 * i];
      const di = back[2 * i + 1] - q[2 * i + 1];
      const d2 = dr * dr + di * di;
      const m2 = q[2 * i] ** 2 + q[2 * i + 1] ** 2;
      num += d2;
      den += m2;
      errL[l] += d2;
      magL[l] += m2;
    }
  }
  let worstL = 0;
  let worstRel = 0;
  for (let l = 0; l <= lmax; l++) {
    const rel = Math.sqrt(errL[l] / Math.max(magL[l], 1e-300));
    if (rel > worstRel) {
      worstRel = rel;
      worstL = l;
    }
  }
  console.log(
    `  ${String(lmax).padEnd(6)} ${`${nlat}x${nphi}`.padEnd(11)} ` +
      `${Math.sqrt(num / den).toExponential(3).padEnd(18)} ` +
      `l=${worstL}: ${worstRel.toExponential(3)}   (${nlm} coefficients)`,
  );
  plan.destroy();
}
