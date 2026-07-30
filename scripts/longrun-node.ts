/**
 * Long-run sanity check: run Schnakenberg to t = 100 on desktop WebGPU and
 * confirm the pattern saturates into O(1)-contrast spots rather than decaying or
 * blowing up. Short runs cannot tell a growing instability from a diverging one.
 *
 *   vite-node scripts/longrun-node.ts [lmax]
 */
import { requestShtDevice } from '../src/sht/sht.ts';
import { ModelSession } from '../src/mgpu/session.ts';
import { mModelByKey, defaultParams } from '../src/mgpu/registry.ts';
import { installWebGpu, errMsg, NO_ADAPTER_HINT } from './nodeWebGpu.ts';

const lmax = Number(process.argv[2] ?? 31);
const model = mModelByKey('schnakenberg')!;
const params = defaultParams(model);

const runtime = await installWebGpu();
const device = await requestShtDevice().catch((e: unknown) => {
  throw new Error(`${errMsg(e)}\n${NO_ADAPTER_HINT}`);
});
const session = await ModelSession.create({ device, model, params, lmax });
session.seed(1);
console.log(`longrun — models/${model.key}.m at lmax ${lmax}, ${runtime}\n`);

const nsteps = Math.round(100 / params.dt);
const BATCH = 50;
const t0 = performance.now();
let lo = 0;
let hi = 0;
for (let s = 0; s < nsteps; s += BATCH) {
  session.step(Math.min(BATCH, nsteps - s));
  const u = await session.read(model.species[0]);
  lo = Infinity;
  hi = -Infinity;
  for (const v of u) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (session.steps % 400 === 0) {
    console.log(
      `t=${session.t.toFixed(1).padStart(5)}  u in [${lo.toFixed(4)}, ${hi.toFixed(4)}]  ` +
        `contrast ${(hi - lo).toFixed(4)}`,
    );
  }
}
const ms = (performance.now() - t0) / nsteps;

const contrast = hi - lo;
const saturated = Number.isFinite(contrast) && contrast > 0.5 && hi < 10;
console.log(
  `\n${saturated ? 'PASS' : 'FAIL'}  pattern saturated: contrast ${contrast.toFixed(4)} ` +
    `after ${session.steps} steps (${ms.toFixed(1)} ms/step)`,
);

session.destroy();
device.destroy();
process.exit(saturated ? 0 : 1);
