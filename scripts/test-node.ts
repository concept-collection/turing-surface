/**
 * The whole suite on desktop WebGPU (Google Dawn), against the real pipeline:
 * MATLAB source -> numbl lowering -> generated WGSL -> GPU.
 *
 * The same four check modules run in the browser (test.html), so both GPU
 * stacks get the same guarantees. Run through vite-node, which is what resolves
 * numbl's compiler sources and the `?raw` model imports:
 *
 *   npm run test:node
 */
import { requestShtDevice } from '../src/sht/sht.ts';
import { installWebGpu, errMsg, NO_ADAPTER_HINT } from './nodeWebGpu.ts';
import { transformChecks } from '../test/transformChecks.ts';
import { analyticChecks } from '../test/analyticChecks.ts';
import { modelChecks } from '../test/modelChecks.ts';
import { geometryChecks } from '../test/geometryChecks.ts';
import { fluxChecks } from '../test/fluxChecks.ts';

let failures = 0;
const check = (name: string, ok: boolean, detail: string): void => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`);
  if (!ok) failures++;
};
const log = (s: string): void => console.log(s);

/**
 * These checks compile MATLAB to compute shaders, so unlike the old CPU-solver
 * suite they need a GPU. `--skip-without-gpu` lets a runner that has none say so
 * and move on — CI uses it, because the browser suite runs the very same check
 * modules on SwiftShader. A plain local run still fails loudly, so a missing GPU
 * is never mistaken for a pass.
 */
const skipWithoutGpu = process.argv.includes('--skip-without-gpu');

let runtime: string;
let device: GPUDevice;
try {
  runtime = await installWebGpu();
  device = await requestShtDevice();
} catch (e) {
  const detail = `${errMsg(e)}\n${NO_ADAPTER_HINT}`;
  if (skipWithoutGpu) {
    console.log(`SKIP  no WebGPU available here, so these checks did not run.\n${detail}`);
    process.exit(0);
  }
  console.error(`test-node: ${detail}`);
  process.exit(1);
}
console.log(`turing-surface tests — ${runtime}\n`);

await transformChecks(device, check, log);
await analyticChecks(device, check, log);
await modelChecks(device, check, log);
await geometryChecks(device, check, log);
await fluxChecks(device, check, log);

console.log(failures === 0 ? '\nAll tests passed.' : `\n${failures} failed.`);
process.exit(failures === 0 ? 0 : 1);
