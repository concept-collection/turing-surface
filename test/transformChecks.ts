/**
 * The WGSL spherical-harmonic transforms against the f64 CPU reference.
 *
 * This is the one place a second implementation is still the right oracle: the
 * transforms are vendored shtns-webgpu, and `src/sht/reference.ts` is its direct-
 * summation f64 twin. Everything above them (the .m models) is checked against
 * closed-form answers instead — see analyticChecks.ts.
 */
import { ShtPlan } from '../src/sht/sht.ts';
import { ShtReference, randomSpectrum } from '../src/sht/reference.ts';
import { gridForLmax } from '../src/sht/layout.ts';
import type { Check, Log } from './analyticChecks.ts';

function relL2(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return Math.sqrt(num / Math.max(den, 1e-300));
}

export async function transformChecks(
  device: GPUDevice,
  check: Check,
  _log: Log,
): Promise<void> {
  const lmax = 31;
  const { nlat, nphi } = gridForLmax(lmax, 1);
  const cfg = { lmax, mmax: lmax, nlat, nphi };

  const plan = await ShtPlan.create(device, cfg);
  const ref = new ShtReference(cfg);

  const q = randomSpectrum(cfg, 42);
  const q64 = new Float64Array(q);

  const spatGpu = await plan.synth(new Float32Array(q64));
  const spatCpu = ref.synth(q64);
  const errSynth = relL2(spatGpu, spatCpu);

  const qGpu = await plan.analys(new Float32Array(spatCpu));
  const qCpu = ref.analys(new Float64Array(spatCpu));
  const errAnalys = relL2(qGpu, qCpu);

  check(
    'transforms: WGSL fp32 vs f64 CPU reference',
    errSynth < 1e-4 && errAnalys < 1e-4,
    `synth ${errSynth.toExponential(2)}, analys ${errAnalys.toExponential(2)}`,
  );

  plan.destroy();
}
