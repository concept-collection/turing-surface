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
import { DerivPlan } from '../src/sht/deriv.ts';
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

  // ---- f64 reference dtheta/dphi vs an independent closed form -----------
  // x(theta,phi) = sin(theta)*cos(phi) is exactly degree 1, so quadrature
  // recovers it to f64 round-off; comparing its dtheta/dphi against the
  // grid-space analytic derivatives (not derived from the same recurrence
  // being tested) catches a sign or indexing error the random-spectrum check
  // below, which compares two implementations of the same formula, would not.
  {
    const x = new Float64Array(nlat * nphi);
    for (let i = 0; i < nlat; i++) {
      const st = ref.st[i];
      for (let j = 0; j < nphi; j++) {
        const phi = (2 * Math.PI * j) / nphi;
        x[i * nphi + j] = st * Math.cos(phi);
      }
    }
    const X = ref.analys(x);
    const dThetaX = ref.dtheta(X);
    const dPhiX = ref.dphi(X);

    let errNum = 0;
    let norm = 0;
    for (let i = 0; i < nlat; i++) {
      const ct = ref.ct[i];
      const st = ref.st[i];
      for (let j = 0; j < nphi; j++) {
        const phi = (2 * Math.PI * j) / nphi;
        const k = i * nphi + j;
        const wantTheta = ct * Math.cos(phi);
        const wantPhi = -st * Math.sin(phi);
        errNum += (dThetaX[k] - wantTheta) ** 2 + (dPhiX[k] - wantPhi) ** 2;
        norm += wantTheta * wantTheta + wantPhi * wantPhi;
      }
    }
    const relErr = Math.sqrt(errNum / Math.max(norm, 1e-300));
    check(
      'deriv: f64 reference dtheta/dphi match the closed form on x = sin(theta)cos(phi)',
      relErr < 1e-6,
      `rel L2 error ${relErr.toExponential(2)}`,
    );
  }

  // ---- WGSL fp32 dtheta/dphi vs the (now closed-form-verified) f64 reference
  {
    const deriv = await DerivPlan.create(device, plan);

    const dThetaGpu = await deriv.dtheta(new Float32Array(q64));
    const dThetaCpu = ref.dtheta(q64);
    const errDtheta = relL2(dThetaGpu, dThetaCpu);

    const dPhiGpu = await deriv.dphi(new Float32Array(q64));
    const dPhiCpu = ref.dphi(q64);
    const errDphi = relL2(dPhiGpu, dPhiCpu);

    check(
      'deriv: WGSL fp32 dtheta/dphi vs f64 CPU reference',
      errDtheta < 1e-4 && errDphi < 1e-4,
      `dtheta ${errDtheta.toExponential(2)}, dphi ${errDphi.toExponential(2)}`,
    );
    deriv.destroy();
  }

  plan.destroy();
}
