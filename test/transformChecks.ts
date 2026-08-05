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

    // The undivided theta derivative sin(theta)*dtheta(u) — the flux-form
    // Laplace-Beltrami scheme's step 1 and the flux-metric precompute's
    // input — is the same shuffle+synthesis with the divide skipped, so it
    // gets the same oracle.
    const sinDthetaGpu = await deriv.sinDtheta(new Float32Array(q64));
    const sinDthetaCpu = ref.sinDtheta(q64);
    const errSinDtheta = relL2(sinDthetaGpu, sinDthetaCpu);
    check(
      'deriv: WGSL fp32 sinDtheta (undivided) vs f64 CPU reference',
      errSinDtheta < 1e-4,
      `sinDtheta ${errSinDtheta.toExponential(2)}`,
    );
    deriv.destroy();
  }

  // ---- batched transforms reproduce the scalar transforms ------------------
  // A batch walks the Legendre recurrence once for K fields with per-lane
  // arithmetic textually identical to the scalar kernel's, so each lane must
  // agree with the scalar path to shader-compiler latitude (FMA contraction
  // may differ between the two modules; nothing else may).
  {
    const { nlat: gl, nphi: gp } = plan.cfg;
    const npts = gl * gp;
    const sizes = [];
    for (let k = 2; k <= plan.batchK; k += 2) sizes.push(k);
    check(
      'batch: plan compiled batched pipelines',
      plan.batchK >= 2,
      `batchK = ${plan.batchK} (${sizes.map((s) => `x${s}`).join(', ') || 'none'})`,
    );
    for (const K of sizes) {
      const qs = Array.from({ length: K }, (_, k) => randomSpectrum(cfg, 1000 + k));
      const qBufs = qs.map((q, k) => {
        const b = device.createBuffer({
          label: `batch-test-q${k}`,
          size: 8 * plan.nlm,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(b, 0, q as Float32Array<ArrayBuffer>);
        return b;
      });
      const spatBufs = qs.map((_, k) =>
        device.createBuffer({
          label: `batch-test-spat${k}`,
          size: 4 * npts,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        }),
      );
      const qOutBufs = qs.map((_, k) =>
        device.createBuffer({
          label: `batch-test-qout${k}`,
          size: 8 * plan.nlm,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        }),
      );
      const stage = device.createBuffer({
        label: 'batch-test-stage',
        size: K * (4 * npts + 8 * plan.nlm),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      // One pass: batched synthesis of all K, then batched analysis back.
      const synthB = plan.createSynthBatchBinding(
        qs.map((_, k) => ({ qlmIn: qBufs[k], spatOut: spatBufs[k] })),
      );
      const analysB = plan.createAnalysBatchBinding(
        qs.map((_, k) => ({ spatIn: spatBufs[k], qlmOut: qOutBufs[k] })),
      );
      const enc = device.createCommandEncoder({ label: 'batch-test' });
      const pass = enc.beginComputePass();
      plan.encodeSynthBatchInto(pass, synthB);
      plan.encodeAnalysBatchInto(pass, analysB);
      pass.end();
      for (let k = 0; k < K; k++) {
        enc.copyBufferToBuffer(spatBufs[k], 0, stage, k * 4 * npts, 4 * npts);
        enc.copyBufferToBuffer(qOutBufs[k], 0, stage, K * 4 * npts + k * 8 * plan.nlm, 8 * plan.nlm);
      }
      device.queue.submit([enc.finish()]);
      await stage.mapAsync(GPUMapMode.READ);
      const raw = new Float32Array(stage.getMappedRange().slice(0));
      stage.unmap();

      let worstSynth = 0;
      let worstAnalys = 0;
      for (let k = 0; k < K; k++) {
        const spatLane = raw.subarray(k * npts, (k + 1) * npts);
        const qLane = raw.subarray(K * npts + k * 2 * plan.nlm, K * npts + (k + 1) * 2 * plan.nlm);
        const spatScalar = await plan.synth(qs[k]);
        const qScalar = await plan.analys(spatScalar);
        worstSynth = Math.max(worstSynth, relL2(spatLane, spatScalar));
        worstAnalys = Math.max(worstAnalys, relL2(qLane, qScalar));
      }
      check(
        `batch: x${K} lanes match the scalar transforms`,
        worstSynth < 1e-6 && worstAnalys < 1e-6,
        `synth ${worstSynth.toExponential(2)}, analys ${worstAnalys.toExponential(2)} ` +
          `across ${K} lanes`,
      );
      for (const b of [...qBufs, ...spatBufs, ...qOutBufs, stage]) b.destroy();
    }
  }

  plan.destroy();
}
