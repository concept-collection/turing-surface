/**
 * Which half of a transform is wrong on this device?
 *
 *   npx vite-node scripts/diagnose-sht.ts [--lmax 63] [--seed 12345]
 *
 * A transform is two stages, and both directions share code, so a single
 * pass/fail says very little:
 *
 *   synthesis:  qlm --[leg_synth]--> fm --[fft_synth | dft_synth]--> spat
 *   analysis:   spat --[fft_analys | dft_analys]--> fm --[leg_analys]--> qlm
 *
 * This reads the intermediate `fm` back out and compares each stage against
 * src/sht/reference.ts (f64, direct summation) on its own:
 *
 *  - fm wrong                        -> the Legendre stage
 *  - fm right but spat wrong         -> the Fourier stage
 *  - both right in DFT, wrong in FFT -> the WGSL FFT specifically
 *
 * and breaks the error down by m and by latitude, because "only high m" or "only
 * near the poles" points straight at the rescaled recurrence, while "every m
 * equally" points at indexing.
 *
 * Written for a report of `npm run test:node` failing on a GPU the transforms
 * have not run on before. Nothing here is a benchmark.
 */
import { ShtPlan, requestShtDevice, describeAdapter } from '../src/sht/sht.ts';
import { ShtReference, randomSpectrum } from '../src/sht/reference.ts';
import { gridForLmax, isPowerOfTwo, type ShtConfig } from '../src/sht/layout.ts';
import { fftThreads } from '../src/sht/wgsl/fourier.ts';
import { installWebGpu, errMsg, NO_ADAPTER_HINT } from './nodeWebGpu.ts';

const argv = process.argv.slice(2);
if (argv.includes('--help') || argv.includes('-h')) {
  console.log(`usage: npx vite-node scripts/diagnose-sht.ts [options]

  --lmax <n>      spherical harmonic truncation (default 63, the app's)
  --seed <n>      seed of the test spectrum (default 12345)
  --fourier <m>   only test this stage: fft | dft (default: both)
  --help

Compares each stage of each direction against the f64 CPU reference and says
which one is wrong. See the header of this file.`);
  process.exit(0);
}
const flag = (name: string, dflt: string): string => {
  const i = argv.indexOf(`--${name}`);
  if (i >= 0 && argv[i + 1] !== undefined) return argv[i + 1];
  const eq = argv.find((a) => a.startsWith(`--${name}=`));
  return eq ? eq.slice(name.length + 3) : dflt;
};
const lmax = Number(flag('lmax', '63'));
const seed = Number(flag('seed', '12345'));
const only = flag('fourier', '');

/** Relative L2 of a against b, both flat. */
function relL2(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < b.length; i++) {
    const d = (a[i] ?? NaN) - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return Math.sqrt(num / Math.max(den, 1e-300));
}

function anyNonFinite(a: ArrayLike<number>): boolean {
  for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) return true;
  return false;
}

/** The Fourier half of a synthesis, on the host, from whatever fm it is given.
 *  Mirrors ShtReference.synth's inner loop — so feeding it the GPU's own fm says
 *  what the Fourier stage should have produced from the input it actually had. */
function fourierSynth(cfg: ShtConfig, fm: ArrayLike<number>): Float64Array {
  const { mmax, nlat, nphi } = cfg;
  const spat = new Float64Array(nlat * nphi);
  for (let i = 0; i < nlat; i++) {
    for (let j = 0; j < nphi; j++) {
      const phi = (2 * Math.PI * j) / nphi;
      let v = fm[2 * i];
      for (let m = 1; m <= mmax; m++) {
        const o = 2 * (m * nlat + i);
        v += 2 * (fm[o] * Math.cos(m * phi) - fm[o + 1] * Math.sin(m * phi));
      }
      spat[i * nphi + j] = v;
    }
  }
  return spat;
}

/** Worst offender along one axis of the [m][ilat] complex fm array. */
function fmBreakdown(
  cfg: ShtConfig,
  got: ArrayLike<number>,
  want: ArrayLike<number>,
): { byM: { m: number; rel: number }[]; worstLat: { ilat: number; rel: number } } {
  const { mmax, nlat } = cfg;
  const byM: { m: number; rel: number }[] = [];
  const latNum = new Float64Array(nlat);
  const latDen = new Float64Array(nlat);
  for (let m = 0; m <= mmax; m++) {
    let num = 0;
    let den = 0;
    for (let i = 0; i < nlat; i++) {
      for (let c = 0; c < 2; c++) {
        const k = 2 * (m * nlat + i) + c;
        const d = (got[k] ?? NaN) - want[k];
        num += d * d;
        den += want[k] * want[k];
        latNum[i] += d * d;
        latDen[i] += want[k] * want[k];
      }
    }
    byM.push({ m, rel: Math.sqrt(num / Math.max(den, 1e-300)) });
  }
  let worstLat = { ilat: 0, rel: 0 };
  for (let i = 0; i < nlat; i++) {
    const rel = Math.sqrt(latNum[i] / Math.max(latDen[i], 1e-300));
    if (rel > worstLat.rel) worstLat = { ilat: i, rel };
  }
  return { byM, worstLat };
}

const OK = 1e-4; // fp32 through these transforms lands near 1e-6; 1e-4 is generous
const verdict = (rel: number): string => (rel < OK ? 'ok  ' : 'WRONG');

let device: GPUDevice | null = null;
try {
  const runtime = await installWebGpu();
  device = await requestShtDevice().catch((e: unknown) => {
    throw new Error(`${errMsg(e)}\n${NO_ADAPTER_HINT}`);
  });
  const adapter = await describeAdapter(device);

  const { nlat, nphi } = gridForLmax(lmax, 3);
  const cfg: ShtConfig = { lmax, mmax: lmax, nlat, nphi };
  const ref = new ShtReference(cfg);
  const qlm = randomSpectrum(cfg, seed);

  console.log('turing-surface — which stage of the transform is wrong?\n');
  console.log(`  device    ${adapter || '(unknown)'}\n            ${runtime}`);
  console.log(`  grid      lmax ${cfg.lmax} · ${nlat}×${nphi} · nlm ${ref.nlm}`);
  console.log(
    `  limits    maxComputeWorkgroupStorageSize ${device.limits.maxComputeWorkgroupStorageSize}` +
      `, maxComputeInvocationsPerWorkgroup ${device.limits.maxComputeInvocationsPerWorkgroup}`,
  );
  console.log(
    `  the FFT stage needs a power-of-two nphi (${isPowerOfTwo(nphi)}), ` +
      `16*nphi = ${16 * nphi} bytes of\n            workgroup storage and ` +
      `${fftThreads(nphi)} invocations per workgroup\n`,
  );

  // reference values, computed once
  const fmRef = ref.legendreSynth(qlm);
  const spatRef = ref.synth(qlm);
  const qlmRef = ref.analys(spatRef);

  const modes: ('fft' | 'dft')[] =
    only === 'fft' || only === 'dft' ? [only] : ['fft', 'dft'];
  const summary: string[] = [];

  for (const mode of modes) {
    let plan: ShtPlan | null = null;
    try {
      plan = await ShtPlan.create(device, cfg, { fourier: mode });
    } catch (e) {
      console.log(`${mode.toUpperCase()} stage: unavailable — ${errMsg(e)}\n`);
      continue;
    }

    const stageFm = device.createBuffer({
      size: 8 * (cfg.mmax + 1) * nlat,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const stageSpat = device.createBuffer({
      size: 4 * nlat * nphi,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const read = async (buf: GPUBuffer): Promise<Float32Array> => {
      await buf.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(buf.getMappedRange().slice(0));
      buf.unmap();
      return out;
    };

    // --- synthesis, stopping to look at fm on the way through ---
    device.queue.writeBuffer(plan.qlmIn, 0, qlm as Float32Array<ArrayBuffer>);
    const enc = device.createCommandEncoder();
    plan.encodeSynth(enc);
    enc.copyBufferToBuffer(plan.fmBuf, 0, stageFm, 0, 8 * (cfg.mmax + 1) * nlat);
    enc.copyBufferToBuffer(plan.spatBuf, 0, stageSpat, 0, 4 * nlat * nphi);
    device.queue.submit([enc.finish()]);
    const fmGpu = await read(stageFm);
    const spatGpu = await read(stageSpat);

    const legSynthRel = relL2(fmGpu, fmRef);
    const synthRel = relL2(spatGpu, spatRef);
    // The Fourier stage judged on its own input, not on the reference's: if the
    // Legendre stage is already wrong, comparing spat with spatRef only repeats
    // that. This asks whether the Fourier stage did the right thing with the fm
    // it was actually handed.
    const fourierRel = relL2(spatGpu, fourierSynth(cfg, fmGpu));

    // --- analysis, for contrast: same two stages, opposite order ---
    const spatIn = Float32Array.from(spatRef);
    device.queue.writeBuffer(plan.spatBuf, 0, spatIn as Float32Array<ArrayBuffer>);
    const enc2 = device.createCommandEncoder();
    plan.encodeAnalys(enc2);
    enc2.copyBufferToBuffer(plan.fmBuf, 0, stageFm, 0, 8 * (cfg.mmax + 1) * nlat);
    device.queue.submit([enc2.finish()]);
    const fmAnalysGpu = await read(stageFm);
    const qlmGpu = await plan.analys(spatIn);
    // forward Fourier of the reference field, in the reference's own normalization
    const gmRef = new Float64Array(2 * (cfg.mmax + 1) * nlat);
    for (let i = 0; i < nlat; i++) {
      for (let m = 0; m <= cfg.mmax; m++) {
        let re = 0;
        let im = 0;
        for (let j = 0; j < nphi; j++) {
          const phi = (2 * Math.PI * j) / nphi;
          re += spatRef[i * nphi + j] * Math.cos(m * phi);
          im -= spatRef[i * nphi + j] * Math.sin(m * phi);
        }
        gmRef[2 * (m * nlat + i)] = re;
        gmRef[2 * (m * nlat + i) + 1] = im;
      }
    }
    const analysFourierRel = relL2(fmAnalysGpu, gmRef);
    const analysRel = relL2(qlmGpu, qlmRef);

    console.log(`${mode.toUpperCase()} stage — plan chose ${plan.fourierMode.toUpperCase()}\n`);
    console.log(`  synthesis   qlm -> fm -> spat`);
    console.log(
      `    ${verdict(legSynthRel)}  leg_synth      fm vs f64 reference          ` +
        `${legSynthRel.toExponential(2)}${anyNonFinite(fmGpu) ? '   (has NaN/Inf)' : ''}`,
    );
    console.log(
      `    ${verdict(fourierRel)}  ${mode}_synth      spat vs host Fourier of that fm  ` +
        `${fourierRel.toExponential(2)}${anyNonFinite(spatGpu) ? '   (has NaN/Inf)' : ''}`,
    );
    console.log(
      `    ${verdict(synthRel)}  end to end     spat vs f64 reference        ` +
        `${synthRel.toExponential(2)}`,
    );
    console.log(`\n  analysis    spat -> fm -> qlm`);
    console.log(
      `    ${verdict(analysFourierRel)}  ${mode}_analys     fm vs f64 reference          ` +
        `${analysFourierRel.toExponential(2)}`,
    );
    console.log(
      `    ${verdict(analysRel)}  end to end     qlm vs f64 reference        ` +
        `${analysRel.toExponential(2)}`,
    );

    if (legSynthRel >= OK) {
      const { byM, worstLat } = fmBreakdown(cfg, fmGpu, fmRef);
      const bad = byM.filter((e) => e.rel >= OK);
      const good = byM.filter((e) => e.rel < OK);
      console.log(`\n  leg_synth is wrong. Where:`);
      console.log(
        `    ${bad.length} of ${byM.length} orders m are wrong` +
          (good.length
            ? `; the ones that are right are m = ${good.slice(0, 12).map((e) => e.m).join(', ')}` +
              (good.length > 12 ? ', ...' : '')
            : ' (all of them)'),
      );
      if (bad.length) {
        const first = bad[0];
        const worst = bad.reduce((a, b) => (b.rel > a.rel ? b : a));
        console.log(
          `    lowest wrong m = ${first.m} (${first.rel.toExponential(2)}), ` +
            `worst m = ${worst.m} (${worst.rel.toExponential(2)})`,
        );
      }
      const theta = (Math.acos(ref.ct[worstLat.ilat]) * 180) / Math.PI;
      console.log(
        `    worst latitude ilat = ${worstLat.ilat} of ${nlat} ` +
          `(theta = ${theta.toFixed(1)}°, sin(theta) = ` +
          `${ref.st[worstLat.ilat].toExponential(2)}), rel ${worstLat.rel.toExponential(2)}`,
      );
      console.log(
        `    If only high m are wrong, or only latitudes near the poles where\n` +
          `    sin(theta) is small, the rescaled seed (sinpow_rescaled in\n` +
          `    src/sht/wgsl/common.ts) is the place to look. If every m is wrong by\n` +
          `    a similar amount, it is indexing or the dispatch, not the recurrence.\n` +
          `    Either way, follow it term by term from here:\n` +
          `      npx vite-node scripts/diagnose-leg.ts --lmax ${lmax} --m 0`,
      );
    }
    console.log();

    summary.push(
      `${mode}: leg_synth ${verdict(legSynthRel).trim()}, ${mode}_synth ` +
        `${verdict(fourierRel).trim()}, ${mode}_analys ${verdict(analysFourierRel).trim()}, ` +
        `leg_analys ${verdict(analysRel).trim()}`,
    );

    stageFm.destroy();
    stageSpat.destroy();
    plan.destroy();
  }

  console.log('summary');
  for (const s of summary) console.log(`  ${s}`);
  device.destroy();
} catch (e) {
  device?.destroy();
  console.error(`diagnose-sht: ${errMsg(e)}`);
  process.exit(1);
}
