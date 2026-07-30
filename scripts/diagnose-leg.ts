/**
 * Where does leg_synth's recurrence go wrong?
 *
 *   npx vite-node scripts/diagnose-leg.ts [--lmax 63] [--m 0]
 *
 * Follow-up to scripts/diagnose-sht.ts, which narrows a bad transform down to
 * one shader. This reads that shader's recurrence out term by term.
 *
 * The trick is to probe the production shader rather than a copy of it: with
 * qlm set to a single 1 at coefficient (l0, m) and zero everywhere else,
 *
 *   fm[m][ilat] = sum_l Q_lm ytilde_l^m(theta_i) = ytilde_l0^m(theta_i)
 *
 * so one synthesis per l0 hands back exactly the recurrence value at that l, for
 * every latitude at once, computed by the same code the app runs. Sweeping l0
 * from m to lmax gives the whole sequence, and comparing with legendreRow (f64)
 * says which term first disagrees:
 *
 *  - wrong at l = m           -> the seed (amm, or sinpow_rescaled)
 *  - wrong at l = m+1         -> a_{m+1}^m, i.e. the ab buffer as the shader reads it
 *  - right until some l, then -> the two-at-a-time advance in the loop
 *    growing steadily
 *  - a constant wrong factor  -> a scale error, not an instability
 */
import { ShtPlan, requestShtDevice, describeAdapter } from '../src/sht/sht.ts';
import { ShtReference } from '../src/sht/reference.ts';
import { legendreRow } from '../src/sht/coeffs.ts';
import { gridForLmax, lmIndex, type ShtConfig } from '../src/sht/layout.ts';
import { installWebGpu, errMsg, NO_ADAPTER_HINT } from './nodeWebGpu.ts';

const argv = process.argv.slice(2);
if (argv.includes('--help') || argv.includes('-h')) {
  console.log(`usage: npx vite-node scripts/diagnose-leg.ts [options]

  --lmax <n>   spherical harmonic truncation (default 63, the app's)
  --m <n>      the order to follow (default 0, which needs no rescaling at all
                 and so isolates the plain recurrence)
  --lats <i,j> latitudes to sample (default 0,1,mid,last)
  --all        print every l, not just the interesting ones
  --help`);
  process.exit(0);
}
const flag = (name: string, dflt: string): string => {
  const i = argv.indexOf(`--${name}`);
  if (i >= 0 && argv[i + 1] !== undefined) return argv[i + 1];
  const eq = argv.find((a) => a.startsWith(`--${name}=`));
  return eq ? eq.slice(name.length + 3) : dflt;
};
const lmax = Number(flag('lmax', '63'));
const m = Number(flag('m', '0'));
const showAll = argv.includes('--all');

if (!Number.isInteger(m) || m < 0 || m > lmax) {
  console.error(`diagnose-leg: --m must be an integer in [0, lmax]`);
  process.exit(2);
}

let device: GPUDevice | null = null;
let plan: ShtPlan | null = null;
try {
  const runtime = await installWebGpu();
  device = await requestShtDevice().catch((e: unknown) => {
    throw new Error(`${errMsg(e)}\n${NO_ADAPTER_HINT}`);
  });
  const adapter = await describeAdapter(device);

  const { nlat, nphi } = gridForLmax(lmax, 3);
  const cfg: ShtConfig = { lmax, mmax: lmax, nlat, nphi };
  const ref = new ShtReference(cfg);
  // The Fourier stage plays no part here — only fm is read.
  plan = await ShtPlan.create(device, cfg, { fourier: 'dft' });

  const lats = flag('lats', '')
    ? flag('lats', '').split(',').map(Number)
    : [0, 1, nlat >> 1, nlat - 1];

  console.log('turing-surface — following leg_synth\'s recurrence term by term\n');
  console.log(`  device    ${adapter || '(unknown)'}\n            ${runtime}`);
  console.log(`  grid      lmax ${lmax} · ${nlat}×${nphi}`);
  console.log(`  order     m = ${m}${m === 0 ? '  (no rescaling: sinpow_rescaled returns 1, ny = 0)' : ''}`);
  console.log(
    `  latitudes ${lats
      .map((i) => `${i} (theta ${((Math.acos(ref.ct[i]) * 180) / Math.PI).toFixed(1)}°)`)
      .join(', ')}\n`,
  );

  const fmBytes = 8 * (cfg.mmax + 1) * nlat;
  const stageFm = device.createBuffer({
    size: fmBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const qlm = new Float32Array(2 * ref.nlm);
  const row = new Float64Array(lmax + 1);

  /** fm[m][ilat] after a synthesis of the unit spectrum at (l0, m). */
  const probe = async (l0: number): Promise<Float32Array> => {
    qlm.fill(0);
    qlm[2 * lmIndex(lmax, l0, m)] = 1;
    device!.queue.writeBuffer(plan!.qlmIn, 0, qlm as Float32Array<ArrayBuffer>);
    const enc = device!.createCommandEncoder();
    plan!.encodeSynth(enc);
    enc.copyBufferToBuffer(plan!.fmBuf, 0, stageFm, 0, fmBytes);
    device!.queue.submit([enc.finish()]);
    await stageFm.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(stageFm.getMappedRange().slice(0));
    stageFm.unmap();
    return out;
  };

  const rows: { l: number; rels: number[]; ratios: number[] }[] = [];
  for (let l0 = m; l0 <= lmax; l0++) {
    const fm = await probe(l0);
    const rels: number[] = [];
    const ratios: number[] = [];
    for (const ilat of lats) {
      legendreRow(ref.coeffs, lmax, m, ref.ct[ilat], ref.st[ilat], row);
      const want = row[l0 - m];
      const got = fm[2 * (m * nlat + ilat)];
      rels.push(Math.abs(got - want) / Math.max(Math.abs(want), 1e-300));
      ratios.push(want === 0 ? NaN : got / want);
    }
    rows.push({ l: l0, rels, ratios });
  }

  // Loose on purpose. A forward Legendre recurrence in fp32 loses relative
  // accuracy as it goes — by l = 63 a few 1e-5 is normal, and worse at the
  // latitudes where the terms nearly cancel. What we are hunting is a structural
  // error, which shows up as a ratio far from 1, not as a slow drift.
  const OK = 1e-2;
  const bad = (r: { rels: number[] }): boolean => r.rels.some((x) => !(x < OK));
  const firstBad = rows.find(bad);

  const head = `    l   ` + lats.map((i) => `ilat ${String(i).padStart(3)}`.padStart(14)).join('');
  console.log(head);
  console.log(`  ${'-'.repeat(head.length)}`);
  for (const r of rows) {
    // every l when --all; otherwise the seed, the first step, the first failure
    // and its neighbours, and a tail sample — enough to see the shape
    const near = firstBad ? Math.abs(r.l - firstBad.l) <= 3 : false;
    const interesting =
      showAll || r.l <= m + 2 || near || r.l >= lmax - 1 || (r.l - m) % 8 === 0;
    if (!interesting) continue;
    const cells = r.rels
      .map((rel, k) =>
        (rel < OK
          ? `ok ${rel.toExponential(1)}`
          : `${r.ratios[k] > 1e3 || r.ratios[k] < -1e3 ? '' : 'x'}${r.ratios[k].toExponential(2)}`
        ).padStart(14),
      )
      .join('');
    console.log(`  ${String(r.l).padStart(4)}  ${cells}${bad(r) ? '  <-- wrong' : ''}`);
  }
  console.log(
    `\n  cells are "ok <relative error>" when the term is right, and the ratio got/want\n` +
      `  when it is not. A few 1e-5 by l = ${lmax} is normal: an fp32 forward recurrence\n` +
      `  loses relative accuracy as it goes, worst where consecutive terms nearly cancel.`,
  );

  if (!firstBad) {
    console.log(`\n  Every term of the m = ${m} recurrence is right on this device.`);
    console.log(
      `  So the problem is not the recurrence itself — try another --m, or look at\n` +
        `  the accumulation into acc rather than the values going into it.`,
    );
  } else {
    const steps = Math.floor((firstBad.l - m) / 2);
    console.log(`\n  First wrong term: l = ${firstBad.l}, which is`);
    if (firstBad.l === m) {
      console.log(
        `    the seed itself — amm[m] or sinpow_rescaled, before any recurrence runs.`,
      );
    } else if (firstBad.l === m + 1) {
      console.log(
        `    y1's initializer, ab[base + 1].x * ct * y0 — so a_{m+1}^m as the shader\n` +
          `    reads it, or the very first multiply. No loop iteration has run yet.`,
      );
    } else {
      console.log(
        `    ${steps} advance${steps === 1 ? '' : 's'} into the loop (l = m + ${firstBad.l - m}).`,
      );
      const ok = rows.filter((r) => !bad(r)).map((r) => r.l);
      console.log(
        `    Terms that are right: l = ${ok.slice(0, 10).join(', ')}` +
          (ok.length > 10 ? ', ...' : ''),
      );
      const parity = new Set(ok.map((l) => (l - m) % 2));
      if (parity.size === 1) {
        console.log(
          `    Every correct term has (l - m) % 2 == ${[...parity][0]}, and every wrong one\n` +
            `    the other parity. The loop carries two values per iteration — y0 for even\n` +
            `    offsets and y1 for odd — so one of the two is being updated wrongly while\n` +
            `    the other is fine.`,
        );
      }
    }
  }

  stageFm.destroy();
  plan.destroy();
  device.destroy();
} catch (e) {
  plan?.destroy();
  device?.destroy();
  console.error(`diagnose-leg: ${errMsg(e)}`);
  process.exit(1);
}
