/**
 * Why is the terminal faster than the browser?
 *
 * Measures the *same* solver work — same .m, same kernels, batched, nothing read
 * back, no rendering on either side — in the terminal (Dawn, in-process) and in a
 * real browser, and splits the result so the gap attributes itself:
 *
 *   node scripts/compare-perf.mjs [--lmax 63] [--steps 300] [--preset schnak-spots]
 *
 * The browser side runs `test.html?soak=`, which has no renderer at all. So:
 *
 *  - if the two agree, the solver is equally fast in the browser, and whatever
 *    the app shows on top of this is readback, rendering, and animation pacing.
 *  - if the browser is slower here, it is the GPU stack itself: submits crossing
 *    into the GPU process, or Metal/Vulkan execution differing between Chrome's
 *    Dawn and node-webgpu's.
 *
 * CPU command encoding is reported for both, because it is the one cost that can
 * make a fast GPU irrelevant — and it is usually *cheaper* in the browser, which
 * defers commands to the GPU process instead of validating them inline.
 *
 * Requires `npm run build` first, and desktop WebGPU for the terminal side.
 */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import { spawnSync } from 'node:child_process';
import puppeteer from 'puppeteer-core';

const argv = process.argv.slice(2);
const flag = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  if (i >= 0 && argv[i + 1] !== undefined) return argv[i + 1];
  const eq = argv.find((a) => a.startsWith(`--${name}=`));
  return eq ? eq.slice(name.length + 3) : dflt;
};
const lmax = flag('lmax', '63');
const steps = flag('steps', '300');
const preset = flag('preset', 'schnak-spots');

console.log(`comparing solver rate — preset ${preset}, lmax ${lmax}, ${steps} steps\n`);

// ---- terminal ------------------------------------------------------------
const bench = spawnSync(
  'npx',
  [
    'vite-node', 'scripts/bench.ts', '--json',
    '--preset', preset, '--lmax', lmax, '--steps', steps, '--warmup', '30',
  ],
  { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 },
);
if (bench.status !== 0) {
  console.error(bench.stdout ?? '');
  console.error(bench.stderr ?? '');
  console.error('compare-perf: the terminal run failed');
  process.exit(1);
}
const desktop = JSON.parse(bench.stdout);

// ---- browser -------------------------------------------------------------
const DIST = new URL('../dist/', import.meta.url).pathname;
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
const server = createServer(async (req, res) => {
  try {
    const path = req.url === '/' ? '/index.html' : req.url.split('?')[0];
    const data = await readFile(join(DIST, path));
    res.writeHead(200, {
      'content-type': MIME[extname(path)] ?? 'application/octet-stream',
    });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end('not found');
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

const flagSets = [
  ['--headless=new', '--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan'],
  ['--headless=new', '--no-sandbox', '--enable-unsafe-webgpu',
    '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader'],
];

let soak = null;
let lastError = '';
for (const args of flagSets) {
  let browser;
  try {
    browser = await puppeteer.launch({
      executablePath: process.env.CHROME_PATH ?? '/usr/bin/google-chrome',
      args,
    });
    const page = await browser.newPage();
    page.on('pageerror', (e) => (lastError = e.message));
    await page.goto(`http://127.0.0.1:${port}/test.html?soak=${steps}&lmax=${lmax}`, {
      waitUntil: 'load',
    });
    await page.waitForFunction(() => window.__SOAK__ !== undefined, { timeout: 600000 });
    soak = await page.evaluate(() => window.__SOAK__);
    await browser.close();
    break;
  } catch (e) {
    lastError = e.message ?? String(e);
    await browser?.close();
  }
}
server.close();

if (!soak) {
  console.error(`compare-perf: the browser run failed: ${lastError}`);
  process.exit(1);
}

// ---- report --------------------------------------------------------------
const d = desktop.throughput;
const row = (label, total, encode, adapter, fourier) => {
  console.log(`  ${label.padEnd(10)} ${total.toFixed(3)} ms/step` +
    `   encoding ${encode.toFixed(3)} ms/step (${((100 * encode) / total).toFixed(0)}%)` +
    `   ${fourier.toUpperCase()}`);
  console.log(`  ${''.padEnd(10)} ${adapter}`);
};
console.log('solver only, batched, nothing read back, no rendering:\n');
row('terminal', d.msPerStep, d.encodeMsPerStep, desktop.backend.adapter, desktop.digest?.fourier ?? 'fft');
row('browser', soak.solverMsPerStep, soak.encodeMsPerStep, soak.adapter, soak.fourier);

const ratio = soak.solverMsPerStep / d.msPerStep;
console.log(`\n  browser / terminal = ${ratio.toFixed(2)}x`);

// Before reading anything into the ratio: are these even the same GPU? A browser
// quietly falling back to a software adapter is a common cause of "the browser is
// much slower", and it makes the comparison meaningless rather than informative.
const software = (a) => /swiftshader|llvmpipe|software|basic render/i.test(a ?? '');
const desktopAdapter = desktop.backend.adapter ?? '';
if (software(soak.adapter) !== software(desktopAdapter)) {
  console.log(
    `\n  STOP  these are not the same device. One side is a software renderer:\n` +
      `    terminal: ${desktopAdapter}\n    browser:  ${soak.adapter}\n` +
      `  The ratio above compares different hardware and means nothing. If it is the\n` +
      `  browser that fell back, that IS the answer — check chrome://gpu for why\n` +
      `  (hardware acceleration disabled, or the GPU blocklisted).`,
  );
} else if (desktopAdapter && soak.adapter && desktopAdapter !== soak.adapter) {
  console.log(
    `\n  NOTE  the two report different adapters, which may just be different\n` +
      `  naming for the same GPU — but check it is not a second GPU:\n` +
      `    terminal: ${desktopAdapter}\n    browser:  ${soak.adapter}`,
  );
}

if (desktop.digest && desktop.digest.fourier !== soak.fourier) {
  console.log(
    `\n  NOTE  different Fourier stage (${desktop.digest.fourier} vs ${soak.fourier}).\n` +
      `  Those are different algorithms with different cost — that is the difference,\n` +
      `  not a symptom of it.`,
  );
} else if (ratio < 1.3) {
  console.log(
    `\n  The solver runs at the same rate in both. Anything the app shows beyond\n` +
      `  this is its readback per species, the colormapping, competing with the\n` +
      `  renderer for the GPU, and animation pacing — not the computation.`,
  );
} else {
  console.log(
    `\n  The browser is slower at the same solver work, with no renderer involved,\n` +
      `  so it is the GPU stack rather than anything above it: every submit crosses\n` +
      `  into the GPU process, and Chrome's Dawn and node-webgpu's need not compile\n` +
      `  or schedule these shaders identically. Note also that an animation-paced\n` +
      `  page can leave the GPU in a low-power state where a continuous benchmark\n` +
      `  boosts it; this soak hammers it continuously, so if the app is slower than\n` +
      `  this number, that is a likely reason.`,
  );
}
console.log(
  `\n  Correctness is a separate question: scripts/compare-env.mjs checks that the\n` +
    `  two environments compute the same state.`,
);
