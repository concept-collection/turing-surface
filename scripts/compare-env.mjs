/**
 * Is the browser computing the same thing as the terminal?
 *
 * Runs one identical spec in both — same model source, parameters, lmax, seed and
 * step count — and compares the final spectral state. The pipeline is
 * deterministic given that spec (seeded PRNG, then fixed arithmetic), so the two
 * should agree to fp32 round-off. They will not agree bit for bit: GPUs differ in
 * fused-multiply-add and other latitude fp32 allows. They should agree to far
 * better than any real difference in what is being computed.
 *
 * Both sides build their spec through the same parseArgs, so neither can quietly
 * use a different default.
 *
 *   node scripts/compare-env.mjs [--lmax 31] [--steps 200] [--preset schnak-spots]
 *
 * Requires `npm run build` first (it serves dist/), and desktop WebGPU for the
 * terminal side.
 */
import { createServer } from 'node:http';
import { readFile, unlink } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { extname, join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import puppeteer from 'puppeteer-core';

// ---- spec, defaulted small enough to be quick in a browser ----------------
const argv = process.argv.slice(2);
const flag = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  if (i >= 0 && argv[i + 1] !== undefined) return argv[i + 1];
  const eq = argv.find((a) => a.startsWith(`--${name}=`));
  return eq ? eq.slice(name.length + 3) : dflt;
};
const lmax = flag('lmax', '31');
const steps = flag('steps', '200');
const preset = flag('preset', 'schnak-spots');
const seed = flag('seed', '1');
const tolerance = Number(flag('tolerance', '2e-3'));

const statePath = join(tmpdir(), `turing-surface-desktop-${process.pid}.json`);

// ---- desktop -------------------------------------------------------------
console.log(`comparing environments — preset ${preset}, lmax ${lmax}, ${steps} steps, seed ${seed}\n`);
console.log('desktop (Dawn):');
const bench = spawnSync(
  'npx',
  [
    'vite-node', 'scripts/bench.ts',
    '--preset', preset, '--lmax', lmax, '--seed', seed,
    '--steps', steps, '--warmup', '10',
    '--dump-state', statePath,
  ],
  { encoding: 'utf8' },
);
if (bench.status !== 0) {
  console.error(bench.stdout ?? '');
  console.error(bench.stderr ?? '');
  console.error('compare-env: the desktop run failed');
  process.exit(1);
}
const desktop = JSON.parse(readFileSync(statePath, 'utf8'));
console.log(`  ${fmt(desktop.digest)}`);
console.log(`  adapter: ${desktop.digest.adapter}`);

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

let browserState = null;
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
    const url =
      `http://127.0.0.1:${port}/test.html?state=1&preset=${preset}` +
      `&lmax=${lmax}&seed=${seed}&steps=${steps}`;
    await page.goto(url, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__STATE__ !== undefined, { timeout: 180000 });
    browserState = await page.evaluate(() => window.__STATE__);
    await browser.close();
    break;
  } catch (e) {
    lastError = e.message ?? String(e);
    await browser?.close();
  }
}
server.close();
await unlink(statePath).catch(() => {});

if (!browserState) {
  console.error(`\ncompare-env: the browser run failed: ${lastError}`);
  process.exit(1);
}

console.log('\nbrowser:');
console.log(`  ${fmt(browserState.digest)}`);
console.log(`  adapter: ${browserState.digest.adapter}`);

// ---- compare -------------------------------------------------------------
const a = desktop.state;
const b = browserState.state;
if (a.length !== b.length) {
  console.error(`\nFAIL  different state sizes: ${a.length} vs ${b.length}`);
  process.exit(1);
}
let num = 0;
let den = 0;
let worst = 0;
for (let i = 0; i < a.length; i++) {
  const d = a[i] - b[i];
  num += d * d;
  den += b[i] * b[i];
  worst = Math.max(worst, Math.abs(d));
}
const rel = Math.sqrt(num / Math.max(den, 1e-300));

console.log('\ndifference:');
console.log(`  relative L2   ${rel.toExponential(3)}`);
console.log(`  worst element ${worst.toExponential(3)}`);
if (desktop.digest.fourier !== browserState.digest.fourier) {
  console.log(
    `  NOTE  different Fourier stage (${desktop.digest.fourier} vs ` +
      `${browserState.digest.fourier}) — those are different algorithms, so they ` +
      `round differently. That alone can explain a difference in the values.`,
  );
}

const ok = rel < tolerance;
console.log(
  `\n${ok ? 'PASS' : 'FAIL'}  the two environments compute the same thing ` +
    `(relative L2 ${rel.toExponential(2)}, tolerance ${tolerance.toExponential(1)})`,
);
process.exit(ok ? 0 : 1);

function fmt(d) {
  const g = (v) => v.toPrecision(9);
  return (
    `n=${d.n} min=${g(d.min)} max=${g(d.max)} mean=${g(d.mean)} rms=${g(d.rms)} ` +
    `fourier=${d.fourier}`
  );
}
