/**
 * Headless GPU test runner: serves dist/, opens test.html in headless
 * Chrome (falling back to the SwiftShader software WebGPU adapter when no
 * hardware GPU is available), and reports the suite results.
 *
 * Run after `vite build`:  node scripts/test-gpu.mjs [--sweep]
 *
 * --sweep adds the niter x geometry sweep, which the page leaves out by
 * default because it is far too slow here to belong in CI: every session it
 * builds recompiles its whole unrolled step (445 kernels at niter 8), and
 * software WebGPU compiles those at about a second each. See
 * test/geometryChecks.ts.
 */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import puppeteer from 'puppeteer-core';

const DIST = new URL('../dist/', import.meta.url).pathname;
const CHROME = process.env.CHROME_PATH ?? '/usr/bin/google-chrome';
const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
};

const server = createServer(async (req, res) => {
  try {
    const path = req.url === '/' ? '/index.html' : req.url.split('?')[0];
    const data = await readFile(join(DIST, path));
    res.writeHead(200, { 'content-type': MIME[extname(path)] ?? 'application/octet-stream' });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end('not found');
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

const flagSets = [
  // hardware first, then SwiftShader (software) WebGPU
  ['--headless=new', '--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan'],
  ['--headless=new', '--no-sandbox', '--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader'],
];

const query = process.argv.includes('--sweep') ? '?sweep=1' : '';

let final = null;
for (const flags of flagSets) {
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    // A copy: puppeteer splices --enable-features out of the array it is given
    // and re-adds it merged with its own, which would drop it from the
    // diagnostic below and make a failure look like it ran with fewer flags.
    args: [...flags],
    // Puppeteer's default is 180 s, and it bounds the CDP call that
    // waitForFunction polls inside — so without this the wait below silently
    // caps at 3 minutes no matter what timeout it is given, and a suite that
    // runs longer fails as 'Runtime.callFunctionOn timed out' with no results.
    protocolTimeout: 900_000,
  });
  try {
    const page = await browser.newPage();
    page.on('console', (msg) => console.log(`  [page] ${msg.text()}`));
    page.on('pageerror', (err) => console.log(`  [pageerror] ${err.message}`));
    await page.goto(`http://127.0.0.1:${port}/test.html${query}`, { waitUntil: 'load' });
    const results = await page.waitForFunction(() => window.__RESULTS__, { timeout: 600_000 });
    final = await results.jsonValue();
  } catch (e) {
    console.error(`run with flags [${flags.join(' ')}] failed: ${e.message}`);
  } finally {
    await browser.close();
  }
  if (final && !final.fatal) break;
  console.log('retrying with next flag set…');
}
server.close();

if (!final || final.fatal) {
  console.error(`GPU tests could not run: ${final?.fatal ?? 'no results'}`);
  process.exit(2);
}
console.log(final.ok ? 'GPU SUITE: PASS' : 'GPU SUITE: FAIL');
process.exit(final.ok ? 0 : 1);
