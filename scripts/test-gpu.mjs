/**
 * Headless GPU test runner: serves dist/, opens test.html in headless
 * Chrome (falling back to the SwiftShader software WebGPU adapter when no
 * hardware GPU is available), and reports the suite results.
 *
 * Run after `vite build`:  node scripts/test-gpu.mjs
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

let final = null;
for (const flags of flagSets) {
  const browser = await puppeteer.launch({ executablePath: CHROME, args: flags });
  try {
    const page = await browser.newPage();
    page.on('console', (msg) => console.log(`  [page] ${msg.text()}`));
    page.on('pageerror', (err) => console.log(`  [pageerror] ${err.message}`));
    await page.goto(`http://127.0.0.1:${port}/test.html`, { waitUntil: 'load' });
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
