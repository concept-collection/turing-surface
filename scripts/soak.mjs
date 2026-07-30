/**
 * Soak test: drive the demo page for many steps and report JS heap growth and
 * any crash, distinguishing a page crash from a renderer/driver death.
 *
 * Usage: node scripts/soak.mjs [steps] [lmax] [backend]
 *   e.g. node scripts/soak.mjs 1500 63 webgpu
 */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import puppeteer from 'puppeteer-core';

const steps = Number(process.argv[2] ?? 1000);
const lmax = process.argv[3] ?? '63';
const backend = process.argv[4] ?? 'webgpu';
const DIST = new URL('../dist/', import.meta.url).pathname;
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };

const server = createServer(async (req, res) => {
  try {
    const path = req.url === '/' ? '/index.html' : req.url.split('?')[0];
    const data = await readFile(join(DIST, path));
    res.writeHead(200, { 'content-type': MIME[extname(path)] ?? 'application/octet-stream' });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end();
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH ?? '/usr/bin/google-chrome',
  args: ['--headless=new', '--no-sandbox', '--enable-unsafe-webgpu',
    '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1000, height: 900 });

let crashed = null;
page.on('error', (e) => { crashed = `page crash: ${e.message}`; });
page.on('pageerror', (e) => { crashed = `page error: ${e.message}`; });
page.on('console', (m) => {
  const t = m.text();
  if (!/GL Driver Message|Failed to load resource/.test(t)) console.log('  [page]', t);
});

await page.goto(`http://127.0.0.1:${port}/index.html`, { waitUntil: 'load' });
await page.waitForFunction(() => /grid/.test(document.getElementById('stats')?.textContent ?? ''), { timeout: 120_000 });
await page.select('#lmax', lmax);
await page.select('#backend', backend);
await page.waitForFunction(() => /grid/.test(document.getElementById('stats')?.textContent ?? ''), { timeout: 120_000 });
await page.click('#runpause');

const readStep = () =>
  page.evaluate(() => {
    const m = document.getElementById('stats')?.textContent?.match(/\((\d+) steps\)/);
    return m ? Number(m[1]) : 0;
  });
const heapMB = async () => {
  const m = await page.metrics();
  return (m.JSHeapUsedSize / 1048576).toFixed(1);
};

const t0 = Date.now();
let last = 0;
let stalls = 0;
try {
  while (last < steps) {
    await new Promise((r) => setTimeout(r, 5000));
    if (crashed) throw new Error(crashed);
    const now = await readStep();
    console.log(`  step ${now}  heap ${await heapMB()} MB  (+${now - last} in 5s)`);
    if (now === last) {
      if (++stalls >= 6) throw new Error(`stalled at step ${now}`);
    } else stalls = 0;
    last = now;
  }
  console.log(`SOAK PASS: ${last} steps in ${((Date.now() - t0) / 1000).toFixed(0)}s, heap ${await heapMB()} MB`);
} catch (e) {
  console.error(`SOAK FAIL at step ${last}: ${e.message}`);
  process.exitCode = 1;
} finally {
  await browser.close().catch(() => {});
  server.close();
}
