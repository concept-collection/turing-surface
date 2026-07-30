/** Screenshot the demo page (dist/) in headless Chrome. Usage: node scripts/screenshot.mjs out.png [light|dark] */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import puppeteer from 'puppeteer-core';

const out = process.argv[2] ?? 'demo.png';
const scheme = process.argv[3] ?? 'light';
const minSteps = Number(process.argv[4] ?? 200);
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
await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: scheme }]);
page.on('console', (m) => console.log('  [page]', m.text()));
await page.goto(`http://127.0.0.1:${port}/index.html`, { waitUntil: 'load' });
// the sim starts paused; wait for setup to finish, then press Run
await page.waitForFunction(() => /grid/.test(document.getElementById('stats')?.textContent ?? ''), { timeout: 120_000 });
await page.click('#runpause');
await page.waitForFunction(
  (min) => {
    const s = document.getElementById('stats');
    const e = document.getElementById('err');
    const m = s && s.textContent.match(/\((\d+) steps\)/);
    return (m && Number(m[1]) >= min) || (e && e.textContent.length > 4);
  },
  { timeout: 600_000 },
  minSteps,
);
await new Promise((r) => setTimeout(r, 300));
await page.screenshot({ path: out });
console.log('screenshot:', out);
console.log('stats:', await page.$eval('#stats', (el) => el.textContent));
const err = await page.$eval('#err', (el) => el.textContent);
if (err) console.log('err:', err);
await browser.close();
server.close();
