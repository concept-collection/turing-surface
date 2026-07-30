/**
 * Smoke-check a deployed URL in headless Chrome: load it, press Run, and
 * confirm the solver actually advances.  Usage: node scripts/check-live.mjs [url]
 */
import puppeteer from 'puppeteer-core';

const url = process.argv[2] ?? 'https://concept-collection.github.io/turing-surface/';
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH ?? '/usr/bin/google-chrome',
  args: ['--headless=new', '--no-sandbox', '--enable-unsafe-webgpu',
    '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1000, height: 900 });
const problems = [];
page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
page.on('requestfailed', (r) => problems.push(`request failed: ${r.url()}`));
page.on('console', (m) => {
  if (m.type() === 'error' && !/GL Driver|favicon/.test(m.text())) {
    problems.push(`console error: ${m.text()}`);
  }
});

try {
  await page.goto(url, { waitUntil: 'load', timeout: 60_000 });
  await page.waitForFunction(
    () => /grid/.test(document.getElementById('stats')?.textContent ?? ''),
    { timeout: 120_000 },
  );
  console.log('initial:', await page.$eval('#stats', (el) => el.textContent));
  await page.click('#runpause');
  await page.waitForFunction(
    () => {
      const m = document.getElementById('stats')?.textContent?.match(/\((\d+) steps\)/);
      return m && Number(m[1]) >= 20;
    },
    { timeout: 180_000 },
  );
  console.log('running:', await page.$eval('#stats', (el) => el.textContent));
  const panels = await page.$$eval('.sphere-box canvas', (els) => els.length);
  console.log('sphere canvases:', panels);
  if (problems.length) {
    console.log('PROBLEMS:');
    for (const p of new Set(problems)) console.log('  ' + p);
    process.exitCode = 1;
  } else {
    console.log('LIVE CHECK: PASS');
  }
} catch (e) {
  console.error(`LIVE CHECK FAIL: ${e.message}`);
  for (const p of new Set(problems)) console.error('  ' + p);
  process.exitCode = 1;
} finally {
  await browser.close();
}
