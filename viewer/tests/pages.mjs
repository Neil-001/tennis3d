// Exercise the real hosted examples under a GitHub Pages project-path prefix.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = fileURLToPath(new URL('../../', import.meta.url));
const site = resolve(root, '_site');
const manifest = JSON.parse(await readFile(resolve(root, 'examples/gallery-v2/gallery.json'), 'utf8'));
const html = await readFile(resolve(site, 'index.html'), 'utf8');
const payload = JSON.parse(html.match(/<script id="embedded-gallery" type="application\/json">(.*?)<\/script>/s)[1]);
assert.equal(payload.readOnly, true);
assert.equal(payload.clips.length, 20);
let rows = 0;
for (const [i, clip] of manifest.clips.entries()) {
  const text = await readFile(resolve(root, 'examples/gallery-v2', clip.result_path), 'utf8');
  assert.equal(payload.clips[i].resultText, text);
  rows += JSON.parse(text).frames.length;
  const media = await readFile(resolve(site, 'media', `${clip.id}.mp4`));
  assert.equal(createHash('sha256').update(media).digest('hex'), clip.playback_sha256);
}
assert.equal(rows, 10749);

const server = createServer(async (request, response) => {
  try {
    if (!['GET', 'HEAD'].includes(request.method)) { response.writeHead(405).end(); return; }
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    if (!pathname.startsWith('/tennis3d/')) { response.writeHead(404).end(); return; }
    const path = resolve(site, pathname.slice('/tennis3d/'.length) || 'index.html');
    if (!path.startsWith(`${site}/`)) { response.writeHead(404).end(); return; }
    const data = await readFile(path);
    const type = path.endsWith('.mp4') ? 'video/mp4' : 'text/html; charset=utf-8';
    const range = request.headers.range?.match(/^bytes=(\d+)-(\d*)$/);
    if (range) {
      const start = Number(range[1]), end = Math.min(data.length - 1, range[2] ? Number(range[2]) : data.length - 1);
      response.writeHead(206, { 'Content-Type': type, 'Accept-Ranges': 'bytes', 'Content-Range': `bytes ${start}-${end}/${data.length}`, 'Content-Length': end - start + 1 });
      response.end(request.method === 'HEAD' ? undefined : data.subarray(start, end + 1));
    } else {
      response.writeHead(200, { 'Content-Type': type, 'Content-Length': data.length });
      response.end(request.method === 'HEAD' ? undefined : data);
    }
  } catch { response.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const url = process.env.PAGES_URL ?? `http://127.0.0.1:${server.address().port}/tennis3d/`;
let browser;
try {
  browser = await chromium.launch({ headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    const context = await browser.newContext({ viewport, acceptDownloads: true });
    const page = await context.newPage(), errors = [], api = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => {
      if (request.url().includes('/api/') || !['GET', 'HEAD'].includes(request.method())) api.push(request.url());
    });
    await page.goto(url);
    assert.equal(await page.locator('#gallery .run-card').count(), 20);
    for (const id of ['launch', 'processing', 'file', 'video-file', 'demo', 'runs']) assert.equal(await page.locator(`#${id}`).isVisible(), false);
    assert.equal(await page.locator('#gallery .run-note').count(), 0);
    for (const [index, clip] of manifest.clips.entries()) {
      await page.locator('#gallery .run-card').nth(index).click();
      await page.waitForFunction(() => document.getElementById('video').readyState >= 2);
      assert.equal(await page.locator('#error').isVisible(), false);
      assert.equal(await page.locator('#video-error').isVisible(), false);
      const text = payload.clips[index].resultText, frames = JSON.parse(text).frames;
      const gap = frames.findIndex(frame => frame.reconstruction === 'unavailable');
      const available = frames.findIndex(frame => frame.xyz_m);
      for (const i of new Set([frames.length - 1, available, gap, 0].filter(i => i >= 0))) {
        const presented = await page.evaluate(time => new Promise((resolve, reject) => {
          const video = document.getElementById('video'), bar = document.getElementById('time');
          const timer = setTimeout(() => reject(new Error(`Video frame timeout at ${time}`)), 15000);
          video.requestVideoFrameCallback((now, frame) => { clearTimeout(timer); resolve(frame.mediaTime); });
          bar.value = String(time); bar.dispatchEvent(new Event('input'));
        }), frames[i].t_s);
        assert.ok(Math.abs(presented - frames[i].t_s) < 2e-6, `${clip.id}: video time ${presented} differs from source time ${frames[i].t_s}`);
        await page.waitForFunction(() => !document.getElementById('video').seeking);
        assert.match(await page.locator('#state').textContent(), new RegExp(`Frame ${i} at `));
        assert.equal(await page.locator('#position').textContent(), frames[i].xyz_m
          ? `X ${frames[i].xyz_m[0].toFixed(3)} · Y ${frames[i].xyz_m[1].toFixed(3)} · Z ${frames[i].xyz_m[2].toFixed(3)} m` : 'XYZ unavailable');
        assert.match(await page.locator('#source-timing').textContent(), new RegExp(`Source PTS ${payload.clips[index].timeline.frames[i].pts} × `));
      }
      const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#export').click()]);
      assert.equal(await readFile(await download.path(), 'utf8'), text);
    }
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({ path: process.env.PAGES_SCREENSHOT_DIR ? `${process.env.PAGES_SCREENSHOT_DIR}/${viewport.width}.png` : `${site}/qualification-${viewport.width}.png`, fullPage: true });
    assert.deepEqual(errors, []);
    assert.deepEqual(api, []);
    await context.close();
  }
  console.log('Pages dashboard passed: 40 example openings, 10,749 exact rows, source PTS, endpoint/gap/available-frame seeks, video decoding, exact exports, no API writes, desktop and mobile.');
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
