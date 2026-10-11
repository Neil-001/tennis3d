// The served app with the shipped example gallery: every clip's included video, result and source times.
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../../', import.meta.url));
const examples = `${root}/examples/gallery-v2`;
const manifest = JSON.parse(await readFile(`${examples}/gallery.json`, 'utf8'));
// The first clip is checked at every frame and the others at a few. GALLERY_FRAME_STRIDE=1 checks every frame of all.
const stride = Number(process.env.GALLERY_FRAME_STRIDE || 0);
const work = await mkdtemp(join(tmpdir(), 'tennis3d-gallery-'));
const server = spawn('uv', ['run', '--locked', 'tennis3d', 'dashboard', '--no-open', '--port', '0',
  '--home', join(work, 'home'), '--gallery', `${examples}/gallery.json`], { cwd: root, stdio: ['ignore', 'pipe', 'inherit'] });
let browser;
try {
  const url = await new Promise((resolve, reject) => {
    let text = '';
    server.stdout.on('data', chunk => { text += chunk; const found = text.match(/http:\/\/127\.0\.0\.1:\d+\//); if (found) resolve(found[0]); });
    server.on('exit', code => reject(new Error(`App exited with ${code}: ${text}`)));
  });
  browser = await chromium.launch({ headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, acceptDownloads: true });
  const page = await context.newPage();
  const errors = [], external = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('request', r => { if (/^https?:/.test(r.url()) && !r.url().startsWith(url)) external.push(r.url()); });
  await page.goto(url);
  await page.locator('#gallery .run-card').first().waitFor();
  assert.equal(await page.locator('#gallery .run-card').count(), 20);
  assert.doesNotMatch(await page.locator('#gallery').innerText(), /no video|video preparing/);
  const video = page.locator('#video');
  const open = async index => {
    await page.locator('#gallery .run-card').nth(index).click();
    const id = manifest.clips[index].id;
    await page.waitForFunction(id => {
      const v = document.getElementById('video');
      return v.getAttribute('src')?.includes(id) && v.readyState >= 2 && document.getElementById('metadata').textContent.includes(id);
    }, id);
    assert.equal(await page.locator('#video-error').isVisible(), false);
  };
  // Seek the transport bar to a source frame and report the video frame the browser then presents.
  const present = time => page.evaluate(time => new Promise((resolve, reject) => {
    const v = document.getElementById('video'), bar = document.getElementById('time');
    const timer = setTimeout(() => reject(new Error(`No video frame was presented at ${time} s.`)), 15000);
    v.requestVideoFrameCallback((now, frame) => { clearTimeout(timer); resolve({ media: frame.mediaTime, bar: Number(bar.value), width: frame.width, height: frame.height }); });
    bar.value = String(time); bar.dispatchEvent(new Event('input', { bubbles: true }));
  }), time);
  let checked = 0, gaps = 0;
  for (const [index, clip] of manifest.clips.entries()) {
    const text = await readFile(`${examples}/${clip.id}/result.json`, 'utf8'), frames = JSON.parse(text).frames;
    await open(index);
    const last = frames.length - 1, gap = frames.findIndex(f => f.reconstruction === 'unavailable');
    const step = stride || (index ? Math.ceil(last / 5) : 1);
    const picks = new Set([last, ...Array.from({ length: Math.ceil(last / step) }, (_, i) => last - (i + 1) * step).filter(i => i > 0), 0]);
    if (gap > 0) picks.add(gap);
    // The browser reports frame times in whole microseconds.
    let previous = -1;
    for (const i of [...picks].sort((a, b) => a === 0 ? 1 : b === 0 ? -1 : b - a)) {
      const shown = await present(frames[i].t_s);
      assert.ok(Math.abs(shown.media - frames[i].t_s) < 2e-6 && Math.abs(shown.bar - frames[i].t_s) < 1e-9, `${clip.id} frame ${i}: video presented ${shown.media} s for source time ${frames[i].t_s} s.`);
      assert.deepEqual([shown.width, shown.height], [1920, 1080]);
      // The finished seek leaves the transport on the exact source frame time.
      assert.ok(await page.evaluate(time => new Promise(resolve => {
        const v = document.getElementById('video'), done = () => resolve(Math.abs(Number(document.getElementById('time').value) - time) < 1e-9);
        if (v.seeking) v.addEventListener('seeked', () => setTimeout(done), { once: true }); else done();
      }), frames[i].t_s), `${clip.id} frame ${i}: the transport left the source frame time.`);
      assert.match(await page.locator('#state').textContent(), new RegExp(`Frame ${i} at `));
      if (frames[i].reconstruction === 'unavailable') { assert.equal(await page.locator('#position').textContent(), 'XYZ unavailable'); gaps++; }
      assert.notEqual(i, previous); previous = i; checked++;
    }
    const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#export').click()]);
    assert.equal(await readFile(await download.path(), 'utf8'), text, `${clip.id}: exported result differs from the shipped file.`);
  }
  assert.ok(gaps > 0, 'At least one shipped clip must show a gap beside its video frame.');
  // Playing drives the 3D clock from the video and stops at the last source frame.
  const play = async () => {
    await page.getByRole('button', { name: 'Play', exact: true }).click();
    await page.waitForFunction(() => document.getElementById('video').currentTime > .3);
    const lag = await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => {
      const v = document.getElementById('video'); resolve(Math.abs(v.currentTime - Number(document.getElementById('time').value)));
    })));
    assert.ok(lag < .1, `The 3D clock is ${lag} s from the playing video.`);
    await page.getByRole('button', { name: 'Pause', exact: true }).click();
    assert.equal(await video.evaluate(v => v.paused), true);
  };
  // The last clip is at its first frame.
  await play();
  const end = Number(await page.locator('#time').getAttribute('max'));
  await present(end - .5);
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await page.waitForFunction(end => document.getElementById('play').textContent === 'Play' && Number(document.getElementById('time').value) === end, end);
  const range = await page.request.get(`${url}api/gallery/${manifest.clips[0].id}/video`, { headers: { Range: 'bytes=0-9' } });
  assert.equal(range.status(), 206);
  assert.equal(range.headers()['content-type'], 'video/mp4');
  // The wheel zooms the scene beside a shipped clip while the left button is held.
  const canvas = page.locator('#scene canvas'), box = await canvas.boundingBox();
  const distance = async () => { const [x, y, z] = (await canvas.getAttribute('data-camera')).split(' ').map(Number); return Math.hypot(x, y, z - .5); };
  const before = await distance(), cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + 40, cy + 10, { steps: 3 });
  await page.mouse.wheel(0, -120);
  await page.mouse.move(cx + 60, cy + 20, { steps: 2 });
  await page.mouse.wheel(0, -120);
  const held = await distance();
  assert.ok(Math.abs(held - before * .95 ** 2.4) < 1e-3 * before, `Held wheel steps must zoom before release: ${before} to ${held}.`);
  await page.mouse.up();
  await page.mouse.wheel(0, 240);
  assert.ok(Math.abs(await distance() - before) < 1e-3 * before, 'A released wheel step must zoom back.');
  // A phone-sized window plays the same clip without widening the page.
  await page.setViewportSize({ width: 390, height: 844 });
  await open(1);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  await video.scrollIntoViewIfNeeded();
  const size = await video.boundingBox();
  assert.ok(size.width > 300 && size.width <= 390 && size.height > 100);
  await play();
  assert.deepEqual(errors, []);
  assert.deepEqual(external, []);
  console.log(`Shipped gallery checks passed: 20 included videos, ${checked} seeks at exact source frame times including ${gaps} gaps, unchanged exports, desktop and mobile playback.`);
} finally {
  await browser?.close();
  server.kill();
  await rm(work, { recursive: true, force: true });
}
