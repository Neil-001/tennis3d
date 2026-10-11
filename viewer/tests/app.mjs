// The served app with a precomputed example and no model weights installed.
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { execFileSync, spawn } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../../', import.meta.url));
const work = await mkdtemp(join(tmpdir(), 'tennis3d-app-'));
const sha = async path => createHash('sha256').update(await readFile(path)).digest('hex');
const resultPath = `${root}/tests/fixtures/trajectory.json`, videoPath = join(work, 'clip.mp4');
execFileSync('uv', ['run', '--locked', 'python', '-c', `
import av, numpy as np
from fractions import Fraction
with av.open(${JSON.stringify(videoPath)}, 'w') as out:
    s = out.add_stream('libx264', rate=25); s.width, s.height, s.pix_fmt = 64, 48, 'yuv420p'
    s.codec_context.time_base = Fraction(1, 1000)
    for i in range(12):
        f = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 20, dtype=np.uint8), format='rgb24')
        f.pts, f.time_base = i * 40, Fraction(1, 1000)
        out.mux(s.encode(f))
    out.mux(s.encode())
`], { cwd: root });
// A second example ships its own VP9 display copy at 60000/1001 frames a second, which this browser can decode.
const timedVideo = join(work, 'timed.mp4'), timedResult = join(work, 'timed.json'), timed = JSON.parse(await readFile(resultPath, 'utf8'));
for (const [index, frame] of timed.frames.entries()) frame.t_s = index * 1001 / 60000;
await writeFile(timedResult, JSON.stringify(timed));
execFileSync('uv', ['run', '--locked', 'python', '-c', `
import av, numpy as np
from fractions import Fraction
with av.open(${JSON.stringify(timedVideo)}, 'w') as out:
    s = out.add_stream('libvpx-vp9', rate=Fraction(60000, 1001)); s.width, s.height, s.pix_fmt = 64, 48, 'yuv420p'
    s.codec_context.time_base = Fraction(1, 60000)
    for i in range(${timed.frames.length}):
        f = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 40, dtype=np.uint8), format='rgb24')
        f.pts, f.time_base = i * 1001, Fraction(1, 60000)
        out.mux(s.encode(f))
    out.mux(s.encode())
`], { cwd: root });
await writeFile(join(work, 'gallery.json'), JSON.stringify({ schema_version: 1, clips: [{
  id: 'example', title: 'Example <b>rally</b>', note: 'partial: 9 of 12 source frames have exported XYZ', source_path: videoPath, source_sha256: await sha(videoPath),
  result_path: resultPath, result_sha256: await sha(resultPath), rights: { redistribution: 'unknown' } }, {
  id: 'timed', title: 'Timed', playback_path: timedVideo, playback_sha256: await sha(timedVideo),
  result_path: timedResult, result_sha256: await sha(timedResult), rights: { redistribution: 'unknown' } }] }));
const server = spawn('uv', ['run', '--locked', 'tennis3d', 'dashboard', '--no-open', '--port', '0',
  '--home', join(work, 'home'), '--gallery', join(work, 'gallery.json')], { cwd: root, stdio: ['ignore', 'pipe', 'inherit'] });
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
  // Without weights the app explains what to do and refuses new videos.
  await page.getByText('Model weights are missing').waitFor();
  assert.match(await page.locator('#runtime').textContent(), /tennis3d setup/);
  assert.equal(await page.locator('#upload').isDisabled(), true);
  assert.equal(await page.locator('#connection').textContent(), 'Running on this computer');
  assert.equal(await page.locator('#device option[value="intel-gpu"]').count(), 0);
  // The menu reflects server capability; these UI fixtures do not qualify a GPU.
  let intelAvailable = true;
  await page.route('**/api/state', async route => {
    const response = await route.fetch();
    const state = await response.json();
    state.intel_gpu = { status: intelAvailable ? 'available' : 'unavailable' };
    await route.fulfill({ response, json: state });
  });
  await page.reload();
  await page.locator('#device option[value="intel-gpu"]').waitFor({ state: 'attached' });
  assert.equal(await page.locator('#device option[value="intel-gpu"]').textContent(), 'Intel GPU detection · CPU lifting');
  await page.locator('#device').selectOption('intel-gpu');
  intelAvailable = false;
  await page.waitForFunction(() => !document.querySelector('#device option[value="intel-gpu"]'));
  assert.equal(await page.locator('#device').inputValue(), 'auto');
  // Precomputed examples still open, with markup in titles shown as text.
  await page.locator('#gallery .run-card').first().waitFor();
  assert.equal(await page.locator('#gallery-count').textContent(), '2');
  assert.equal(await page.locator('#runs-count').textContent(), '');
  assert.equal(await page.locator('#gallery b').count(), 0);
  // The manifest's coverage note is not card text; the opened result reports coverage.
  assert.equal(await page.locator('#gallery .run-note').count(), 0);
  assert.doesNotMatch(await page.locator('#gallery').innerText(), /partial:/);
  await page.locator('#gallery .run-card').first().click();
  await page.waitForFunction(() => /% coverage/.test(document.getElementById('kind').textContent));
  const fixture = JSON.parse(await readFile(resultPath, 'utf8'));
  assert.match(await page.locator('#coverage').innerText(), new RegExp(`${fixture.frames.length}\\s+source frames`, 'i'));
  assert.equal(await page.locator('#processing').isVisible(), false);
  await page.locator('#frames-detail summary').click();
  assert.equal(await page.locator('#frame-rows tr').count(), fixture.frames.length);
  for (const state of ['estimated', 'inferred', 'unavailable']) {
    await page.locator('#frame-filter').selectOption(state);
    assert.equal(await page.locator('#frame-rows tr').count(), fixture.frames.filter(f => f.reconstruction === state).length);
  }
  await page.locator('#frame-filter').selectOption('all');
  const target = fixture.frames.at(-1);
  await page.locator('#frame-rows tr').last().locator('button').click();
  assert.match(await page.locator('#state').textContent(), new RegExp(`Frame ${target.frame_index} `));
  // The video copy is served with ranges; a browser without H.264 says so.
  await page.waitForFunction(() => document.getElementById('video').readyState > 0 || !document.getElementById('video-error').hidden);
  const range = await page.request.get(`${url}api/gallery/example/video`, { headers: { Range: 'bytes=0-9' } });
  assert.equal(range.status(), 206);
  const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#export').click()]);
  assert.equal(await readFile(await download.path(), 'utf8'), await readFile(resultPath, 'utf8'));
  // Seeking to a source frame presents that video frame and keeps the exact frame time on the transport.
  await page.locator('#gallery .run-card').nth(1).click();
  await page.waitForFunction(() => { const v = document.getElementById('video'); return v.getAttribute('src')?.includes('timed') && v.readyState >= 2; });
  const present = time => page.evaluate(time => new Promise(resolve => {
    const bar = document.getElementById('time');
    document.getElementById('video').requestVideoFrameCallback((now, frame) => resolve(frame.mediaTime));
    bar.value = String(time); bar.dispatchEvent(new Event('input', { bubbles: true }));
  }), time);
  for (const index of [5, 2, 4]) {
    assert.ok(Math.abs(await present(timed.frames[index].t_s) - timed.frames[index].t_s) < 2e-6, `Seeking to frame ${index} presented another video frame.`);
    assert.match(await page.locator('#state').textContent(), new RegExp(`Frame ${index} at `));
    await page.waitForFunction(() => !document.getElementById('video').seeking);
    assert.ok(Math.abs(Number(await page.locator('#time').inputValue()) - timed.frames[index].t_s) < 1e-9, 'A finished seek moved the transport off the source frame time.');
  }
  // Stepping back from a frame lands on the frame before it.
  await page.getByRole('button', { name: 'Previous source frame' }).click();
  assert.match(await page.locator('#state').textContent(), /Frame 3 at /);
  // A video that plays to its end leaves the transport on the last source frame.
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await page.waitForFunction(() => document.getElementById('video').ended);
  assert.ok(Math.abs(Number(await page.locator('#time').inputValue()) - timed.frames.at(-1).t_s) < 1e-9, 'The transport stopped before the last source frame.');
  assert.equal(await page.locator('#play').textContent(), 'Play');
  // The wheel zooms the scene while the left button is held and moving.
  const canvas = page.locator('#scene canvas'), box = await canvas.boundingBox();
  const distance = async () => { const [x, y, z] = (await canvas.getAttribute('data-camera')).split(' ').map(Number); return Math.hypot(x, y, z - .5); };
  const before = await distance(), cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + 40, cy + 10, { steps: 3 });
  await page.mouse.wheel(0, -120);
  await page.mouse.move(cx + 60, cy + 20, { steps: 2 });
  assert.ok(Math.abs(await distance() - before * .95 ** 1.2) < 1e-3 * before, 'A held wheel step must zoom before release.');
  await page.mouse.up();
  await page.mouse.wheel(0, 120);
  assert.ok(Math.abs(await distance() - before) < 1e-3 * before, 'A released wheel step must zoom back.');
  // Cross-site style requests are refused.
  assert.equal((await page.request.post(`${url}api/runs?name=a.mp4`, { data: 'x' })).status(), 403);
  intelAvailable = true;
  await page.waitForFunction(() => !!document.querySelector('#device option[value="intel-gpu"]'));
  await page.locator('#device').selectOption('intel-gpu');
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  assert.deepEqual(errors, []);
  assert.deepEqual(external, []);
  console.log('Served app checks passed: missing-weights guidance, precomputed example, frame inspection, export and request confinement.');
} finally {
  await browser?.close();
  server.kill();
  await rm(work, { recursive: true, force: true });
}
