// A private gallery build must preserve bytes and source timing under file://.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';
import { embeddedGallery } from '../../scripts/embedded_gallery.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const work = await mkdtemp(join(tmpdir(), 'tennis3d-offline-gallery-'));
const sha = text => createHash('sha256').update(text).digest('hex');
let browser;
try {
  const video = 'hashed video placeholder; no codec is exercised by this fixture';
  const data = JSON.parse(await readFile(`${root}/tests/fixtures/trajectory.json`, 'utf8'));
  data.source.sha256 = sha(video);
  data.source.time_base = [1, 1000000];
  data.source.name = '</script><img src=x onerror="window.injected=true">';
  const resultText = JSON.stringify(data, null, 2) + '\n';
  const timeline = { source_sha256: sha(video), endpoint_rule: 'Fixture exact PTS', frames: data.frames.map((f, index) => ({
    frame_index: index, original_frame_index: 100 + index, pts: Math.round((data.source.time_origin_s + f.t_s) * 1000000), time_base: [1, 1000000] })) };
  const timelineText = JSON.stringify(timeline);
  for (const [name, text] of [['source.mp4', video], ['result.json', resultText], ['timeline.json', timelineText]]) await writeFile(join(work, name), text);
  const clip = { id: 'example-1', title: '<b>Source example</b>', note: 'Raw and exported gaps remain distinct.', source_path: 'source.mp4', source_sha256: sha(video),
    result_path: 'result.json', result_sha256: sha(resultText), timeline_path: 'timeline.json', timeline_sha256: sha(timelineText) };
  const manifestPath = join(work, 'gallery.json'), output = join(work, 'offline.html');
  await writeFile(manifestPath, JSON.stringify({ schema_version: 1, clips: [clip] }));
  assert.equal((await embeddedGallery(manifestPath, () => true)).clips[0].resultText, resultText);
  await writeFile(join(work, 'source.mp4'), 'changed');
  await assert.rejects(embeddedGallery(manifestPath, () => true), /SHA-256 mismatch/);
  await writeFile(join(work, 'source.mp4'), video);
  const changed = structuredClone(timeline); changed.frames.at(-1).pts += 1;
  const changedText = JSON.stringify(changed);
  await writeFile(join(work, 'timeline.json'), changedText);
  await writeFile(manifestPath, JSON.stringify({ schema_version: 1, clips: [{ ...clip, timeline_sha256: sha(changedText) }] }));
  await assert.rejects(embeddedGallery(manifestPath, () => true), /Source PTS does not match/);
  for (const mismatch of ['origin', 'time_base']) {
    const changed = structuredClone(timeline);
    for (const row of changed.frames) {
      if (mismatch === 'origin') row.pts += 1000000;
      else { row.pts *= 2; row.time_base = [1, 2000000]; }
    }
    const text = JSON.stringify(changed);
    await writeFile(join(work, 'timeline.json'), text);
    await writeFile(manifestPath, JSON.stringify({ schema_version: 1, clips: [{ ...clip, timeline_sha256: sha(text) }] }));
    await assert.rejects(embeddedGallery(manifestPath, () => true), /Source timeline origin or time base/);
  }
  await writeFile(join(work, 'timeline.json'), timelineText);
  await writeFile(manifestPath, JSON.stringify({ schema_version: 1, clips: [clip] }));
  execFileSync('node', ['scripts/build_viewer.mjs', '--gallery', manifestPath, '--output', output], { cwd: root });
  browser = await chromium.launch({ headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const context = await browser.newContext({ offline: true, acceptDownloads: true, viewport: { width: 1280, height: 900 } });
  const page = await context.newPage(), errors = [], requests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (/^https?:/.test(request.url())) requests.push(request.url()); });
  await page.goto(pathToFileURL(output).href);
  assert.equal(await page.locator('#demo').isVisible(), false);
  assert.equal(await page.locator('#saved-runs-heading').isVisible(), false);
  await page.locator('#gallery .run-card').click();
  assert.equal(await page.locator('#gallery b').count(), 0);
  assert.equal(await page.locator('#gallery .run-note').count(), 0);
  assert.doesNotMatch(await readFile(output, 'utf8'), /Raw and exported gaps remain distinct/);
  assert.equal(await page.evaluate(() => window.injected), undefined);
  assert.equal(await page.locator('#error').isVisible(), false);
  for (const [index, row] of data.frames.entries()) {
    await page.locator('#time').evaluate((el, t) => { el.value = String(t); el.dispatchEvent(new Event('input')); }, row.t_s);
    assert.match(await page.locator('#state').textContent(), new RegExp(`Frame ${index} `));
    assert.match(await page.locator('#source-timing').textContent(), new RegExp(`Source PTS ${timeline.frames[index].pts} × 1/1000000 s`));
    if (!row.xyz_m) assert.equal(await page.locator('#position').textContent(), 'XYZ unavailable');
  }
  const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#export').click()]);
  assert.equal(await readFile(await download.path(), 'utf8'), resultText);
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  assert.deepEqual(errors, []); assert.deepEqual(requests, []);
  console.log('Offline embedded gallery passed: source hash/PTS checks, safe titles, exact export bytes, mobile layout and no network requests.');
} finally {
  await browser?.close();
  await rm(work, { recursive: true, force: true });
}
