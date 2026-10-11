// Independent campaigns deliberately reuse segment and trial IDs.
import assert from 'node:assert/strict';
import { execFileSync, spawn } from 'node:child_process';
import { mkdir, mkdtemp, readFile, readdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const artifacts = process.env.DASHBOARD_ARTIFACTS || path.join(root, 'viewer/test-artifacts');
await mkdir(artifacts, { recursive: true });
const runs = await mkdtemp(path.join(artifacts, 'dashboard-multirun-'));
const current = path.join(runs, 'current'), archive = path.join(runs, 'archive');
const python = process.env.DASHBOARD_PYTHON || path.join(root, '.venv/bin/python');
const env = { ...process.env, PYTHONPATH: root };
for (const [run, phase] of [[current, 'multirun-current'], [archive, 'multirun-archive']]) {
  execFileSync(python, ['tests/dashboard_fixture.py', run, phase], { cwd: root, env });
}
const files = async () => {
  const result = {};
  for (const entry of await readdir(runs, { recursive: true, withFileTypes: true })) {
    if (entry.isFile()) {
      const filename = path.join(entry.parentPath || entry.path, entry.name);
      result[path.relative(runs, filename)] = (await readFile(filename)).toString('base64');
    }
  }
  return result;
};
const before = await files();
const server = spawn(python, ['-m', 'tennis3d', 'dashboard', '--run', current, '--archive-run', archive, '--archive-run', archive, '--port', '0'], { cwd: root, env, stdio: ['ignore', 'pipe', 'pipe'] });
const base = await new Promise((resolve, reject) => {
  const timeout = setTimeout(() => reject(Error('dashboard server did not start')), 10000);
  server.stdout.on('data', data => {
    const match = data.toString().match(/http:\/\/127\.0\.0\.1:\d+/);
    if (match) { clearTimeout(timeout); resolve(match[0]); }
  });
  server.on('exit', code => { clearTimeout(timeout); reject(Error(`dashboard exited ${code}`)); });
});
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });
  const errors = [], requests = [];
  page.on('pageerror', error => errors.push(error.stack || error.message));
  page.on('request', request => requests.push(request.url()));
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector('#segment').options.length === 2);
  const archiveKey = await page.locator('#segment option').first().getAttribute('value');
  assert.equal(await page.locator('#segment').inputValue(), 'fixture-shared', 'fresh pages open the current campaign');
  assert.match(await page.locator('#segment option').first().textContent(), /fixture-shared.*archived/);
  assert.match(await page.locator('#live').textContent(), /current-live/);
  assert.doesNotMatch(await page.locator('#live').textContent(), /Search stopped|archive-live/);
  assert.match(await page.locator('#connection').textContent(), /Connected/);
  await page.getByRole('button', { name: 'reference-001', exact: true }).click();
  await page.locator('#compare-best').check();
  await page.waitForFunction(() => details?.metrics?.campaign === 'multirun-current' && comparison?.data?.metrics?.campaign === 'multirun-current');
  assert.match(await page.locator('#best').textContent(), /2\.000/);
  assert.deepEqual(await page.evaluate(() => [details.curve[0].loss, comparison.data.curve[0].loss]), [3, 2]);
  await page.selectOption('#segment', archiveKey);
  await page.getByRole('button', { name: 'reference-001', exact: true }).click();
  await page.waitForFunction(() => details?.metrics?.campaign === 'multirun-archive' && comparison?.data?.metrics?.campaign === 'multirun-archive');
  assert.match(await page.locator('#best').textContent(), /12\.000/);
  assert.deepEqual(await page.evaluate(() => [details.curve[0].loss, comparison.data.curve[0].loss]), [13, 12]);
  assert.match(await page.locator('#segment-info').textContent(), /Archived/);
  assert.match(await page.locator('#live').textContent(), /Viewing archived segment.*Recorded search stopped.*Current campaign: current-live/);
  assert.match(await page.locator('#connection').textContent(), /Connected/, 'an unfinished archived trial must not make the current campaign stale');
  await page.waitForTimeout(2200);
  assert.equal(await page.locator('#segment').inputValue(), archiveKey, 'polling preserves archive selection');
  await page.screenshot({ path: path.join(artifacts, 'dashboard-archived-campaign.png'), fullPage: true });

  // A response for an identical trial ID in an archive must not overwrite current data.
  let release, held;
  await page.route('**/api/trial?*', async route => {
    const query = new URL(route.request().url()).searchParams;
    if (query.get('segment') === archiveKey && query.get('id') === 'reference-001') {
      const response = await route.fetch();
      const data = await response.json();
      held = true;
      await new Promise(resolve => { release = resolve; });
      data.curve = [{ loss: 999, train_seconds: 999 }];
      await route.fulfill({ json: data });
    } else await route.continue();
  });
  await page.getByRole('button', { name: 'reference-001', exact: true }).click();
  const deadline = Date.now() + 10000;
  while (!held && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 10));
  assert.ok(held);
  await page.selectOption('#segment', 'fixture-shared');
  await page.getByRole('button', { name: 'reference-001', exact: true }).click();
  await page.waitForFunction(() => details?.metrics?.campaign === 'multirun-current' && comparison?.data?.metrics?.campaign === 'multirun-current');
  release();
  await page.unrouteAll({ behavior: 'wait' });
  assert.deepEqual(await page.evaluate(() => [details.curve[0].loss, comparison.data.curve[0].loss]), [3, 2]);
  assert.doesNotMatch(await page.locator('#live').textContent(), /Viewing archived/);
  assert.deepEqual(errors, []);
  const detailRequests = requests.filter(url => new URL(url).pathname === '/api/trial');
  assert.ok(detailRequests.length > 0);
  assert.ok(detailRequests.every(url => new URL(url).searchParams.has('segment')));
  assert.ok(detailRequests.some(url => new URL(url).searchParams.get('segment') === archiveKey && new URL(url).searchParams.get('id') === 'best-001'));
  assert.deepEqual(await files(), before, 'monitoring leaves both campaign directories unchanged');

  // Archive corruption stays visible without disconnecting current monitoring.
  await page.selectOption('#segment', archiveKey);
  await page.getByRole('button', { name: 'reference-001', exact: true }).click();
  await page.waitForFunction(() => details?.metrics?.campaign === 'multirun-archive' && comparison?.data?.metrics?.campaign === 'multirun-archive');
  const archiveLog = path.join(archive, 'experiments.jsonl'), archiveBytes = await readFile(archiveLog);
  await writeFile(archiveLog, 'bad\n');
  await page.waitForFunction(() => document.querySelector('#warning').textContent.includes('Archive unavailable'));
  assert.equal(await page.locator('#segment option').count(), 1);
  assert.equal(await page.locator('#segment').inputValue(), 'fixture-shared');
  assert.match(await page.locator('#connection').textContent(), /Connected/);
  assert.match(await page.locator('#live').textContent(), /current-live/);
  await page.getByRole('button', { name: 'reference-001', exact: true }).click();
  await page.waitForFunction(() => details?.metrics?.campaign === 'multirun-current' && comparison?.data?.metrics?.campaign === 'multirun-current');
  await writeFile(archiveLog, archiveBytes);
  await page.waitForFunction(() => document.querySelector('#segment').options.length === 2 && !document.querySelector('#warning').textContent);
  await page.selectOption('#segment', archiveKey);
  await page.getByRole('button', { name: 'reference-001', exact: true }).click();
  await page.waitForFunction(() => details?.metrics?.campaign === 'multirun-archive' && comparison?.data?.metrics?.campaign === 'multirun-archive');
  assert.deepEqual(await files(), before, 'restoring the fixture archive preserves its original bytes');
  console.log('Dashboard multirun browser passed: current default, archived selector, duplicate segment/trial IDs, segment-scoped details and comparison, stale-response race, live status, unchanged campaign files, and isolated/recoverable archive corruption.');
} finally { await browser.close(); server.kill('SIGTERM'); }
