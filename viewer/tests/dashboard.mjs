// Browser behavior against fabricated history, never comparable training scores.
import assert from 'node:assert/strict';
import { execFileSync, spawn } from 'node:child_process';
import { mkdir, mkdtemp } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const artifacts = process.env.DASHBOARD_ARTIFACTS || path.join(root, 'viewer/test-artifacts');
await mkdir(artifacts, { recursive: true });
const run = await mkdtemp(path.join(artifacts, 'dashboard-'));
const python = process.env.DASHBOARD_PYTHON || path.join(root, '.venv/bin/python');
const env = { ...process.env, PYTHONPATH: root };
function fixture(phase) { execFileSync(python, ['tests/dashboard_fixture.py', run, phase], { cwd: root, env }); }
fixture('initial');
const server = spawn(python, ['-m', 'tennis3d', 'dashboard', '--run', run, '--port', '0'], { cwd: root, env, stdio: ['ignore', 'pipe', 'pipe'] });
const base = await new Promise((resolve, reject) => {
  const timeout = setTimeout(() => reject(Error('dashboard server did not start')), 10000);
  server.stdout.on('data', data => { const match = data.toString().match(/http:\/\/127\.0\.0\.1:\d+/); if (match) { clearTimeout(timeout); resolve(match[0]); } });
  server.on('exit', code => reject(Error(`dashboard exited ${code}`)));
});
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });
  await page.addInitScript(() => {
    // Record actual canvas drawing calls, including path geometry and point colors.
    window.draws = {};
    window.chartListeners = {};
    const addListener = EventTarget.prototype.addEventListener;
    EventTarget.prototype.addEventListener = function (type, ...args) {
      if (['history', 'curve'].includes(this.id)) {
        const key = `${this.id}:${type}`;
        window.chartListeners[key] = (window.chartListeners[key] || 0) + 1;
      }
      return addListener.call(this, type, ...args);
    };
    for (const method of ['scale', 'beginPath', 'moveTo', 'lineTo', 'stroke', 'arc']) {
      const original = CanvasRenderingContext2D.prototype[method];
      CanvasRenderingContext2D.prototype[method] = function (...args) {
        const id = this.canvas.id;
        if (method === 'scale') window.draws[id] = [];
        (window.draws[id] ||= []).push({ method, args, color: method === 'arc' ? this.fillStyle : this.strokeStyle });
        return original.apply(this, args);
      };
    }
  });
  const errors = [], requests = [];
  page.on('pageerror', error => errors.push(error.stack || error.message));
  page.on('request', request => requests.push(request.url()));
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector('#trials').children.length === 1);
  await page.selectOption('#segment', 'fixture-standard');
  assert.match(await page.locator('#best').textContent(), /2\.500/);
  assert.match(await page.locator('#baseline').textContent(), /3\.000/);
  assert.equal(await page.locator('#trials tr').count(), 4);
  const history = await page.evaluate(() => window.draws.history);
  assert.equal(history.filter(call => call.method === 'arc' && call.color === '#267044').length, 0);
  assert.equal(history.filter(call => call.method === 'arc' && call.color === '#2868b5').length, 1);
  assert.equal(history.filter(call => call.method === 'arc' && call.color === '#c97c15').length, 1);
  assert.equal(history.filter(call => call.method === 'arc' && call.color === '#8793a1').length, 1);
  const retained = history.filter(call => ['moveTo', 'lineTo'].includes(call.method) && call.color === '#267044');
  assert.equal(retained.length, 7);
  for (let i = 1; i < retained.length; i++) {
    assert.ok(retained[i].args[0] === retained[i - 1].args[0] || retained[i].args[1] === retained[i - 1].args[1], 'retained line must only move horizontally or vertically');
  }
  assert.equal(retained[0].args[1], retained[1].args[1]);
  assert.equal(retained[1].args[0], retained[2].args[0]);
  assert.ok(retained[2].args[1] > retained[1].args[1], 'new best drops at attempt 2');
  assert.equal(history.filter(call => call.method === 'stroke' && call.color === '#b63829').length, 1);
  const hoverAt = async (id, value, xmax, lane = false) => {
    await page.locator(`#${id}`).scrollIntoViewIfNeeded();
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const bounds = await page.locator(`#${id}`).boundingBox();
    const bottom = bounds.height - (id === 'history' ? 72 : 40);
    await page.mouse.move(bounds.x + 72 + value / xmax * (bounds.width - 92), bounds.y + (lane ? bottom + 38 : 60));
  };
  const historyTip = page.locator('#history-tooltip'), curveTip = page.locator('#curve-tooltip');
  const initialLayout = await page.locator('#history').boundingBox();
  const historyImage = await page.locator('#history').evaluate(c => c.toDataURL());
  const historyMarkers = history.filter(call => call.method === 'arc');
  const markerPixels = async color => page.locator('#history').evaluate((c, marker) => {
    const ctx = c.getContext('2d'), scale = window.devicePixelRatio;
    return [...ctx.getImageData(Math.floor(marker.args[0] * scale) - 4, Math.floor(marker.args[1] * scale) - 4, 9, 9).data];
  }, historyMarkers.find(call => call.color === color));
  for (const [attempt, color] of [[1, '#c97c15'], [2, '#2868b5']]) {
    const before = await markerPixels(color);
    await hoverAt('history', attempt, 4);
    assert.deepEqual(await markerPixels(color), before, `${color} marker pixels must keep their colors on hover`);
    assert.ok(await page.locator('#history').evaluate((c, image) => c.toDataURL() === image, historyImage), 'hover must preserve every canvas pixel');
  }
  await hoverAt('history', 4, 4);
  await historyTip.waitFor({ state: 'visible' });
  assert.match(await historyTip.textContent(), /Attempt 4 · fixture-4\nStatus: discarded\nXYZ error: 2.8 m\nBest retained: 2.5 m/);
  assert.deepEqual(await page.locator('#history').boundingBox(), initialLayout, 'tooltip must not shift chart layout');
  assert.deepEqual(await page.evaluate(() => window.draws.history.filter(call => call.method === 'arc')), historyMarkers, 'hover must preserve marker geometry and colors');
  await hoverAt('history', 2.8, 4, true);
  assert.match(await historyTip.textContent(), /Attempt 3 · fixture-3\nStatus: checks-failed\nXYZ error: unscored\nBest retained: 2.5 m · fixture-2\nDeliberate fixture failure/);
  await page.waitForTimeout(2200);
  assert.equal(await page.locator('.chart-tooltip').count(), 2, 'polling must reuse tooltip elements');
  assert.deepEqual(await page.evaluate(() => Object.values(window.chartListeners)), [1, 1, 1, 1, 1, 1], 'polling must not attach duplicate canvas listeners');
  assert.equal(await historyTip.isVisible(), true, 'hover updates across polling without moving the pointer');
  await page.mouse.move(0, 0);
  assert.equal(await historyTip.isVisible(), false);
  assert.ok(await page.locator('#history').evaluate((c, before) => c.toDataURL() === before, historyImage), 'leaving hover must restore the exact canvas pixels');
  assert.equal(await page.locator('#compare-best').isChecked(), false);
  await page.getByRole('button', { name: 'fixture-1', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('#details').textContent.includes('fixture score'));
  let mode = 'normal', held = null, release = null;
  await page.route('**/api/trial?*', async route => {
    const response = await route.fetch();
    const data = await response.json();
    if (data.trial.trial === 'fixture-1' && mode === 'selected-failure') return route.fulfill({ status: 422, json: { error: 'Fixture selected fetch failure' } });
    if (data.trial.trial === 'fixture-2') {
      if (mode === 'failure') return route.fulfill({ status: 422, json: { error: 'Fixture fetch failure' } });
      if (mode === 'different') data.trial.loss_label = 'Fixture absolute error (metres)';
      if (mode === 'unknown') delete data.trial.loss_label;
      if (mode === 'missing') data.curve = [];
      if (mode === 'delay') {
        mode = 'normal';
        held = data.trial.trial;
        await new Promise(resolve => { release = resolve; });
        // A stale response is visibly different if it incorrectly gets applied.
        data.curve = [{ loss: 999, train_seconds: 999 }];
      }
    }
    await route.fulfill({ json: data });
  });
  const hasGreenCurve = () => page.evaluate(() => window.draws.curve.some(call => call.method === 'stroke' && call.color === '#267044'));
  const refreshComparison = async () => {
    await page.locator('#compare-best').uncheck();
    await page.locator('#compare-best').check();
  };
  await page.locator('#compare-best').check();
  await page.waitForFunction(() => document.querySelector('#curve-legend').textContent.includes('Green: best retained trial fixture-2'));
  assert.ok(await hasGreenCurve());
  const curves = await page.evaluate(() => window.draws.curve);
  const blue = curves.find(call => call.method === 'arc' && call.color === '#2868b5');
  const green = curves.find(call => call.method === 'arc' && call.color === '#267044');
  assert.ok(green.args[0] > blue.args[0], 'different elapsed seconds share the same x-axis');
  assert.ok(green.args[1] < blue.args[1], 'higher training loss shares the same log y-axis');
  const curveImage = await page.locator('#curve').evaluate(c => c.toDataURL());
  await hoverAt('curve', 7.2, 200);
  await curveTip.waitFor({ state: 'visible' });
  assert.match(await curveTip.textContent(), /Selected trial fixture-1\nTime: 7 s · loss: 0.14285714285714285 · step: 7/);
  assert.match(await curveTip.textContent(), /Best retained trial fixture-2\nTime: 8 s · loss: 0.5 · step: 4/);
  assert.doesNotMatch(await curveTip.textContent(), /Time: 7.2 s/, 'tooltip must report stored samples, not interpolated values');
  assert.ok(await page.locator('#curve').evaluate((c, image) => c.toDataURL() === image, curveImage), 'training hover must preserve every canvas pixel');
  await page.screenshot({ path: path.join(artifacts, 'dashboard-training-hover.png') });
  assert.equal(await curveTip.isVisible(), true);
  await page.evaluate(() => { document.querySelector('#compare-best').checked = false; document.querySelector('#compare-best').dispatchEvent(new Event('change')); });
  assert.equal(await curveTip.isVisible(), false, 'comparison context changes must clear hover');
  await page.locator('#compare-best').check();
  await page.waitForFunction(() => document.querySelector('#curve-legend').textContent.includes('Green: best retained trial fixture-2'));
  await page.mouse.move(0, 0);
  await page.screenshot({ path: path.join(artifacts, 'dashboard-comparison.png'), fullPage: true });
  await page.locator('#compare-best').uncheck();
  assert.equal(await hasGreenCurve(), false);
  for (const [testMode, message] of [['different', 'different loss definitions or units'], ['unknown', 'not recorded for both trials'], ['missing', 'no positive training-loss curve'], ['failure', 'Fixture fetch failure']]) {
    mode = testMode;
    await refreshComparison();
    await page.waitForFunction(message => document.querySelector('#comparison-status').textContent.includes(message), message);
    assert.equal(await hasGreenCurve(), false);
  }
  mode = 'selected-failure';
  await refreshComparison();
  await page.waitForFunction(() => document.querySelector('#comparison-status').textContent.includes('Fixture selected fetch failure'));
  assert.equal(await hasGreenCurve(), false);
  mode = 'normal';
  await page.getByRole('button', { name: 'fixture-2', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('#details').textContent.includes('fixture-2'));
  assert.match(await page.locator('#comparison-status').textContent(), /selected trial is the current best/);
  assert.equal(await hasGreenCurve(), false);

  // Responses from a prior selection, segment, or retained best must not win a race.
  const startDelayed = async () => {
    held = null; mode = 'delay';
    await page.getByRole('button', { name: 'fixture-1', exact: true }).click();
    const deadline = Date.now() + 10000;
    while (!held && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 10));
    assert.ok(held, 'comparison request must reach the delayed response');
  };
  await startDelayed();
  await page.getByRole('button', { name: 'fixture-4', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('#details').textContent.includes('fixture-4'));
  release();
  await page.waitForTimeout(100);
  assert.match(await page.locator('#details').textContent(), /fixture-4/);
  assert.equal(await page.evaluate(() => comparison.data.curve.some(p => p.loss === 999)), false);

  await startDelayed();
  await page.selectOption('#segment', 'fixture-smoke');
  await page.waitForFunction(() => document.querySelector('#details').textContent.includes('fixture-live'));
  release();
  await page.waitForTimeout(100);
  assert.match(await page.locator('#comparison-status').textContent(), /no retained best trial/);
  assert.equal(await hasGreenCurve(), false);
  await page.selectOption('#segment', 'fixture-standard');
  await startDelayed();
  fixture('best-switch');
  await page.waitForFunction(() => document.querySelector('#curve-legend').textContent.includes('Green: best retained trial fixture-5'));
  release();
  await page.waitForTimeout(100);
  assert.match(await page.locator('#curve-legend').textContent(), /fixture-5/);
  assert.equal(await page.evaluate(() => comparison.data.trial.trial), 'fixture-5');
  assert.equal(await page.evaluate(() => comparison.data.curve.some(p => p.loss === 999)), false);
  await page.unrouteAll({ behavior: 'wait' });
  await page.getByRole('button', { name: 'fixture-3', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('#detail-summary').textContent.includes('Deliberate fixture failure'));
  await page.screenshot({ path: path.join(artifacts, 'dashboard-history.png'), fullPage: true });
  // Baselines and diagnostics cannot establish records merely by having a lower score.
  fixture('records');
  await page.waitForFunction(() => [...document.querySelector('#segment').options].some(o => o.value === 'fixture-records'));
  await page.selectOption('#segment', 'fixture-records');
  const records = await page.evaluate(() => window.draws.history);
  assert.equal(records.filter(call => call.method === 'arc' && call.color === '#2868b5').length, 1);
  assert.equal(records.filter(call => call.method === 'arc' && call.color === '#c97c15').length, 1);
  assert.equal(records.filter(call => call.method === 'arc' && call.color === '#8793a1').length, 6);
  assert.equal(records.filter(call => call.method === 'arc' && call.color === '#267044').length, 0);
  await hoverAt('history', 1, 10);
  assert.match(await historyTip.textContent(), /Baseline: small \(diagnostic\)\nXYZ error: 0.1 m\nBest retained: none/);
  await hoverAt('history', 4, 10);
  assert.match(await historyTip.textContent(), /Status: discarded\nXYZ error: 3 m\nBest retained: 3 m/);
  await hoverAt('history', 6, 10);
  assert.match(await historyTip.textContent(), /Baseline: reference\nXYZ error: 2.8 m\nBest retained: 2.5 m/);
  await hoverAt('history', 9, 10, true);
  assert.match(await historyTip.textContent(), /<img src=x onerror=alert\(1\)>/);
  assert.equal(await historyTip.locator('img').count(), 0, 'log text must never become markup');
  await page.screenshot({ path: path.join(artifacts, 'dashboard-history-hover.png') });
  assert.equal(await historyTip.isVisible(), true);
  await hoverAt('history', 10, 10);
  assert.match(await historyTip.textContent(), /Status: running\nXYZ error: unscored\nBest retained: 2.5 m/);
  await page.evaluate(() => { document.querySelector('#segment').value = 'fixture-smoke'; document.querySelector('#segment').dispatchEvent(new Event('change')); });
  assert.equal(await historyTip.isVisible(), false, 'segment context changes must clear hover');
  // Resize and high-DPI coordinates use CSS pixels while the canvas keeps a scaled backing buffer.
  await page.setViewportSize({ width: 390, height: 844 });
  await page.selectOption('#segment', 'fixture-records');
  await hoverAt('history', 10, 10);
  const tipBounds = await historyTip.boundingBox();
  assert.ok(tipBounds.x >= 0 && tipBounds.x + tipBounds.width <= 390 && tipBounds.y >= 0 && tipBounds.y + tipBounds.height <= 844);
  await page.screenshot({ path: path.join(artifacts, 'dashboard-mobile-hover.png') });
  assert.equal(await historyTip.isVisible(), true);
  const retina = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 });
  await retina.goto(base);
  await retina.waitForFunction(() => document.querySelector('#segment').options.length === 3);
  await retina.selectOption('#segment', 'fixture-records');
  await retina.locator('#history').scrollIntoViewIfNeeded();
  await retina.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const retinaBounds = await retina.locator('#history').boundingBox();
  await retina.mouse.move(retinaBounds.x + retinaBounds.width - 20, retinaBounds.y + 50);
  assert.match(await retina.locator('#history-tooltip').textContent(), /Attempt 10 · fixture-record-10/);
  assert.equal(await retina.locator('#history').evaluate(c => c.width === c.clientWidth * 2), true);
  await retina.close();
  fixture('records-finished');
  await page.setViewportSize({ width: 1280, height: 1000 });
  await page.selectOption('#segment', 'fixture-smoke'); fixture('live');
  await page.waitForFunction(() => document.querySelector('#live').textContent.includes('training'));
  assert.equal(await page.locator('#trials tr').count(), 1);
  await page.waitForFunction(() => document.querySelector('#details').textContent.includes('"steps": 2'));
  assert.match(await page.locator('#loss-label').textContent(), /squared Euclidean position loss, XYZ divided by 10 m/);
  const curve = await page.locator('#curve').evaluate(canvas => canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data.some(value => value !== 0));
  assert.ok(curve);
  fixture('stale');
  await page.waitForFunction(() => document.querySelector('#connection').textContent.includes('Stale progress'));
  fixture('interrupted');
  await page.waitForFunction(() => document.querySelector('#live').textContent.includes('Search stopped'));
  assert.match(await page.locator('#trials').textContent(), /interrupted/);
  server.kill('SIGTERM');
  await page.waitForFunction(() => document.querySelector('#connection').textContent.includes('Disconnected'));
  await page.screenshot({ path: path.join(artifacts, 'dashboard-disconnected.png'), fullPage: true });
  assert.deepEqual(errors, []);
  assert.ok(requests.every(url => url.startsWith(base)));
  console.log('Dashboard browser passed: strict record/grey baseline markers, retained steps, discrete history and per-series recorded-sample hover, failures, safe text, polling/context cleanup, mobile/DPR, comparison compatibility and races, live/stale/stop/disconnect, local assets.');
} finally { await browser.close(); server.kill('SIGTERM'); }
