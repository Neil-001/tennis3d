import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { mkdir, readFile } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
const root = fileURLToPath(new URL('../../', import.meta.url));
const fixture = JSON.parse(await readFile(`${root}/viewer/example.json`, 'utf8'));
await mkdir(`${root}/viewer/test-artifacts`, { recursive: true });
const browser = await chromium.launch({ headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
try {
  const context = await browser.newContext({ offline: true, viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  const errors = [], requests = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('request', r => { if (/^https?:/.test(r.url())) requests.push(r.url()); });
  await page.goto(pathToFileURL(`${root}/dist/viewer.html`).href);
  await page.getByRole('button', { name: 'Load example' }).click();
  assert.equal(await page.locator('#error').isVisible(), false);
  assert.match(await page.locator('#kind').textContent(), /Illustrative fixture · partial/);
  assert.equal(await page.locator('#diagnostic').isVisible(), true);
  const seek = async time => page.locator('#time').evaluate((el, t) => { el.value = String(t); el.dispatchEvent(new Event('input', { bubbles: true })); }, time);
  await seek(.375);
  assert.match(await page.locator('#state').textContent(), /Playback inferred · Frame 1 at 0.200 s: estimated\/detected/);
  assert.match(await page.locator('#position').textContent(), /Y -6.000/);
  await seek(1.06);
  assert.match(await page.locator('#state').textContent(), /Playback unavailable · Frame 4 at 1.050 s: estimated\/detected/);
  await seek(1.5);
  assert.equal(await page.locator('#position').textContent(), 'XYZ unavailable');
  await seek(2);
  assert.match(await page.locator('#position').textContent(), /Y 7.000/);
  assert.match(await page.locator('#timestamp').textContent(), /2.000 \/ 2.000/);
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await page.waitForTimeout(180);
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  const at1 = Number(await page.locator('#time').inputValue());
  assert.ok(at1 > 0 && at1 < 1);
  await seek(0);
  await page.locator('#speed').selectOption('4');
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await page.waitForTimeout(180);
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  assert.ok(Number(await page.locator('#time').inputValue()) > at1 * 1.7);
  await seek(1.9);
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await page.getByRole('button', { name: 'Play', exact: true }).waitFor();
  assert.equal(await page.locator('#time').inputValue(), '2');
  await page.getByRole('button', { name: 'hit 0.000 s' }).click();
  assert.equal(await page.locator('#time').inputValue(), '0');
  // Frame buttons and arrow keys land on exact source frame times, including the fractional end.
  const times = fixture.frames.map(f => f.t_s), at = async () => Number(await page.locator('#time').inputValue());
  assert.equal(await page.locator('#frame-back').isDisabled(), true);
  await page.getByRole('button', { name: 'Next source frame' }).click();
  assert.equal(await at(), times[1]);
  await seek((times[2] + times[3]) / 2);
  await page.getByRole('button', { name: 'Previous source frame' }).click();
  assert.equal(await at(), times[2]);
  await page.locator('h1').click();
  await page.keyboard.press('ArrowRight');
  assert.equal(await at(), times[3]);
  await page.keyboard.press('ArrowLeft');
  assert.equal(await at(), times[2]);
  await seek(times.at(-1) - .01);
  await page.getByRole('button', { name: 'Next source frame' }).click();
  assert.equal(await at(), times.at(-1));
  assert.match(await page.locator('#state').textContent(), new RegExp(`Frame ${times.length - 1} `));
  assert.equal(await page.locator('#frame-forward').isDisabled(), true);
  // The bundled typeface is embedded in the page; nothing is fetched for it.
  assert.equal(await page.evaluate(async () => { await document.fonts.ready; return document.fonts.check('14px "Google Sans Code"') && [...document.fonts].every(f => f.status === 'loaded'); }), true);
  await seek(.9);
  for (const view of ['Perspective', 'Top', 'Side']) {
    await page.getByRole('button', { name: view, exact: true }).click();
    await page.screenshot({ path: `${root}/viewer/test-artifacts/desktop-${view.toLowerCase()}.png`, fullPage: true });
  }
  await page.getByRole('button', { name: 'Reset view' }).click();
  const canvas = page.locator('#scene canvas');
  const box = await canvas.boundingBox();
  // The camera orbits the fixed target (0, 0, 0.5); the canvas reports its position.
  const view = async () => {
    const [x, y, z] = (await canvas.getAttribute('data-camera')).split(' ').map(Number);
    return { distance: Math.hypot(x, y, z - .5), azimuth: Math.atan2(y, x) };
  };
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2, near = (a, b) => Math.abs(a - b) < 1e-3 * b;
  const start = await view();
  await page.mouse.move(cx, cy);
  // The wheel zooms while the left button is held and moving, and the drag continues afterwards.
  await page.mouse.down();
  await page.mouse.move(cx + 40, cy + 10, { steps: 3 });
  const dragged = await view();
  assert.ok(near(dragged.distance, start.distance) && dragged.azimuth !== start.azimuth, 'Dragging must orbit at a constant distance.');
  await page.mouse.wheel(0, -120);
  await page.mouse.move(cx + 60, cy + 20, { steps: 2 });
  await page.mouse.wheel(0, -120);
  const held = await view();
  assert.ok(near(held.distance, start.distance * .95 ** 2.4), `Two held wheel steps must zoom in before release: ${start.distance} to ${held.distance}.`);
  await page.mouse.wheel(0, 240);
  assert.ok(near((await view()).distance, start.distance), 'A held wheel step back must zoom out by the same scale.');
  await page.mouse.wheel(0, -240);
  await page.mouse.move(cx + 80, cy + 30, { steps: 2 });
  const after = await view();
  assert.ok(near(after.distance, held.distance) && after.azimuth !== held.azimuth, 'The drag must keep orbiting after a held wheel step.');
  await page.mouse.up();
  await page.mouse.move(cx + 120, cy + 60, { steps: 2 });
  assert.deepEqual(await view(), after, 'Moving after release must not orbit.');
  // A released wheel step zooms once, by the same scale as a held one.
  await page.mouse.wheel(0, -120);
  assert.ok(near((await view()).distance, held.distance * .95 ** 1.2), 'A released wheel step must zoom once.');
  // Held and released zoom both stop at the camera limits, and a preset restores the view.
  await page.mouse.down();
  for (let i = 0; i < 12; i++) await page.mouse.wheel(0, -2000);
  assert.ok(near((await view()).distance, 2), 'Held zoom must stop at the nearest distance.');
  for (let i = 0; i < 12; i++) await page.mouse.wheel(0, 4000);
  assert.ok(near((await view()).distance, 400), 'Held zoom must stop at the farthest distance.');
  await page.mouse.up();
  await page.mouse.wheel(0, 4000);
  assert.ok(near((await view()).distance, 400), 'Released zoom must stop at the farthest distance.');
  await page.getByRole('button', { name: 'Reset view' }).click();
  assert.deepEqual(await view(), start, 'Reset must restore the starting view.');
  await page.mouse.down(); await page.mouse.move(cx + 80, cy + 30, { steps: 5 }); await page.mouse.wheel(0, -120); await page.mouse.up();
  const settleResize = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  await settleResize();
  const beforeZeroSize = await canvas.screenshot();
  await page.locator('#scene').evaluate(el => { el.style.width = '0'; el.style.height = '0'; });
  await settleResize();
  await page.locator('#scene').evaluate(el => { el.style.width = ''; el.style.height = ''; });
  await settleResize();
  assert.ok(beforeZeroSize.equals(await canvas.screenshot()), 'A temporary zero-sized scene must preserve the camera.');
  const upload = async (data, name = 'test.json') => {
    await page.locator('#file').setInputFiles({ name, mimeType: 'application/json', buffer: Buffer.from(typeof data === 'string' ? data : JSON.stringify(data)) });
    // The change handler reads the file asynchronously and clears the input when done.
    await page.waitForFunction(() => !document.getElementById('file').value);
  };
  await upload('{broken');
  assert.match(await page.locator('#error').textContent(), /Could not load/);
  const hostile = structuredClone(fixture);
  hostile.source.name = '<img src=x onerror="window.metadataExecuted=true">';
  await upload(hostile, '<svg onload=alert(1)>.json');
  assert.equal(await page.locator('#error').isVisible(), false);
  assert.match(await page.locator('#source').textContent(), /<img src=x/);
  assert.equal(await page.locator('#source img, #source svg').count(), 0);
  assert.equal(await page.evaluate(() => window.metadataExecuted), undefined);
  await upload(await readFile(`${root}/tests/fixtures/trajectory.json`, 'utf8'));
  assert.equal(await page.locator('#diagnostic').isVisible(), false);
  await seek(.2);
  assert.match(await page.locator('#position').textContent(), /Z 1.650/);
  const failed = structuredClone(fixture);
  failed.failure_reasons = ['unsupported_shot']; failed.status = 'failed';
  await upload(failed);
  assert.equal(await page.locator('#play').isDisabled(), true);
  assert.equal(await page.locator('#position').textContent(), 'XYZ unavailable');
  const empty = structuredClone(failed);
  empty.frames = []; empty.source.frame_count = 0; empty.synthetic_diagnostics = null;
  empty.summary = { valid_frame_fraction: 0, detected_frame_fraction: 0 };
  await upload(empty);
  assert.equal(await page.locator('#time').isDisabled(), true);
  const single = structuredClone(fixture);
  single.frames = [single.frames[0]]; single.source.frame_count = 1;
  single.status = 'ok'; single.synthetic_diagnostics = null;
  single.summary = { valid_frame_fraction: 1, detected_frame_fraction: 1 };
  await upload(single);
  assert.equal(await page.locator('#play').isDisabled(), true);
  assert.match(await page.locator('#position').textContent(), /X 0.500/);
  // The file name wraps only as far as its row requires, whatever the length of the label beside it.
  const heading = () => page.evaluate(() => {
    const name = document.getElementById('source'), kind = document.getElementById('kind'), head = name.parentElement;
    const lines = el => { const range = document.createRange(); range.selectNodeContents(el); return new Set([...range.getClientRects()].map(r => Math.round(r.top))).size; };
    const probe = name.cloneNode(true);
    probe.removeAttribute('id'); probe.style.cssText = 'position:absolute;visibility:hidden;white-space:nowrap;flex:none';
    head.append(probe); const natural = probe.getBoundingClientRect().width; probe.remove();
    const style = getComputedStyle(head), row = head.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    const box = head.getBoundingClientRect(), k = kind.getBoundingClientRect(), n = name.getBoundingClientRect();
    return { nameLines: lines(name), needed: Math.ceil(natural / row), kindLines: lines(kind), sameRow: k.top < n.bottom && k.bottom > n.top,
      inside: k.left >= box.left && k.right <= box.right + .5, fits: document.documentElement.scrollWidth <= window.innerWidth };
  });
  const labelled = structuredClone(fixture);
  labelled.source.name = 'gallery-v2-02-03.mp4';
  const longLabel = 'Stage 4 production: Muon transformer, selected EMA at step 83,334; lifting on shared detection and calibration';
  for (const [width, height] of [[1280, 900], [390, 844]]) {
    await page.setViewportSize({ width, height });
    for (const label of [longLabel, 'Lifter A', undefined]) {
      labelled.provenance.model_label = label;
      await upload(labelled, 'result.json');
      assert.match(await page.locator('#source').textContent(), /^result\.json · gallery-v2-02-03\.mp4$/);
      const layout = await heading(), where = `${width}px, label ${label ?? 'absent'}`;
      assert.ok(layout.nameLines <= layout.needed, `File name broke into ${layout.nameLines} lines where ${layout.needed} fit (${where}).`);
      assert.ok(layout.inside && layout.fits, `Label left its card or widened the page (${where}).`);
      if (label === longLabel) assert.ok(!layout.sameRow, `Long label squeezed the file name instead of moving below it (${where}).`);
      else if (width === 1280) assert.ok(layout.sameRow && layout.kindLines === 1, `Short label left the file name's row (${where}).`);
    }
    await page.screenshot({ path: `${root}/viewer/test-artifacts/heading-${width}.png` });
  }
  await page.setViewportSize({ width: 1280, height: 900 });
  await settleResize();
  for (const path of process.argv.slice(2)) {
    const truth = JSON.parse(await readFile(path, 'utf8'));
    await upload(truth, path.split('/').at(-1));
    assert.equal(await page.locator('#error').isVisible(), false);
    const kindText = await page.locator('#kind').textContent();
    const expectedKind = { synthetic_ground_truth: 'Synthetic ground truth', fixture: 'Illustrative fixture', model_estimate: 'Model estimate' }[truth.provenance.trajectory_kind];
    assert.ok(kindText.includes(expectedKind));
    if (truth.provenance.model_label) assert.ok(kindText.includes(truth.provenance.model_label));
    assert.equal(await page.locator('#events button').count(), (truth.synthetic_diagnostics?.events ?? []).length);
    for (const [i, event] of (truth.synthetic_diagnostics?.events ?? []).entries()) {
      await page.locator('#events button').nth(i).click();
      assert.ok(Math.abs(Number(await page.locator('#time').inputValue()) - event.t_s) < 1e-9);
    }
    await seek(truth.frames.at(-1).t_s);
    assert.match(await page.locator('#state').textContent(), new RegExp(`Frame ${truth.frames.length - 1} `));
    await page.locator('#trail').selectOption('1000000');
    await page.getByRole('button', { name: 'Perspective', exact: true }).click();
    await page.screenshot({ path: `${root}/viewer/test-artifacts/${path.split('/').at(-1)}.png`, fullPage: true });
    console.log(`Verified result ${path}: ${truth.frames.length} frames, ${(truth.synthetic_diagnostics?.events ?? []).length} events.`);
  }
  await page.getByRole('button', { name: 'Load example' }).click();
  await seek(.9);
  await page.getByRole('button', { name: 'Top', exact: true }).click();
  // Resize the live scene without reloading data or reselecting a preset.
  await page.setViewportSize({ width: 390, height: 844 });
  await settleResize();
  await page.screenshot({ path: `${root}/viewer/test-artifacts/mobile.png`, fullPage: true });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  // The canvas keeps its previous pixel width until its resize observer runs; the panel must clip it meanwhile.
  assert.equal(await page.evaluate(() => {
    document.querySelector('#scene canvas').style.width = '1280px';
    return document.documentElement.scrollWidth <= window.innerWidth;
  }), true);
  assert.deepEqual(errors, []);
  assert.deepEqual(requests, []);
  console.log('Offline file:// Chromium checks passed at desktop and mobile sizes; no page errors or network requests.');
} finally { await browser.close(); }

// Also qualify the optional embedded gallery build.
await import("./offline-gallery.mjs");
