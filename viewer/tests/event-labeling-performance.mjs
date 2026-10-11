import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp, mkdir, readFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';

// This regression always creates synthetic media and writes only its temporary labels.
const repo = fileURLToPath(new URL('../../', import.meta.url));
const root = await mkdtemp(path.join(os.tmpdir(), 'tennis-events-performance-'));
const evidence = process.env.EVENT_EVIDENCE || root;
await mkdir(evidence, {recursive:true});
const python = process.env.EVENT_PYTHON || 'uv';
const prefix = process.env.EVENT_PYTHON ? [] : ['run', '--extra', 'labeling', 'python'];
const pythonEnv = {...process.env,PYTHONPATH:[repo,process.env.PYTHONPATH].filter(Boolean).join(path.delimiter)};
async function run(args) {
  return new Promise((resolve, reject) => {
    const child = spawn(python, [...prefix, ...args], {cwd:repo,env:pythonEnv});
    let output = '';
    child.stdout.on('data', data => output += data);
    child.stderr.on('data', data => output += data);
    child.on('error', reject);
    child.on('exit', code => code === 0 ? resolve(output) : reject(new Error(output)));
  });
}
await run(['tests/event_labeling_fixture.py', root]);
const labels = path.join(root, 'isolated-performance-labels');
const server = spawn(python, [...prefix, '-m', 'tennis3d.labeling.dashboard', '--inventory', path.join(root, 'inventory.json'), '--cache', path.join(root, 'previews'), '--labels', labels, '--port', '0'], {cwd:repo,env:pythonEnv});
let output = '', browser;
server.stdout.on('data', data => output += data);
server.stderr.on('data', data => output += data);
try {
  const url = await new Promise((resolve, reject) => {
    const deadline = setTimeout(() => reject(new Error(`Server startup timed out: ${output}`)), 120000);
    server.on('error', error => { clearTimeout(deadline); reject(error); });
    server.on('exit', code => { clearTimeout(deadline); reject(new Error(`Server exited ${code}: ${output}`)); });
    server.stdout.on('data', () => {
      const match = output.match(/Labeling dashboard: (http:\/\/localhost:\d+)/);
      if (match) { clearTimeout(deadline); resolve(match[1]); }
    });
  });
  browser = await chromium.launch({headless:true});
  const page = await browser.newPage({viewport:{width:1440,height:1000}});
  page.on('dialog', dialog => dialog.accept());
  const errors = [], requests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (/\/frames\/\d+/.test(request.url())) requests.push(request.url()); });
  const catalog = await (await page.request.get(url + '/api/gallery')).json();
  const cid = catalog.clips[0].clip_id;
  const detail = await (await page.request.get(`${url}/api/clips/${cid}`)).json();
  const diskEvents = [{frame_index:2,type:'bounce'}];
  const receipt = await page.request.post(`${url}/api/clips/${cid}/labels`, {headers:{'X-Label-Token':catalog.token,Origin:url},data:{revision:0,identity:detail.identity,events:diskEvents,annotator:'TEST competing submission'}});
  assert.equal(receipt.status(), 200);
  const key = `tennis-events:${catalog.inventory_sha256}:${cid}`;
  const positionKey = `tennis-events-position:${catalog.inventory_sha256}`;
  const legacy = {events:[{frame_index:7,type:'hit'}],revision:0,dirty:true,selected:7,annotator:'TEST old unsaved draft'};
  const otherKey = 'tennis-events:other-inventory:legacy-clip';
  const otherDraft = JSON.stringify({events:[{frame_index:11,type:'bounce'}],revision:4,dirty:true,selected:11,annotator:'TEST untouched draft'});
  await page.addInitScript(({key,positionKey,legacy,otherKey,otherDraft}) => {
    if (!localStorage.getItem('performance-test-seeded')) {
      localStorage.setItem(key, JSON.stringify(legacy)); localStorage.setItem(positionKey, '0');
      localStorage.setItem(otherKey, otherDraft); localStorage.setItem('performance-test-seeded', 'yes');
    }
    window.__decodeCalls = [];
    const decode = HTMLImageElement.prototype.decode;
    HTMLImageElement.prototype.decode = function() {
      window.__decodeCalls.push(this.src);
      window.__decodesActive = (window.__decodesActive || 0) + 1;
      window.__decodePeak = Math.max(window.__decodePeak || 0,window.__decodesActive);
      let decoded = decode.call(this);
      if (window.__holdNextDecode) {
        window.__holdNextDecode = false;
        decoded = decoded.then(() => new Promise(resolve => {
          window.__decodeHeld = true;
          window.__releaseDecode = () => { window.__heldDecodeReleased = true; resolve(); };
        }));
      }
      return decoded.finally(() => window.__decodesActive--);
    };
    const presented = HTMLVideoElement.prototype.requestVideoFrameCallback;
    HTMLVideoElement.prototype.requestVideoFrameCallback = function(callback) {
      return presented.call(this, (now, metadata) => { if (!window.__suppressVideoFrames) callback(now, metadata); });
    };
  }, {key,positionKey,legacy,otherKey,otherDraft});
  const waitOriginal = async index => {
    await page.waitForFunction(index => {
      const image = document.getElementById('original');
      return !document.getElementById('hit').disabled && !image.hidden && image.complete && document.getElementById('loading').hidden && document.getElementById('frame-readout').textContent.startsWith(`Frame ${index + 1} /`);
    }, index);
  };
  const settlePrefetch = () => page.waitForFunction(() => prefetchActive.size === 0);
  const seek = async index => {
    await page.locator('#scrub').evaluate((slider,index) => { slider.value = String(index); slider.dispatchEvent(new Event('input',{bubbles:true})); }, index);
    await waitOriginal(index);
  };
  const paint = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const pixels = async index => {
    const rgb = await page.locator('#original').evaluate(async image => {
      const canvas = document.createElement('canvas'); canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
      const context = canvas.getContext('2d'); context.drawImage(image,0,0);
      const rgba = context.getImageData(0,0,canvas.width,canvas.height).data;
      const rgb = new Uint8Array(canvas.width * canvas.height * 3);
      for (let i = 0, j = 0; i < rgba.length; i += 4) { rgb[j++] = rgba[i]; rgb[j++] = rgba[i+1]; rgb[j++] = rgba[i+2]; }
      const hash = await crypto.subtle.digest('SHA-256', rgb);
      return {sha:Array.from(new Uint8Array(hash), x => x.toString(16).padStart(2,'0')).join(''),width:canvas.width,height:canvas.height};
    });
    assert.equal(rgb.sha, detail.frames[index].decoded_rgb_sha256, `Displayed pixels must identify original frame ${index}`);
    assert.deepEqual([rgb.width,rgb.height], [96,64]);
  };
  const checkCache = async () => {
    const cache = await page.evaluate(() => ({bytes:cacheBytes,entries:frameCache.size,prefetch:prefetchActive.size,loaders:activeImages.size,decodePeak:window.__decodePeak,decoded:[...frameCache.values()].reduce((sum,entry) => sum + entry.bytes,0)}));
    assert(cache.entries <= 12, JSON.stringify(cache)); assert(cache.bytes <= 64 * 1024 * 1024, JSON.stringify(cache));
    assert.equal(cache.bytes, cache.decoded); assert(cache.prefetch <= 2, JSON.stringify(cache)); assert(cache.loaders <= 3, JSON.stringify(cache)); assert(cache.decodePeak <= 3, JSON.stringify(cache));
    return cache;
  };
  await page.goto(url); await waitOriginal(7); await settlePrefetch(); await pixels(7);
  assert.match(await page.locator('#status').textContent(), /newer disk submission/);
  assert.equal(await page.locator('#annotator').inputValue(), legacy.annotator);
  assert.equal(await page.locator('#state').textContent(), 'Unsaved draft');
  const beforeMark = {requests:requests.length,decodes:await page.evaluate(() => window.__decodeCalls.length)};
  assert.equal(beforeMark.decodes,3,'Initial original and two neighbors each decode once');
  assert.equal(beforeMark.requests,3,'Only the selected original and its two neighbors are fetched');
  await page.locator('#hit').click(); await page.locator('#bounce').click(); await paint();
  assert.equal(await page.locator('#event-count').textContent(), '2');
  assert.equal(requests.length, beforeMark.requests, 'Marking confirmed originals must not request frames');
  assert.equal(await page.evaluate(() => window.__decodeCalls.length), beforeMark.decodes, 'Marking confirmed originals must not decode again');
  const warmStart = performance.now();
  await page.locator('#forward-frame').click(); await waitOriginal(8);
  await page.locator('#back-frame').click(); await waitOriginal(7);
  const warmStepsMs = performance.now() - warmStart;
  assert.equal(requests.filter(request => /\/frames\/(7|8)\?/.test(request)).length,2,'Prefetched neighbors must be reused');
  await settlePrefetch();

  // Hold an already decoded old image so its continuation runs after a newer paint.
  await page.evaluate(() => { window.__holdNextDecode = true; });
  await page.locator('#scrub').evaluate(slider => { slider.value = '12'; slider.dispatchEvent(new Event('input',{bubbles:true})); });
  await page.waitForFunction(() => window.__decodeHeld);
  assert.equal(await page.locator('#hit').isDisabled(), true);
  assert.equal(await page.locator('#forward-frame').isDisabled(), false);
  await seek(25); await pixels(25);
  await page.evaluate(() => window.__releaseDecode()); await paint(); await pixels(25);
  assert.match(await page.locator('#frame-readout').textContent(), /^Frame 26 \//);
  await settlePrefetch();

  await seek(12); await settlePrefetch();
  await page.evaluate(() => {
    for (let i = 0; i < 7; i++) document.body.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));
  });
  await waitOriginal(19); await pixels(19); await settlePrefetch();
  assert.equal(await page.locator('#event-count').textContent(), '2', 'Rapid navigation must not edit events');
  await checkCache();

  // Delay confirmation after pause. Preview must freeze in the click turn, while
  // navigation remains usable and marking waits for a real decoded original.
  await seek(0); await settlePrefetch();
  await page.reload(); await waitOriginal(0); await settlePrefetch();
  let releaseFrames, heldFrames = 0;
  const frameGate = new Promise(resolve => releaseFrames = resolve);
  await page.route('**/frames/**', async route => { heldFrames++; await frameGate; await route.continue().catch(() => {}); });
  await page.locator('#play').click();
  await page.waitForFunction(() => !document.getElementById('video').paused && playbackFrame >= 8);
  const paused = await page.evaluate(() => {
    const index = playbackFrame; document.getElementById('play').click();
    return {index,paused:video.paused,time:video.currentTime,preview:!video.hidden,markDisabled:document.getElementById('hit').disabled};
  });
  assert(paused.paused && paused.preview && paused.markDisabled);
  await paint(); assert.equal(await page.locator('#video').evaluate(video => video.currentTime), paused.time);
  assert.equal(await page.locator('#forward-frame').isDisabled(), false);
  await page.locator('h1').click(); await page.keyboard.press('h');
  assert.equal(await page.locator('#event-count').textContent(), '2');
  await page.locator('#forward-frame').click(); releaseFrames();
  await waitOriginal(paused.index + 1); await pixels(paused.index + 1);
  assert(heldFrames > 0); assert.equal(await page.locator('#video').evaluate(video => video.paused), true);
  await page.unroute('**/frames/**'); await settlePrefetch();

  // With no presented-frame callback there is no trustworthy preview identity.
  await seek(0); await settlePrefetch();
  await page.evaluate(() => { window.__suppressVideoFrames = true; });
  await page.locator('#play').click();
  await page.waitForFunction(() => !document.getElementById('video').paused && document.getElementById('video').currentTime > .12);
  const unknownBefore = requests.length;
  const unknown = await page.evaluate(() => { document.getElementById('play').click(); return {paused:video.paused,mapped:playbackFrame,markDisabled:document.getElementById('hit').disabled}; });
  assert(unknown.paused && unknown.markDisabled); assert.equal(unknown.mapped,null);
  assert.equal(requests.length,unknownBefore,'Unmapped pauses must not guess a source from currentTime');
  await page.evaluate(() => { window.__suppressVideoFrames = false; });
  await seek(31); await pixels(31); await settlePrefetch();

  const failedFrame = url => new URL(url).pathname.endsWith('/frames/35');
  await page.route(failedFrame, route => route.fulfill({status:404,body:'TEST unavailable original'}));
  await page.locator('#scrub').evaluate(slider => { slider.value = '35'; slider.dispatchEvent(new Event('input',{bubbles:true})); });
  await page.waitForFunction(() => document.getElementById('status').textContent.includes('Could not verify'));
  assert.equal(await page.locator('#hit').isDisabled(),true);
  assert.equal(await page.locator('#forward-frame').isDisabled(),false);
  await page.locator('h1').click(); await page.keyboard.press('h');
  assert.equal(await page.locator('#event-count').textContent(),'2');
  await page.unroute(failedFrame);
  const failedRecoveryRequests = requests.length;
  await seek(31); await pixels(31); await settlePrefetch();
  assert.equal(requests.length,failedRecoveryRequests,'Returning to the displayed confirmed original must not refetch it');
  assert.equal(await page.locator('#loading').isHidden(),true);
  assert.equal(await page.locator('#scrub').inputValue(),'31');
  assert.equal(await page.locator('#display-mode').textContent(),'Original frame');
  assert.equal(await page.locator('#status').evaluate(status => status.classList.contains('error')),false);
  assert.doesNotMatch(await page.locator('#status').textContent(),/Could not verify/);

  // Actual backend rejection of an expired process token must retain the legacy draft.
  await page.route('**/labels', route => route.continue({headers:{...route.request().headers(),'x-label-token':'TEST expired process token'}}));
  await page.locator('#submit').click();
  await page.waitForFunction(() => document.getElementById('status').textContent.includes('Refresh this page'));
  assert.equal(await page.locator('#state').textContent(),'Unsaved draft');
  const retained = await page.evaluate(key => JSON.parse(localStorage.getItem(key)),key);
  assert.deepEqual(retained.events,[{frame_index:7,type:'hit'},{frame_index:7,type:'bounce'}]);
  assert.equal(retained.revision,0); assert.equal(retained.dirty,true); assert.equal(retained.selected,31); assert.equal(retained.annotator,legacy.annotator);
  assert.equal(JSON.parse(await readFile(path.join(labels,cid+'.json'),'utf8')).revision,1);
  await page.unroute('**/labels');
  await page.reload(); await waitOriginal(31); await settlePrefetch();
  assert.equal(await page.locator('#event-count').textContent(),'2');
  await page.locator('#submit').click();
  await page.waitForFunction(() => document.getElementById('status').textContent.includes('newer submission'));
  assert.equal(JSON.parse(await readFile(path.join(labels,cid+'.json'),'utf8')).revision,1);
  await page.locator('#reload-saved').click(); await page.locator('#confirm-reload').click(); await waitOriginal(0);
  assert.equal(await page.locator('#event-count').textContent(),'1');
  await seek(31); await page.locator('#hit').click(); await page.locator('#submit').click();
  await page.waitForFunction(() => document.getElementById('status').textContent.includes('Saved to local disk'));
  const saved = JSON.parse(await readFile(path.join(labels,cid+'.json'),'utf8'));
  assert.equal(saved.revision,2);
  const marked = saved.events.find(event => event.type === 'hit');
  for (const field of ['pts','time_base','original_frame_index','decoded_rgb_sha256']) assert.deepEqual(marked[field],detail.frames[31][field]);

  for (let index = 3; index < detail.frames.length; index += 3) { await seek(index); await settlePrefetch(); await checkCache(); }
  const cache = await checkCache();
  assert.equal(await page.evaluate(key => localStorage.getItem(key),otherKey),otherDraft);
  assert.equal(await page.evaluate(key => localStorage.getItem(key),positionKey),'0');
  assert(requests.every(request => new URL(request).searchParams.get('identity') === catalog.inventory_sha256));
  await page.screenshot({path:path.join(evidence,'performance-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({path:path.join(evidence,'performance-narrow.png'),fullPage:true});
  assert((await page.evaluate(() => window.__decodePeak)) <= 3,'Decode concurrency must stay bounded');
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,isolated_labels:labels,evidence,warm_steps_ms:warmStepsMs,frame_requests:requests.length,cache,checks:'One decode per fetched PNG, zero fetch/decode on confirmed marks, cached neighbor steps, late decode cancellation, accumulated rapid arrows, synchronous frozen pause, navigation during confirmation, unmapped preview rejection, unavailable-original rejection, exact displayed RGB/PTS, legacy stale drafts and annotator, actual expired-token rejection, revision conflict, immutable unrelated drafts, bounded cache and prefetch, narrow viewport'},null,2));
} finally { if (browser) await browser.close(); server.kill('SIGTERM'); }
