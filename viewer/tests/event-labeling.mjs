import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp, mkdir, readFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const root = await mkdtemp(path.join(os.tmpdir(), 'tennis-events-test-'));
const evidence = process.env.EVENT_EVIDENCE || root;
await mkdir(evidence, {recursive:true});
const python = process.env.EVENT_PYTHON || 'uv';
const prefix = process.env.EVENT_PYTHON ? [] : ['run','--extra','labeling','python'];
async function run(args) {
  return await new Promise((resolve,reject) => {
    const child = spawn(python,[...prefix,...args],{cwd:repo}); let out='';
    child.stdout.on('data',x=>out+=x); child.stderr.on('data',x=>out+=x);
    child.on('exit',code=>code === 0 ? resolve(out) : reject(new Error(out)));
  });
}
let inventory = process.env.EVENT_INVENTORY;
if (!inventory) { await run(['tests/event_labeling_fixture.py',root]); inventory = path.join(root,'inventory.json'); }
const labels = path.join(root,'isolated-test-labels');
const server = spawn(python,[...prefix,'-m','tennis3d.labeling.dashboard','--inventory',inventory,'--cache',process.env.EVENT_CACHE || path.join(root,'previews'),'--labels',labels,'--port','0'],{cwd:repo});
let output='';
server.stdout.on('data',data=>output+=data); server.stderr.on('data',data=>output+=data);
const url = await new Promise((resolve,reject)=> {
  const deadline = setTimeout(()=>reject(new Error(`Server startup timed out: ${output}`)),120000);
  server.on('exit',code=>{clearTimeout(deadline);reject(new Error(`Server exited ${code}: ${output}`));});
  server.stdout.on('data',()=>{const match=output.match(/Labeling dashboard: (http:\/\/localhost:\d+)/);if(match){clearTimeout(deadline);resolve(match[1]);}});
});
const browser = await chromium.launch({headless:true});
const page = await browser.newPage({viewport:{width:1440,height:1000}});
page.on('dialog', dialog => dialog.accept());
const errors=[];page.on('pageerror',error=>errors.push(error.message));
const frame = () => page.locator('#frame-readout').textContent();
const waitOriginal = async () => {
  await page.locator('#hit').waitFor({state:'visible'});
  await page.waitForFunction(()=>!document.getElementById('hit').disabled && !document.getElementById('original').hidden && document.getElementById('loading').hidden);
};
try {
  await page.goto(url); await waitOriginal();
  const catalog = await (await page.request.get(url+'/api/gallery')).json();
  const cid = catalog.clips[0].clip_id;
  const detail = await (await page.request.get(url+`/api/clips/${cid}`)).json();
  assert.match(await frame(), /Frame 1 \//);
  // Theme, typeface and its license come from this server only.
  assert.equal(await page.evaluate(async()=>{await document.fonts.ready;return document.fonts.check('14px "Google Sans Code"');}),true);
  assert.match(await (await page.request.get(url+'/font-license.txt')).text(),/SIL OPEN FONT LICENSE Version 1\.1/);
  await page.locator('#annotator').fill('TEST browser verification, not human truth');
  await page.locator('#play').click();
  await page.waitForFunction(()=>!document.getElementById('video').paused && !document.getElementById('video').hidden && document.getElementById('video').currentTime > .15);
  await page.locator('#play').click(); await waitOriginal();
  assert(await page.locator('#video').evaluate(v=>v.paused));
  await page.locator('#scrub').evaluate((s,max)=>{s.value=String(Math.floor(max*.3));s.dispatchEvent(new Event('input',{bubbles:true}));},detail.frames.length-1);
  await waitOriginal();
  const chosen = Math.floor((detail.frames.length-1)*.3);
  assert.match(await frame(),new RegExp(`Frame ${chosen+1} /`));
  await page.locator('#forward-frame').click();await waitOriginal();
  assert.match(await frame(),new RegExp(`Frame ${chosen+2} /`));
  await page.locator('#back-frame').click();await waitOriginal();
  await page.locator('h1').click();await page.keyboard.press('h');await waitOriginal();
  await page.waitForFunction(()=>document.getElementById('event-count').textContent==='1');
  await page.locator('#hit').click();await waitOriginal();assert.equal(await page.locator('#event-count').textContent(),'1');
  await page.locator('#bounce').click();await waitOriginal();assert.equal(await page.locator('#event-count').textContent(),'2');
  await page.locator('#undo').click();assert.equal(await page.locator('#event-count').textContent(),'1');
  await page.locator('#bounce').click();await waitOriginal();
  await page.locator('#events li').first().getByRole('button',{name:/Delete/}).click();assert.equal(await page.locator('#event-count').textContent(),'1');
  await page.locator('#undo').click();assert.equal(await page.locator('#event-count').textContent(),'2');
  await page.locator('#forward-frame').click();await waitOriginal();
  await page.locator('#markers button').first().click();await waitOriginal();assert.match(await frame(),new RegExp(`Frame ${chosen+1} /`));
  await page.locator('#submit').click();await page.getByRole('status').filter({hasText:'Saved to local disk'}).waitFor();
  let saved=JSON.parse(await readFile(path.join(labels,cid+'.json'),'utf8'));
  assert.equal(saved.revision,1);assert.equal(saved.events.length,2);
  for(const event of saved.events){assert.equal(event.frame_index,chosen);assert.equal(event.pts,detail.frames[chosen].pts);assert.deepEqual(event.time_base,detail.frames[chosen].time_base);assert.equal(event.original_frame_index,detail.frames[chosen].original_frame_index);assert.equal(event.decoded_rgb_sha256,detail.frames[chosen].decoded_rgb_sha256);}
  await page.screenshot({path:path.join(evidence,'desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(evidence,'narrow.png'),fullPage:true});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth));
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('#next').click();await waitOriginal();assert.match(await page.locator('#clip-name').textContent(),/clip 2 of/);
  await page.reload();await waitOriginal();assert.match(await page.locator('#progress').textContent(),/^1 \//);
  await page.locator('#hit').click();await waitOriginal();
  await page.locator('#previous').click();await waitOriginal();assert.equal(await page.locator('#event-count').textContent(),'2');
  await page.locator('#next').click();await waitOriginal();assert.equal(await page.locator('#event-count').textContent(),'1');assert.equal(await page.locator('#state').textContent(),'Unsaved draft');
  await page.locator('#events li').first().getByRole('button',{name:/Delete/}).click();
  await page.locator('#submit').click();await page.getByRole('status').filter({hasText:'Saved to local disk'}).waitFor();
  const second=JSON.parse(await readFile(path.join(labels,catalog.clips[1].clip_id+'.json'),'utf8'));assert.deepEqual(second.events,[]);assert.equal(second.submitted,true);
  await page.locator('#previous').click();await waitOriginal();
  await page.locator('#events li').first().getByRole('button',{name:/Delete/}).click();
  await page.locator('#submit').click();await page.getByRole('status').filter({hasText:'Saved to local disk'}).waitFor();
  saved=JSON.parse(await readFile(path.join(labels,cid+'.json'),'utf8'));assert.equal(saved.revision,2);assert.equal(saved.events.length,1);
  await page.evaluate(async ({cid,token,identity,events})=>{
    const response=await fetch(`/api/clips/${cid}/labels`,{method:'POST',headers:{'Content-Type':'application/json','X-Label-Token':token},body:JSON.stringify({revision:2,identity,events,annotator:'TEST competing tab'})});if(!response.ok)throw new Error('Test competing submission failed');
  },{cid,token:catalog.token,identity:detail.identity,events:saved.events.map(e=>({frame_index:e.frame_index,type:e.type}))});
  await page.locator('#forward-frame').click();await waitOriginal();await page.locator('#hit').click();await waitOriginal();
  await page.locator('#submit').click();await page.getByRole('status').filter({hasText:'newer submission'}).waitFor();
  assert.equal(JSON.parse(await readFile(path.join(labels,cid+'.json'),'utf8')).revision,3);
  await page.reload();await waitOriginal();assert.equal(await page.locator('#event-count').textContent(),'2');assert.equal(await page.locator('#state').textContent(),'Unsaved draft');
  await page.locator('#reload-saved').click();await page.locator('#confirm-reload').click();await waitOriginal();assert.equal(await page.locator('#event-count').textContent(),'1');
  await page.locator('#hit').click();await waitOriginal();
  await page.route('**/labels',route=>route.fulfill({status:500,contentType:'application/json',body:JSON.stringify({error:'TEST disk unavailable; draft retained'})}));
  await page.locator('#submit').click();await page.getByRole('status').filter({hasText:'disk unavailable'}).waitFor();assert.equal(await page.locator('#state').textContent(),'Unsaved draft');
  await page.unroute('**/labels');
  assert.equal(JSON.parse(await readFile(path.join(labels,cid+'.json'),'utf8')).revision,3);
  // Exercise actual browser codec and original-frame decoding throughout a supplied real inventory.
  if(process.env.EVENT_INVENTORY){
    for(let i=0;i<catalog.clips.length;i++){
      await page.locator('#gallery').selectOption(String(i));await waitOriginal();
      await page.locator('#play').click();await page.waitForFunction(()=>!document.getElementById('video').paused && !document.getElementById('video').hidden && document.getElementById('video').currentTime>.08);
      await page.locator('#play').click();await waitOriginal();
      await page.locator('#forward-frame').click();await waitOriginal();
      assert(await page.locator('#original').evaluate(img=>img.complete && img.naturalWidth>0));
    }
  }
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,url,isolated_labels:labels,evidence,clip_count:catalog.clips.length,checks:'VFR playback, pause, scrub, original-frame steps, exact Hit/Bounce source mapping, duplicate/edit/undo/marker seek, empty save, disk receipt, resume, drafts, correction, conflicts, save error, desktop/narrow'},null,2));
} finally { await browser.close(); server.kill('SIGTERM'); }
