'use strict';
const $ = id => document.getElementById(id);
const video = $('video');
let catalog, detail, position = 0, selected = 0, events = [], revision = 0;
let dirty = false, ready = false, loading = false, clipLoading = false, saving = false, history = [], generation = 0;
let frameCallback = null, playbackFrame = null, targetFrame = 0, confirmedFrame = null, frameError = false;
let foreground = null, displayedImage = null, cacheBytes = 0, prefetchQueue = [];
const frameCache = new Map(), prefetchActive = new Set(), imageQueue = [], activeImages = new Set();
// Retained decoded pixels. The three loader slots also bound transient decode work.
const CACHE_BYTES = 64 * 1024 * 1024, CACHE_ENTRIES = 12;
function discardImage(entry) {
  if (frameCache.get(entry.index) === entry) frameCache.delete(entry.index);
  entry.controller.abort();
  const queued = imageQueue.indexOf(entry);
  if (queued >= 0) { imageQueue.splice(queued, 1); entry.reject(new DOMException('Aborted', 'AbortError')); }
  if (entry !== displayedImage) {
    if (entry.decodingImage) entry.decodingImage.src = '';
    if (entry.image) entry.image.src = '';
    entry.decodingImage = null; entry.image = null;
  }
  if (entry.bytes) { cacheBytes -= entry.bytes; entry.bytes = 0; }
  if (entry !== displayedImage && entry.url) { URL.revokeObjectURL(entry.url); entry.url = null; }
}
function trimCache(extraBytes = 0, extraEntries = 0, keep = null) {
  for (const entry of frameCache.values()) {
    if (cacheBytes + extraBytes <= CACHE_BYTES && frameCache.size + extraEntries <= CACHE_ENTRIES) break;
    if (entry !== displayedImage && entry !== keep && entry !== foreground) discardImage(entry);
  }
}
function touchImage(entry) {
  if (frameCache.get(entry.index) !== entry) return;
  frameCache.delete(entry.index); frameCache.set(entry.index, entry);
}
function cancelPending(keep = null) {
  prefetchQueue = [];
  for (const entry of frameCache.values()) if (!entry.image && entry !== keep) discardImage(entry);
}
function clearImages() {
  cancelPending();
  for (const entry of frameCache.values()) discardImage(entry);
  frameCache.clear(); foreground = null; confirmedFrame = null;
  if (displayedImage?.url) URL.revokeObjectURL(displayedImage.url);
  displayedImage = null; cacheBytes = 0;
  const old = $('original'), blank = new Image();
  blank.id = 'original'; blank.alt = old.alt; blank.hidden = true; old.replaceWith(blank);
}
function pumpImages() {
  while (activeImages.size < 3 && imageQueue.length) {
    const priority = imageQueue.indexOf(foreground);
    const entry = imageQueue.splice(priority < 0 ? 0 : priority, 1)[0];
    activeImages.add(entry);
    entry.load().then(entry.resolve, entry.reject).finally(() => { activeImages.delete(entry); pumpImages(); });
  }
}
function getImage(index, priority = false) {
  let entry = frameCache.get(index);
  if (entry) { touchImage(entry); if (priority) { foreground = entry; pumpImages(); } return entry; }
  trimCache(0, 1);
  entry = {index, controller:new AbortController(), image:null, bytes:0, url:null};
  frameCache.set(index, entry);
  const clip = detail.clip_id, identity = catalog.inventory_sha256;
  entry.promise = new Promise((resolve, reject) => { entry.resolve = resolve; entry.reject = reject; });
  entry.load = async () => {
    try {
      const response = await fetch(`/api/clips/${encodeURIComponent(clip)}/frames/${index}?identity=${encodeURIComponent(identity)}`, {signal:entry.controller.signal});
      if (!response.ok) throw new Error('Original frame unavailable');
      const blob = await response.blob();
      if (entry.controller.signal.aborted) throw new DOMException('Aborted', 'AbortError');
      entry.url = URL.createObjectURL(blob);
      const image = new Image(); entry.decodingImage = image; image.src = entry.url;
      await image.decode();
      if (entry.controller.signal.aborted) throw new DOMException('Aborted', 'AbortError');
      const bytes = image.naturalWidth * image.naturalHeight * 4;
      trimCache(bytes, 0, entry);
      if (entry !== foreground && cacheBytes + bytes > CACHE_BYTES) throw new DOMException('Prefetch memory limit', 'AbortError');
      entry.decodingImage = null; entry.image = image; entry.bytes = bytes; cacheBytes += bytes;
      return entry;
    } catch (error) { discardImage(entry); throw error; }
  };
  imageQueue.push(entry); if (priority) foreground = entry; pumpImages();
  return entry;
}
function pumpPrefetch() {
  while (prefetchActive.size < 2 && prefetchQueue.length) {
    const index = prefetchQueue.shift();
    if (frameCache.has(index)) continue;
    const entry = getImage(index); prefetchActive.add(entry);
    entry.promise.catch(() => {}).finally(() => { prefetchActive.delete(entry); pumpPrefetch(); });
  }
}
function prefetchNeighbors(index) {
  prefetchQueue = [index - 1, index + 1].filter(i => i >= 0 && i < detail.frames.length);
  pumpPrefetch();
}
const message = (text, error = false) => { $('status').textContent = text; $('status').classList.toggle('error', error); };
const draftKey = () => `tennis-events:${catalog.inventory_sha256}:${detail.clip_id}`;
function storeDraft() {
  try {
    localStorage.setItem(draftKey(), JSON.stringify({ events, revision, dirty, selected, annotator: $('annotator').value }));
    localStorage.setItem(`tennis-events-position:${catalog.inventory_sha256}`, String(position));
  } catch { message('Browser draft storage is unavailable. Submit before leaving this clip.', true); }
}
function buttons() {
  $('hit').disabled = $('bounce').disabled = !ready || loading || saving;
  $('submit').disabled = loading || saving || !ready;
  $('play').disabled = saving || !detail || (video.paused && (loading || !ready));
  $('undo').disabled = history.length === 0 || saving;
  $('back-frame').disabled = saving || clipLoading || !detail || targetFrame <= 0;
  $('forward-frame').disabled = saving || clipLoading || !detail || targetFrame >= detail.frames.length - 1;
  $('previous').disabled = position === 0 || saving;
  $('next').disabled = !catalog || position === catalog.clips.length - 1 || saving;
  $('gallery').disabled = $('reload-saved').disabled = saving;
  $('scrub').disabled = saving || clipLoading || !detail;
  $('state').textContent = dirty ? 'Unsaved draft' : revision ? `Submitted · revision ${revision}` : 'Not submitted';
  $('play').textContent = video.paused ? 'Play' : 'Pause';
}
function render() {
  $('events').replaceChildren(); $('markers').replaceChildren(); $('event-count').textContent = events.length;
  for (const event of [...events].sort((a,b) => a.frame_index-b.frame_index || a.type.localeCompare(b.type))) {
    const index = event.frame_index;
    const text = `${event.type === 'hit' ? 'Hit' : 'Bounce'} · frame ${index + 1} · ${detail.times[index].toFixed(3)} s`;
    const item = document.createElement('li'), jump = document.createElement('button'), remove = document.createElement('button');
    jump.textContent = text; jump.className = `event-${event.type}`; jump.onclick = () => selectFrame(index);
    remove.textContent = '×'; remove.setAttribute('aria-label', `Delete ${text}`); remove.onclick = () => edit(events.filter(e => e !== event));
    remove.disabled = saving; jump.disabled = saving;
    item.append(jump,remove); $('events').append(item);
    const marker = document.createElement('button'); marker.className = `marker ${event.type}`; marker.title = text; marker.setAttribute('aria-label', text);
    marker.style.left = `${index / Math.max(1, detail.frames.length - 1) * 100}%`; marker.onclick = () => selectFrame(index); marker.disabled = saving; $('markers').append(marker);
  }
  buttons();
}
function readout() {
  const f = detail.frames[selected];
  $('frame-readout').textContent = `Frame ${selected + 1} / ${detail.frames.length} · ${detail.times[selected].toFixed(3)} s · source frame ${f.original_frame_index}`;
  $('scrub').value = selected;
}
function stopPlayback() {
  video.pause();
  if (frameCallback !== null) video.cancelVideoFrameCallback(frameCallback);
  frameCallback = null; playbackFrame = null;
  buttons();
}
function finishOriginalSelection() {
  ready = true; loading = false;
  $('loading').hidden = true; $('loading').textContent = 'Loading original frame…';
  video.hidden = true; $('display-mode').textContent = 'Original frame'; readout();
  if (frameError) { message('Original frame confirmed. Labels have not changed.'); frameError = false; }
}
async function selectFrame(index) {
  if (!detail || saving || clipLoading) return false;
  index = Math.max(0, Math.min(detail.frames.length - 1, index));
  stopPlayback();
  targetFrame = index;
  if (!loading && confirmedFrame?.clip === detail.clip_id && confirmedFrame.index === index && !$('original').hidden) {
    selected = index; finishOriginalSelection(); storeDraft(); buttons(); return true;
  }
  const request = ++generation, clip = detail.clip_id;
  let entry = frameCache.get(index);
  foreground = entry || null; cancelPending(entry);
  ready = false; loading = true; $('loading').textContent = 'Loading original frame…'; $('loading').hidden = false;
  $('scrub').value = index; buttons();
  try {
    entry = entry || getImage(index, true); foreground = entry; pumpImages();
    await entry.promise;
    if (request !== generation) return false;
    // Swap the already decoded node. No asynchronous write can replace a newer frame.
    const image = entry.image, previous = displayedImage;
    image.id = 'original'; image.alt = 'Selected original video frame'; image.hidden = false;
    $('original').replaceWith(image); displayedImage = entry;
    if (previous && !frameCache.has(previous.index) && previous.url) { URL.revokeObjectURL(previous.url); previous.url = null; }
    video.hidden = true; selected = index; confirmedFrame = {clip,index};
    $('display-mode').textContent = 'Original frame'; readout();
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    if (request !== generation) return false;
    foreground = null; finishOriginalSelection();
    touchImage(entry); trimCache(); storeDraft(); buttons(); prefetchNeighbors(index);
    return true;
  } catch (error) {
    if (request !== generation) return false;
    foreground = null; loading = false; ready = false; frameError = true; $('loading').textContent = 'Original frame unavailable';
    message('Could not verify the selected original frame. Labels have not changed. Try seeking again.', true); buttons(); return false;
  }
}
function edit(next) {
  if (saving) return;
  history.push(structuredClone(events)); events = next; dirty = true; storeDraft(); render();
}
async function mark(kind) {
  if (!ready || loading || saving) return;
  const clip = detail.clip_id, index = video.paused ? confirmedFrame?.index : playbackFrame;
  if (index === null || index === undefined) return;
  if (!await selectFrame(index) || detail.clip_id !== clip || confirmedFrame?.index !== index || loading) return;
  if (events.some(e => e.frame_index === selected && e.type === kind)) { message('That event is already marked on this frame.'); return; }
  edit([...events, {frame_index:selected,type:kind}]); message(`${kind === 'hit' ? 'Hit' : 'Bounce'} marked at frame ${selected + 1}. Submit to save to disk.`);
}
function nearest(time) {
  let lo = 0, hi = detail.playback_times.length - 1;
  while (lo < hi) { const mid = Math.floor((lo + hi) / 2); if (detail.playback_times[mid] < time) lo = mid + 1; else hi = mid; }
  if (lo > 0 && Math.abs(detail.playback_times[lo - 1] - time) < Math.abs(detail.playback_times[lo] - time)) lo--;
  return Math.abs(detail.playback_times[lo] - time) <= 0.002 ? lo : null;
}
async function togglePlayback() {
  if (saving || !detail) return;
  if (!video.paused) {
    const index = playbackFrame;
    // Pause in this input turn, before the original-frame request or decode.
    stopPlayback();
    if (index !== null) await selectFrame(index);
    else { generation++; ready = false; message('The paused preview has no confirmed source frame. Step or seek before labeling.', true); buttons(); }
    return;
  }
  if (!ready || loading) return;
  if (!video.requestVideoFrameCallback) { message('This browser cannot identify presented video frames. Frame seeking and labeling still work.', true); return; }
  const request = ++generation;
  cancelPending(); foreground = null; confirmedFrame = null;
  video.currentTime = detail.playback_times[selected];
  playbackFrame = null; ready = false; video.hidden = false; $('original').hidden = true; buttons();
  const presented = (_, metadata) => {
    if (request !== generation) return;
    const index = nearest(metadata.mediaTime);
    if (index === null) { stopPlayback(); ready = false; message('Playback frame mapping failed. Seek to an original frame before labeling.', true); buttons(); return; }
    selected = index; targetFrame = index; playbackFrame = index; ready = true;
    $('original').hidden = true; video.hidden = false; $('display-mode').textContent = 'Playback preview'; readout(); buttons();
    frameCallback = video.requestVideoFrameCallback(presented);
  };
  frameCallback = video.requestVideoFrameCallback(presented);
  try { await video.play(); buttons(); } catch { stopPlayback(); await selectFrame(selected); message('Video playback failed. Original-frame seeking remains available.', true); }
}
async function loadClip(index, discard = false) {
  if (saving) return;
  if (detail) storeDraft();
  stopPlayback(); generation++; clearImages(); ready = false; loading = true; clipLoading = true; frameError = false; video.hidden = true;
  $('loading').textContent = 'Loading clip…'; $('loading').hidden = false; buttons();
  const request = generation;
  try {
    const response = await fetch(`/api/clips/${encodeURIComponent(catalog.clips[index].clip_id)}`);
    if (!response.ok) throw new Error('Clip data could not be loaded.');
    const data = await response.json();
    if (request !== generation) return;
    clipLoading = false; position = index; detail = data; revision = data.labels.revision; events = data.labels.events.map(e => ({frame_index:e.frame_index,type:e.type})); dirty = false; history = []; selected = 0; targetFrame = 0;
    if (!discard) {
      const draft = JSON.parse(localStorage.getItem(draftKey()) || 'null');
      if (draft && draft.dirty && Array.isArray(draft.events) && draft.events.every(e => Number.isInteger(e.frame_index) && e.frame_index >= 0 && e.frame_index < data.frames.length && ['hit','bounce'].includes(e.type))) {
        events = draft.events; revision = draft.revision; dirty = true; selected = Number.isInteger(draft.selected) ? Math.max(0, Math.min(data.frames.length - 1, draft.selected)) : 0; targetFrame = selected; $('annotator').value = draft.annotator || 'Local annotator';
        message(revision !== data.labels.revision ? 'A newer disk submission exists. Your draft is retained. Reload saved labels explicitly to resolve the conflict.' : 'Recovered unsaved draft from this browser.', revision !== data.labels.revision);
      } else message(data.labels.submitted ? 'Previously submitted labels loaded from disk.' : 'Ready to label. Submit also accepts an empty event list.');
    } else message('Latest submitted labels loaded.');
    $('match').textContent = catalog.clips[index].recording_id.replaceAll('_',' ');
    $('clip-name').textContent = `${detail.clip_id} · clip ${index + 1} of ${catalog.clips.length}`;
    $('gallery').value = index; $('scrub').max = detail.frames.length - 1;
    video.src = `/api/clips/${encodeURIComponent(detail.clip_id)}/video`; video.load();
    render(); await selectFrame(selected); updateProgress();
  } catch (error) { if (request !== generation) return; clipLoading = false; loading = false; ready = false; message(error.message, true); buttons(); }
}
function updateProgress() {
  const count = catalog.clips.filter(c => c.submitted).length;
  $('progress').textContent = `${count} / ${catalog.clips.length} submitted · clip ${position + 1}`;
  for (let i = 0; i < catalog.clips.length; i++) $('gallery').options[i].textContent = `${i + 1}. ${catalog.clips[i].clip_id}${catalog.clips[i].submitted ? ' · submitted' : ''}`;
}
async function submit() {
  if (saving || loading || !ready) return;
  if (!video.paused && (playbackFrame === null || !await selectFrame(playbackFrame))) return;
  saving = true; render(); message('Saving to local disk…');
  try {
    const response = await fetch(`/api/clips/${encodeURIComponent(detail.clip_id)}/labels`, {method:'POST',headers:{'Content-Type':'application/json','X-Label-Token':catalog.token},body:JSON.stringify({revision,identity:detail.identity,events,annotator:$('annotator').value})});
    if (response.status === 403) throw new Error('The labeling server restarted or this page has an expired save token. Refresh this page and submit again. Your browser draft is retained.');
    const receipt = await response.json();
    if (!response.ok || !receipt.saved || receipt.labels.clip_id !== detail.clip_id || receipt.labels.identity.inventory_sha256 !== catalog.inventory_sha256) throw new Error(receipt.error || 'Save receipt was not confirmed.');
    revision = receipt.labels.revision; dirty = false; history = []; detail.labels = receipt.labels;
    catalog.clips[position].submitted = true; catalog.clips[position].revision = revision;
    storeDraft(); updateProgress(); message(`Saved to local disk · revision ${revision} · ${events.length} events.`);
  } catch (error) { message(error.message, true); storeDraft(); }
  finally { saving = false; render(); }
}
$('hit').onclick = () => mark('hit'); $('bounce').onclick = () => mark('bounce'); $('play').onclick = togglePlayback;
$('back-frame').onclick = () => selectFrame(targetFrame - 1); $('forward-frame').onclick = () => selectFrame(targetFrame + 1);
$('scrub').oninput = () => { if (!saving) selectFrame(Number($('scrub').value)); };
$('speed').onchange = () => { video.playbackRate = Number($('speed').value); };
$('undo').onclick = () => { if (!saving && history.length) { events = history.pop(); dirty = true; storeDraft(); render(); message('Last edit undone.'); } };
$('submit').onclick = submit; $('previous').onclick = () => loadClip(position - 1); $('next').onclick = () => loadClip(position + 1);
$('gallery').onchange = () => loadClip(Number($('gallery').value)); $('annotator').oninput = storeDraft;
$('reload-saved').onclick = () => $('reload-dialog').showModal(); $('cancel-reload').onclick = () => $('reload-dialog').close();
$('confirm-reload').onclick = () => { $('reload-dialog').close(); loadClip(position, true); };
video.onended = () => { if (playbackFrame !== null) selectFrame(playbackFrame); else { stopPlayback(); ready = false; buttons(); } };
window.addEventListener('keydown', event => {
  if (event.target.matches('input,select,textarea') || $('reload-dialog').open || event.ctrlKey || event.metaKey || event.altKey) return;
  const key = event.key.toLowerCase();
  if ([' ','h','b','arrowleft','arrowright'].includes(key)) event.preventDefault();
  if (key === ' ') togglePlayback(); if (key === 'h') mark('hit'); if (key === 'b') mark('bounce');
  if (!saving && !clipLoading && detail && key === 'arrowleft') selectFrame(targetFrame - 1);
  if (!saving && !clipLoading && detail && key === 'arrowright') selectFrame(targetFrame + 1);
});
window.addEventListener('beforeunload', event => { if (detail) storeDraft(); if (dirty) { event.preventDefault(); event.returnValue = ''; } });
(async () => {
  try {
    const response = await fetch('/api/gallery'); if (!response.ok) throw new Error('Gallery could not be loaded.'); catalog = await response.json();
    catalog.clips.forEach((c,i) => { const option = document.createElement('option'); option.value = i; option.textContent = c.clip_id; $('gallery').append(option); });
    const previous = localStorage.getItem(`tennis-events-position:${catalog.inventory_sha256}`);
    const index = previous !== null && Number.isInteger(Number(previous)) && Number(previous) >= 0 && Number(previous) < catalog.clips.length ? Number(previous) : Math.max(0,catalog.clips.findIndex(c => !c.submitted));
    await loadClip(index);
  } catch (error) { message(error.message, true); }
})();
