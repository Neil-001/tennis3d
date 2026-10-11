import { parseResult, validateResult } from './validate.js';
import { sample, snap } from './playback.js';
import { createScene } from './scene.js';
import { drawProjection } from './projection.js';
import { eventColors } from './court.js';
import { connect } from './server.js';
import { connectOffline } from './offline.js';
import { validateTimeline } from './timeline.js';
import { showFrames, showSummary } from './inspect.js';
import example from './example.json';
const el = id => document.getElementById(id);
const video = el('video');
// `mediaOffset` is the video time of the first source frame.
let sourceTimeline, scene, result, resultText, resultName, time = 0, playing = false, lastWallTime, loadSerial = 0, mediaOffset = 0, videoUrl;
const webglMessage = 'This browser could not start WebGL. Enable hardware acceleration or use a WebGL-capable browser.';
try { scene = createScene(el('scene')); }
catch { el('error').hidden = false; el('error').textContent = webglMessage; }
const duration = () => result?.frames.at(-1)?.t_s ?? 0;
// Browsers keep frame times in whole microseconds and store a seek target up to a few
// microseconds low, which presents the frame before the one at `time`. Seeks aim this much
// later, far less than the time between two frames.
const seekMargin = 1e-3;
const videoTime = () => time + mediaOffset + seekMargin;
const hasVideo = () => !!video.getAttribute('src') && video.readyState > 0 && !video.error;
const playable = () => !!result && duration() > 0 && (hasVideo() || (!!scene && result.status !== 'failed'));
function pause() {
  playing = false; el('play').textContent = 'Play';
  if (!video.paused) video.pause();
}
function seek(value) {
  time = Math.min(duration(), Math.max(0, value));
  if (hasVideo()) video.currentTime = videoTime();
  draw();
}
function draw() {
  if (!result) return;
  const { index, time: at } = snap(result.frames, time), frame = result.frames[index];
  const current = sample(result.frames, at);
  // A declared input-contract failure is a report, not a playable reconstruction.
  if (result.status === 'failed') { current.xyz = null; current.state = 'unavailable'; }
  scene?.update(result.status === 'failed' ? [] : result.frames, at, current, Number(el('trail').value));
  el('time').value = String(time);
  el('frame-back').disabled = !frame || (index === 0 && at <= frame.t_s);
  el('frame-forward').disabled = !frame || index >= result.frames.length - 1;
  el('timestamp').textContent = `${time.toFixed(3)} / ${duration().toFixed(3)} s`;
  el('position').textContent = current.xyz ? `X ${current.xyz[0].toFixed(3)} · Y ${current.xyz[1].toFixed(3)} · Z ${current.xyz[2].toFixed(3)} m` : 'XYZ unavailable';
  el('state').textContent = frame ? `Playback ${current.state} · Frame ${index} at ${frame.t_s.toFixed(3)} s: ${frame.reconstruction}/${frame.observation}${frame.reason_codes.length ? ' · ' + frame.reason_codes.join(', ') : ''}` : 'No source frames';
  const sourceRow = sourceTimeline?.frames[index];
  el('source-timing').hidden = !sourceRow;
  el('source-timing').textContent = sourceRow ? `Source PTS ${sourceRow.pts} × ${sourceRow.time_base[0]}/${sourceRow.time_base[1]} s${sourceRow.original_frame_index === undefined ? '' : ` · Original frame ${sourceRow.original_frame_index}`}` : '';
  if (result.synthetic_diagnostics) {
    drawProjection(el('projection'), result, index);
    const p = result.synthetic_diagnostics.frames[index];
    el('projection-time').textContent = p ? `Source frame ${index} at ${p.t_s.toFixed(3)} s. Projections are not interpolated.` : 'No source frames';
    el('projection-state').textContent = p ? `${p.ball_clean_uv ? 'Clean projection available' : 'Clean projection unavailable'}; ${p.ball_observed_uv ? 'observation available' : 'observation missing'}. Image ${result.source.width} × ${result.source.height} px.` : '';
  }
}
function setVideo(url, label, offset = 0) {
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = url?.startsWith('blob:') ? url : null;
  mediaOffset = offset;
  el('video-error').hidden = true;
  if (url) video.src = url; else { video.removeAttribute('src'); video.load(); }
  el('video-empty').hidden = !!url;
  el('video-label').textContent = label;
}
function clear(message) {
  sourceTimeline = null; el('source-timing').hidden = true;
  result = resultText = null; pause(); time = 0;
  el('source').textContent = message; el('kind').textContent = 'No result yet';
  el('time').disabled = el('play').disabled = el('export').disabled = el('frame-back').disabled = el('frame-forward').disabled = true;
  el('result-summary').hidden = el('diagnostic').hidden = true;
  el('metadata').textContent = 'No file loaded.';
  el('position').textContent = 'XYZ unavailable'; el('state').textContent = 'No source frame';
  showFrames(null, seek);
  scene?.setEvents([]); scene?.update([], 0, { index: -1, xyz: null, state: 'unavailable' }, 0);
}
function load(data, name, text, timeline = null) {
  sourceTimeline = validateTimeline(data, timeline);
  result = data; resultName = name; resultText = text ?? JSON.stringify(data); pause(); time = 0;
  el('error').hidden = !!scene;
  if (!scene) el('error').textContent = webglMessage;
  // Uploaded videos are stored under a generated name that is not worth showing.
  el('source').textContent = /^source\.\w+$/.test(data.source.name) ? name : `${name} · ${data.source.name}`;
  const kind = { synthetic_ground_truth: 'Synthetic ground truth', fixture: 'Illustrative fixture', model_estimate: 'Model estimate' }[data.provenance.trajectory_kind];
  el('kind').textContent = `${kind} · ${data.status} · ${(data.summary.valid_frame_fraction * 100).toFixed(1)}% coverage${data.provenance.model_label ? ` · ${data.provenance.model_label}` : ''}`;
  el('metadata').textContent = JSON.stringify({ source: data.source, provenance: data.provenance, summary: data.summary, failure_reasons: data.failure_reasons }, null, 2);
  el('time').max = String(duration());
  el('time').disabled = !data.frames.length;
  el('play').disabled = !playable();
  el('export').disabled = false;
  const diagnostic = data.synthetic_diagnostics;
  el('diagnostic').hidden = !diagnostic;
  el('events').replaceChildren();
  for (const event of diagnostic?.events ?? []) {
    const button = document.createElement('button');
    button.textContent = `${event.kind} ${event.t_s.toFixed(3)} s`;
    button.style.color = eventColors[event.kind];
    button.addEventListener('click', () => { pause(); seek(event.t_s); });
    el('events').append(button);
  }
  showSummary(data); showFrames(data, value => { pause(); seek(value); });
  scene?.setEvents(data.status === 'failed' ? [] : diagnostic?.events ?? []);
  scene?.preset('perspective'); seek(0);
}
function fail(name, error) {
  el('error').textContent = `Could not load ${name}: ${error.message}`;
  el('error').hidden = false;
}
el('file').addEventListener('change', async event => {
  const file = event.target.files[0];
  if (!file) return;
  const serial = ++loadSerial;
  pause();
  try {
    const text = await file.text();
    if (serial !== loadSerial) return;
    const data = parseResult(text);
    library?.deselect(); setVideo(null, 'Optional playback');
    load(data, file.name, text);
  } catch (error) {
    if (serial === loadSerial) fail(file.name, error);
  }
  event.target.value = '';
});
el('video-file').addEventListener('change', event => {
  const file = event.target.files[0];
  if (file) setVideo(URL.createObjectURL(file), 'Local video · assumed to start at the first source frame');
  event.target.value = '';
});
el('demo').addEventListener('click', () => {
  ++loadSerial; library?.deselect(); setVideo(null, 'Optional playback');
  load(validateResult(structuredClone(example)), 'Built-in analytical example');
});
el('export').addEventListener('click', () => {
  if (!resultText) return;
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([resultText], { type: 'application/json' }));
  link.download = `${(resultName || 'tennis3d').replace(/\.[^.]+$/, '').replace(/[^\w.-]+/g, '_')}.result.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});
el('play').addEventListener('click', () => {
  if (playing) return pause();
  if (time >= duration()) seek(0);
  playing = true; lastWallTime = performance.now(); el('play').textContent = 'Pause';
  if (hasVideo()) video.play().catch(() => pause());
  draw();
});
el('time').addEventListener('input', () => { pause(); seek(Number(el('time').value)); });
// Step by source frame. Back from between two frames lands on the earlier one first.
function step(delta) {
  if (!result?.frames.length) return;
  pause();
  const { index, time: at } = snap(result.frames, time);
  const target = delta < 0 && at > result.frames[index].t_s ? index : index + delta;
  seek(result.frames[Math.min(result.frames.length - 1, Math.max(0, target))].t_s);
}
el('frame-back').addEventListener('click', () => step(-1));
el('frame-forward').addEventListener('click', () => step(1));
window.addEventListener('keydown', event => {
  if (event.target.matches('input,select,textarea,video') || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
  event.preventDefault(); step(event.key === 'ArrowLeft' ? -1 : 1);
});
el('trail').addEventListener('change', draw);
el('speed').addEventListener('change', () => { video.playbackRate = Number(el('speed').value); });
el('court').addEventListener('change', () => scene?.setCourtVisible(el('court').checked));
for (const button of document.querySelectorAll('[data-view]')) button.addEventListener('click', () => scene?.preset(button.dataset.view));
document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
// The video's own controls drive the same clock as the transport bar.
video.addEventListener('play', () => { playing = true; el('play').textContent = 'Pause'; });
video.addEventListener('pause', () => { playing = false; el('play').textContent = 'Play'; });
// A video that ends pauses before the next tick, so its last position is read here.
for (const type of ['seeked', 'ended']) video.addEventListener(type, () => {
  if (!result) return;
  const at = video.currentTime - mediaOffset;
  // The video is where `seek` put it, so `time` is already the exact frame time.
  if (type === 'seeked' && Math.abs(at - seekMargin - time) < seekMargin / 2) return;
  time = Math.min(duration(), Math.max(0, at)); draw();
});
video.addEventListener('loadedmetadata', () => {
  video.playbackRate = Number(el('speed').value);
  if (result) { video.currentTime = videoTime(); el('play').disabled = !playable(); }
});
video.addEventListener('error', () => {
  if (!video.getAttribute('src')) return;
  el('video-error').textContent = 'This browser cannot play the video. The 3D reconstruction is unaffected; try Chrome, Edge, Firefox or Safari.';
  el('video-error').hidden = false;
  if (result) el('play').disabled = !playable();
});
function tick(now) {
  if (playing && result) {
    if (hasVideo()) {
      time = Math.min(duration(), Math.max(0, video.currentTime - mediaOffset));
      if (video.ended || time >= duration()) pause();
    } else {
      time = Math.min(duration(), time + (now - lastWallTime) / 1000 * Number(el('speed').value));
      if (time === duration()) pause();
    }
    draw();
  }
  lastWallTime = now;
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
// Served by the local app: add processing, saved runs and examples.
const viewer = {
  clear, setVideo, fail,
  loadText(text, name, timeline) { ++loadSerial; load(parseResult(text), name, text, timeline); },
  loaded: () => !!result,
};
const library = connectOffline(viewer) ?? (/^https?:$/.test(location.protocol) ? connect(viewer) : null);
