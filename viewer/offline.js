// Embedded examples work from file:// without a server or network requests.
const el = id => document.getElementById(id);

export function connectOffline(viewer) {
  const payload = el('embedded-gallery');
  if (!payload) return null;
  const gallery = JSON.parse(payload.textContent), clips = gallery.clips;
  const readOnly = gallery.readOnly === true;
  if (readOnly) {
    for (const id of ['file', 'video-file']) el(id).closest('label').hidden = true;
  }
  let selected = null;
  el('library').hidden = false;
  el('library').setAttribute('aria-label', 'Precomputed examples');
  for (const id of ['saved-runs-heading', 'runs-empty', 'runs', 'demo']) el(id).hidden = true;
  el('gallery-empty').hidden = !!clips.length;
  el('gallery-count').textContent = String(clips.length);
  el('connection').textContent = readOnly ? 'Example gallery' : 'Offline examples';
  el('privacy').textContent = readOnly
    ? 'Read-only examples. Run Tennis3D locally to reconstruct your own videos.'
    : 'Examples are embedded in this page. Add a matching local video for source playback.';
  function render() {
    el('gallery').replaceChildren(...clips.map(clip => {
      const button = document.createElement('button');
      button.className = 'run-card';
      button.dataset.clipId = clip.id;
      button.setAttribute('aria-current', String(selected === clip.id));
      for (const [name, text] of [['run-title', clip.title], ['run-caption', readOnly ? 'Precomputed · video and 3D playback' : 'Precomputed · source video optional']]) {
        if (!text) continue;
        const span = document.createElement('span');
        span.className = name; span.textContent = text; button.append(span);
      }
      button.addEventListener('click', () => {
        selected = clip.id;
        viewer.setVideo(readOnly ? clip.videoUrl : null, readOnly ? 'Source playback' : 'Add the matching local source video', readOnly ? clip.mediaOffset : 0);
        try { viewer.loadText(clip.resultText, clip.title, clip.timeline); }
        catch (error) { viewer.fail(clip.title, error); }
        render();
      });
      return button;
    }));
  }
  render();
  if (readOnly && clips.length) el('gallery').firstElementChild.click();
  return { deselect() { selected = null; render(); } };
}
