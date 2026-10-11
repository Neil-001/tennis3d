// Client for the loopback app: upload, progress, cancel, saved runs and examples.
const el = id => document.getElementById(id);
const headers = { 'X-Tennis3D': '1' };
const stageText = {
  checking: 'Checking the video',
  playback: 'Preparing the video for playback',
  detect: 'Finding the ball and the court',
  calibrate: 'Estimating the camera',
  lift: 'Reconstructing the 3D path',
  export: 'Checking the result',
  complete: 'Finishing',
};
const stageOrder = ['playback', 'detect', 'calibrate', 'lift', 'export'];
const statusText = { created: 'Ready', queued: 'Waiting', running: 'Processing', complete: 'Done', failed: 'Failed', cancelled: 'Cancelled', interrupted: 'Stopped' };
const active = run => run.status === 'queued' || run.status === 'running';

export function connect(viewer) {
  let state = null, selected = null, shown = null, timer;
  const api = async (path, options = {}) => {
    const response = await fetch(path, { ...options, headers: { ...headers, ...options.headers } });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || `Request failed (${response.status})`);
    return response;
  };
  const selectedRun = () => selected?.kind === 'run' ? state.runs.find(run => run.id === selected.id) : null;
  const report = error => { el('error').textContent = error.message; el('error').hidden = false; };

  function card(title, caption, current, onClick, note) {
    const button = document.createElement('button');
    button.className = 'run-card';
    button.setAttribute('aria-current', String(current));
    for (const [name, text] of [['run-title', title], ['run-caption', caption], ['run-note', note]]) {
      if (!text) continue;
      const line = document.createElement('span');
      line.className = name; line.textContent = text;
      button.append(line);
    }
    button.addEventListener('click', onClick);
    return button;
  }
  function progressText(run) {
    const progress = run.progress;
    if (run.status === 'queued') return 'Waiting for the current video to finish.';
    if (!progress) return '';
    const { completed, total, unit } = progress;
    if (unit === 'stages') return '';
    return total ? `${Math.min(completed, total)} of ${total} ${unit}` : `${completed} ${unit}`;
  }
  function renderRun(run) {
    el('processing').hidden = !run || (run.status === 'complete' && !run.error);
    if (!run) return;
    el('run-name').textContent = run.name;
    el('run-status').textContent = statusText[run.status];
    const step = stageOrder.indexOf(run.stage);
    el('run-stage').textContent = run.status === 'running'
      ? `${step >= 0 ? `Step ${step + 1} of ${stageOrder.length} · ` : ''}${stageText[run.stage] ?? 'Starting'}`
      : run.status === 'complete' ? 'Finished with a limited result.' : '';
    const bar = el('run-progress'), progress = run.progress;
    bar.hidden = !active(run);
    if (run.status === 'running' && progress?.total && progress.unit !== 'stages') bar.value = Math.min(1, progress.completed / progress.total);
    else bar.removeAttribute('value');
    el('run-progress-text').textContent = active(run) ? progressText(run) : '';
    el('run-error').hidden = !run.error;
    el('run-error').textContent = run.error ?? '';
    el('cancel').hidden = !active(run);
    el('start').hidden = active(run) || run.status === 'complete';
    el('start').textContent = run.status === 'created' ? 'Start reconstruction' : 'Start again';
    el('start').disabled = !state.ready;
    el('delete').hidden = active(run);
  }
  function render() {
    el('launch').hidden = el('library').hidden = false;
    el('demo').hidden = !!state.gallery.length;
    el('connection').textContent = 'Running on this computer';
    el('privacy').textContent = 'Videos and results stay on this computer. Nothing is uploaded to the internet.';
    el('runtime').textContent = state.ready
      ? `Ready · Automatic uses ${state.cuda ? 'the NVIDIA GPU' : 'the CPU'}. Reconstruction can take several minutes per clip.`
      : state.problems.join(' ');
    el('runtime').classList.toggle('failure', !state.ready);
    el('upload').disabled = !state.ready;
    el('upload').parentElement.classList.toggle('disabled', !state.ready);
    el('device').querySelector('[value=cuda]').disabled = !state.cuda;
    const intel = el('device').querySelector('[value="intel-gpu"]');
    if (state.intel_gpu?.status === 'available' && !intel) {
      const option = document.createElement('option');
      option.value = 'intel-gpu';
      option.textContent = 'Intel GPU detection · CPU lifting';
      el('device').append(option);
    } else if (state.intel_gpu?.status !== 'available' && intel) {
      if (el('device').value === 'intel-gpu') el('device').value = 'auto';
      intel.remove();
    }
    el('runs-empty').hidden = !!state.runs.length;
    el('runs').replaceChildren(...state.runs.map(run => card(run.name,
      active(run) ? statusText[run.status] : `${statusText[run.status]} · ${new Date(run.created).toLocaleString()}`,
      selected?.kind === 'run' && selected.id === run.id, () => select({ kind: 'run', id: run.id }))));
    el('gallery-empty').hidden = !!state.gallery.length;
    el('gallery').replaceChildren(...state.gallery.map(clip => card(clip.title,
      `Precomputed${clip.video === 'ready' ? '' : clip.video === 'preparing' ? ' · video preparing' : ' · no video'}`,
      selected?.kind === 'gallery' && selected.id === clip.id, () => select({ kind: 'gallery', id: clip.id }), clip.note)));
    el('runs-count').textContent = state.runs.length ? String(state.runs.length) : '';
    el('gallery-count').textContent = state.gallery.length ? String(state.gallery.length) : '';
    renderRun(selectedRun());
  }
  // Load whatever the selected item has now; repeat as processing produces more.
  async function show() {
    if (!selected) return;
    const item = selected.kind === 'run' ? selectedRun() : state.gallery.find(clip => clip.id === selected.id);
    if (!item) return deselect();
    const base = `/api/${selected.kind === 'run' ? 'runs' : 'gallery'}/${encodeURIComponent(item.id)}`;
    const wantVideo = selected.kind === 'run' ? item.has_video : item.video === 'ready';
    const key = `${selected.kind}/${item.id}`;
    if (shown?.key !== key) {
      shown = { key, video: false, result: 'none' };
      viewer.setVideo(null, 'Source video');
      viewer.clear(selected.kind === 'run' ? `${item.name} · no result yet` : item.title);
    }
    const current = shown;
    if (wantVideo && !current.video) {
      current.video = true;
      viewer.setVideo(`${base}/video`, 'Source video', item.media_offset_s ?? 0);
    }
    // Polling calls this repeatedly; only a changed selection abandons a load.
    if (item.has_result && current.result === 'none') {
      current.result = 'pending';
      try {
        const text = await (await api(`${base}/result`)).text();
        const timeline = item.has_timeline ? await (await api(`${base}/timeline`)).json() : null;
        if (shown !== current) return;
        viewer.loadText(text, item.name ?? item.title, timeline);
        current.result = 'loaded';
      } catch (error) {
        if (shown !== current) return;
        current.result = 'failed';
        viewer.fail(item.name ?? item.title, error);
      }
    }
  }
  function select(item) { selected = item; el('error').hidden = true; render(); show(); }
  function deselect() { selected = shown = null; if (state) render(); }
  async function refresh() {
    clearTimeout(timer);
    try {
      state = await (await api('/api/state')).json();
      render(); show();
    } catch {
      el('connection').textContent = 'App stopped. Start Tennis3D again.';
    }
    const busy = state && (state.runs.some(active) || state.gallery.some(clip => clip.video === 'preparing'));
    timer = setTimeout(refresh, busy ? 700 : 5000);
  }
  function upload(file) {
    const request = new XMLHttpRequest();
    request.open('POST', `/api/runs?name=${encodeURIComponent(file.name)}&device=${el('device').value}`);
    request.setRequestHeader('X-Tennis3D', '1');
    request.setRequestHeader('Content-Type', 'application/octet-stream');
    el('runtime').textContent = 'Copying the video…';
    request.upload.addEventListener('progress', event => {
      if (event.lengthComputable) el('runtime').textContent = `Copying the video… ${Math.round(100 * event.loaded / event.total)}%`;
    });
    request.addEventListener('load', async () => {
      let body = {};
      try { body = JSON.parse(request.responseText); } catch { /* reported below */ }
      if (request.status !== 201) report(new Error(body.error || 'The video could not be added.'));
      else selected = { kind: 'run', id: body.id };
      await refresh();
    });
    request.addEventListener('error', () => { report(new Error('The video could not be copied. Is Tennis3D still running?')); refresh(); });
    request.send(file);
  }
  el('upload').addEventListener('change', event => {
    const file = event.target.files[0];
    el('error').hidden = true;
    if (file) upload(file);
    event.target.value = '';
  });
  const act = (button, run) => el(button).addEventListener('click', async () => {
    const id = selectedRun()?.id;
    if (!id) return;
    try { await run(id); } catch (error) { report(error); }
    refresh();
  });
  act('cancel', id => api(`/api/runs/${id}/cancel`, { method: 'POST' }));
  act('start', id => { shown = null; return api(`/api/runs/${id}/start?device=${el('device').value}`, { method: 'POST' }); });
  act('delete', async id => {
    if (!confirm('Delete this run and its saved result from this computer?')) return;
    await api(`/api/runs/${id}`, { method: 'DELETE' });
    viewer.setVideo(null, 'Optional playback'); viewer.clear('Open a result or choose an example to explore a trajectory.');
    deselect();
  });
  refresh();
  return { deselect };
}
