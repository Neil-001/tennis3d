// Coverage summary and the per-source-frame table. Rows are never merged or interpolated.
const el = id => document.getElementById(id);
const pageSize = 100;
export const reasonText = {
  long_gap: 'ball not seen for too long',
  invalid_camera: 'court camera could not be estimated',
  ambiguous_track: 'more than one plausible ball',
  window_disagreement: 'model estimates disagreed',
  implausible_prediction: 'physically implausible position',
  unsupported_shot: 'camera cut or unsupported shot',
  decoder_error: 'video could not be decoded',
};
const describe = codes => codes.map(code => reasonText[code] ?? code).join('; ');
const statusText = {
  ok: 'Every source frame has a 3D position.',
  partial: 'Some source frames have no usable 3D position. They are shown as gaps and are not filled in.',
  failed: 'No usable 3D trajectory was produced for this video. The source frames remain listed below.',
};

function stat(value, label) {
  const item = document.createElement('div');
  item.className = 'coverage-stat';
  const caption = document.createElement('span');
  caption.textContent = label;
  item.append(value, caption);
  return item;
}

export function showSummary(data) {
  const frames = data.frames, count = state => frames.filter(f => f.reconstruction === state).length;
  const stats = document.createElement('div');
  stats.className = 'coverage-stats';
  stats.append(
    stat(`${(data.summary.valid_frame_fraction * 100).toFixed(1)}%`, '3D coverage'),
    stat(String(frames.length), 'source frames'),
    stat(String(count('estimated')), 'from seen ball'),
    stat(String(count('inferred')), 'inferred'),
    stat(String(count('unavailable')), 'gaps'),
    stat(`${(frames.at(-1)?.t_s ?? 0).toFixed(2)} s`, 'duration'));
  el('coverage').replaceChildren(stats);
  el('result-message').textContent = statusText[data.status];
  el('failure-reasons').hidden = !data.failure_reasons.length;
  el('failure-reasons').textContent = data.failure_reasons.length ? `Reason: ${describe(data.failure_reasons)}.` : '';
  el('result-summary').hidden = false;
}

let rows = [], page = 0, seekTo = () => {};
function render() {
  const pages = Math.max(1, Math.ceil(rows.length / pageSize));
  page = Math.min(pages - 1, Math.max(0, page));
  const body = el('frame-rows');
  body.replaceChildren();
  for (const frame of rows.slice(page * pageSize, (page + 1) * pageSize)) {
    const row = document.createElement('tr');
    row.dataset.state = frame.reconstruction;
    const go = document.createElement('button');
    go.textContent = String(frame.frame_index);
    go.title = 'Show this frame';
    go.addEventListener('click', () => seekTo(frame.t_s));
    const first = document.createElement('td');
    first.append(go);
    row.append(first);
    for (const text of [
      frame.t_s.toFixed(3),
      { estimated: 'valid', inferred: 'inferred', unavailable: 'gap' }[frame.reconstruction],
      frame.observation.replaceAll('_', ' '),
      frame.xyz_m ? frame.xyz_m.map(v => v.toFixed(2)).join(', ') : '—',
      describe(frame.reason_codes),
    ]) {
      const cell = document.createElement('td');
      cell.textContent = text;
      row.append(cell);
    }
    body.append(row);
  }
  el('frame-count').textContent = rows.length ? `${rows.length} rows · page ${page + 1} of ${pages}` : 'No rows';
  el('frames-prev').disabled = page === 0;
  el('frames-next').disabled = page >= pages - 1;
}
let current = null;
function filter() {
  const wanted = el('frame-filter').value;
  rows = (current?.frames ?? []).filter(frame => wanted === 'all' || frame.reconstruction === wanted);
  page = 0; render();
}
export function showFrames(data, seek) {
  current = data; seekTo = seek; filter();
}
el('frame-filter').addEventListener('change', filter);
el('frames-prev').addEventListener('click', () => { page--; render(); });
el('frames-next').addEventListener('click', () => { page++; render(); });
