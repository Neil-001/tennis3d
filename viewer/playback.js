// Binary search preserves source cadence. No frame-number-to-time conversion.
export function frameAt(frames, time) {
  if (!frames.length) return -1;
  let lo = 0, hi = frames.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (frames[mid].t_s <= time) lo = mid + 1;
    else hi = mid;
  }
  return Math.max(0, lo - 1);
}

// Video seeks land a fraction of a millisecond off a frame's timestamp. Snap to
// that frame so its row and its sampled position always agree.
export function snap(frames, time, tolerance = 1e-4) {
  const index = frameAt(frames, time + tolerance), frame = frames[index];
  return { index, time: frame && Math.abs(time - frame.t_s) <= tolerance ? frame.t_s : time };
}

export function sample(frames, time) {
  const index = frameAt(frames, time);
  if (index < 0) return { index, xyz: null, state: 'unavailable' };
  const a = frames[index], b = frames[index + 1];
  if (time === a.t_s || !b) return { index, xyz: a.xyz_m, state: a.reconstruction };
  if (!a.xyz_m || !b.xyz_m) return { index, xyz: null, state: 'unavailable' };
  const alpha = (time - a.t_s) / (b.t_s - a.t_s);
  return { index, xyz: a.xyz_m.map((v, j) => v * (1 - alpha) + b.xyz_m[j] * alpha),
    state: a.reconstruction === 'inferred' || b.reconstruction === 'inferred' ? 'inferred' : 'estimated' };
}

export function trailSegments(frames, time, seconds) {
  const segments = { estimated: [], inferred: [] };
  if (seconds <= 0) return segments;
  const start = Math.max(0, time - seconds);
  for (let i = Math.max(1, frameAt(frames, start) + 1); i < frames.length; i++) {
    const a = frames[i - 1], b = frames[i];
    if (a.t_s >= time) break;
    if (!a.xyz_m || !b.xyz_m) continue;
    const from = sample(frames, Math.max(start, a.t_s)).xyz;
    const to = sample(frames, Math.min(time, b.t_s)).xyz;
    const key = a.reconstruction === 'inferred' || b.reconstruction === 'inferred' ? 'inferred' : 'estimated';
    if (from && to) segments[key].push(...from, ...to);
  }
  return segments;
}
