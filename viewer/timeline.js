// A separate source receipt must join the exact rows of the result being shown.
export function validateTimeline(result, timeline) {
  if (!timeline) return null;
  if (timeline.source_sha256 !== result.source.sha256 || !Array.isArray(timeline.frames) || timeline.frames.length !== result.frames.length) throw new Error('Source timeline does not match this result.');
  let origin;
  for (const [index, row] of timeline.frames.entries()) {
    const base = row.time_base;
    if (row.frame_index !== index || !Number.isSafeInteger(row.pts) || !Array.isArray(base) || base.length !== 2 || !base.every(v => Number.isSafeInteger(v) && v > 0)) throw new Error('Source timeline contains an invalid exact row.');
    const at = row.pts * base[0] / base[1];
    if (index === 0 && (Math.abs(at - result.source.time_origin_s) > 1e-9 || !Array.isArray(result.source.time_base) || base.some((v, i) => v !== result.source.time_base[i]))) throw new Error('Source timeline origin or time base does not match this result.');
    origin ??= at;
    if (result.frames[index].frame_index !== index || Math.abs(result.frames[index].t_s - (at - origin)) > 1e-9) throw new Error('Source PTS does not match the result row time.');
  }
  return timeline;
}
