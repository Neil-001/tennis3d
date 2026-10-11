import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { validateTimeline } from '../viewer/timeline.js';

async function verified(path, expected) {
  if (!/^[a-f0-9]{64}$/.test(expected ?? '')) throw new Error(`Missing SHA-256 for ${path}`);
  const hash = createHash('sha256');
  for await (const block of createReadStream(path)) hash.update(block);
  if (hash.digest('hex') !== expected) throw new Error(`SHA-256 mismatch for ${path}`);
}

export async function embeddedGallery(path, validate, { playback = false } = {}) {
  const manifest = JSON.parse(await readFile(path, 'utf8'));
  if (manifest.schema_version !== 1 || !Array.isArray(manifest.clips)) throw new Error('Unsupported gallery manifest');
  const ids = new Set(), clips = [];
  for (const clip of manifest.clips) {
    if (!/^[A-Za-z0-9_.-]{1,100}$/.test(clip.id) || ['.', '..'].includes(clip.id) || ids.has(clip.id)) throw new Error('Invalid or repeated gallery clip id');
    ids.add(clip.id);
    for (const name of [playback ? 'playback' : 'source', 'result', 'timeline']) await verified(resolve(dirname(path), clip[`${name}_path`]), clip[`${name}_sha256`]);
    const resultText = await readFile(resolve(dirname(path), clip.result_path), 'utf8');
    const result = JSON.parse(resultText);
    if (!validate(result)) throw new Error(`Invalid result for ${clip.id}: ${JSON.stringify(validate.errors)}`);
    const timeline = JSON.parse(await readFile(resolve(dirname(path), clip.timeline_path), 'utf8'));
    if (result.source.sha256 !== clip.source_sha256) throw new Error(`Source hash mismatch for ${clip.id}`);
    validateTimeline(result, timeline);
    if (playback && (!Number.isFinite(clip.media_offset_s ?? 0) || (clip.media_offset_s ?? 0) < 0)) throw new Error(`Invalid media offset for ${clip.id}`);
    clips.push({ ...(playback ? { videoUrl: `./media/${clip.id}.mp4`, mediaOffset: clip.media_offset_s ?? 0 } : {}), id: clip.id, title: clip.title ?? clip.id, rights: clip.rights, resultText,
      timeline: { endpoint_rule: timeline.endpoint_rule, source_sha256: timeline.source_sha256,
        frames: timeline.frames.map(({ frame_index, original_frame_index, pts, time_base }) => ({ frame_index, original_frame_index, pts, time_base })) } });
  }
  return { clips };
}
