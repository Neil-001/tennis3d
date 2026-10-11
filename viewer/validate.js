import structural from './generated/validate.cjs';
import { visit } from 'jsonc-parser';

function requireState(condition, message) {
  if (!condition) throw new Error(message);
}

export function parseResult(text) {
  const keys = [];
  visit(text, {
    onObjectBegin: () => keys.push(new Set()),
    onObjectProperty: key => {
      requireState(!keys.at(-1).has(key), `Duplicate JSON key: ${key}`);
      keys.at(-1).add(key);
    },
    onObjectEnd: () => keys.pop(),
    onError: () => { throw new Error('Malformed JSON. Select a result JSON file.'); },
  }, { disallowComments: true, allowTrailingComma: false });
  return validateResult(JSON.parse(text));
}

export function validateResult(data) {
  requireState(data?.schema_version === '1.0' || data?.schema_version === undefined, 'Unsupported schema version. Expected 1.0.');
  requireState(structural(data), `Invalid result: ${structural.errors?.[0]?.instancePath || '/'} ${structural.errors?.[0]?.message || ''}`);
  const { frames, source, provenance, synthetic_diagnostics: diagnostic } = data;
  requireState(frames.length === source.frame_count, 'Frame count does not match source.');
  let valid = 0, detected = 0;
  frames.forEach((f, i) => {
    const check = (c, message) => requireState(c, `Frame ${i}: ${message}`);
    check(f.frame_index === i, 'indices must be contiguous from zero.');
    check(i ? f.t_s > frames[i - 1].t_s : f.t_s === 0, 'timestamps must start at zero and strictly increase.');
    const unavailable = f.reconstruction === 'unavailable';
    check(unavailable === (f.xyz_m === null), 'XYZ contradicts availability.');
    check(f.camera_valid || unavailable, 'invalid camera requires unavailable XYZ.');
    check(f.reason_codes.includes('invalid_camera') === !f.camera_valid, 'camera state contradicts reasons.');
    check(unavailable === !!f.reason_codes.length && new Set(f.reason_codes).size === f.reason_codes.length, 'invalid reason codes.');
    check(unavailable || (f.observation === 'detected') === (f.reconstruction === 'estimated'), 'observation contradicts reconstruction.');
    check((f.observation === 'duplicate') === (f.duplicate_of !== null), 'duplicate_of contradicts observation.');
    check(f.duplicate_of === null || f.duplicate_of < i, 'duplicate must reference an earlier frame.');
    valid += !unavailable;
    detected += f.observation === 'detected';
  });
  const n = frames.length;
  requireState(Math.abs(data.summary.valid_frame_fraction - (n ? valid / n : 0)) <= 1e-9 &&
    Math.abs(data.summary.detected_frame_fraction - (n ? detected / n : 0)) <= 1e-9, 'Summary coverage does not match frames.');
  const status = data.failure_reasons.length || !valid ? 'failed' : valid === n ? 'ok' : 'partial';
  requireState(data.status === status, 'Status contradicts frames or failure reasons.');
  if (diagnostic) {
    requireState(provenance.calibration_mode === 'synthetic' && ['fixture', 'synthetic_ground_truth'].includes(provenance.trajectory_kind), 'Synthetic diagnostics require synthetic provenance.');
    requireState(diagnostic.frames.length === n && diagnostic.frames.every((f, i) => f.frame_index === i && f.t_s === frames[i].t_s), 'Diagnostic frames must exactly match the result timeline.');
    let previous = 0;
    for (const event of diagnostic.events) {
      requireState(n > 0 && event.t_s >= previous && event.t_s <= frames.at(-1).t_s, 'Events must be ordered within the result timeline.');
      previous = event.t_s;
    }
  }
  return data;
}
