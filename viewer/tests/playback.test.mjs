import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { sample, snap, trailSegments } from '../playback.js';
import { parseResult, validateResult } from '../validate.js';
const fixture = () => JSON.parse(readFileSync(new URL('../example.json', import.meta.url)));

test('irregular time interpolation, exact endpoints and gaps', () => {
  const { frames } = fixture();
  assert.deepEqual(sample(frames, 0).xyz, frames[0].xyz_m);
  assert.deepEqual(sample(frames, 2).xyz, frames.at(-1).xyz_m);
  assert.equal(sample(frames, 0.375).state, 'inferred');
  assert.ok(Math.abs(sample(frames, 0.375).xyz[1] - (-6)) < 1e-12);
  for (const t of [1.06, 1.35, 1.5, 1.6, 1.79]) assert.equal(sample(frames, t).xyz, null);
  assert.deepEqual(sample(frames, 1.8).xyz, frames[7].xyz_m);
  assert.equal(trailSegments(frames, 2, 0).estimated.length, 0);
  assert.equal(trailSegments(frames, 2, 100).estimated.length, 12);
  assert.equal(trailSegments(frames, 2, 100).inferred.length, 18);
  assert.equal(sample([], 0).index, -1);
  assert.deepEqual(sample([frames[0]], 0).xyz, frames[0].xyz_m);
});

test('both fixtures validate, defaults match Python', () => {
  validateResult(fixture());
  parseResult(readFileSync(new URL('../../tests/fixtures/trajectory.json', import.meta.url), 'utf8'));
  const data = fixture();
  delete data.schema_version;
  delete data.frames[0].duplicate_of;
  delete data.frames[0].reason_codes;
  delete data.failure_reasons;
  delete data.synthetic_diagnostics.events;
  delete data.coordinates;
  delete data.court;
  assert.equal(validateResult(data).frames[0].duplicate_of, null);
});

test('reject malformed, contradictory and misaligned data before rendering', () => {
  const mutations = [
    d => d.schema_version = '2', d => d.source.frame_count++, d => d.frames[1].t_s = 0,
    d => d.frames[0].frame_index = 2, d => d.frames[0].xyz_m[0] = Infinity,
    d => d.frames[0].xyz_m = null, d => d.frames[0].observation = 'missing',
    d => d.frames[0].camera_valid = false, d => d.frames[0].reason_codes = ['long_gap'],
    d => d.frames[0].duplicate_of = 0, d => d.summary.valid_frame_fraction = 1,
    d => d.source.width = '640', d => d.status = 'ok', d => d.extra = true,
    d => d.synthetic_diagnostics.frames[1].t_s += .001,
    d => d.synthetic_diagnostics.frames[0].court_uv.pop(),
    d => d.synthetic_diagnostics.events[0].t_s = 3,
    d => d.provenance.trajectory_kind = 'model_estimate',
  ];
  for (const mutate of mutations) { const data = fixture(); mutate(data); assert.throws(() => validateResult(data)); }
  for (const value of ['{"schema_version":"1.0","schema_version":"1.0"}', '{/*comment*/}', '{"x":1,}', '{"x":NaN}', '{"x":1e999}', 'null']) assert.throws(() => parseResult(value));
});

test('a rounded video seek keeps the frame row and its position together', () => {
  const frames = [0, 1, 2].map(i => ({ t_s: i * 0.1668335, xyz_m: i === 0 ? null : [i, 0, 1], reconstruction: i === 0 ? 'unavailable' : 'estimated' }));
  for (const seek of [0.166833, 0.166834, 0.1668335]) {
    const at = snap(frames, seek);
    assert.equal(at.index, 1);
    assert.deepEqual(sample(frames, at.time).xyz, frames[1].xyz_m);
  }
  assert.equal(snap(frames, 0.1).index, 0);
  assert.equal(sample(frames, snap(frames, 0.1).time).xyz, null);
});
