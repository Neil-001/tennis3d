// Build only the read-only example page and its verified playback copies.
import { execFileSync } from 'node:child_process';
import { readFile, mkdir, copyFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const output = resolve(root, '_site');
const manifestPath = resolve(root, 'examples/gallery-v2/gallery.json');
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
await mkdir(resolve(output, 'media'), { recursive: true });
// This validates all result, timeline and playback hashes before copying media.
execFileSync('node', [resolve(root, 'scripts/build_viewer.mjs'), '--gallery', manifestPath,
  '--read-only', '--output', resolve(output, 'index.html')], { cwd: root, stdio: 'inherit' });
for (const clip of manifest.clips) {
  await copyFile(resolve(dirname(manifestPath), clip.playback_path), resolve(output, 'media', `${clip.id}.mp4`));
}
await writeFile(resolve(output, '.nojekyll'), '');
console.log(`Built read-only dashboard with ${manifest.clips.length} playback clips.`);
