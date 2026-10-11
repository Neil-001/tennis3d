import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { embeddedGallery } from './embedded_gallery.mjs';
const { values } = parseArgs({ options: { gallery: { type: 'string' }, output: { type: 'string' }, 'read-only': { type: 'boolean', default: false } } });
if (values.gallery && !values.output) throw new Error('--gallery requires --output to keep private examples out of the packaged dashboard.');
if (values['read-only'] && (!values.gallery || !values.output)) throw new Error('--read-only requires --gallery and --output.');
const root = fileURLToPath(new URL('../', import.meta.url));
const require = createRequire(new URL('../viewer/package.json', import.meta.url));
const { build } = require('esbuild');
const Ajv = require('ajv/dist/2020');
const standalone = require('ajv/dist/standalone').default;
const { schema, geometry } = JSON.parse(execFileSync('uv', ['run', '--locked', 'python', 'scripts/viewer_contract.py'], { cwd: root }));
const ajv = new Ajv({ strict: false, strictNumbers: true, useDefaults: true, code: { source: true, esm: false } });
await mkdir(`${root}/viewer/generated`, { recursive: true });
await writeFile(`${root}/viewer/generated/validate.cjs`, standalone(ajv, ajv.compile(schema)));
await writeFile(`${root}/viewer/generated/court.json`, JSON.stringify(geometry));
const output = await build({ entryPoints: [`${root}/viewer/app.js`], bundle: true, minify: true,
  write: false, format: 'iife', target: 'es2022', legalComments: 'inline' });
const template = await readFile(`${root}/viewer/index.html`, 'utf8');
// One self-contained page: the shared theme carries its typeface as a data URI.
const theme = `${root}/tennis3d/resources/theme`, fontName = 'GoogleSansCode-latin.woff2';
const font = (await readFile(`${theme}/${fontName}`)).toString('base64');
const style = (await readFile(`${theme}/theme.css`, 'utf8')).replace(`url(${fontName})`, () => `url(data:font/woff2;base64,${font})`)
  + await readFile(`${root}/viewer/style.css`, 'utf8');
if (!style.includes('data:font/woff2')) throw new Error('The theme no longer references its bundled typeface.');
const licenses = await Promise.all(['three/LICENSE', 'jsonc-parser/LICENSE.md', 'ajv/LICENSE', 'fast-uri/LICENSE'].map(async path =>
  `${path}\n${await readFile(`${root}/viewer/node_modules/${path}`, 'utf8')}`));
licenses.push(`Google Sans Code (typeface)\n${await readFile(`${theme}/OFL.txt`, 'utf8')}`);
const escape = text => text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
let html = template.replace('/* INLINE_STYLE */', () => style)
  .replace('/* INLINE_SCRIPT */', () => output.outputFiles[0].text.replaceAll('</script', '<\\/script'))
  .replace('<!-- LICENSES -->', () => `<pre>${escape(licenses.join('\n\n'))}</pre>`);
if (values.gallery) {
  const gallery = await embeddedGallery(resolve(values.gallery), ajv.compile(schema), { playback: values['read-only'] });
  if (values['read-only']) {
    gallery.readOnly = true;
    html = html.replace('<body>', '<body data-read-only>')
      .replace('Your rally, in 3D', 'Explore tennis in 3D')
      .replace('One camera in, a metric 3D ball path out, played back beside the source video.', 'Twenty tennis clips with precomputed 3D ball paths. Choose an example and explore its video, trajectory and source frames.')
      .replace('Tennis3D is a local video reconstruction app.', 'This is the read-only Tennis3D example gallery. Run the local app to reconstruct your own videos.')
      .replace('<span>Tennis3D · GPL-3.0-only</span>', '<a href="https://github.com/Neil-001/tennis3d">Get Tennis3D · GPL-3.0-only</a>');
  }
  html = html.replace('<!-- EMBEDDED_GALLERY -->', () => `<script id="embedded-gallery" type="application/json">${JSON.stringify(gallery).replaceAll('<', '\\u003c')}</script>`);
}
if (values.output) {
  await writeFile(resolve(values.output), html);
  console.log(`Built ${resolve(values.output)} (${Buffer.byteLength(html)} bytes)`);
} else {
  await mkdir(`${root}/dist`, { recursive: true });
  await writeFile(`${root}/dist/viewer.html`, html);
  await mkdir(`${root}/tennis3d/resources`, { recursive: true });
  await writeFile(`${root}/tennis3d/resources/dashboard.html`, html);
  console.log(`Built dist/viewer.html and tennis3d/resources/dashboard.html (${Buffer.byteLength(html)} bytes each)`);
}
