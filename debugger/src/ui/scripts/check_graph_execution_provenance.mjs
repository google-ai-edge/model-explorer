/**
 * @license
 * Copyright 2026 The AI Edge Model Explorer Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

import {strict as assert} from 'node:assert';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {existsSync, mkdtempSync, readFileSync, rmSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

// The served visualizer bundle must be exactly the recorded upstream bytes plus the recorded patch.
const root = fileURLToPath(new URL('..', import.meta.url));
const upstreamBundle = path.join(root, 'public/graph_execution/upstream/model_explorer');
const provenance = JSON.parse(readFileSync(path.join(upstreamBundle, 'PROVENANCE.json'), 'utf8'));
const sha256 = (file) => createHash('sha256').update(readFileSync(file)).digest('hex');
for (const [file, expected] of Object.entries(provenance.sha256))
  assert.equal(
    sha256(path.join(upstreamBundle, file)),
    expected,
    `${file} differs from PROVENANCE.json`,
  );
assert.ok(provenance.patch?.file, 'PROVENANCE.json names the replayable patch');
const archPreview = path.join(root, 'public/graph_arch_preview');
const archProvenance = JSON.parse(readFileSync(path.join(archPreview, 'PROVENANCE.json'), 'utf8'));
for (const [file, expected] of Object.entries(archProvenance.sha256))
  assert.equal(
    sha256(path.join(archPreview, file)),
    expected,
    `graph_arch_preview/${file} differs from PROVENANCE.json`,
  );
const upstream = path.join(
  root,
  'node_modules/.cache/model_explorer_upstream/package/dist/main_browser.js',
);
if (existsSync(upstream)) {
  const temp = mkdtempSync(path.join(root, 'node_modules/.cache', 'provenance-'));
  try {
    const rebuilt = path.join(temp, 'main_browser.js');
    execFileSync('patch', [
      '-s',
      '-o',
      rebuilt,
      upstream,
      path.join(upstreamBundle, provenance.patch.file),
    ]);
    assert.equal(
      sha256(rebuilt),
      provenance.sha256['dist/main_browser.js'],
      'patch does not reproduce the served bundle',
    );
  } finally {
    rmSync(temp, {recursive: true, force: true});
  }
  console.log(
    'PASS: visualizer bundle, worker and patch hashes match PROVENANCE.json; the patch rebuilds the bundle from the upstream tarball',
  );
} else {
  console.log(
    'PASS: visualizer bundle, worker and patch hashes match PROVENANCE.json (replay skipped: run scripts/replay_model_explorer_patch.mjs to cache the upstream tarball)',
  );
}
