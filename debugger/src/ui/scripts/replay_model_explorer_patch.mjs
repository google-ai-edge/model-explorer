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

import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {existsSync, mkdirSync, readFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

// Fetch the pinned upstream tarball (network), verify it, and prove that
// local-changes.patch turns its browser bundle into the served one.
const root = fileURLToPath(new URL('..', import.meta.url));
const modelExplorerDir =
    path.join(root, 'public/graph-execution/upstream/model-explorer');
const provenance = JSON.parse(
    readFileSync(path.join(modelExplorerDir, 'PROVENANCE.json'), 'utf8'),
);
const cache = path.join(root, 'node_modules/.cache/model-explorer-upstream');
mkdirSync(cache, {recursive: true});
const tarball =
    path.join(cache, `${provenance.package}-${provenance.version}.tgz`);
if (!existsSync(tarball)) {
  execFileSync(
      'npm',
      [
        'pack', `${provenance.package}@${provenance.version}`,
        '--pack-destination', cache
      ],
      {stdio: 'inherit'},
  );
}
const sha1 = createHash('sha1').update(readFileSync(tarball)).digest('hex');
if (sha1 !== provenance.sha1) {
  throw new Error(
      `Tarball sha1 ${sha1} differs from PROVENANCE.json ${provenance.sha1}`);
}
execFileSync('tar', ['xzf', tarball, '-C', cache]);
const upstream = path.join(cache, 'package/dist/main_browser.js');
const rebuilt = path.join(cache, 'rebuilt_main_browser.js');
execFileSync('patch', [
  '-s',
  '-o',
  rebuilt,
  upstream,
  path.join(modelExplorerDir, provenance.patch.file),
]);
const sha256 = createHash('sha256').update(readFileSync(rebuilt)).digest('hex');
if (sha256 !== provenance.sha256['dist/main_browser.js']) {
  throw new Error(
      `Rebuilt bundle ${sha256} differs from the served bundle ${
          provenance.sha256['dist/main_browser.js']}`,
  );
}
console.log(
    `PASS: ${provenance.package}@${
        provenance
            .version} + local-changes.patch reproduces dist/main_browser.js (${
        sha256})`,
);
