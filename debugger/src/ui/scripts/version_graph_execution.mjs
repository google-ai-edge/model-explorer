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

import {readFile, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

// Bind the Angular island, frame document and executable dependencies to their bytes.
// Keep this generated chain committed so direct ng builds are safe as well as npm builds.
const root = fileURLToPath(new URL('..', import.meta.url));
const directory = path.join(root, 'public/graph_execution');
const check = process.argv.includes('--check');
const edits = new Map();
const digest = (data) => createHash('sha256').update(data).digest('hex').slice(0, 16);
const read = (file) =>
  edits.has(file) ? Promise.resolve(edits.get(file)) : readFile(file, 'utf8');
const controllerPath = path.join(directory, 'execution_view.js');
const workerHash = digest(
  await readFile(path.join(directory, 'upstream/model_explorer/dist/worker.js')),
);
const controller = await read(controllerPath);
const workerPattern =
  /workerScriptPath = 'upstream\/model_explorer\/dist\/worker\.js(?:\?v=[a-f0-9]+)?'/;
if (!workerPattern.test(controller)) throw new Error('Execution worker path not found');
edits.set(
  controllerPath,
  controller.replace(
    workerPattern,
    `workerScriptPath = 'upstream/model_explorer/dist/worker.js?v=${workerHash}'`,
  ),
);

const indexPath = path.join(directory, 'index.html');
let index = await read(indexPath);
for (const asset of [
  'execution.css',
  'execution_data.js',
  'execution_view.js',
  'upstream/model_explorer/dist/main_browser.js',
]) {
  const hash = digest(await read(path.join(directory, asset)));
  const escaped = asset.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pattern = new RegExp(`((?:src|href)=")${escaped}(?:\\?v=[a-f0-9]+)?"`);
  if (!pattern.test(index)) throw new Error('Execution dependency not found: ' + asset);
  index = index.replace(pattern, `$1${asset}?v=${hash}"`);
}
edits.set(indexPath, index);
const componentPath = path.join(root, 'src/features/graph/execution_graph/execution_graph.ts');
const component = await read(componentPath);
const framePattern = /src="graph_execution\/index\.html(?:\?v=[a-f0-9]+)?"/;
if (!framePattern.test(component)) throw new Error('Execution frame URL not found');
edits.set(
  componentPath,
  component.replace(framePattern, `src="graph_execution/index.html?v=${digest(index)}"`),
);

const changed = [];
for (const [file, content] of edits) {
  if (content === (await readFile(file, 'utf8'))) continue;
  changed.push(path.relative(root, file));
  if (!check) await writeFile(file, content);
}
if (check && changed.length)
  throw new Error(
    'Stale Graph execution asset versions. Run node scripts/version_graph_execution.mjs: ' +
      changed.join(', '),
  );
console.log(
  check
    ? 'PASS: Graph frame/dependency content versions match'
    : `Graph execution asset versions updated (${changed.length} files).`,
);
