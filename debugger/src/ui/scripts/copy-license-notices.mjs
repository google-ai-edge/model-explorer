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

// Angular lists the npm packages it bundled; the notices below cover the
// lazy-loaded Plotly asset and what ships from public/.
const output = 'dist/model_explorer_debugger/browser/3rdpartylicenses.txt';
const modelExplorerDir = 'public/graph-execution/upstream/model-explorer';
const notices = [
  [
    'Plotly.js — MIT', 'node_modules/plotly.js-dist-min/LICENSE'
  ],
  [
    'ai-edge-model-explorer-visualizer (modified; see PROVENANCE) — Apache-2.0',
    `${modelExplorerDir}/LICENSE`,
  ],
  [
    'ai-edge-model-explorer-visualizer local changes',
    `${modelExplorerDir}/PROVENANCE.json`,
  ],
  [
    'Material Icons (public/fonts) — Apache-2.0',
    'public/fonts/LICENSE-material-icons.txt'
  ],
];
let text =
    await readFile('dist/model_explorer_debugger/3rdpartylicenses.txt', 'utf8');
for (const [title, file] of notices) {
  const body = (await readFile(file, 'utf8')).trim();
  text +=
      `\n\n${'='.repeat(72)}\n${title}\n${file}\n${'='.repeat(72)}\n${body}\n`;
}
await writeFile(output, text);
console.log(`Wrote ${output} with ${notices.length} bundled-resource notices.`);
