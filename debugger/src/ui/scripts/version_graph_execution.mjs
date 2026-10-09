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
 */

import {createHash} from 'node:crypto';
import {readFile, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

/**
 * Strips balanced curly-brace declarations (interfaces, declare global) from TS
 * source.
 */
function stripBalancedBlocks(source, headerRegex) {
  let result = source;
  while (true) {
    const match = headerRegex.exec(result);
    if (!match) {
      break;
    }
    const start = match.index;
    const openBrace = result.indexOf('{', start + match[0].length - 1);
    if (openBrace === -1) {
      break;
    }
    let depth = 0;
    let end = openBrace;
    for (; end < result.length; end++) {
      if (result[end] === '{') {
        depth++;
      } else if (result[end] === '}') {
        depth--;
        if (depth === 0) {
          end++;
          break;
        }
      }
    }
    result = result.slice(0, start) + result.slice(end);
  }
  return result;
}

/**
 * Normalizes TS or JS runtime source into a canonical token stream so any
 * behavioral drift between `runtime/execution_*.ts` and
 * `public/graph-execution/execution-*.js` fails verification immediately.
 */
export function normalizeRuntimeSource(rawSource, isTs) {
  let text =
      rawSource.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
  if (isTs) {
    text = text.replace(/import\s+type\s+\{[\s\S]*?\}\s+from\s+'[^']+';/g, '');
    text = text.replace(
        /^\s*(?:export\s+)?(?:declare\s+)?type\s+[A-Za-z_]\w*\s*=[^;]+;/gm, '');
    text = stripBalancedBlocks(
        text,
        /(?:export\s+)?(?:declare\s+)?interface\s+[A-Za-z_]\w*(?:\s+extends\s+[^{]+)?\s*\{/);
    text = stripBalancedBlocks(text, /declare\s+global\s*\{/);
    text = text.replace(
        /^\s*private\s+(?:readonly\s+)?[A-Za-z_]\w*(?:[^;(=\n]*?)(?:=\s*[^;]+)?;\s*$/gm,
        '');
    text = text.replace(/\bexport\s+(?=(?:class|function|const)\b)/g, '');
    text = text.replace(/\bprivate\s+readonly\s+/g, '');
    text = text.replace(/\bprivate\s+/g, '');
    text = text.replace(
        /\bnew\s+(Map|Set)\s*<(?:[^<>]|<[^<>]*>)*>\s*\(/g, 'new $1(');
    text = text.replace(
        /\(\s*([A-Za-z_]\w*)\s*\)\s*:\s*\1\s+is\s+string\s*=>/g, '($1) =>');
    text = text.replace(
        /(?<!\.)\b(function\s+[A-Za-z_]\w*|async\s+render|constructor|post|showStatus|initializationFailed|applyTheme|saveViewport|selectedNodes|syncSelection|readable|initialize|ensureViewport|updateData|selectHit|showTooltip|renderLabels|perform)\s*\(([^()]*)\)\s*(?::\s*[^{]+)?\{/g,
        (_, fnName, rawParams) => {
          let cleanParams = rawParams.replace(/\bprivate\s+readonly\s+/g, '');
          while (/<[^<>]+>/.test(cleanParams)) {
            cleanParams = cleanParams.replace(/<[^<>]+>/g, '');
          }
          cleanParams = cleanParams.replace(/:\s*[^=,)]+(?=\s*=)/g, '')
                            .replace(/:\s*[^,)]+/g, '');
          return `${fnName}(${cleanParams}) {`;
        });
    text = text.replace(
        /\b(const|let)\s+([A-Za-z_]\w*)\s*:\s*[^=;]+=/g, '$1 $2 =');
    text = text.replace(/\(event:\s*MessageEvent\)/g, '(event)');
  } else {
    text = text.replace(/^\s*\(\(\)\s*=>\s*\{\s*'use strict';/, '');
    text = text.replace(/\}\)\(\);\s*$/, '');
    text = text.replace(
        /constructor\s*\(\s*dataAdapter\s*\)\s*\{\s*(?:this\.(?:dataAdapter|host|tooltip|views|latest|current|element|generation|ready|initialized|frame|desiredSelections|choices|suppressSelectionEvents|hover|pointer|pointerDownCoordinates|metricKey|lastHit|selectionKey|priorSelection|pendingReveal|pendingAction|mappingSnapshot|paintedTheme|themeWarning)\s*=\s*[^;]+;\s*)+/,
        'constructor(dataAdapter) {');
  }
  return text.replace(/,\s*([)\]}])/g, '$1')
      .replace(/\s+/g, ' ')
      .replace(/\s*([{}();,:[\]=.])\s*/g, '$1')
      .trim();
}

// Bind the Angular island, frame document and executable dependencies to their
// bytes. Keep this generated chain committed so direct ng builds are safe as
// well as npm builds.
const root = fileURLToPath(new URL('..', import.meta.url));
const directory = path.join(root, 'public/graph-execution');
const check = process.argv.includes('--check');
const edits = new Map();
const digest = (data) =>
    createHash('sha256').update(data).digest('hex').slice(0, 16);
const read = (file) =>
    edits.has(file) ? Promise.resolve(edits.get(file)) : readFile(file, 'utf8');

const modelExplorerDist = 'upstream/model-explorer/dist';
const workerAsset = `${modelExplorerDist}/worker.js`;
const workerHash = digest(await readFile(path.join(directory, workerAsset)));
const workerTargets = [
  path.join(directory, 'execution-view.js'),
  path.join(
      root, 'src/features/graph/execution_graph/runtime/execution_view.ts'),
];
const workerPattern = new RegExp(
    `((?:WORKER_SCRIPT_PATH|workerScriptPath)\\s*=\\s*)'${
        workerAsset.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?:\\?v=[a-f0-9]+)?'`,
    'g',
);

for (const targetPath of workerTargets) {
  const source = await read(targetPath);
  if (!workerPattern.test(source)) {
    throw new Error(`Execution worker path not found in ${
        path.relative(root, targetPath)}`);
  }
  workerPattern.lastIndex = 0;
  edits.set(
      targetPath,
      source.replace(
          workerPattern,
          `$1'${workerAsset}?v=${workerHash}'`,
          ),
  );
}

const indexPath = path.join(directory, 'index.html');
let index = await read(indexPath);
for (const asset
         of ['execution.css',
             'execution-data.js',
             'execution-view.js',
             `${modelExplorerDist}/main_browser.js`,
]) {
  const hash = digest(await read(path.join(directory, asset)));
  const escaped = asset.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pattern = new RegExp(`((?:src|href)=")${escaped}(?:\\?v=[a-f0-9]+)?"`);
  if (!pattern.test(index)) {
    throw new Error('Execution dependency not found: ' + asset);
  }
  index = index.replace(pattern, `$1${asset}?v=${hash}"`);
}
edits.set(indexPath, index);
const componentPath = path.join(
    root,
    'src/features/graph/execution_graph/execution_graph.ts',
);
const component = await read(componentPath);
const framePattern = /src="graph-execution\/index\.html(?:\?v=[a-f0-9]+)?"/;
if (!framePattern.test(component)) {
  throw new Error('Execution frame URL not found');
}
edits.set(
    componentPath,
    component.replace(
        framePattern,
        `src="graph-execution/index.html?v=${digest(index)}"`,
        ),
);

const runtimePairs = [
  [
    'src/features/graph/execution_graph/runtime/execution_data.ts',
    'public/graph-execution/execution-data.js',
  ],
  [
    'src/features/graph/execution_graph/runtime/execution_view.ts',
    'public/graph-execution/execution-view.js',
  ],
];
for (const [tsRel, jsRel] of runtimePairs) {
  const tsContent = await read(path.join(root, tsRel));
  const jsContent = await read(path.join(root, jsRel));
  const normalizedTs = normalizeRuntimeSource(tsContent, true);
  const normalizedJs = normalizeRuntimeSource(jsContent, false);
  if (normalizedTs !== normalizedJs) {
    let diffPos = 0;
    while (diffPos < normalizedTs.length && diffPos < normalizedJs.length &&
           normalizedTs[diffPos] === normalizedJs[diffPos]) {
      diffPos++;
    }
    const tsExcerpt =
        normalizedTs.slice(Math.max(0, diffPos - 40), diffPos + 80);
    const jsExcerpt =
        normalizedJs.slice(Math.max(0, diffPos - 40), diffPos + 80);
    throw new Error(
        `Runtime parity mismatch between ${tsRel} and ${jsRel} at offset ${
            diffPos}:\n` +
        `  TS: ${tsExcerpt}\n` +
        `  JS: ${jsExcerpt}`);
  }
}

const changed = [];
for (const [file, content] of edits) {
  if (content === (await readFile(file, 'utf8'))) {
    continue;
  }
  changed.push(path.relative(root, file));
  if (!check) {
    await writeFile(file, content);
  }
}
if (check && changed.length) {
  throw new Error(
      'Stale Graph execution asset versions. Run node' +
          ' scripts/version_graph_execution.mjs: ' + changed.join(', '),
  );
}
console.log(
    check ?
        'PASS: Graph frame/dependency content versions and TS/JS parity match' :
        `Graph execution asset versions updated (${changed.length} files).`,
);
