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
import {readdirSync, readFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

// data/ and shared/ are leaves: features/ and app/ may import them, never the
// reverse.
const root = fileURLToPath(new URL('../src/', import.meta.url));
const leaves = ['data', 'shared'];
const forbidden =
    ['features', 'app'].map((name) => path.join(root, name) + path.sep);
function files(directory) {
  return readdirSync(directory, {withFileTypes: true}).flatMap((entry) => {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) return files(file);
    return /\.ts$/.test(entry.name) && !/\.test\./.test(entry.name) ? [file] :
                                                                      [];
  });
}
const violations = [];
for (const leaf of leaves)
  for (const file of files(path.join(root, leaf))) {
    const source = readFileSync(file, 'utf8');
    for (const match of source.matchAll(
             /from\s+'([^']+)'|import\(\s*'([^']+)'\s*\)/g)) {
      const specifier = match[1] ?? match[2];
      if (!specifier.startsWith('.')) continue;
      const target = path.resolve(path.dirname(file), specifier);
      if (forbidden.some((prefix) => target.startsWith(prefix)))
        violations.push(`${path.relative(root, file)} → ${specifier}`);
    }
  }
assert.deepEqual(violations, [], 'leaf layers import feature or app code');
console.log(`PASS: src/${
    leaves.join(' and src/')} import nothing from src/features or src/app`);
