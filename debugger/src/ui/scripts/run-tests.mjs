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

import {spawnSync} from 'node:child_process';
import {readdirSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const feature = process.argv[2];
if (feature && !/^[a-z_]+$/.test(feature))
  throw new Error('Invalid feature name');

function run(args) {
  const result =
      spawnSync(process.execPath, args, {cwd: root, stdio: 'inherit'});
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

function testsIn(directory) {
  return readdirSync(directory, {withFileTypes: true}).flatMap((entry) => {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) return testsIn(file);
    return /\.test\.(?:mjs|cjs)$/.test(entry.name) ? [file] : [];
  });
}

const checks =
    readdirSync(path.join(root, 'scripts'))
        .filter((name) => /^check_.*\.mjs$/.test(name))
        .filter((name) => !feature || name.startsWith(`check_${feature}`))
        .sort();
const featureTests =
    testsIn(
        path.join(root, 'src', ...(feature ? ['features', feature] : [])),
        )
        .sort();
if (!checks.length && !featureTests.length) throw new Error('No tests found');
console.log(
    `Running ${checks.length} regression scripts and ${
        featureTests.length} feature test files.`,
);
for (const check of checks) run([path.join('scripts', check)]);
if (featureTests.length) run(['--test', ...featureTests]);
