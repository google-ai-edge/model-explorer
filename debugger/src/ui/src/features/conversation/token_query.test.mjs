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

import assert from 'node:assert/strict';
import test from 'node:test';
import {fileURLToPath} from 'node:url';
import {build} from 'esbuild';

const {outputFiles} = await build({
  entryPoints: [fileURLToPath(new URL('./token_query.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {compileQuery, queryFields} = await import(
  'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
);

test('unknown propagation, types, percent-as-percent and invalid syntax', () => {
  assert.equal(compileQuery('NOT token_match')({token_match: null}), null);
  assert.equal(compileQuery('NOT token_match')({token_match: false}), true);
  assert.equal(
    compileQuery('kl > 0.005 AND relative_l2 > 0.4%')({kl: 0.01, relative_l2: 0.5}),
    true,
  );
  assert.equal(compileQuery('NOT (kl > 0.005)')({kl: null}), null);
  assert.throws(() => compileQuery('window.alert(1)'));
  assert.throws(() => compileQuery('token_match > true'));
});

test('requiresMetrics and the blank-formula message', () => {
  assert.equal(compileQuery('NOT token_match').requiresMetrics, false);
  assert.equal(compileQuery('token_match OR js > 1').requiresMetrics, true);
  assert.throws(() => compileQuery('  '), /reset to NOT token_match/);
  assert.deepEqual(Object.keys(queryFields)[0], 'token_match');
});
