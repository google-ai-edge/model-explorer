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
  entryPoints: [fileURLToPath(new URL('./find_navigator_state.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {findNavigatorState} = await import(
  'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
);

test('no results disables everything and shows a dash', () => {
  assert.deepEqual(findNavigatorState({index: -1, total: 0}), {
    firstDisabled: true,
    previousDisabled: true,
    nextDisabled: true,
    label: '— / 0',
  });
});

test('first, middle and last results enable the right directions', () => {
  assert.deepEqual(findNavigatorState({index: 0, total: 3}), {
    firstDisabled: true,
    previousDisabled: true,
    nextDisabled: false,
    label: '1 / 3',
  });
  assert.deepEqual(findNavigatorState({index: 1, total: 3}), {
    firstDisabled: false,
    previousDisabled: false,
    nextDisabled: false,
    label: '2 / 3',
  });
  assert.deepEqual(findNavigatorState({index: 2, total: 3}), {
    firstDisabled: false,
    previousDisabled: false,
    nextDisabled: true,
    label: '3 / 3',
  });
});

test('busy and pending block navigation; pending hides the count', () => {
  assert.equal(findNavigatorState({index: 1, total: 3, busy: true}).nextDisabled, true);
  assert.equal(findNavigatorState({index: 1, total: 3, busy: true}).label, '2 / 3');
  const pending = findNavigatorState({index: -1, total: 0, pending: true});
  assert.equal(pending.label, '… / …');
  assert.equal(pending.firstDisabled, true);
});
