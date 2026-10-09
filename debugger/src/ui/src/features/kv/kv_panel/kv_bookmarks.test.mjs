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
  entryPoints: [fileURLToPath(new URL('./kv_bookmarks.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {
  KV_BOOKMARK_CODEC: codec,
  kvBookmarkLabel,
  kvBookmarkKey,
} = await import(
  'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
);

test('legacy entries keep only their location; invalid ones are dropped', () => {
  assert.deepEqual(codec.parse({context: 'c', layer: 2, position: 5, kind: 'key'}), {
    context: 'c',
    layer: 2,
    position: 5,
  });
  assert.deepEqual(codec.parse({context: 'c', layer: null, position: 0}), {
    context: 'c',
    layer: null,
    position: 0,
  });
  for (const bad of [
    null,
    1,
    {context: 3, layer: 0, position: 0},
    {context: 'c', layer: -1, position: 0},
    {context: 'c', layer: 0, position: -1},
    {context: 'c', layer: 0.5, position: 0},
  ])
    assert.equal(codec.parse(bad), null);
});

test('identity, label and storage key', () => {
  assert.equal(codec.identity({context: 'c', layer: null, position: 7}), '["c",7,null]');
  assert.equal(
    kvBookmarkLabel({context: 'c', layer: null, position: 7}),
    'Position 7 · All layers',
  );
  assert.equal(kvBookmarkLabel({context: 'c', layer: 3, position: 7}), 'Position 7 · Layer 3');
  assert.equal(kvBookmarkKey('cap'), 'debugger.kv-bookmarks.cap');
});
