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
  entryPoints: [fileURLToPath(new URL('./bookmark_store.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
  plugins: [
    {
      name: 'angular-signal-stub',
      setup(builder) {
        builder.onResolve({filter: /^@angular\/core$/}, () => ({path: 'core', namespace: 'stub'}));
        builder.onLoad({filter: /.*/, namespace: 'stub'}, () => ({
          contents: `export function signal(value){const read=()=>value;read.set=next=>value=next;read.update=fn=>value=fn(value);return read;}`,
        }));
      },
    },
  ],
});
const {BookmarkStore, decodeBookmarks, encodeBookmarks, BOOKMARK_SAVE_ERROR} = await import(
  'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
);
const codec = {
  parse: (value) => (value && Number.isInteger(value.n) ? {n: value.n} : null),
  identity: (value) => String(value.n),
};
function memoryStorage(failing = false) {
  const data = new Map();
  return {
    data,
    getItem: (key) => data.get(key) ?? null,
    setItem(key, value) {
      if (failing) throw new DOMException('quota', 'QuotaExceededError');
      data.set(key, value);
    },
  };
}

test('decode accepts the legacy bare array and the versioned envelope', () => {
  assert.deepEqual(decodeBookmarks('[{"n":1},{"n":2}]', codec), [{n: 1}, {n: 2}]);
  assert.deepEqual(decodeBookmarks('{"version":1,"items":[{"n":3}]}', codec), [{n: 3}]);
});

test('decode isolates corrupt entries, drops duplicates and survives bad JSON', () => {
  assert.deepEqual(decodeBookmarks('[null,{"n":1},false,{"n":"x"},{"n":1},{"n":2}]', codec), [
    {n: 1},
    {n: 2},
  ]);
  for (const raw of [null, '{', 'null', '{}', '[false,null,0]', '{"items":5}'])
    assert.deepEqual(decodeBookmarks(raw, codec), []);
});

test('encode writes the envelope that decode reads back', () => {
  const text = encodeBookmarks([{n: 4}]);
  assert.equal(text, '{"version":1,"items":[{"n":4}]}');
  assert.deepEqual(decodeBookmarks(text, codec), [{n: 4}]);
});

test('store reads per key, caches replaced lists and reports a refused write', () => {
  const storage = memoryStorage();
  storage.data.set('k.a', '[{"n":1}]');
  const store = new BookmarkStore(codec, () => storage);
  assert.deepEqual(store.read('k.a'), [{n: 1}]);
  assert.deepEqual(store.read('k.b'), []);
  assert.equal(store.set('k.a', [{n: 2}, {n: 2}, {n: 1}]), true);
  assert.deepEqual(store.read('k.a'), [{n: 2}, {n: 1}]);
  assert.equal(storage.data.get('k.a'), '{"version":1,"items":[{"n":2},{"n":1}]}');
  assert.equal(store.error(), '');
  const failing = new BookmarkStore(codec, () => memoryStorage(true));
  assert.equal(failing.set('k', [{n: 9}]), false);
  assert.equal(failing.error(), BOOKMARK_SAVE_ERROR);
  assert.deepEqual(failing.read('k'), [{n: 9}], 'the list stays usable for this page');
});

test('a missing storage object reads as empty instead of throwing', () => {
  const store = new BookmarkStore(codec, () => {
    throw new ReferenceError('localStorage is not defined');
  });
  assert.deepEqual(store.read('k'), []);
  assert.equal(store.set('k', [{n: 1}]), false);
});
