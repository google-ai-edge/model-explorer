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
  entryPoints: [fileURLToPath(new URL('./kv_find_controller.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
  plugins: [
    {
      name: 'angular-stub',
      setup(builder) {
        builder.onResolve({filter: /^@angular\/core$/}, () => ({path: 'core', namespace: 'stub'}));
        builder.onLoad({filter: /.*/, namespace: 'stub'}, () => ({
          contents: `export const computed=read=>read; export function signal(value){const read=()=>value;read.set=next=>value=next;read.update=fn=>value=fn(value);return read;}`,
        }));
      },
    },
  ],
});
const {KvFindController, KV_FIND_PAGE} = await import(
  'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
);
const page = (offset, total) => ({
  offset,
  total,
  results: Array.from({length: Math.min(KV_FIND_PAGE, total - offset)}, (_, i) => ({
    layer: 0,
    position: offset + i,
    kind: 'key',
  })),
});
function fixture(total = 100) {
  const log = {selected: [], shown: [], popup: []};
  let data = page(0, total),
    loading = false;
  const host = {
    data: () => data,
    loading: () => loading,
    select: (result) => log.selected.push(result.position),
    setResults: (show) => log.shown.push(show),
    openPopup: () => log.popup.push('open'),
    closePopup: () => log.popup.push('close'),
  };
  const find = new KvFindController(host);
  find.receive(data, 0);
  return {find, log, setData: (next) => (data = next), setLoading: (value) => (loading = value)};
}

test('navigation within the loaded page selects at once; beyond it requests the page and selects on arrival', () => {
  const f = fixture(100);
  f.find.first();
  assert.deepEqual(f.log.selected, [0]);
  assert.equal(f.find.resultIndex(), 0);
  f.find.move(1);
  assert.equal(f.find.resultIndex(), 1);
  f.find.navigate(57);
  assert.equal(f.find.resultPage(), 1, 'page two requested');
  assert.deepEqual(f.log.selected, [0, 1], 'nothing selected until the page lands');
  f.setData(page(40, 100));
  f.find.receive(page(40, 100), 1);
  assert.deepEqual(f.log.selected, [0, 1, 57]);
  assert.equal(f.find.resultIndex(), 57);
  f.find.receive(page(40, 100), 1);
  assert.deepEqual(f.log.selected, [0, 1, 57], 'a pending jump is honoured once');
});

test('out-of-range, loading and a shrunken result set are handled', () => {
  const f = fixture(100);
  f.find.navigate(-1);
  f.find.navigate(100);
  f.setLoading(true);
  f.find.navigate(3);
  assert.deepEqual(f.log.selected, []);
  f.setLoading(false);
  f.find.resultPage.set(2);
  f.find.receive({offset: 80, total: 50, results: []}, 2);
  assert.equal(f.find.resultPage(), 1, 'the page moves back into range');
  f.find.navigate(5);
  f.find.reset();
  assert.equal(f.find.resultIndex(), -1);
});

test('formula application validates first and drives the results panel', () => {
  const f = fixture(0);
  f.find.open();
  assert.deepEqual(f.log.popup, ['open']);
  f.find.draftFormula = 'relative_l2 >';
  f.find.apply();
  assert.match(f.find.queryError(), /Expected a metric or value/);
  assert.equal(f.find.formula(), '');
  f.find.draftFormula = ' relative_l2 > 10% ';
  f.find.apply();
  assert.equal(f.find.formula(), 'relative_l2 > 10%');
  assert.deepEqual(f.log.shown, [true]);
  assert.deepEqual(f.log.popup, ['open', 'close']);
  f.find.showResults.set(true);
  f.find.clear();
  assert.equal(f.find.formula(), '');
  assert.equal(f.find.showResults(), false);
  assert.equal(f.find.resultPages(), 1);
});
