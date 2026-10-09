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

import {bundleSources} from './lib/angular_harness.mjs';

const load = async (file) =>
    (await bundleSources({'*': 'src/features/kv/kv_panel/' + file}, 'kv-view-'))
        .module;
const {defaultKvView, parseKvView, restoreKvView, KvViewCache, kvContextRange} =
    await load('kv_view_state.ts');
const plain = (value) => JSON.parse(JSON.stringify(value));
const identity = {
  captureId: 'capture-a',
  turn: 2
};
const context = (id, turn = 2) => ({
  id,
  label: id,
  turn,
  source: 'snapshot',
  phase: 'decode',
  moment: 'terminal',
  step: 9,
  forward_id: 10,
  runtime: 'CPU',
  layers: ['key', 'value'].map((kind) => ({
                                 layer: 3,
                                 kind,
                                 status: 'ok',
                                 head_count: 2,
                                 position_start: 10,
                                 position_count: 90,
                                 cells: [],
                               })),
});
const analysis = {
  contexts: [
    context('first'),
    {
      ...context('second'),
      moment: 'prefill_post',
      phase: 'prefill',
      step: null,
      forward_id: 1
    },
  ],
};
const snapshot = {
  ...defaultKvView(),
  contextId: 'first',
  kind: 'value',
  metric: 'max_abs',
  head: '1',
  colorScale: 'thresholds',
  view: 'trends',
  formula: 'relative_l2 > 10%',
  layer: 3,
  position: 25,
  selectionEnd: 40,
  column: true,
  zoomRange: {start: 20, end: 60},
  zoomHistory: [{start: 10, end: 100}],
  panelWidth: 420,
  showDetails: false,
  showResults: true,
  infoExpanded: false,
  savedExpanded: true,
  expandedHeads: {'value:1': true},
  headViews: {
    'value:1':
        {view: 'scatter', numericalExpanded: false, channelsExpanded: true}
  },
};
{
  const cache = new KvViewCache();
  cache.save(identity, snapshot);
  const saved = cache.read(identity);
  assert.deepEqual(plain(saved), plain(snapshot));
  saved.zoomHistory[0].end = 13;
  assert.equal(
      cache.read(identity).zoomHistory[0].end,
      100,
      'reads cannot mutate stored snapshots',
  );
  assert.equal(cache.read({...identity, turn: 1}), undefined);
  assert.equal(cache.read({...identity, captureId: 'capture-b'}), undefined);
  const restored = restoreKvView(cache.read(identity), analysis, identity);
  assert.deepEqual(plain(restored.state), plain(snapshot));
  assert.equal(restored.entryMissing, false);
  assert.equal('turn' in restored.state, false);
  assert.equal('phase' in restored.state, false);
  console.log(
      'PASS: KV return snapshot preserves view/Find/range/panels/head chart; capture and Turn are isolated and never restored as navigation',
  );
}
{
  for (const invalid
           of [null,
               {},
               {...snapshot, version: 8},
               {...snapshot, metric: 'fake'},
               {...snapshot, formula: 'javascript()'},
               {...snapshot, zoomRange: {start: 2, end: 1}},
               {...snapshot, headViews: {'value:1': {view: 'fake'}}},
               {...snapshot, position: Infinity},
  ]) {
    assert.equal(parseKvView(invalid), null);
  }
  const old = {
    ...snapshot,
    contextId: 'deleted',
    position: 999,
    selectionEnd: 1000,
    layer: 999,
    head: '999',
    expandedHeads: {'key:999': true},
    headViews: {'key:999': snapshot.headViews['value:1']},
  };
  const result = restoreKvView(old, analysis, identity);
  assert.equal(result.state.contextId, 'first');
  assert.equal(result.state.layer, 3);
  assert.equal(result.state.head, 'max');
  assert.equal(result.state.position, 10);
  assert.equal(result.state.selectionEnd, null);
  assert.equal(result.state.zoomRange, null);
  assert.deepEqual(plain(result.state.headViews), {});
  assert.equal(result.state.formula, snapshot.formula);
  assert.match(result.notice, /no longer available/);
  const shortened = restoreKvView(
      snapshot,
      {
        contexts: [
          {
            ...context('first'),
            layers: context('first').layers.map(
                (row) => ({...row, position_count: 20})),
          },
        ],
      },
      identity,
  );
  assert.equal(shortened.state.selectionEnd, null);
  assert.equal(shortened.state.zoomRange, null);
  assert.equal(shortened.state.zoomHistory.length, 0);
  const absent = restoreKvView(snapshot, {contexts: []}, identity);
  assert.equal(absent.state.contextId, '');
  assert.equal(kvContextRange(undefined), null);
  assert.equal(
      kvContextRange({
        ...context('missing'),
        layers: context('missing').layers.map((row) => ({
                                                ...row,
                                                position_start: null,
                                                position_count: null,
                                              })),
      }),
      null,
  );
  console.log(
      'PASS: unsupported snapshots and outdated observation/layer/head/ranges safely reset; missing metadata stays unavailable',
  );
}
{
  const entry = {
    sessionId: 'capture-a',
    turn: 2,
    moment: 'prefill_post',
    phase: 'prefill',
    step: null,
    forward_id: 1,
    runtime: 'CPU',
  };
  const explicit = restoreKvView(snapshot, analysis, identity, entry);
  assert.equal(explicit.state.contextId, 'second');
  assert.equal(explicit.state.zoomRange, null);
  assert.equal(explicit.state.selectionEnd, null);
  for (const patch
           of [{runtime: 'GPU'},
               {forward_id: 99},
               {step: 1},
               {moment: 'prefill_pre'},
               {phase: 'decode'},
  ]) {
    const missing =
        restoreKvView(snapshot, analysis, identity, {...entry, ...patch});
    assert.equal(missing.entryMissing, true);
    assert.equal(
        missing.state.contextId, '',
        'must not silently show a different observation');
  }
  assert.equal(
      restoreKvView(snapshot, analysis, identity, {
        ...entry,
        sessionId: 'other'
      }).state.contextId,
      'first',
  );
  // A native context keyed by logical identity has no step/forward of its own;
  // the entry resolves through its target snapshot and never through a foreign
  // forward.
  const native = {
    contexts: [
      {
        ...context('native'),
        step: null,
        forward_id: null,
        runtime: 'LiteRT-LM',
        snapshots: {
          ref: [{
            forward_id: 24,
            phase: 'decode',
            moment: 'terminal',
            runtime: 'LiteRT-LM'
          }],
          target: [{
            forward_id: 33,
            phase: 'decode',
            moment: 'terminal',
            runtime: 'LiteRT-LM'
          }],
        },
      },
    ],
  };
  const nativeEntry = {
    sessionId: 'capture-a',
    turn: 2,
    moment: 'terminal',
    phase: 'decode',
    step: 147,
    forward_id: 33,
    runtime: 'LiteRT-LM',
  };
  assert.equal(
      restoreKvView(snapshot, native, identity, nativeEntry).state.contextId,
      'native');
  for (const patch
           of [{forward_id: 24},
               {moment: 'prefill_post'},
               {phase: 'prefill'},
               {runtime: 'CPU'},
  ]) {
    const missing =
        restoreKvView(snapshot, native, identity, {...nativeEntry, ...patch});
    assert.equal(missing.entryMissing, true, JSON.stringify(patch));
  }
  console.log(
      'PASS: a native logical-identity context resolves its entry through the target snapshot only',
  );
  assert.equal(
      restoreKvView(snapshot, analysis, identity, {...entry, turn: 9})
          .state.contextId,
      'first',
  );
  assert.equal(
      restoreKvView(snapshot, {contexts: [context('foreign', 9)]}, identity)
          .state.contextId,
      '',
  );
  console.log(
      'PASS: explicit Token-to-KV observation wins over cache and matches every captured identity field',
  );
}

const {KvDataController} = await load('kv_data.ts');
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((a, b) => {
    resolve = a;
    reject = b;
  });
  return {promise, resolve, reject};
};
const flush = async () => {
  for (let i = 0; i < 10; i++) await Promise.resolve();
};
const calls = [];
const api = Object.fromEntries(
    ['kvMetadata', 'kvRange', 'kvSelection', 'kvFind'].map(
        (method) =>
            [method,
             (...args) => {
               const result = deferred();
               calls.push({method, args, ...result});
               return result.promise;
             },
]),
);
const data = new KvDataController(api);
{
  const received = [];
  data.loadMetadata(identity, (value) => received.push(value));
  const old = calls.at(-1);
  data.loadMetadata(
      {...identity, captureId: 'capture-b', turn: 3},
      (value) => received.push(value),
  );
  const current = calls.at(-1);
  assert.equal(old.args.at(-1).aborted, true);
  current.resolve({contexts: [context('new', 3)]});
  await flush();
  old.resolve(analysis);
  await flush();
  assert.equal(data.metadata.data().contexts[0].id, 'new');
  assert.equal(received.length, 1);
  data.loadMetadata(identity, () => {});
  calls.at(-1).reject(new Error('unavailable'));
  await flush();
  assert.equal(data.metadata.data(), null);
  assert.match(data.metadata.error(), /unavailable/);
  data.loadMetadata(identity, () => {});
  assert.equal(data.metadata.error(), '');
  calls.at(-1).resolve(analysis);
  await flush();
  assert.equal(data.metadata.loading(), false);
  console.log(
      'PASS: metadata captures identity at dispatch, ignores late responses, clears stale values on error and supports retry',
  );
}
{
  const query = {
    ...identity,
    contextId: 'first',
    selection: {start: 10, end: 20, layer: 3}
  };
  data.loadSelection(query);
  const old = calls.at(-1);
  data.loadSelection({...query, contextId: 'second'});
  const current = calls.at(-1);
  old.reject(new Error('late failure'));
  current.resolve({context_id: 'first', start: 10, end: 20, rows: []});
  await flush();
  assert.equal(data.selection.data(), null);
  assert.match(data.selection.error(), /did not match/);
  data.loadSelection(query);
  calls.at(-1).resolve({
    context_id: 'first',
    start: 10,
    end: 20,
    rows: [{metrics: {relative_l2: {value: null}}}],
  });
  await flush();
  assert.equal(data.selection.error(), '');
  assert.equal(data.selection.data().rows[0].metrics.relative_l2.value, null);
  const range = {
    ...identity,
    contextId: 'first',
    range: {start: 10, end: 30},
    bins: 10,
    kind: 'key',
    head: 'max',
    formula: '',
  };
  const before = calls.length;
  const cancel = data.loadRange(range);
  cancel();
  await new Promise((resolve) => setTimeout(resolve, 120));
  assert.equal(
      calls.length, before, 'cancelled resize debounce cannot issue a request');
  data.loadScale(range);
  const scale = calls.at(-1);
  assert.equal(scale.args[5], 1);
  assert.equal(scale.args[8], '');
  data.loadFind(
      {...identity, contextId: 'first', formula: 'max_abs > 0', page: 2},
      () => {});
  const find = calls.at(-1);
  assert.equal(find.args[4], 80);
  data.destroy();
  assert.equal(scale.args.at(-1).aborted, true);
  assert.equal(find.args.at(-1).aborted, true);
  find.resolve({total: 1, offset: 0, results: []});
  await flush();
  assert.equal(data.find.data(), null);
  console.log(
      'PASS: selection identity mismatch rejected, null metrics preserved, debounce cancelled, scale bounded, Find paged, destroy aborts every request',
  );
}
