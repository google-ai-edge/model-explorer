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

import {test} from 'node:test';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import vm from 'node:vm';
import {build} from 'esbuild';

const {outputFiles} = await build({
  entryPoints: [fileURLToPath(new URL('../graph_inspection_service.ts', import.meta.url))],
  bundle: true,
  write: false,
  platform: 'node',
  format: 'cjs',
  plugins: [
    {
      name: 'service-test-dependencies',
      setup(builder) {
        builder.onResolve(
          {filter: /^@angular\/core$|\/(report_api_service|report_state_service)$/},
          (args) => ({path: args.path.split('/').at(-1), namespace: 'stub'}),
        );
        builder.onLoad({filter: /.*/, namespace: 'stub'}, ({path}) => ({
          contents:
            path === 'core'
              ? `
      export const Injectable=()=>value=>value;
      export const inject=token=>globalThis.dependencies[token.name];
      export const computed=read=>read;
      export const untracked=read=>read();
      export const effect=read=>globalThis.effects.push(read);
      export class DestroyRef {}
      export function signal(value){const read=()=>value;read.set=next=>value=next;read.update=fn=>value=fn(value);return read;}
    `
              : `export class ${path === 'report_api_service' ? 'ReportApiService' : 'ReportStateService'} {}`,
        }));
      },
    },
  ],
});
function signal(value) {
  const read = () => value;
  read.set = (next) => (value = next);
  return read;
}
function deferred() {
  let resolve;
  const promise = new Promise((done) => (resolve = done));
  return {promise, resolve};
}
const flush = async () => {
  for (let index = 0; index < 8; index++) await Promise.resolve();
};
function fixture() {
  const baseline = {
    status: 'ok',
    reference: 'r',
    target: 't',
    shape: [1],
    metrics: {CosSim: {value: 0, status: 'ok'}},
  };
  const tensors = ['r', 'r2', 't', 't2'].map((id) => ({
    id,
    run: id.startsWith('r') ? 'ref' : 'target',
    graph: 'g',
    node: 'n',
    output: id,
    layer: 0,
    batch: 0,
    sample: 's',
    shape: [1],
    dtype: 'float32',
  }));
  const details = {
    node: {id: 'semantic', label: 'Semantic', namespace: '', incomingEdges: []},
    anchor: null,
    parameters: {},
    sources: [],
    tensors,
    executions: ['ref', 'target'].map((id) => ({
      id,
      graphs: [
        {id: 'g', nodes: [{id: 'n', label: 'Operation', incomingEdges: [], outputsMetadata: []}]},
      ],
    })),
    saved: {status: 'not_saved'},
  };
  const pending = [],
    effects = [],
    calls = [];
  const state = {
    captureId: signal('a'),
    batchId: signal(0),
    layer: signal(0),
    comparison: signal({batch: 0, rows: [{layer: 0, anchor: 'a', ...baseline}]}),
    refreshMetrics() {
      calls.push('refresh');
      this.comparison.set(null);
    },
  };
  const api = {
    node: async () => details,
    compareSelection(capture, payload, signal) {
      const wait = deferred();
      pending.push({capture, payload, signal, ...wait});
      return wait.promise;
    },
    saveMapping: async (capture, payload) => {
      calls.push({capture, payload});
      return {status: 'saved', record: payload, comparison: {...baseline, ...payload}};
    },
    removeMapping: async () => ({status: 'not_saved'}),
  };
  const module = {exports: {}};
  vm.runInNewContext(outputFiles[0].text, {
    module,
    exports: module.exports,
    effects,
    AbortController,
    dependencies: {ReportApiService: api, ReportStateService: state, DestroyRef: {onDestroy() {}}},
  });
  const service = new module.exports.GraphInspectionService();
  return {
    service,
    state,
    api,
    details,
    baseline,
    pending,
    calls,
    effects,
    context: {layer: 0, batch: 0, semantic: 'anchor:a'},
  };
}

test('initial pair retains captured zero; changed endpoints clear metrics and Cancel restores saved pair', async () => {
  const f = fixture();
  await f.service.select(f.context);
  assert.equal(f.service.metrics().CosSim.value, 0);
  assert.equal(f.service.scalarMetrics().CosSim, 0);
  f.service.beginMapping();
  f.service.selectTensor({side: 'target', id: 't2'});
  assert.equal(f.service.result(), null);
  assert.equal(f.service.canSave(), false);
  f.service.cancelMapping();
  assert.equal(f.pending[0].signal.aborted, true);
  assert.equal(f.service.selectedTarget(), 't');
  assert.equal(f.service.metrics().CosSim.value, 0);
  f.pending[0].resolve({...f.baseline, target: 't2', metrics: {CosSim: {value: 1, status: 'ok'}}});
  await flush();
  assert.equal(f.service.selectedTarget(), 't');
  assert.equal(f.service.metrics().CosSim.value, 0);
});
test('operation selection is separate; unknown tensors and operation mapping are rejected', async () => {
  const f = fixture();
  await f.service.select(f.context);
  f.service.selectOperation({side: 'target', graphId: 'g', nodeId: 'n'});
  assert.equal(f.service.result(), null);
  assert.equal(f.service.canMap(), false);
  assert.equal(Object.keys(f.service.scalarMetrics()).length, 0);
  f.service.beginMapping();
  assert.equal(f.service.mapping(), false);
  f.service.selectTensor({side: 'target', id: 'unknown'});
  assert.equal(f.service.operation().nodeId, 'n');
  f.service.selectTensor({side: 'ref', id: 'r'});
  assert.equal(f.service.operation(), null);
  assert.equal(f.service.metrics().CosSim.value, 0);
});
test('compare and save use captured IDs and service persistence; no optimistic metrics', async () => {
  const f = fixture();
  await f.service.select(f.context);
  f.service.beginMapping();
  f.service.selectTensor({side: 'target', id: 't2'});
  f.pending[0].resolve({...f.baseline, target: 't2'});
  await flush();
  assert.equal(f.service.canSave(), true);
  await f.service.saveMapping();
  assert.equal(f.service.mapping(), false);
  assert.equal(f.service.selectedTarget(), 't2');
  assert.equal(f.calls[0].capture, 'a');
  assert.equal(f.calls[0].payload.semantic, 'anchor:a');
  assert.equal(f.calls[1], 'refresh');
});
test('capture, batch, and layer changes discard selection and abort pending comparison', async () => {
  for (const [field, value] of [
    ['captureId', 'b'],
    ['batchId', 1],
    ['layer', 1],
  ]) {
    const f = fixture();
    await f.service.select(f.context);
    f.service.beginMapping();
    f.service.selectTensor({side: 'target', id: 't2'});
    f.state[field].set(value);
    f.effects[0]();
    assert.equal(f.service.context(), null);
    assert.equal(f.service.details(), null);
    assert.equal(f.pending[0].signal.aborted, true);
  }
});
test('late node response cannot overwrite a newer selection', async () => {
  const f = fixture(),
    first = deferred();
  let oldSignal;
  f.api.node = (_capture, _layer, _batch, _semantic, signal) => {
    oldSignal = signal;
    return first.promise;
  };
  const old = f.service.select(f.context);
  f.api.node = async () => ({...f.details, node: {id: 'new'}});
  await f.service.select({...f.context, semantic: 'anchor:b'});
  first.resolve(f.details);
  await old;
  assert.equal(oldSignal.aborted, true);
  assert.equal(f.service.details().node.id, 'new');
});
test('remove mapping waits for refreshed baseline and preserves missing values', async () => {
  const f = fixture();
  await f.service.select(f.context);
  await f.service.removeMapping();
  assert.equal(f.service.result(), null);
  assert.equal(f.service.selectedReference(), '');
  f.state.comparison.set({
    batch: 0,
    rows: [
      {
        layer: 0,
        anchor: 'a',
        reference: 'r',
        target: 't',
        shape: [1],
        status: 'shape_mismatch',
        metrics: {},
      },
    ],
  });
  f.effects[1]();
  f.service.beginMapping();
  assert.equal(f.service.canSave(), false);
  assert.equal(f.service.result().status, 'shape_mismatch');
});
