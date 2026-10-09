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

import {build} from 'esbuild';
import assert from 'node:assert/strict';
import test from 'node:test';
import {fileURLToPath} from 'node:url';

const {outputFiles} = await build({
  entryPoints: [fileURLToPath(new URL('./graph_view_state.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {graphViewKey, parseGraphView, resolveGraphView} = await import(
  'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
);
const snapshot = () => ({
  version: 1,
  query: {
    text: 'residual',
    anchor: '',
    metric: 'CosSim',
    operator: 'lt',
    threshold: '0',
    withMetrics: false,
  },
  metric: 'CosSim',
  details: true,
  results: false,
  trends: true,
  detailsWidth: 350,
  trendHeight: 140,
  context: {layer: 0, semantic: 'anchor:output'},
  execution: true,
  viewport: {layer: 0, zoom: 1.2, left: 40, top: 150},
});
const model = {
  layers: [{def: 0}, {def: 1}],
  semantic_graph: [
    {nodes: [], anchors: [{id: 'output'}]},
    {nodes: [], anchors: [{id: 'other'}]},
  ],
};

test('Graph return snapshots preserve query, panel and canvas state with independent copies', () => {
  const original = snapshot();
  const restored = resolveGraphView(original, model, 0, ['CosSim']);
  assert.deepEqual(restored, original);
  original.query.text = 'edited';
  original.viewport.top = 900;
  assert.equal(restored.query.text, 'residual');
  assert.equal(restored.viewport.top, 150);
  assert.notEqual(graphViewKey('capture:1', 2), graphViewKey('capture', 12));
  assert.notEqual(graphViewKey('capture', 1), graphViewKey('capture', 2));
});

test('returning to another layer discards old selection and viewport without changing current context', () => {
  const restored = resolveGraphView(snapshot(), model, 1, ['CosSim']);
  assert.equal(restored.context, null);
  assert.equal(restored.viewport, null);
  assert.equal(restored.execution, false);
  assert.equal(restored.query.text, 'residual');
});

test('changed graph metadata rejects stale semantic selection and unsupported metrics', () => {
  const source = snapshot();
  source.context.semantic = 'anchor:missing';
  source.metric = 'obsolete';
  source.query.metric = 'obsolete';
  const restored = resolveGraphView(source, model, 0, ['RMSE']);
  assert.equal(restored.context, null);
  assert.equal(restored.execution, false);
  assert.equal(restored.metric, 'RMSE');
  assert.equal(restored.query.metric, 'RMSE');
});

test('unknown versions and invalid viewport values are rejected; panel sizes are bounded', () => {
  assert.equal(parseGraphView({...snapshot(), version: 99}), null);
  assert.equal(parseGraphView({...snapshot(), query: {text: 'partial'}}), null);
  assert.equal(
    parseGraphView({...snapshot(), viewport: {...snapshot().viewport, zoom: NaN}}),
    null,
  );
  assert.equal(parseGraphView({...snapshot(), viewport: {...snapshot().viewport, left: -1}}), null);
  assert.equal(parseGraphView({...snapshot(), detailsWidth: 900}).detailsWidth, 560);
});
