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
import {build} from 'esbuild';
const {outputFiles} = await build({
  entryPoints: [new URL('./stage_analysis.ts', import.meta.url).pathname],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {validStagePanels, restoreLogitPanel, numberStagePanels} = await import(
  'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
);
const panel = {
  id: 'analysis-1',
  turn: 1,
  stage: 'decode',
  kind: 'tokens',
  open: true,
  trend: {open: true, metric: 'kl', start: 0, end: 10},
};
test('reject invalid bookmarks before partially restoring a panel list', () => {
  assert.equal(validStagePanels([panel]), true);
  for (const list of [
    [panel, panel],
    [{...panel, stage: 'prefill'}],
    [{...panel, trend: {...panel.trend, metric: 'invented'}}],
    [{...panel, kv: {version: 1}}],
    [{...panel, id: '<script>'}],
  ])
    assert.equal(validStagePanels(list), false);
});
test('graph panels retain captured observation settings independently', () => {
  assert.equal(
    validStagePanels([
      {
        ...panel,
        kind: 'graph',
        trend: undefined,
        graph: {batchId: 4, metric: 'relative_l2', anchor: 'output'},
      },
    ]),
    true,
  );
  assert.equal(
    validStagePanels([
      {...panel, kind: 'graph', graph: {batchId: '4', metric: 'relative_l2', anchor: 'output'}},
    ]),
    false,
  );
});

test('retains Layer Activation inspection bookmarks and rejects invalid tensor coordinates', () => {
  const inspection = {
    plot: 'scatter',
    position: 44,
    layer: 2,
    elementRange: {start: 0, end: 16},
    valueRange: {start: -2.5, end: 3.5},
    selection: {channel: 4},
    reduction: 'mean',
  };
  const saved = {
    ...panel,
    kind: 'activations',
    trend: undefined,
    graph: {
      batchId: null,
      metric: 'cosine_similarity',
      anchor: 'output',
      mode: 'inspection',
      inspection,
    },
  };
  assert.equal(validStagePanels([saved]), true);
  for (const patch of [
    {position: -1},
    {position: 0.5},
    {layer: NaN},
    {plot: 'heatmap'},
    {selection: {channel: -1}},
    {elementRange: {start: -1, end: 2}},
    {valueRange: {start: 2, end: 2}},
    {reduction: 'sum'},
  ])
    assert.equal(
      validStagePanels([
        {...saved, graph: {...saved.graph, inspection: {...inspection, ...patch}}},
      ]),
      false,
    );
  assert.equal(validStagePanels([{...saved, graph: {...saved.graph, mode: 'unknown'}}]), false);
});

test('retains Layer Activation runtime visibility across bookmark serialization', () => {
  const saved = {
    ...panel,
    kind: 'activations',
    trend: undefined,
    graph: {
      batchId: null,
      metric: 'cosine_similarity',
      anchor: 'output',
      mode: 'inspection',
      inspection: {plot: 'index', position: 44, layer: 2},
    },
  };
  for (const visibleRuntimes of [
    undefined,
    Object.freeze([]),
    Object.freeze(['ref']),
    Object.freeze(['target']),
    Object.freeze(['ref', 'target']),
    Object.freeze(['target', 'ref']),
  ]) {
    const source = {
      ...saved,
      graph: {...saved.graph, inspection: {...saved.graph.inspection, visibleRuntimes}},
    };
    assert.equal(validStagePanels([source]), true);
    const restored = JSON.parse(JSON.stringify([source]));
    assert.equal(validStagePanels(restored), true);
    assert.deepEqual(restored[0].graph.inspection.visibleRuntimes, visibleRuntimes);
    assert.deepEqual(restoreLogitPanel(restored[0]), restored[0]);
  }
  for (const visibleRuntimes of [
    null,
    false,
    'ref',
    {},
    ['Reference'],
    ['ref', 'unknown'],
    ['ref', 'ref'],
    ['target', 'target'],
    ['ref', 'target', 'ref'],
    [0],
    [null],
    [undefined],
  ])
    assert.equal(
      validStagePanels([
        {
          ...saved,
          graph: {...saved.graph, inspection: {...saved.graph.inspection, visibleRuntimes}},
        },
      ]),
      false,
      `visibleRuntimes=${JSON.stringify(visibleRuntimes)}`,
    );
});

test('restore old logits bookmarks into the matching independent analysis', () => {
  for (const metric of ['entropy', 'margin', 'selected_probability']) {
    const restored = restoreLogitPanel({...panel, trend: {...panel.trend, metric}});
    assert.equal(restored.kind, 'distribution');
    assert.equal(restored.trend.metric, metric);
    assert.equal(validStagePanels([restored]), true);
    assert.equal(validStagePanels([{...restored, stage: 'prefill'}]), false);
  }
  assert.equal(restoreLogitPanel(panel).trend.metric, 'kl');
  assert.equal(
    restoreLogitPanel({...panel, trend: {...panel.trend, metric: 'js'}}).trend.metric,
    'js',
  );
  assert.equal(
    restoreLogitPanel({...panel, trend: {...panel.trend, metric: 'relative_l2'}}).trend.metric,
    'logit_rmse',
  );
});

test('restores logits inspection settings without moving KL/JS or losing the comparison range', () => {
  const settings = {
    ...panel.trend,
    logitView: 'inspection',
    logitPlot: 'scatter',
    logitElementRange: {start: 128, end: 256},
    logitValueRange: {start: -9.5, end: 2.25},
    logitElementReduction: 'max',
    logitTransform: 'centered',
    logitPosition: 42,
    logitOffset: 128,
    logitOrder: 'absolute_delta',
    logitSortDirection: 'ascending',
    positionRange: {start: 32, end: 64},
  };
  for (const metric of ['logit_rmse', 'logit_max_abs', 'kl', 'js']) {
    const saved = {...panel, trend: {...settings, metric}};
    const restored = restoreLogitPanel(saved);
    assert.equal(restored.kind, 'tokens');
    assert.deepEqual(restored.trend, saved.trend);
    assert.equal(validStagePanels([restored]), true);
  }
  assert.equal(restoreLogitPanel({...panel, trend: undefined}).trend.metric, 'logit_rmse');
});

test('moves legacy distribution divergence bookmarks back to Final Logits', () => {
  for (const metric of ['kl', 'js']) {
    const saved = {
      ...panel,
      kind: 'distribution',
      instanceNumber: 3,
      trend: {...panel.trend, metric},
    };
    const restored = restoreLogitPanel(saved);
    assert.equal(restored.kind, 'tokens');
    assert.deepEqual(restored.trend, saved.trend);
    assert.equal(restored.instanceNumber, undefined);
    assert.equal(validStagePanels([restored]), true);
  }
  const saved = {
    ...panel,
    kind: 'distribution',
    instanceNumber: 3,
    trend: {...panel.trend, metric: 'entropy'},
  };
  assert.deepEqual(restoreLogitPanel(saved), saved);
});

test('validates persisted logits view, plot, reduction, transform, position, page offset, and column sorting', () => {
  const validValues = {
    samplingView: ['comparison', 'inspection'],
    samplingPosition: [0, 42],
    samplingTopK: [1, 10, 1000],
    logitView: ['comparison', 'inspection'],
    logitPlot: ['index', 'scatter'],
    logitElementReduction: ['min', 'max', 'mean'],
    logitTransform: ['raw', 'centered'],
    logitPosition: [0, 42],
    logitOffset: [0, 256],
    logitOrder: ['index', 'ref', 'target', 'delta', 'absolute_delta'],
    logitSortDirection: ['ascending', 'descending'],
  };
  const invalidValues = {
    samplingView: [null, 'table', true],
    samplingPosition: [null, -1, 0.5, '42', NaN, Infinity],
    samplingTopK: [null, 0, -1, 0.5, '10', NaN, Infinity, 1001],
    logitView: [null, 'table', true],
    logitPlot: [null, 'line', 'table', true],
    logitElementReduction: [null, 'sum', 1],
    logitTransform: [null, 'softmax', 1],
    logitPosition: [null, -1, 0.5, '42', NaN, Infinity],
    logitOffset: [null, -1, 0.5, '256', NaN, Infinity],
    logitOrder: [null, 'value', false],
    logitSortDirection: [null, 'asc', 'desc', false, 1],
  };
  for (const [field, values] of Object.entries(validValues))
    for (const value of values)
      assert.equal(validStagePanels([{...panel, trend: {...panel.trend, [field]: value}}]), true);
  for (const [field, values] of Object.entries(invalidValues))
    for (const value of values)
      assert.equal(
        validStagePanels([{...panel, trend: {...panel.trend, [field]: value}}]),
        false,
        `${field}=${String(value)}`,
      );
});

test('keeps legacy inspection bookmarks valid without a saved plot or element viewport', () => {
  const saved = {...panel, trend: {...panel.trend, logitView: 'inspection'}};
  const restored = restoreLogitPanel(saved);
  assert.equal(validStagePanels([restored]), true);
  assert.deepEqual(restored.trend, saved.trend);
  assert.equal(restored.trend.logitPlot, undefined);
  assert.equal(restored.trend.logitElementRange, undefined);
  assert.equal(restored.trend.logitValueRange, undefined);
});

test('retains legacy absolute-difference ordering without inventing a saved direction', () => {
  const saved = {...panel, trend: {...panel.trend, logitOrder: 'absolute_delta'}};
  assert.equal(validStagePanels([saved]), true);
  const restored = restoreLogitPanel(saved);
  assert.equal(restored.trend.logitOrder, 'absolute_delta');
  assert.equal(restored.trend.logitSortDirection, undefined);
});

test('validates element index ranges separately from signed floating-point value ranges', () => {
  const validValues = {
    logitElementRange: [
      {start: 0, end: 1},
      {start: 200000, end: 262144},
    ],
    logitValueRange: [
      {start: -9.5, end: -2.25},
      {start: -1.25, end: 3.5},
      {start: 0, end: 1},
    ],
  };
  const invalidValues = {
    logitElementRange: [
      null,
      {},
      [],
      '0–10',
      {start: -1, end: 2},
      {start: 0.5, end: 2},
      {start: 0, end: 2.5},
      {start: 2, end: 2},
      {start: 3, end: 2},
      {start: NaN, end: 2},
      {start: 0, end: Infinity},
      {start: '0', end: 2},
    ],
    logitValueRange: [
      null,
      {},
      [],
      '-1–2',
      {start: 2, end: 2},
      {start: 3, end: -2},
      {start: -Infinity, end: 2},
      {start: 0, end: Infinity},
      {start: NaN, end: 2},
      {start: -1, end: NaN},
      {start: '-1', end: 2},
    ],
  };
  for (const [field, values] of Object.entries(validValues))
    for (const value of values)
      assert.equal(validStagePanels([{...panel, trend: {...panel.trend, [field]: value}}]), true);
  for (const [field, values] of Object.entries(invalidValues))
    for (const value of values)
      assert.equal(
        validStagePanels([{...panel, trend: {...panel.trend, [field]: value}}]),
        false,
        `${field}=${JSON.stringify(value)}`,
      );
});

test('sync point subsets survive bookmark validation and malformed selections are rejected', () => {
  const graphPanel = {
    ...panel,
    kind: 'graph',
    trend: undefined,
    graph: {batchId: null, metric: 'cosine', anchor: 'mean'},
  };
  for (const anchorSelection of [undefined, null, ['pre_attention_norm', 'mlp_output']]) {
    assert.equal(
      validStagePanels([{...graphPanel, graph: {...graphPanel.graph, anchorSelection}}]),
      true,
    );
  }
  for (const anchorSelection of [[], ['mlp_output', 'mlp_output'], [0], 'mlp_output']) {
    assert.equal(
      validStagePanels([{...graphPanel, graph: {...graphPanel.graph, anchorSelection}}]),
      false,
    );
  }
});

test('default view numbers start at zero in the UI and remain stable across restore and removal', () => {
  const restored = numberStagePanels([
    {...panel, kind: 'kv'},
    {...panel, id: 'analysis-2', kind: 'kv'},
    {...panel, id: 'analysis-3', kind: 'graph'},
  ]);
  assert.deepEqual(
    restored.map((p) => p.instanceNumber - 1),
    [0, 1, 0],
  );
  const remaining = numberStagePanels(restored.slice(1));
  assert.equal(remaining[0].instanceNumber - 1, 1);
  assert.deepEqual(numberStagePanels(remaining), remaining);
  assert.equal(validStagePanels(restored), true);
});

test('preserves unified sync point scope and legacy inspection anchors in bookmarks', () => {
  for (const kind of ['activations', 'graph']) {
    const saved = {
      ...panel,
      kind,
      trend: undefined,
      graph: {
        batchId: null,
        metric: 'cosine_similarity',
        anchor: 'min',
        anchorSelection: ['attention_output'],
        mode: 'inspection',
        inspection: {plot: 'index', position: 44, layer: 2},
      },
    };
    for (const anchorScope of [undefined, 'output', 'internal', 'all'])
      for (const anchor of [undefined, 'output', 'attention_output', 'x'.repeat(128)]) {
        const item = {
          ...saved,
          graph: {
            ...saved.graph,
            anchorScope,
            inspection: {...saved.graph.inspection, anchor},
          },
        };
        const restored = JSON.parse(JSON.stringify([item]));
        assert.equal(validStagePanels(restored), true);
        assert.deepEqual(restored[0].graph.inspection.anchor, anchor);
        assert.deepEqual(restored[0].graph.anchorScope, anchorScope);
      }
    for (const anchor of [null, '', '   ', 3, false, 'x'.repeat(129)])
      assert.equal(
        validStagePanels([
          {
            ...saved,
            graph: {...saved.graph, inspection: {...saved.graph.inspection, anchor}},
          },
        ]),
        false,
      );
    for (const anchorScope of [null, '', 'legacy', 3, false])
      assert.equal(
        validStagePanels([
          {
            ...saved,
            graph: {...saved.graph, anchorScope},
          },
        ]),
        false,
      );
  }
});

test('shares Layer Activation numbering with legacy intermediate views without losing their configuration', () => {
  const legacy = {
    ...panel,
    kind: 'graph',
    instanceNumber: 1,
    trend: undefined,
    graph: {batchId: 4, metric: 'relative_l2', anchor: 'mean', anchorSelection: ['mlp_output']},
  };
  const activation = {
    ...panel,
    id: 'analysis-2',
    kind: 'activations',
    instanceNumber: 1,
    trend: undefined,
    graph: {batchId: null, metric: 'cosine_similarity', anchor: 'output'},
  };
  const restored = numberStagePanels([
    legacy,
    activation,
    {...activation, id: 'analysis-3', instanceNumber: undefined},
  ]);
  assert.deepEqual(
    restored.map((p) => p.instanceNumber),
    [1, 2, 3],
  );
  assert.equal(restored[0], legacy);
  assert.deepEqual(restored[1].graph, activation.graph);
  assert.deepEqual(numberStagePanels(restored), restored);
  assert.deepEqual(
    numberStagePanels(restored.slice(1)).map((p) => p.instanceNumber),
    [2, 3],
  );
  assert.equal(validStagePanels(restored), true);
});

 test('retains Graph Diff as an Intermediate Tensors comparison view', () => {
 const graph = {batchId: null, metric: 'cosine_similarity', anchor: 'output', view: 'graph'};
 assert.equal(validStagePanels([{...panel, kind: 'activations', trend: undefined, graph}]), true);
 assert.equal(validStagePanels([{...panel, kind: 'activations', trend: undefined, graph: {...graph, view: 'unknown'}}]), false);
 });
