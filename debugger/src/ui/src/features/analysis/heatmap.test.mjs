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
import {fileURLToPath} from 'node:url';
async function load(path) {
  const {outputFiles} = await build({
    entryPoints: [fileURLToPath(new URL(path, import.meta.url))],
    bundle: true,
    write: false,
    format: 'esm',
    platform: 'node',
  });
  return import(
    'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
  );
}
const {
  HEATMAP_DEFINITIONS,
  heatmapAvailable,
  heatmapControls,
  reduceMetricValues,
  defaultPositionReduction,
} = await load('./heatmap_model.ts');
const {heatmapTraces} = await load('./heatmap_plot.ts');
const {kvHeatmapData} = await load('../kv/kv_panel/kv_heatmap_adapter.ts');

test('positional analysis rejects layer-only captures for the whole analysis', () => {
  for (const id of ['activations', 'intermediate'])
    assert.equal(heatmapAvailable(HEATMAP_DEFINITIONS[id], ['layer', 'syncPoint']), false);
  assert.equal(
    heatmapAvailable(HEATMAP_DEFINITIONS.intermediate, ['position', 'layer', 'syncPoint']),
    true,
  );
});
test('dimension controls share select/reduce semantics and preserve sparse member IDs', () => {
  assert.equal(heatmapControls(HEATMAP_DEFINITIONS.activations, {}, {})[0].id, 'position');
  assert.equal(defaultPositionReduction('cosine_similarity'), 'min');
  assert.equal(defaultPositionReduction('relative_l2'), 'max');
  const heads = heatmapControls(
    HEATMAP_DEFINITIONS.kv,
    {head: '7'},
    {head: [{id: '7', label: 'Head 7'}]},
  );
  assert.equal(heads[1].value, '7');
  assert.deepEqual(
    heads[1].choices.map((x) => x.id),
    ['7', 'min', 'max', 'mean'],
  );
  const points = heatmapControls(
    HEATMAP_DEFINITIONS.intermediate,
    {syncPoint: 'attention'},
    {syncPoint: [{id: 'attention', label: 'Attention'}]},
  );
  assert.equal(points[1].choices[0].id, 'attention');
  assert.equal(reduceMetricValues([null, 0.9, NaN, 0.2], 'min'), 0.2);
  assert.equal(reduceMetricValues([null, 0.9, 0.3], 'mean'), 0.6);
  assert.equal(reduceMetricValues([null, NaN], 'max'), null);
});
test('single-position KV adaptation keeps K/V identity, head policy and missing cells', () => {
  const bin = {start: 10, end: 11};
  const rows = [
    {
      layer: 4,
      bins: [{...bin, metrics: {cosine_similarity: {min: 0.8, max: 0.8, valid_count: 1}}}],
    },
  ];
  const data = kvHeatmapData(
    [4, 9],
    [{...bin, source: bin, overview: false}],
    [
      {kind: 'key', rows, overview: []},
      {kind: 'value', rows: [], overview: []},
    ],
    'cosine_similarity',
    'min',
    String,
  );
  assert.equal(data.planes[0].cells[0][0].value, 0.8);
  assert.equal(data.planes[0].cells[1][0].value, null);
  assert.equal(data.planes[1].cells[0][0].value, null);
  assert.deepEqual(data.planes[0].cells[0][0].evidence, {...bin, layer: 4, kind: 'key'});
  assert.equal(data.planes[0].cells[0][0].hover, 'Layer 4 · Position 10');
  const traces = heatmapTraces(data, {
    normalize: (v) => (v === null ? null : (1 - v) / 2),
    low: 'white',
    high: 'red',
    marker: 'black',
    filtered: false,
  });
  assert.equal(traces[0].yaxis, 'y');
  assert.equal(traces[1].yaxis, 'y2');
  assert.deepEqual(traces[0].x, [9.5, 10.5]);
  assert.deepEqual(traces[1].x, traces[0].x);
  assert.equal(traces[1].z[0][0], null);
  assert.deepEqual(traces[0].colorscale, traces[1].colorscale);
});
test('sampled overview keeps full evidence interval and worst similarity, not a fabricated position', () => {
  const source = {start: 0, end: 8};
  const data = kvHeatmapData(
    [2],
    [{start: 4, end: 8, source, overview: true}],
    [
      {
        kind: 'value',
        rows: [],
        overview: [
          {
            layer: 2,
            bins: [{...source, metrics: {cosine_similarity: {min: 0.2, max: 0.9, valid_count: 8}}}],
          },
        ],
      },
    ],
    'cosine_similarity',
    '3',
    String,
  );
  assert.equal(data.planes[0].cells[0][0].value, 0.2);
  assert.equal(data.planes[0].cells[0][0].evidence.start, 0);
});
test('generic heatmaps support single non-KV planes and reject more than two or malformed matrices', () => {
  const data = {
    layers: [2],
    positions: [{start: 0, end: 1}],
    planes: [
      {
        id: 'activation',
        label: 'Activations',
        cells: [[{value: 0.9, evidence: {layer: 2}, hover: 'Activation'}]],
      },
    ],
  };
  const style = {normalize: (v) => v, low: 'white', high: 'red', marker: 'black', filtered: false};
  assert.equal(heatmapTraces(data, style)[0].name, 'Activations');
  assert.throws(
    () => heatmapTraces({...data, planes: [...data.planes, ...data.planes, ...data.planes]}, style),
    /one or two/,
  );
  assert.throws(() => heatmapTraces({...data, layers: [2, 3]}, style), /dimensions/);
});

test('Positions reducers use real summaries after Heads and preserve missing means', () => {
  const bin = {start: 0, end: 3};
  const summary = {min: 0.1, max: 0.9, mean: 0.4, valid_count: 3};
  const render = (mode, value = summary) =>
    kvHeatmapData(
      [0],
      [{...bin, source: bin, overview: false}],
      [
        {
          kind: 'key',
          rows: [{layer: 0, bins: [{...bin, metrics: {cosine_similarity: value}}]}],
          overview: [],
        },
      ],
      'cosine_similarity',
      'mean',
      String,
      mode,
    ).planes[0].cells[0][0];
  assert.equal(render('min').value, 0.1);
  assert.equal(render('max').value, 0.9);
  assert.equal(render('mean').value, 0.4);
  assert.equal(render('mean').hover, 'Layer 0 · Position 0–2');
  assert.equal(render('mean', {...summary, mean: undefined}).value, null);
  assert.equal(render('mean', {...summary, valid_count: 0}).value, null);
});

const {tensorBins, tensorRangeRows} = await load('./position_tensor_model.ts');
test('tensor reductions apply sync points before position bins and preserve global coordinates', () => {
  const row = (position, anchor, value) => ({
    position,
    layer: 7,
    anchor,
    status: 'ok',
    metrics: {CosSim: {value, status: 'ok'}},
  });
  const source = [row(32, 'a', 0.2), row(32, 'b', 0.8), row(33, 'a', 0.6), row(33, 'b', null)];
  const bins = tensorBins({start: 32, end: 34}, 1);
  assert.deepEqual(bins, [{start: 32, end: 34}]);
  const result = tensorRangeRows(source, [7], bins, 'mean', 'Activations');
  const stats = result[0].bins[0].metrics.cosine_similarity;
  assert.equal(stats.min, 0.5);
  assert.equal(stats.max, 0.6);
  assert.equal(stats.mean, 0.55);
  assert.equal(stats.min_position, 32);
  const missing = tensorRangeRows(source, [8], bins, 'mean', 'Activations')[0].bins[0].metrics
    .cosine_similarity;
  assert.equal(missing.valid_count, 0);
  assert.equal(missing.mean, null);
});

const {heatmapMissingRanges} = await load('./heatmap_model.ts');
test('unavailable hatch runs preserve plane, layer and gaps between valid cells', () => {
  const cells = (values) => values.map((value) => ({value, evidence: null, hover: ''}));
  assert.deepEqual(
    heatmapMissingRanges({
      layers: [4],
      positions: [
        {start: 10, end: 11},
        {start: 11, end: 12},
        {start: 12, end: 13},
      ],
      planes: [
        {id: 'key', label: 'K', cells: [cells([null, null, 0])]},
        {id: 'value', label: 'V', cells: [cells([0, null, 1])]},
      ],
    }),
    [
      {plane: 0, layer: 4, start: 10, end: 12},
      {plane: 1, layer: 4, start: 11, end: 12},
    ],
  );
});

const {positionAxisLabels} = await load('./heatmap_plot.ts');
test('wide aggregate cells show inclusive ranges, narrow cells use a sparse Position scale', () => {
  const bins = [
    {start: 128, end: 136},
    {start: 136, end: 144},
  ];
  assert.deepEqual(positionAxisLabels(bins, {start: 128, end: 144}, 400).ticktext, [
    '128–135',
    '136–143',
  ]);
  const sparse = positionAxisLabels(bins, {start: 128, end: 144}, 60);
  assert.equal(sparse.tickmode, 'array');
  assert.deepEqual(sparse.tickvals, [131.5]);
  assert.deepEqual(sparse.ticktext, ['128–135']);
});

const {trendPoints} = await load('./trend_points.ts');
test('Trends uses the Heatmap bin reduction, centers and preserves missing intervals', () => {
  const bins = [
    {start: 17, end: 23},
    {start: 23, end: 29},
    {start: 32, end: 38},
  ];
  const summary = {min: 0.2, mean: 0.6, max: 0.9, valid_count: 6};
  for (const reduction of ['min', 'mean', 'max']) {
    assert.deepEqual(trendPoints(bins, [summary, null, summary], reduction), [
      {position: 19.5, value: summary[reduction], bin: 0},
      null,
      null,
      {position: 34.5, value: summary[reduction], bin: 2},
    ]);
  }
  assert.deepEqual(trendPoints([bins[0]], [{...summary, valid_count: 0}], 'mean'), [null]);
});
