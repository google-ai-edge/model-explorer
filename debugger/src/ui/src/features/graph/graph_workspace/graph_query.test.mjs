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
  entryPoints: [fileURLToPath(new URL('./graph_query.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
});
const {emptyGraphQuery, graphAnchor, graphQueryCount, matchesGraphQuery} = await import(
  'data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64')
);

const model = {
  layers: [
    {def: 0, attrs: {}},
    {def: 1, attrs: {}},
  ],
  semantic_graph: [
    {
      anchors: [
        {
          id: 'residual',
          of: 'add:2',
          label: 'Attention result',
          semantic: 'Residual after attention',
        },
      ],
    },
    {
      anchors: [
        {id: 'residual', of: 'mlp:7', label: 'MLP result', semantic: 'Residual after feed forward'},
      ],
    },
  ],
};
const row = (metrics = {CosSim: {value: 0, status: 'ok'}}, overrides = {}) => ({
  layer: 0,
  anchor: 'residual',
  reference: 'REF_tensor_17',
  target: 'target_tensor_42',
  shape: [1, 4],
  status: 'ok',
  metrics,
  ...overrides,
});
const query = (overrides = {}) => ({...emptyGraphQuery(), ...overrides});
const matches = (candidate, filters = {}, semantic = model) =>
  matchesGraphQuery(candidate, semantic, query(filters));

test('blank filters keep rows with missing comparisons and count only applied conditions', () => {
  assert.equal(matches(row({})), true);
  assert.equal(matches(row({}), {text: ' \n ', threshold: ' \t '}), true);
  assert.equal(graphQueryCount(query({text: ' ', threshold: ' '})), 0);
  assert.equal(graphQueryCount(query({threshold: '0'})), 1);
  assert.equal(
    graphQueryCount(query({text: 'a', anchor: 'residual', threshold: '0', withMetrics: true})),
    4,
  );
  const draft = emptyGraphQuery();
  draft.text = 'draft';
  assert.equal(emptyGraphQuery().text, '');
});

test('anchor lookup uses the row layer definition and handles missing semantic metadata', () => {
  assert.equal(graphAnchor(model, row()).label, 'Attention result');
  assert.equal(graphAnchor(model, row({}, {layer: 1})).label, 'MLP result');
  assert.equal(graphAnchor(model, row({}, {layer: 99})), undefined);
  assert.equal(graphAnchor(null, row()), undefined);
  assert.equal(matches(row(), {anchor: 'residual'}), true);
  assert.equal(matches(row(), {anchor: 'res'}), false);
});

test('name search covers both tensor IDs and the correct anchor ID, label and semantic name', () => {
  for (const text of [
    ' REF_TENSOR ',
    'TARGET_tensor_42',
    'RESIDUAL',
    'Attention result',
    'after attention',
  ]) {
    assert.equal(matches(row(), {text}), true, text);
  }
  assert.equal(matches(row({}, {layer: 1}), {text: 'MLP result'}), true);
  assert.equal(matches(row({}, {layer: 1}), {text: 'Attention result'}), false);
  assert.equal(matches(row(), {text: 'not present'}), false);
  assert.equal(matches(row(), {text: 'ref_tensor'}, null), true);
});

test('zero is a captured metric and strict zero thresholds do not behave like missing values', () => {
  assert.equal(matches(row(), {withMetrics: true}), true);
  assert.equal(matches(row(), {threshold: '0', operator: 'lt'}), false);
  assert.equal(matches(row(), {threshold: '0', operator: 'gt'}), false);
  assert.equal(matches(row(), {threshold: '0.01', operator: 'lt'}), true);
  assert.equal(matches(row(), {threshold: '-0.01', operator: 'gt'}), true);
});

test('missing and nonfinite metrics fail numerical conditions and captured-metric filtering', () => {
  for (const value of [null, undefined, NaN, Infinity, -Infinity]) {
    const candidate = row({CosSim: {value, status: 'unavailable'}});
    assert.equal(matches(candidate, {withMetrics: true}), false, String(value));
    assert.equal(matches(candidate, {threshold: '1', operator: 'lt'}), false, String(value));
    assert.equal(matches(candidate, {threshold: '-1', operator: 'gt'}), false, String(value));
  }
  assert.equal(matches(row({}), {threshold: '1'}), false);
  assert.equal(matches(row({CosSim: undefined}), {withMetrics: true}), false);
  assert.equal(
    matches(row({CosSim: {value: null, status: 'unavailable'}, RMSE: {value: 0, status: 'ok'}}), {
      withMetrics: true,
    }),
    true,
  );
});

test('numerical filtering reads its selected metric independently of other valid metrics', () => {
  const candidate = row({CosSim: {value: 0.9, status: 'ok'}, RMSE: {value: 0.02, status: 'ok'}});
  assert.equal(matches(candidate, {metric: 'RMSE', threshold: '.1', operator: 'lt'}), true);
  assert.equal(matches(candidate, {metric: 'CosSim', threshold: '.1', operator: 'lt'}), false);
  assert.equal(matches(candidate, {metric: 'Relative L2', threshold: '1'}), false);
});

test('invalid thresholds fail closed; decimal and scientific notation remain valid', () => {
  for (const threshold of [
    'not a number',
    'Infinity',
    '-Infinity',
    'NaN',
    '1e999',
    '0; throw Error()',
  ]) {
    assert.equal(matches(row(), {threshold}), false, threshold);
  }
  assert.equal(matches(row(), {threshold: ' 1e-3 '}), true);
  assert.equal(matches(row(), {threshold: '-1e-3', operator: 'gt'}), true);
});

test('text, anchor, numerical and availability conditions all constrain the same row', () => {
  const filters = {text: 'target', anchor: 'residual', threshold: '.1', withMetrics: true};
  assert.equal(matches(row(), filters), true);
  assert.equal(matches(row({}, {reference: null, target: null}), filters), false);
  assert.equal(matches(row(undefined, {anchor: 'other'}), filters), false);
  assert.equal(matches(row({CosSim: {value: 0.5, status: 'ok'}}), filters), false);
});
