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
import {strict as assert} from 'node:assert';
import {fileURLToPath} from 'node:url';

const {outputFiles} = await build({
  entryPoints: [fileURLToPath(new URL(
      '../src/features/kv/kv_panel/kv_range_geometry.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node'
});
const geometry = await import(
    'data:text/javascript;base64,' +
    Buffer.from(outputFiles[0].text).toString('base64'));
const {
  KV_RANGE_MAX_BINS,
  KV_RANGE_MIN_COLUMN_WIDTH,
  KV_RANGE_MAX_EXACT_COLUMN_WIDTH,
  KV_RANGE_ROW_HEIGHT,
  rangeCapacity,
  rangeColumnWidth,
  rangeSelectionFrame,
  rangeBinAtPosition,
  rangeBinAtX,
  rangeFromBins,
  rangeFromBrush,
  rangeXAtPosition,
  rangeDragStarted,
  rangeOverlappingBins,
  rangeTrendSegments,
  rangeKeyboardSelection,
  rangeKeyboardBrush
} = geometry;
const bins = [
  {start: 131000, end: 131024}, {start: 131024, end: 131048},
  {start: 131048, end: 131072}
];
assert.equal(KV_RANGE_MAX_BINS, 1024);
assert.equal(KV_RANGE_MIN_COLUMN_WIDTH, 4);
assert.equal(KV_RANGE_MAX_EXACT_COLUMN_WIDTH, 24);
assert.equal(KV_RANGE_ROW_HEIGHT, 24);
assert.equal(rangeCapacity(0), 1);
assert.equal(rangeCapacity(84), 1);
assert.equal(rangeCapacity(112), 8);
assert.equal(rangeCapacity(1104), 256);
assert.equal(rangeCapacity(4176), 1024);
assert.equal(rangeCapacity(100000), 1024);
const exactBins = (start, count) => Array.from(
    {length: count}, (_, i) => ({start: start + i, end: start + i + 1}));
assert.equal(rangeColumnWidth(1104, [], null), 4);
assert.equal(rangeColumnWidth(1104, bins, {start: 131000, end: 131072}), 4);
assert.equal(
    rangeColumnWidth(1104, exactBins(100, 4), {start: 100, end: 104}), 24);
assert.equal(
    rangeColumnWidth(1104, exactBins(100, 64), {start: 100, end: 164}), 16);
assert.equal(
    rangeColumnWidth(1104, exactBins(100, 256), {start: 100, end: 356}), 4);
assert.equal(
    rangeColumnWidth(84, exactBins(100, 8), {start: 100, end: 108}), 4);
assert.equal(
    rangeColumnWidth(1104, exactBins(100, 4), {start: 0, end: 131072}), 4,
    'stale exact bins cannot expand a coarse loading range');
assert.equal(
    rangeColumnWidth(
        1104, [{start: 1, end: 2}, {start: 3, end: 4}], {start: 1, end: 3}),
    4,
    'sparse/mismatched positions do not pretend to be exact contiguous bins');
const narrowFrame = rangeSelectionFrame({x: 64, y: 36, width: 4, height: 24});
assert.deepEqual(narrowFrame, {x: 64.5, y: 36.5, width: 3, height: 23});
assert.ok(
    narrowFrame.width > 0 && narrowFrame.x + narrowFrame.width < 68,
    'a dense selected column keeps its contrast frame');
assert.equal(rangeBinAtPosition(bins, 131023), 0);
assert.equal(rangeBinAtPosition(bins, 131024), 1);
assert.equal(rangeBinAtPosition(bins, 131072), -1);
for (const uniform of [true, false]) {
  assert.equal(rangeBinAtX(bins, 63, 64, 96, uniform), -1);
  assert.equal(rangeBinAtX(bins, 96, 64, 96, uniform), 1);
  assert.equal(rangeBinAtX(bins, 160, 64, 96, uniform), 2);
  assert.equal(rangeBinAtX(bins, -500, 64, 96, uniform, true), 0);
  assert.equal(rangeBinAtX(bins, 900, 64, 96, uniform, true), 2);
}
assert.deepEqual(rangeFromBins(bins, 2, 0), {start: 131000, end: 131072});
assert.deepEqual(rangeFromBins(bins, 0, 2), rangeFromBins(bins, 2, 0));
assert.equal(rangeFromBins([], 0, 0), null);
assert.equal(rangeDragStarted(30, 34.9), false);
assert.equal(rangeDragStarted(30, 35), true);
assert.equal(rangeDragStarted(30, 25), true);
for (const uniform of [true, false]) {
  const single = [{start: 0, end: 131072}];
  assert.deepEqual(
      rangeFromBrush(single, 72, 88, 64, 32, uniform),
      {start: 32768, end: 98304});
  assert.deepEqual(
      rangeFromBrush(single, 88, 72, 64, 32, uniform),
      {start: 32768, end: 98304});
  assert.deepEqual(
      rangeFromBrush(single, -200, 900, 64, 32, uniform), single[0]);
  assert.deepEqual(
      rangeFromBrush([{start: 7, end: 8}], 65, 72, 64, 32, uniform),
      {start: 7, end: 8});
  assert.equal(rangeXAtPosition(single, 32768, 64, 32, uniform), 72);
}
assert.deepEqual(
    rangeFromBrush(
        [{start: 0, end: 100}, {start: 100, end: 150}], 72, 112, 64, 64, true),
    {start: 25, end: 125});
for (const uniform of [true, false]) {
  const single = [{start: 0, end: 131072}];
  const first =
      rangeKeyboardBrush(single, single[0], null, -1, 64, 32, uniform);
  assert.deepEqual(first.range, {start: 0, end: 98304});
  const second =
      rangeKeyboardBrush(single, single[0], first, -1, 64, 32, uniform);
  assert.deepEqual(second.range, {start: 0, end: 65536});
  assert.deepEqual(
      rangeKeyboardBrush(single, single[0], second, 1, 64, 32, uniform).range,
      first.range);
}
assert.deepEqual(
    rangeOverlappingBins(bins, {start: 131023, end: 131025}), [0, 1]);
assert.equal(rangeOverlappingBins(bins, {start: 0, end: 1}), null);
const summary = (min, max, min_position, max_position, valid_count = 24) =>
    ({min, max, min_position, max_position, valid_count});
const segments = rangeTrendSegments(bins, [
  summary(1, 9, 131020, 131001), summary(2, 8, 131025, 131046),
  summary(3, 7, 131069, 131050)
]);
assert.deepEqual(
    segments[0].map(point => point.position),
    [131001, 131020, 131025, 131046, 131050, 131069]);
const gap = rangeTrendSegments(bins, [
  summary(1, 9, 131020, 131001), summary(null, null, null, null, 0),
  summary(3, 7, 131069, 131050)
]);
assert.deepEqual(
    gap.map(segment => segment.map(point => point.bin)), [[0, 0], [2, 2]]);
assert.equal(rangeTrendSegments(bins, [null, null, null]).length, 0);
assert.deepEqual(
    rangeTrendSegments([{start: 10, end: 11}], [summary(0, 0, 10, 10, 1)]),
    [[{position: 10, value: 0, bin: 0}]]);
assert.equal(
    rangeTrendSegments([{start: 10, end: 11}], [summary(1, 2, 9, 12, 1)])
        .length,
    0);
assert.equal(
    rangeTrendSegments(
        [{start: 10, end: 11}, {start: 20, end: 21}],
        [summary(1, 1, 10, 10, 1), summary(2, 2, 20, 20, 1)])
        .length,
    2);
assert.deepEqual(
    rangeKeyboardSelection(bins, [2, 7], {...bins[0], layer: 2}, 'ArrowRight'),
    {...bins[1], layer: 2});
assert.deepEqual(
    rangeKeyboardSelection(bins, [2, 7], {...bins[2], layer: 2}, 'ArrowRight'),
    {...bins[2], layer: 2});
assert.deepEqual(
    rangeKeyboardSelection(bins, [2, 7], {...bins[1], layer: 2}, 'ArrowDown'),
    {...bins[1], layer: 7});
assert.deepEqual(
    rangeKeyboardSelection(bins, [2, 7], {...bins[1], layer: 2}, 'c'),
    {...bins[1], layer: null});
assert.deepEqual(
    rangeKeyboardSelection(
        bins, [2, 7], {...bins[1], layer: null}, 'ArrowRight'),
    {...bins[2], layer: null});
assert.equal(
    rangeKeyboardSelection(
        bins, [2, 7], {...bins[1], layer: 2}, 'ArrowDown', 2),
    null);
assert.deepEqual(
    rangeKeyboardSelection(bins, [2, 7], null, 'End', 7),
    {...bins[2], layer: 7});
assert.equal(rangeKeyboardSelection([], [], null, 'ArrowRight'), null);

// Dense layout changes presentation only: every exact Position remains
// addressable and plotted.
const denseExact = exactBins(130048, 1024),
      denseSpan = 1024 * KV_RANGE_MIN_COLUMN_WIDTH;
assert.equal(
    rangeColumnWidth(4176, denseExact, {start: 130048, end: 131072}), 4);
for (let index = 0; index < denseExact.length; index++) {
  const center = 64 + (index + .5) * 4;
  assert.equal(rangeBinAtX(denseExact, center, 64, denseSpan, true), index);
  assert.equal(rangeBinAtPosition(denseExact, 130048 + index), index);
}
const denseTrend = rangeTrendSegments(
    denseExact,
    denseExact.map(
        (bin, index) => summary(index, index, bin.start, bin.start, 1)));
assert.equal(denseTrend.length, 1);
assert.equal(denseTrend[0].length, 1024);
assert.equal(denseTrend[0][0].position, 130048);
assert.equal(denseTrend[0].at(-1).position, 131071);
assert.deepEqual(
    rangeKeyboardSelection(
        denseExact, [0, 31], {start: 131070, end: 131071, layer: 31},
        'ArrowRight'),
    {start: 131071, end: 131072, layer: 31});
const denseCoarse = Array.from(
    {length: 1024},
    (_, index) => ({start: index * 128, end: (index + 1) * 128}));
assert.equal(rangeColumnWidth(4176, denseCoarse, {start: 0, end: 131072}), 4);
assert.equal(rangeDragStarted(65, 73), true);
assert.deepEqual(
    rangeFromBrush(denseCoarse, 65, 73, 64, denseSpan, true),
    {start: 32, end: 288});
assert.deepEqual(
    rangeFromBrush(denseCoarse, 73, 65, 64, denseSpan, true),
    {start: 32, end: 288});
const expanded = exactBins(1000, 64),
      expandedWidth =
          rangeColumnWidth(1104, expanded, {start: 1000, end: 1064});
assert.equal(expandedWidth, 16);
assert.equal(
    rangeBinAtX(expanded, 64 + 16 * 31 + 8, 64, 64 * expandedWidth, true), 31);
assert.deepEqual(
    rangeFromBrush(expanded, 64 + 16, 64 + 48, 64, 64 * expandedWidth, true),
    {start: 1001, end: 1003});
console.log(
    'PASS: dense 4px/24px geometry, stable 1024-bin capacity, visible selection frames, exact Position preservation, half-open/reversed brushing, extrema gaps and keyboard selection');
