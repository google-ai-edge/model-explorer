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

import type {KvAnalysisLayer, KvTokenStructure} from '../../../data/contracts/kv';
import {defaultTensorLocation} from './kv_tensor_location';

const source = (start = 0, count = 100, heads = 4): KvTokenStructure => ({
  status: 'ok',
  shape: [heads, 8],
  dtype: 'float32',
  head_count: heads,
  channel_count: 8,
  batch_count: 1,
  position_start: start,
  position_count: count,
});
const row = (layer: number, ref = source(), target: KvTokenStructure | null = source()) =>
  ({layer, kind: 'key', token_structure: {ref, target}}) as KvAnalysisLayer;
const empty = {position: null, layer: null, head: null};

describe('default tensor location', () => {
  it('uses the current range and numeric layer/head order', () => {
    expect(
      defaultTensorLocation([row(7), row(2)], {start: 15, end: 40}, null, empty, [3, 1]),
    ).toEqual({position: 15, layer: 2, head: 1});
  });
  it('skips alignment gaps in favor of a paired position', () => {
    expect(
      defaultTensorLocation(
        [row(0)],
        {start: 32, end: 35},
        {
          start: 32,
          pairs: [
            [null, 0],
            [1, null],
            [2, 1],
          ],
        },
        empty,
        [0, 1],
      ),
    ).toEqual({position: 34, layer: 0, head: 0});
  });
  it('prefers paired captured intervals, then the first readable single side', () => {
    expect(
      defaultTensorLocation(
        [row(0, source(), null), row(1, source(20), source(25))],
        {start: 15, end: 40},
        null,
        empty,
        [0],
      ),
    ).toEqual({position: 25, layer: 1, head: 0});
    expect(
      defaultTensorLocation([row(0, source(20), null)], {start: 15, end: 40}, null, empty, [0]),
    ).toEqual({position: 20, layer: 0, head: 0});
  });
  it('retains a previous valid position, layer and head, including single-side values', () => {
    const preferred = {position: 27, layer: 7, head: 3};
    expect(
      defaultTensorLocation(
        [row(0), row(7, source(), null)],
        {start: 15, end: 40},
        null,
        preferred,
        [0],
      ),
    ).toEqual(preferred);
  });
  it('fills a selected column or layer without ignoring that coordinate', () => {
    expect(
      defaultTensorLocation(
        [row(7), row(2)],
        {start: 15, end: 40},
        null,
        {...empty, position: 21},
        [0],
      ),
    ).toEqual({position: 21, layer: 2, head: 0});
    expect(
      defaultTensorLocation([row(7), row(2)], {start: 15, end: 40}, null, {...empty, layer: 7}, [
        0,
      ]),
    ).toEqual({position: 15, layer: 7, head: 0});
  });
  it('replaces invalid coordinates and leaves uncaptured ranges empty', () => {
    expect(
      defaultTensorLocation(
        [row(2)],
        {start: 15, end: 40},
        null,
        {position: 120, layer: 9, head: 8},
        [2],
      ),
    ).toEqual({position: 15, layer: 2, head: 2});
    expect(
      defaultTensorLocation([row(0, source(80), null)], {start: 15, end: 40}, null, empty, [0]),
    ).toBeNull();
  });
});
