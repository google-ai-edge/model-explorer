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

import {kvInputMetadata, kvOutputMetadata} from './kv_analysis_metadata';
import type {KvAnalysisLayer, KvTokenStructure} from '../../../data/contracts/kv';

function structure(channels = 128, count = 35): KvTokenStructure {
  return {
    status: 'ok',
    shape: [4, channels],
    dtype: 'float16',
    head_count: 4,
    channel_count: channels,
    batch_count: 1,
    position_start: 32,
    position_count: count,
  };
}
function layer(index: number, channels = 128): KvAnalysisLayer {
  return {
    layer: index,
    kind: 'key',
    status: 'ok',
    head_count: 4,
    position_start: 32,
    position_count: 35,
    cells: [],
    token_structure: {ref: structure(channels), target: structure(channels)},
  };
}
describe('KV logical analysis metadata', () => {
  it('uses Position-first full context geometry, reduces Heads in the output, and omits Batch from the display', () => {
    const rows = [layer(0), layer(1)];
    expect(kvInputMetadata(rows, 'ref')).toMatchObject({
      shape: '[35, 2, 4, 128]',
      dtype: 'float16',
    });
    expect(kvOutputMetadata(rows)).toMatchObject({
      dimensions: 'Position × Layer',
      shape: '[35, 2]',
      dtype: 'float64',
    });
  });
  it('does not invent a rectangular input for varying channel dimensions, while its output can be rectangular', () => {
    const rows = [layer(0), layer(1, 256)];
    expect(kvInputMetadata(rows, 'ref').shape).toBe('[35, 2, 4, Mixed]');
    expect(kvInputMetadata(rows, 'ref').layers[1].value).toBe('[35, 4, 256] · float16');
    expect(kvOutputMetadata(rows).shape).toBe('[35, 2]');
  });
  it('keeps runtime dtype and shape differences separate', () => {
    const rows = [layer(0)];
    rows[0].token_structure!.target = {...structure(128, 30), dtype: 'float32'};
    expect(kvInputMetadata(rows, 'target')).toMatchObject({
      shape: '[30, 1, 4, 128]',
      dtype: 'float32',
    });
    expect(kvOutputMetadata(rows).shape).toBe('—');
  });
  it('reports sizes independently of coordinate origins and keeps unavailable dimensions unknown', () => {
    expect(kvInputMetadata([], 'ref').shape).toBe('—');
    const rows = [layer(0), layer(1)];
    rows[1].token_structure!.ref!.position_start = 64;
    rows[1].token_structure!.target!.position_start = 64;
    expect(kvInputMetadata(rows, 'ref').shape).toBe('[35, 2, 4, 128]');
    expect(kvOutputMetadata(rows).shape).toBe('[35, 2]');
    rows[0].token_structure = {ref: null, target: null};
    expect(kvInputMetadata(rows, 'ref').shape).toBe('[—, 2, —, —]');
    expect(kvOutputMetadata([rows[0]]).shape).toBe('—');
  });
});

it('marks only varying dimensions as Mixed in inputs and outputs', () => {
  const rows = [layer(0), layer(1)];
  for (const side of ['ref', 'target'] as const) {
    rows[1].token_structure![side] = {...structure(256, 30), head_count: 8};
  }
  expect(kvInputMetadata(rows, 'ref').shape).toBe('[Mixed, 2, Mixed, Mixed]');
  expect(kvOutputMetadata(rows).shape).toBe('[Mixed, 2]');
});

it('counts scoped runtime positions without treating alignment gaps as input tokens', () => {
  const rows = [layer(3)];
  const scope = {
    range: {start: 100, end: 104},
    heads: [1, 3],
    alignment: {
      start: 100,
      pairs: [
        [32, 32],
        [33, null],
        [null, 33],
        [34, 34],
      ] as [number | null, number | null][],
    },
  };
  expect(kvInputMetadata(rows, 'ref', scope).shape).toBe('[3, 1, 2, 128]');
  expect(kvInputMetadata(rows, 'target', {...scope, range: {start: 100, end: 102}}).shape).toBe(
    '[1, 1, 2, 128]',
  );
  expect(kvOutputMetadata(rows, scope).shape).toBe('[4, 1]');
  expect(kvInputMetadata(rows, 'ref', {range: {start: 34, end: 36}, heads: [0]}).shape).toBe(
    '[2, 1, 1, 128]',
  );
});
