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

import {
  pairedChannels,
  scatterExtent,
  scatterRelayout,
  scatterRepresentatives,
  validValueRange,
} from './kv_scatter_model';
import {defaultKvView, parseKvView} from './kv_view_state';
import type {KvHeadChannel} from '../../../data/contracts/kv';
const row = (channel: number, ref: number | null, target: number | null) =>
  ({channel, ref, target, delta: null, abs_delta: null, status: 'ok'}) as KvHeadChannel;
describe('raw channel scatter domains', () => {
  it('keeps every small tensor element, including points outside the current zoom', () => {
    const rows = [row(7, 1, 2), row(8, 8, 9), row(9, null, 1)];
    const result = scatterRepresentatives(rows, {start: 0, end: 5});
    expect(result.rows.map((r) => r.channel)).toEqual([7, 8]);
    expect(result).toMatchObject({total: 2, visible: 1, rendered: 2, grouped: false});
  });
  it('retains the largest absolute difference per cell, selected identity, and deterministic ties', () => {
    const rows = [
      row(0, 1, 2),
      row(1, 1, 4),
      row(2, 2, 4),
      row(3, 8, 8),
      row(4, 1, 9),
      row(5, 9, 1),
    ];
    const range = {start: 0, end: 10};
    expect(scatterRepresentatives(rows, range, null, 4).rows.map((r) => r.channel)).toEqual([
      1, 3, 4, 5,
    ]);
    expect(
      scatterRepresentatives([...rows].reverse(), range, 0, 4).rows.map((r) => r.channel),
    ).toEqual([0, 1, 3, 4, 5]);
    expect(scatterRepresentatives(rows, range, null, 4, false).rows.map((r) => r.channel)).toEqual([
      0, 3, 4, 5,
    ]);
    const zoomed = scatterRepresentatives(rows, {start: 0, end: 5}, null, 4);
    expect(zoomed.rows.map((r) => r.channel)).toEqual([0, 1, 2]);
    expect(zoomed.visible).toBe(3);
    expect(zoomed.grouped).toBe(false);
  });
  it('shows every visible raw element after a large tensor is zoomed to the point limit', () => {
    const rows = Array.from({length: 5000}, (_, index) =>
      row(index, index < 4096 ? 1 : 10, index < 4096 ? 1 : 10),
    );
    expect(scatterRepresentatives(rows, {start: 0, end: 11}).grouped).toBe(true);
    const zoomed = scatterRepresentatives(rows, {start: 0, end: 2}, 65);
    expect(zoomed).toMatchObject({total: 5000, visible: 4096, rendered: 4096, grouped: false});
    expect(zoomed.rows).toEqual(rows.slice(0, 4096));
    expect(zoomed.rows[65]).toBe(rows[65]);
    const selectedOutside = scatterRepresentatives(rows, {start: 0, end: 2}, 4999);
    expect(selectedOutside.rows).toEqual(zoomed.rows);
    expect(selectedOutside.grouped).toBe(false);
  });
  it('bounds a full vocabulary while full extents still include every paired element', () => {
    const rows = Array.from({length: 262144}, (_, index) =>
      row(index, index % 512, Math.floor(index / 512)),
    );
    rows.push(row(262144, -1000, 2000));
    const range = scatterExtent([rows]);
    const result = scatterRepresentatives(rows, range, 131328);
    expect(result.total).toBe(262145);
    expect(result.visible).toBe(262145);
    expect(result.rendered).toBeLessThanOrEqual(4097);
    expect(result.grouped).toBe(true);
    expect(result.rows.some((r) => r.channel === 262144)).toBe(true);
    expect(result.rows.some((r) => r.channel === 131328)).toBe(true);
    expect(range.start).toBeLessThan(-1000);
    expect(range.end).toBeGreaterThan(2000);
  });
  it('uses all paired channels across both caches, retaining Head dimension identity', () => {
    const key = [row(120, -0.125, 0.75), row(4095, 3, 4), row(3, null, 1000)];
    expect(pairedChannels(key).map((r) => r.channel)).toEqual([120, 4095]);
    const range = scatterExtent([key, [row(20, -8, 2)]]);
    expect(range.start).toBeLessThan(-8);
    expect(range.end).toBeGreaterThan(4);
    expect(range.end).toBeLessThan(5);
  });
  it('provides finite nonzero domains for empty and constant data', () => {
    expect(scatterExtent([[]])).toEqual({start: -1, end: 1});
    expect(validValueRange(scatterExtent([[row(0, 0, 0)]]))).toBe(true);
  });
  it('unites negative fractional zoom bounds from either cache without integer clamping', () => {
    expect(
      scatterRelayout({
        'xaxis2.range[0]': -0.6,
        'xaxis2.range[1]': 0.3,
        'yaxis2.range[0]': -0.4,
        'yaxis2.range[1]': 0.7,
      }),
    ).toEqual({start: -0.6, end: 0.7});
    expect(scatterRelayout({'xaxis.range': [-2.5, 1.5]})).toEqual({start: -2.5, end: 1.5});
    expect(scatterRelayout({autosize: true})).toBeNull();
    expect(scatterRelayout({'xaxis.range': [1, 1]})).toBeNull();
  });
  it('persists value zoom separately from channel zoom and rejects malformed domains', () => {
    const saved = {
      ...defaultKvView(),
      tensor: {
        active: true,
        plot: 'scatter',
        position: 32,
        layer: 0,
        head: 0,
        channelRange: {start: 10, end: 20},
        valueRange: {start: -0.8, end: 1.3},
      },
    };
    expect(parseKvView(saved)?.tensor).toEqual(saved.tensor);
    for (const valueRange of [
      {start: 1, end: 1},
      {start: 2, end: 1},
      {start: NaN, end: 1},
      {start: '0', end: 1},
    ])
      expect(parseKvView({...saved, tensor: {...saved.tensor, valueRange}})).toBeNull();
  });
});
