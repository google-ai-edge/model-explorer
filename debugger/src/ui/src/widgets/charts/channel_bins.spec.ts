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
  channelBins,
  channelBinSize,
  channelRepresentative,
  reducedRepresentative,
} from './channel_bins';
import type {KvHeadChannel} from '../../server/contracts/kv';
const rows = Array.from(
  {length: 4096},
  (_, channel) =>
    ({
      channel,
      ref: channel,
      target: channel * 2,
      delta: channel,
      abs_delta: channel,
      status: 'ok',
    }) satisfies KvHeadChannel,
);
describe('head dimension range binning', () => {
  it('bounds dense chart size and restores exact values when zoomed', () => {
    const full = {start: 0, end: 4096};
    expect(channelBinSize(full, 128)).toBe(32);
    const bins = channelBins(rows, full, 128, 'mean');
    expect(bins).toHaveLength(128);
    expect(bins[0]).toEqual({start: 0, end: 32, ref: 15.5, target: 31});
    expect(bins.at(-1)?.end).toBe(4096);
    expect(channelBins(rows, {start: 1025, end: 1028}, 128, 'mean')).toEqual([
      {start: 1025, end: 1026, ref: 1025, target: 2050},
      {start: 1026, end: 1027, ref: 1026, target: 2052},
      {start: 1027, end: 1028, ref: 1027, target: 2054},
    ]);
  });
  it('reduces runtimes independently and preserves missing/nonfinite data as gaps', () => {
    expect(channelBins(rows, {start: 0, end: 3}, 1, 'min')[0]).toMatchObject({ref: 0, target: 0});
    expect(channelBins(rows, {start: 0, end: 3}, 1, 'max')[0]).toMatchObject({ref: 2, target: 4});
    expect(
      channelBins([{...rows[0], ref: null}, rows[1]], {start: 0, end: 2}, 1, 'mean')[0],
    ).toMatchObject({ref: null, target: 1});
    expect(channelBins([rows[0]], {start: 0, end: 2}, 1, 'min')[0]).toMatchObject({
      ref: null,
      target: null,
    });
    expect(
      channelBins([{...rows[0], target: Infinity}], {start: 0, end: 1}, 1, 'mean')[0].target,
    ).toBeNull();
  });
  it('selects the clicked runtime extreme, breaks ties by index, and uses the first Mean element', () => {
    const values = [
      {...rows[2], ref: -5, target: 9},
      {...rows[1], ref: -5, target: -3},
      {...rows[0], ref: 8, target: 4},
    ];
    const bin = {start: 0, end: 3};
    expect(channelRepresentative(values, bin, 'min', 'ref')).toBe(1);
    expect(channelRepresentative(values, bin, 'min', 'target')).toBe(1);
    expect(channelRepresentative(values, bin, 'max', 'ref')).toBe(0);
    expect(channelRepresentative(values, bin, 'max', 'target')).toBe(2);
    expect(channelRepresentative(values, bin, 'mean', 'ref')).toBe(0);
    expect(channelRepresentative(values, bin, 'mean', 'target')).toBe(0);
  });
  it('falls back to Target only for side-free axis selection and never selects an incomplete bin', () => {
    const values = [
      {...rows[0], ref: null, target: 5},
      {...rows[1], target: -2},
    ];
    const bin = {start: 0, end: 2};
    expect(channelRepresentative(values, bin, 'min', 'ref')).toBeNull();
    expect(channelRepresentative(values, bin, 'min')).toBe(1);
    expect(channelRepresentative(rows, bin, 'max')).toBe(1);
    expect(channelRepresentative(values.slice(1), bin, 'min')).toBeNull();
    expect(
      channelRepresentative([{...rows[0], ref: Infinity, target: NaN}], {start: 0, end: 1}, 'mean'),
    ).toBeNull();
  });
  it("chooses a reduced bin's element as channelRepresentative chooses it", () => {
    const bin = {start: 4, end: 8, ref: 1, target: 2, refIndex: 5, targetIndex: 7};
    expect(reducedRepresentative(bin, 'ref')).toBe(5);
    expect(reducedRepresentative(bin, 'target')).toBe(7);
    expect(reducedRepresentative(bin)).toBe(5);
    expect(reducedRepresentative({...bin, ref: null, refIndex: null})).toBe(7);
    expect(reducedRepresentative({...bin, ref: null, refIndex: null}, 'ref')).toBeNull();
  });
});
