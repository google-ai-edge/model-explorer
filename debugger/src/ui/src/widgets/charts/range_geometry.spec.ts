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
  RANGE_MAX_BINS,
  RANGE_MIN_COLUMN_WIDTH,
  rangeCapacity,
  rangeKeyboardSelection,
  rangeOverlappingBins,
} from './range_geometry';

describe('range geometry', () => {
  const bins = [
    {start: 131000, end: 131024},
    {start: 131024, end: 131048},
    {start: 131048, end: 131072},
  ];
  const exactBins = (start: number, count: number) =>
    Array.from({length: count}, (_, i) => ({start: start + i, end: start + i + 1}));

  it('dense 8px column geometry and a stable 1024-bin capacity', () => {
    expect(RANGE_MAX_BINS).toBe(1024);
    expect(RANGE_MIN_COLUMN_WIDTH).toBe(8);
    expect(rangeCapacity(0)).toBe(1);
    expect(rangeCapacity(84)).toBe(1);
    expect(rangeCapacity(112)).toBe(4);
    expect(rangeCapacity(1104)).toBe(128);
    expect(rangeCapacity(4176)).toBe(512);
    expect(rangeCapacity(100000)).toBe(1024);
  });

  it('a range finds the bins it overlaps; keyboard selection moves by bin and layer and stops at the ends', () => {
    expect(rangeOverlappingBins(bins, {start: 131023, end: 131025})).toStrictEqual([0, 1]);
    expect(rangeOverlappingBins(bins, {start: 0, end: 1})).toBe(null);
    expect(
      rangeKeyboardSelection(bins, [2, 7], {...bins[0], layer: 2}, 'ArrowRight'),
    ).toStrictEqual({
      ...bins[1],
      layer: 2,
    });
    expect(
      rangeKeyboardSelection(bins, [2, 7], {...bins[2], layer: 2}, 'ArrowRight'),
    ).toStrictEqual({
      ...bins[2],
      layer: 2,
    });
    expect(rangeKeyboardSelection(bins, [2, 7], {...bins[1], layer: 2}, 'ArrowDown')).toStrictEqual(
      {
        ...bins[1],
        layer: 7,
      },
    );
    expect(rangeKeyboardSelection(bins, [2, 7], {...bins[1], layer: 2}, 'c')).toStrictEqual({
      ...bins[1],
      layer: null,
    });
    expect(
      rangeKeyboardSelection(bins, [2, 7], {...bins[1], layer: null}, 'ArrowRight'),
    ).toStrictEqual({
      ...bins[2],
      layer: null,
    });
    expect(rangeKeyboardSelection(bins, [2, 7], {...bins[1], layer: 2}, 'ArrowDown', 2)).toBe(null);
    expect(rangeKeyboardSelection(bins, [2, 7], null, 'End', 7)).toStrictEqual({
      ...bins[2],
      layer: 7,
    });
    expect(rangeKeyboardSelection([], [], null, 'ArrowRight')).toBe(null);
  });

  it('keyboard selection reaches the last exact position of a dense layout', () => {
    const denseExact = exactBins(130048, 1024);
    expect(
      rangeKeyboardSelection(
        denseExact,
        [0, 31],
        {start: 131070, end: 131071, layer: 31},
        'ArrowRight',
      ),
    ).toStrictEqual({start: 131071, end: 131072, layer: 31});
  });
});
