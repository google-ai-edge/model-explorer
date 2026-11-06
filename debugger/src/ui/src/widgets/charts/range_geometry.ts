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

import type {KvRangeBin} from '../../server/contracts/kv';
import {RangeSelection} from './range_selection';

export const RANGE_MAX_BINS = 1024;
export const RANGE_MIN_COLUMN_WIDTH = 8;
export const RANGE_LEFT = 64;
export const RANGE_RIGHT = 16;

/** Width is the chart's content box, excluding its outer padding. */
export function rangeCapacity(width: number): number {
  return Math.max(
    1,
    Math.min(
      RANGE_MAX_BINS,
      Math.floor((width - RANGE_LEFT - RANGE_RIGHT) / RANGE_MIN_COLUMN_WIDTH),
    ),
  );
}

/** Integer partitioning must not create a narrower remainder column. */
export function rangeBinCount(range: KvRangeBin | null, capacity: number): number {
  if (!range) {
    return 1;
  }
  const count = Math.max(1, range.end - range.start);
  const minimumSpan = Math.ceil(count / Math.max(1, capacity));
  return Math.max(1, Math.floor(count / minimumSpan));
}

export function rangeOverlappingBins(
  bins: readonly KvRangeBin[],
  selection: KvRangeBin | null,
): [number, number] | null {
  if (!selection) {
    return null;
  }
  let first = -1;
  let last = -1;
  for (let i = 0; i < bins.length; i++) {
    if (bins[i].end > selection.start && bins[i].start < selection.end) {
      if (first < 0) {
        first = i;
      }
      last = i;
    }
  }
  return first < 0 ? null : [first, last];
}

export function rangeKeyboardSelection(
  bins: readonly KvRangeBin[],
  layers: readonly number[],
  selection: RangeSelection | null,
  key: string,
  lineChartLayer: number | null = null,
): RangeSelection | null {
  if (!bins.length) {
    return null;
  }
  const overlap = rangeOverlappingBins(bins, selection);
  let index = overlap?.[0] ?? 0;
  let layer: number | null = selection?.layer ?? lineChartLayer ?? layers[0] ?? null;
  if (selection?.layer === null) {
    layer = null;
  }
  if (lineChartLayer !== null && layer !== null) {
    layer = lineChartLayer;
  }
  if (key === 'ArrowLeft') {
    index = Math.max(0, index - 1);
  } else if (key === 'ArrowRight') {
    index = overlap ? Math.min(bins.length - 1, index + 1) : 0;
  } else if (key === 'Home') {
    index = 0;
  } else if (key === 'End') {
    index = bins.length - 1;
  } else if ((key === 'ArrowUp' || key === 'ArrowDown') && lineChartLayer === null && layers.length) {
    const current = Math.max(0, layers.indexOf(layer ?? layers[0]));
    layer =
      layers[Math.max(0, Math.min(layers.length - 1, current + (key === 'ArrowUp' ? -1 : 1)))];
  } else if (key.toLowerCase() === 'c') {
    layer = null;
  } else if (key !== 'Enter') {
    return null;
  }
  return {...bins[index], layer};
}
