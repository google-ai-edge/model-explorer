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

import type {
  KvRangeBin,
  KvRangeMetricSummary,
} from '../../../data/contracts/kv';
import {KvRangeSelection} from './kv_range_types';

export const KV_RANGE_MAX_BINS = 1024;
export const KV_RANGE_MIN_COLUMN_WIDTH = 4;
export const KV_RANGE_MAX_EXACT_COLUMN_WIDTH = 24;
export const KV_RANGE_ROW_HEIGHT = 24;
export const KV_RANGE_LEFT = 64;
export const KV_RANGE_RIGHT = 16;
export const KV_RANGE_TOP = 36;
export const KV_RANGE_DRAG_THRESHOLD = 5;

export interface KvTrendPoint {
  position: number;
  value: number;
  bin: number;
}
export interface KvRangeBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Width is the chart's content box, excluding its outer padding. */
export function rangeCapacity(width: number): number {
  return Math.max(
    1,
    Math.min(
      KV_RANGE_MAX_BINS,
      Math.floor(
        (width - KV_RANGE_LEFT - KV_RANGE_RIGHT) / KV_RANGE_MIN_COLUMN_WIDTH,
      ),
    ),
  );
}

/** Coarse intervals stay dense; exact positions grow only when spare horizontal space exists. */
export function rangeColumnWidth(
  width: number,
  bins: readonly KvRangeBin[],
  visibleRange: KvRangeBin | null,
): number {
  const exact =
    visibleRange !== null &&
    visibleRange.end - visibleRange.start === bins.length &&
    bins.length > 0 &&
    bins.every(
      (bin, index) =>
        bin.start === visibleRange.start + index && bin.end === bin.start + 1,
    );
  if (!exact) return KV_RANGE_MIN_COLUMN_WIDTH;
  return Math.max(
    KV_RANGE_MIN_COLUMN_WIDTH,
    Math.min(
      KV_RANGE_MAX_EXACT_COLUMN_WIDTH,
      (width - KV_RANGE_LEFT - KV_RANGE_RIGHT) / bins.length,
    ),
  );
}

/** A half-pixel inset keeps a visible stroke even on the narrowest data column. */
export function rangeSelectionFrame(box: KvRangeBox): KvRangeBox {
  return {
    x: box.x + 0.5,
    y: box.y + 0.5,
    width: Math.max(1, box.width - 1),
    height: Math.max(1, box.height - 1),
  };
}

/** Bins are ordered, non-overlapping, half-open logical position intervals. */
export function rangeBinAtPosition(
  bins: readonly KvRangeBin[],
  position: number,
): number {
  let low = 0,
    high = bins.length - 1;
  while (low <= high) {
    const middle = (low + high) >>> 1,
      bin = bins[middle];
    if (position < bin.start) high = middle - 1;
    else if (position >= bin.end) low = middle + 1;
    else return middle;
  }
  return -1;
}

export function rangeBinAtX(
  bins: readonly KvRangeBin[],
  x: number,
  left: number,
  width: number,
  uniform: boolean,
  clamp = false,
): number {
  if (!bins.length || !Number.isFinite(x) || width <= 0) return -1;
  if (!clamp && (x < left || x > left + width)) return -1;
  const fraction = Math.max(0, Math.min(1, (x - left) / width));
  if (uniform)
    return Math.min(bins.length - 1, Math.floor(fraction * bins.length));
  const start = bins[0].start,
    end = bins[bins.length - 1].end;
  if (fraction === 1) return bins.length - 1;
  return rangeBinAtPosition(bins, start + fraction * (end - start));
}

export function rangeFromBins(
  bins: readonly KvRangeBin[],
  first: number,
  last: number,
): KvRangeBin | null {
  if (
    !bins.length ||
    first < 0 ||
    last < 0 ||
    first >= bins.length ||
    last >= bins.length
  )
    return null;
  return {
    start: bins[Math.min(first, last)].start,
    end: bins[Math.max(first, last)].end,
  };
}

/** Continuous position mapping lets a single coarse bucket be brushed into a smaller range. */
export function rangePositionAtX(
  bins: readonly KvRangeBin[],
  x: number,
  left: number,
  width: number,
  uniform: boolean,
): number | null {
  if (!bins.length || !Number.isFinite(x) || width <= 0) return null;
  const fraction = Math.max(0, Math.min(1, (x - left) / width));
  if (!uniform)
    return (
      bins[0].start + fraction * (bins[bins.length - 1].end - bins[0].start)
    );
  if (fraction === 1) return bins[bins.length - 1].end;
  const offset = fraction * bins.length,
    index = Math.floor(offset),
    bin = bins[index];
  return bin.start + (offset - index) * (bin.end - bin.start);
}

export function rangeXAtPosition(
  bins: readonly KvRangeBin[],
  position: number,
  left: number,
  width: number,
  uniform: boolean,
): number {
  if (!bins.length) return left;
  const start = bins[0].start,
    end = bins[bins.length - 1].end;
  if (position <= start) return left;
  if (position >= end) return left + width;
  if (!uniform)
    return left + ((position - start) / Math.max(1, end - start)) * width;
  const index = rangeBinAtPosition(bins, position);
  if (index < 0) {
    const next = bins.findIndex((bin) => bin.start > position);
    return left + ((next < 0 ? bins.length : next) / bins.length) * width;
  }
  const bin = bins[index];
  return (
    left +
    ((index + (position - bin.start) / (bin.end - bin.start)) / bins.length) *
      width
  );
}

export function rangeFromBrush(
  bins: readonly KvRangeBin[],
  firstX: number,
  lastX: number,
  left: number,
  width: number,
  uniform: boolean,
): KvRangeBin | null {
  const first = rangePositionAtX(bins, firstX, left, width, uniform),
    last = rangePositionAtX(bins, lastX, left, width, uniform);
  if (first === null || last === null) return null;
  const extentStart = bins[0].start,
    extentEnd = bins[bins.length - 1].end;
  const start = Math.max(
    extentStart,
    Math.min(extentEnd - 1, Math.floor(Math.min(first, last))),
  );
  return {
    start,
    end: Math.min(
      extentEnd,
      Math.max(start + 1, Math.ceil(Math.max(first, last))),
    ),
  };
}

export function rangeKeyboardBrush(
  bins: readonly KvRangeBin[],
  selection: KvRangeBin | null,
  current: {anchorX: number; focusX: number} | null,
  direction: -1 | 1,
  left: number,
  width: number,
  uniform: boolean,
): {anchorX: number; focusX: number; range: KvRangeBin} | null {
  if (!bins.length || width <= 0) return null;
  const selected = selection ?? bins[0];
  const anchorX =
    current?.anchorX ??
    rangeXAtPosition(bins, selected.start, left, width, uniform);
  const previous =
    current?.focusX ??
    rangeXAtPosition(bins, selected.end, left, width, uniform);
  // At very low capacity a quarter-bucket step keeps keyboard zoom usable inside one coarse bucket.
  const focusX = Math.max(
    left,
    Math.min(
      left + width,
      previous + (direction * width) / Math.max(4, bins.length),
    ),
  );
  const range = rangeFromBrush(bins, anchorX, focusX, left, width, uniform);
  return range ? {anchorX, focusX, range} : null;
}

export function rangeDragStarted(
  firstClientX: number,
  currentClientX: number,
): boolean {
  return Math.abs(currentClientX - firstClientX) >= KV_RANGE_DRAG_THRESHOLD;
}

export function rangeOverlappingBins(
  bins: readonly KvRangeBin[],
  selection: KvRangeBin | null,
): [number, number] | null {
  if (!selection) return null;
  let first = -1,
    last = -1;
  for (let i = 0; i < bins.length; i++)
    if (bins[i].end > selection.start && bins[i].start < selection.end) {
      if (first < 0) first = i;
      last = i;
    }
  return first < 0 ? null : [first, last];
}

/** Each bucket contributes its actual extrema in position order, never min-then-max order. */
export function rangeTrendSegments(
  bins: readonly KvRangeBin[],
  summaries: readonly (KvRangeMetricSummary | null)[],
): KvTrendPoint[][] {
  const segments: KvTrendPoint[][] = [];
  let current: KvTrendPoint[] = [];
  const finish = () => {
    if (current.length) segments.push(current);
    current = [];
  };
  for (let i = 0; i < bins.length; i++) {
    const bin = bins[i],
      summary = summaries[i];
    if (!summary || summary.valid_count <= 0) {
      finish();
      continue;
    }
    const candidates = [
      {position: summary.min_position, value: summary.min},
      {position: summary.max_position, value: summary.max},
    ]
      .filter(
        (point): point is {position: number; value: number} =>
          point.position !== null &&
          Number.isInteger(point.position) &&
          point.position >= bin.start &&
          point.position < bin.end &&
          point.value !== null &&
          Number.isFinite(point.value),
      )
      .sort((a, b) => a.position - b.position);
    if (!candidates.length) {
      finish();
      continue;
    }
    // A sparse gap in supplied intervals is also a discontinuity.
    if (i > 0 && bins[i - 1].end !== bin.start) finish();
    for (const point of candidates) {
      const previous = current[current.length - 1];
      if (
        previous?.position === point.position &&
        previous.value === point.value
      )
        continue;
      current.push({...point, bin: i});
    }
  }
  finish();
  return segments;
}

export function rangeKeyboardSelection(
  bins: readonly KvRangeBin[],
  layers: readonly number[],
  selection: KvRangeSelection | null,
  key: string,
  trendLayer: number | null = null,
): KvRangeSelection | null {
  if (!bins.length) return null;
  const overlap = rangeOverlappingBins(bins, selection);
  let index = overlap?.[0] ?? 0;
  let layer: number | null =
    selection?.layer ?? trendLayer ?? layers[0] ?? null;
  if (selection?.layer === null) layer = null;
  if (trendLayer !== null && layer !== null) layer = trendLayer;
  if (key === 'ArrowLeft') index = Math.max(0, index - 1);
  else if (key === 'ArrowRight')
    index = overlap ? Math.min(bins.length - 1, index + 1) : 0;
  else if (key === 'Home') index = 0;
  else if (key === 'End') index = bins.length - 1;
  else if (
    (key === 'ArrowUp' || key === 'ArrowDown') &&
    trendLayer === null &&
    layers.length
  ) {
    const current = Math.max(0, layers.indexOf(layer ?? layers[0]));
    layer =
      layers[
        Math.max(
          0,
          Math.min(layers.length - 1, current + (key === 'ArrowUp' ? -1 : 1)),
        )
      ];
  } else if (key.toLowerCase() === 'c') layer = null;
  else if (key !== 'Enter') return null;
  return {...bins[index], layer};
}
