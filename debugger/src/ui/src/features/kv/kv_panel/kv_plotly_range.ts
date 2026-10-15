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

import type {KvRangeBin, KvRangeRow} from '../../../data/contracts/kv';

/** Plotly uses continuous cell edges; captured Position ranges are integer, half-open. */
export function plotlyPositionRange(
  event: Record<string, unknown>,
  full: KvRangeBin,
): KvRangeBin | null {
  if (event['xaxis.autorange'] === true) return {...full};
  const pair = event['xaxis.range'];
  const a = Array.isArray(pair) ? Number(pair[0]) : Number(event['xaxis.range[0]']);
  const b = Array.isArray(pair) ? Number(pair[1]) : Number(event['xaxis.range[1]']);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  const start = Math.max(full.start, Math.min(full.end - 1, Math.round(Math.min(a, b) + 0.5)));
  return {start, end: Math.max(start + 1, Math.min(full.end, Math.round(Math.max(a, b) + 0.5)))};
}

/** Keep bounded full-observation context while replacing the viewport with finer bins.
 * A clipped overview cell retains its original evidence interval for hover/selection.
 */
export function plotlyColumns(
  bins: readonly KvRangeBin[],
  overview: readonly KvRangeBin[],
  range: KvRangeBin,
) {
  return [
    ...overview
      .filter((b) => b.start < range.start)
      .map((b) => ({...b, end: Math.min(b.end, range.start), source: b, overview: true})),
    ...bins.map((b) => ({...b, source: b, overview: false})),
    ...overview
      .filter((b) => b.end > range.end)
      .map((b) => ({...b, start: Math.max(b.start, range.end), source: b, overview: true})),
  ];
}
export function plotlyCellIndex(rows: readonly KvRangeRow[]) {
  return new Map(
    rows.map((row) => [
      row.layer,
      new Map(row.bins.map((bin) => [`${bin.start}:${bin.end}`, bin])),
    ]),
  );
}

/** Hit-test the whole Position axis, not only the sparsely rendered tick labels.
 * Preserve recorded conversation coordinates and the full evidence interval. */
export function plotlyColumnSelection(bins: readonly KvRangeBin[], position: number) {
  const bin = bins.find((bin) => position >= bin.start - 0.5 && position < bin.end - 0.5);
  return bin ? {...bin, layer: null} : null;
}

export interface HeatmapHighlightBox {
  left: number;
  top: number;
  width: number;
  height: number;
}
/** Project one selection to every K/V plane; preserve all unselected chart data. */
export function heatmapHighlightBoxes(
  x: {_offset: number; _length: number; d2p: (value: number) => number},
  axes: readonly {_offset: number; _length: number; d2p: (value: number) => number}[],
  layers: readonly number[],
  selection: {start: number; end: number; layer: number | null},
): HeatmapHighlightBox[] {
  const left = Math.max(0, x.d2p(selection.start - 0.5));
  const right = Math.min(x._length, x.d2p(selection.end - 0.5));
  const row = selection.layer === null ? null : layers.indexOf(selection.layer);
  if (right <= left || row === -1) return [];
  return axes
    .map((axis) => {
      const top =
        row === null ? 0 : Math.max(0, Math.min(axis.d2p(row - 0.5), axis.d2p(row + 0.5)));
      const bottom =
        row === null
          ? axis._length
          : Math.min(axis._length, Math.max(axis.d2p(row - 0.5), axis.d2p(row + 0.5)));
      return {
        left: x._offset + left,
        top: axis._offset + top,
        width: right - left,
        height: bottom - top,
      };
    })
    .filter((box) => box.height > 0);
}
