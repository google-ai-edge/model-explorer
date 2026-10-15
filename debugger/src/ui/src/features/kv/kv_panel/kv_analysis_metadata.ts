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
  KvAnalysisLayer,
  KvTokenStructure,
  KvPositionAlignment,
} from '../../../data/contracts/kv';

export interface KvLogicalMetadata {
  dimensions: string;
  shape: string;
  dtype: string;
  layers: {label: string; value: string}[];
}
const inputAxes = ['Position', 'Head', 'Head dimension'];
const outputAxes = ['Position'];
function shapeText(shape: readonly (number | string)[] | null) {
  return shape ? `[${shape.join(', ')}]` : '—';
}
function combinedShape(shapes: (number[] | null)[], axes: string[]) {
  if (!shapes.some(Boolean)) return '—';
  const dimensions = axes.map((_, i) =>
    shapes.some((shape) => shape?.[i] == null)
      ? '—'
      : uniform(shapes.map((shape) => String(shape![i]))),
  );
  return shapeText([dimensions[0], shapes.length, ...dimensions.slice(1)]);
}
function uniform(values: string[]) {
  return !values.length ? '—' : values.every((value) => value === values[0]) ? values[0] : 'Mixed';
}
function geometry(structure: KvTokenStructure | null | undefined): number[] | null {
  if (!structure || !['ok', 'empty'].includes(structure.status)) return null;
  const {position_count: positions, head_count: heads, channel_count: channels} = structure;
  return positions === null || heads === null || channels === null
    ? null
    : [positions, heads, channels];
}
export interface KvMetadataScope {
  range: {start: number; end: number};
  alignment?: KvPositionAlignment | null;
  heads: readonly number[];
}
function scopedGeometry(
  structure: KvTokenStructure | null | undefined,
  side: 'ref' | 'target',
  scope?: KvMetadataScope,
) {
  const shape = geometry(structure);
  if (!shape || !scope) return shape;
  if (structure?.position_start == null) return null;
  const start = structure.position_start,
    end = start + shape[0];
  let positions: number;
  if (scope.alignment) {
    const alignment = scope.alignment;
    const selected = alignment.pairs
      .slice(
        Math.max(0, scope.range.start - alignment.start),
        Math.max(0, scope.range.end - alignment.start),
      )
      .map((pair) => pair[side === 'ref' ? 0 : 1]);
    positions = new Set(selected.filter((p): p is number => p != null && p >= start && p < end))
      .size;
  } else {
    positions = Math.max(0, Math.min(end, scope.range.end) - Math.max(start, scope.range.start));
  }
  return [
    positions,
    new Set(scope.heads.filter((head) => head >= 0 && head < shape[1])).size,
    shape[2],
  ];
}
/** Logical metadata for the selected analysis inputs and head-reduced outputs. */
export function kvInputMetadata(
  rows: KvAnalysisLayer[],
  side: 'ref' | 'target',
  scope?: KvMetadataScope,
): KvLogicalMetadata {
  const entries = rows.map((row) => {
    const structure = row.token_structure?.[side];
    return {
      row,
      shape: scopedGeometry(structure, side, scope),
      dtype: structure?.dtype ?? '—',
    };
  });
  return {
    dimensions: 'Position × Layer × Head × Head dimension',
    shape: combinedShape(
      entries.map((entry) => entry.shape),
      inputAxes,
    ),
    dtype: uniform(entries.map((entry) => entry.dtype)),
    layers: entries.map((entry) => ({
      label: `L${entry.row.layer}`,
      value: `${shapeText(entry.shape)} · ${entry.dtype}`,
    })),
  };
}
export function kvOutputMetadata(
  rows: KvAnalysisLayer[],
  scope?: KvMetadataScope,
): KvLogicalMetadata {
  const shapes = rows.map((row) => {
    const ref = row.token_structure?.ref,
      target = row.token_structure?.target;
    const a = geometry(ref),
      b = geometry(target);
    if (scope) {
      // Output positions use the analysis coordinate grid, including unavailable aligned pairs.
      // Each selected Position/Layer has one value after selecting or reducing Heads.
      if (!a || !b || a[2] !== b[2] || ref?.batch_count !== target?.batch_count) return null;
      return [Math.max(0, scope.range.end - scope.range.start)];
    }
    // Do not manufacture a rectangular output when the operands' coordinates differ.
    return a &&
      b &&
      a.every((n, i) => n === b[i]) &&
      ref?.position_start === target?.position_start &&
      ref?.batch_count === target?.batch_count
      ? a.slice(0, 1)
      : null;
  });
  return {
    dimensions: 'Position × Layer',
    shape: combinedShape(shapes, outputAxes),
    // The comparison engine accumulates in float64; this is a logical result, not a stored tensor.
    dtype: shapes.some(Boolean) ? 'float64' : '—',
    layers: rows.map((row, i) => ({
      label: `L${row.layer}`,
      value: shapeText(shapes[i]),
    })),
  };
}
