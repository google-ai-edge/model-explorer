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

/** Heatmaps always compare global Position × Layer. Other dimensions are selected,
 * reduced after metric calculation, or split into at most two synchronized planes. */
export type HeatmapReduction = 'min' | 'max' | 'mean';
export const defaultPositionReduction = (metric: string): HeatmapReduction =>
  metric === 'cosine_similarity' ? 'min' : 'max';
const POSITION_DIMENSION: HeatmapDimension = {
  id: 'position',
  label: 'Position bins',
  modes: ['min', 'max', 'mean'],
  defaultMode: 'min',
};
export interface HeatmapDimension {
  id: string;
  label: string;
  modes: readonly ('select' | 'min' | 'max' | 'mean')[];
  defaultMode: 'select' | 'min' | 'max' | 'mean';
}
export interface HeatmapDefinition {
  id: 'activations' | 'intermediate' | 'kv';
  dimensions: readonly HeatmapDimension[];
  split?: {dimension: string; members: readonly string[]};
  requiredDimensions: readonly string[];
  defaultMetric: string;
}
export const HEATMAP_DEFINITIONS: Record<HeatmapDefinition['id'], HeatmapDefinition> = {
  activations: {
    id: 'activations',
    dimensions: [POSITION_DIMENSION],
    requiredDimensions: ['position', 'layer'],
    defaultMetric: 'cosine_similarity',
  },
  intermediate: {
    id: 'intermediate',
    dimensions: [
      POSITION_DIMENSION,
      {
        id: 'syncPoint',
        label: 'Sync Points',
        modes: ['select', 'min', 'max', 'mean'],
        defaultMode: 'select',
      },
    ],
    requiredDimensions: ['position', 'layer', 'syncPoint'],
    defaultMetric: 'cosine_similarity',
  },
  kv: {
    id: 'kv',
    dimensions: [
      POSITION_DIMENSION,
      {id: 'head', label: 'Heads', modes: ['select', 'min', 'max', 'mean'], defaultMode: 'min'},
    ],
    split: {dimension: 'kind', members: ['key', 'value']},
    requiredDimensions: ['position', 'layer', 'head', 'kind'],
    defaultMetric: 'cosine_similarity',
  },
};
export function heatmapAvailable(
  definition: HeatmapDefinition,
  capturedDimensions: readonly string[],
) {
  return definition.requiredDimensions.every((dimension) => capturedDimensions.includes(dimension));
}
export interface HeatmapChoice {
  id: string;
  label: string;
}
export interface HeatmapControl {
  disabled?: boolean;
  id: string;
  label: string;
  value: string;
  choices: readonly HeatmapChoice[];
}
export function heatmapControls(
  definition: HeatmapDefinition,
  values: Record<string, string>,
  members: Record<string, readonly HeatmapChoice[]>,
): HeatmapControl[] {
  return definition.dimensions.map((dimension) => ({
    id: dimension.id,
    label: dimension.label,
    value: values[dimension.id] ?? dimension.defaultMode,
    choices: dimension.modes.flatMap((mode) =>
      mode === 'select'
        ? [...(members[dimension.id] ?? [])]
        : [{id: mode, label: mode[0].toUpperCase() + mode.slice(1)}],
    ),
  }));
}
export interface HeatmapCell {
  value: number | null;
  evidence: unknown;
  hover: string;
  matched?: boolean;
}
export interface HeatmapPlane {
  id: string;
  label: string;
  cells: readonly (readonly HeatmapCell[])[];
}
export interface HeatmapData {
  layers: readonly number[];
  positions: readonly {start: number; end: number}[];
  planes: readonly HeatmapPlane[];
}
/** Input cells are already metric results, never raw tensor elements. */
export function reduceMetricValues(
  values: readonly (number | null)[],
  mode: 'min' | 'max' | 'mean',
) {
  const valid = values.filter((value): value is number => value !== null && Number.isFinite(value));
  if (!valid.length) return null;
  return mode === 'mean'
    ? valid.reduce((sum, value) => sum + value, 0) / valid.length
    : valid.reduce((result, value) =>
        mode === 'min' ? Math.min(result, value) : Math.max(result, value),
      );
}

/** Coalesce adjacent unavailable cells without covering valid comparison results. */
export function heatmapMissingRanges(data: HeatmapData) {
  const ranges: {plane: number; layer: number; start: number; end: number}[] = [];
  data.planes.forEach((plane, p) =>
    plane.cells.forEach((row, y) => {
      let current: (typeof ranges)[number] | null = null;
      row.forEach((cell, x) => {
        const position = data.positions[x];
        if (cell.value !== null && Number.isFinite(cell.value)) {
          current = null;
          return;
        }
        if (current && current.end === position.start) current.end = position.end;
        else {
          current = {plane: p, layer: data.layers[y], ...position};
          ranges.push(current);
        }
      });
    }),
  );
  return ranges;
}
