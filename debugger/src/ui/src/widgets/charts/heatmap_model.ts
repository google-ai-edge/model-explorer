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

import type {MetricDescription} from '../../server/contracts/metrics';

/** Heatmaps always compare global Position × Layer. Other dimensions are selected, or reduced
 * after metric calculation. */
export type HeatmapReduction = 'min' | 'max' | 'mean';
/** Position bins show their worst value by default: the least of a metric for which larger is
 *  better, as the catalog says, else the largest. */
export function defaultPositionReduction(
  description: MetricDescription | undefined,
): HeatmapReduction {
  return description?.larger_is_better ? 'min' : 'max';
}
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
}
export const HEATMAP_DEFINITIONS: Record<HeatmapDefinition['id'], HeatmapDefinition> = {
  activations: {
    id: 'activations',
    dimensions: [POSITION_DIMENSION],
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
  },
  kv: {
    id: 'kv',
    dimensions: [
      POSITION_DIMENSION,
      {id: 'head', label: 'Heads', modes: ['select', 'min', 'max', 'mean'], defaultMode: 'min'},
    ],
  },
};
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
/** Coalesce adjacent unavailable cells without covering valid comparison results. */
export function heatmapMissingRanges(
  data: HeatmapData,
): {plane: number; layer: number; start: number; end: number}[] {
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
        if (current && current.end === position.start) {
          current.end = position.end;
        } else {
          current = {plane: p, layer: data.layers[y], ...position};
          ranges.push(current);
        }
      });
    }),
  );
  return ranges;
}
