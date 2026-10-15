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

import type {KvMetric, KvRangeRow} from '../../../data/contracts/kv';
import {
  defaultPositionReduction,
  type HeatmapReduction,
  type HeatmapData,
} from '../../analysis/heatmap_model';
import {plotlyCellIndex, plotlyColumns} from './kv_plotly_range';

/** The range API already computes Metric per Head before reducing Heads. Keep its
 * evidence interval intact even when an overview cell is clipped for display. */
export function kvHeatmapData(
  layers: readonly number[],
  columns: ReturnType<typeof plotlyColumns>,
  tensors: readonly {kind: string; rows: readonly KvRangeRow[]; overview: readonly KvRangeRow[]}[],
  metric: KvMetric,
  _head: string,
  _format: (value: number | null) => string,
  reduction: HeatmapReduction = defaultPositionReduction(metric),
): HeatmapData {
  return {
    layers,
    positions: columns,
    planes: tensors.map((tensor) => {
      const detail = plotlyCellIndex(tensor.rows),
        overview = plotlyCellIndex(tensor.overview);
      return {
        id: tensor.kind,
        label:
          tensor.kind === 'key'
            ? 'Key cache'
            : tensor.kind === 'value'
              ? 'Value cache'
              : tensor.kind,
        cells: layers.map((layer) =>
          columns.map((column) => {
            const bin = (column.overview ? overview : detail)
              .get(layer)
              ?.get(`${column.source.start}:${column.source.end}`);
            const summary = bin?.metrics[metric];
            const value = summary?.valid_count ? (summary[reduction] ?? null) : null;
            return {
              value,
              evidence: {...column.source, layer, kind: tensor.kind},
              hover: `Layer ${layer} · Position ${column.source.start}${column.source.end - column.source.start > 1 ? '–' + (column.source.end - 1) : ''}`,
              matched: !column.overview && (bin?.match_count ?? 0) > 0,
            };
          }),
        ),
      };
    }),
  };
}
