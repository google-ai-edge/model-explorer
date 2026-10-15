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

import {rangeBinCount} from '../kv/kv_panel/kv_range_geometry';
import type {PositionTensorRow} from '../../data/contracts/position_analysis';
import type {KvRangeBin, KvRangeRow, KvRangeMetricSummary} from '../../data/contracts/kv';
import {reduceMetricValues, type HeatmapReduction} from './heatmap_model';

export const TENSOR_METRICS = [
  {id: 'cosine_similarity', label: 'Cosine similarity', source: 'CosSim'},
  {id: 'relative_l2', label: 'Relative L2 (%)', source: 'Relative L2'},
  {id: 'max_abs', label: 'Max abs error', source: 'Max abs error'},
] as const;

export function tensorBins(range: KvRangeBin, capacity: number): KvRangeBin[] {
  const count = rangeBinCount(range, capacity),
    length = range.end - range.start;
  return Array.from({length: count}, (_, i) => ({
    start: range.start + Math.floor((i * length) / count),
    end: range.start + Math.floor(((i + 1) * length) / count),
  }));
}
/** Metric per tensor → Sync Points → Position bins. Missing values stay null. */
export function tensorRangeRows(
  source: readonly PositionTensorRow[],
  layers: readonly number[],
  bins: readonly KvRangeBin[],
  anchor: string,
  kind: string,
): KvRangeRow[] {
  const indexed = new Map<string, PositionTensorRow[]>();
  for (const row of source) {
    if (row.status !== 'ok' || (!['min', 'max', 'mean'].includes(anchor) && row.anchor !== anchor))
      continue;
    const key = `${row.layer}:${row.position}`;
    const group = indexed.get(key) ?? [];
    group.push(row);
    indexed.set(key, group);
  }
  return layers.map((layer) => ({
    layer,
    kind,
    bins: bins.map((bin) => {
      const metrics: Record<string, KvRangeMetricSummary> = {};
      for (const metric of TENSOR_METRICS) {
        const values: {position: number; value: number}[] = [];
        for (let position = bin.start; position < bin.end; position++) {
          const cells = (indexed.get(`${layer}:${position}`) ?? []).map(
            (r) => r.metrics[metric.source]?.value ?? null,
          );
          const value = reduceMetricValues(
            cells,
            ['min', 'max', 'mean'].includes(anchor) ? (anchor as HeatmapReduction) : 'mean',
          );
          if (value !== null) values.push({position, value});
        }
        const min = values.reduce<(typeof values)[number] | null>(
          (a, b) => (!a || b.value < a.value ? b : a),
          null,
        );
        const max = values.reduce<(typeof values)[number] | null>(
          (a, b) => (!a || b.value > a.value ? b : a),
          null,
        );
        metrics[metric.id] = {
          min: min?.value ?? null,
          max: max?.value ?? null,
          mean: reduceMetricValues(
            values.map((v) => v.value),
            'mean',
          ),
          min_position: min?.position ?? null,
          max_position: max?.position ?? null,
          valid_count: values.length,
        };
      }
      metrics['cosine_distance'] = {
        min: null,
        max: null,
        mean: null,
        min_position: null,
        max_position: null,
        valid_count: 0,
      };
      return {...bin, metrics: metrics as KvRangeRow['bins'][number]['metrics']};
    }),
  }));
}
