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

import type {ComparisonRow, Semantic} from '../../../data/contracts/types';

export interface GraphQuery {
  text: string;
  anchor: string;
  metric: string;
  operator: 'lt' | 'gt';
  threshold: string;
  withMetrics: boolean;
}
export const emptyGraphQuery = (): GraphQuery => ({
  text: '',
  anchor: '',
  metric: 'CosSim',
  operator: 'lt',
  threshold: '',
  withMetrics: false,
});
export function graphMetricValue(
  row: ComparisonRow | undefined,
  name: string,
): number | null {
  const metric = row?.metrics[name];
  return metric?.status === 'ok' &&
    metric.value != null &&
    Number.isFinite(metric.value)
    ? metric.value
    : null;
}
export function graphAnchor(model: Semantic | null, row: ComparisonRow) {
  const layer = model?.layers[row.layer];
  return layer
    ? model?.semantic_graph[layer.def]?.anchors.find((a) => a.id === row.anchor)
    : undefined;
}
export function matchesGraphQuery(
  row: ComparisonRow,
  model: Semantic | null,
  query: GraphQuery,
) {
  const anchor = graphAnchor(model, row);
  if (query.anchor && row.anchor !== query.anchor) return false;
  const text = query.text.trim().toLowerCase();
  if (
    text &&
    ![row.reference, row.target, row.anchor, anchor?.label, anchor?.semantic]
      .join(' ')
      .toLowerCase()
      .includes(text)
  )
    return false;
  const value = graphMetricValue(row, query.metric);
  if (
    query.withMetrics &&
    !Object.keys(row.metrics).some(
      (name) => graphMetricValue(row, name) != null,
    )
  )
    return false;
  if (query.threshold.trim()) {
    const threshold = Number(query.threshold);
    if (!Number.isFinite(threshold) || value == null || !Number.isFinite(value))
      return false;
    if (query.operator === 'lt' ? value >= threshold : value <= threshold)
      return false;
  }
  return true;
}
export function graphQueryCount(query: GraphQuery) {
  return [
    query.text.trim(),
    query.anchor,
    query.threshold.trim(),
    query.withMetrics,
  ].filter(Boolean).length;
}
