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

export type KvMetric = 'relative_l2' | 'max_abs' | 'cosine_distance';
export type KvMetrics = Record<KvMetric, number | null>;
export type KvTokenMetric =
  | 'cosine_similarity'
  | 'relative_l2'
  | 'rmse'
  | 'max_abs';
export interface KvTokenMetrics {
  position: number;
  status: string;
  metrics: Record<KvTokenMetric, number | null>;
  metric_status: Record<KvTokenMetric, string>;
}
export interface KvSnapshotIdentity {
  id: string | null;
  snapshot_id: number | null;
  runtime: string | null;
  turn: number | null;
  phase: string | null;
  step: number | null;
  forward_id: number | null;
  moment: string | null;
  signature: string | null;
  edge: string | null;
}
export interface KvCell {
  position: number;
  head: number;
  status?: string;
  metrics: KvMetrics;
}
/** One logical token, per batch, with KV head and channel axes validated by the backend. */
export interface KvTokenStructure {
  status: string;
  reason?: string;
  shape: number[] | null;
  dtype: string | null;
  head_count: number | null;
  channel_count: number | null;
  batch_count: number | null;
  position_start: number | null;
  position_count: number | null;
}
export interface KvAnalysisLayer {
  layer: number;
  kind: string;
  status: string;
  reason?: string;
  reference?: string;
  target?: string;
  pair_id?: string;
  shape?: number[];
  head_count: number | null;
  position_start: number | null;
  position_count: number | null;
  cells: KvCell[];
  metrics?: KvMetrics;
  summary_metrics?: Record<string, {value: number | null; status: string}>;
  comparison_basis?: string;
  canonical_layout?: string[];
  token_structure?: {
    ref: KvTokenStructure | null;
    target: KvTokenStructure | null;
  };
  token_metrics?: KvTokenMetrics[];
}
export interface KvAnalysisContext {
  id: string;
  label: string;
  turn: number;
  phase: string;
  step: number | null;
  forward_id: number | null;
  moment: string;
  source: 'snapshot' | 'explicit';
  runtime: string | null;
  layers: KvAnalysisLayer[];
  aggregation?: string;
  observations?: Record<
    string,
    {phase: string; step?: number; forward_id?: number}
  >;
  snapshots?: {ref: KvSnapshotIdentity[]; target: KvSnapshotIdentity[]};
  processed_token_count?: number;
  /** Present when the logical context was derived from a runtime dump (see the data contract). */
  basis?: string;
  context_basis?: string;
}
export interface KvAnalysis {
  contexts: KvAnalysisContext[];
  notice?: string;
}

/** Logical position intervals always use an exclusive end. */
export interface KvRangeBin {
  start: number;
  end: number;
}
export interface KvRangeMetricSummary {
  min: number | null;
  max: number | null;
  min_position: number | null;
  max_position: number | null;
  valid_count: number;
}
export interface KvRangeCell extends KvRangeBin {
  metrics: Record<KvMetric, KvRangeMetricSummary>;
  match_count?: number;
}
export interface KvRangeRow {
  layer: number;
  kind?: string;
  status?: string;
  reason?: string;
  bins: KvRangeCell[];
}
export interface KvRangeResponse extends KvRangeBin {
  context_id: string;
  bins: KvRangeBin[];
  rows: KvRangeRow[];
}
export interface KvRangeDetailMetric {
  value: number | null;
  position: number | null;
  valid_count: number;
  status: string;
}
export interface KvRangeDetailRow {
  layer: number;
  kind: string;
  status: string;
  reason?: string;
  metrics: Record<KvTokenMetric, KvRangeDetailMetric>;
  token_metrics?: KvTokenMetrics[];
}
export interface KvRangeDetails extends KvRangeBin {
  context_id: string;
  rows: KvRangeDetailRow[];
}
export interface KvFindResult {
  layer: number;
  position: number;
  kind: string;
  heads: number[];
  metrics: KvMetrics;
}
export interface KvFindResponse {
  total: number;
  offset: number;
  results: KvFindResult[];
  complete?: boolean;
  unavailable_rows?: unknown[];
}

export type KvHeadMetric =
  | 'cosine_similarity'
  | 'cosine_distance'
  | 'relative_l2'
  | 'mean_abs'
  | 'rmse'
  | 'max_abs';
export interface KvHeadChannel {
  channel: number;
  ref: number | string | null;
  target: number | string | null;
  delta: number | null;
  abs_delta: number | null;
  status: string;
}
export interface KvHeadSource {
  runtime: string | null;
  source_dtype: string;
  source_shape: number[];
  comparison_dtype: string;
  comparison_shape: number[];
  layout: string[];
  axis_order?: number[];
  dequantization?: {scale: number; zero_point: number; formula: string} | null;
  values_mode: 'comparison' | 'stored';
  resource_id?: string;
  pair_id?: string;
}
export interface KvHeadEvidence {
  selection: {
    turn: number;
    context_id: string;
    layer: number;
    kind: string;
    position: number;
    head: number;
    batch: number | null;
  };
  status: string;
  reason?: string;
  batch_count: number | null;
  channel_count: number | null;
  shape: number[] | null;
  metrics: Record<KvHeadMetric, number | null>;
  metric_status: Record<KvHeadMetric, string>;
  channels: KvHeadChannel[];
  largest_channel: number | null;
  sources: {ref: KvHeadSource | null; target: KvHeadSource | null};
  comparison_basis?: string;
  calculation_dtype: 'float64';
  delta_definition: 'Target - Reference';
}
