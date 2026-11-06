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

/** GET /api/kv-analysis, /api/kv-range and /api/kv-head. */

import type {ComparisonAlignment, ComparisonStage, StagePositions} from './capture';
import type {Reason} from './metrics';

export type KvMetric = 'relative_l2' | 'max_abs' | 'cosine_distance' | 'cosine_similarity';
export type KvMetrics = Record<Exclude<KvMetric, 'cosine_similarity'>, number | null> & {
  cosine_similarity?: number | null;
};
/** One logical token, per batch, with KV head and channel axes validated by the backend. */
export interface KvTokenStructure {
  status: string;
  reasons?: Reason[];
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
  reasons?: Reason[];
  reference?: string;
  target?: string;
  pair_id?: string;
  shape?: number[];
  head_count: number | null;
  position_start: number | null;
  position_count: number | null;
  metrics?: KvMetrics;
  summary_metrics?: Record<string, {value: number | null; status: string}>;
  comparison_basis?: string;
  canonical_layout?: string[];
  token_structure?: {ref: KvTokenStructure | null; target: KvTokenStructure | null};
}
export interface KvAnalysisContext {
  id: string;
  label: string;
  turn: number;
  phase: string;
  step: number | null;
  moment: string;
  source: 'snapshot' | 'explicit';
  runtime: string | null;
  layers: KvAnalysisLayer[];
  aggregation?: string;
  processed_token_count?: number;
  /** Present when the logical context was derived from a runtime dump (see the data contract). */
  basis?: string;
  context_basis?: string;
}
export interface KvAnalysis {
  contexts: KvAnalysisContext[];
  notice?: string;
}

/** Conversation-global captured positions (across turns, including Prefill).
 * Intervals use an exclusive end. Never add turn lengths to cumulative KV coordinates. */
/** A stage's rows as KV columns: which stage and alignment (all a KV request names; the server
 *  looks the positions up itself), and the positions the server's comparison gives them, which
 *  the panels show. A null position is a one-sided gap. */
export interface KvPositionAlignment extends StagePositions {
  stage: ComparisonStage;
  alignment: ComparisonAlignment;
}
export interface KvRangeBin {
  start: number;
  end: number;
}
/** The server's statistic of values at positions (analysis/statistics): the finite values'
 *  minimum, maximum and mean, the first position of each extremum, and how many were finite. */
export interface KvRangeMetricSummary {
  mean?: number | null;
  min: number | null;
  max: number | null;
  min_position: number | null;
  max_position: number | null;
  valid_count: number;
  /** Intermediate tensors' bins, for a min or max over sync points: the sync point of each
   *  extremum at its position. */
  min_anchor?: string | null;
  max_anchor?: string | null;
  /** Intermediate tensors' bins: why the bin's positions without a value have none. */
  reasons?: Reason[];
}
/** The statistics of one tensor's elements on each side and of their difference, by element. */
export type ElementStatistics = Record<
  'ref' | 'target' | 'delta' | 'abs_delta',
  KvRangeMetricSummary
>;
/** One axis of a shape across layers: their one size, the sizes where they differ, or null where
 *  a layer has no shape. */
export type ShapeAxis = number | number[] | null;
/** A logical tensor, as a details' Metadata lists it: its axes, its shape (null when no layer has
 *  one), its dtype (a list where the layers' differ, null when unknown) and each layer's own. */
export interface LogicalTensor {
  dimensions: string[];
  shape: ShapeAxis[] | null;
  dtype: string | string[] | null;
  layers?: {layer: number; shape: number[] | null; dtype?: string | null}[];
}
/** What a kv-range answer compared: each side's input and the metric's output. */
export interface KvRangeMetadata {
  ref: LogicalTensor;
  target: LogicalTensor;
  output: LogicalTensor;
}
export interface KvRangeCell extends KvRangeBin {
  metrics: Record<Exclude<KvMetric, 'cosine_similarity'>, KvRangeMetricSummary> & {
    cosine_similarity?: KvRangeMetricSummary;
  };
}
export interface KvRangeRow {
  layer: number;
  kind?: string;
  status?: string;
  reasons?: Reason[];
  bins: KvRangeCell[];
}
export interface KvRangeResponse extends KvRangeBin {
  heads?: number[] | null;
  /** The layers the request named; null for every layer. */
  layers?: number[] | null;
  context_id: string;
  /** The alignment the request read through, when it named one. */
  stage?: ComparisonStage;
  alignment?: ComparisonAlignment;
  bins: KvRangeBin[];
  rows: KvRangeRow[];
  metadata: KvRangeMetadata;
}

export type KvHeadMetric =
  | 'cosine_similarity'
  | 'cosine_distance'
  | 'relative_l2'
  | 'norm_ratio'
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
/** One KV head's channels at a column, both sides, of a kv-head answer. */
export interface KvHeadEvidence {
  head: number;
  status: string;
  reasons?: Reason[];
  batch_count: number | null;
  channel_count: number | null;
  shape: number[] | null;
  metrics: Record<KvHeadMetric, number | null>;
  metric_status: Record<KvHeadMetric, string>;
  channels: KvHeadChannel[];
  largest_channel: number | null;
  sources: {ref: KvHeadSource | null; target: KvHeadSource | null};
  comparison_basis?: string;
  calculation_dtype: string;
  delta_definition: 'Target - Reference';
  /** The statistics of the channels' values and differences, by channel. */
  statistics: ElementStatistics;
}
/** GET /api/kv-head: every KV head of a layer and kind at one column, in one answer. */
export interface KvHeadsEvidence {
  selection: {
    turn: number;
    context_id: string;
    layer: number;
    kind: string;
    position: number;
    batch: number | null;
    /** With an alignment: each side's position of the column, and the alignment named. */
    positions?: {ref: number | null; target: number | null};
    stage?: ComparisonStage;
    alignment?: ComparisonAlignment;
  };
  /** Each KV head the layer's stored tensors have, in order. */
  heads: KvHeadEvidence[];
}
