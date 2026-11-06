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

/** POST /api/token-analysis. */

import type {ComparisonAlignment, ComparisonStage, RowAddress} from './capture';
import type {KvRangeBin, KvRangeMetricSummary, LogicalTensor} from './kv';
import type {Reason} from './metrics';

/** Rows of a turn's Decode by index, and with `first_divergence` the first divergence's pair
 *  (its row, filled in by step); at most 128 in all. */
export interface TokenAnalysisRequest extends RowAddress {
  rows: number[];
  first_divergence?: boolean;
}

/** The metrics of a token pair, by the names the server answers with; GET /api/metrics describes
 *  each one. */
export const TOKEN_METRIC_KEYS = [
  'kl',
  'js',
  'relative_l2',
  'cosine_similarity',
  'cosine_distance',
  'norm_ratio',
  'max_abs',
] as const;
export type TokenMetricKey = (typeof TOKEN_METRIC_KEYS)[number];
export interface TokenMetric {
  value: number | null;
  /** Why the value is missing; none when it is there. */
  reasons: Reason[];
}
export interface FinalLogitsMetrics {
  rmse: TokenMetric;
  max_abs: TokenMetric;
  max_abs_index: number | null;
}
export interface TokenProbability {
  probability: number;
  rank: number;
  logit: number | string;
  selected: boolean;
}
export interface TokenLogitsMetadata {
  /** Shape and dtype of the loaded logits source, before float64 metric calculation. */
  source_shape: number[];
  source_dtype: string;
  /** Actual vocabulary-vector slice used for comparison and softmax; never a flattened sequence. */
  comparison_shape: number[];
  /** Captured vocabulary identity when supplied by the source forward. */
  vocab_identity?: string;
}
export interface TokenDistributionSide {
  entropy?: number | null;
  selected_id?: number | null;
  selected_rank?: number | null;
  /** Probability of the selected token, and of the most likely minus the second most likely
   *  token: properties of one side's own distribution, valid in any context. */
  selected_probability?: number | null;
  margin?: number | null;
  size?: number;
  source_forward_id?: number;
  /** Optional for compatibility with older responses and unavailable distributions. */
  logits_metadata?: TokenLogitsMetadata;
  /** This side's own reasons, of its side. */
  reasons?: Reason[];
}
export interface SamplingParameters {
  temperature: number | null;
  top_k: number | null;
  top_p: number | null;
  seed: number | null;
}
export interface SamplingDistributionSide extends TokenDistributionSide {
  /** The vocabulary ids the distribution spans, not the displayed row count. */
  candidate_count: number | null;
  /** The softmax of the captured full-vocabulary logits divided by `temperature`, before top-k,
   *  top-p or any other constraint; null where the side has no logits (its `reasons` say why). */
  basis: 'logits_softmax' | null;
  /** The temperature the logits were divided by: the sampler's when positive, else 1. */
  temperature: number | null;
  sampler: SamplingParameters | null;
}
export interface SamplingProbability {
  probability: number;
  /** Zero-probability candidates have no rank. */
  rank: number | null;
  selected: boolean;
}
export interface SamplingDistributionRow {
  id: number;
  label: string | null;
  ref: SamplingProbability | null;
  target: SamplingProbability | null;
}
export interface SamplingDistribution {
  context: TokenContext;
  vocabulary_compatible: boolean;
  /** Whether the rows hold every id of nonzero probability; the server lists the most likely
   *  ids of either side, so a missing id is not zero. */
  rows_complete: boolean;
  reasons: Reason[];
  ref: SamplingDistributionSide;
  target: SamplingDistributionSide;
  rows: SamplingDistributionRow[];
}
/** Whether both sides fed the same tokens before the pair; `unknown` when a token the Runner did not
 *  name could make them either; null when the pair is one-sided. Logits compare in every context. */
export type TokenContext = 'same' | 'different' | 'unknown' | null;
/** How the per-token evidence was obtained: a recorded trace/eager capture, or bindings inferred
 *  from the native dump under greedy decoding (`inferred_greedy_argmax`). */
export type TokenEvidenceBasis = 'recorded' | 'inferred_greedy_argmax' | string;
export interface TokenAnalysisPair {
  /** The row this pair is, as the request named it: final logits name it the same way. */
  stage: ComparisonStage;
  alignment: ComparisonAlignment;
  row: number;
  first_divergence: boolean;
  /** The steps the server found for the row. */
  ref_step: number | null;
  target_step: number | null;
  /** Absent for older captures; a per-side object when the two sides differ. */
  basis?: TokenEvidenceBasis | Partial<Record<'ref' | 'target', TokenEvidenceBasis>> | null;
  metrics: Record<TokenMetricKey, TokenMetric>;
  /** Optional for older responses; each transform compares full vocabulary logits. */
  final_logits?: {
    raw: FinalLogitsMetrics;
    centered: FinalLogitsMetrics;
  };
  /** Post-sampler evidence; never fall back to the T=1 logits distribution below. */
  sampling_distribution?: SamplingDistribution;
  distribution: {
    compatible: boolean;
    reasons: Reason[];
    /** `different` once the two generations have diverged. */
    context?: TokenContext;
    /** Why the compared logits read different tokens on the two sides, in the server's words. */
    notes?: Reason[];
    ref: TokenDistributionSide;
    target: TokenDistributionSide;
    rows: {
      id: number;
      label: string | null;
      ref: TokenProbability | null;
      target: TokenProbability | null;
      delta: number | null;
    }[];
  };
}
export interface TokenAnalysisResponse extends RowAddress {
  pairs: TokenAnalysisPair[];
}

/** POST /api/token-range: one metric of a turn's Decode rows, binned by column. Its columns are the
 *  KV analyses': row k of the stage is its first column + k. */
export interface TokenRangeRequest extends RowAddress {
  analysis: 'final_logits' | 'sampling';
  /** The catalog's id of one of the analysis' metrics. */
  metric: string;
  start: number;
  end: number;
  bins: number;
}
/** A bin's columns and the statistic of its values. */
export type TokenRangeCell = KvRangeBin & KvRangeMetricSummary;
/** Per side, what the rows of the range compared: for Final Logits, how many have logits and
 *  the distinct sizes, dtypes and identities they report; for Sampling, the distinct vocabulary
 *  sizes and sampler settings, null where a row reports none. */
export interface TokenRangeSideMetadata {
  count?: number;
  vocabulary_sizes: (number | null)[];
  dtypes?: (string | null)[];
  vocabulary_identities?: (string | null)[];
  sampler?: Record<'temperature' | 'top_k' | 'top_p' | 'seed', (number | null)[]>;
}
export interface TokenRangeResponse extends RowAddress {
  analysis: 'final_logits' | 'sampling';
  metric: string;
  start: number;
  end: number;
  bins: KvRangeBin[];
  /** Each series' statistics per bin: Final Logits' one of the pair (side null), Sampling's one
   *  per side. */
  rows: {side: 'ref' | 'target' | null; bins: TokenRangeCell[]}[];
  /** The statistics per bin of every series together. */
  all: TokenRangeCell[];
  metadata: {
    ref: TokenRangeSideMetadata;
    target: TokenRangeSideMetadata;
    output: LogicalTensor;
  };
  /** The stage's largest KL, its row and step (the target's, else the ref's); the first
   *  divergence's pair stands for its row. Null without one. */
  peak_kl: {value: number; row: number; step: number} | null;
}
