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

export type TokenMetricKey =
  | 'kl'
  | 'js'
  | 'relative_l2'
  | 'cosine_distance'
  | 'norm_ratio'
  | 'max_abs_error';
export interface TokenMetric {
  value: number | null;
  reason: string | null;
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
  reason?: string | null;
}
export interface SamplingParameters {
  temperature: number | null;
  top_k: number | null;
  top_p: number | null;
  seed: number | null;
}
export interface SamplingDistributionSide extends TokenDistributionSide {
  /** Nonzero candidates in the post-sampler distribution, not displayed row count. */
  candidate_count: number | null;
  /** Current native support reconstructs only the validated deterministic greedy policy. */
  basis: 'reconstructed_greedy' | null;
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
  context: 'same' | 'different' | null;
  vocabulary_compatible: boolean;
  /** Rows contain the complete nonzero support of all available sides. */
  rows_complete: boolean;
  reason: string | null;
  ref: SamplingDistributionSide;
  target: SamplingDistributionSide;
  rows: SamplingDistributionRow[];
}
/** How the per-token evidence was obtained: a recorded trace/eager capture, or bindings inferred
 *  from the native dump under greedy decoding (`inferred_greedy_argmax`). */
export type TokenEvidenceBasis = 'recorded' | 'inferred_greedy_argmax' | string;
export interface TokenAnalysisPair {
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
    reason: string | null;
    /** `different` once the two generations have diverged: rows keep each side's own candidates. */
    context?: 'same' | 'different' | null;
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
export interface TokenAnalysisResponse {
  turn: number;
  pairs: TokenAnalysisPair[];
}
