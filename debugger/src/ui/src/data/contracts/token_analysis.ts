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
export interface TokenProbability {
  probability: number;
  rank: number;
  logit: number | string;
  selected: boolean;
}
export interface TokenDistributionSide {
  entropy?: number;
  selected_id?: number | null;
  selected_rank?: number | null;
  /** Probability of the selected token, and of the most likely minus the second most likely
   *  token: properties of one side's own distribution, valid in any context. */
  selected_probability?: number | null;
  margin?: number | null;
  size?: number;
  source_forward_id?: number;
  reason?: string | null;
}
/** How the per-token evidence was obtained: a recorded trace/eager capture, or bindings inferred
 *  from the native dump under greedy decoding (`inferred_greedy_argmax`). */
export type TokenEvidenceBasis = 'recorded' | 'inferred_greedy_argmax' | string;
export interface TokenAnalysisPair {
  ref_step: number | null;
  target_step: number | null;
  /** Absent for older captures; a per-side object when the two sides differ. */
  basis?:
    | TokenEvidenceBasis
    | Partial<Record<'ref' | 'target', TokenEvidenceBasis>>
    | null;
  metrics: Record<TokenMetricKey, TokenMetric>;
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
