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

import type {FinalLogitsMetrics, TokenDistributionSide} from './token_analysis';

export type FinalLogitsTransform = 'raw' | 'centered';
export type FinalLogitsOrder = 'index' | 'ref' | 'target' | 'delta' | 'absolute_delta';
export type FinalLogitsDirection = 'ascending' | 'descending';

export interface FinalLogitsRequest {
  turn: number;
  ref_step: number | null;
  target_step: number | null;
  transform: FinalLogitsTransform;
  offset: number;
  limit: number;
  order: FinalLogitsOrder;
  /** Explicit sorting keeps non-finite/missing values last and breaks value ties by index. */
  direction?: FinalLogitsDirection;
}

export interface FinalLogitsRow {
  /** Vocabulary index; sorting never changes this identity. */
  index: number;
  /** Null when the shared vocabulary label is unavailable or not established. */
  label: string | null;
  /** Non-finite captured values use 'NaN', 'Infinity', or '-Infinity'. */
  ref: number | string | null;
  target: number | string | null;
  delta: number | string | null;
  absolute_delta: number | string | null;
}

export interface FinalLogitsResponse {
  turn: number;
  ref_step: number | null;
  target_step: number | null;
  transform: FinalLogitsTransform;
  compatible: boolean;
  reason: string | null;
  total: number;
  /** Row offset after full-vector sorting; only legacy absolute_delta requests may fall back. */
  offset: number;
  limit: number;
  rows: FinalLogitsRow[];
  ref: TokenDistributionSide;
  target: TokenDistributionSide;
  metrics: FinalLogitsMetrics;
}

export type FinalLogitsVectorRequest = Pick<
  FinalLogitsRequest,
  'turn' | 'ref_step' | 'target_step' | 'transform'
>;

export interface FinalLogitsVectorResponse
  extends Omit<FinalLogitsResponse, 'offset' | 'limit' | 'rows'> {
  /** All captured elements are represented; oversized vectors fail instead of truncating. */
  complete: true;
  /** Array index is the vocabulary index. Null means the entire side/transform is unavailable. */
  values: {
    ref: Array<number | string | null> | null;
    target: Array<number | string | null> | null;
  };
}
