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

/** POST /api/match-rate. */

import type {RowAddress} from './capture';

/** Rows of one stage by index, each once; without `rows`, every row of the stage. */
export interface MatchRateRequest extends RowAddress {
  rows?: number[];
}
/** Match rate computed across the requested rows of a stage. */
export interface MatchRateResponse extends RowAddress {
  /**
   * The share of the rows that are the same; null when no row is named or one named row's
   * sameness is unknown.
   */
  match_rate: number | null;
}
