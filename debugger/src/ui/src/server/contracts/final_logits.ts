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

/** POST /api/final-logits, /api/final-logits-summary, /api/final-logits-bins and
 *  /api/final-logits-points. */

import type {RowAddress} from './capture';
import type {ElementStatistics} from './kv';
import type {Reason} from './metrics';
import type {FinalLogitsMetrics, TokenDistributionSide} from './token_analysis';

export type FinalLogitsTransform = 'raw' | 'centered';
export type FinalLogitsOrder = 'index' | 'ref' | 'target' | 'delta' | 'absolute_delta';
export type FinalLogitsDirection = 'ascending' | 'descending';

/** One Decode row, as a token-analysis pair names it (`row`, `first_divergence`). */
export interface FinalLogitsRequest extends RowAddress {
  row: number;
  first_divergence?: boolean;
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

export interface FinalLogitsResponse extends RowAddress {
  row: number;
  first_divergence: boolean;
  /** The steps the server found for the row. */
  ref_step: number | null;
  target_step: number | null;
  transform: FinalLogitsTransform;
  compatible: boolean;
  reasons: Reason[];
  /** Why the compared vectors read different tokens on the two sides, in the server's words. */
  notes?: Reason[];
  total: number;
  /** Row offset after full-vector sorting; only legacy absolute_delta requests may fall back. */
  offset: number;
  limit: number;
  rows: FinalLogitsRow[];
  ref: TokenDistributionSide;
  target: TokenDistributionSide;
  metrics: FinalLogitsMetrics;
}

/** A Decode row's logits, as every request about its whole vectors names them. */
export type FinalLogitsAddress = Pick<
  FinalLogitsRequest,
  'turn' | 'stage' | 'alignment' | 'row' | 'first_divergence' | 'transform'
>;

/** What every answer about a row's whole vectors repeats. */
interface FinalLogitsEcho extends RowAddress {
  row: number;
  first_divergence: boolean;
  ref_step: number | null;
  target_step: number | null;
  transform: FinalLogitsTransform;
  compatible: boolean;
  /** The vocabulary's length: the longer side's. */
  total: number;
}

/** POST /api/final-logits-summary: what is known of both whole vectors, but their values. */
export interface FinalLogitsSummaryResponse
  extends Omit<FinalLogitsResponse, 'offset' | 'limit' | 'rows'> {
  /** Each vector's statistics and, when they are compared, their difference's, by vocabulary
   *  index; the statistics of a side without a vector have no values. */
  statistics: ElementStatistics;
}

/** POST /api/final-logits-bins: the Inspection line chart's bins (analysis/vector_drawing). */
export interface FinalLogitsBinsRequest extends FinalLogitsAddress {
  /** The vocabulary range shown, both or neither (all of it); the server keeps it inside. */
  start?: number;
  end?: number;
  /** How many bins the chart has room for. */
  capacity: number;
  reduction: 'min' | 'max' | 'mean';
}

/** Bins of [start, end), `size` elements each but the last: one entry per bin in each list.
 *  A value is null where its side's bin is a gap; an index is the bin's representative. */
export interface FinalLogitsBins {
  start: number;
  end: number;
  size: number;
  ref: Array<number | null>;
  target: Array<number | null>;
  ref_index: Array<number | null>;
  target_index: Array<number | null>;
}

export interface FinalLogitsBinsResponse extends FinalLogitsEcho {
  /** The range shown, kept inside the vocabulary. */
  range: {start: number; end: number};
  capacity: number;
  reduction: FinalLogitsBinsRequest['reduction'];
  /** The bins of the whole vocabulary, and of `range`. */
  overview: FinalLogitsBins;
  visible: FinalLogitsBins;
}

/** POST /api/final-logits-points: the Inspection scatter plot's points (analysis/vector_drawing). */
/** The points of a value range shown; without one, final-logits-points draws the paired values'
 *  extent (`FinalLogitsPointsResponse`). */
export interface FinalLogitsPointsRequest extends FinalLogitsAddress {
  start: number;
  end: number;
}

export interface FinalLogitsPointsResponse extends FinalLogitsEcho {
  /** The paired values' extent, padded; -1..1 without a paired element. */
  extent: {start: number; end: number};
  range: {start: number; end: number};
  /** The vocabulary indices whose two logits are both finite, compared, as half-open runs. */
  paired_runs: Array<[number, number]>;
  /** How many indices are paired, and inside `range`; whether the points were grouped. */
  paired: number;
  visible: number;
  grouped: boolean;
  /** The points drawn, by increasing index; a selected index is not among them for its own sake. */
  points: {index: number[]; ref: number[]; target: number[]};
}

/** The points of a value range: what no range changes, the extent and the paired runs, comes with
 *  the points of the whole extent alone. */
export type FinalLogitsZoomedPointsResponse = Omit<
  FinalLogitsPointsResponse,
  'extent' | 'paired_runs'
>;
