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

import type {KvRangeBin, KvRangeMetricSummary} from '../../../data/contracts/kv';
import type {TokenPair} from '../token_alignment';

export interface AnalysisAxisToken {
  position: number;
  pair: TokenPair;
}
export interface AnalysisAxisBin extends KvRangeBin {
  representative?: number | null;
}
export interface AnalysisAxisColumn extends AnalysisAxisBin {
  tokens: AnalysisAxisToken[];
  before: boolean;
  after: boolean;
  mismatchCount: number;
  unknownCount: number;
}

/** Same extrema as the chart; mean and unavailable extrema use the first bin position. */
export function representativePosition(
  bin: KvRangeBin,
  reduction: string,
  summary?: KvRangeMetricSummary,
): number {
  const extreme =
    reduction === 'min'
      ? summary?.min_position
      : reduction === 'max'
        ? summary?.max_position
        : null;
  return extreme != null && extreme >= bin.start && extreme < bin.end ? extreme : bin.start;
}

/** Linear scan of sorted aligned tokens; no tensor values or token IDs are compared here. */
export function axisColumns(
  tokens: readonly AnalysisAxisToken[],
  bins: readonly AnalysisAxisBin[],
): AnalysisAxisColumn[] {
  let cursor = 0;
  return bins.map((bin) => {
    while (cursor < tokens.length && tokens[cursor].position < bin.start) cursor++;
    const first = cursor;
    while (cursor < tokens.length && tokens[cursor].position < bin.end) cursor++;
    const members = tokens.slice(first, cursor);
    const requested = bin.representative ?? bin.start;
    const representative = members.find((t) => t.position === requested) ?? members[0];
    const displayed = bin.end - bin.start > 3 ? (representative ? [representative] : []) : members;
    return {
      ...bin,
      tokens: displayed,
      before: !!displayed.length && displayed[0].position > bin.start,
      after: !!displayed.length && displayed[displayed.length - 1].position < bin.end - 1,
      mismatchCount: members.filter((t) => t.pair.match === false).length,
      unknownCount:
        members.filter((t) => t.pair.match === null).length +
        Math.max(0, bin.end - bin.start - members.length),
    };
  });
}

export function tokenPages(
  widths: readonly number[],
  available: number,
): {start: number; end: number}[] {
  const pages: {start: number; end: number}[] = [];
  let start = 0,
    used = 0;
  widths.forEach((width, index) => {
    const next = Math.min(width, available);
    if (index > start && used + next > available) {
      pages.push({start, end: index});
      start = index;
      used = 0;
    }
    used += next;
  });
  if (widths.length) pages.push({start, end: widths.length});
  return pages;
}
