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

import type {KvHeadChannel, KvRangeBin} from '../../server/contracts/kv';
export interface ValueRange {
  start: number;
  end: number;
}
export function validValueRange(value: unknown): value is ValueRange {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const r = value as ValueRange;
  return (
    typeof r.start === 'number' &&
    typeof r.end === 'number' &&
    Number.isFinite(r.start) &&
    Number.isFinite(r.end) &&
    r.end > r.start
  );
}
export function pairedChannels(rows: readonly KvHeadChannel[]): PairedChannel[] {
  return rows.filter(isPairedChannel);
}
export type PairedChannel = KvHeadChannel & {ref: number; target: number};
export function isPairedChannel(row: KvHeadChannel): row is PairedChannel {
  return (
    typeof row.ref === 'number' &&
    Number.isFinite(row.ref) &&
    typeof row.target === 'number' &&
    Number.isFinite(row.target)
  );
}
export const SCATTER_POINT_LIMIT = 4096;
export interface ScatterRepresentation {
  rows: readonly PairedChannel[];
  total: number;
  visible: number;
  rendered: number;
  grouped: boolean;
}

/** Keep exact elements, never averages. Each occupied value-grid cell retains its
 * largest absolute difference (then lowest index); a selected element is added.
 * Small tensors retain all points. The full input still determines the extent.
 */
export function scatterRepresentatives(
  rows: readonly KvHeadChannel[],
  range: ValueRange,
  selected: number | null = null,
  limit = SCATTER_POINT_LIMIT,
  comparisonAllowed = true,
): ScatterRepresentation {
  const cells = Math.max(1, Math.floor(Math.sqrt(limit)));
  const grid = new Map<number, PairedChannel>();
  const small: PairedChannel[] = [];
  const visibleRows: PairedChannel[] = [];
  let total = 0;
  let visible = 0;
  let selectedRow: PairedChannel | undefined;
  const score = (row: PairedChannel) => (comparisonAllowed ? Math.abs(row.target - row.ref) : 0);
  const cell = (value: number) =>
    Math.max(
      0,
      Math.min(cells - 1, Math.floor(((value - range.start) / (range.end - range.start)) * cells)),
    );
  for (const row of rows) {
    if (!isPairedChannel(row)) {
      continue;
    }
    total++;
    if (total <= limit) {
      small.push(row);
    }
    if (
      row.ref < range.start ||
      row.ref > range.end ||
      row.target < range.start ||
      row.target > range.end
    ) {
      continue;
    }
    visible++;
    if (visible <= limit) {
      visibleRows.push(row);
    }
    if (row.channel === selected) {
      selectedRow = row;
    }
    const key = cell(row.ref) * cells + cell(row.target);
    const previous = grid.get(key);
    if (
      !previous ||
      score(row) > score(previous) ||
      (score(row) === score(previous) && row.channel < previous.channel)
    ) {
      grid.set(key, row);
    }
  }
  const grouped = visible > limit;
  const result = total <= limit ? small : grouped ? [...grid.values()] : visibleRows;
  if (total > limit) {
    result.sort((a, b) => a.channel - b.channel);
  }
  return withSelected(
    {rows: result, total, visible, rendered: result.length, grouped},
    selectedRow,
    range,
  );
}
/** Grouped points keep the selected element when it lies in the range, in id order. */
export function withSelected(
  representation: ScatterRepresentation,
  selected: PairedChannel | undefined,
  range: ValueRange,
): ScatterRepresentation {
  if (
    !representation.grouped ||
    !selected ||
    selected.ref < range.start ||
    selected.ref > range.end ||
    selected.target < range.start ||
    selected.target > range.end ||
    representation.rows.some((row) => row.channel === selected.channel)
  ) {
    return representation;
  }
  const rows = [...representation.rows, selected].sort((a, b) => a.channel - b.channel);
  return {...representation, rows, rendered: rows.length};
}
/** What a scatter plot draws when another chose its points (the server, for a vocabulary). */
export interface ScatterView {
  /** The extent of every paired element (`scatterExtent`). */
  full: ValueRange;
  /** `scatterRepresentatives` at the chart's range, without the selected element. */
  visible: ScatterRepresentation;
  /** `scatterRepresentatives` at `full`: the overview's points. */
  overview: readonly PairedChannel[];
  /** The paired elements' ids, as increasing half-open runs. */
  paired: readonly KvRangeBin[];
  /** Elements by id: those drawn, and any other looked up (the selected one). */
  elements: ReadonlyMap<number, KvHeadChannel>;
}
/** Ids in increasing order, by position: as a list, or as half-open runs. */
export interface IdSequence {
  length: number;
  at(index: number): number;
  /** The position of an id, -1 when it is not one. */
  indexOf(id: number): number;
}
export function idList(ids: readonly number[]): IdSequence {
  return {length: ids.length, at: (index) => ids[index], indexOf: (id) => ids.indexOf(id)};
}
export function idRuns(runs: readonly KvRangeBin[]): IdSequence {
  const before: number[] = [];
  let length = 0;
  for (const run of runs) {
    before.push(length);
    length += run.end - run.start;
  }
  return {
    length,
    at: (index) => {
      let run = 0;
      while (run + 1 < runs.length && before[run + 1] <= index) {
        run++;
      }
      return runs[run].start + index - before[run];
    },
    indexOf: (id) => {
      const run = runs.findIndex((r) => id >= r.start && id < r.end);
      return run < 0 ? -1 : before[run] + id - runs[run].start;
    },
  };
}
export function scatterExtent(groups: readonly (readonly KvHeadChannel[])[]): ValueRange {
  let start = Infinity;
  let end = -Infinity;
  for (const rows of groups) {
    for (const row of rows) {
      if (!isPairedChannel(row)) {
        continue;
      }
      start = Math.min(start, row.ref, row.target);
      end = Math.max(end, row.ref, row.target);
    }
  }
  if (!Number.isFinite(start)) {
    return {start: -1, end: 1};
  }
  const pad = (end - start || Math.abs(start) || 1) * 0.05;
  return {start: start - pad, end: end + pad};
}
/** A box on either plane becomes one shared, square numerical domain. */
export function scatterRelayout(event: Record<string, unknown>): ValueRange | null {
  const values: number[] = [];
  for (const axis of ['xaxis', 'yaxis', 'xaxis2', 'yaxis2']) {
    const pair = event[`${axis}.range`];
    if (Array.isArray(pair)) {
      values.push(...pair.filter((n): n is number => typeof n === 'number' && Number.isFinite(n)));
    } else {
      for (const i of [0, 1]) {
        const n = event[`${axis}.range[${i}]`];
        if (typeof n === 'number' && Number.isFinite(n)) {
          values.push(n);
        }
      }
    }
  }
  const range = {start: Math.min(...values), end: Math.max(...values)};
  return values.length >= 2 && validValueRange(range) ? range : null;
}
