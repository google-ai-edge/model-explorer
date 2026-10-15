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

import {
  axisColumns,
  representativePosition,
  tokenPages,
  type AnalysisAxisToken,
} from './analysis_token_model';
import type {KvRangeMetricSummary} from '../../../data/contracts/kv';

const tokens = (count: number): AnalysisAxisToken[] =>
  Array.from({length: count}, (_, i) => ({position: 32 + i, pair: {index: i, match: true}}));
const summary: KvRangeMetricSummary = {
  min: 0.1,
  max: 0.9,
  min_position: 39,
  max_position: 33,
  valid_count: 8,
};
describe('fullscreen token strip model', () => {
  it('uses chart extrema and the first token for mean or unavailable extrema', () => {
    const bin = {start: 32, end: 40};
    expect(representativePosition(bin, 'min', summary)).toBe(39);
    expect(representativePosition(bin, 'max', summary)).toBe(33);
    expect(representativePosition(bin, 'mean', summary)).toBe(32);
    expect(representativePosition(bin, 'min', {...summary, min_position: 100})).toBe(32);
    expect(representativePosition(bin, 'max', {...summary, max_position: null})).toBe(32);
    expect(representativePosition(bin, 'min')).toBe(32);
  });
  it('displays the first captured token when no representative is specified or captured', () => {
    const data = tokens(8).slice(1);
    const columns = axisColumns(data, [
      {start: 32, end: 36},
      {start: 36, end: 40, representative: 100},
    ]);
    expect(columns.map((column) => column.tokens.map((token) => token.position))).toEqual([
      [33],
      [36],
    ]);
  });
  it('colors the whole bin on any mismatch even when the representative matches', () => {
    const data = tokens(8);
    data[7].pair.match = false;
    const [column] = axisColumns(data, [{start: 32, end: 40, representative: 35}]);
    expect(column.mismatchCount).toBe(1);
    expect(column.tokens.map((t) => t.position)).toEqual([35]);
    expect(column.tokens[0].pair.match).toBe(true);
    expect([column.start, column.end]).toEqual([32, 40]);
    expect([column.before, column.after]).toEqual([true, true]);
  });
  it('keeps all positions in small bins as one selectable column and handles edge extrema', () => {
    const data = tokens(8);
    const columns = axisColumns(data, [
      {start: 32, end: 35},
      {start: 35, end: 40, representative: 39},
    ]);
    expect(columns[0].tokens).toHaveLength(3);
    expect(columns[1].tokens.map((t) => t.position)).toEqual([39]);
    expect([columns[1].before, columns[1].after]).toEqual([true, false]);
  });
  it('does not turn missing capture into a token mismatch', () => {
    const data = tokens(2);
    data[0].pair.match = null;
    expect(axisColumns(data, [{start: 32, end: 36}])[0]).toMatchObject({
      mismatchCount: 0,
      unknownCount: 3,
    });
  });
  it('paginates whole columns to the available width without gaps or duplicates', () => {
    expect(tokenPages([80, 120, 60, 110], 200)).toEqual([
      {start: 0, end: 2},
      {start: 2, end: 4},
    ]);
    expect(tokenPages([80, 120, 60, 110], 500)).toEqual([{start: 0, end: 4}]);
    expect(tokenPages([240, 70], 100)).toEqual([
      {start: 0, end: 1},
      {start: 1, end: 2},
    ]);
    expect(tokenPages([], 100)).toEqual([]);
  });
});
