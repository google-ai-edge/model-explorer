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
  plotlyColumns,
  plotlyPositionRange,
  plotlyColumnSelection,
  heatmapHighlightBoxes,
} from './kv_plotly_range';

describe('KV Plotly Position contract', () => {
  const full = {start: 0, end: 131072};
  it('converts slider and drag events into exact inclusive cell boundaries', () => {
    expect(plotlyPositionRange({'xaxis.range': [65529.5, 65545.5]}, full)).toEqual({
      start: 65530,
      end: 65546,
    });
    expect(plotlyPositionRange({'xaxis.range[0]': -20, 'xaxis.range[1]': 999999}, full)).toEqual(
      full,
    );
    expect(plotlyPositionRange({'xaxis.autorange': true}, full)).toEqual(full);
    expect(plotlyPositionRange({'yaxis.range': [0, 1]}, full)).toBeNull();
    expect(plotlyPositionRange({'xaxis.range': [12, 12]}, full)).toEqual({start: 13, end: 14});
  });
  it('selects complete global Position bins between sparse ticks, without rebasing by turn', () => {
    const secondTurn = [
      {start: 120, end: 121},
      {start: 121, end: 129},
    ];
    expect(plotlyColumnSelection(secondTurn, 120)).toEqual({start: 120, end: 121, layer: null});
    expect(plotlyColumnSelection(secondTurn, 123.4)).toEqual({start: 121, end: 129, layer: null});
    expect(plotlyColumnSelection(secondTurn, 0)).toBeNull();
    expect(plotlyColumnSelection(secondTurn, 128.5)).toBeNull();
  });
  it('links hover/selection across K/V without hiding the remaining grid', () => {
    const x = {_offset: 60, _length: 100, d2p: (v: number) => (v + 0.5) * 10};
    const axes = [30, 100].map((top) => ({
      _offset: top,
      _length: 40,
      d2p: (v: number) => (v + 0.5) * 20,
    }));
    expect(heatmapHighlightBoxes(x, axes, [2, 4], {start: 3, end: 4, layer: 4})).toEqual([
      {left: 90, top: 50, width: 10, height: 20},
      {left: 90, top: 120, width: 10, height: 20},
    ]);
    expect(heatmapHighlightBoxes(x, axes, [2, 4], {start: 8, end: 15, layer: null})).toEqual([
      {left: 140, top: 30, width: 20, height: 40},
      {left: 140, top: 100, width: 20, height: 40},
    ]);
    expect(heatmapHighlightBoxes(x, axes, [2, 4], {start: 10, end: 11, layer: null})).toEqual([]);
  });
  it('keeps full-domain slider context without replacing detailed cells by coarse bins', () => {
    const columns = plotlyColumns(
      [
        {start: 4, end: 5},
        {start: 5, end: 6},
      ],
      [{start: 0, end: 10}],
      {start: 4, end: 6},
    );
    expect(columns.map(({start, end}) => [start, end])).toEqual([
      [0, 4],
      [4, 5],
      [5, 6],
      [6, 10],
    ]);
    expect(columns.map((c) => c.overview)).toEqual([true, false, false, true]);
    expect(columns[0].source).toEqual({start: 0, end: 10});
  });
});
