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

import type {MetricDescription} from '../../server/contracts/metrics';
import type {KvRangeMetricSummary, KvRangeRow} from '../../server/contracts/kv';
import {
  defaultPositionReduction,
  HEATMAP_DEFINITIONS,
  HeatmapReduction,
  heatmapControls,
  heatmapMissingRanges,
} from './heatmap_model';
import {heatmapTraces, positionAxisLabels} from './heatmap_plot';
import {lineChartPoints} from './line_chart_points';
import {rangeHeatmapData} from './range_heatmap_data';

/** What the specs read of a Plotly trace heatmapTraces builds. */
interface Trace {
  name: string;
  x: number[];
  yaxis: string;
  z: (number | null)[][];
  colorscale: unknown;
}

describe('heatmaps', () => {
  it('dimension controls share select/reduce semantics and preserve sparse member IDs', () => {
    expect(heatmapControls(HEATMAP_DEFINITIONS.activations, {}, {})[0].id).toBe('position');
    // The worst value by default: the least where larger is better.
    expect(defaultPositionReduction({larger_is_better: true} as MetricDescription)).toBe('min');
    expect(defaultPositionReduction({larger_is_better: false} as MetricDescription)).toBe('max');
    expect(defaultPositionReduction(undefined)).toBe('max');
    const heads = heatmapControls(
      HEATMAP_DEFINITIONS.kv,
      {head: '7'},
      {head: [{id: '7', label: 'Head 7'}]},
    );
    expect(heads[1].value).toBe('7');
    expect(heads[1].choices.map((x) => x.id)).toStrictEqual(['7', 'min', 'max', 'mean']);
    const points = heatmapControls(
      HEATMAP_DEFINITIONS.intermediate,
      {syncPoint: 'attention'},
      {syncPoint: [{id: 'attention', label: 'Attention'}]},
    );
    expect(points[1].choices[0].id).toBe('attention');
  });

  it('single-position KV adaptation keeps K/V identity, head policy and missing cells', () => {
    const bin = {start: 10, end: 11};
    const rows = [
      {
        layer: 4,
        bins: [{...bin, metrics: {cosine_similarity: {min: 0.8, max: 0.8, valid_count: 1}}}],
      },
    ] as KvRangeRow[];
    const data = rangeHeatmapData(
      [4, 9],
      [{...bin, source: bin, overview: false}],
      [
        {kind: 'key', rows, overview: []},
        {kind: 'value', rows: [], overview: []},
      ],
      'cosine_similarity',
      'min',
      String,
      'min',
    );
    expect(data.planes[0].cells[0][0].value).toBe(0.8);
    expect(data.planes[0].cells[1][0].value).toBe(null);
    expect(data.planes[1].cells[0][0].value).toBe(null);
    expect(data.planes[0].cells[0][0].evidence).toStrictEqual({...bin, layer: 4, kind: 'key'});
    expect(data.planes[0].cells[0][0].hover).toBe('Layer 4 · Position 10');
    const traces = heatmapTraces(data, {
      normalize: (v: number | null) => (v === null ? null : (1 - v) / 2),
      low: 'white',
      high: 'red',
    }) as Trace[];
    expect(traces[0].yaxis).toBe('y');
    expect(traces[1].yaxis).toBe('y2');
    expect(traces[0].x).toStrictEqual([9.5, 10.5]);
    expect(traces[1].x).toStrictEqual(traces[0].x);
    expect(traces[1].z[0][0]).toBe(null);
    expect(traces[0].colorscale).toStrictEqual(traces[1].colorscale);
  });

  it('sampled overview keeps full evidence interval and worst similarity, not a fabricated position', () => {
    const source = {start: 0, end: 8};
    const data = rangeHeatmapData(
      [2],
      [{start: 4, end: 8, source, overview: true}],
      [
        {
          kind: 'value',
          rows: [],
          overview: [
            {
              layer: 2,
              bins: [
                {...source, metrics: {cosine_similarity: {min: 0.2, max: 0.9, valid_count: 8}}},
              ],
            },
          ] as KvRangeRow[],
        },
      ],
      'cosine_similarity',
      '3',
      String,
      'min',
    );
    expect(data.planes[0].cells[0][0].value).toBe(0.2);
    expect((data.planes[0].cells[0][0].evidence as {start: number}).start).toBe(0);
  });

  it('generic heatmaps support single non-KV planes and reject more than two or malformed matrices', () => {
    const data = {
      layers: [2],
      positions: [{start: 0, end: 1}],
      planes: [
        {
          id: 'activation',
          label: 'Activations',
          cells: [[{value: 0.9, evidence: {layer: 2}, hover: 'Activation'}]],
        },
      ],
    };
    const style = {normalize: (v: number | null) => v, low: 'white', high: 'red'};
    expect((heatmapTraces(data, style)[0] as Trace).name).toBe('Activations');
    expect(() =>
      heatmapTraces({...data, planes: [...data.planes, ...data.planes, ...data.planes]}, style),
    ).toThrow(/one or two/);
    expect(() => heatmapTraces({...data, layers: [2, 3]}, style)).toThrow(/dimensions/);
  });

  it('Positions reducers use real summaries after Heads and preserve missing means', () => {
    const bin = {start: 0, end: 3};
    const summary = {min: 0.1, max: 0.9, mean: 0.4, valid_count: 3} as KvRangeMetricSummary;
    const render = (mode: HeatmapReduction, value = summary) =>
      rangeHeatmapData(
        [0],
        [{...bin, source: bin, overview: false}],
        [
          {
            kind: 'key',
            rows: [
              {layer: 0, bins: [{...bin, metrics: {cosine_similarity: value}}]},
            ] as KvRangeRow[],
            overview: [],
          },
        ],
        'cosine_similarity',
        'mean',
        String,
        mode,
      ).planes[0].cells[0][0];
    expect(render('min').value).toBe(0.1);
    expect(render('max').value).toBe(0.9);
    expect(render('mean').value).toBe(0.4);
    expect(render('mean').hover).toBe('Layer 0 · Position 0–2');
    expect(render('mean', {...summary, mean: undefined}).value).toBe(null);
    expect(render('mean', {...summary, valid_count: 0}).value).toBe(null);
  });

  it('unavailable hatch runs preserve plane, layer and gaps between valid cells', () => {
    const cells = (values: (number | null)[]) =>
      values.map((value) => ({value, evidence: null, hover: ''}));
    expect(
      heatmapMissingRanges({
        layers: [4],
        positions: [
          {start: 10, end: 11},
          {start: 11, end: 12},
          {start: 12, end: 13},
        ],
        planes: [
          {id: 'key', label: 'K', cells: [cells([null, null, 0])]},
          {id: 'value', label: 'V', cells: [cells([0, null, 1])]},
        ],
      }),
    ).toStrictEqual([
      {plane: 0, layer: 4, start: 10, end: 12},
      {plane: 1, layer: 4, start: 11, end: 12},
    ]);
  });

  it('wide aggregate cells show inclusive ranges, narrow cells use a sparse Position scale', () => {
    const bins = [
      {start: 128, end: 136},
      {start: 136, end: 144},
    ];
    expect(positionAxisLabels(bins, {start: 128, end: 144}, 400).ticktext).toStrictEqual([
      '128–135',
      '136–143',
    ]);
    const sparse = positionAxisLabels(bins, {start: 128, end: 144}, 60);
    expect(sparse.tickmode).toBe('array');
    expect(sparse.tickvals).toStrictEqual([131.5]);
    expect(sparse.ticktext).toStrictEqual(['128–135']);
  });

  it('Line chart uses the Heatmap bin reduction, centers and preserves missing intervals', () => {
    const bins = [
      {start: 17, end: 23},
      {start: 23, end: 29},
      {start: 32, end: 38},
    ];
    const summary = {min: 0.2, mean: 0.6, max: 0.9, valid_count: 6} as KvRangeMetricSummary;
    for (const reduction of ['min', 'mean', 'max'] as const) {
      expect(lineChartPoints(bins, [summary, null, summary], reduction)).toStrictEqual([
        {position: 19.5, value: summary[reduction], bin: 0},
        null,
        null,
        {position: 34.5, value: summary[reduction], bin: 2},
      ]);
    }
    expect(lineChartPoints([bins[0]], [{...summary, valid_count: 0}], 'mean')).toStrictEqual([
      null,
    ]);
  });
});
