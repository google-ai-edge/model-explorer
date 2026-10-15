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

import {rangeChartGeometry, rangeChartHeight} from './range_chart_geometry';
import type {HeatmapData} from './heatmap_model';

/** Plotly projection only: no capture API, KV types, or component state. */
export function heatmapTraces(
  data: HeatmapData,
  style: {
    normalize: (value: number | null) => number | null;
    low: string;
    high: string;
    marker: string;
    filtered: boolean;
  },
) {
  if (data.planes.length < 1 || data.planes.length > 2)
    throw new Error('Heatmap requires one or two planes');
  const traces: unknown[] = [];
  data.planes.forEach((plane, index) => {
    if (
      plane.cells.length !== data.layers.length ||
      plane.cells.some((row) => row.length !== data.positions.length)
    )
      throw new Error('Heatmap dimensions do not match');
    const yaxis = index ? 'y2' : 'y';
    traces.push({
      type: 'heatmap',
      yaxis,
      name: plane.label,
      x: data.positions.length
        ? [data.positions[0].start - 0.5, ...data.positions.map((p) => p.end - 0.5)]
        : [],
      y: data.layers.map((_, i) => i),
      z: plane.cells.map((row) => row.map((cell) => style.normalize(cell.value))),
      customdata: plane.cells.map((row) => row.map((cell) => cell.evidence)),
      text: plane.cells.map((row) => row.map((cell) => cell.hover)),
      hovertemplate: '%{text}<extra></extra>',
      hoverongaps: true,
      colorscale: [
        [0, style.low],
        [1, style.high],
      ],
      zmin: 0,
      zmax: 1,
      showscale: false,
      zsmooth: false,
      xgap: 0,
      ygap: 1,
    });
    if (style.filtered) {
      const matches = plane.cells.flatMap((row, y) =>
        row.flatMap((cell, x) =>
          cell.matched
            ? [
                {
                  x: (data.positions[x].start + data.positions[x].end - 1) / 2,
                  y,
                  evidence: cell.evidence,
                },
              ]
            : [],
        ),
      );
      traces.push({
        type: 'scatter',
        yaxis,
        mode: 'markers',
        x: matches.map((p) => p.x),
        y: matches.map((p) => p.y),
        customdata: matches.map((p) => p.evidence),
        marker: {size: 4, color: style.marker},
        hoverinfo: 'skip',
        showlegend: false,
      });
    }
  });
  return traces;
}

export function heatmapChartHeight(layers: number, planes = 1) {
  return rangeChartHeight(layers, planes);
}

/** Every plane shares the same Layer mapping and a single external Position axis. */
export function heatmapAxes(data: HeatmapData, height: number) {
  const dual = data.planes.length === 2;
  const {fraction: planeFraction, plotHeight} = rangeChartGeometry(height, dual);
  const planeHeight = plotHeight * (dual ? planeFraction : 1);
  const stride = Math.max(1, Math.ceil(data.layers.length / Math.max(1, planeHeight / 18)));
  const ticks = data.layers.map((id, index) => ({id, index})).filter((t) => t.index % stride === 0);
  const axis = (domain: number[]) => ({
    domain,
    range: [data.layers.length - 0.5, -0.5],
    tickvals: ticks.map((t) => t.index),
    ticktext: ticks.map((t) => `L${t.id}`),
    fixedrange: true,
    showgrid: false,
    zeroline: false,
  });
  return {
    yaxis: axis(dual ? [1 - planeFraction, 1] : [0, 1]),
    ...(dual ? {yaxis2: axis([0, planeFraction])} : {}),
    annotations: dual
      ? data.planes.map((plane, index) => ({
          text: plane.label,
          x: 0,
          y: index ? planeFraction : 1,
          yanchor: 'bottom',
          yshift: 6,
          xref: 'paper',
          yref: 'paper',
          showarrow: false,
          xanchor: 'left',
        }))
      : [],
  };
}

/** Sparse labels always name actual bins and stay centered on those bins. */
export function positionAxisLabels(
  positions: readonly {start: number; end: number}[],
  range: {start: number; end: number},
  width: number,
) {
  const visible = positions.filter((p) => p.start >= range.start && p.end <= range.end);
  const tickvals: number[] = [],
    ticktext: string[] = [];
  let previousRight = -Infinity;
  for (const bin of visible) {
    const text = bin.end - bin.start === 1 ? String(bin.start) : `${bin.start}–${bin.end - 1}`;
    const center = (bin.start + bin.end - 1) / 2;
    const pixel = ((center - range.start + 0.5) / (range.end - range.start)) * width;
    const halfWidth = text.length * 3.5;
    if (pixel - halfWidth < previousRight + 12) continue;
    tickvals.push(center);
    ticktext.push(text);
    previousRight = pixel + halfWidth;
  }
  return {tickmode: 'array', tickvals, ticktext};
}
