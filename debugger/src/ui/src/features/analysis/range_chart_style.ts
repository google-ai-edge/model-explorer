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

import {rangeChartGeometry} from './range_chart_geometry';
import {plotlyColor} from '../../shared/plotly_loader';
import type {KvRangeBin} from '../../data/contracts/kv';
import {type AnalysisRuntime, runtimeSeries} from './runtime_series';

/** Shared geometry and visual language for metric and raw-value range charts. */
export function rangeChartStyle(host: HTMLElement, height: number, dual: boolean) {
  const color = (name: string) => plotlyColor(host, `var(${name})`);
  const surface = color('--kv-canvas'),
    muted = color('--kv-muted'),
    rule = color('--kv-line');
  const {margin, plotHeight, fraction} = rangeChartGeometry(height, dual);
  return {
    surface,
    muted,
    rule,
    fraction,
    layout: {
      paper_bgcolor: surface,
      plot_bgcolor: surface,
      font: {
        family: getComputedStyle(host).getPropertyValue('--me-font-family').trim(),
        size: 11,
        color: muted,
      },
      margin,
      showlegend: false,
      hovermode: 'closest',
      hoverlabel: {bgcolor: surface, bordercolor: rule, font: {color: muted}},
    },
    xaxis(range: KvRangeBin, full: KvRangeBin, title: string) {
      return {
        anchor: dual ? 'y2' : 'y',
        range: [range.start - 0.5, range.end - 0.5],
        tickformat: 'd',
        showgrid: false,
        zeroline: false,
        title: {text: title, standoff: 6},
        rangeslider: {
          visible: true,
          range: [full.start - 0.5, full.end - 0.5],
          thickness: Math.min(0.5, (dual ? 72 : 40) / plotHeight),
          bgcolor: surface,
          bordercolor: rule,
          borderwidth: 1,
          yaxis: {rangemode: 'auto'},
        },
      };
    },
    yaxis(domain: number[], title: string, tickformat = '.3~g') {
      return {
        domain,
        fixedrange: true,
        title: {text: title, standoff: 4},
        rangemode: 'tozero',
        tickformat,
        gridcolor: rule,
        zeroline: false,
      };
    },
    labels(labels: readonly string[]) {
      return labels.map((text, i) => ({
        text,
        x: 0,
        y: i ? fraction : 1,
        xref: 'paper',
        yref: 'paper',
        xanchor: 'left',
        yanchor: 'bottom',
        yshift: 6,
        showarrow: false,
      }));
    },
  };
}
export function rangeLineStyle(color: string, emphasized = false) {
  return {line: {color, width: emphasized ? 3 : 1.5}, marker: {size: 3, color}};
}

export function runtimeLineStyle(host: HTMLElement, runtime: AnalysisRuntime, emphasized = false) {
  const series = runtimeSeries(runtime);
  const style = rangeLineStyle(plotlyColor(host, series.color), emphasized);
  return {...style, line: {...style.line, dash: series.dash}};
}
