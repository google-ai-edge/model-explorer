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
import {plotlyColor} from '../plotly_loader';
import type {KvRangeBin} from '../../server/contracts/kv';
import {type Runtime, runtimeSeries} from './runtime_series';

/** Shared style primitives returned by `rangeChartStyle`. */
export interface RangeChartStyle {
  surface: string;
  muted: string;
  rule: string;
  fraction: number;
  layout: Record<string, unknown>;
  xaxis: (range: KvRangeBin, full: KvRangeBin, title: string) => Record<string, unknown>;
  yaxis: (domain: number[], title: string, tickformat?: string) => Record<string, unknown>;
  labels: (labels: readonly string[]) => Record<string, unknown>[];
}

/** Shared geometry and visual language for metric and raw-value range charts. */
export function rangeChartStyle(host: HTMLElement, height: number, dual: boolean): RangeChartStyle {
  const color = (name: string): string => plotlyColor(host, `var(${name})`);
  const surface = color('--chart-canvas');
  const muted = color('--chart-muted');
  const rule = color('--chart-line');
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
export function rangeLineStyle(
  color: string,
  emphasized = false,
): {line: {color: string; width: number}; marker: {size: number; color: string}} {
  return {line: {color, width: emphasized ? 3 : 1.5}, marker: {size: 3, color}};
}

export function runtimeLineStyle(
  host: HTMLElement,
  runtime: Runtime,
  emphasized = false,
): {
  line: {color: string; width: number; dash: 'solid' | 'dash'};
  marker: {size: number; color: string};
} {
  const series = runtimeSeries(runtime);
  const style = rangeLineStyle(plotlyColor(host, series.color), emphasized);
  return {...style, line: {...style.line, dash: series.dash}};
}

/** The y range of a line chart whose every value is one value `c` (a greedy sampler's entropy 0
 *  bits or selected probability 1 at every step, say): from zero to past `c`, padded so that the
 *  line is drawn clear of the axis' edge rather than on it, where it reads as no line at all.
 *  Null when the values differ, or none is finite: the axis ranges itself. */
export function flatLineRange(
  values: readonly (number | null | undefined)[],
): [number, number] | null {
  const finite = values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
  if (!finite.length || finite.some((v) => v !== finite[0])) {
    return null;
  }
  const c = finite[0];
  const span = Math.abs(c) || 1;
  const pad = span * 0.08;
  return [Math.min(0, c) - pad, Math.max(0, c) + pad + (c === 0 ? span : 0)];
}
