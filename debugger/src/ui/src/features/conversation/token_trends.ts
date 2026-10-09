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

/** Range and min/max/gap sampling adapted from the candidates Plotly/flow virtualizer. */
import type {CapturedToken} from '../../data/contracts/capture';
import {TokenMetricKey} from '../../data/contracts/token_analysis';
import {capturedTokenLabel} from './conversation_content';
/** Quantities of one runtime's own distribution. They need no shared context, so they stay
 *  meaningful after the two generations diverge, drawn as one line per runtime. */
export const SIDE_TREND_METRICS = [
  {
    key: 'selected_probability',
    label: 'Per runtime · Selected probability',
    unit: 'probability',
    reading: 'higher = more confident in the chosen token',
  },
  {
    key: 'entropy',
    label: 'Per runtime · Entropy',
    unit: 'bits',
    reading: 'higher = less certain',
  },
  {
    key: 'margin',
    label: 'Per runtime · Top-2 margin',
    unit: 'probability',
    reading: 'higher = more decisive, near 0 = a near tie',
  },
] as const;
export type SideTrendMetric = (typeof SIDE_TREND_METRICS)[number]['key'];
export type TokenTrendMetric = TokenMetricKey | SideTrendMetric;
export const isSideTrendMetric = (key: string): key is SideTrendMetric =>
  SIDE_TREND_METRICS.some((metric) => metric.key === key);
export interface TokenTrendState {
  open: boolean;
  metric: TokenTrendMetric;
  start: number;
  end: number;
}
export function normalizeTrendRange(
  length: number,
  range: number[],
): [number, number] {
  const max = Math.max(1, length - 1),
    lo = Math.max(0, Math.min(max - 1, Math.floor(Math.min(...range)))),
    hi = Math.max(lo + 1, Math.min(max, Math.ceil(Math.max(...range))));
  return [lo, hi];
}
export function trendTickStep(range: number[]) {
  const raw = Math.max(1, (range[1] - range[0]) / 7),
    unit = 10 ** Math.floor(Math.log10(raw));
  return Math.max(1, ([1, 2, 5, 10].find((n) => n * unit >= raw) || 10) * unit);
}
export function sampleTokenTrend(
  source: (number | null)[],
  range: number[],
  width: number,
) {
  const bins = Math.max(32, Math.min(600, Math.floor(width / 2))),
    indices = new Set<number>();
  if (source.length) {
    indices.add(0);
    indices.add(source.length - 1);
  }
  function add(lo: number, hi: number, budget: number) {
    const step = Math.max(1, Math.ceil((hi - lo + 1) / budget));
    for (let start = lo; start <= hi; start += step) {
      const end = Math.min(hi + 1, start + step);
      let min: number | null = null,
        max: number | null = null,
        gap: number | null = null;
      for (let index = start; index < end; index++) {
        const value = source[index];
        if (value == null || !Number.isFinite(value)) {
          gap = index;
          continue;
        }
        if (min === null || value < source[min]!) min = index;
        if (max === null || value > source[max]!) max = index;
      }
      if (min !== null) indices.add(min);
      if (max !== null) indices.add(max);
      if (gap !== null) indices.add(gap);
    }
  }
  if (source.length <= 1200) source.forEach((_, i) => indices.add(i));
  else {
    add(0, source.length - 1, bins);
    add(range[0], Math.min(source.length - 1, range[1]), bins);
  }
  const x = [...indices].sort((a, b) => a - b);
  let max = 0,
    available = 0;
  for (let i = range[0]; i <= Math.min(source.length - 1, range[1]); i++) {
    const value = source[i];
    if (value != null && Number.isFinite(value)) {
      max = Math.max(max, value);
      available++;
    }
  }
  return {x, y: x.map((i) => source[i]), max, available};
}
/** Plotly hover labels parse a small HTML subset, so token text has to reach them escaped. */
export const plotText = (text: string) =>
  text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
/** One hover line for one runtime: its generation step, its token as a quoted string so that
 *  whitespace stays legible, and its value when the metric is per runtime. A gap reads ∅. */
export function trendTokenLine(
  name: string,
  token: CapturedToken | undefined,
  value?: string,
) {
  if (!token) return `${name}: ∅`;
  return (
    `${name} · step ${token.step} ${plotText(JSON.stringify(capturedTokenLabel(token)))}` +
    (value === undefined ? '' : `: <b>${value}</b>`)
  );
}
