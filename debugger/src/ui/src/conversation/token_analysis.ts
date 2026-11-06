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

import type {TokenPair} from './token_alignment';
import type {
  TokenMetricKey,
  TokenMetric,
  TokenAnalysisPair,
} from '../server/contracts/token_analysis';
import type {MetricLookup} from '../server/metric_catalog';
import {notCaptured} from '../server/reasons';
import {formatSignificant} from '../widgets/format/format';
import {tokenHeat} from '../widgets/metric_scale/metric_scale';
export type TokenColor = 'token_match' | 'none' | TokenMetricKey;
export const analysisKey = (
  turn: number,
  ref: number | null | undefined,
  target: number | null | undefined,
): string => `${turn}:${ref ?? '-'}:${target ?? '-'}`;
export const pairKey = (turn: number, pair: TokenPair): string =>
  analysisKey(turn, pair.ref?.step, pair.target?.step);
/** The metrics Token Diff colors tokens by: the catalog's, in its order. */
export const tokenMetrics = (catalog: MetricLookup) => catalog.choices('token_diff');
export const colorLabel = (catalog: MetricLookup, key: TokenColor): string =>
  key === 'none' ? '-' : key === 'token_match' ? 'Token · Match' : catalog.label(key, 'token_diff');
/** A target token's heat, as a CSS percentage, by the catalog's heat scale of the metric. */
export function heatLevel(
  catalog: MetricLookup,
  color: TokenColor,
  record: TokenAnalysisPair | undefined,
): string | null {
  if (color === 'none' || color === 'token_match') {
    return null;
  }
  const heat = tokenHeat(catalog.metric(color), record?.metrics[color]?.value);
  return heat === null ? null : heat * 100 + '%';
}
/** Metrics of the logits take the logits palette; those of activations the activation one. */
export function heatPalette(catalog: MetricLookup, color: TokenColor): 'logits' | 'activation' {
  return catalog.analysis('token_diff')?.groups[color] === 'Logits' ? 'logits' : 'activation';
}
export function metricText(metric: TokenMetric | undefined): string {
  return metric?.value == null
    ? notCaptured(metric?.reasons)
      ? 'Not captured'
      : 'Unavailable'
    : formatSignificant(metric.value, 5);
}
/** The fields a token query reads: the match and every metric the server answered with. */
export function queryRecord(
  pair: TokenPair,
  analysis: TokenAnalysisPair | undefined,
): Record<string, number | boolean | null> {
  if (!analysis) {
    return {token_match: pair.match};
  }
  return {
    token_match: pair.match,
    ...Object.fromEntries(
      Object.entries(analysis.metrics).map(([key, metric]) => [key, metric?.value ?? null]),
    ),
  };
}
