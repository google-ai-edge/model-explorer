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

import type {
  TokenAnalysisPair,
  TokenMetric,
  TokenMetricKey,
} from '../../data/contracts/token_analysis';
import {formatSignificant} from '../../shared/format/format';
import type {TokenPair} from './token_alignment';
import {TOKEN_METRICS} from './token_metric_metadata';
export type TokenColor = 'token_match' | 'none' | TokenMetricKey;
export const analysisKey = (
  turn: number,
  ref: number | null | undefined,
  target: number | null | undefined,
) => `${turn}:${ref ?? '-'}:${target ?? '-'}`;
export const pairKey = (turn: number, pair: TokenPair) =>
  analysisKey(turn, pair.ref?.step, pair.target?.step);
export const colorLabel = (key: TokenColor) =>
  key === 'none'
    ? 'None'
    : key === 'token_match'
      ? 'Token · Match'
      : (TOKEN_METRICS.find((m) => m.key === key)?.label ?? 'None');
export const TOKEN_COLOR_MAX: Record<TokenMetricKey, number> = {
  kl: 0.01,
  js: 1,
  relative_l2: 1,
  cosine_distance: 0.02,
  norm_ratio: 1.2,
  max_abs_error: 0.1,
};
export const normalizeColor = (value: boolean | TokenColor): TokenColor =>
  value === true ? 'token_match' : value === false ? 'none' : value;
export function heatLevel(
  color: TokenColor,
  record: TokenAnalysisPair | undefined,
) {
  if (color === 'none' || color === 'token_match') return null;
  const value = record?.metrics[color]?.value;
  return value == null
    ? null
    : Math.min(100, Math.max(0, (value / TOKEN_COLOR_MAX[color]) * 100)) + '%';
}
export function heatPalette(color: TokenColor) {
  return color === 'kl' || color === 'js' ? 'logits' : 'activation';
}
/** The value as the UI presents it (cosine similarity for the recorded cosine distance). */
export function presentValue(key: TokenMetricKey, value: number) {
  const present = TOKEN_METRICS.find((m) => m.key === key)?.present;
  return present ? present(value) : value;
}
export function metricText(
  metric: TokenMetric | undefined,
  key?: TokenMetricKey,
) {
  return metric?.value == null
    ? metric?.reason && !metric.reason.toLowerCase().includes('not captured')
      ? 'Unavailable'
      : 'Not captured'
    : formatSignificant(
        key ? presentValue(key, metric.value) : metric.value,
        5,
      );
}
export function queryRecord(
  pair: TokenPair,
  analysis: TokenAnalysisPair | undefined,
) {
  if (!analysis) return {token_match: pair.match};
  const distance = analysis.metrics['cosine_distance']?.value;
  return {
    token_match: pair.match,
    ...Object.fromEntries(
      TOKEN_METRICS.map((m) => [
        m.key,
        analysis?.metrics[m.key]?.value ?? null,
      ]),
    ),
    cosine_similarity: distance == null ? null : 1 - distance,
  };
}
