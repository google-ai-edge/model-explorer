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

import {TokenMetricKey} from '../../server/contracts/token_analysis';
/** Fields of one runtime's own sampling distribution, which the Sampling panel draws as one line
 *  per runtime; they need no shared context, so they stay meaningful after the two generations
 *  diverge. Their labels and units are the metric catalog's. */
export const PER_RUNTIME_METRICS = ['selected_probability', 'entropy', 'margin'] as const;
export type PerRuntimeMetric = (typeof PER_RUNTIME_METRICS)[number];
/** The Final Logits panel's keys of the server's whole-vector `rmse` and `max_abs`, as views
 *  save them. */
export const FINAL_LOGIT_METRICS = ['logit_rmse', 'logit_max_abs'] as const;
export type FinalLogitMetric = (typeof FINAL_LOGIT_METRICS)[number];
export type LogitPanelMetric = TokenMetricKey | PerRuntimeMetric | FinalLogitMetric;
export function isFinalLogitMetric(key: string): key is FinalLogitMetric {
  return (FINAL_LOGIT_METRICS as readonly string[]).includes(key);
}
export function isPerRuntimeMetric(key: string): key is PerRuntimeMetric {
  return (PER_RUNTIME_METRICS as readonly string[]).includes(key);
}
/** The metric catalog's id of a panel metric key. */
export function catalogMetricId(key: string): string {
  return key === 'logit_rmse' ? 'rmse' : key === 'logit_max_abs' ? 'max_abs' : key;
}
/** The panel key of a Final Logits metric the catalog names. */
export function finalLogitKey(id: string): LogitPanelMetric {
  return (id === 'rmse' ? 'logit_rmse' : id === 'max_abs' ? 'logit_max_abs' : id) as LogitPanelMetric;
}
export interface LogitPanelState {
  logitView?: 'comparison' | 'inspection';
  logitPlot?: 'index' | 'scatter';
  logitElementRange?: {start: number; end: number};
  logitValueRange?: {start: number; end: number};
  logitElementReduction?: 'min' | 'max' | 'mean';
  /** @deprecated Accepted in old saved views; all Final Logits UI values use raw logits. */
  logitTransform?: 'raw' | 'centered';
  logitPosition?: number;
  logitOffset?: number;
  logitOrder?: 'index' | 'ref' | 'target' | 'delta' | 'absolute_delta';
  logitSortDirection?: 'ascending' | 'descending';
  positionRange?: {start: number; end: number};
  positionReduction?: 'min' | 'max' | 'mean';
  open: boolean;
  metric: LogitPanelMetric;
}
