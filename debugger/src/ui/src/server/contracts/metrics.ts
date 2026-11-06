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

/** GET /api/metrics: what every metric is and how it is shown, as the server describes it. */

/** How a value is written: fixed decimals or significant digits, after an optional `scale`. */
export interface MetricFormat {
  style: 'fixed' | 'significant';
  decimals?: number;
  digits?: number;
  /** The value is multiplied by it for display only, as a probability shown in percent. */
  scale?: number;
  suffix?: string;
}
/** Color scale thresholds and heat legend metadata for a metric. */
export interface MetricColor {
  /** The four steps of a stepped color scale, away from `ideal`; null for a continuous one. */
  thresholds: number[] | null;
  /** The value at which a token's heat is full. */
  heat_full: number | null;
  /**
   * What a legend of the token heat writes where the heat is none (the ideal value) and where
   * it is full ('1 ± 20%' when both sides of the ideal reach it); null without a heat scale.
   */
  heat_labels: {none: string; full: string} | null;
}
/** Full catalog metadata describing a single comparison metric. */
export interface MetricDescription {
  /** The name every response uses for the metric. */
  id: string;
  label: string;
  unit: string;
  /** Written after a number, as `%`; empty when nothing is. */
  unit_label: string;
  measures: string;
  formula: string;
  range: string;
  reading: string;
  larger_is_better: boolean | null;
  /** The value of two identical sides. */
  ideal: number | null;
  bounds: [number | null, number | null];
  format: MetricFormat;
  color: MetricColor;
  /** What the metric compares, when it is the same in every analysis. */
  inputs: string | null;
  /** The analyses that show it by default. */
  analyses: string[];
}
/**
 * Why a value is missing, or two sides are not compared, as the server says it: the UI decides on
 * `code`, one of `MetricCatalog.reasons`, and shows `text` (`reasonText`, `server/reasons.ts`).
 */
export interface Reason {
  code: string;
  /** The side it is of; null for both sides, or for the pair. */
  side: 'ref' | 'target' | null;
  text: string;
}
/** One reason code and its words, which the UI shows where a response gives the code alone. */
export interface ReasonDescription {
  code: string;
  text: string;
}
/** One analysis stage view and the metrics it offers in the catalog. */
export interface MetricAnalysis {
  id: string;
  label: string;
  /** The metrics it offers, in order. */
  metrics: string[];
  default_metric: string;
  /** How the KV heads are combined by default; KV cache only. */
  default_head?: string;
  /** The group each metric's label is prefixed with, as `Logits`. */
  groups: Record<string, string>;
  /** What it compares, unless a metric names its own inputs. */
  inputs: string;
}
/** Full metric catalog response from GET /api/metrics. */
export interface MetricCatalog {
  metrics: MetricDescription[];
  analyses: MetricAnalysis[];
  /** Every reason code a response may carry, each once. */
  reasons: ReasonDescription[];
}
