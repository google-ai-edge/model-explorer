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

/** Shared grouped choices for the KV toolbar and Add analysis preview. */
export const KV_VISUALIZATIONS = [
  {label: 'Tensor Comparison · Heatmap', data: 'Metric', chart: 'Heatmap'},
  {label: 'Tensor Comparison · Line chart', data: 'Metric', chart: 'Line chart'},
  {label: 'Tensor Inspection · Line chart', data: 'Tensor', chart: 'Line chart'},
  {label: 'Tensor Inspection · Scatter plot', data: 'Tensor', chart: 'Scatter plot'},
] as const;
export const KV_VISUALIZATION_OPTIONS = KV_VISUALIZATIONS.map((option) => option.label);
export const KV_VISUALIZATION_GROUPS = [
  {label: 'Tensor Comparison', data: 'Metric'},
  {label: 'Tensor Inspection', data: 'Tensor'},
].map(({label, data}) => ({
  label,
  options: KV_VISUALIZATIONS.filter((option) => option.data === data).map((option) => option.label),
}));
export const KV_VISUALIZATION_LABELS = Object.fromEntries(
  KV_VISUALIZATIONS.map((option) => [option.label, option.chart]),
);
