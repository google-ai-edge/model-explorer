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

import {analysisMetricDisplayFormat} from '../../shared/format/format';
import type {ExplorerHelp} from '../../shared/explorer_info_value/explorer_help_term';

/** Descriptions of the metric values, before head/sync-point or position aggregation. */
const METRIC_HELP: Record<string, ExplorerHelp['rows']> = {
  logit_rmse: [
    {label: 'Measures', value: 'Root mean square error across all vocabulary elements'},
    {label: 'Formula', value: 'sqrt(mean((Target − Reference)²))'},
    {label: 'Inputs', value: 'Raw captured final logits, before softmax; no centering'},
    {label: 'Reading', value: '0: identical values; larger errors have more weight'},
  ],
  logit_max_abs: [
    {label: 'Measures', value: 'Largest absolute error across all vocabulary elements'},
    {label: 'Formula', value: 'max(abs(Target − Reference))'},
    {label: 'Inputs', value: 'Raw captured final logits, before softmax; no centering'},
    {label: 'Reading', value: '0: identical values; inspect the corresponding vocabulary index'},
  ],
  cosine_similarity: [
    {label: 'Measures', value: 'Direction similarity of Reference and Target vectors'},
    {label: 'Range', value: '−1 to 1'},
    {label: 'Reading', value: '1: same direction · 0: orthogonal · −1: opposite'},
    {label: 'Unit', value: 'Unitless'},
  ],
  relative_l2: [
    {label: 'Measures', value: 'L2 error divided by the Reference L2 norm'},
    {label: 'Range', value: '0% and above'},
    {label: 'Reading', value: '0%: identical values'},
    {label: 'Unit', value: 'Percent'},
  ],
  max_abs: [
    {label: 'Measures', value: 'Largest absolute element difference'},
    {label: 'Range', value: '0 and above'},
    {label: 'Reading', value: '0: identical values'},
    {label: 'Unit', value: 'Same as tensor values'},
  ],
  kl: [
    {label: 'Measures', value: 'Kullback–Leibler divergence'},
    {label: 'Direction', value: 'Reference → Target'},
    {label: 'Inputs', value: 'Full captured vocabulary logits for the sampled token'},
    {label: 'Transform', value: 'Softmax (T=1), independent of generation settings'},
    {label: 'Filtering', value: 'No top-k, top-p or sampling constraints'},
    {label: 'Reading', value: '0: identical distributions'},
    {label: 'Unit', value: 'Nats'},
  ],
  js: [
    {label: 'Measures', value: 'Jensen–Shannon divergence'},
    {label: 'Inputs', value: 'Full captured vocabulary logits for the sampled token'},
    {label: 'Transform', value: 'Softmax (T=1), independent of generation settings'},
    {label: 'Filtering', value: 'No top-k, top-p or sampling constraints'},
    {label: 'Unit', value: 'Bits'},
    {label: 'Range', value: '0 to 1 bit'},
    {label: 'Reading', value: '0: identical · 1: disjoint support'},
    {label: 'Direction', value: 'Symmetric'},
  ],
  entropy: [
    {label: 'Measures', value: 'Average information; fractional bits are valid'},
    {label: 'Range', value: '0 to log₂(vocabulary size)'},
    {label: 'Reading', value: 'Higher: more spread across tokens'},
    {label: 'Unit', value: 'Bits (base-2 logarithm)'},
    {label: 'Formula', value: 'H = −Σ p(token) log₂ p(token)'},
    {label: '0 bits', value: 'One token has 100% probability'},
    {label: '1 bit', value: 'Two tokens each have 50% probability'},
    {label: '≈0.469 bits', value: 'Two tokens with probabilities 90% and 10%'},
  ],
  selected_probability: [
    {label: 'Measures', value: 'Probability assigned to the selected token'},
    {label: 'Scope', value: 'Each runtime’s output distribution'},
    {label: 'Range', value: '0% to 100%'},
    {label: 'Unit', value: 'Percent'},
  ],
  margin: [
    {label: 'Measures', value: 'Highest minus second-highest token probability'},
    {label: 'Range', value: '0 to 100 percentage points'},
    {label: 'Reading', value: 'Higher: larger separation between top two tokens'},
    {label: 'Unit', value: 'Percentage points'},
  ],
};

export function metricHelp(metric: string, label: string): ExplorerHelp | undefined {
  const rows = METRIC_HELP[metric];
  if (!rows) return undefined;
  const display = analysisMetricDisplayFormat(metric);
  return {
    title: label,
    rows: [
      ...rows,
      {label: 'Precision', value: `${display.precision} (display only)`},
      {label: 'Notation', value: display.notation},
      {label: 'Rounding', value: display.rounding},
    ],
  };
}
