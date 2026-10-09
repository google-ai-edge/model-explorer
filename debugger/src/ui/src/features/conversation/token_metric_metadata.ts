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

import type {TokenMetricKey} from '../../data/contracts/token_analysis';
export interface TokenMetricDefinition {
  key: TokenMetricKey;
  label: string;
  unit: string;
  description: string;
  /** Which way the value points, shown beside the unit of a trend chart. */
  reading: string;
  /** How a recorded value is shown; the Server records cosine distance, the UI shows similarity. */
  present?: (value: number) => number;
}
/** Units and field descriptions inherited from candidates Formula reference. */
export const TOKEN_METRICS: readonly TokenMetricDefinition[] = [
  {
    key: 'kl',
    label: 'Logits · Kullback–Leibler divergence',
    unit: 'nats',
    description: 'Kullback–Leibler divergence from Reference to Target.',
    reading: '0 = identical distributions',
  },
  {
    key: 'js',
    label: 'Logits · Jensen–Shannon divergence',
    unit: 'bits (0–1)',
    description:
      'Symmetric difference between Reference and Target distributions.',
    reading: '0 = identical, 1 = disjoint',
  },
  {
    key: 'relative_l2',
    label: 'Activation · Relative L2',
    unit: 'percent; 1 = 1%',
    description: 'L2 error relative to the Reference tensor norm.',
    reading: '0% = identical tensors',
  },
  {
    key: 'cosine_distance',
    label: 'Activation · Cosine similarity',
    unit: 'unitless (1 = identical)',
    description:
      'Cosine similarity between the paired tensors (shown as one minus the recorded distance).',
    reading: '1 = same direction',
    present: (value) => 1 - value,
  },
  {
    key: 'norm_ratio',
    label: 'Activation · Norm ratio',
    unit: '1 = equal norms',
    description: 'Target tensor norm divided by the Reference tensor norm.',
    reading: '1 = equal norms',
  },
  {
    key: 'max_abs_error',
    label: 'Activation · Max absolute error',
    unit: 'tensor units',
    description: 'Largest absolute element difference in the paired tensors.',
    reading: '0 = identical tensors',
  },
];
export const TOKEN_FIELD_HELP: Record<string, string> = {
  token_match: 'Token · Match — true: match; false: mismatch',
  ...Object.fromEntries(
    TOKEN_METRICS.map((m) => [m.key, m.label + ' — ' + m.unit]),
  ),
  cosine_distance:
    'Activation · Cosine distance — unitless (0 = identical); 1 − cosine_similarity',
  cosine_similarity:
    'Activation · Cosine similarity — unitless (1 = identical)',
};
export const TOKEN_QUERY_EXAMPLES = [
  {formula: 'NOT token_match', description: 'Find mismatched tokens.'},
  {
    formula: 'kl > 0.005 AND relative_l2 > 0.4%',
    description: 'KL above 0.005 nats and relative L2 above 0.4%.',
  },
];
