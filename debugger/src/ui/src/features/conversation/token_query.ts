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

import {
  compileMetricQuery,
  type MetricQuery,
  type MetricQueryLanguage,
  type Truth,
} from '../../shared/metric_query/metric_query';

/** Token-analysis fields; relative_l2 is already a percentage, so `0.4%` means 0.4. */
export const queryFields: Record<string, 'number' | 'boolean'> = {
  token_match: 'boolean',
  js: 'number',
  kl: 'number',
  relative_l2: 'number',
  cosine_distance: 'number',
  cosine_similarity: 'number',
  norm_ratio: 'number',
  max_abs_error: 'number',
};
const TOKEN_QUERY_LANGUAGE: MetricQueryLanguage = {
  fields: Object.fromEntries(
    Object.entries(queryFields).map(([name, type]) => [name, {type}]),
  ),
  percentField: 'relative_l2',
  percentScale: 1,
};
export type TokenQueryRow = Record<string, number | boolean | null | undefined>;
export type TokenQuery = MetricQuery<TokenQueryRow>;
export type {Truth};
/** The default filter is `NOT token_match`; a blank formula is an error, not match-all. */
export function compileQuery(source: string): TokenQuery {
  if (!source.trim())
    throw new Error('Enter a formula, or reset to NOT token_match');
  return compileMetricQuery<TokenQueryRow>(source, TOKEN_QUERY_LANGUAGE);
}
