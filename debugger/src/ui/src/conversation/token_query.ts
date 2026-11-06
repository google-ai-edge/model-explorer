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
} from './metric_query/metric_query';
import {TOKEN_METRIC_KEYS} from '../server/contracts/token_analysis';
import type {MetricLookup} from '../server/metric_catalog';

/** Token-analysis fields: the match and every metric of a token pair. */
export const queryFields: Record<string, 'number' | 'boolean'> = {
  token_match: 'boolean',
  ...Object.fromEntries(TOKEN_METRIC_KEYS.map((key) => [key, 'number' as const])),
};
export const TOKEN_QUERY_EXAMPLES = [
  {formula: 'NOT token_match', description: 'Find mismatched tokens.'},
  {
    formula: 'kl > 0.005 AND relative_l2 > 0.4%',
    description: 'KL above 0.005 nats and relative L2 above 0.4%.',
  },
];
/**
 * The query language as the catalog describes the metrics: `%` goes with the metric whose unit
 * is percent, whose values already are percentages, so `0.4%` means 0.4.
 */
function tokenQueryLanguage(catalog: MetricLookup | null): MetricQueryLanguage {
  return {
    fields: Object.fromEntries(Object.entries(queryFields).map(([name, type]) => [name, {type}])),
    percentField: TOKEN_METRIC_KEYS.find((key) => catalog?.metric(key)?.unit === 'percent') ?? null,
    percentScale: 1,
  };
}
export type TokenQueryRow = Record<string, number | boolean | null | undefined>;
export type TokenQuery = MetricQuery<TokenQueryRow>;
export type {Truth};
/**
 * The default filter is `NOT token_match`; a blank formula is an error, not match-all. With no
 * catalog, no metric takes `%`.
 */
export function compileQuery(source: string, catalog: MetricLookup | null): TokenQuery {
  if (!source.trim()) {
    throw new Error('Enter a formula, or reset to NOT token_match');
  }
  return compileMetricQuery<TokenQueryRow>(source, tokenQueryLanguage(catalog));
}
/**
 * Whether `source` would compile against a catalog that says one of the metrics is a percentage:
 * a formula that does not compile without the catalog, but this way, waits for the catalog.
 */
export function compilesWithSomeCatalog(source: string): boolean {
  if (!source.trim()) {
    return false;
  }
  return TOKEN_METRIC_KEYS.some((key) => {
    try {
      compileMetricQuery<TokenQueryRow>(source, {...tokenQueryLanguage(null), percentField: key});
      return true;
    } catch {
      return false;
    }
  });
}
/**
 * The formula as compileQuery() compiles it; one it cannot compile against `catalog` (a `%`
 * before the catalog has loaded) finds nothing, rather than throwing.
 */
export function compileQueryOrNothing(source: string, catalog: MetricLookup | null): TokenQuery {
  try {
    return compileQuery(source, catalog);
  } catch {
    return Object.assign(() => null, {requiresMetrics: false});
  }
}
