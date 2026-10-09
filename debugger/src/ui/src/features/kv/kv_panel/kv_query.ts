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
  type MetricQueryLanguage,
} from '../../../shared/metric_query/metric_query';

/** All metrics use raw values; relative_l2 is a ratio, so 10% means 0.1. */
export interface KvQueryMetrics {
  relative_l2: number | null;
  max_abs: number | null;
  cosine_distance: number | null;
}
const KV_QUERY_LANGUAGE: MetricQueryLanguage = {
  fields: {
    relative_l2: {type: 'number'},
    max_abs: {type: 'number'},
    max_abs_delta: {type: 'number', key: 'max_abs'},
    cosine_distance: {type: 'number'},
  },
  percentField: 'relative_l2',
  percentScale: 0.01,
};
/** Mirrors kv_formula.py on the server, which does the actual filtering; only true matches. */
export function compileKvQuery(
  expression: string,
): (metrics: KvQueryMetrics) => boolean {
  const query = compileMetricQuery<KvQueryMetrics>(
    expression,
    KV_QUERY_LANGUAGE,
  );
  return (metrics) => query(metrics) === true;
}
