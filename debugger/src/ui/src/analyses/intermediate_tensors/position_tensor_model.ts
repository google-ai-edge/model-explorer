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

/** The field a position-analysis row named each metric by before rows named them by the metric
 *  catalog's id, by that id: a view saved then names its metric by the field. */
export const TENSOR_METRIC_SOURCES: Readonly<Record<string, string>> = {
  cosine_similarity: 'CosSim',
  relative_l2: 'Relative L2',
  max_abs: 'Max abs error',
};
/** The catalog's id of a saved metric; older views name one by its row field. */
export const tensorMetricId = (metric: string): string =>
  Object.entries(TENSOR_METRIC_SOURCES).find(([, source]) => source === metric)?.[0] ?? metric;
