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

import type {MetricLookup} from '../../server/metric_catalog';
import {metricDisplayRules} from '../../widgets/format/format';
import type {ExplorerHelp} from '../../widgets/explorer_info_value/explorer_help_term';

/** What a metric is, as the server describes it, before head, sync-point or position
 *  aggregation; undefined while the catalog has no description of it. */
export function metricHelp(
  catalog: MetricLookup,
  metric: string,
  analysis: string,
  label: string,
): ExplorerHelp | undefined {
  const description = catalog.metric(metric);
  if (!description) {
    return undefined;
  }
  const display = metricDisplayRules(description);
  const inputs = catalog.inputs(metric, analysis);
  return {
    title: label,
    rows: [
      {label: 'Measures', value: description.measures},
      {label: 'Formula', value: description.formula},
      ...(inputs ? [{label: 'Inputs', value: inputs}] : []),
      {label: 'Range', value: description.range},
      {label: 'Reading', value: description.reading},
      {label: 'Unit', value: description.unit},
      {label: 'Precision', value: `${display.precision} (display only)`},
      {label: 'Notation', value: display.notation},
      {label: 'Rounding', value: display.rounding},
    ],
  };
}
