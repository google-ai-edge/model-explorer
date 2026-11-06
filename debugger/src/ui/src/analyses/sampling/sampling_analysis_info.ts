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

import type {LogicalTensor} from '../../server/contracts/kv';
import type {TokenRangeSideMetadata} from '../../server/contracts/token_analysis';
import type {ExplorerInfoItem} from '../../widgets/explorer_info_list/explorer_info_list';
import type {MetricDescription} from '../../server/contracts/metrics';
import {metricDisplayRules} from '../../widgets/format/format';

function valuesText(values: readonly (string | undefined)[]): string {
  return [...new Set(values.map((value) => value ?? 'Not reported'))].join(' · ') || 'Not reported';
}

/** Distribution properties reported for the selected positions on one runtime, as the server
 *  lists their distinct values. */
export function samplingDistributionMetadata(metadata: TokenRangeSideMetadata): ExplorerInfoItem[] {
  const display = (values: readonly (number | null)[]): string => {
    if (values.every((value) => value === null)) {
      return '-';
    }
    const distinct = [...new Set(values)];
    return distinct.length === 1
      ? String(distinct[0])
      : `Mixed (${valuesText(distinct.map((value) => (value === null ? undefined : String(value))))})`;
  };
  const parameters = (
    [
      ['Temperature', 'temperature'],
      ['Top-k', 'top_k'],
      ['Top-p', 'top_p'],
      ['Seed', 'seed'],
    ] as const
  ).map(([label, key]) => ({label, values: metadata.sampler?.[key] ?? []}));
  const summary = parameters.some(({values}) => values.some((value) => value !== null))
    ? parameters.map(({label, values}) => `${label}=${display(values)}`).join(' · ')
    : '-';
  return [
    {
      label: 'Vocabulary size',
      value: display(metadata.vocabulary_sizes),
      expandable: true,
      wrap: true,
    },
    {
      label: 'Sampling parameters',
      value: summary,
      expandable: true,
    },
  ];
}

/** The metric output grid retains both runtime slots, including missing values; its shape and
 *  dtype are the server's. */
export function samplingMetricMetadata(
  metric: MetricDescription | undefined,
  range: {start: number; end: number},
  alignment: 'content' | 'steps',
  output: LogicalTensor | null,
): ExplorerInfoItem[] {
  const shape = output?.shape ?? null;
  const count = shape ? String(shape[1]) : '-';
  return [
    {
      label: 'Dimensions',
      value: '[Runtime, Token position]',
      segments: [
        {text: '['},
        {
          text: 'Runtime',
          help: {
            title: 'Runtime',
            rows: [
              {label: 'Order', value: 'Reference, Target'},
              {label: 'Count', value: '2'},
              {
                label: 'Meaning',
                value: 'Each runtime uses its own sampling distribution and selected token',
              },
            ],
          },
        },
        {text: ', '},
        {
          text: 'Token position',
          help: {
            title: 'Token position',
            rows: [
              {label: 'Alignment', value: alignment === 'content' ? 'Context' : 'Steps'},
              {label: 'Range', value: `${range.start}–${range.end - 1}`},
              {label: 'Count', value: count},
              {
                label: 'Counting',
                value:
                  'All selected chart coordinate slots, including unavailable values and alignment gaps',
              },
            ],
          },
        },
        {text: ']'},
      ],
    },
    {label: 'Shape', value: shape ? `[${shape.join(', ')}]` : '—'},
    {label: 'Dtype', value: typeof output?.dtype === 'string' ? output.dtype : '—'},
    {label: 'Display precision', value: metricDisplayRules(metric).precision},
  ];
}
