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

import type {TokenAnalysisPair} from '../../data/contracts/token_analysis';
import type {ExplorerInfoItem} from '../../shared/explorer_info_list/explorer_info_list';
import {analysisMetricDisplayFormat} from '../../shared/format/format';

interface SamplingPoint {
  hasRef: boolean;
  hasTarget: boolean;
  analysis?: TokenAnalysisPair;
}

type SamplingSide = NonNullable<TokenAnalysisPair['sampling_distribution']>['ref'];
type SamplingMetric = 'entropy' | 'selected_probability' | 'margin';
type Reason = {label: string; count: number; description: string};

const notCaptured = 'Sampling probabilities not captured';

function presentOn(point: SamplingPoint, side: 'ref' | 'target'): boolean {
  return side === 'ref' ? point.hasRef : point.hasTarget;
}

function available(source: SamplingSide | undefined): boolean {
  return (
    source?.basis === 'reconstructed_greedy' &&
    Number.isInteger(source.candidate_count) &&
    source.candidate_count! > 0
  );
}

/** Each side has its own denominator; alignment gaps are not missing probability captures. */
export function samplingCoverage(
  points: readonly SamplingPoint[],
  side: 'ref' | 'target',
  metric?: SamplingMetric,
) {
  const present = points.filter((point) => presentOn(point, side));
  const reasons = new Map<string, {count: number; descriptions: Set<string>}>();
  let valid = 0;
  for (const point of present) {
    const source = point.analysis?.sampling_distribution?.[side];
    let label: string;
    let description: string;
    if (!point.analysis) {
      label = 'Pending analysis';
      description = 'Token analysis has not returned for these positions.';
    } else if (!available(source)) {
      label = notCaptured;
      description = source?.reason ?? notCaptured;
    } else if (metric && !Number.isFinite(source?.[metric])) {
      label = 'Metric unavailable';
      description = source?.reason ?? 'The selected metric is unavailable for these positions.';
    } else {
      valid++;
      continue;
    }
    const reason = reasons.get(label) ?? {count: 0, descriptions: new Set<string>()};
    reason.count++;
    reason.descriptions.add(description);
    reasons.set(label, reason);
  }
  return {
    valid,
    total: present.length,
    gaps: points.length - present.length,
    reasons: [...reasons].map(
      ([label, reason]): Reason => ({
        label,
        count: reason.count,
        description: [...reason.descriptions].join('; '),
      }),
    ),
  };
}

function valuesText(values: readonly (string | undefined)[]): string {
  return [...new Set(values.map((value) => value ?? 'Not reported'))].join(' · ') || 'Not reported';
}

function uniformValue(values: readonly (string | undefined)[]): string {
  if (!values.length || values.every((value) => value === undefined)) return '—';
  return values.every((value) => value === values[0]) ? values[0]! : 'Mixed';
}

function numberText(value: number | null | undefined): string | undefined {
  return value != null && Number.isFinite(value) ? String(value) : undefined;
}

function metadataItem(
  label: string,
  values: readonly (string | undefined)[],
  description: string,
): ExplorerInfoItem {
  const value = uniformValue(values);
  return {
    label,
    value,
    segments: [
      {
        text: value,
        help: {
          title: label,
          rows: [
            {label: 'Meaning', value: description},
            {label: 'Values', value: valuesText(values)},
          ],
        },
      },
    ],
  };
}

/** Sampler evidence only: the earlier softmax(T=1) distribution is never a fallback. */
export function samplingSourceItems(
  points: readonly SamplingPoint[],
  side: 'ref' | 'target',
): ExplorerInfoItem[] {
  const present = points.filter((point) => presentOn(point, side));
  const sources = present.map((point) => point.analysis?.sampling_distribution?.[side]);
  const coverage = samplingCoverage(points, side);
  const count = `${coverage.valid} / ${coverage.total}`;
  const basis = sources.map((source) => (available(source) ? 'Reconstructed greedy' : undefined));
  const status = !coverage.total
    ? 'No positions'
    : coverage.valid === coverage.total
      ? 'Available'
      : coverage.valid
        ? 'Partially available'
        : present.every((point) => !point.analysis)
          ? 'Pending analysis'
          : notCaptured;
  return [
    {label: 'Status', value: status, expandable: true},
    {
      label: 'Available positions',
      value: count,
      segments: [
        {
          text: count,
          help: {
            title: 'Sampling distribution availability',
            rows: [
              {label: 'Available', value: String(coverage.valid)},
              {label: 'Expected', value: String(coverage.total)},
              {label: 'Counting', value: 'Positions present on this side; alignment gaps excluded'},
              ...(coverage.gaps ? [{label: 'Alignment gaps', value: String(coverage.gaps)}] : []),
              ...coverage.reasons.map((reason) => ({
                label: `${reason.label} (${reason.count})`,
                value: reason.description,
              })),
            ],
          },
        },
      ],
    },
    metadataItem(
      'Basis',
      basis,
      'reconstructed_greedy: a one-hot distribution reconstructed from verified greedy decoding. It is not a captured probability tensor or softmax(T=1).',
    ),
    metadataItem(
      'Vocabulary size',
      sources.map((source) => numberText(source?.size)),
      'Vocabulary size reported with the sampling analysis.',
    ),
    metadataItem(
      'Source forwards',
      sources.map((source) => numberText(source?.source_forward_id)),
      'Source forward IDs for these positions.',
    ),
    ...(
      [
        ['Temperature', 'temperature'],
        ['Top-k', 'top_k'],
        ['Top-p', 'top_p'],
        ['Seed', 'seed'],
      ] as const
    ).map(([label, key]) =>
      metadataItem(
        label,
        sources.map((source) => numberText(source?.sampler?.[key])),
        'Reported sampler configuration; missing settings are not inferred.',
      ),
    ),
  ];
}

/** Distribution properties reported for the selected positions on one runtime. */
export function samplingDistributionMetadata(
  points: readonly SamplingPoint[],
  side: 'ref' | 'target',
): ExplorerInfoItem[] {
  const sources = points
    .filter((point) => presentOn(point, side))
    .map((point) => point.analysis?.sampling_distribution?.[side]);
  const display = (values: readonly (string | undefined)[]) => {
    const value = uniformValue(values);
    return value === 'Mixed' ? `Mixed (${valuesText(values)})` : value === '—' ? '-' : value;
  };
  const parameters = (
    [
      ['Temperature', 'temperature'],
      ['Top-k', 'top_k'],
      ['Top-p', 'top_p'],
      ['Seed', 'seed'],
    ] as const
  ).map(([label, key]) => ({
    label,
    values: sources.map((source) => numberText(source?.sampler?.[key])),
  }));
  const summary = parameters.some(({values}) => values.some((value) => value !== undefined))
    ? parameters.map(({label, values}) => `${label}=${display(values)}`).join(' · ')
    : '-';
  return [
    {
      label: 'Vocabulary size',
      value: display(sources.map((source) => numberText(source?.size))),
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

/** The metric output grid retains both runtime slots, including missing values. */
export function samplingMetricMetadata(
  metric: SamplingMetric,
  range: {start: number; end: number},
  alignment: 'content' | 'steps',
): ExplorerInfoItem[] {
  const count = Math.max(0, range.end - range.start);
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
              {label: 'Count', value: String(count)},
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
    {label: 'Shape', value: `[2, ${count}]`},
    {label: 'Dtype', value: 'float64'},
    {label: 'Display precision', value: analysisMetricDisplayFormat(metric).precision},
  ];
}
