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

import type {TokenAnalysisPair, TokenDistributionSide} from '../../data/contracts/token_analysis';
import type {ExplorerInfoItem} from '../../shared/explorer_info_list/explorer_info_list';

interface LogitPoint {
  hasRef: boolean;
  hasTarget: boolean;
  analysis?: TokenAnalysisPair;
}

type MissingReason = {label: string; count: number; description: string};

const reasons = {
  gap: ['Alignment gap', 'The current alignment has no token on one side.'],
  pending: ['Pending analysis', 'Token analysis has not returned for these positions.'],
  context: [
    'Different / unproven context',
    'Matching token contexts are not established. Each side may still have complete logits.',
  ],
  vocabulary: [
    'Vocabulary not comparable',
    'Matching captured vocabulary identities and sizes are not established.',
  ],
  infinite: [
    'Infinite divergence',
    'The divergence is infinite and is excluded from finite Min, Max, and Mean statistics.',
  ],
  logits: ['Logits unavailable', 'A usable full-vocabulary logits distribution is unavailable.'],
  metric: ['Metric unavailable', 'The selected metric could not be calculated.'],
} as const;

function hasLogits(side: TokenDistributionSide): boolean {
  return !!side.logits_metadata || (Number.isFinite(side.size) && side.size! > 0);
}

/** Count eligibility for the selected metric, independently of each side's source availability. */
export function logitCoverage(points: readonly LogitPoint[], metric: 'kl' | 'js') {
  let valid = 0;
  const counts = new Map<keyof typeof reasons, number>();
  const details = new Map<keyof typeof reasons, Set<string>>();
  for (const point of points) {
    const analysis = point.analysis;
    let kind: keyof typeof reasons;
    if (!point.hasRef || !point.hasTarget) kind = 'gap';
    else if (!analysis) kind = 'pending';
    else if (Number.isFinite(analysis.metrics[metric].value)) {
      valid++;
      continue;
    } else {
      const detail = [analysis.metrics[metric].reason, analysis.distribution.reason]
        .filter(Boolean)
        .join(' ');
      if (analysis.distribution.context === 'different' || /context|diverged/i.test(detail)) {
        kind = 'context';
      } else if (
        /(?:matching|mismatched|incompatible).*vocab|vocab.*(?:mismatch|identit|ids|sizes|differ)/i.test(
          detail,
        )
      ) {
        kind = 'vocabulary';
      } else if (/infinit/i.test(detail) || analysis.metrics[metric].value === Infinity) {
        kind = 'infinite';
      } else if (
        !hasLogits(analysis.distribution.ref) ||
        !hasLogits(analysis.distribution.target)
      ) {
        kind = 'logits';
      } else kind = 'metric';
    }
    counts.set(kind, (counts.get(kind) ?? 0) + 1);
    if (analysis && kind !== 'gap') {
      const reported = [analysis.metrics[metric].reason, analysis.distribution.reason];
      if (kind === 'logits') {
        reported.push(
          analysis.distribution.ref.reason ?? null,
          analysis.distribution.target.reason ?? null,
        );
      }
      const descriptions = details.get(kind) ?? new Set<string>();
      for (const description of reported) if (description) descriptions.add(description);
      details.set(kind, descriptions);
    }
  }
  return {
    valid,
    total: points.length,
    reasons: Object.entries(reasons).flatMap(([kind, [label, description]]): MissingReason[] => {
      const count = counts.get(kind as keyof typeof reasons) ?? 0;
      const reported = [...(details.get(kind as keyof typeof reasons) ?? [])];
      return count
        ? [
            {
              label,
              count,
              description: `${description}${reported.length ? ` Reported: ${reported.join('; ')}` : ''}`,
            },
          ]
        : [];
    }),
  };
}

function valuesText(values: (string | undefined)[]) {
  return [...new Set(values.map((value) => value ?? 'Not reported'))].join(' · ') || 'Not reported';
}

function uniformValue(values: (string | undefined)[]) {
  if (!values.length || values.every((value) => value === undefined)) return '—';
  return values.every((value) => value === values[0])
    ? values[0]!
    : `Mixed (${valuesText(values)})`;
}

/** Logical input geometry, like KV metadata: one captured vocabulary vector per position. */
export function logitSourceItems(
  points: readonly LogitPoint[],
  side: 'ref' | 'target',
): ExplorerInfoItem[] {
  const present = points.filter((point) => (side === 'ref' ? point.hasRef : point.hasTarget));
  const analyzed = present.flatMap((point) => (point.analysis ? [point.analysis] : []));
  const sources = analyzed.map((analysis) => analysis.distribution[side]);
  const available = sources.filter(hasLogits);
  const vocabularySizes = available.map((source) => {
    const dimensions = source.logits_metadata?.comparison_shape;
    const size = dimensions ? (dimensions.length === 1 ? dimensions[0] : undefined) : source.size;
    return Number.isInteger(size) && size! > 0 ? String(size) : undefined;
  });
  const shape = available.length ? `[${available.length}, ${uniformValue(vocabularySizes)}]` : '—';
  const dtypes = available.map((source) => source.logits_metadata?.source_dtype);
  const dtype = uniformValue(dtypes);
  return [
    {
      label: 'Dimensions',
      value: '[Token position, Vocabulary]',
      segments: [
        {text: '['},
        {
          text: 'Token position',
          help: {
            title: 'Token position',
            rows: [
              {label: 'Count', value: String(available.length)},
              {
                label: 'Counting',
                value: 'Captured logits vectors; alignment gaps and unavailable inputs excluded',
              },
            ],
          },
        },
        {text: ', '},
        {
          text: 'Vocabulary',
          help: {
            title: 'Vocabulary',
            rows: [
              {label: 'Dimension', value: 'Vocabulary token IDs in each captured logits vector'},
              {label: 'Size', value: valuesText(vocabularySizes)},
              {
                label: 'Identity',
                value: valuesText(
                  available.map((source) => source.logits_metadata?.vocab_identity),
                ),
              },
            ],
          },
        },
        {text: ']'},
      ],
    },
    {label: 'Shape', value: shape, expandable: true},
    {label: 'Dtype', value: dtype, expandable: true},
  ];
}
