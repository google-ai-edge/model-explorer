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

import type {TokenRangeSideMetadata} from '../../server/contracts/token_analysis';
import {samplingDistributionMetadata, samplingMetricMetadata} from './sampling_analysis_info';
import {fixtureMetric} from '../../server/testing/metric_catalog_fixture';

function item(items: ReturnType<typeof samplingMetricMetadata>, label: string) {
  return items.find((entry) => entry.label === label)!;
}
function helpRows(items: ReturnType<typeof samplingMetricMetadata>, label: string) {
  return item(items, label).segments?.flatMap((segment) => segment.help?.rows ?? []) ?? [];
}

describe('Distribution and metric metadata', () => {
  /** The server's lists of one side's distinct vocabulary sizes and sampler settings. */
  const side = (
    sizes: (number | null)[],
    sampler: Partial<NonNullable<TokenRangeSideMetadata['sampler']>> = {},
  ): TokenRangeSideMetadata => ({
    vocabulary_sizes: sizes,
    sampler: {temperature: [0], top_k: [1], top_p: [1], seed: [0], ...sampler},
  });
  const grid = (count: number) => ({
    dimensions: ['Runtime', 'Position'],
    shape: [2, count],
    dtype: 'float64',
  });

  it('shows only vocabulary size and sampling parameters, preserving zero configuration values', () => {
    const items = samplingDistributionMetadata(side([4]));
    expect(items.map(({label, value}) => [label, value])).toEqual([
      ['Vocabulary size', '4'],
      ['Sampling parameters', 'Temperature=0 · Top-k=1 · Top-p=1 · Seed=0'],
    ]);
    const repeated = samplingDistributionMetadata(
      side([32000, 32000], {
        temperature: [0.8, 0.8],
        top_k: [40, 40],
        top_p: [0.9, 0.9],
        seed: [0, 0],
      }),
    );
    expect(repeated.map(({label, value}) => [label, value])).toEqual([
      ['Vocabulary size', '32000'],
      ['Sampling parameters', 'Temperature=0.8 · Top-k=40 · Top-p=0.9 · Seed=0'],
    ]);
    expect(items.every((entry) => entry.segments === undefined)).toBe(true);
    expect(JSON.stringify(items)).not.toMatch(
      /Dimensions|Shape|Dtype|Source forwards|Basis|Available positions|Status|float16/,
    );
  });

  it('keeps missing sampling properties unknown', () => {
    const items = samplingDistributionMetadata(
      side([null], {temperature: [null], top_k: [null], top_p: [null], seed: [null]}),
    );
    expect(items.map(({label, value}) => [label, value])).toEqual([
      ['Vocabulary size', '-'],
      ['Sampling parameters', '-'],
    ]);
    expect(samplingDistributionMetadata({vocabulary_sizes: []})[1].value).toBe('-');
  });

  it('writes the settings that differ across positions as Mixed', () => {
    const unsupported = samplingDistributionMetadata(
      side([8], {temperature: [0.8], top_k: [40], top_p: [0.9], seed: [null]}),
    );
    expect(unsupported[0].value).toBe('8');
    expect(unsupported[1].value).toBe('Temperature=0.8 · Top-k=40 · Top-p=0.9 · Seed=-');
    const mixed = samplingDistributionMetadata(
      side([4, 8, null], {
        temperature: [0, 0.8, null],
        top_k: [1, 40, null],
        top_p: [1, 0.9, null],
        seed: [0, null],
      }),
    );
    expect(mixed.map(({label, value}) => [label, value])).toEqual([
      ['Vocabulary size', 'Mixed (4 · 8 · Not reported)'],
      [
        'Sampling parameters',
        'Temperature=Mixed (0 · 0.8 · Not reported) · Top-k=Mixed (1 · 40 · Not reported) · Top-p=Mixed (1 · 0.9 · Not reported) · Seed=Mixed (0 · Not reported)',
      ],
    ]);
    expect(mixed.every((item) => item.segments === undefined)).toBe(true);
  });

  it('describes the selected output grid with the server shape, dtype and display precision', () => {
    const items = samplingMetricMetadata(
      fixtureMetric('entropy'),
      {start: 100, end: 103},
      'content',
      grid(3),
    );
    expect(items.map(({label, value}) => [label, value])).toEqual([
      ['Dimensions', '[Runtime, Token position]'],
      ['Shape', '[2, 3]'],
      ['Dtype', 'float64'],
      ['Display precision', '6 significant digits'],
    ]);
    expect(helpRows(items, 'Dimensions')).toContainEqual({label: 'Alignment', value: 'Context'});
    expect(helpRows(items, 'Dimensions')).toContainEqual({label: 'Range', value: '100–102'});
    expect(items.slice(1).every((item) => item.segments === undefined)).toBe(true);
    const loading = samplingMetricMetadata(
      fixtureMetric('entropy'),
      {start: 0, end: 1},
      'content',
      null,
    );
    expect(item(loading, 'Shape').value).toBe('—');
    expect(item(loading, 'Dtype').value).toBe('—');
  });

  it.each(['selected_probability', 'margin'] as const)(
    'uses the same plain fields for single-position %s',
    (metric) => {
      const items = samplingMetricMetadata(
        fixtureMetric(metric),
        {start: 9, end: 10},
        'steps',
        grid(1),
      );
      expect(item(items, 'Shape')).toEqual({label: 'Shape', value: '[2, 1]'});
      expect(item(items, 'Dtype')).toEqual({label: 'Dtype', value: 'float64'});
      expect(item(items, 'Display precision')).toEqual({
        label: 'Display precision',
        value: '2 decimal places',
      });
      expect(helpRows(items, 'Dimensions')).toContainEqual({label: 'Alignment', value: 'Steps'});
      expect(helpRows(items, 'Dimensions')).toContainEqual({label: 'Count', value: '1'});
    },
  );
});
