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
import {
  samplingCoverage,
  samplingDistributionMetadata,
  samplingMetricMetadata,
  samplingSourceItems,
} from './sampling_analysis_info';

function analysis(): TokenAnalysisPair {
  const metric = {value: null, reason: null};
  return {
    ref_step: 0,
    target_step: 0,
    metrics: {
      kl: metric,
      js: metric,
      relative_l2: metric,
      cosine_distance: metric,
      norm_ratio: metric,
      max_abs_error: metric,
    },
    distribution: {
      compatible: true,
      reason: null,
      context: 'same',
      ref: {entropy: 2.5, selected_probability: 0.4, margin: 0.1, size: 4},
      target: {entropy: 2.5, selected_probability: 0.4, margin: 0.1, size: 4},
      rows: [],
    },
    sampling_distribution: {
      context: 'same',
      vocabulary_compatible: true,
      rows_complete: true,
      reason: null,
      ref: {
        entropy: 0,
        selected_probability: 1,
        margin: 1,
        size: 4,
        source_forward_id: 0,
        basis: 'reconstructed_greedy',
        candidate_count: 1,
        sampler: {temperature: 0, top_k: 1, top_p: 1, seed: 0},
      },
      target: {
        entropy: 0,
        selected_probability: 1,
        margin: 1,
        size: 4,
        source_forward_id: 0,
        basis: 'reconstructed_greedy',
        candidate_count: 1,
        sampler: {temperature: 0, top_k: 1, top_p: 1, seed: 0},
      },
      rows: [],
    },
  };
}

const point = (record?: TokenAnalysisPair) => ({hasRef: true, hasTarget: true, analysis: record});
const item = (items: ReturnType<typeof samplingSourceItems>, label: string) =>
  items.find((entry) => entry.label === label)!;
const helpRows = (items: ReturnType<typeof samplingSourceItems>, label: string) =>
  item(items, label).segments?.flatMap((segment) => segment.help?.rows ?? []) ?? [];

describe('Sampling distribution coverage', () => {
  it('never uses the earlier T=1 distribution when sampling data is absent', () => {
    const older = analysis();
    delete older.sampling_distribution;
    const points = [point(older)];
    expect(samplingCoverage(points, 'ref', 'entropy')).toMatchObject({
      valid: 0,
      total: 1,
      gaps: 0,
      reasons: [{label: 'Sampling probabilities not captured', count: 1}],
    });
    const items = samplingSourceItems(points, 'ref');
    expect(item(items, 'Status').value).toBe('Sampling probabilities not captured');
    expect(item(items, 'Basis').value).toBe('—');
    expect(item(items, 'Vocabulary size').value).toBe('—');
    expect(item(items, 'Temperature').value).toBe('—');
  });

  it('counts zero entropy as available and excludes side-specific alignment gaps', () => {
    const record = analysis();
    const points = [point(record), {...point(), hasRef: false}, point()];
    expect(samplingCoverage(points, 'ref', 'entropy')).toEqual({
      valid: 1,
      total: 2,
      gaps: 1,
      reasons: [
        {
          label: 'Pending analysis',
          count: 1,
          description: 'Token analysis has not returned for these positions.',
        },
      ],
    });
    expect(samplingCoverage(points, 'target', 'entropy')).toMatchObject({
      valid: 1,
      total: 3,
      gaps: 0,
    });
  });

  it('counts a missing selected-token metric separately from an available distribution', () => {
    const record = analysis();
    record.sampling_distribution!.ref.selected_probability = null;
    record.sampling_distribution!.ref.reason = 'Selected token is outside the vocabulary';
    expect(samplingCoverage([point(record)], 'ref')).toMatchObject({valid: 1, reasons: []});
    expect(samplingCoverage([point(record)], 'ref', 'entropy')).toMatchObject({valid: 1});
    expect(samplingCoverage([point(record)], 'ref', 'selected_probability')).toMatchObject({
      valid: 0,
      reasons: [
        {
          label: 'Metric unavailable',
          count: 1,
          description: 'Selected token is outside the vocabulary',
        },
      ],
    });
  });

  it('preserves unavailable sampler evidence and reasons without inventing probabilities', () => {
    const record = analysis();
    record.sampling_distribution!.ref = {
      basis: null,
      candidate_count: null,
      sampler: {temperature: 0.8, top_k: 40, top_p: 0.9, seed: null},
      reason: 'Stochastic sampling probabilities were not captured',
    };
    expect(samplingCoverage([point(record)], 'ref', 'entropy')).toMatchObject({
      valid: 0,
      total: 1,
      reasons: [
        {
          label: 'Sampling probabilities not captured',
          count: 1,
          description: 'Stochastic sampling probabilities were not captured',
        },
      ],
    });
    const items = samplingSourceItems([point(record)], 'ref');
    expect(item(items, 'Temperature').value).toBe('0.8');
    expect(item(items, 'Top-k').value).toBe('40');
    expect(item(items, 'Seed').value).toBe('—');
    expect(item(items, 'Available positions').value).toBe('0 / 1');
  });
});

describe('Sampling source metadata', () => {
  it('reports mixed settings and preserves all reported values in help', () => {
    const first = analysis();
    const second = analysis();
    second.sampling_distribution!.ref.source_forward_id = 3;
    second.sampling_distribution!.ref.sampler!.temperature = 0.2;
    second.sampling_distribution!.ref.sampler!.seed = null;
    const items = samplingSourceItems([point(first), point(second)], 'ref');
    expect(item(items, 'Status').value).toBe('Available');
    expect(item(items, 'Available positions').value).toBe('2 / 2');
    expect(item(items, 'Basis').value).toBe('Reconstructed greedy');
    expect(item(items, 'Vocabulary size').value).toBe('4');
    expect(item(items, 'Temperature').value).toBe('Mixed');
    expect(helpRows(items, 'Temperature')).toContainEqual({label: 'Values', value: '0 · 0.2'});
    expect(item(items, 'Source forwards').value).toBe('Mixed');
    expect(helpRows(items, 'Source forwards')).toContainEqual({label: 'Values', value: '0 · 3'});
    expect(item(items, 'Seed').value).toBe('Mixed');
    expect(helpRows(items, 'Seed')).toContainEqual({label: 'Values', value: '0 · Not reported'});
  });

  it('keeps valid side metadata across different contexts and partial availability', () => {
    const record = analysis();
    record.sampling_distribution!.context = 'different';
    record.sampling_distribution!.vocabulary_compatible = false;
    record.sampling_distribution!.target = {
      basis: null,
      candidate_count: null,
      sampler: null,
      reason: 'No final probabilities available',
    };
    const points = [point(record), point()];
    const items = samplingSourceItems(points, 'ref');
    expect(item(items, 'Status').value).toBe('Partially available');
    expect(item(items, 'Available positions').value).toBe('1 / 2');
    expect(samplingCoverage(points, 'ref', 'margin').valid).toBe(1);
    expect(samplingCoverage(points, 'target', 'margin').valid).toBe(0);
    expect(helpRows(items, 'Available positions')).toContainEqual({
      label: 'Pending analysis (1)',
      value: 'Token analysis has not returned for these positions.',
    });
  });
});

describe('Distribution and metric metadata', () => {
  it('shows only vocabulary size and sampling parameters, preserving zero configuration values', () => {
    const first = analysis();
    first.sampling_distribution!.ref.logits_metadata = {
      source_shape: [1, 32, 4],
      comparison_shape: [4],
      source_dtype: 'float16',
    };
    const items = samplingDistributionMetadata([point(first), {...point(), hasRef: false}], 'ref');
    expect(items.map(({label, value}) => [label, value])).toEqual([
      ['Vocabulary size', '4'],
      ['Sampling parameters', 'Temperature=0 · Top-k=1 · Top-p=1 · Seed=0'],
    ]);
    expect(items.every((item) => item.segments === undefined)).toBe(true);
    expect(JSON.stringify(items)).not.toMatch(
      /Dimensions|Shape|Dtype|Source forwards|Basis|Available positions|Status|float16/,
    );
  });

  it('keeps missing sampling properties unknown without falling back to the logits distribution', () => {
    const older = analysis();
    delete older.sampling_distribution;
    const items = samplingDistributionMetadata([point(older)], 'ref');
    expect(items.map(({label, value}) => [label, value])).toEqual([
      ['Vocabulary size', '-'],
      ['Sampling parameters', '-'],
    ]);
    expect(items.every((item) => item.segments === undefined)).toBe(true);
  });

  it('retains reported settings for unsupported probabilities and mixed values across positions', () => {
    const first = analysis();
    const second = analysis();
    second.sampling_distribution!.ref = {
      basis: null,
      candidate_count: null,
      sampler: {temperature: 0.8, top_k: 40, top_p: 0.9, seed: null},
      source_forward_id: 3,
      size: 8,
      reason: 'Stochastic sampling probabilities were not captured',
    };
    const unsupported = samplingDistributionMetadata([point(second)], 'ref');
    expect(item(unsupported, 'Vocabulary size').value).toBe('8');
    expect(item(unsupported, 'Sampling parameters').value).toBe(
      'Temperature=0.8 · Top-k=40 · Top-p=0.9 · Seed=-',
    );
    const mixed = samplingDistributionMetadata(
      [point(first), point(second), point(), {...point(), hasRef: false}],
      'ref',
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

  it('describes the selected output grid with plain shape, dtype and display precision', () => {
    const items = samplingMetricMetadata('entropy', {start: 100, end: 103}, 'content');
    expect(items.map(({label, value}) => [label, value])).toEqual([
      ['Dimensions', '[Runtime, Token position]'],
      ['Shape', '[2, 3]'],
      ['Dtype', 'float64'],
      ['Display precision', '6 significant digits'],
    ]);
    expect(helpRows(items, 'Dimensions')).toContainEqual({label: 'Alignment', value: 'Context'});
    expect(helpRows(items, 'Dimensions')).toContainEqual({label: 'Range', value: '100–102'});
    expect(items.slice(1).every((item) => item.segments === undefined)).toBe(true);
  });

  it.each(['selected_probability', 'margin'] as const)(
    'uses the same plain fields for single-position %s',
    (metric) => {
      const items = samplingMetricMetadata(metric, {start: 9, end: 10}, 'steps');
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
