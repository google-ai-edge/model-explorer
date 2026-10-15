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
import {logitCoverage, logitSourceItems} from './logit_analysis_info';

function analysis(): TokenAnalysisPair {
  return {
    ref_step: 0,
    target_step: 0,
    basis: 'recorded',
    metrics: {
      kl: {value: 0.2, reason: null},
      js: {value: 0.1, reason: null},
      relative_l2: {value: null, reason: null},
      cosine_distance: {value: null, reason: null},
      norm_ratio: {value: null, reason: null},
      max_abs_error: {value: null, reason: null},
    },
    distribution: {
      compatible: true,
      reason: null,
      context: 'same',
      ref: {size: 4, source_forward_id: 0},
      target: {size: 4, source_forward_id: 0},
      rows: [],
    },
  };
}

const point = (record?: TokenAnalysisPair) => ({hasRef: true, hasTarget: true, analysis: record});
const item = (items: ReturnType<typeof logitSourceItems>, label: string) =>
  items.find((entry) => entry.label === label)!;
const helpRows = (items: ReturnType<typeof logitSourceItems>, label: string) =>
  item(items, label).segments?.flatMap((segment) => segment.help?.rows ?? []) ?? [];

describe('Logit comparison coverage', () => {
  it('separates alignment gaps, pending analysis, and different contexts with captured logits', () => {
    const different = analysis();
    different.distribution.context = 'different';
    different.distribution.compatible = false;
    different.metrics.kl = {
      value: null,
      reason: 'Generated histories have diverged or are unproven',
    };
    const points = [{...point(), hasTarget: false}, point(), point(different), point(analysis())];
    expect(logitCoverage(points, 'kl')).toEqual({
      valid: 1,
      total: 4,
      reasons: [
        expect.objectContaining({label: 'Alignment gap', count: 1}),
        expect.objectContaining({label: 'Pending analysis', count: 1}),
        expect.objectContaining({label: 'Different / unproven context', count: 1}),
      ],
    });
    expect(item(logitSourceItems([point(different)], 'ref'), 'Shape').value).toBe('[1, 4]');
  });

  it('counts finite JS independently of infinite KL', () => {
    const infinite = analysis();
    infinite.metrics.kl = {
      value: null,
      reason: 'Divergence is infinite: target assigns zero probability',
    };
    expect(logitCoverage([point(infinite)], 'kl')).toMatchObject({
      valid: 0,
      reasons: [{label: 'Infinite divergence', count: 1}],
    });
    expect(logitCoverage([point(infinite)], 'js')).toEqual({valid: 1, total: 1, reasons: []});
  });

  it('distinguishes vocabulary, unusable logits, and otherwise unavailable metrics', () => {
    const vocabulary = analysis();
    vocabulary.metrics.kl = {
      value: null,
      reason: 'Matching captured vocabulary IDs and sizes are required',
    };
    const invalid = analysis();
    invalid.metrics.kl = {value: null, reason: 'Logits contain invalid values'};
    invalid.distribution.ref = {reason: 'Logits contain invalid values'};
    const failed = analysis();
    failed.metrics.kl = {value: null, reason: 'Calculation failed'};
    const notCaptured = analysis();
    notCaptured.metrics.kl = {value: null, reason: 'Full-vocabulary logits not captured'};
    notCaptured.distribution.ref = {reason: 'Full-vocabulary logits not captured'};
    expect(
      logitCoverage([point(vocabulary), point(invalid), point(failed), point(notCaptured)], 'kl')
        .reasons,
    ).toEqual([
      expect.objectContaining({label: 'Vocabulary not comparable', count: 1}),
      expect.objectContaining({label: 'Logits unavailable', count: 2}),
      expect.objectContaining({label: 'Metric unavailable', count: 1}),
    ]);
    expect(item(logitSourceItems([point(invalid)], 'ref'), 'Shape').value).toBe('—');
    expect(logitCoverage([point(failed)], 'kl').reasons[0].description).toContain(
      'Calculation failed',
    );
  });
});

describe('Logit source metadata', () => {
  it('shows mixed captured dtypes directly while keeping the logical vocabulary geometry', () => {
    const prefill = analysis();
    prefill.distribution.ref.logits_metadata = {
      source_shape: [1, 32, 4],
      source_dtype: 'float16',
      comparison_shape: [4],
      vocab_identity: 'vocab-A',
    };
    const decode = analysis();
    decode.distribution.ref.logits_metadata = {
      source_shape: [1, 1, 4],
      source_dtype: 'float32',
      comparison_shape: [4],
      vocab_identity: 'vocab-A',
    };
    const items = logitSourceItems([point(prefill), point(decode)], 'ref');
    expect(items.map(({label, value}) => [label, value])).toEqual([
      ['Dimensions', '[Token position, Vocabulary]'],
      ['Shape', '[2, 4]'],
      ['Dtype', 'Mixed (float16 · float32)'],
    ]);
    expect(item(items, 'Shape').segments).toBeUndefined();
    expect(item(items, 'Dtype').segments).toBeUndefined();
    expect(helpRows(items, 'Dimensions')).toContainEqual({label: 'Identity', value: 'vocab-A'});
  });

  it('shows varying vocabulary sizes directly without adding batch or sequence dimensions', () => {
    const smaller = analysis(),
      larger = analysis();
    smaller.distribution.ref.logits_metadata = {
      source_shape: [1, 4],
      source_dtype: 'float16',
      comparison_shape: [4],
    };
    larger.distribution.ref.size = 8;
    larger.distribution.ref.logits_metadata = {
      source_shape: [1, 8],
      source_dtype: 'float16',
      comparison_shape: [8],
    };
    const items = logitSourceItems([point(smaller), point(larger)], 'ref');
    expect(item(items, 'Shape').value).toBe('[2, Mixed (4 · 8)]');
    expect(item(items, 'Dtype').value).toBe('float16');
    expect(helpRows(items, 'Dimensions')).toContainEqual({label: 'Size', value: '4 · 8'});
  });

  it('keeps equal vocabulary sizes numeric when the reported identities differ', () => {
    const first = analysis(),
      second = analysis();
    for (const [record, identity] of [
      [first, 'vocab-A'],
      [second, 'vocab-B'],
    ] as const) {
      record.distribution.ref.logits_metadata = {
        source_shape: [1, 4],
        source_dtype: 'float16',
        comparison_shape: [4],
        vocab_identity: identity,
      };
    }
    const items = logitSourceItems([point(first), point(second)], 'ref');
    expect(item(items, 'Shape').value).toBe('[2, 4]');
    expect(helpRows(items, 'Dimensions')).toContainEqual({
      label: 'Identity',
      value: 'vocab-A · vocab-B',
    });
  });

  it('uses old reported sizes and keeps both unknown and known source dtypes visible', () => {
    const older = analysis();
    const items = logitSourceItems([point(older)], 'ref');
    expect(item(items, 'Shape').value).toBe('[1, 4]');
    expect(item(items, 'Dtype').value).toBe('—');
    const newer = analysis();
    newer.distribution.ref.logits_metadata = {
      source_shape: [1, 4],
      source_dtype: 'float16',
      comparison_shape: [4],
    };
    const partial = logitSourceItems([point(older), point(newer)], 'ref');
    expect(item(partial, 'Shape').value).toBe('[2, 4]');
    expect(item(partial, 'Dtype').value).toBe('Mixed (Not reported · float16)');
    expect(item(partial, 'Dtype').segments).toBeUndefined();
  });

  it('counts usable input vectors independently of comparison eligibility and selected-token warnings', () => {
    const different = analysis(),
      infinite = analysis(),
      vocabulary = analysis(),
      selected = analysis();
    different.distribution.context = 'different';
    different.distribution.compatible = false;
    different.metrics.kl = {value: null, reason: 'Generated histories have diverged'};
    infinite.metrics.kl = {value: null, reason: 'Divergence is infinite'};
    vocabulary.distribution.compatible = false;
    vocabulary.distribution.reason = 'Matching captured vocabulary IDs and sizes are required';
    vocabulary.metrics.kl = {value: null, reason: vocabulary.distribution.reason};
    selected.distribution.ref.reason = 'Selected token ID is outside captured vocabulary';
    const points = [different, infinite, vocabulary, selected].map(point);
    expect(logitCoverage(points, 'kl').valid).toBe(1);
    for (const side of ['ref', 'target'] as const)
      expect(item(logitSourceItems(points, side), 'Shape').value).toBe('[4, 4]');
  });

  it('excludes pending analysis, unavailable sources and side-specific alignment gaps from shape', () => {
    const missing = analysis();
    missing.distribution.target = {reason: 'Full-vocabulary logits not captured'};
    const points = [{...point(analysis()), hasTarget: false}, point(), point(missing)];
    const target = logitSourceItems(points, 'target');
    expect(item(target, 'Shape').value).toBe('—');
    expect(item(target, 'Dtype').value).toBe('—');
    expect(helpRows(target, 'Dimensions')).toContainEqual({label: 'Count', value: '0'});
    expect(item(logitSourceItems(points, 'ref'), 'Shape').value).toBe('[2, 4]');
    expect(item(logitSourceItems([], 'ref'), 'Shape').value).toBe('—');
    expect(target.slice(1).every((item) => item.segments === undefined)).toBe(true);
  });

  it('excludes unavailable inputs from source dtype summaries', () => {
    const available = analysis(),
      unavailable = analysis();
    available.distribution.ref.logits_metadata = {
      source_shape: [1, 4],
      source_dtype: 'float16',
      comparison_shape: [4],
    };
    unavailable.distribution.ref = {reason: 'Full-vocabulary logits not captured'};
    const items = logitSourceItems([point(available), point(unavailable)], 'ref');
    expect(item(items, 'Shape').value).toBe('[1, 4]');
    expect(item(items, 'Dtype').value).toBe('float16');
  });

  it('uses the actual vocabulary slice without reinterpreting unsupported slice geometry', () => {
    const record = analysis();
    delete record.distribution.ref.size;
    record.distribution.ref.logits_metadata = {
      source_shape: [1, 32, 4],
      source_dtype: 'float16',
      comparison_shape: [4],
    };
    expect(item(logitSourceItems([point(record)], 'ref'), 'Shape').value).toBe('[1, 4]');
    record.distribution.ref.logits_metadata.comparison_shape = [1, 4];
    record.distribution.ref.size = 4;
    expect(item(logitSourceItems([point(record)], 'ref'), 'Shape').value).toBe('[1, —]');
  });
});
