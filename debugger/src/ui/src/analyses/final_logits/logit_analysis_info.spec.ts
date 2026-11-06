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
import {logitSourceItems} from './logit_analysis_info';

const item = (items: ReturnType<typeof logitSourceItems>, label: string) =>
  items.find((entry) => entry.label === label)!;
const helpRows = (items: ReturnType<typeof logitSourceItems>, label: string) =>
  item(items, label).segments?.flatMap((segment) => segment.help?.rows ?? []) ?? [];

describe('Logit source metadata', () => {
  /** The server's metadata of one side's vectors over a range. */
  const side = (fields: Partial<TokenRangeSideMetadata>): TokenRangeSideMetadata => ({
    count: 2,
    vocabulary_sizes: [4],
    dtypes: ['float16'],
    vocabulary_identities: ['vocab-A'],
    ...fields,
  });

  it('shows mixed captured dtypes directly while keeping the logical vocabulary geometry', () => {
    const items = logitSourceItems(side({dtypes: ['float16', 'float32']}));
    expect(items.map(({label, value}) => [label, value])).toEqual([
      ['Dimensions', '[Token position, Vocabulary]'],
      ['Shape', '[2, 4]'],
      ['Dtype', 'Mixed (float16 · float32)'],
    ]);
    expect(item(items, 'Shape').segments).toBeUndefined();
    expect(item(items, 'Dtype').segments).toBeUndefined();
    expect(helpRows(items, 'Dimensions')).toContainEqual({label: 'Identity', value: 'vocab-A'});
    expect(helpRows(items, 'Dimensions')).toContainEqual({label: 'Count', value: '2'});
  });

  it('shows varying vocabulary sizes directly without adding batch or sequence dimensions', () => {
    const items = logitSourceItems(side({vocabulary_sizes: [4, 8]}));
    expect(item(items, 'Shape').value).toBe('[2, Mixed (4 · 8)]');
    expect(item(items, 'Dtype').value).toBe('float16');
    expect(helpRows(items, 'Dimensions')).toContainEqual({label: 'Size', value: '4 · 8'});
  });

  it('keeps equal vocabulary sizes numeric when the reported identities differ', () => {
    const items = logitSourceItems(side({vocabulary_identities: ['vocab-A', 'vocab-B']}));
    expect(item(items, 'Shape').value).toBe('[2, 4]');
    expect(helpRows(items, 'Dimensions')).toContainEqual({
      label: 'Identity',
      value: 'vocab-A · vocab-B',
    });
  });

  it('keeps both unknown and known source dtypes visible', () => {
    expect(item(logitSourceItems(side({dtypes: [null]})), 'Dtype').value).toBe('—');
    const partial = logitSourceItems(side({dtypes: [null, 'float16']}));
    expect(item(partial, 'Dtype').value).toBe('Mixed (Not reported · float16)');
    expect(item(partial, 'Dtype').segments).toBeUndefined();
  });

  it('shows no shape for a side without vectors', () => {
    const target = logitSourceItems(
      side({count: 0, vocabulary_sizes: [], dtypes: [], vocabulary_identities: []}),
    );
    expect(item(target, 'Shape').value).toBe('—');
    expect(item(target, 'Dtype').value).toBe('—');
    expect(helpRows(target, 'Dimensions')).toContainEqual({label: 'Count', value: '0'});
    expect(helpRows(target, 'Dimensions')).toContainEqual({
      label: 'Identity',
      value: 'Not reported',
    });
    expect(target.slice(1).every((item) => item.segments === undefined)).toBe(true);
  });

  it('writes a size the server could not read from the vocabulary slice as unknown', () => {
    expect(item(logitSourceItems(side({count: 1, vocabulary_sizes: [null]})), 'Shape').value).toBe(
      '[1, —]',
    );
  });
});
