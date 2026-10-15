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

import {tensorHover} from './kv_tensor_hover';

describe('tensor hover values', () => {
  const rows = [
    {channel: 0, ref: 2, target: 5, delta: 3, abs_delta: 3, status: 'ok'},
    {channel: 1, ref: 7, target: 1, delta: -6, abs_delta: 6, status: 'ok'},
  ];
  it('keeps Bin on raw elements and reports Target minus Reference for both cache groups', () => {
    const tip = tensorHover([rows, rows], {start: 0, end: 1}, 'min');
    expect(tip.properties).toContainEqual({label: 'Bin', value: '-'});
    expect(tip.columns).toEqual(['Ref', 'Target', 'Δ', '|Δ|']);
    expect(tip.rows?.[0].values).toEqual(['2', '5', '3', '3']);
    expect(tip.rows?.[1].label).toBe('Value cache');
  });
  it('does not subtract independent extrema from different elements', () => {
    const tip = tensorHover([rows, rows], {start: 0, end: 2}, 'min');
    expect(tip.properties).toContainEqual({label: 'Bin', value: 'Minimum'});
    expect(tip.properties).toContainEqual({label: 'Index', value: '0–1'});
    expect(tip.columns).toEqual(['Ref', 'Target']);
    expect(tip.rows?.[0].values).toEqual(['2', '1']);
  });
  it('retains readable single-side values without inventing a difference', () => {
    const tip = tensorHover(
      [[{channel: 0, ref: 2, target: null, delta: null, abs_delta: null, status: 'missing'}]],
      {start: 0, end: 1},
      'mean',
    );
    expect(tip.rows?.[0].values[0]).toBe('2');
    expect(tip.rows?.[0].values[2]).toBe(tip.rows?.[0].values[1]);
  });
  it('uses vocabulary labels and keeps both sides readable when comparison is unavailable', () => {
    const tip = tensorHover([rows], {start: 0, end: 1}, 'mean', {
      dimensionLabel: 'Vocabulary index',
      groupLabels: ['Final logits'],
      comparisonAllowed: false,
    });
    expect(tip.properties).toContainEqual({label: 'Dimension', value: 'Vocabulary index'});
    expect(tip.rows?.[0].label).toBe('Final logits');
    expect(tip.rows?.[0].values.slice(0, 2)).toEqual(['2', '5']);
    expect(tip.rows?.[0].values.slice(2)).toEqual(['Unavailable', 'Unavailable']);
  });
});
