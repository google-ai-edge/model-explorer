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

import {sortTensorValueRows, type TensorValueSortColumn} from './tensor_value_sort';

describe('Tensor Values global sorting', () => {
  const rows = [
    {index: 2, values: [-4, 3, 7, 7]},
    {index: 0, values: [8, 2, -6, 6]},
    {index: 1, values: [2, 0, -2, 2]},
  ];
  const indices = (values: readonly {index: number}[]) => values.map((row) => row.index);

  it.each([
    ['ref', [2, 1, 0]],
    ['target', [1, 0, 2]],
    ['delta', [0, 1, 2]],
    ['absolute_delta', [1, 0, 2]],
  ] satisfies [TensorValueSortColumn, number[]][])(
    'sorts %s by its own signed numeric values',
    (column, expected) => {
      expect(indices(sortTensorValueRows(rows, column, 'ascending'))).toEqual(expected);
      expect(indices(sortTensorValueRows(rows, column, 'descending'))).toEqual(
        [...expected].reverse(),
      );
    },
  );

  it('sorts Index in both directions regardless of value availability', () => {
    const values = [
      {index: 8, values: [null]},
      {index: 1, values: []},
      {index: 4, values: ['NaN']},
    ];
    expect(indices(sortTensorValueRows(values, 'index', 'ascending'))).toEqual([1, 4, 8]);
    expect(indices(sortTensorValueRows(values, 'index', 'descending'))).toEqual([8, 4, 1]);
  });

  it('uses ascending index for equal numeric values in either direction', () => {
    const values = [
      {index: 9, values: [3]},
      {index: 2, values: [3]},
      {index: 5, values: [3]},
      {index: 1, values: [-3]},
    ];
    expect(indices(sortTensorValueRows(values, 'ref', 'ascending'))).toEqual([1, 2, 5, 9]);
    expect(indices(sortTensorValueRows(values, 'ref', 'descending'))).toEqual([2, 5, 9, 1]);
  });

  it('keeps null, missing, string and nonfinite values last with deterministic index order', () => {
    const values = [
      {index: 9, values: [NaN]},
      {index: 7, values: [Infinity]},
      {index: 5, values: [-Infinity]},
      {index: 3, values: [null]},
      {index: 1, values: [undefined]},
      {index: 8, values: ['100']},
      {index: 6, values: ['-Infinity']},
      {index: 4, values: [0]},
      {index: 2, values: [-3]},
      {index: 10, values: []},
    ];
    const unavailable = [1, 3, 5, 6, 7, 8, 9, 10];
    expect(indices(sortTensorValueRows(values, 'ref', 'ascending'))).toEqual([
      2,
      4,
      ...unavailable,
    ]);
    expect(indices(sortTensorValueRows(values, 'ref', 'descending'))).toEqual([
      4,
      2,
      ...unavailable,
    ]);
  });

  it('returns a new array while preserving source rows and values', () => {
    const values = Object.freeze([
      Object.freeze({index: 2, values: Object.freeze([2, 4, 2, 2]), label: 'second'}),
      Object.freeze({index: 1, values: Object.freeze([1, 2, 1, 1]), label: 'first'}),
    ]);
    const sorted = sortTensorValueRows(values, 'ref', 'ascending');
    expect(sorted).not.toBe(values);
    expect(indices(values)).toEqual([2, 1]);
    expect(sorted[0]).toBe(values[1]);
    expect(sorted[1]).toBe(values[0]);
    expect(sorted[0].label).toBe('first');
    expect(values[0].values).toEqual([2, 4, 2, 2]);
    expect(sortTensorValueRows([], 'index', 'ascending')).toEqual([]);
  });

  it('orders the complete tensor across page boundaries before selecting a 16-row page', () => {
    const values = Array.from({length: 34}, (_, index) => ({index, values: [100 - index]}));
    const sorted = sortTensorValueRows(values, 'ref', 'ascending');
    expect(indices(sorted.slice(0, 16))).toEqual(
      Array.from({length: 16}, (_, index) => 33 - index),
    );
    expect(indices(sorted.slice(16, 32))).toEqual(
      Array.from({length: 16}, (_, index) => 17 - index),
    );
    expect(indices(sorted.slice(32, 48))).toEqual([1, 0]);
    expect(indices(sortTensorValueRows(values, 'ref', 'descending').slice(16, 32))).toEqual(
      Array.from({length: 16}, (_, index) => 16 + index),
    );
  });
});
