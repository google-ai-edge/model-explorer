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

export type TensorValueSortColumn = 'index' | 'ref' | 'target' | 'delta' | 'absolute_delta';
export type TensorValueSortDirection = 'ascending' | 'descending';

/** Sort the complete tensor before paging, keeping unavailable values last in either direction. */
export function sortTensorValueRows<
  T extends {index: number; values: readonly (number | string | null | undefined)[]},
>(rows: readonly T[], column: TensorValueSortColumn, direction: TensorValueSortDirection): T[] {
  const order = direction === 'ascending' ? 1 : -1;
  const valueIndex = {ref: 0, target: 1, delta: 2, absolute_delta: 3};
  return [...rows].sort((a, b) => {
    if (column === 'index') return (a.index - b.index) * order;
    const av = a.values[valueIndex[column]],
      bv = b.values[valueIndex[column]];
    const aFinite = typeof av === 'number' && Number.isFinite(av);
    const bFinite = typeof bv === 'number' && Number.isFinite(bv);
    if (aFinite !== bFinite) return aFinite ? -1 : 1;
    return (aFinite && bFinite ? (av - bv) * order : 0) || a.index - b.index;
  });
}
