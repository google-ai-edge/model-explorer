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

import type {
  KvAnalysisLayer,
  KvPositionAlignment,
  KvRangeBin,
  KvTokenStructure,
} from '../../../data/contracts/kv';

export interface TensorLocation {
  position: number;
  layer: number;
  head: number;
}

/** Choose raw-data coordinates using captured structure, never metric extrema. */
export function defaultTensorLocation(
  layers: readonly KvAnalysisLayer[],
  range: KvRangeBin,
  alignment: KvPositionAlignment | null,
  preferred: {position: number | null; layer: number | null; head: number | null},
  selectedHeads: readonly number[],
): TensorLocation | null {
  const ids = [...new Set(layers.map((row) => row.layer))].sort((a, b) => a - b);
  const sourcePosition = (position: number, side: number) =>
    alignment ? (alignment.pairs[position - alignment.start]?.[side] ?? null) : position;
  const count = (source: KvTokenStructure | null | undefined, position: number | null) =>
    source?.status === 'ok' &&
    position !== null &&
    source.position_start !== null &&
    source.position_count !== null &&
    position >= source.position_start &&
    position < source.position_start + source.position_count &&
    (source.channel_count ?? 0) > 0
      ? (source.head_count ?? 0)
      : 0;
  const at = (position: number, layer: number, paired: boolean): TensorLocation | null => {
    const counts = layers
      .filter((row) => row.layer === layer)
      .map((row) => {
        const ref = count(row.token_structure?.ref, sourcePosition(position, 0));
        const target = count(row.token_structure?.target, sourcePosition(position, 1));
        return paired ? Math.min(ref, target) : Math.max(ref, target);
      });
    const available = Math.max(0, ...counts);
    if (!available) return null;
    const head =
      preferred.head !== null && preferred.head < available && preferred.head >= 0
        ? preferred.head
        : ([...selectedHeads].sort((a, b) => a - b).find((head) => head >= 0 && head < available) ??
          0);
    return {position, layer, head};
  };
  // Preserve a valid explicit or previous location, including readable one-sided data.
  if (preferred.position !== null && preferred.layer !== null) {
    const current = at(preferred.position, preferred.layer, false);
    if (current) return current;
  }
  const find = (position: number | null, layer: number | null) => {
    const candidates = layers.filter((row) => layer === null || row.layer === layer);
    // Without alignment, availability changes only at a captured interval boundary.
    const positions =
      position !== null
        ? [position]
        : alignment
          ? alignment.pairs
              .map((_, index) => alignment.start + index)
              .filter((p) => p >= range.start && p < range.end)
          : [
              ...new Set([
                range.start,
                ...candidates.flatMap((row) =>
                  [row.token_structure?.ref, row.token_structure?.target].map(
                    (source) => source?.position_start ?? range.start,
                  ),
                ),
              ]),
            ]
              .filter((p) => p >= range.start && p < range.end)
              .sort((a, b) => a - b);
    const eligible = ids.filter((id) => layer === null || id === layer);
    for (const paired of [true, false]) {
      for (const p of positions) {
        for (const id of eligible) {
          const value = at(p, id, paired);
          if (value) return value;
        }
      }
    }
    return null;
  };
  return find(preferred.position, preferred.layer) ?? find(null, null);
}
