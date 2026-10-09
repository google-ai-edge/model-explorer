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

import type {BookmarkCodec} from '../../../shared/bookmark_store/bookmark_store';

/** A saved KV cache location: one position in one context, for one layer or all. */
export interface SavedPoint {
  context: string;
  layer: number | null;
  position: number;
}
export const kvBookmarkKey = (captureId: string | null) =>
  'debugger.kv-bookmarks.' + captureId;
/** Older entries also carried the Display K/V choice; only the location is kept. */
export const KV_BOOKMARK_CODEC: BookmarkCodec<SavedPoint> = {
  parse(value) {
    if (!value || typeof value !== 'object') return null;
    const point = value as Record<string, unknown>;
    const layer = point['layer'],
      position = point['position'];
    if (
      typeof point['context'] !== 'string' ||
      !(
        layer === null ||
        (Number.isInteger(layer) && (layer as number) >= 0)
      ) ||
      !Number.isInteger(position) ||
      (position as number) < 0
    )
      return null;
    return {
      context: point['context'],
      layer: layer as number | null,
      position: position as number,
    };
  },
  identity: (point) =>
    JSON.stringify([point.context, point.position, point.layer]),
};
export function kvBookmarkLabel(point: SavedPoint) {
  return point.layer === null
    ? `Position ${point.position} · All layers`
    : `Position ${point.position} · Layer ${point.layer}`;
}
