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

import type {BookmarkCodec} from '../../shared/bookmark_store/bookmark_store';

export interface ConversationBookmark {
  id?: string;
  title?: string;
  view?: unknown;
  turn: number;
  index: number;
  alignment: 'content' | 'steps';
}

/** View snapshots are validated when opened, so an expired snapshot remains removable. */
export const CONVERSATION_BOOKMARK_CODEC: BookmarkCodec<ConversationBookmark> =
  {
    parse(value) {
      if (!value || typeof value !== 'object' || Array.isArray(value))
        return null;
      const item = value as Record<string, unknown>;
      if (
        !Number.isInteger(item['turn']) ||
        (item['turn'] as number) < 1 ||
        !Number.isInteger(item['index']) ||
        (item['index'] as number) < -1 ||
        (item['index'] === -1 &&
          (!item['view'] || typeof item['view'] !== 'object')) ||
        !['content', 'steps'].includes(item['alignment'] as string)
      )
        return null;
      if (
        (item['id'] !== undefined &&
          (typeof item['id'] !== 'string' || !item['id'])) ||
        (item['title'] !== undefined && typeof item['title'] !== 'string')
      )
        return null;
      return value as ConversationBookmark;
    },
    identity: (item) =>
      item.id ?? `${item.turn}:${item.alignment}:${item.index}`,
  };
