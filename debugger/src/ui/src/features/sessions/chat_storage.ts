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

/** Browser-local drafts are separate from captured/runtime conversation history. */
export interface StoredChats {
  chats: Record<string, {id: string; name: string}[]>;
  drafts: Record<string, string>;
  active: Record<string, string>;
}
export const emptyChats = (): StoredChats => ({
  chats: {},
  drafts: {},
  active: {},
});
/** Applied on write and on read; longer drafts are truncated rather than lost. */
export const MAX_DRAFT_LENGTH = 16000;
/** Keep only entries for Sessions the server still lists; empty drafts are dropped too. */
export function pruneStoredChats(
  stored: StoredChats,
  sessionIds: Iterable<string>,
): StoredChats {
  const known = new Set(sessionIds);
  const owned = (key: string) => known.has(key.split(':')[0]);
  return {
    chats: Object.fromEntries(
      Object.entries(stored.chats).filter(([session]) => known.has(session)),
    ),
    drafts: Object.fromEntries(
      Object.entries(stored.drafts).filter(
        ([key, value]) => owned(key) && value !== '',
      ),
    ),
    active: Object.fromEntries(
      Object.entries(stored.active).filter(([session]) => known.has(session)),
    ),
  };
}
export function parseStoredChats(raw: string | null): StoredChats {
  const result = emptyChats();
  try {
    const data = JSON.parse(raw ?? 'null');
    if (data?.version !== 1) return result;
    for (const [session, chats] of Object.entries(data.chats ?? {})) {
      if (!Array.isArray(chats)) continue;
      const seen = new Set<string>();
      result.chats[session] = chats
        .filter(
          (c) =>
            c &&
            typeof c.id === 'string' &&
            c.id !== 'capture' &&
            typeof c.name === 'string' &&
            !seen.has(c.id) &&
            !!seen.add(c.id),
        )
        .map((c) => ({id: c.id, name: c.name.slice(0, 160)}));
    }
    for (const [key, value] of Object.entries(data.drafts ?? {}))
      if (typeof value === 'string')
        result.drafts[key] = value.slice(0, MAX_DRAFT_LENGTH);
    for (const [session, id] of Object.entries(data.active ?? {}))
      if (typeof id === 'string' && id.length > 0)
        result.active[session] = id as string;
  } catch {
    /* Corrupt or unavailable storage must not prevent opening a capture. */
  }
  return result;
}
